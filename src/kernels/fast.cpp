#include "kernels/fast.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace eklow::fast {

namespace {

inline float half_at(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(uint16_t(p[0] | p[1] << 8))); }

// The 6-bit value (0..63) of element i of a Q6_K block (ql[128], qh[64], int8 scales[16], fp16 d),
// as ggml lays it out.
inline int q6k_value(const uint8_t* blk, int i) {
  const int half = i / 128, r = i % 128, l = r % 32, quarter = r / 32;
  const uint8_t* ql = blk + half * 64;
  const uint8_t* qh = blk + 128 + half * 32;
  switch (quarter) {
    case 0: return (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
    case 1: return (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    case 2: return (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
    default: return (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
  }
}

}  // namespace

Q6 repack_q6k(sycl::queue& q, const ref::Weight& raw) {
  if (raw.type != gguf::Type::Q6_K || raw.k % 256) throw std::runtime_error("fast::repack_q6k needs a Q6_K matrix");
  const int64_t k = raw.k, n = raw.n;
  const size_t lo = size_t(n * k / 2), hi = size_t(n * k / 4), sc = size_t(n * k / 16), d = size_t(n * k / 256) * 2;
  auto* p = sycl::malloc_device<uint8_t>(lo + hi + sc + d, q);
  if (!p) throw std::runtime_error("fast::repack_q6k: out of device memory");
  Q6 w{k, n, p, p + lo, reinterpret_cast<int8_t*>(p + lo + hi), reinterpret_cast<sycl::half*>(p + lo + hi + sc)};
  const ref::Weight r = raw;
  // One work-item per 4 elements: one byte of high bits, two of low nibbles.
  q.parallel_for(sycl::range<2>(size_t(n), size_t(k / 4)), [=](sycl::id<2> id) {
    const int64_t row = int64_t(id[0]), e = int64_t(id[1]) * 4;
    const uint8_t* blk = r.data + uint64_t(row) * r.row_bytes + uint64_t(e / 256) * 210;
    int v[4];
    for (int j = 0; j < 4; ++j) v[j] = q6k_value(blk, int((e + j) % 256));
    const int64_t at = row * k + e;
    w.lo[at / 2] = uint8_t((v[0] & 15) | (v[1] & 15) << 4);
    w.lo[at / 2 + 1] = uint8_t((v[2] & 15) | (v[3] & 15) << 4);
    w.hi[at / 4] = uint8_t(v[0] >> 4 | (v[1] >> 4) << 2 | (v[2] >> 4) << 4 | (v[3] >> 4) << 6);
    if (e % 16 == 0) w.sc[at / 16] = int8_t(blk[192 + (e % 256) / 16]);
    if (e % 256 == 0) w.d[at / 256] = sycl::bit_cast<sycl::half>(uint16_t(blk[208] | blk[209] << 8));
  }).wait();
  return w;
}

void free_q6(sycl::queue& q, Q6& w) {
  sycl::free(w.lo, q);
  w = Q6{};
}

void embed(sycl::queue& q, const Q6& table, const int32_t* tokens, int64_t n_tok, float scale, float* out) {
  const Q6 w = table;
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(w.k)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
    const int64_t e = int64_t(tokens[t]) * w.k + i;
    const int v = (w.lo[e / 2] >> (4 * (e % 2)) & 15) | (w.hi[e / 4] >> (2 * (e % 4)) & 3) << 4;
    out[t * w.k + i] = float(w.d[e / 256]) * float(w.sc[e / 16]) * float(v - 32) * scale;
  });
}

void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out) {
  constexpr int kWg = 256;
  q.parallel_for(sycl::nd_range<1>(size_t(rows * kWg), kWg), [=](sycl::nd_item<1> it) {
    const int64_t r = int64_t(it.get_group(0));
    const int lid = int(it.get_local_id(0));
    const float* xr = x + r * cols;
    float ss = 0;
    for (int64_t i = lid; i < cols; i += kWg) ss += xr[i] * xr[i];
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(cols) + eps);
    for (int64_t i = lid; i < cols; i += kWg) out[r * cols + i] = xr[i] * inv * (w ? w[i] : 1.0f);
  });
}

void rope_neox(sycl::queue& q, float* x, const int32_t* pos, int64_t n_tok, int64_t n_head, int64_t head_dim,
               int64_t n_rot, const float* inv_freq) {
  const int64_t half = n_rot / 2;
  q.parallel_for(sycl::range<3>(size_t(n_tok), size_t(n_head), size_t(half)), [=](sycl::id<3> id) {
    const int64_t t = int64_t(id[0]), h = int64_t(id[1]), i = int64_t(id[2]);
    const float theta = float(pos[t]) * inv_freq[i];
    const float c = sycl::cos(theta), s = sycl::sin(theta);
    float* v = x + (t * n_head + h) * head_dim;
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  });
}

void gelu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
    const float g = gate[id[0]];
    out[id[0]] = 0.5f * g * (1.0f + sycl::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * up[id[0]];
  });
}

void qkv_heads(sycl::queue& q, const float* qkv, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim,
               bool has_v, const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const float* inv_freq,
               float* Q, float* K, float* V) {
  const int32_t H = int32_t(n_head), Hkv = int32_t(n_head_kv), D = int32_t(head_dim), half = D / 2;
  const int32_t row = (H + (has_v ? 2 : 1) * Hkv) * D, slots = H + 2 * Hkv;
  q.parallel_for(sycl::nd_range<1>(size_t(n_tok * slots * half), size_t(half)), [=](sycl::nd_item<1> it) {
    const int32_t g = int32_t(it.get_group(0)), t = g / slots, s = g % slots, i = int32_t(it.get_local_id(0));
    const float* src;
    const float* w;
    float* dst;
    bool rope = true;
    if (s < H) {
      src = qkv + t * row + s * D, w = q_norm, dst = Q + (t * H + s) * D;
    } else if (s < H + Hkv) {
      src = qkv + t * row + H * D + (s - H) * D, w = k_norm, dst = K + (t * Hkv + s - H) * D;
    } else {
      const int32_t h = s - H - Hkv;
      src = qkv + t * row + (has_v ? H * D + Hkv * D : H * D) + h * D, w = nullptr, dst = V + (t * Hkv + h) * D, rope = false;
    }
    float a = src[i], b = src[i + half];
    const float ss = sycl::reduce_over_group(it.get_group(), a * a + b * b, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(D) + eps);
    a *= inv * (w ? w[i] : 1.0f);
    b *= inv * (w ? w[i + half] : 1.0f);
    if (rope) {
      const float theta = float(pos[t]) * inv_freq[i], c = sycl::cos(theta), sn = sycl::sin(theta);
      const float a2 = a * c - b * sn;
      b = a * sn + b * c;
      a = a2;
    }
    dst[i] = a;
    dst[i + half] = b;
  });
}

void residual_norm_quant(sycl::queue& q, const float* o, const float* w_o, const float* x, float scale, float* x_out,
                         const float* w_next, float eps, int64_t rows, int64_t cols, xmx::Activations* act, float* h_out) {
  // Eight columns per work-item; the four work-items of a 32-block share its max through
  // sub-group shuffles.
  if (cols % 128 || cols > 8192) throw std::runtime_error("fast::residual_norm_quant: unsupported width");
  const int32_t C = int32_t(cols), kb = C / 32;
  int8_t* aq = nullptr;
  float* as = nullptr;
  sycl::half* ah = nullptr;
  if (act && w_next) {
    xmx::set_rows(*act, rows, cols);
    if (act->fp16) ah = act->h;
    else aq = act->q, as = act->s;
  }
  q.parallel_for(sycl::nd_range<1>(size_t(rows * cols / 8), size_t(cols / 8)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t r = int32_t(it.get_group(0)), c0 = int32_t(it.get_local_id(0)) * 8;
    float v[8];
    for (int i = 0; i < 8; ++i) v[i] = x[r * C + c0 + i];
    if (o) {
      float ov[8], ss = 0;
      for (int i = 0; i < 8; ++i) {
        ov[i] = o[r * C + c0 + i];
        ss += ov[i] * ov[i];
      }
      ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
      const float inv = 1.0f / sycl::sqrt(ss / float(C) + eps);
      for (int i = 0; i < 8; ++i) {
        v[i] = (v[i] + ov[i] * inv * w_o[c0 + i]) * scale;
        x_out[r * C + c0 + i] = v[i];
      }
    }
    if (!w_next) return;
    float ss = 0;
    for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(C) + eps);
    float amax = 0;
    for (int i = 0; i < 8; ++i) {
      v[i] *= inv * w_next[c0 + i];
      amax = sycl::fmax(amax, sycl::fabs(v[i]));
      if (h_out) h_out[r * C + c0 + i] = v[i];
    }
    if (ah) {
      for (int i = 0; i < 8; ++i) ah[xmx::act16_index(r, c0 + i, C / 16)] = sycl::half(v[i]);
      return;
    }
    if (!aq) return;
    auto sg = it.get_sub_group();
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
    const float d = xmx::quantize_run(v, amax, aq + xmx::act_index(r, c0, kb));
    if (c0 % 32 == 0) as[xmx::act_scale_index(r, c0, kb)] = d;
  });
}

void gelu_quant(sycl::queue& q, const float* gu, int64_t rows, int64_t n, xmx::Activations& act) {
  if (n % 512) throw std::runtime_error("fast::gelu_quant: width not a multiple of 512");
  xmx::set_rows(act, rows, n);
  int8_t* aq = act.q;
  float* as = act.s;
  sycl::half* ah = act.fp16 ? act.h : nullptr;
  const int32_t N = int32_t(n), kb = N / 32;
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(rows), size_t(n / 8)), sycl::range<2>(1, 64)),
                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                   const int32_t r = int32_t(it.get_global_id(0)), c0 = int32_t(it.get_global_id(1)) * 8;
                   float v[8], amax = 0;
                   for (int i = 0; i < 8; ++i) {
                     const float g = gu[r * 2 * N + c0 + i], u = gu[r * 2 * N + N + c0 + i];
                     v[i] = 0.5f * g * (1.0f + sycl::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
                     amax = sycl::fmax(amax, sycl::fabs(v[i]));
                   }
                   if (ah) {
                     for (int i = 0; i < 8; ++i) ah[xmx::act16_index(r, c0 + i, N / 16)] = sycl::half(v[i]);
                     return;
                   }
                   auto sg = it.get_sub_group();
                   amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
                   amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
                   const float d = xmx::quantize_run(v, amax, aq + xmx::act_index(r, c0, kb));
                   if (c0 % 32 == 0) as[xmx::act_scale_index(r, c0, kb)] = d;
                 });
}

void add_scale(sycl::queue& q, const float* a, const float* b, float s, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = (a[id[0]] + b[id[0]]) * s; });
}

void softcap(sycl::queue& q, float* x, float cap, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { x[id[0]] = cap * sycl::tanh(x[id[0]] / cap); });
}

void matmul_q6(sycl::queue& q, const Q6& w, const float* x, int64_t rows, float* out) {
  // One sub-group per output column, 16 sub-groups per work-group. Per 256-element super-block
  // each lane takes 16 consecutive elements: 8 bytes of low nibbles, 4 of high bits, one scale,
  // for up to 8 input rows per pass, then the sub-group sums its lanes.
  constexpr int kSg = 16, kRows = 8, kWg = 256;
  const Q6 ww = w;
  if (ww.n % (kWg / kSg)) throw std::runtime_error("fast::matmul_q6: rows not a multiple of 16");
  for (int64_t r0 = 0; r0 < rows; r0 += kRows) {
    const int64_t rc = std::min<int64_t>(kRows, rows - r0);
    q.parallel_for(sycl::nd_range<1>(size_t(ww.n * kSg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
      const int64_t v = int64_t(it.get_global_id(0)) / kSg;
      const int lane = int(it.get_sub_group().get_local_id()[0]);
      const int64_t k = ww.k;
      float acc[kRows] = {};
      for (int64_t b = 0; b < k / 256; ++b) {
        const int64_t e = v * k + b * 256 + lane * 16;
        const sycl::uint2 lo = *reinterpret_cast<const sycl::uint2*>(ww.lo + e / 2);
        const uint32_t hi = *reinterpret_cast<const uint32_t*>(ww.hi + e / 4);
        const float s = float(ww.d[e / 256]) * float(ww.sc[e / 16]);
        float wv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const uint32_t l = j < 8 ? lo.x() : lo.y();
          wv[j] = float(int((l >> (4 * (j % 8))) & 15 | ((hi >> (2 * j)) & 3) << 4) - 32);
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          if (r >= rc) break;
          const sycl::float4* xr = reinterpret_cast<const sycl::float4*>(x + (r0 + r) * k + b * 256 + lane * 16);
          float dot = 0;
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const sycl::float4 xv = xr[c];
            dot += xv.x() * wv[4 * c] + xv.y() * wv[4 * c + 1] + xv.z() * wv[4 * c + 2] + xv.w() * wv[4 * c + 3];
          }
          acc[r] += dot * s;
        }
      }
      auto sg = it.get_sub_group();
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        if (r >= rc) break;
        const float sum = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
        if (lane == 0) out[(r0 + r) * ww.n + v] = sum;
      }
    });
  }
}

}  // namespace eklow::fast
