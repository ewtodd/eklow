#include "kernels/fast.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace eklow::fast {

namespace {

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
          wv[j] = float(int(((l >> (4 * (j % 8))) & 15) | ((hi >> (2 * j)) & 3) << 4) - 32);
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

void moe_route(sycl::queue& q, const float* logits, int64_t n_tok, int64_t n_expert, int64_t n_used, const float* down_scale,
               int32_t* ids, float* weights) {
  constexpr int kLanes = 32;
  if (n_expert % kLanes || n_expert / kLanes > 8) throw std::runtime_error("fast::moe_route: unsupported expert count");
  const int32_t NE = int32_t(n_expert), NU = int32_t(n_used);
  // One sub-group per token; lane l holds experts l, l + 32, ... so every reduction is a shuffle.
  // Ties go to the lowest expert id (the host code's partial_sort left them unspecified).
  q.parallel_for(sycl::nd_range<1>(size_t(n_tok * kLanes), kLanes), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kLanes)]] {
    const int32_t t = int32_t(it.get_group(0)), l = int32_t(it.get_local_id(0));
    auto sg = it.get_sub_group();
    const float* lg = logits + t * NE;
    float p[8];
    float mx = -1e30f;
    for (int i = 0; i < 8; ++i) {
      p[i] = i * kLanes < NE ? lg[i * kLanes + l] : -1e30f;
      mx = sycl::fmax(mx, p[i]);
    }
    mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
    float sum = 0;
    for (int i = 0; i < 8; ++i) {
      p[i] = i * kLanes < NE ? sycl::exp(p[i] - mx) : 0.0f;
      sum += p[i];
    }
    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
    for (int i = 0; i < 8; ++i) p[i] /= sum;
    float wsum = 0, wsel[16];
    int32_t sel[16];
    uint32_t taken = 0;  // bit i: this lane's expert i * 32 + l is chosen
    for (int j = 0; j < NU; ++j) {
      float best = -1.0f;
      int32_t bi = 0;
      for (int i = 0; i < 8; ++i)
        if (i * kLanes < NE && !(taken >> i & 1) && p[i] > best) best = p[i], bi = i * kLanes + l;
      const float gbest = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
      const int32_t gi = sycl::reduce_over_group(sg, best == gbest ? bi : 0x7fffffff, sycl::minimum<int32_t>());
      if (gi % kLanes == l) taken |= 1u << (gi / kLanes);
      sel[j] = gi;
      wsel[j] = gbest;
      wsum += gbest;
    }
    const float denom = sycl::fmax(wsum, 6.103515625e-5f);
    if (l < NU) {
      ids[t * NU + l] = sel[l];
      weights[t * NU + l] = wsel[l] / denom * down_scale[sel[l]];
    }
  });
}

void moe_combine(sycl::queue& q, const float* y, const float* weights, int64_t n_tok, int64_t n_used, int64_t cols, float* out,
                 const int32_t* rows, const float* extra) {
  const int32_t NU = int32_t(n_used), C = int32_t(cols);
  q.parallel_for(sycl::range<1>(size_t(n_tok * cols)), [=](sycl::id<1> id) {
    const int32_t t = int32_t(id[0]) / C, j = int32_t(id[0]) % C;
    float s = 0;
    for (int i = 0; i < NU; ++i) {
      const int32_t r = rows ? rows[t * NU + i] : t * NU + i;
      if (r >= 0) s += weights[t * NU + i] * y[size_t(r) * C + j];  // r < 0: a cold expert, in `extra`
    }
    out[id[0]] = extra ? s + extra[id[0]] : s;
  });
}

void moe_group(sycl::queue& q, const int32_t* ids, int64_t n_pairs, int64_t n_expert, int64_t tile, int32_t* scratch, int32_t* tiles,
               int32_t* row_of_pair, const uint8_t* cold) {
  constexpr int kWg = 256;
  if (n_expert > kWg) throw std::runtime_error("fast::moe_group: too many experts");
  const int32_t P = int32_t(n_pairs), NE = int32_t(n_expert), TM = int32_t(tile);
  const int32_t nch = (P + kWg - 1) / kWg;
  int32_t* counts = scratch;           // [nch][NE]: the chunk's pairs per expert, then its offset in the expert's segment
  int32_t* segs = scratch + nch * NE;  // [NE]: first row of each expert's segment
  // Per chunk of 256 pairs, a histogram by expert.
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(kWg), h);
    h.parallel_for(sycl::nd_range<1>(size_t(nch) * kWg, kWg), [=](sycl::nd_item<1> it) {
      const int32_t c = int32_t(it.get_group(0)), lid = int32_t(it.get_local_id(0));
      cnt[lid] = 0;
      sycl::group_barrier(it.get_group());
      const int32_t p = c * kWg + lid;
      if (p < P && !(cold && cold[ids[p]]))
        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(
            cnt[ids[p]])
            .fetch_add(1);
      sycl::group_barrier(it.get_group());
      if (lid < NE) counts[c * NE + lid] = cnt[lid];
    });
  });
  // One work-group: the chunks' offsets within each expert, then the tile list.
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> total(sycl::range<1>(kWg), h), first(sycl::range<1>(kWg + 1), h);
    h.parallel_for(sycl::nd_range<1>(kWg, kWg), [=](sycl::nd_item<1> it) {
      const int32_t e = int32_t(it.get_local_id(0));
      int32_t run = 0;
      if (e < NE) {
        for (int32_t c = 0; c < nch; ++c) {
          const int32_t v = counts[c * NE + e];
          counts[c * NE + e] = run;
          run += v;
        }
      }
      total[e] = run;
      sycl::group_barrier(it.get_group());
      if (e == 0) {
        int32_t t = 0;
        for (int32_t x = 0; x < NE; ++x) {
          first[x] = t;
          t += (total[x] + TM - 1) / TM;
        }
        first[NE] = t;
        tiles[0] = t;
      }
      sycl::group_barrier(it.get_group());
      if (e < NE) {
        segs[e] = first[e] * TM;
        for (int32_t j = first[e]; j < first[e + 1]; ++j) {
          tiles[4 + 4 * j] = e;
          tiles[5 + 4 * j] = j * TM;
          tiles[6 + 4 * j] = sycl::min(TM, total[e] - (j - first[e]) * TM);
        }
      }
    });
  });
  // Per chunk: each pair's rank among the chunk's pairs of its expert.
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> sid(sycl::range<1>(kWg), h);
    h.parallel_for(sycl::nd_range<1>(size_t(nch) * kWg, kWg), [=](sycl::nd_item<1> it) {
      const int32_t c = int32_t(it.get_group(0)), lid = int32_t(it.get_local_id(0));
      const int32_t p = c * kWg + lid;
      sid[lid] = p < P && !(cold && cold[ids[p]]) ? ids[p] : -1;
      sycl::group_barrier(it.get_group());
      if (p >= P) return;
      const int32_t e = sid[lid];
      if (e < 0) {
        row_of_pair[p] = -1;  // cold: no row
        return;
      }
      int32_t rank = 0;
      for (int32_t k = 0; k < lid; ++k) rank += sid[k] == e;
      row_of_pair[p] = segs[e] + counts[c * NE + e] + rank;
    });
  });
}

void moe_gather(sycl::queue& q, const float* x, const int32_t* row_of_pair, int64_t n_pairs, int64_t n_used, int64_t cols,
                sycl::half* out) {
  if (cols % 16) throw std::runtime_error("fast::moe_gather: width not a multiple of 16");
  const int32_t NU = int32_t(n_used), C = int32_t(cols), kh = C / 16;
  q.parallel_for(sycl::range<2>(size_t(n_pairs), size_t(kh)), [=](sycl::id<2> id) {
    const int32_t p = int32_t(id[0]), j = int32_t(id[1]);
    if (row_of_pair[p] < 0) return;
    const sycl::float4* src = reinterpret_cast<const sycl::float4*>(x + int64_t(p / NU) * C + j * 16);
    sycl::vec<sycl::half, 4>* dst = reinterpret_cast<sycl::vec<sycl::half, 4>*>(out + xmx::act16_index(row_of_pair[p], int64_t(j) * 16, kh));
    sycl::float4 v[4];
    for (int c = 0; c < 4; ++c) v[c] = src[c];
    for (int c = 0; c < 4; ++c) dst[c] = v[c].convert<sycl::half, sycl::rounding_mode::automatic>();
  });
}

void matmul_f32(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out) {
  constexpr int kTok = 4, kChunk = 256;  // tokens per work-group; floats of K per staged chunk
  if (k % 4 || n > 256 || n % 32) throw std::runtime_error("fast::matmul_f32: unsupported shape");
  const int32_t K = int32_t(k), N = int32_t(n), M = int32_t(n_tok);
  const int32_t groups = (M + kTok - 1) / kTok;
  // One work-group per 4 tokens, one work-item per output column: the tokens' chunk of K is staged in
  // shared memory and read by all the columns, each item reading its weight row in float4s.
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> xs(sycl::range<1>(kTok * kChunk), h);
    h.parallel_for(sycl::nd_range<1>(size_t(groups) * size_t(N), size_t(N)), [=](sycl::nd_item<1> it) {
      const int32_t g = int32_t(it.get_group(0)), c = int32_t(it.get_local_id(0));
      const sycl::float4* wr = reinterpret_cast<const sycl::float4*>(w + int64_t(c) * K);
      float acc[kTok] = {};
      for (int32_t k0 = 0; k0 < K; k0 += kChunk) {
        const int32_t len = sycl::min<int32_t>(kChunk, K - k0);
        for (int32_t i = c; i < kTok * kChunk; i += N) {
          const int32_t t = g * kTok + i / kChunk, kk = i % kChunk;
          xs[i] = t < M && kk < len ? x[int64_t(t) * K + k0 + kk] : 0.0f;
        }
        sycl::group_barrier(it.get_group());
        for (int32_t kk = 0; kk < len; kk += 4) {
          const sycl::float4 wv = wr[(k0 + kk) / 4];
          for (int t = 0; t < kTok; ++t) {
            const sycl::float4 xv = *reinterpret_cast<const sycl::float4*>(&xs[t * kChunk + kk]);
            acc[t] += wv.x() * xv.x() + wv.y() * xv.y() + wv.z() * xv.z() + wv.w() * xv.w();
          }
        }
        sycl::group_barrier(it.get_group());
      }
      for (int t = 0; t < kTok; ++t)
        if (g * kTok + t < M) out[int64_t(g * kTok + t) * N + c] = acc[t];
    });
  });
}

void matmul_f32_small(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out) {
  constexpr int kWg = 64;
  if (k % 4 || n_tok > 4) throw std::runtime_error("fast::matmul_f32_small: unsupported shape");
  const int32_t K4 = int32_t(k / 4), M = int32_t(n_tok), N = int32_t(n);
  // One work-group per output column, its items striding over K in float4s; the row is read once for all tokens.
  q.parallel_for(sycl::nd_range<1>(size_t(n * kWg), kWg), [=](sycl::nd_item<1> it) {
    const int32_t c = int32_t(it.get_group(0)), lid = int32_t(it.get_local_id(0));
    const sycl::float4* wr = reinterpret_cast<const sycl::float4*>(w + int64_t(c) * K4 * 4);
    float acc[4] = {0, 0, 0, 0};
    for (int32_t i = lid; i < K4; i += kWg) {
      const sycl::float4 wv = wr[i];
      for (int32_t t = 0; t < M; ++t) {
        const sycl::float4 xv = reinterpret_cast<const sycl::float4*>(x + int64_t(t) * K4 * 4)[i];
        acc[t] += wv.x() * xv.x() + wv.y() * xv.y() + wv.z() * xv.z() + wv.w() * xv.w();
      }
    }
    for (int32_t t = 0; t < M; ++t) {
      const float s = sycl::reduce_over_group(it.get_group(), acc[t], sycl::plus<float>());
      if (lid == 0) out[t * N + c] = s;
    }
  });
}

namespace {

// Eight values per work-item to the int8 activations of `rows` rows (the four work-items of a
// 32-block share its max through sub-group shuffles).
inline void quant8(sycl::sub_group sg, float (&v)[8], int8_t* aq, float* as, int32_t r, int32_t c0, int32_t kb) {
  float amax = 0;
  for (int i = 0; i < 8; ++i) amax = sycl::fmax(amax, sycl::fabs(v[i]));
  amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
  amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
  const float d = xmx::quantize_run(v, amax, aq + xmx::act_index(r, c0, kb));
  if (c0 % 32 == 0) as[xmx::act_scale_index(r, c0, kb)] = d;
}

}  // namespace

void residual_norm_quant3(sycl::queue& q, const float* o, const float* w_o, const float* x, float* x_out, float eps, int64_t rows,
                          int64_t cols, xmx::Activations& a1, const float* w1, xmx::Activations& a2, const float* w2,
                          float* r_out, const float* w3) {
  if (cols % 128 || cols > 8192) throw std::runtime_error("fast::residual_norm_quant3: unsupported width");
  const int32_t C = int32_t(cols), kb = C / 32;
  xmx::set_rows(a1, rows, cols);
  xmx::set_rows(a2, rows, cols);
  if (a1.fp16 || a2.fp16) throw std::runtime_error("fast::residual_norm_quant3: int8 rows only");
  int8_t *q1 = a1.q, *q2 = a2.q;
  float *s1 = a1.s, *s2 = a2.s;
  q.parallel_for(sycl::nd_range<1>(size_t(rows * cols / 8), size_t(cols / 8)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t r = int32_t(it.get_group(0)), c0 = int32_t(it.get_local_id(0)) * 8;
    float v[8], ov[8], ss = 0;
    for (int i = 0; i < 8; ++i) {
      v[i] = x[r * C + c0 + i];
      ov[i] = o[r * C + c0 + i];
      ss += ov[i] * ov[i];
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    float inv = 1.0f / sycl::sqrt(ss / float(C) + eps);
    ss = 0;
    for (int i = 0; i < 8; ++i) {
      v[i] += ov[i] * inv * w_o[c0 + i];
      x_out[r * C + c0 + i] = v[i];
      ss += v[i] * v[i];
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    inv = 1.0f / sycl::sqrt(ss / float(C) + eps);
    float n[8];
    for (int i = 0; i < 8; ++i) {
      n[i] = v[i] * inv;
      r_out[r * C + c0 + i] = n[i] * w3[c0 + i];
    }
    auto sg = it.get_sub_group();
    float t[8];
    for (int i = 0; i < 8; ++i) t[i] = n[i] * w1[c0 + i];
    quant8(sg, t, q1, s1, r, c0, kb);
    for (int i = 0; i < 8; ++i) t[i] = n[i] * w2[c0 + i];
    quant8(sg, t, q2, s2, r, c0, kb);
  });
}

void moe_tail(sycl::queue& q, const float* dense, const float* w_d, const float* y, const float* weights, int64_t n_used,
              const float* w_m, const float* w_post, const float* x2, float scale, float* x_out, const float* w_next, float eps,
              int64_t rows, int64_t cols, xmx::Activations* act, float* d_dump, float* m_dump, float* c_dump) {
  if (cols % 128 || cols > 8192) throw std::runtime_error("fast::moe_tail: unsupported width");
  const int32_t C = int32_t(cols), kb = C / 32, NU = int32_t(n_used);
  int8_t* aq = nullptr;
  float* as = nullptr;
  if (act && w_next) {
    xmx::set_rows(*act, rows, cols);
    if (act->fp16) throw std::runtime_error("fast::moe_tail: int8 rows only");
    aq = act->q, as = act->s;
  }
  q.parallel_for(sycl::nd_range<1>(size_t(rows * cols / 8), size_t(cols / 8)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t r = int32_t(it.get_group(0)), c0 = int32_t(it.get_local_id(0)) * 8;
    auto grp = it.get_group();
    float d[8], m[8], ssd = 0, ssm = 0;
    for (int i = 0; i < 8; ++i) {
      d[i] = dense[r * C + c0 + i];
      ssd += d[i] * d[i];
      m[i] = 0;
    }
    // The experts' weighted sum, in the order j = 0 .. n_used - 1.
    for (int j = 0; j < NU; ++j) {
      const float wj = weights[r * NU + j];
      const float* yj = y + size_t(r * NU + j) * C + c0;
      for (int i = 0; i < 8; ++i) m[i] += wj * yj[i];
    }
    for (int i = 0; i < 8; ++i) ssm += m[i] * m[i];
    ssd = sycl::reduce_over_group(grp, ssd, sycl::plus<float>());
    ssm = sycl::reduce_over_group(grp, ssm, sycl::plus<float>());
    const float invd = 1.0f / sycl::sqrt(ssd / float(C) + eps), invm = 1.0f / sycl::sqrt(ssm / float(C) + eps);
    float c[8], ssc = 0;
    for (int i = 0; i < 8; ++i) {
      d[i] = d[i] * invd * w_d[c0 + i];
      m[i] = m[i] * invm * w_m[c0 + i];
      if (d_dump) d_dump[r * C + c0 + i] = d[i];
      if (m_dump) m_dump[r * C + c0 + i] = m[i];
      c[i] = d[i] + m[i];
      if (c_dump) c_dump[r * C + c0 + i] = c[i];
      ssc += c[i] * c[i];
    }
    ssc = sycl::reduce_over_group(grp, ssc, sycl::plus<float>());
    const float invc = 1.0f / sycl::sqrt(ssc / float(C) + eps);
    float v[8], ss = 0;
    for (int i = 0; i < 8; ++i) {
      v[i] = (x2[r * C + c0 + i] + c[i] * invc * w_post[c0 + i]) * scale;
      x_out[r * C + c0 + i] = v[i];
      ss += v[i] * v[i];
    }
    if (!w_next) return;
    ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(C) + eps);
    for (int i = 0; i < 8; ++i) v[i] *= inv * w_next[c0 + i];
    quant8(it.get_sub_group(), v, aq, as, r, c0, kb);
  });
}

}  // namespace eklow::fast
