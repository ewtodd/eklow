#include "kernels/qwen_fast.hpp"

#include <stdexcept>

namespace eklow::qwenk {

namespace {

// The activations a kernel writes, in the form matmul takes for `rows` rows: int8 with a scale per 32-block or fp16.
struct ActOut {
  int8_t* aq = nullptr;
  float* as = nullptr;
  sycl::half* ah = nullptr;
};

ActOut act_out(xmx::Activations* act, int64_t rows, int64_t cols) {
  ActOut o;
  if (!act) return o;
  xmx::set_rows(*act, rows, cols);
  if (act->fp16) o.ah = act->h;
  else o.aq = act->q, o.as = act->s;
  return o;
}

// v (eight values of row r from column c0, by one work-item of a work-group that covers the row) -> rmsnorm * w -> h_out and
// the activations. Eight columns per work-item; the four work-items of a 32-block share its max through sub-group shuffles.
inline void norm_store(const sycl::nd_item<1>& it, float (&v)[8], const float* w, float eps, int32_t C, int32_t r, int32_t c0, float* h_out,
                       const ActOut& a) {
  float ss = 0;
  for (int i = 0; i < 8; ++i) ss += v[i] * v[i];
  ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
  const float inv = 1.0f / sycl::sqrt(ss / float(C) + eps);
  float amax = 0;
  for (int i = 0; i < 8; ++i) {
    v[i] *= inv * w[c0 + i];
    amax = sycl::fmax(amax, sycl::fabs(v[i]));
    if (h_out) h_out[int64_t(r) * C + c0 + i] = v[i];
  }
  if (a.ah) {
    for (int i = 0; i < 8; ++i) a.ah[xmx::act16_index(r, c0 + i, C / 16)] = sycl::half(v[i]);
    return;
  }
  if (!a.aq) return;
  auto sg = it.get_sub_group();
  amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
  amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
  const float d = xmx::quantize_run(v, amax, a.aq + xmx::act_index(r, c0, C / 32));
  if (c0 % 32 == 0) a.as[xmx::act_scale_index(r, c0, C / 32)] = d;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

void attn_heads(sycl::queue& q, const float* qkv, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, int64_t n_rot,
                const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const int* sections, const float* inv_freq,
                float* Q, float* K, float* V, float* G) {
  const int32_t H = int32_t(n_head), Hkv = int32_t(n_head_kv), D = int32_t(head_dim), half = D / 2, R = int32_t(n_rot);
  if (D > 512 || R > D || R % 2 || R / 2 > half) throw std::runtime_error("qwenk::attn_heads: unsupported head shape");
  const int32_t row = (2 * H + 2 * Hkv) * D, slots = H + 2 * Hkv, n_t = int32_t(n_tok);
  const int32_t s0 = sections[0], s1 = sections[1], s2 = sections[2];
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(size_t(D)), h);
    h.parallel_for(sycl::nd_range<1>(size_t(n_tok * slots * half), size_t(half)), [=](sycl::nd_item<1> it) {
      const int32_t g = int32_t(it.get_group(0)), t = g / slots, s = g % slots, i = int32_t(it.get_local_id(0));
      if (s >= H + Hkv) {  // V as it is
        const int32_t hh = s - H - Hkv;
        const float* src = qkv + int64_t(t) * row + (2 * H + Hkv) * D + hh * D;
        float* dst = V + (int64_t(t) * Hkv + hh) * D;
        dst[i] = src[i];
        dst[i + half] = src[i + half];
        return;
      }
      const float* src;
      const float* w;
      float* dst;
      if (s < H) {
        src = qkv + int64_t(t) * row + s * 2 * D, w = q_norm, dst = Q + (int64_t(t) * H + s) * D;
        float* gd = G + (int64_t(t) * H + s) * D;
        gd[i] = src[D + i];
        gd[i + half] = src[D + i + half];
      } else {
        src = qkv + int64_t(t) * row + 2 * H * D + (s - H) * D, w = k_norm, dst = K + (int64_t(t) * Hkv + s - H) * D;
      }
      float a = src[i], b = src[i + half];
      const float ss = sycl::reduce_over_group(it.get_group(), a * a + b * b, sycl::plus<float>());
      const float inv = 1.0f / sycl::sqrt(ss / float(D) + eps);
      slm[i] = a * inv * w[i];
      slm[i + half] = b * inv * w[i + half];
      sycl::group_barrier(it.get_group());
      float x = slm[i];
      if (i < R) {
        const int32_t p = i % (R / 2);
        // The component of this pair (an unclaimed pair would take the fourth, which is 0; the config refuses sections that leave one).
        const int32_t comp = p % 3 == 0 && p < 3 * s0 ? 0 : p % 3 == 1 && p < 3 * s1 ? 1 : p % 3 == 2 && p < 3 * s2 ? 2 : 3;
        const float theta = (comp < 3 ? float(pos[comp * n_t + t]) : 0.0f) * inv_freq[p], c = sycl::cos(theta), sn = sycl::sin(theta);
        x = i < R / 2 ? x * c - slm[i + R / 2] * sn : slm[i - R / 2] * sn + x * c;
      }
      dst[i] = x;
      dst[i + half] = slm[i + half];
    });
  });
}

void gate_quant(sycl::queue& q, const float* x, const float* gate, int64_t rows, int64_t cols, xmx::Activations& act) {
  if (cols % 1024) throw std::runtime_error("qwenk::gate_quant: width not a multiple of 1024");
  const int32_t C = int32_t(cols);
  const ActOut a = act_out(&act, rows, cols);
  q.parallel_for(sycl::nd_range<1>(size_t(rows * cols / 8), 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int64_t gid = int64_t(it.get_global_id(0));
    const int32_t r = int32_t(gid / (C / 8)), c0 = int32_t(gid % (C / 8)) * 8;
    float v[8], amax = 0;
    for (int i = 0; i < 8; ++i) {
      v[i] = x[int64_t(r) * C + c0 + i] * sigmoid(gate[int64_t(r) * C + c0 + i]);
      amax = sycl::fmax(amax, sycl::fabs(v[i]));
    }
    if (a.ah) {
      for (int i = 0; i < 8; ++i) a.ah[xmx::act16_index(r, c0 + i, C / 16)] = sycl::half(v[i]);
      return;
    }
    auto sg = it.get_sub_group();
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
    const float d = xmx::quantize_run(v, amax, a.aq + xmx::act_index(r, c0, C / 32));
    if (c0 % 32 == 0) a.as[xmx::act_scale_index(r, c0, C / 32)] = d;
  });
}

void add_norm_quant(sycl::queue& q, const float* o, const float* x, const float* w, float eps, int64_t rows, int64_t cols, float* x_out,
                    xmx::Activations* act, float* h_out) {
  if (cols % 128 || cols > 8192) throw std::runtime_error("qwenk::add_norm_quant: unsupported width");
  const int32_t C = int32_t(cols);
  const ActOut a = act_out(act, rows, cols);
  q.parallel_for(sycl::nd_range<1>(size_t(rows * cols / 8), size_t(cols / 8)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t r = int32_t(it.get_group(0)), c0 = int32_t(it.get_local_id(0)) * 8;
    float v[8];
    for (int i = 0; i < 8; ++i) v[i] = x[int64_t(r) * C + c0 + i];
    if (o)
      for (int i = 0; i < 8; ++i) {
        v[i] += o[int64_t(r) * C + c0 + i];
        x_out[int64_t(r) * C + c0 + i] = v[i];
      }
    norm_store(it, v, w, eps, C, r, c0, h_out, a);
  });
}

void moe_tail(sycl::queue& q, const float* x2, const float* y, const float* weights, int64_t n_used, const float* moe, const float* sgate,
              int64_t ld_sg, const float* shared, const float* w_next, float eps, int64_t rows, int64_t cols, float* x_out,
              xmx::Activations* act, float* h_out) {
  if (cols % 128 || cols > 8192) throw std::runtime_error("qwenk::moe_tail: unsupported width");
  const int32_t C = int32_t(cols), NU = int32_t(n_used);
  const int32_t LD = int32_t(ld_sg);
  const ActOut a = act_out(w_next ? act : nullptr, rows, cols);
  q.parallel_for(sycl::nd_range<1>(size_t(rows * cols / 8), size_t(cols / 8)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t r = int32_t(it.get_group(0)), c0 = int32_t(it.get_local_id(0)) * 8;
    const float sg = sigmoid(sgate[int64_t(r) * LD]);
    float v[8];
    for (int i = 0; i < 8; ++i) {
      float s;
      if (y) {
        s = 0;
        for (int j = 0; j < NU; ++j) s += weights[r * NU + j] * y[(int64_t(r) * NU + j) * C + c0 + i];
      } else {
        s = moe[int64_t(r) * C + c0 + i];
      }
      v[i] = x2[int64_t(r) * C + c0 + i] + (s + sg * shared[int64_t(r) * C + c0 + i]);
      x_out[int64_t(r) * C + c0 + i] = v[i];
    }
    if (w_next) norm_store(it, v, w_next, eps, C, r, c0, h_out, a);
  });
}

void moe_accumulate(sycl::queue& q, float* x, const float* y, const float* weights, const int32_t* rows, int64_t n_used, const float* extra,
                    const int32_t* extra_row, const float* shared, const float* sgate, int64_t ld_sg, int64_t n_tok, int64_t cols) {
  const int32_t C = int32_t(cols), NU = int32_t(n_used);
  const int64_t LD = ld_sg;
  q.parallel_for(sycl::range<1>(size_t(n_tok * cols)), [=](sycl::id<1> id) {
    const int32_t t = int32_t(id[0]) / C, j = int32_t(id[0]) % C;
    float s = 0;
    if (y)
      for (int i = 0; i < NU; ++i) {
        const int32_t r = rows[t * NU + i];
        if (r >= 0) s += weights[t * NU + i] * y[size_t(r) * C + j];
      }
    if (extra) {
      const int32_t er = extra_row[t];
      if (er >= 0) s += extra[size_t(er) * C + j];
    }
    if (shared) s += sigmoid(sgate[int64_t(t) * LD]) * shared[id[0]];
    x[id[0]] += s;
  });
}

void to_half(sycl::queue& q, const float* x, int64_t n, sycl::half* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = sycl::half(x[id[0]]); });
}

void gather_rows(sycl::queue& q, const float* x, const int32_t* idx, int64_t n, int64_t cols, float* dst, int64_t ld, int64_t col0) {
  const int32_t C = int32_t(cols);
  q.parallel_for(sycl::range<1>(size_t(n * cols)), [=](sycl::id<1> id) {
    const int64_t r = int64_t(id[0]) / C, c = int64_t(id[0]) % C;
    dst[r * ld + col0 + c] = x[int64_t(idx[r]) * C + c];
  });
}

void copy_cols(sycl::queue& q, const float* src, int64_t rows, int64_t n, float* dst, int64_t ld, int64_t c0) {
  q.parallel_for(sycl::range<1>(size_t(rows * n)), [=](sycl::id<1> id) {
    const int64_t r = int64_t(id[0]) / n, c = int64_t(id[0]) % n;
    dst[r * ld + c0 + c] = src[id[0]];
  });
}

}  // namespace eklow::qwenk
