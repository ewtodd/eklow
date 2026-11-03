#include "kernels/gemma_fast.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace eklow::fast {

void embed(sycl::queue& q, const Q6& table, const int32_t* tokens, int64_t n_tok, float scale, float* out) {
  const Q6 w = table;
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(w.k)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
    const int64_t e = int64_t(tokens[t]) * w.k + i;
    const int v = (w.lo[e / 2] >> (4 * (e % 2)) & 15) | (w.hi[e / 4] >> (2 * (e % 4)) & 3) << 4;
    out[t * w.k + i] = float(w.d[e / 256]) * float(w.sc[e / 16]) * float(v - 32) * scale;
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
