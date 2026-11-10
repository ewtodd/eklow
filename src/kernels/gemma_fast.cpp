#include "kernels/gemma_fast.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace eklow::fast {

void qkv_heads(sycl::queue& q, const float* qs, int64_t ld_q, const float* ks, int64_t ld_k, const float* vs, int64_t ld_v,
               int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, const float* q_norm, const float* k_norm,
               float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K, float* V) {
  const int32_t H = int32_t(n_head), Hkv = ks ? int32_t(n_head_kv) : 0, D = int32_t(head_dim), half = D / 2;
  const int32_t lq = int32_t(ld_q), lk = int32_t(ld_k), lv = int32_t(ld_v), slots = H + 2 * Hkv;
  const bool own_v = vs != nullptr;
  q.parallel_for(sycl::nd_range<1>(size_t(n_tok * slots * half), size_t(half)), [=](sycl::nd_item<1> it) {
    const int32_t g = int32_t(it.get_group(0)), t = g / slots, s = g % slots, i = int32_t(it.get_local_id(0));
    const float* src;
    const float* w;
    float* dst;
    bool rope = true;
    if (s < H) {
      src = qs + t * lq + s * D, w = q_norm, dst = Q + (t * H + s) * D;
    } else if (s < H + Hkv) {
      src = ks + t * lk + (s - H) * D, w = k_norm, dst = K + (t * Hkv + s - H) * D;
    } else {
      const int32_t h = s - H - Hkv;
      src = own_v ? vs + t * lv + h * D : ks + t * lk + h * D, w = nullptr, dst = V + (t * Hkv + h) * D, rope = false;
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

void ple_inputs(sycl::queue& q, float* pl, const ref::Weight& table, const int32_t* tokens, int64_t n_tok, int64_t n_layer,
                int64_t width, const float* norm, float eps, float in_scale, float table_scale, float out_scale) {
  if (table.type != gguf::Type::Q8_0 || width % 32 || width > 1024 || table.k != n_layer * width)
    throw std::runtime_error("fast::ple_inputs: the per-layer table must be Q8_0 rows of n_layer x width");
  const int32_t W = int32_t(width), NL = int32_t(n_layer);
  const uint8_t* tab = table.data;
  const uint64_t row_bytes = table.row_bytes;
  q.parallel_for(sycl::nd_range<1>(size_t(n_tok * n_layer * width), size_t(width)), [=](sycl::nd_item<1> it) {
    const int32_t g = int32_t(it.get_group(0)), t = g / NL, l = g % NL, i = int32_t(it.get_local_id(0));
    float* p = pl + (int64_t(t) * NL + l) * W;
    const float v = p[i] * in_scale;
    const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(W) + eps);
    const int32_t e = l * W + i;
    const uint8_t* blk = tab + uint64_t(tokens[t]) * row_bytes + uint64_t(e / 32) * 34;
    const float d = float(sycl::bit_cast<sycl::half>(uint16_t(blk[0] | blk[1] << 8)));
    const float tv = d * float(int8_t(blk[2 + e % 32]));
    p[i] = (v * inv * norm[i] + tv * table_scale) * out_scale;
  });
}

void ple_gate(sycl::queue& q, const float* g, const float* pl, int64_t layer, int64_t n_layer, int64_t rows, int64_t width,
              xmx::Activations& act) {
  if (width % 32 || width > 1024) throw std::runtime_error("fast::ple_gate: unsupported width");
  xmx::set_rows(act, rows, width);
  const int32_t W = int32_t(width), kb = W / 32, l = int32_t(layer), NL = int32_t(n_layer);
  sycl::half* ah = act.fp16 ? act.h : nullptr;
  int8_t* aq = act.fp16 ? nullptr : act.q;
  float* as = act.fp16 ? nullptr : act.s;
  q.parallel_for(sycl::nd_range<1>(size_t(rows * width / 8), size_t(width / 8)), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t r = int32_t(it.get_group(0)), c0 = int32_t(it.get_local_id(0)) * 8;
    float v[8], amax = 0;
    for (int i = 0; i < 8; ++i) {
      const float x = g[r * W + c0 + i];
      v[i] = 0.5f * x * (1.0f + sycl::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))) *
             pl[(int64_t(r) * NL + l) * W + c0 + i];
      amax = sycl::fmax(amax, sycl::fabs(v[i]));
    }
    if (ah) {
      for (int i = 0; i < 8; ++i) ah[xmx::act16_index(r, c0 + i, W / 16)] = sycl::half(v[i]);
      return;
    }
    auto sg = it.get_sub_group();
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
    const float d = xmx::quantize_run(v, amax, aq + xmx::act_index(r, c0, kb));
    if (c0 % 32 == 0) as[xmx::act_scale_index(r, c0, kb)] = d;
  });
}

}  // namespace eklow::fast
