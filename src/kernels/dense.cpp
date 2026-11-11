#include "kernels/dense.hpp"

#include <stdexcept>

namespace eklow::dense {

// A work-group of 256 per row: the one-row-a-work-item ref::rmsnorm was vega's reader's largest stage (2026-11-10, 3 x
// 3,000 tokens: 317 ms of 908 with a wait after each stage).
void norm_quant(sycl::queue& q, float* x, const float* add, const float* w, int64_t rows, int64_t E, float eps, xmx::Activations& act,
                int64_t act_rows, float* h) {
  constexpr int kLocal = 256;
  if (E % kLocal || E / kLocal > 8) throw std::runtime_error("dense::norm_quant: width not a multiple of 256 up to 2048");
  xmx::set_rows(act, act_rows, E, true);
  if (rows <= 0) return;
  sycl::half* ah = act.h;
  const int32_t per = int32_t(E / kLocal), EE = int32_t(E);
  q.parallel_for(sycl::nd_range<1>(size_t(rows) * kLocal, kLocal), [=](sycl::nd_item<1> it) {
    const int64_t r = int64_t(it.get_group(0));
    const int32_t l = int32_t(it.get_local_id(0));
    float v[8], ss = 0;
    for (int32_t j = 0; j < per; ++j) {
      const int64_t at = r * EE + j * kLocal + l;
      v[j] = x[at] + (add ? add[at] : 0.0f);
      if (add) x[at] = v[j];
      ss += v[j] * v[j];
    }
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(EE) + eps);
    for (int32_t j = 0; j < per; ++j) {
      const int32_t c = j * kLocal + l;
      const float y = v[j] * inv * w[c];
      if (h) h[r * EE + c] = y;
      ah[xmx::act16_index(r, c, EE / 16)] = sycl::half(y);
    }
  });
}

void qk_heads(sycl::queue& q, const float* qkv, int64_t rows, int64_t n_head, int64_t n_head_kv, int64_t D, const float* q_norm,
              const float* k_norm, float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K, float* V) {
  if (D > 512 || D % 2) throw std::runtime_error("dense::qk_heads: unsupported head size");
  if (rows <= 0) return;
  const int32_t H = int32_t(n_head), Hkv = int32_t(n_head_kv), DD = int32_t(D), half = DD / 2, row = (H + 2 * Hkv) * DD, slots = H + 2 * Hkv;
  q.parallel_for(sycl::nd_range<1>(size_t(rows) * slots * half, size_t(half)), [=](sycl::nd_item<1> it) {
    const int32_t g = int32_t(it.get_group(0)), t = g / slots, s = g % slots, i = int32_t(it.get_local_id(0));
    const float* src = qkv + int64_t(t) * row + s * DD;
    if (s >= H + Hkv) {
      float* dst = V + (int64_t(t) * Hkv + s - H - Hkv) * DD;
      dst[i] = src[i];
      dst[i + half] = src[i + half];
      return;
    }
    const float* w = s < H ? q_norm : k_norm;
    float* dst = s < H ? Q + (int64_t(t) * H + s) * DD : K + (int64_t(t) * Hkv + s - H) * DD;
    const float a = src[i], b = src[i + half];
    const float ss = sycl::reduce_over_group(it.get_group(), a * a + b * b, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(DD) + eps);
    const float x0 = a * inv * w[i], x1 = b * inv * w[i + half];
    const float theta = float(pos[t]) * inv_freq[i], c = sycl::cos(theta), sn = sycl::sin(theta);
    dst[i] = x0 * c - x1 * sn;
    dst[i + half] = x0 * sn + x1 * c;
  });
}

}  // namespace eklow::dense
