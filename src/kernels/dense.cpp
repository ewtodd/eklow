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

}  // namespace eklow::dense
