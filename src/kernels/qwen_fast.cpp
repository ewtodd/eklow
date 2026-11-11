#include "kernels/qwen_fast.hpp"

#include <stdexcept>

namespace eklow::qwenk {

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

void gather_rows(sycl::queue& q, const float* x, const int32_t* idx, int64_t n, int64_t cols, float* dst, int64_t ld, int64_t col0) {
  const int32_t C = int32_t(cols);
  q.parallel_for(sycl::range<1>(size_t(n * cols)), [=](sycl::id<1> id) {
    const int64_t r = int64_t(id[0]) / C, c = int64_t(id[0]) % C;
    dst[r * ld + col0 + c] = x[int64_t(idx[r]) * C + c];
  });
}

}  // namespace eklow::qwenk
