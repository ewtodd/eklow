#include "kernels/ref_dflash.hpp"

#include <limits>

namespace eklow::ref {

void attention_ranges(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t key0, int64_t n_tok, int64_t n_head,
                      int64_t n_head_kv, int64_t head_dim, const int32_t* lo, const int32_t* hi, float scale, float* out) {
  const int64_t group = n_head / n_head_kv;
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(n_head)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), h = int64_t(id[1]), kvh = h / group;
    const float* qv = Q + (t * n_head + h) * head_dim;
    float m = -std::numeric_limits<float>::infinity();
    for (int64_t j = lo[t]; j <= hi[t]; ++j) {
      const float* kv = K + ((j - key0) * n_head_kv + kvh) * head_dim;
      float s = 0;
      for (int64_t d = 0; d < head_dim; ++d) s += qv[d] * kv[d];
      m = sycl::fmax(m, s * scale);
    }
    float l = 0;
    float acc[256];
    for (int64_t d = 0; d < head_dim; ++d) acc[d] = 0;
    for (int64_t j = lo[t]; j <= hi[t]; ++j) {
      const float* kv = K + ((j - key0) * n_head_kv + kvh) * head_dim;
      const float* vv = V + ((j - key0) * n_head_kv + kvh) * head_dim;
      float s = 0;
      for (int64_t d = 0; d < head_dim; ++d) s += qv[d] * kv[d];
      const float p = sycl::exp(s * scale - m);
      l += p;
      for (int64_t d = 0; d < head_dim; ++d) acc[d] += p * vv[d];
    }
    for (int64_t d = 0; d < head_dim; ++d) out[(t * n_head + h) * head_dim + d] = acc[d] / l;
  });
}

void topk_rows(sycl::queue& q, const float* x, int64_t rows, int64_t n, int64_t k, int32_t* ids, float* vals) {
  q.parallel_for(sycl::range<1>(size_t(rows)), [=](sycl::id<1> id) {
    const int64_t r = int64_t(id[0]);
    const float* xr = x + r * n;
    int32_t* ir = ids + r * k;
    float* vr = vals + r * k;
    int64_t have = 0;
    for (int64_t i = 0; i < n; ++i) {
      const float v = xr[i];
      // Ids ascend, so a value equal to the worst kept loses to it.
      if (have == k && !(v > vr[k - 1])) continue;
      int64_t pos = have < k ? have++ : k - 1;
      while (pos > 0 && v > vr[pos - 1]) {
        vr[pos] = vr[pos - 1];
        ir[pos] = ir[pos - 1];
        --pos;
      }
      vr[pos] = v;
      ir[pos] = int32_t(i);
    }
  });
}

void silu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
    const float g = gate[id[0]];
    out[id[0]] = g / (1.0f + sycl::exp(-g)) * up[id[0]];
  });
}

void round_acts(sycl::queue& q, const float* x, int64_t rows, int64_t cols, bool fp16, float* out) {
  if (fp16) {
    q.parallel_for(sycl::range<1>(size_t(rows * cols)), [=](sycl::id<1> id) { out[id[0]] = float(sycl::half(x[id[0]])); });
    return;
  }
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(cols / 32)), [=](sycl::id<2> id) {
    const float* xb = x + int64_t(id[0]) * cols + int64_t(id[1]) * 32;
    float* ob = out + int64_t(id[0]) * cols + int64_t(id[1]) * 32;
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xb[j]));
    const float d = amax / 127.0f, inv = d != 0 ? 1.0f / d : 0.0f;
    for (int j = 0; j < 32; ++j) ob[j] = float(int8_t(sycl::round(xb[j] * inv))) * d;
  });
}

void jitter(sycl::queue& q, float* x, int64_t n, float eps) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
    const uint32_t h = uint32_t(id[0]) * 2654435761u;
    x[id[0]] *= 1.0f + eps * (float((h >> 8) & 0xffff) / 32768.0f - 1.0f);
  });
}

}  // namespace eklow::ref
