#include "kernels/inkling_fast.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eklow::inkling {

namespace {

// Few rows (decode): one 256-lane work-group a row.
void depth_mix_wide(sycl::queue& q, const States& st, const float* query, int n_query, int64_t rows, int64_t n_embd, float eps,
                    float* out) {
  constexpr int kWg = 256, kS = 8, kQ = 4;
  q.parallel_for(sycl::nd_range<1>(size_t(rows * kWg), kWg), [=](sycl::nd_item<1> it) {
    const int64_t r = int64_t(it.get_group(0));
    const int lid = int(it.get_local_id(0));
    float ss[kS], dq[kQ][kS];
    for (int j = 0; j < kS; ++j) {
      ss[j] = 0;
      for (int i = 0; i < kQ; ++i) dq[i][j] = 0;
    }
    for (int64_t e = lid; e < n_embd; e += kWg) {
      float qv[kQ];
      for (int i = 0; i < kQ; ++i) qv[i] = i < n_query ? query[int64_t(i) * n_embd + e] : 0.0f;
      for (int j = 0; j < kS; ++j) {
        if (j >= st.n) break;
        const float x = st.p[j][r * n_embd + e];
        ss[j] += x * x;
        for (int i = 0; i < kQ; ++i) dq[i][j] += x * qv[i];
      }
    }
    float inv[kS], w[kQ][kS];
    for (int j = 0; j < kS; ++j) {
      if (j >= st.n) break;
      inv[j] = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), ss[j], sycl::plus<float>()) / float(n_embd) + eps);
      for (int i = 0; i < kQ; ++i) {
        if (i >= n_query) break;
        w[i][j] = sycl::reduce_over_group(it.get_group(), dq[i][j], sycl::plus<float>());
      }
    }
    const float sc = 1.0f / sycl::sqrt(float(n_embd));
    for (int i = 0; i < kQ; ++i) {
      if (i >= n_query) break;
      float mx = -INFINITY;
      for (int j = 0; j < kS; ++j) {
        if (j >= st.n) break;
        w[i][j] = w[i][j] * inv[j] * sc;
        mx = sycl::fmax(mx, w[i][j]);
      }
      float sum = 0;
      for (int j = 0; j < kS; ++j) {
        if (j >= st.n) break;
        sum += (w[i][j] = sycl::exp(w[i][j] - mx));
      }
      for (int j = 0; j < kS; ++j) {
        if (j >= st.n) break;
        w[i][j] = w[i][j] / sum * inv[j];
      }
    }
    for (int64_t e = lid; e < n_embd; e += kWg) {
      float x[kS];
      for (int j = 0; j < kS; ++j)
        if (j < st.n) x[j] = st.p[j][r * n_embd + e];
      for (int i = 0; i < kQ; ++i) {
        if (i >= n_query) break;
        float acc = 0;
        for (int j = 0; j < kS; ++j)
          if (j < st.n) acc += w[i][j] * x[j];
        out[(int64_t(i) * rows + r) * n_embd + e] = acc;
      }
    }
  });
}

}  // namespace

void depth_mix(sycl::queue& q, const States& s, const float* query, int n_query, int64_t rows, int64_t n_embd, float eps, float* out) {
  if (s.n < 1 || s.n > 8 || n_query < 1 || n_query > 4) throw std::runtime_error("inkling::depth_mix: bad state or query count");
  if (rows <= 64) return depth_mix_wide(q, s, query, n_query, rows, n_embd, eps, out);
  // Many rows (prefill): one sub-group of 16 a row, 8 rows a work-group: the sums of squares and the products with each query reduce over the
  // sub-group (shuffles, no barriers).
  constexpr int kSg = 16, kRows = 8, kS = 8, kQ = 4;
  const States st = s;
  const int64_t groups = (rows + kRows - 1) / kRows;
  q.parallel_for(sycl::nd_range<1>(size_t(groups * kRows * kSg), kRows * kSg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
    const int64_t r = int64_t(it.get_global_id(0)) / kSg;
    if (r >= rows) return;
    const auto sg = it.get_sub_group();
    const int lane = int(sg.get_local_id()[0]);
    float ss[kS], dq[kQ][kS];
#pragma unroll
    for (int j = 0; j < kS; ++j) {
      ss[j] = 0;
#pragma unroll
      for (int i = 0; i < kQ; ++i) dq[i][j] = 0;
    }
    for (int64_t e = lane; e < n_embd; e += kSg) {
      float qv[kQ];
#pragma unroll
      for (int i = 0; i < kQ; ++i) qv[i] = i < n_query ? query[int64_t(i) * n_embd + e] : 0.0f;
#pragma unroll
      for (int j = 0; j < kS; ++j)
        if (j < st.n) {
          const float x = st.p[j][r * n_embd + e];
          ss[j] += x * x;
#pragma unroll
          for (int i = 0; i < kQ; ++i) dq[i][j] += x * qv[i];
        }
    }
    float w[kQ][kS], inv[kS];
    const float sc = 1.0f / sycl::sqrt(float(n_embd));
#pragma unroll
    for (int j = 0; j < kS; ++j)
      if (j < st.n) inv[j] = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, ss[j], sycl::plus<float>()) / float(n_embd) + eps);
#pragma unroll
    for (int i = 0; i < kQ; ++i) {
      if (i >= n_query) break;
      float mx = -INFINITY;
#pragma unroll
      for (int j = 0; j < kS; ++j)
        if (j < st.n) {
          w[i][j] = sycl::reduce_over_group(sg, dq[i][j], sycl::plus<float>()) * inv[j] * sc;
          mx = sycl::fmax(mx, w[i][j]);
        }
      float sum = 0;
#pragma unroll
      for (int j = 0; j < kS; ++j)
        if (j < st.n) sum += (w[i][j] = sycl::exp(w[i][j] - mx));
#pragma unroll
      for (int j = 0; j < kS; ++j)
        if (j < st.n) w[i][j] = w[i][j] / sum * inv[j];
    }
    for (int64_t e = lane; e < n_embd; e += kSg) {
      float x[kS];
#pragma unroll
      for (int j = 0; j < kS; ++j)
        if (j < st.n) x[j] = st.p[j][r * n_embd + e];
#pragma unroll
      for (int i = 0; i < kQ; ++i) {
        if (i >= n_query) break;
        float acc = 0;
#pragma unroll
        for (int j = 0; j < kS; ++j)
          if (j < st.n) acc += w[i][j] * x[j];
        out[(int64_t(i) * rows + r) * n_embd + e] = acc;
      }
    }
  });
}

void short_conv(sycl::queue& q, const float* x, int64_t ld_x, int64_t T, int64_t channels, const Chunks& ch, const int32_t* slot,
                const int32_t* base, float* conv, int64_t state_stride, int64_t hist_stride, const float* w, const float* dyn,
                int64_t heads, float* out) {
  const bool has_dyn = dyn != nullptr;
  const float s = 1.0f / std::sqrt(float((has_dyn ? 2 : 1) * 4 + 1));
  const int64_t per_head = has_dyn ? channels / heads : 1;
  q.parallel_for(sycl::range<2>(size_t(T), size_t(channels)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), c = int64_t(id[1]);
    const int64_t b = base[t];
    const float* hist = conv + int64_t(slot[t]) * state_stride;
    auto at = [&](int64_t back) {
      const int64_t p = t - back;
      return p >= b ? x[p * ld_x + c] : hist[(3 + (p - b)) * hist_stride + c];
    };
    float acc = 0;
    for (int k = 0; k < 4; ++k) acc += w[c * 4 + k] * at(3 - k);
    if (has_dyn) {
      const float* d = dyn + (t * heads + c / per_head) * 4;
      for (int lag = 0; lag < 4; ++lag) acc += d[lag] * at(lag);
    }
    out[t * channels + c] = (x[t * ld_x + c] + acc) * s;
  });
  // the new state: each chunk's last three inputs, older state where the chunk has fewer
  const Chunks cc = ch;
  q.parallel_for(sycl::range<2>(size_t(cc.count), size_t(channels)), [=](sycl::id<2> id) {
    const int k = int(id[0]);
    const int64_t c = int64_t(id[1]);
    const int64_t b = cc.base[k], n = cc.n[k];
    float* hist = conv + int64_t(cc.slot[k]) * state_stride;
    float h[3];
    for (int64_t i = 0; i < 3; ++i) {
      const int64_t p = n - 3 + i;  // within the chunk
      h[i] = p >= 0 ? x[(b + p) * ld_x + c] : hist[(3 + p) * hist_stride + c];
    }
    for (int64_t i = 0; i < 3; ++i) hist[i * hist_stride + c] = h[i];
  });
}

void heads(sycl::queue& q, const float* qr, int64_t ld_qr, const float* k, const float* v, const int32_t* pos, int64_t T,
           int64_t n_head, int64_t n_kv, int64_t D, int64_t n_rel, const float* q_clip, const float* rel_clip, float qk_clip,
           float rel_clip_t, bool full, float alpha, float floor, float eps, float* Q, float* R, float* K, float* V) {
  // rows: n_head of Q, n_head of R, n_kv of K, n_kv of V, per token; one sub-group of 16 a row
  constexpr int kSg = 16;
  const int64_t per = 2 * n_head + 2 * n_kv;
  q.parallel_for(sycl::nd_range<1>(size_t(T * per * kSg), 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
    const int64_t g = int64_t(it.get_global_id(0)) / kSg;
    const int lane = int(it.get_sub_group().get_local_id()[0]);
    const int64_t t = g / per, j = g % per;
    float tau = 1.0f;
    if (full) tau = alpha * sycl::log(sycl::fmax(1.0f, (float(pos[t]) + 1.0f) * (1.0f / floor))) + 1.0f;
    const float* src;
    const float* scale = nullptr;
    float* dst;
    int64_t n;
    float thr = 0, mul = 1.0f;
    bool norm = false;
    if (j < n_head) {
      src = qr + t * ld_qr + j * D, scale = q_clip + j * D, dst = Q + (t * n_head + j) * D, n = D, thr = qk_clip, mul = tau;
    } else if (j < 2 * n_head) {
      const int64_t h = j - n_head;
      src = qr + t * ld_qr + n_head * D + h * n_rel, scale = rel_clip + h * n_rel, dst = R + (t * n_head + h) * n_rel, n = n_rel,
      thr = rel_clip_t, mul = tau;
    } else if (j < 2 * n_head + n_kv) {
      const int64_t h = j - 2 * n_head;
      src = k + (t * n_kv + h) * D, dst = K + (t * n_kv + h) * D, n = D, thr = qk_clip;
    } else {
      const int64_t h = j - 2 * n_head - n_kv;
      src = v + (t * n_kv + h) * D, dst = V + (t * n_kv + h) * D, n = D, norm = true;
    }
    float ss = 0;
    for (int64_t i = lane; i < n; i += kSg) {
      const float a = scale ? src[i] * scale[i] : src[i];
      ss += a * a;
    }
    ss = sycl::reduce_over_group(it.get_sub_group(), ss, sycl::plus<float>());
    float f;
    if (norm) f = 1.0f / sycl::sqrt(ss / float(n) + eps);
    else f = sycl::clamp(thr / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f) * mul;
    for (int64_t i = lane; i < n; i += kSg) dst[i] = (scale ? src[i] * scale[i] : src[i]) * f;
  });
}

void gate(sycl::queue& q, float* att, const float* g, int64_t ld_g, int64_t T, int64_t n) {
  q.parallel_for(sycl::range<2>(size_t(T), size_t(n)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
    att[t * n + i] *= 1.0f / (1.0f + sycl::exp(-g[t * ld_g + i]));
  });
}

void matmul_f32(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t ld_x, int64_t T, float* out,
                int64_t ld_out) {
  constexpr int kSg = 16, kWg = 128;
  const int64_t outs = T * n, groups = (outs * kSg + kWg - 1) / kWg;
  q.parallel_for(sycl::nd_range<1>(size_t(groups * kWg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
    const int64_t o = int64_t(it.get_global_id(0)) / kSg;
    const int lane = int(it.get_sub_group().get_local_id()[0]);
    const bool live = o < outs;
    const int64_t t = live ? o / n : 0, j = live ? o % n : 0;
    float acc = 0;
    if (live)
      for (int64_t i = lane; i < k; i += kSg) acc += w[j * k + i] * x[t * ld_x + i];
    acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
    if (live && lane == 0) out[t * ld_out + j] = acc;
  });
}

void control_input(sycl::queue& q, const float* cur, const float* msg, int64_t T, int64_t n_embd, float eps, float* out) {
  constexpr int kWg = 256;
  q.parallel_for(sycl::nd_range<1>(size_t(T * 2 * kWg), kWg), [=](sycl::nd_item<1> it) {
    const int64_t g = int64_t(it.get_group(0)), t = g / 2, which = g % 2;
    const int lid = int(it.get_local_id(0));
    const float* x = (which == 0 ? cur : msg) + t * n_embd;
    float ss = 0;
    for (int64_t e = lid; e < n_embd; e += kWg) ss += x[e] * x[e];
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(n_embd) + eps);
    for (int64_t e = lid; e < n_embd; e += kWg) out[t * 2 * n_embd + which * n_embd + e] = x[e] * inv;
  });
}

void gather_rows(sycl::queue& q, const float* x, const int32_t* idx, int64_t rows, int64_t n, float* out) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(n)), [=](sycl::id<2> id) {
    out[int64_t(id[0]) * n + int64_t(id[1])] = x[int64_t(idx[id[0]]) * n + int64_t(id[1])];
  });
}

}  // namespace eklow::inkling
