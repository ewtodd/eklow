#include "kernels/inkling_fast.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eklow::inkling {

namespace {

// Few rows (decode): one 256-lane work-group a row.
void depth_mix_wide(sycl::queue& q, const States& st, const float* query, int n_query, int64_t rows, int64_t n_embd, float eps,
                    float* out) {
  // A work-group a row. Its sums (each state's squares, each state against each query: up to 40) were 40 group reductions
  // one after another, each with its barriers (0.11-0.15 ms a call at one token); they reduce over each sub-group by
  // shuffles, meet in local memory once, and every thread reads the totals.
  constexpr int kWg = 256, kS = 8, kQ = 4, kSg = 16, kNsg = kWg / kSg, kV = kS + kQ * kS;
  q.submit([&](sycl::handler& h) {
  sycl::local_accessor<float, 1> part(sycl::range<1>(kNsg * kV), h);
  sycl::local_accessor<float, 1> tot(sycl::range<1>(kV), h);
  h.parallel_for(sycl::nd_range<1>(size_t(rows * kWg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
    const int64_t r = int64_t(it.get_group(0));
    const int lid = int(it.get_local_id(0));
    const auto sg = it.get_sub_group();
    const int lane = int(sg.get_local_id()[0]), sgid = int(sg.get_group_id()[0]);
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
    for (int j = 0; j < kS; ++j) {
      if (j >= st.n) break;
      const float a = sycl::reduce_over_group(sg, ss[j], sycl::plus<float>());
      if (lane == 0) part[sgid * kV + j] = a;
      for (int i = 0; i < kQ; ++i) {
        if (i >= n_query) break;
        const float b = sycl::reduce_over_group(sg, dq[i][j], sycl::plus<float>());
        if (lane == 0) part[sgid * kV + kS + i * kS + j] = b;
      }
    }
    sycl::group_barrier(it.get_group());
    if (lid < kV) {
      float a = 0;
      for (int g = 0; g < kNsg; ++g) a += part[g * kV + lid];
      tot[lid] = a;
    }
    sycl::group_barrier(it.get_group());
    float inv[kS], w[kQ][kS];
    for (int j = 0; j < kS; ++j) {
      if (j >= st.n) break;
      inv[j] = 1.0f / sycl::sqrt(tot[j] / float(n_embd) + eps);
      for (int i = 0; i < kQ; ++i) {
        if (i >= n_query) break;
        w[i][j] = tot[kS + i * kS + j];
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
  });
}

// A decode step's rows: a work-group a row was one work-group on the card (60 us a call). Each row splits into slices of
// 256 across work-groups: the first kernel writes each slice's sums (squares and products with the queries, 40 at most),
// the second adds a row's slices, takes the weights and writes the slice's output.
void depth_mix_few(sycl::queue& q, const States& st, const float* query, int n_query, int64_t rows, int64_t n_embd, float eps,
                   float* out, float* part) {
  constexpr int kWg = 256, kS = 8, kQ = 4, kSg = 16, kNsg = kWg / kSg, kV = kS + kQ * kS;
  const int64_t slices = (n_embd + kWg - 1) / kWg;
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> sub(sycl::range<1>(kNsg * kV), h);
    h.parallel_for(sycl::nd_range<1>(size_t(rows * slices * kWg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
      const int64_t g = int64_t(it.get_group(0)), r = g / slices, sl = g % slices;
      const int lid = int(it.get_local_id(0));
      const auto sg = it.get_sub_group();
      const int lane = int(sg.get_local_id()[0]), sgid = int(sg.get_group_id()[0]);
      const int64_t e = sl * kWg + lid;
      float qv[kQ];
#pragma unroll
      for (int i = 0; i < kQ; ++i) qv[i] = (i < n_query && e < n_embd) ? query[int64_t(i) * n_embd + e] : 0.0f;
#pragma unroll
      for (int j = 0; j < kS; ++j) {
        if (j >= st.n) break;
        const float x = e < n_embd ? st.p[j][r * n_embd + e] : 0.0f;
        const float a = sycl::reduce_over_group(sg, x * x, sycl::plus<float>());
        if (lane == 0) sub[sgid * kV + j] = a;
#pragma unroll
        for (int i = 0; i < kQ; ++i) {
          if (i >= n_query) break;
          const float b = sycl::reduce_over_group(sg, x * qv[i], sycl::plus<float>());
          if (lane == 0) sub[sgid * kV + kS + i * kS + j] = b;
        }
      }
      sycl::group_barrier(it.get_group());
      if (lid < kV) {
        float a = 0;
        for (int k = 0; k < kNsg; ++k) a += sub[k * kV + lid];
        part[(r * slices + sl) * kV + lid] = a;
      }
    });
  });
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> wts(sycl::range<1>(kQ * kS), h);
    h.parallel_for(sycl::nd_range<1>(size_t(rows * slices * kWg), kWg), [=](sycl::nd_item<1> it) {
      const int64_t g = int64_t(it.get_group(0)), r = g / slices, sl = g % slices;
      const int lid = int(it.get_local_id(0));
      if (lid == 0) {
        float tot[kV];
        for (int v = 0; v < kV; ++v) tot[v] = 0;
        for (int64_t k = 0; k < slices; ++k)
          for (int v = 0; v < kV; ++v) tot[v] += part[(r * slices + k) * kV + v];
        const float sc = 1.0f / sycl::sqrt(float(n_embd));
        for (int i = 0; i < kQ; ++i) {
          if (i >= n_query) break;
          float w[kS], inv[kS], mx = -INFINITY;
          for (int j = 0; j < kS; ++j) {
            if (j >= st.n) break;
            inv[j] = 1.0f / sycl::sqrt(tot[j] / float(n_embd) + eps);
            w[j] = tot[kS + i * kS + j] * inv[j] * sc;
            mx = sycl::fmax(mx, w[j]);
          }
          float sum = 0;
          for (int j = 0; j < kS; ++j) {
            if (j >= st.n) break;
            sum += (w[j] = sycl::exp(w[j] - mx));
          }
          for (int j = 0; j < kS; ++j) {
            if (j >= st.n) break;
            wts[i * kS + j] = w[j] / sum * inv[j];
          }
        }
      }
      sycl::group_barrier(it.get_group());
      const int64_t e = sl * kWg + lid;
      if (e >= n_embd) return;
      float x[kS];
      for (int j = 0; j < kS; ++j)
        if (j < st.n) x[j] = st.p[j][r * n_embd + e];
      for (int i = 0; i < kQ; ++i) {
        if (i >= n_query) break;
        float acc = 0;
        for (int j = 0; j < kS; ++j)
          if (j < st.n) acc += wts[i * kS + j] * x[j];
        out[(int64_t(i) * rows + r) * n_embd + e] = acc;
      }
    });
  });
}

}  // namespace

void depth_mix(sycl::queue& q, const States& s, const float* query, int n_query, int64_t rows, int64_t n_embd, float eps, float* out,
               float* scratch) {
  if (s.n < 1 || s.n > 8 || n_query < 1 || n_query > 4) throw std::runtime_error("inkling::depth_mix: bad state or query count");
  // With scratch, every row count: rows split across work-groups. At a prefill sub-batch of 1,024 rows it beats a sub-group
  // a row (whose 16 lanes walked 6 states x 2,304 twice: in.mix + ffn.mix 245 ms of a 1,024-token prefill).
  if (scratch) return depth_mix_few(q, s, query, n_query, rows, n_embd, eps, out, scratch);
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
  // Few outputs over long rows (the controllers' 4,608-wide down projection, the router, the sinks, the dynamic taps at a
  // decode step): a sub-group an output left the card nearly idle while each walked its row 16 floats at a time (80-180 us
  // a call). A work-group an output splits the row 256 ways and reduces once.
  if (k >= 1024 && T * n <= 8192) {
    constexpr int kWgRow = 256;
    q.parallel_for(sycl::nd_range<1>(size_t(T * n * kWgRow), kWgRow), [=](sycl::nd_item<1> it) {
      const int64_t o = int64_t(it.get_group(0)), t = o / n, j = o % n;
      const int lid = int(it.get_local_id(0));
      const float* wr = w + j * k;
      const float* xr = x + t * ld_x;
      float acc = 0;
      for (int64_t i = lid; i < k; i += kWgRow) acc += wr[i] * xr[i];
      acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
      if (lid == 0) out[t * ld_out + j] = acc;
    });
    return;
  }
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

namespace {

// A sub-group a token, PER experts a lane (n_expert = 16 PER): with PER a compile-time count the values stay in registers
// (a runtime bound left them in private memory: 42 us a call).
template <int PER>
void route_kernel(sycl::queue& q, const float* logits, const float* bias, int64_t T, int64_t n_used, float w_scale, int32_t* ids,
                  float* weights) {
  constexpr int kSg = 16;
  constexpr int64_t NE = int64_t(PER) * kSg;
  q.parallel_for(sycl::nd_range<1>(size_t(T * kSg), kSg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
    const auto sg = it.get_sub_group();
    const int64_t t = int64_t(it.get_group(0));
    const int lane = int(sg.get_local_id()[0]);
    const float* l = logits + t * NE;
    float v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int64_t e = int64_t(i) * kSg + lane;
      v[i] = 1.0f / (1.0f + sycl::exp(-l[e])) + (bias ? bias[e] : 0.0f);
    }
    int32_t mine = -1;  // lane s keeps the s-th pick
    for (int s = 0; s < int(n_used); ++s) {
      float bv = -INFINITY;
      int32_t be = INT32_MAX;
#pragma unroll
      for (int i = 0; i < PER; ++i) {
        const int32_t e = int32_t(i * kSg + lane);
        if (v[i] > bv || (v[i] == bv && e < be)) bv = v[i], be = e;
      }
      const float m = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
      const int32_t win = sycl::reduce_over_group(sg, bv == m ? be : INT32_MAX, sycl::minimum<int32_t>());
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (i * kSg + lane == win) v[i] = -INFINITY;
      if (lane == s) mine = win;
    }
    const float g = lane < int(n_used) ? 1.0f / (1.0f + sycl::exp(-l[mine])) : 0.0f;
    const float sum = sycl::reduce_over_group(sg, g, sycl::plus<float>());
    if (lane < int(n_used)) {
      ids[t * n_used + lane] = mine;
      weights[t * n_used + lane] = g / (sum + 1e-20f) * w_scale;
    }
  });
}

}  // namespace

void route(sycl::queue& q, const float* logits, const float* bias, int64_t T, int64_t n_expert, int64_t n_used, float w_scale,
           int32_t* ids, float* weights) {
  if (n_used > 16) throw std::runtime_error("inkling::route: at most 16 experts a token");
  switch (n_expert) {
    case 64: return route_kernel<4>(q, logits, bias, T, n_used, w_scale, ids, weights);
    case 128: return route_kernel<8>(q, logits, bias, T, n_used, w_scale, ids, weights);
    case 256: return route_kernel<16>(q, logits, bias, T, n_used, w_scale, ids, weights);
    default: throw std::runtime_error("inkling::route: 64, 128 or 256 experts");
  }
}

void control(sycl::queue& q, const float* cur, const float* msg, const float* down, const float* gain, const float* forget, int64_t T,
             int64_t n_embd, int64_t rank, float eps, float* cd, float* out) {
  // 1. cd[t][r] = down[r] . [cur/rms(cur), msg/rms(msg)]: a work-group an output, the two sums of squares and the two dot
  //    products in one pass, reduced over sub-groups and once through local memory.
  {
    constexpr int kWg = 256, kSg = 16, kNsg = kWg / kSg;
    q.submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> part(sycl::range<1>(kNsg * 4), h);
      h.parallel_for(sycl::nd_range<1>(size_t(T * rank * kWg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
        const int64_t o = int64_t(it.get_group(0)), t = o / rank, r = o % rank;
        const int lid = int(it.get_local_id(0));
        const auto sg = it.get_sub_group();
        const float* c = cur + t * n_embd;
        const float* m = msg + t * n_embd;
        const float* w = down + r * 2 * n_embd;
        float sc = 0, sm = 0, dc = 0, dm = 0;
        for (int64_t e = lid; e < n_embd; e += kWg) {
          const float xc = c[e], xm = m[e];
          sc += xc * xc, sm += xm * xm;
          dc += w[e] * xc, dm += w[n_embd + e] * xm;
        }
        const int g = int(sg.get_group_id()[0]);
        const float a0 = sycl::reduce_over_group(sg, sc, sycl::plus<float>()), a1 = sycl::reduce_over_group(sg, sm, sycl::plus<float>());
        const float a2 = sycl::reduce_over_group(sg, dc, sycl::plus<float>()), a3 = sycl::reduce_over_group(sg, dm, sycl::plus<float>());
        if (sg.get_local_id()[0] == 0) part[g * 4] = a0, part[g * 4 + 1] = a1, part[g * 4 + 2] = a2, part[g * 4 + 3] = a3;
        sycl::group_barrier(it.get_group());
        if (lid == 0) {
          float b0 = 0, b1 = 0, b2 = 0, b3 = 0;
          for (int i = 0; i < kNsg; ++i) b0 += part[i * 4], b1 += part[i * 4 + 1], b2 += part[i * 4 + 2], b3 += part[i * 4 + 3];
          const float ic = 1.0f / sycl::sqrt(b0 / float(n_embd) + eps), im = 1.0f / sycl::sqrt(b1 / float(n_embd) + eps);
          cd[t * rank + r] = b2 * ic + b3 * im;
        }
      });
    });
  }
  // 2. out = cur - tanh(forget . cd) cur + tanh(gain . cd) msg: a sub-group an element, its lanes along the rank, so that
  //    the gain and forget rows are read in 64-byte runs (a work-item an element read them 256 bytes apart).
  {
    constexpr int kSg = 16, kWg = 128;
    const int64_t outs = T * n_embd;
    q.parallel_for(sycl::nd_range<1>(size_t((outs * kSg + kWg - 1) / kWg * kWg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
      const int64_t o = int64_t(it.get_global_id(0)) / kSg;
      const auto sg = it.get_sub_group();
      const int lane = int(sg.get_local_id()[0]);
      const bool live = o < outs;
      const int64_t t = live ? o / n_embd : 0, e = live ? o % n_embd : 0;
      float g = 0, f = 0;
      if (live)
        for (int64_t k = lane; k < rank; k += kSg) {
          const float d = cd[t * rank + k];
          g += gain[e * rank + k] * d, f += forget[e * rank + k] * d;
        }
      g = sycl::reduce_over_group(sg, g, sycl::plus<float>());
      f = sycl::reduce_over_group(sg, f, sycl::plus<float>());
      if (live && lane == 0) {
        const float c = cur[t * n_embd + e];
        out[t * n_embd + e] = c - sycl::tanh(f) * c + sycl::tanh(g) * msg[t * n_embd + e];
      }
    });
  }
}

void gather_cols(sycl::queue& q, const float* x, int64_t ld, int64_t rows, int64_t n, float* out) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(n)), [=](sycl::id<2> id) {
    out[int64_t(id[0]) * n + int64_t(id[1])] = x[int64_t(id[0]) * ld + int64_t(id[1])];
  });
}

void contrast(sycl::queue& q, float* y, const float* first, float w, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) { y[i] += w * (y[i] - first[i]); });
}

sycl::event pack_host(sycl::queue& q, const int32_t* ids, const float* w, const float* x, int64_t T, int64_t n_used, int64_t n_embd,
                      int32_t* hids, float* hw, float* hx) {
  return q.parallel_for(sycl::range<2>(size_t(T), size_t(n_embd)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), e = int64_t(id[1]);
    hx[t * n_embd + e] = x[t * n_embd + e];
    if (e < n_used) {
      hids[t * n_used + e] = ids[t * n_used + e];
      hw[t * n_used + e] = w[t * n_used + e];
    }
  });
}

}  // namespace eklow::inkling
