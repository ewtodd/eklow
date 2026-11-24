#include "kernels/ref_inkling.hpp"

#include <algorithm>
#include <cmath>

namespace eklow::ref::inkling {

void depth_mix(sycl::queue& q, const float* const* states, int n_states, const float* query, int n_query, int64_t n_tok,
               int64_t n_embd, float eps, float* out) {
  constexpr int kMaxStates = 8, kMaxQuery = 4;
  q.parallel_for(sycl::range<1>(size_t(n_tok)), [=](sycl::id<1> id) {
     const int64_t t = int64_t(id[0]);
     float inv[kMaxStates], w[kMaxQuery][kMaxStates];
     for (int s = 0; s < n_states; ++s) {
       const float* x = states[s] + t * n_embd;
       float ss = 0;
       for (int64_t e = 0; e < n_embd; ++e) ss += x[e] * x[e];
       inv[s] = 1.0f / sycl::sqrt(ss / float(n_embd) + eps);
     }
     const float sc = 1.0f / sycl::sqrt(float(n_embd));
     for (int qi = 0; qi < n_query; ++qi) {
       const float* qv = query + int64_t(qi) * n_embd;
       float mx = -INFINITY;
       for (int s = 0; s < n_states; ++s) {
         const float* x = states[s] + t * n_embd;
         float d = 0;
         for (int64_t e = 0; e < n_embd; ++e) d += x[e] * qv[e];
         w[qi][s] = d * inv[s] * sc;
         mx = sycl::fmax(mx, w[qi][s]);
       }
       float sum = 0;
       for (int s = 0; s < n_states; ++s) sum += (w[qi][s] = sycl::exp(w[qi][s] - mx));
       for (int s = 0; s < n_states; ++s) w[qi][s] = w[qi][s] / sum * inv[s];
     }
     for (int64_t e = 0; e < n_embd; ++e) {
       float v[kMaxStates];
       for (int s = 0; s < n_states; ++s) v[s] = states[s][t * n_embd + e];
       for (int qi = 0; qi < n_query; ++qi) {
         float acc = 0;
         for (int s = 0; s < n_states; ++s) acc += w[qi][s] * v[s];
         out[(int64_t(qi) * n_tok + t) * n_embd + e] = acc;
       }
     }
   });
}

void short_conv(sycl::queue& q, const float* x, int64_t n_tok, int64_t channels, float* hist, int64_t hist_stride,
                const float* w, const float* dyn_or_null, int64_t heads, float* out) {
  // SYCL's native_cpu device refuses a null pointer as a kernel argument: flags, and x in place of a missing input
  const bool has_dyn = dyn_or_null != nullptr;
  const float* dyn = has_dyn ? dyn_or_null : x;
  const float s = 1.0f / std::sqrt(float((has_dyn ? 2 : 1) * 4 + 1));
  const int64_t per_head = has_dyn ? channels / heads : 1;
  // x at position t - back (back 0..3), from the history before position 0
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(channels)), [=](sycl::id<2> id) {
     const int64_t t = int64_t(id[0]), c = int64_t(id[1]);
     auto at = [&](int64_t back) {
       const int64_t p = t - back;
       return p >= 0 ? x[p * channels + c] : hist[(3 + p) * hist_stride + c];
     };
     float conv = 0;
     for (int k = 0; k < 4; ++k) conv += w[c * 4 + k] * at(3 - k);
     if (has_dyn) {
       const float* d = dyn + (t * heads + c / per_head) * 4;
       for (int lag = 0; lag < 4; ++lag) conv += d[lag] * at(lag);
     }
     out[t * channels + c] = (x[t * channels + c] + conv) * s;
   });
  // the new history: the last three inputs (older history where there are fewer than three)
  q.single_task([=]() {
     for (int64_t c = 0; c < channels; ++c) {
       float h[3];
       for (int64_t i = 0; i < 3; ++i) {
         const int64_t p = n_tok - 3 + i;
         h[i] = p >= 0 ? x[p * channels + c] : hist[(3 + p) * hist_stride + c];
       }
       for (int64_t i = 0; i < 3; ++i) hist[i * hist_stride + c] = h[i];
     }
   });
}

void l2_clip(sycl::queue& q, float* x, int64_t n_rows, int64_t heads, int64_t dim, const float* scale_or_null,
             float threshold) {
  const bool has_scale = scale_or_null != nullptr;
  const float* scale = has_scale ? scale_or_null : x;
  q.parallel_for(sycl::range<1>(size_t(n_rows)), [=](sycl::id<1> id) {
     const int64_t r = int64_t(id[0]);
     float* v = x + r * dim;
     const float* sc = scale + (r % heads) * dim;
     float ss = 0;
     for (int64_t i = 0; i < dim; ++i) {
       if (has_scale) v[i] *= sc[i];
       ss += v[i] * v[i];
     }
     const float f = sycl::clamp(threshold / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
     for (int64_t i = 0; i < dim; ++i) v[i] *= f;
   });
}

void scale_tokens(sycl::queue& q, float* x, const float* s, int64_t n_tok, int64_t cols) {
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(cols)),
                 [=](sycl::id<2> id) { x[int64_t(id[0]) * cols + int64_t(id[1])] *= s[id[0]]; });
}

void attention(sycl::queue& q, const float* Q, const float* K, const float* V, const float* R, const float* table,
               const float* sink, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, int64_t n_rel,
               int64_t extent, int64_t window, float scale, float* out) {
  const int64_t group = n_head / n_head_kv;
  // a launch of every query of a few heads stays far from xe's job timeout on the card
  for (int64_t h0 = 0; h0 < n_head; ++h0) {
    q.parallel_for(sycl::range<1>(size_t(n_tok)), [=](sycl::id<1> id) {
       const int64_t t = int64_t(id[0]), h = h0, hk = h / group;
       const float* qv = Q + (t * n_head + h) * head_dim;
       const float* rv = R + (t * n_head + h) * n_rel;
       float* o = out + (t * n_head + h) * head_dim;
       for (int64_t d = 0; d < head_dim; ++d) o[d] = 0;
       // the sink is a key of logit `sink` and a zero value
       float m = sink[t * n_head + h], l = 1.0f;
       const int64_t j0 = window > 0 ? sycl::max(int64_t(0), t - window + 1) : 0;
       for (int64_t j = j0; j <= t; ++j) {
         const float* kv = K + (j * n_head_kv + hk) * head_dim;
         float s = 0;
         for (int64_t d = 0; d < head_dim; ++d) s += qv[d] * kv[d];
         s *= scale;
         const int64_t dist = t - j;
         if (dist < extent) {
           const float* tb = table + dist * n_rel;
           float b = 0;
           for (int64_t i = 0; i < n_rel; ++i) b += rv[i] * tb[i];
           s += b;
         }
         const float* vv = V + (j * n_head_kv + hk) * head_dim;
         if (s > m) {
           const float c = sycl::exp(m - s);
           l = l * c + 1.0f;
           for (int64_t d = 0; d < head_dim; ++d) o[d] = o[d] * c + vv[d];
           m = s;
         } else {
           const float p = sycl::exp(s - m);
           l += p;
           for (int64_t d = 0; d < head_dim; ++d) o[d] += p * vv[d];
         }
       }
       for (int64_t d = 0; d < head_dim; ++d) o[d] /= l;
     });
  }
}

void sigmoid_mul(sycl::queue& q, float* x, const float* g, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) { x[i] *= 1.0f / (1.0f + sycl::exp(-g[i])); });
}

void route(sycl::queue& q, const float* logits, const float* bias_or_null, int64_t n_tok, int64_t n_expert,
           int64_t n_used, float w_scale, int32_t* ids, float* weights) {
  const bool has_bias = bias_or_null != nullptr;
  const float* bias = has_bias ? bias_or_null : logits;
  q.parallel_for(sycl::range<1>(size_t(n_tok)), [=](sycl::id<1> id) {
     const int64_t t = int64_t(id[0]);
     const float* l = logits + t * n_expert;
     int32_t* it = ids + t * n_used;
     float* wt = weights + t * n_used;
     // selection by insertion into the n_used best (n_used is 8)
     float best[16];
     for (int64_t s = 0; s < n_used; ++s) best[s] = -INFINITY, it[s] = -1;
     for (int64_t e = 0; e < n_expert; ++e) {
       const float v = 1.0f / (1.0f + sycl::exp(-l[e])) + (has_bias ? bias[e] : 0.0f);
       if (v <= best[n_used - 1]) continue;
       int64_t s = n_used - 1;
       while (s > 0 && v > best[s - 1]) {
         best[s] = best[s - 1];
         it[s] = it[s - 1];
         --s;
       }
       best[s] = v;
       it[s] = int32_t(e);
     }
     float sum = 0;
     for (int64_t s = 0; s < n_used; ++s) sum += (wt[s] = 1.0f / (1.0f + sycl::exp(-l[it[s]])));
     for (int64_t s = 0; s < n_used; ++s) wt[s] = wt[s] / (sum + 1e-20f) * w_scale;
   });
}

void situ(sycl::queue& q, const float* gate_up, int64_t rows, int64_t n_ff, float beta, float lb, float* out) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(n_ff)), [=](sycl::id<2> id) {
     const int64_t r = int64_t(id[0]), i = int64_t(id[1]);
     const float g = gate_up[r * 2 * n_ff + i], u = gate_up[r * 2 * n_ff + n_ff + i];
     const float a = beta * sycl::tanh(g / beta) * (1.0f / (1.0f + sycl::exp(-g)));
     out[r * n_ff + i] = a * (lb > 0 ? lb * sycl::tanh(u / lb) : u);
   });
}

void moe_combine(sycl::queue& q, const float* down, const float* weights, int64_t n_tok, int64_t n_used, int64_t n_embd,
                 float* out) {
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(n_embd)), [=](sycl::id<2> id) {
     const int64_t t = int64_t(id[0]), e = int64_t(id[1]);
     float acc = 0;
     for (int64_t s = 0; s < n_used; ++s) acc += weights[t * n_used + s] * down[(t * n_used + s) * n_embd + e];
     out[t * n_embd + e] = acc;
   });
}

void control_input(sycl::queue& q, const float* cur, const float* msg, int64_t n_tok, int64_t n_embd, float eps, float* out) {
  q.parallel_for(sycl::range<2>(size_t(n_tok), 2), [=](sycl::id<2> id) {
     const int64_t t = int64_t(id[0]);
     const float* x = (id[1] == 0 ? cur : msg) + t * n_embd;
     float* o = out + t * 2 * n_embd + int64_t(id[1]) * n_embd;
     float ss = 0;
     for (int64_t e = 0; e < n_embd; ++e) ss += x[e] * x[e];
     const float inv = 1.0f / sycl::sqrt(ss / float(n_embd) + eps);
     for (int64_t e = 0; e < n_embd; ++e) o[e] = x[e] * inv;
   });
}

void control_mix(sycl::queue& q, const float* cur, const float* msg, const float* g, const float* f, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
     out[i] = cur[i] - sycl::tanh(f[i]) * cur[i] + sycl::tanh(g[i]) * msg[i];
   });
}

namespace {

// fp32 -> fp16 -> fp32, rounding to nearest even (F16C's _cvtss_sh, ggml's GGML_CPU_FP32_TO_FP16 on x86)
inline float round_fp16(float v) {
  uint32_t f = sycl::bit_cast<uint32_t>(v);
  const uint32_t sign = f & 0x80000000u;
  f ^= sign;
  uint32_t h;
  if (f >= (127u + 16u) << 23) {
    h = f > 0x7f800000u ? 0x7e00u : 0x7c00u;
  } else if (f < 113u << 23) {
    const uint32_t magic = ((127u - 15u) + (23u - 10u) + 1u) << 23;
    h = sycl::bit_cast<uint32_t>(sycl::bit_cast<float>(f) + sycl::bit_cast<float>(magic)) - magic;
  } else {
    const uint32_t odd = (f >> 13) & 1u;
    f += (uint32_t(15 - 127) << 23) + 0xfffu + odd;
    h = f >> 13;
  }
  // back to fp32
  const uint32_t e = (h >> 10) & 31u, m = h & 1023u;
  uint32_t r;
  if (e == 0) {
    r = m == 0 ? 0u : sycl::bit_cast<uint32_t>(float(m) * 5.9604644775390625e-8f);  // m * 2^-24
  } else if (e == 31) {
    r = 0x7f800000u | m << 13;
  } else {
    r = (e + 127u - 15u) << 23 | m << 13;
  }
  return sycl::bit_cast<float>(sign | r);
}

}  // namespace

void ggml_cpu_acts(sycl::queue& q, gguf::Type t, const float* x, int64_t rows, int64_t k, float* out) {
  if (t == gguf::Type::Q8_0 || t == gguf::Type::NVFP4 || t == gguf::Type::Q4_0) {
    q.parallel_for(sycl::range<2>(size_t(rows), size_t(k / 32)), [=](sycl::id<2> id) {
       const float* xb = x + int64_t(id[0]) * k + int64_t(id[1]) * 32;
       float* ob = out + int64_t(id[0]) * k + int64_t(id[1]) * 32;
       float amax = 0;
       for (int j = 0; j < 32; ++j) amax = sycl::fmax(amax, sycl::fabs(xb[j]));
       const float d = round_fp16(amax / 127.f), id_ = amax != 0.0f ? 127.f / amax : 0.0f;
       for (int j = 0; j < 32; ++j) ob[j] = sycl::rint(xb[j] * id_) * d;
     });
  } else if (t == gguf::Type::Q5_K || t == gguf::Type::Q4_K || t == gguf::Type::Q6_K) {
    q.parallel_for(sycl::range<2>(size_t(rows), size_t(k / 256)), [=](sycl::id<2> id) {
       const float* xb = x + int64_t(id[0]) * k + int64_t(id[1]) * 256;
       float* ob = out + int64_t(id[0]) * k + int64_t(id[1]) * 256;
       float amax = 0, mx = 0;
       for (int j = 0; j < 256; ++j)
         if (sycl::fabs(xb[j]) > amax) amax = sycl::fabs(xb[j]), mx = xb[j];
       if (amax == 0) {
         for (int j = 0; j < 256; ++j) ob[j] = 0;
         return;
       }
       const float is = -127.f / mx, d = 1.0f / is;
       for (int j = 0; j < 256; ++j) ob[j] = sycl::fmin(127.0f, sycl::rint(is * xb[j])) * d;
     });
  } else if (t == gguf::Type::BF16) {
    q.parallel_for(sycl::range<1>(size_t(rows * k)), [=](sycl::id<1> i) {
       uint32_t b = sycl::bit_cast<uint32_t>(x[i]);
       if ((b & 0x7fffffffu) <= 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
       out[i] = sycl::bit_cast<float>(b & 0xffff0000u);
     });
  } else if (out != x) {
    q.memcpy(out, x, size_t(rows * k) * sizeof(float));
  }
}

}  // namespace eklow::ref::inkling
