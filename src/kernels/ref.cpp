#include "kernels/ref.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace eklow::ref {

namespace {

inline float half_at(const uint8_t* p) {
  uint16_t bits = uint16_t(p[0]) | uint16_t(p[1]) << 8;
  return float(sycl::bit_cast<sycl::half>(bits));
}

// Q4_0 (32 weights, 18 bytes): fp16 d, then 16 bytes; byte j holds weight j in its low nibble and
// weight j + 16 in its high nibble; w = (nibble - 8) * d.
inline float dot_q4_0(const uint8_t* row, const float* x, int64_t k) {
  float acc = 0;
  for (int64_t b = 0; b < k / 32; ++b) {
    const uint8_t* blk = row + b * 18;
    const float d = half_at(blk);
    const float* xb = x + b * 32;
    float s = 0;
    for (int j = 0; j < 16; ++j) {
      const uint8_t v = blk[2 + j];
      s += float(int(v & 15) - 8) * xb[j] + float(int(v >> 4) - 8) * xb[j + 16];
    }
    acc += d * s;
  }
  return acc;
}

// Q6_K (256 weights, 210 bytes): ql[128], qh[64], int8 scales[16], fp16 d; ggml's
// dequantize_row_q6_K, two halves of 128 weights.
inline float q6k_weight(const uint8_t* blk, int i) {
  const uint8_t* ql = blk;
  const uint8_t* qh = blk + 128;
  const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192);
  const float d = half_at(blk + 208);
  const int half = i / 128, r = i % 128, l = r % 32, quarter = r / 32;
  ql += half * 64;
  qh += half * 32;
  sc += half * 8;
  const int is = l / 16;
  int q;
  switch (quarter) {
    case 0: q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
    case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
    case 2: q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); break;
    default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
  }
  return d * float(sc[is + 2 * quarter]) * float(q - 32);
}

inline float dot_q6_k(const uint8_t* row, const float* x, int64_t k) {
  float acc = 0;
  for (int64_t b = 0; b < k / 256; ++b) {
    const uint8_t* blk = row + b * 210;
    float s = 0;
    for (int i = 0; i < 256; ++i) s += q6k_weight(blk, i) * x[b * 256 + i];
    acc += s;
  }
  return acc;
}

inline float weight_at(gguf::Type t, const uint8_t* row, int64_t i) {
  switch (t) {
    case gguf::Type::F32: return reinterpret_cast<const float*>(row)[i];
    case gguf::Type::F16: return half_at(row + 2 * i);
    case gguf::Type::Q4_0: {
      const uint8_t* blk = row + (i / 32) * 18;
      const int j = int(i % 32);
      const uint8_t v = blk[2 + (j % 16)];
      return half_at(blk) * float(int(j < 16 ? v & 15 : v >> 4) - 8);
    }
    case gguf::Type::Q6_K: return q6k_weight(row + (i / 256) * 210, int(i % 256));
    default: return 0;
  }
}

void check_type(const Weight& w) {
  switch (w.type) {
    case gguf::Type::F32:
    case gguf::Type::F16:
    case gguf::Type::Q4_0:
    case gguf::Type::Q6_K: return;
    default: throw std::runtime_error(std::string("ref: no reference kernel for ") + gguf::type_name(w.type));
  }
}

}  // namespace

void embed(sycl::queue& q, const Weight& table, const int32_t* tokens, int64_t n_tok, float scale_by, float* out) {
  check_type(table);
  const Weight w = table;
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(w.k)), [=](sycl::id<2> id) {
     const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
     const uint8_t* row = w.data + uint64_t(tokens[t]) * w.row_bytes;
     out[t * w.k + i] = weight_at(w.type, row, i) * scale_by;
   }).wait();
}

void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out) {
  q.parallel_for(sycl::range<1>(size_t(rows)), [=](sycl::id<1> id) {
     const int64_t r = int64_t(id[0]);
     const float* xr = x + r * cols;
     float ss = 0;
     for (int64_t i = 0; i < cols; ++i) ss += xr[i] * xr[i];
     const float inv = 1.0f / sycl::sqrt(ss / float(cols) + eps);
     for (int64_t i = 0; i < cols; ++i) out[r * cols + i] = xr[i] * inv * (w ? w[i] : 1.0f);
   }).wait();
}

void matmul(sycl::queue& q, const Weight& w, const float* x, int64_t m_rows, float* out) {
  check_type(w);
  // Rows per launch: a fraction of a second for these kernels, about 2e10 multiply-adds for Q4_0;
  // Q6_K and the generic path decode each weight separately and get a tenth of that.
  const double budget = w.type == gguf::Type::Q4_0 ? 2e10 : 2e9;
  const int64_t per = std::max<int64_t>(1, int64_t(budget / double(w.n * w.k)));
  for (int64_t m0 = 0; m0 < m_rows; m0 += per) {
    const int64_t mc = std::min(per, m_rows - m0);
    const Weight ww = w;
    q.parallel_for(sycl::range<2>(size_t(mc), size_t(w.n)), [=](sycl::id<2> id) {
       const int64_t m = m0 + int64_t(id[0]), n = int64_t(id[1]);
       const uint8_t* row = ww.data + uint64_t(n) * ww.row_bytes;
       const float* xm = x + m * ww.k;
       float acc;
       if (ww.type == gguf::Type::Q4_0) acc = dot_q4_0(row, xm, ww.k);
       else if (ww.type == gguf::Type::Q6_K) acc = dot_q6_k(row, xm, ww.k);
       else {
         acc = 0;
         for (int64_t i = 0; i < ww.k; ++i) acc += weight_at(ww.type, row, i) * xm[i];
       }
       out[m * ww.n + n] = acc;
     }).wait();
  }
}

RopeTables rope_tables(int64_t n_tok, int64_t n_rot, double base, const float* freq_factors) {
  const int64_t half = n_rot / 2;
  RopeTables t{std::vector<float>(size_t(n_tok * half)), std::vector<float>(size_t(n_tok * half))};
  for (int64_t p = 0; p < n_tok; ++p)
    for (int64_t i = 0; i < half; ++i) {
      double theta = double(p) * std::pow(base, -2.0 * double(i) / double(n_rot));
      if (freq_factors) theta /= double(freq_factors[i]);
      t.cos[size_t(p * half + i)] = float(std::cos(theta));
      t.sin[size_t(p * half + i)] = float(std::sin(theta));
    }
  return t;
}

void rope_neox(sycl::queue& q, float* x, int64_t n_tok, int64_t n_head, int64_t head_dim, int64_t n_rot,
               const float* cos_tab, const float* sin_tab) {
  const int64_t half = n_rot / 2;
  q.parallel_for(sycl::range<3>(size_t(n_tok), size_t(n_head), size_t(half)), [=](sycl::id<3> id) {
     const int64_t t = int64_t(id[0]), h = int64_t(id[1]), i = int64_t(id[2]);
     const float c = cos_tab[t * half + i], s = sin_tab[t * half + i];
     float* v = x + (t * n_head + h) * head_dim;
     const float a = v[i], b = v[i + half];
     v[i] = a * c - b * s;
     v[i + half] = a * s + b * c;
   }).wait();
}

void attention(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t n_tok, int64_t n_head,
               int64_t n_head_kv, int64_t head_dim, int64_t window, float scale_by, float* out) {
  const int64_t group = n_head / n_head_kv;
  const int64_t keys = window > 0 ? std::min(window, n_tok) : n_tok;
  // Queries in launches of about 2e8 key visits each.
  const int64_t per = std::max<int64_t>(1, int64_t(2e8 / double(n_head * keys)));
  for (int64_t t0 = 0; t0 < n_tok; t0 += per) {
    const int64_t tc = std::min(per, n_tok - t0);
    // One work-group per (query, head), one work-item per dimension; each score is reduced across
    // the group and the softmax is kept online.
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(tc * n_head), size_t(head_dim)),
                                     sycl::range<2>(1, size_t(head_dim))),
                   [=](sycl::nd_item<2> it) {
                     const int64_t th = t0 * n_head + int64_t(it.get_global_id(0)), d = int64_t(it.get_local_id(1));
                     const int64_t t = th / n_head, h = th % n_head, kh = h / group;
                     const float qd = Q[(t * n_head + h) * head_dim + d];
                     const int64_t first = window > 0 ? std::max<int64_t>(0, t - window + 1) : 0;
                     float m = -INFINITY, l = 0, acc = 0;
                     for (int64_t j = first; j <= t; ++j) {
                       const float part = qd * K[(j * n_head_kv + kh) * head_dim + d];
                       const float sc = sycl::reduce_over_group(it.get_group(), part, sycl::plus<float>()) * scale_by;
                       const float mn = sycl::fmax(m, sc);
                       const float corr = sycl::exp(m - mn), p = sycl::exp(sc - mn);
                       l = l * corr + p;
                       acc = acc * corr + p * V[(j * n_head_kv + kh) * head_dim + d];
                       m = mn;
                     }
                     out[(t * n_head + h) * head_dim + d] = acc / l;
                   })
        .wait();
  }
}

void gelu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float g = gate[id[0]];
     // gelu with the tanh approximation, as Gemma's gelu_pytorch_tanh and ggml_gelu
     const float k0 = 0.7978845608028654f, k1 = 0.044715f;
     out[id[0]] = 0.5f * g * (1.0f + sycl::tanh(k0 * (g + k1 * g * g * g))) * up[id[0]];
   }).wait();
}

void add(sycl::queue& q, const float* a, const float* b, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = a[id[0]] + b[id[0]]; }).wait();
}

void scale(sycl::queue& q, float* x, float s, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { x[id[0]] *= s; }).wait();
}

void softcap(sycl::queue& q, float* x, float cap, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { x[id[0]] = cap * sycl::tanh(x[id[0]] / cap); }).wait();
}

}  // namespace eklow::ref
