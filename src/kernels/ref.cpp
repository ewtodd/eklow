#include "kernels/ref.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace eklow::ref {

namespace {

inline float half_at(const uint8_t* p) {
  uint16_t bits = uint16_t(p[0]) | uint16_t(p[1]) << 8;
#ifdef EKLOW_HOST_NO_FP16
  // tools/dflash_test/host-build.sh: SYCL's native_cpu device has no fp16 aspect; the same (exact) conversion by hand.
  const uint32_t sign = uint32_t(bits >> 15) << 31, exp = (bits >> 10) & 31, man = bits & 1023;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      int e = -1;
      uint32_t m = man;
      do {
        ++e;
        m <<= 1;
      } while (!(m & 1024));
      f = sign | uint32_t(127 - 15 - e) << 23 | (m & 1023) << 13;
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | man << 13;
  } else {
    f = sign | (exp + 127 - 15) << 23 | man << 13;
  }
  return sycl::bit_cast<float>(f);
#else
  return float(sycl::bit_cast<sycl::half>(bits));
#endif
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

// K-quant and IQ4_XS blocks (256 weights): decoded as ggml's dequantize_row_q4_K / q5_K / iq4_xs do,
// then dotted.
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// Q4_K (144 bytes): fp16 d, fp16 dmin, 12 bytes of 6-bit scales and mins, 128 bytes of nibbles.
inline void decode_q4_k(const uint8_t* blk, float* y) {
  const float d = half_at(blk), mn = half_at(blk + 2);
  const uint8_t* sc = blk + 4;
  const uint8_t* q = blk + 16;
  int is = 0;
  for (int j = 0; j < 256; j += 64) {
    uint8_t s, m;
    scale_min_k4(is, sc, s, m);
    const float d1 = d * s, m1 = mn * m;
    scale_min_k4(is + 1, sc, s, m);
    const float d2 = d * s, m2 = mn * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
    q += 32;
    is += 2;
  }
}

// Q5_K (176 bytes): as Q4_K, then 32 bytes of high bits before the 128 bytes of low nibbles.
inline void decode_q5_k(const uint8_t* blk, float* y) {
  const float d = half_at(blk), mn = half_at(blk + 2);
  const uint8_t* sc = blk + 4;
  const uint8_t* qh = blk + 16;
  const uint8_t* ql = blk + 48;
  int is = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int j = 0; j < 256; j += 64) {
    uint8_t s, m;
    scale_min_k4(is, sc, s, m);
    const float d1 = d * s, m1 = mn * m;
    scale_min_k4(is + 1, sc, s, m);
    const float d2 = d * s, m2 = mn * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
}

// IQ4_XS (136 bytes): fp16 d, uint16 scales_h, 4 bytes scales_l, 128 bytes of nibbles indexing a
// non-linear table; sub-block ib of 32 has scale d * (ls - 32), ls 6 bits.
inline void decode_iq4_xs(const uint8_t* blk, float* y) {
  constexpr int8_t table[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
  const float d = half_at(blk);
  const uint32_t scales_h = uint32_t(blk[2]) | uint32_t(blk[3]) << 8;
  const uint8_t* scales_l = blk + 4;
  const uint8_t* qs = blk + 8;
  for (int ib = 0; ib < 8; ++ib) {
    const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
    const float dl = d * float(ls - 32);
    for (int j = 0; j < 16; ++j) {
      y[j] = dl * float(table[qs[j] & 0xf]);
      y[j + 16] = dl * float(table[qs[j] >> 4]);
    }
    y += 32;
    qs += 16;
  }
}

inline void decode_block(gguf::Type t, const uint8_t* blk, float* y) {
  if (t == gguf::Type::Q4_K) decode_q4_k(blk, y);
  else if (t == gguf::Type::Q5_K) decode_q5_k(blk, y);
  else decode_iq4_xs(blk, y);
}

inline uint32_t block_bytes(gguf::Type t) { return t == gguf::Type::Q4_K ? 144 : t == gguf::Type::Q5_K ? 176 : 136; }

inline float dot_kblocks(gguf::Type t, const uint8_t* row, const float* x, int64_t k) {
  float acc = 0;
  float y[256];
  for (int64_t b = 0; b < k / 256; ++b) {
    decode_block(t, row + b * block_bytes(t), y);
    float s = 0;
    for (int i = 0; i < 256; ++i) s += y[i] * x[b * 256 + i];
    acc += s;
  }
  return acc;
}

// Q8_0 (32 weights, 34 bytes): fp16 d, then 32 int8; w = q * d.
inline float dot_q8_0(const uint8_t* row, const float* x, int64_t k) {
  float acc = 0;
  for (int64_t b = 0; b < k / 32; ++b) {
    const uint8_t* blk = row + b * 34;
    const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
    float s = 0;
    for (int j = 0; j < 32; ++j) s += float(qs[j]) * x[b * 32 + j];
    acc += half_at(blk) * s;
  }
  return acc;
}

inline float weight_at(gguf::Type t, const uint8_t* row, int64_t i) {
  switch (t) {
    case gguf::Type::F32: return reinterpret_cast<const float*>(row)[i];
    case gguf::Type::F16: return half_at(row + 2 * i);
    case gguf::Type::BF16: return sycl::bit_cast<float>(uint32_t(row[2 * i] | row[2 * i + 1] << 8) << 16);
    case gguf::Type::Q8_0: {
      const uint8_t* blk = row + (i / 32) * 34;
      return half_at(blk) * float(reinterpret_cast<const int8_t*>(blk + 2)[i % 32]);
    }
    case gguf::Type::Q4_0: {
      const uint8_t* blk = row + (i / 32) * 18;
      const int j = int(i % 32);
      const uint8_t v = blk[2 + (j % 16)];
      return half_at(blk) * float(int(j < 16 ? v & 15 : v >> 4) - 8);
    }
    case gguf::Type::Q6_K: return q6k_weight(row + (i / 256) * 210, int(i % 256));
    case gguf::Type::Q4_K:
    case gguf::Type::Q5_K:
    case gguf::Type::IQ4_XS: {
      float y[256];
      decode_block(t, row + (i / 256) * block_bytes(t), y);
      return y[i % 256];
    }
    default: return 0;
  }
}

void check_type(const Weight& w) {
  switch (w.type) {
    case gguf::Type::F32:
    case gguf::Type::F16:
    case gguf::Type::BF16:
    case gguf::Type::Q4_0:
    case gguf::Type::Q8_0:
    case gguf::Type::Q6_K:
    case gguf::Type::Q4_K:
    case gguf::Type::Q5_K:
    case gguf::Type::IQ4_XS: return;
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
   });
}

void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out) {
  q.parallel_for(sycl::range<1>(size_t(rows)), [=](sycl::id<1> id) {
     const int64_t r = int64_t(id[0]);
     const float* xr = x + r * cols;
     float ss = 0;
     for (int64_t i = 0; i < cols; ++i) ss += xr[i] * xr[i];
     const float inv = 1.0f / sycl::sqrt(ss / float(cols) + eps);
     for (int64_t i = 0; i < cols; ++i) out[r * cols + i] = xr[i] * inv * (w ? w[i] : 1.0f);
   });
}

void matmul(sycl::queue& q, const Weight& w, const float* x, int64_t m_rows, float* out) {
  check_type(w);
  // Rows per launch: a fraction of a second for these kernels, about 2e10 multiply-adds for Q4_0;
  // Q6_K and the generic path decode each weight separately and get a tenth of that.
  const double budget = w.type == gguf::Type::Q4_0 || w.type == gguf::Type::Q8_0 ? 2e10 : 2e9;
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
       else if (ww.type == gguf::Type::Q8_0) acc = dot_q8_0(row, xm, ww.k);
       else if (ww.type == gguf::Type::Q6_K) acc = dot_q6_k(row, xm, ww.k);
       else if (ww.type == gguf::Type::Q4_K || ww.type == gguf::Type::Q5_K || ww.type == gguf::Type::IQ4_XS)
         acc = dot_kblocks(ww.type, row, xm, ww.k);
       else {
         acc = 0;
         for (int64_t i = 0; i < ww.k; ++i) acc += weight_at(ww.type, row, i) * xm[i];
       }
       out[m * ww.n + n] = acc;
     });
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
   });
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
                   });
  }
}

void attention_bidir(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t n_tok, int64_t n_head,
                     int64_t n_head_kv, int64_t head_dim, int64_t half_window, float scale_by, float* out) {
  const int64_t group = n_head / n_head_kv;
  const int64_t keys = half_window > 0 ? std::min(2 * half_window + 1, n_tok) : n_tok;
  const int64_t per = std::max<int64_t>(1, int64_t(2e8 / double(n_head * keys)));
  for (int64_t t0 = 0; t0 < n_tok; t0 += per) {
    const int64_t tc = std::min(per, n_tok - t0);
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(tc * n_head), size_t(head_dim)),
                                     sycl::range<2>(1, size_t(head_dim))),
                   [=](sycl::nd_item<2> it) {
                     const int64_t th = t0 * n_head + int64_t(it.get_global_id(0)), d = int64_t(it.get_local_id(1));
                     const int64_t t = th / n_head, h = th % n_head, kh = h / group;
                     const float qd = Q[(t * n_head + h) * head_dim + d];
                     const int64_t first = half_window > 0 ? std::max<int64_t>(0, t - half_window) : 0;
                     const int64_t last = half_window > 0 ? std::min<int64_t>(n_tok - 1, t + half_window) : n_tok - 1;
                     float m = -INFINITY, l = 0, acc = 0;
                     for (int64_t j = first; j <= last; ++j) {
                       const float part = qd * K[(j * n_head_kv + kh) * head_dim + d];
                       const float sc = sycl::reduce_over_group(it.get_group(), part, sycl::plus<float>()) * scale_by;
                       const float mn = sycl::fmax(m, sc);
                       const float corr = sycl::exp(m - mn), p = sycl::exp(sc - mn);
                       l = l * corr + p;
                       acc = acc * corr + p * V[(j * n_head_kv + kh) * head_dim + d];
                       m = mn;
                     }
                     out[(t * n_head + h) * head_dim + d] = acc / l;
                   });
  }
}

void attention_dense(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t n_tok, int64_t n_head,
                     int64_t n_head_kv, int64_t head_dim, float scale_by, float* out) {
  const int64_t group = n_head / n_head_kv;
  // Queries in launches of about 2e9 multiply-adds per matrix product: well under xe's 5 s.
  const int64_t per = std::max<int64_t>(1, int64_t(2e9 / double(n_head * n_tok * head_dim)));
  for (int64_t t0 = 0; t0 < n_tok; t0 += per) {
    const int64_t tc = std::min(per, n_tok - t0);
    q.parallel_for(sycl::range<2>(size_t(tc), size_t(n_head)), [=](sycl::id<2> id) {
       const int64_t t = t0 + int64_t(id[0]), h = int64_t(id[1]), kh = h / group;
       const float* qv = Q + (t * n_head + h) * head_dim;
       float* o = out + (t * n_head + h) * head_dim;
       for (int64_t d = 0; d < head_dim; ++d) o[d] = 0;
       float m = -INFINITY, l = 0;
       for (int64_t j = 0; j < n_tok; ++j) {
         const float* kv = K + (j * n_head_kv + kh) * head_dim;
         const float* vv = V + (j * n_head_kv + kh) * head_dim;
         float s = 0;
         for (int64_t d = 0; d < head_dim; ++d) s += qv[d] * kv[d];
         s *= scale_by;
         const float mn = sycl::fmax(m, s);
         const float corr = sycl::exp(m - mn), p = sycl::exp(s - mn);
         l = l * corr + p;
         for (int64_t d = 0; d < head_dim; ++d) o[d] = o[d] * corr + p * vv[d];
         m = mn;
       }
       for (int64_t d = 0; d < head_dim; ++d) o[d] /= l;
     });
  }
}

void add_rows(sycl::queue& q, float* x, const float* v, int64_t rows, int64_t cols) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(cols)), [=](sycl::id<2> id) {
     x[int64_t(id[0]) * cols + int64_t(id[1])] += v[id[1]];
   });
}

void gelu_quick_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float g = gate[id[0]];
     out[id[0]] = g * (1.0f / (1.0f + sycl::exp(-1.702f * g))) * up[id[0]];
   });
}

void round_bf16(sycl::queue& q, const float* x, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     uint32_t b = sycl::bit_cast<uint32_t>(x[id[0]]);
     if ((b & 0x7fffffffu) <= 0x7f800000u) b += 0x7fffu + ((b >> 16) & 1u);
     out[id[0]] = sycl::bit_cast<float>(b & 0xffff0000u);
   });
}

void clamp(sycl::queue& q, const float* x, float lo, float hi, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = sycl::fmin(sycl::fmax(x[id[0]], lo), hi); });
}

void mul_rows(sycl::queue& q, float* x, const float* v, int64_t rows, int64_t cols) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(cols)), [=](sycl::id<2> id) {
     x[int64_t(id[0]) * cols + int64_t(id[1])] *= v[id[1]];
   });
}

void gelu(sycl::queue& q, const float* x, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float g = x[id[0]];
     const float k0 = 0.7978845608028654f, k1 = 0.044715f;
     out[id[0]] = 0.5f * g * (1.0f + sycl::tanh(k0 * (g + k1 * g * g * g)));
   });
}

void mul(sycl::queue& q, const float* a, const float* b, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = a[id[0]] * b[id[0]]; });
}

void gelu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float g = gate[id[0]];
     // gelu with the tanh approximation, as the Gemma family's gelu_pytorch_tanh and ggml_gelu
     const float k0 = 0.7978845608028654f, k1 = 0.044715f;
     out[id[0]] = 0.5f * g * (1.0f + sycl::tanh(k0 * (g + k1 * g * g * g))) * up[id[0]];
   });
}

void add(sycl::queue& q, const float* a, const float* b, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = a[id[0]] + b[id[0]]; });
}

void scale(sycl::queue& q, float* x, float s, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { x[id[0]] *= s; });
}

void softcap(sycl::queue& q, float* x, float cap, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { x[id[0]] = cap * sycl::tanh(x[id[0]] / cap); });
}

// ---- Qwen3.5/3.6 ----

void silu(sycl::queue& q, const float* x, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float v = x[id[0]];
     out[id[0]] = v / (1.0f + sycl::exp(-v));
   });
}

void swiglu(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float g = gate[id[0]];
     out[id[0]] = g / (1.0f + sycl::exp(-g)) * up[id[0]];
   });
}

void sigmoid(sycl::queue& q, const float* x, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = 1.0f / (1.0f + sycl::exp(-x[id[0]])); });
}

void softplus(sycl::queue& q, const float* x, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     const float v = x[id[0]];
     out[id[0]] = v > 20.0f ? v : sycl::log(1.0f + sycl::exp(v));
   });
}

void mul_by_row_scalar(sycl::queue& q, const float* x, const float* s, int64_t rows, int64_t cols, float* out) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(cols)), [=](sycl::id<2> id) {
     const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
     out[r * cols + c] = x[r * cols + c] * s[r];
   });
}

void causal_conv(sycl::queue& q, const float* x, const float* w, int64_t n_tok, int64_t channels, int64_t kernel,
                 float* out) {
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(channels)), [=](sycl::id<2> id) {
     const int64_t t = int64_t(id[0]), c = int64_t(id[1]);
     float acc = 0;
     for (int64_t j = 0; j < kernel; ++j) {
       const int64_t src = t - (kernel - 1) + j;
       acc += (src >= 0 ? x[src * channels + c] : 0.0f) * w[c * kernel + j];
     }
     out[t * channels + c] = acc;
   });
}

void l2_norm_heads(sycl::queue& q, const float* x, int64_t tok_stride, int64_t n_tok, int64_t n_heads, int64_t cols,
                   float eps, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n_tok * n_heads)), [=](sycl::id<1> id) {
     const int64_t r = int64_t(id[0]);
     const float* xr = x + (r / n_heads) * tok_stride + (r % n_heads) * cols;
     float ss = 0;
     for (int64_t i = 0; i < cols; ++i) ss += xr[i] * xr[i];
     const float scale = 1.0f / sycl::sqrt(ss / float(cols) + eps / float(cols)) * (1.0f / sycl::sqrt(float(cols)));
     for (int64_t i = 0; i < cols; ++i) out[r * cols + i] = xr[i] * scale;
   });
}

void gated_delta_rule(sycl::queue& q, const float* qv, const float* kv, const float* v, int64_t v_stride,
                      const float* g, const float* beta, int64_t n_tok, int64_t n_k_heads, int64_t n_v_heads, int64_t s,
                      float* state, float* out) {
  q.fill(state, 0.0f, size_t(n_v_heads * s * s));
  const float qscale = 1.0f / std::sqrt(float(s));
  // Tokens per launch: each work-item does about 6 * s multiply-adds a token; 256 keeps a launch to a
  // few milliseconds at s = 128 and the 5 s limit out of sight.
  const int64_t per = 256;
  for (int64_t t0 = 0; t0 < n_tok; t0 += per) {
    const int64_t tc = std::min(per, n_tok - t0);
    q.parallel_for(sycl::range<2>(size_t(n_v_heads), size_t(s)), [=](sycl::id<2> id) {
       const int64_t h = int64_t(id[0]), j = int64_t(id[1]), kh = h % n_k_heads;
       float* S = state + (h * s + j) * s;  // column j of the state, key index i fastest
       for (int64_t t = t0; t < t0 + tc; ++t) {
         const float* kt = kv + (t * n_k_heads + kh) * s;
         const float* qt = qv + (t * n_k_heads + kh) * s;
         const float decay = sycl::exp(g[t * n_v_heads + h]);
         float sk = 0;
         for (int64_t i = 0; i < s; ++i) {
           S[i] *= decay;
           sk += S[i] * kt[i];
         }
         const float d = (v[t * v_stride + h * s + j] - sk) * beta[t * n_v_heads + h];
         float o = 0;
         for (int64_t i = 0; i < s; ++i) {
           S[i] += kt[i] * d;
           o += S[i] * (qt[i] * qscale);
         }
         out[(t * n_v_heads + h) * s + j] = o;
       }
     });
  }
}

void gated_rmsnorm(sycl::queue& q, const float* x, const float* z, const float* w, int64_t rows, int64_t cols,
                   float eps, float* out) {
  q.parallel_for(sycl::range<1>(size_t(rows)), [=](sycl::id<1> id) {
     const int64_t r = int64_t(id[0]);
     const float* xr = x + r * cols;
     float ss = 0;
     for (int64_t i = 0; i < cols; ++i) ss += xr[i] * xr[i];
     const float inv = 1.0f / sycl::sqrt(ss / float(cols) + eps);
     for (int64_t i = 0; i < cols; ++i) {
       const float zi = z[r * cols + i];
       out[r * cols + i] = xr[i] * inv * w[i] * (zi / (1.0f + sycl::exp(-zi)));
     }
   });
}

void split_q_gate(sycl::queue& q, const float* full, int64_t n_tok, int64_t n_head, int64_t d, float* qout,
                  float* gate) {
  q.parallel_for(sycl::range<3>(size_t(n_tok), size_t(n_head), size_t(d)), [=](sycl::id<3> id) {
     const int64_t t = int64_t(id[0]), h = int64_t(id[1]), i = int64_t(id[2]);
     const float* src = full + (t * n_head + h) * 2 * d;
     qout[(t * n_head + h) * d + i] = src[i];
     gate[(t * n_head + h) * d + i] = src[d + i];
   });
}

void gather_rows(sycl::queue& q, const float* src, const int32_t* ids, int64_t n, int64_t cols, float* dst) {
  q.parallel_for(sycl::range<2>(size_t(n), size_t(cols)), [=](sycl::id<2> id) {
     const int64_t i = int64_t(id[0]), c = int64_t(id[1]);
     dst[i * cols + c] = src[int64_t(ids[i]) * cols + c];
   });
}

void scatter_add_rows(sycl::queue& q, const float* src, const int32_t* ids, const float* w, int64_t n,
                      int64_t cols, float* dst) {
  q.parallel_for(sycl::range<2>(size_t(n), size_t(cols)), [=](sycl::id<2> id) {
     const int64_t i = int64_t(id[0]), c = int64_t(id[1]);
     dst[int64_t(ids[i]) * cols + c] += w[i] * src[i * cols + c];
   });
}

void split_gate_up(sycl::queue& q, const float* gu, int64_t n, int64_t f, float* gate, float* up) {
  q.parallel_for(sycl::range<2>(size_t(n), size_t(f)), [=](sycl::id<2> id) {
     const int64_t i = int64_t(id[0]), j = int64_t(id[1]);
     gate[i * f + j] = gu[i * 2 * f + j];
     up[i * f + j] = gu[i * 2 * f + f + j];
   });
}

}  // namespace eklow::ref
