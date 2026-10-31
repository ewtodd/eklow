// Host references for eklow-qfmt-test and eklow-cpu-qgemv-test: ggml's dequantization of each format, written from
// ggml-quants.c (dequantize_row_q8_0 / q4_K / q5_K / q6_K / iq4_xs, not from eklow's repack code), random valid
// blocks, and the integer arithmetic of the int8-activation GEMVs as an exact emulation (the card's decode kernels
// and the CPU's both compute it).
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "model/gguf.hpp"

namespace eklow::refs {

constexpr int8_t kIq4[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

inline float h2f(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(uint16_t(p[0] | p[1] << 8))); }
inline uint16_t f2h_bits(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }

// Weights per ggml block and bytes of it.
inline int block_k(gguf::Type t) { return t == gguf::Type::Q8_0 || t == gguf::Type::Q4_0 ? 32 : 256; }
inline int block_bytes(gguf::Type t) {
  switch (t) {
    case gguf::Type::Q4_0: return 18;
    case gguf::Type::Q8_0: return 34;
    case gguf::Type::Q4_K: return 144;
    case gguf::Type::Q5_K: return 176;
    case gguf::Type::Q6_K: return 210;
    case gguf::Type::IQ4_XS: return 136;
    default: return 0;
  }
}
inline uint64_t row_bytes(gguf::Type t, int64_t k) { return uint64_t(k / block_k(t)) * uint64_t(block_bytes(t)); }

// Random valid blocks: random bytes, with fp16 scales (d, dmin) set to small positive normal values (small enough that
// a GLU of random data stays inside fp16).
inline std::vector<uint8_t> random_rows(gguf::Type t, int64_t n, int64_t k, std::mt19937& rng) {
  const uint64_t rb = row_bytes(t, k);
  std::vector<uint8_t> out(size_t(rb * uint64_t(n)));
  for (auto& b : out) b = uint8_t(rng());
  std::uniform_real_distribution<float> du(1e-4f, 4e-4f);
  const int bb = block_bytes(t);
  for (size_t at = 0; at < out.size(); at += size_t(bb)) {
    uint8_t* blk = out.data() + at;
    auto put = [&](int off) {
      const uint16_t h = f2h_bits(du(rng));
      blk[off] = uint8_t(h);
      blk[off + 1] = uint8_t(h >> 8);
    };
    switch (t) {
      case gguf::Type::Q4_0:
      case gguf::Type::Q8_0:
      case gguf::Type::IQ4_XS: put(0); break;
      case gguf::Type::Q4_K:
      case gguf::Type::Q5_K: put(0); put(2); break;
      case gguf::Type::Q6_K: put(208); break;
      default: break;
    }
  }
  return out;
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

// ggml's dequantization of one row of k weights, fp32.
inline void dequant_row(gguf::Type t, const uint8_t* row, int64_t k, float* y) {
  const int bk = block_k(t), bb = block_bytes(t);
  for (int64_t i = 0; i < k / bk; ++i) {
    const uint8_t* x = row + i * bb;
    switch (t) {
      case gguf::Type::Q4_0: {
        const float d = h2f(x);
        for (int j = 0; j < 16; ++j) {
          y[j] = float((x[2 + j] & 0xF) - 8) * d;
          y[j + 16] = float((x[2 + j] >> 4) - 8) * d;
        }
        break;
      }
      case gguf::Type::Q8_0: {
        const float d = h2f(x);
        for (int j = 0; j < 32; ++j) y[j] = d * float(int8_t(x[2 + j]));
        break;
      }
      case gguf::Type::Q4_K: {
        const float d = h2f(x), min = h2f(x + 2);
        const uint8_t* q = x + 16;
        int is = 0;
        float* yy = y;
        for (int j = 0; j < 256; j += 64) {
          uint8_t sc, m;
          get_scale_min_k4(is + 0, x + 4, &sc, &m);
          const float d1 = d * sc, m1 = min * m;
          get_scale_min_k4(is + 1, x + 4, &sc, &m);
          const float d2 = d * sc, m2 = min * m;
          for (int l = 0; l < 32; ++l) *yy++ = d1 * float(q[l] & 0xF) - m1;
          for (int l = 0; l < 32; ++l) *yy++ = d2 * float(q[l] >> 4) - m2;
          q += 32;
          is += 2;
        }
        break;
      }
      case gguf::Type::Q5_K: {
        const float d = h2f(x), min = h2f(x + 2);
        const uint8_t* qh = x + 16;
        const uint8_t* ql = x + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        float* yy = y;
        for (int j = 0; j < 256; j += 64) {
          uint8_t sc, m;
          get_scale_min_k4(is + 0, x + 4, &sc, &m);
          const float d1 = d * sc, m1 = min * m;
          get_scale_min_k4(is + 1, x + 4, &sc, &m);
          const float d2 = d * sc, m2 = min * m;
          for (int l = 0; l < 32; ++l) *yy++ = d1 * float((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
          for (int l = 0; l < 32; ++l) *yy++ = d2 * float((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
          ql += 32;
          is += 2;
          u1 <<= 2;
          u2 <<= 2;
        }
        break;
      }
      case gguf::Type::Q6_K: {
        const float d = h2f(x + 208);
        const uint8_t* ql = x;
        const uint8_t* qh = x + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(x + 192);
        float* yy = y;
        for (int n = 0; n < 256; n += 128) {
          for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = int8_t((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int q2 = int8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = int8_t((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = int8_t((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            yy[l + 0] = d * sc[is + 0] * float(q1);
            yy[l + 32] = d * sc[is + 2] * float(q2);
            yy[l + 64] = d * sc[is + 4] * float(q3);
            yy[l + 96] = d * sc[is + 6] * float(q4);
          }
          yy += 128;
          ql += 64;
          qh += 32;
          sc += 8;
        }
        break;
      }
      case gguf::Type::IQ4_XS: {
        const float d = h2f(x);
        const uint16_t sh = uint16_t(x[2] | x[3] << 8);
        const uint8_t* sl = x + 4;
        const uint8_t* qs = x + 8;
        float* yy = y;
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((sl[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((sh >> 2 * ib) & 3) << 4);
          const float dl = d * float(ls - 32);
          for (int j = 0; j < 16; ++j) {
            yy[j + 0] = dl * float(kIq4[qs[j] & 0xf]);
            yy[j + 16] = dl * float(kIq4[qs[j] >> 4]);
          }
          yy += 32;
          qs += 16;
        }
        break;
      }
      default: break;
    }
    y += bk;
  }
}

// One row of int8 activations by the cards' rule (xmx::quantize_run): per 32-block s = max|x| / 127 and q = round(x / s).
struct QRow {
  std::vector<int> q;
  std::vector<float> s;
};
inline QRow quantize_row(const float* x, int64_t k) {
  QRow r;
  r.q.resize(size_t(k));
  r.s.resize(size_t(k / 32));
  for (int64_t b = 0; b < k / 32; ++b) {
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[b * 32 + j]));
    const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
    for (int j = 0; j < 32; ++j) r.q[size_t(b * 32 + j)] = int(std::round(x[b * 32 + j] * id));
    r.s[size_t(b)] = d;
  }
  return r;
}

// The 256 weights of a K-quant / IQ4_XS superblock as the integers the matrix units multiply (q for Q4_K / Q5_K,
// q - 32 for Q6_K, the table's int8 for IQ4_XS), and per 32-block j the scale dq[2 j] (d sc; Q6_K: of the first 16;
// IQ4_XS: d (ls - 32)) and dq[2 j + 1] (the min dmin m; Q6_K: the scale of the second 16; IQ4_XS: unused), in float.
inline void block_ints(gguf::Type t, const uint8_t* x, int* w, float* dq) {
  if (t == gguf::Type::Q4_K || t == gguf::Type::Q5_K) {
    const float d = h2f(x), dmin = h2f(x + 2);
    const uint8_t* qs = x + (t == gguf::Type::Q5_K ? 48 : 16);
    for (int e = 0; e < 256; ++e) {
      const uint8_t byte = qs[32 * (e / 64) + e % 32];
      int q = (e % 64) >= 32 ? byte >> 4 : byte & 15;
      if (t == gguf::Type::Q5_K) q += ((x[16 + e % 32] >> (e / 32)) & 1) << 4;
      w[e] = q;
    }
    for (int j = 0; j < 8; ++j) {
      uint8_t sc, m;
      get_scale_min_k4(j, x + 4, &sc, &m);
      dq[2 * j] = d * float(sc);
      dq[2 * j + 1] = dmin * float(m);
    }
  } else if (t == gguf::Type::Q6_K) {
    const float d = h2f(x + 208);
    const int8_t* sc = reinterpret_cast<const int8_t*>(x + 192);
    for (int e = 0; e < 256; ++e) {
      const int h = e / 128, r = e % 128, l = r % 32, s = r / 32;
      const uint8_t qlb = x[64 * h + l + 32 * (s & 1)];
      w[e] = ((s >= 2 ? qlb >> 4 : qlb & 15) | ((x[128 + 32 * h + l] >> (2 * s)) & 3) << 4) - 32;
    }
    for (int j = 0; j < 8; ++j) {
      dq[2 * j] = d * float(sc[2 * j]);
      dq[2 * j + 1] = d * float(sc[2 * j + 1]);
    }
  } else {
    const float d = h2f(x);
    const uint16_t sh = uint16_t(x[2] | x[3] << 8);
    for (int e = 0; e < 256; ++e) {
      const uint8_t byte = x[8 + 16 * (e / 32) + e % 16];
      w[e] = kIq4[(e % 32) >= 16 ? byte >> 4 : byte & 15];
    }
    for (int j = 0; j < 8; ++j) {
      const int ls = ((x[4 + j / 2] >> 4 * (j % 2)) & 0xf) | (((sh >> 2 * j) & 3) << 4);
      dq[2 * j] = d * float(ls - 32);
      dq[2 * j + 1] = 0;
    }
  }
}

// The decode GEMV's arithmetic for one output, from the format's own bytes: per 32-block (a tile) the integer dot of
// the activations with the quantized weights, scaled in floats as the kernels do, summed in double.
inline double emulate(gguf::Type t, const uint8_t* row, int64_t k, const QRow& a) {
  const int bb = block_bytes(t);
  double total = 0;
  if (t == gguf::Type::Q4_0 || t == gguf::Type::Q8_0) {
    for (int64_t b = 0; b < k / 32; ++b) {
      const uint8_t* x = row + b * bb;
      long dot = 0;
      for (int j = 0; j < 32; ++j) {
        const int qw = t == gguf::Type::Q8_0 ? int8_t(x[2 + j]) : (j < 16 ? x[2 + j] & 15 : x[2 + j - 16] >> 4) - 8;
        dot += long(a.q[size_t(b * 32 + j)]) * qw;
      }
      total += double(dot) * double(float(a.s[size_t(b)]) * h2f(x));
    }
    return total;
  }
  for (int64_t sb = 0; sb < k / 256; ++sb) {
    int w[256];
    float dq[16];
    block_ints(t, row + sb * bb, w, dq);
    for (int j = 0; j < 8; ++j) {
      const int64_t b = sb * 8 + j;
      const float sa = a.s[size_t(b)];
      long dot = 0, asum = 0, lo = 0, hi = 0;
      for (int kk = 0; kk < 32; ++kk) {
        const long qa = a.q[size_t(b * 32 + kk)], qw = w[j * 32 + kk];
        dot += qa * qw;
        asum += qa;
        (kk < 16 ? lo : hi) += qa * qw;
      }
      if (t == gguf::Type::Q6_K) total += double(sa) * (double(dq[2 * j]) * double(lo) + double(dq[2 * j + 1]) * double(hi));
      else if (t == gguf::Type::IQ4_XS) total += double(sa) * double(dq[2 * j]) * double(dot);
      else total += double(sa) * (double(dq[2 * j]) * double(dot) - double(dq[2 * j + 1]) * double(asum));
    }
  }
  return total;
}

// The fp16 GEMM's weights: one row as the fp16 values (as floats) its dequantization produces, with the roundings
// it does (xmx_fmt.hpp, GemmDec): scales rounded to fp16, products and differences rounded to fp16 in the order of
// the kernel.
inline float to_h(float v) { return float(sycl::half(v)); }
inline void weights16_row(gguf::Type t, const uint8_t* row, int64_t k, float* out) {
  const int bb = block_bytes(t);
  if (t == gguf::Type::Q4_0 || t == gguf::Type::Q8_0) {
    for (int64_t b = 0; b < k / 32; ++b) {
      const uint8_t* x = row + b * bb;
      const float d = h2f(x);
      for (int j = 0; j < 32; ++j) {
        const int qw = t == gguf::Type::Q8_0 ? int8_t(x[2 + j]) : (j < 16 ? x[2 + j] & 15 : x[2 + j - 16] >> 4) - 8;
        out[b * 32 + j] = to_h(float(qw) * d);
      }
    }
    return;
  }
  for (int64_t sb = 0; sb < k / 256; ++sb) {
    int w[256];
    float dq[16];
    block_ints(t, row + sb * bb, w, dq);
    for (int j = 0; j < 8; ++j) {
      const float ds = to_h(dq[2 * j]), dm = to_h(dq[2 * j + 1]), ds2 = to_h(dq[2 * j + 1]);
      for (int kk = 0; kk < 32; ++kk) {
        const float q = float(w[j * 32 + kk]);
        float v;
        if (t == gguf::Type::Q4_K || t == gguf::Type::Q5_K) v = to_h(to_h(q * ds) - dm);
        else if (t == gguf::Type::Q6_K) v = to_h(q * (kk < 16 ? ds : ds2));
        else v = to_h(q * ds);
        out[sb * 256 + j * 32 + kk] = v;
      }
    }
  }
}

}  // namespace eklow::refs
