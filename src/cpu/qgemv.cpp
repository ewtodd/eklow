#include "cpu/qgemv.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace eklow::cpu {

const char* qtype_name(QType t) {
  switch (t) {
    case QType::Q8_0: return "Q8_0";
    case QType::Q5_K: return "Q5_K";
    case QType::NVFP4: return "NVFP4";
  }
  return "?";
}

int qtype_block(QType t) { return t == QType::Q8_0 ? 32 : t == QType::Q5_K ? 256 : 64; }

int qtype_block_bytes(QType t) { return t == QType::Q8_0 ? 34 : t == QType::Q5_K ? 176 : 36; }

namespace {

inline float f16(const uint8_t* p) {
  uint16_t raw;
  std::memcpy(&raw, p, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(raw)));
}

// ggml's get_scale_min_k4: sub-block j's 6-bit scale and min out of the 12 packed bytes.
inline void scale_min(int j, const uint8_t* q, int& d, int& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

inline __m256i dot_u(__m256i w_u8, __m256i a_s8) { return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), w_u8, a_s8); }
// Signed weights against signed activations: |w| x sign(a, w).
inline __m256i dot_s(__m256i w_s8, __m256i a_s8) {
  return _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(w_s8), _mm256_sign_epi8(a_s8, w_s8));
}
inline __m256i ld(const void* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
inline float hsum(__m256 v) {
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  return _mm_cvtss_f32(h);
}

// The activations of R inputs: int8 values, block scales and block sums.
struct In {
  const int8_t* q;
  const float* s;
  const int32_t* sum;
};

// One weight row against R inputs: out[r] = dot. nb is the number of 32-blocks (Q8_0) or superblocks (Q5_K).
template <QType T, int R>
inline void row_dot(const uint8_t* row, int64_t nb, const In* in, float* out) {
  __m256 acc[R];
  float corr[R];
  for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps(), corr[r] = 0.0f;
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  if constexpr (T == QType::Q8_0) {
    for (int64_t b = 0; b < nb; ++b) {
      const uint8_t* p = row + b * 34;
      const __m256i w = ld(p + 2);
      const float d = f16(p);
      for (int r = 0; r < R; ++r)
        acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_s(w, ld(in[r].q + b * 32))), _mm256_set1_ps(d * in[r].s[b]), acc[r]);
    }
  } else {
    for (int64_t sb = 0; sb < nb; ++sb) {
      const uint8_t* blk = row + sb * 176;
      const float d = f16(blk), dmin = f16(blk + 2);
      int sc[8], mn[8];
      for (int j = 0; j < 8; ++j) scale_min(j, blk + 4, sc[j], mn[j]);
      const uint8_t* qs = blk + 48;
      const __m256i qh = ld(blk + 16);
      for (int i = 0; i < 4; ++i) {
        const __m256i q = ld(qs + 32 * i);
        __m256i lo = _mm256_and_si256(q, m0f), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m0f);
        // The 5th bit: bit j of qh's byte l is the high bit of weight 32 j + l of the superblock.
        const __m256i one = _mm256_set1_epi8(1);
        lo = _mm256_or_si256(lo, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * i), one), 4));
        hi = _mm256_or_si256(hi, _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2 * i + 1), one), 4));
        const int64_t b = sb * 8 + 2 * i;
        const float d0 = d * float(sc[2 * i]), d1 = d * float(sc[2 * i + 1]);
        const float m0 = dmin * float(mn[2 * i]), m1 = dmin * float(mn[2 * i + 1]);
        for (int r = 0; r < R; ++r) {
          const __m256i dl = dot_u(lo, ld(in[r].q + b * 32)), dh = dot_u(hi, ld(in[r].q + (b + 1) * 32));
          const float s0 = in[r].s[b], s1 = in[r].s[b + 1];
          acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dl), _mm256_set1_ps(d0 * s0), acc[r]);
          acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dh), _mm256_set1_ps(d1 * s1), acc[r]);
          corr[r] += m0 * s0 * float(in[r].sum[b]) + m1 * s1 * float(in[r].sum[b + 1]);
        }
      }
    }
  }
  for (int r = 0; r < R; ++r) out[r] = hsum(acc[r]) - corr[r];
}

template <QType T, int R>
void rows_t(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const In* in, float* const* y) {
  const uint64_t rb = qrow_bytes(T, k);
  const int64_t nb = k / qtype_block(T);
  float o[R];
  // Software prefetch of the rows ahead: the L2 streamer stops at each 4 KB page and a task is a few dozen rows, so on pinned
  // 4 KB-paged memory the hardware prefetcher alone leaves the cores waiting at every page start.
  constexpr int64_t kAhead = 3;
  for (int64_t r = lo; r < std::min(hi, lo + kAhead); ++r) {
    const char* p = reinterpret_cast<const char*>(rows + uint64_t(r) * rb);
    for (uint64_t off = 0; off < rb; off += 64) _mm_prefetch(p + off, _MM_HINT_T0);
  }
  for (int64_t r = lo; r < hi; ++r) {
    const char* ahead = reinterpret_cast<const char*>(rows + uint64_t(r + kAhead) * rb);
    for (uint64_t off = 0; off < rb; off += 64) _mm_prefetch(ahead + off, _MM_HINT_T0);
    row_dot<T, R>(rows + uint64_t(r) * rb, nb, in, o);
    for (int j = 0; j < R; ++j) y[j][r] = o[j];
  }
}

template <QType T>
void range_t(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks* const* x, int nx, float* const* y) {
  int j = 0;
  for (; j + 4 <= nx; j += 4) {
    In in[4];
    for (int i = 0; i < 4; ++i) in[i] = {x[j + i]->q.data(), x[j + i]->s.data(), x[j + i]->sum.data()};
    rows_t<T, 4>(rows, k, lo, hi, in, y + j);
  }
  if (nx - j >= 2) {  // 2 or 3 left: one pass over the weights, not one each
    const int r = nx - j;
    In in[3];
    for (int i = 0; i < r; ++i) in[i] = {x[j + i]->q.data(), x[j + i]->s.data(), x[j + i]->sum.data()};
    if (r == 2) rows_t<T, 2>(rows, k, lo, hi, in, y + j);
    else rows_t<T, 3>(rows, k, lo, hi, in, y + j);
    return;
  }
  for (; j < nx; ++j) {
    In in[1] = {{x[j]->q.data(), x[j]->s.data(), x[j]->sum.data()}};
    rows_t<T, 1>(rows, k, lo, hi, in, y + j);
  }
}

}  // namespace

void gemv_q_multi(QType t, const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks* const* x, int nx, float* const* y) {
  if (t == QType::NVFP4) throw std::invalid_argument("gemv_q_multi: NVFP4 takes cpu::NvBlocks (nvfp4.hpp)");
  if (k % qtype_block(t)) throw std::invalid_argument("gemv_q_multi: k is not a multiple of the format's block");
  if (t == QType::Q8_0) range_t<QType::Q8_0>(rows, k, lo, hi, x, nx, y);
  else range_t<QType::Q5_K>(rows, k, lo, hi, x, nx, y);
}

void gemv_q(QType t, const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, Pool& pool) {
  const Q8Blocks* xs[1] = {&x};
  pool.for_each(n, 16, [=](int64_t lo, int64_t hi) {
    float* ys[1] = {y};
    gemv_q_multi(t, rows, k, lo, hi, xs, 1, ys);
  });
}

}  // namespace eklow::cpu
