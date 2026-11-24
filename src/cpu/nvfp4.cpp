#include "cpu/nvfp4.hpp"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace eklow::cpu {

void nv_quantize_into(const float* x, int64_t k, NvBlocks& r) {
  if (k % kNvBlock) throw std::invalid_argument("nv_quantize: k is not a multiple of 64");
  const int64_t nb = k / kNvBlock;
  r.k = k;
  r.q.resize(size_t(k));
  r.s.resize(size_t(nb) * 8);
  r.init.resize(size_t(nb) * 8);
  for (int64_t b = 0; b < nb; ++b) {
    int8_t* q = r.q.data() + b * 64;
    for (int s = 0; s < 4; ++s) {
      const float* p = x + b * 64 + s * 16;
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) amax = std::max(amax, std::fabs(p[i]));
      const float sc = amax / 127.0f;
      const float inv = sc != 0.0f ? 1.0f / sc : 0.0f;
      for (int j = 0; j < 8; ++j) {
        q[s * 8 + j] = int8_t(std::lrintf(p[j] * inv));
        q[32 + s * 8 + j] = int8_t(std::lrintf(p[8 + j] * inv));
      }
      r.s[size_t(b) * 8 + 2 * s] = r.s[size_t(b) * 8 + 2 * s + 1] = sc;
    }
    for (int l = 0; l < 8; ++l) {
      int32_t sum = 0;
      for (int i = 0; i < 4; ++i) sum += q[4 * l + i] + q[32 + 4 * l + i];
      r.init[size_t(b) * 8 + l] = -12 * sum;
    }
  }
}

NvBlocks nv_quantize(const float* x, int64_t k) {
  NvBlocks r;
  nv_quantize_into(x, k, r);
  return r;
}

namespace {

// UE4M3 scale bytes as ggml decodes them, halved (the codes are doubled); 0 and 0x7f (NaN) are 0.
struct ScaleLut {
  float v[256];
  ScaleLut() {
    for (int x = 0; x < 256; ++x) {
      if (x == 0 || x == 0x7f) v[x] = 0.0f;
      else {
        const int e = (x >> 3) & 15, m = x & 7;
        v[x] = 0.5f * (e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, e - 7));
      }
    }
  }
};
const ScaleLut kLut;

struct In {
  const int8_t* q;
  const float* s;
  const int32_t* init;
};

inline float hsum(__m256 v) {
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  return _mm_cvtss_f32(h);
}

template <int R>
inline void row_dot(const uint8_t* row, int64_t nb, const In* in, float* out) {
  // code -> doubled E2M1 value + 12
  const __m128i t128 = _mm_setr_epi8(12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0);
  const __m256i tab = _mm256_set_m128i(t128, t128);
  const __m256i m0f = _mm256_set1_epi8(0x0F);
  const __m256i dup = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
  __m256 acc[R];
  for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* p = row + b * kNvBlockBytes;
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p + 4));
    const __m256i wl = _mm256_shuffle_epi8(tab, _mm256_and_si256(raw, m0f));
    const __m256i wh = _mm256_shuffle_epi8(tab, _mm256_and_si256(_mm256_srli_epi16(raw, 4), m0f));
    uint32_t sc;
    std::memcpy(&sc, p, 4);
    const __m128 ws = _mm_set_ps(kLut.v[sc >> 24], kLut.v[(sc >> 16) & 255], kLut.v[(sc >> 8) & 255], kLut.v[sc & 255]);
    const __m256 ws8 = _mm256_permutevar8x32_ps(_mm256_castps128_ps256(ws), dup);
    for (int r = 0; r < R; ++r) {
      const int8_t* a = in[r].q + b * 64;
      __m256i dp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in[r].init + b * 8));
      dp = _mm256_dpbusd_avx_epi32(dp, wl, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)));
      dp = _mm256_dpbusd_avx_epi32(dp, wh, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + 32)));
      acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_mul_ps(ws8, _mm256_loadu_ps(in[r].s + b * 8)), acc[r]);
    }
  }
  for (int r = 0; r < R; ++r) out[r] = hsum(acc[r]);
}

template <int R>
void rows_t(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const In* in, float* const* y) {
  const uint64_t rb = nv_row_bytes(k);
  const int64_t nb = k / kNvBlock;
  float o[R];
  // Software prefetch of the rows ahead (see qgemv.cpp).
  constexpr int64_t kAhead = 3;
  for (int64_t r = lo; r < std::min(hi, lo + kAhead); ++r) {
    const char* p = reinterpret_cast<const char*>(rows + uint64_t(r) * rb);
    for (uint64_t off = 0; off < rb; off += 64) _mm_prefetch(p + off, _MM_HINT_T0);
  }
  for (int64_t r = lo; r < hi; ++r) {
    const char* ahead = reinterpret_cast<const char*>(rows + uint64_t(r + kAhead) * rb);
    for (uint64_t off = 0; off < rb; off += 64) _mm_prefetch(ahead + off, _MM_HINT_T0);
    row_dot<R>(rows + uint64_t(r) * rb, nb, in, o);
    for (int j = 0; j < R; ++j) y[j][r] = o[j];
  }
}

}  // namespace

void gemv_nvfp4_multi(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const NvBlocks* const* x, int nx, float* const* y) {
  if (k % kNvBlock) throw std::invalid_argument("gemv_nvfp4: k is not a multiple of 64");
  int j = 0;
  for (; j + 4 <= nx; j += 4) {
    In in[4];
    for (int i = 0; i < 4; ++i) in[i] = {x[j + i]->q.data(), x[j + i]->s.data(), x[j + i]->init.data()};
    rows_t<4>(rows, k, lo, hi, in, y + j);
  }
  if (nx - j >= 2) {
    const int r = nx - j;
    In in[3];
    for (int i = 0; i < r; ++i) in[i] = {x[j + i]->q.data(), x[j + i]->s.data(), x[j + i]->init.data()};
    if (r == 2) rows_t<2>(rows, k, lo, hi, in, y + j);
    else rows_t<3>(rows, k, lo, hi, in, y + j);
    return;
  }
  for (; j < nx; ++j) {
    In in[1] = {{x[j]->q.data(), x[j]->s.data(), x[j]->init.data()}};
    rows_t<1>(rows, k, lo, hi, in, y + j);
  }
}

void gemv_nvfp4(const uint8_t* rows, int64_t k, int64_t n, const NvBlocks& x, float* y, Pool& pool) {
  const NvBlocks* xs[1] = {&x};
  pool.for_each(n, 16, [=](int64_t lo, int64_t hi) {
    float* ys[1] = {y};
    gemv_nvfp4_multi(rows, k, lo, hi, xs, 1, ys);
  });
}

}  // namespace eklow::cpu
