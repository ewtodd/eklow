#include "cpu/q4_gemv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <thread>
#include <vector>

namespace eklow::cpu {

Q8Blocks quantize(const float* x, int64_t k) {
  const int64_t nblk = k / 32;
  Q8Blocks r;
  r.q.resize(size_t(k));
  r.s.resize(size_t(nblk));
  r.sum.resize(size_t(nblk));
  for (int64_t b = 0; b < nblk; ++b) {
    const float* p = x + b * 32;
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(p[i]));
    const float s = amax / 127.0f;
    const float inv = s != 0.0f ? 1.0f / s : 0.0f;
    int32_t sum = 0;
    for (int i = 0; i < 32; ++i) {
      const int8_t q = int8_t(std::lrintf(p[i] * inv));
      r.q[size_t(b) * 32 + i] = q;
      sum += q;
    }
    r.s[size_t(b)] = s;
    r.sum[size_t(b)] = sum;
  }
  return r;
}

namespace {

// One 18-byte Q4_0 block: fp16 scale at [0,2), 16 bytes of nibbles at [2,18).
constexpr int kBlockBytes = 18;

inline float f16(const uint8_t* p) {
  uint16_t raw;
  std::memcpy(&raw, p, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(raw)));
}

// The block's 32 nibbles as unsigned bytes in value order: low nibbles (values 0..15) in the low
// 128 bits, high nibbles (16..31) in the high, matching Q8Blocks::q. The shift is masked because
// srli_epi16 moves bits across the byte pair.
inline __m256i load_nibbles(const uint8_t* p) {
  const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
  const __m128i m = _mm_set1_epi8(0x0f);
  return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(raw, 4), m), _mm_and_si128(raw, m));
}

// One row against x. Per block: dpbusd gives 8 partial int32 sums, converted to float and scaled
// by d*s into a vector accumulator (no per-block horizontal sum); the -8 * sum(q) term is the
// same for all 32 values, so it goes into a scalar chain: row = hsum(vec) - corr. Two
// accumulator pairs (even/odd blocks) halve the dependent-add chains.
inline float row_dot(const uint8_t* row, int64_t nblk, const Q8Blocks& x) {
  __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
  float corr0 = 0.0f, corr1 = 0.0f;
  const int8_t* q = x.q.data();
  int64_t b = 0;
  for (; b + 1 < nblk; b += 2) {
    const uint8_t* p0 = row + b * kBlockBytes;
    const uint8_t* p1 = p0 + kBlockBytes;
    const float ds0 = f16(p0) * x.s[size_t(b)], ds1 = f16(p1) * x.s[size_t(b + 1)];
    const __m256i d0 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), load_nibbles(p0 + 2),
                                               _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + b * 32)));
    const __m256i d1 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), load_nibbles(p1 + 2),
                                               _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + b * 32 + 32)));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d0), _mm256_set1_ps(ds0), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d1), _mm256_set1_ps(ds1), acc1);
    corr0 += float(8 * x.sum[size_t(b)]) * ds0;
    corr1 += float(8 * x.sum[size_t(b + 1)]) * ds1;
  }
  for (; b < nblk; ++b) {
    const uint8_t* p = row + b * kBlockBytes;
    const float ds = f16(p) * x.s[size_t(b)];
    const __m256i d = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), load_nibbles(p + 2),
                                              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + b * 32)));
    acc0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(ds), acc0);
    corr0 += float(8 * x.sum[size_t(b)]) * ds;
  }
  const __m256 v = _mm256_add_ps(acc0, acc1);
  __m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  h = _mm_hadd_ps(h, h);
  h = _mm_hadd_ps(h, h);
  return _mm_cvtss_f32(h) - (corr0 + corr1);
}

void rows_range(const uint8_t* rows, int64_t nblk, const Q8Blocks& x, float* y, int64_t lo, int64_t hi) {
  const int64_t row_bytes = nblk * kBlockBytes;
  for (int64_t r = lo; r < hi; ++r) y[r] = row_dot(rows + r * row_bytes, nblk, x);
}

}  // namespace

void gemv_q4(const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, int threads) {
  if (k % 32) throw std::invalid_argument("gemv_q4: k is not a multiple of 32");
  const int64_t nblk = k / 32;
  const int t = int(std::min<int64_t>(n, std::max(threads, 1)));
  if (t <= 1) return rows_range(rows, nblk, x, y, 0, n);
  std::vector<std::thread> th;
  th.reserve(size_t(t - 1));
  for (int i = 1; i < t; ++i)
    th.emplace_back([=, &x] { rows_range(rows, nblk, x, y, i * n / t, (i + 1) * n / t); });
  rows_range(rows, nblk, x, y, 0, n / t);
  for (auto& w : th) w.join();
}

Pool::Pool(int threads) {
  for (int i = 1; i < std::max(threads, 1); ++i) workers_.emplace_back([this, i] { run(i); });
}

Pool::~Pool() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  wake_.notify_all();
  for (auto& w : workers_) w.join();
}

void Pool::for_each(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& f) {
  {
    std::lock_guard lk(mu_);
    job_ = &f;
    n_ = n;
    grain_ = grain;
    next_.store(0, std::memory_order_relaxed);
    pending_ = int(workers_.size());
    ++gen_;
  }
  wake_.notify_all();
  for (int64_t lo; (lo = next_.fetch_add(grain)) < n;) f(lo, std::min(n, lo + grain));
  std::unique_lock lk(mu_);
  done_.wait(lk, [this] { return pending_ == 0; });
}

void Pool::run(int) {
  uint64_t seen = 0;
  for (;;) {
    const std::function<void(int64_t, int64_t)>* job;
    int64_t n, grain;
    {
      std::unique_lock lk(mu_);
      wake_.wait(lk, [&] { return stop_ || gen_ != seen; });
      if (stop_) return;
      seen = gen_;
      job = job_;
      n = n_;
      grain = grain_;
    }
    for (int64_t lo; (lo = next_.fetch_add(grain)) < n;) (*job)(lo, std::min(n, lo + grain));
    std::lock_guard lk(mu_);
    if (--pending_ == 0) done_.notify_one();
  }
}

// Four inputs against one row: the weight block is unpacked once and dotted with each input.
template <int R>
inline void row_dot_multi(const uint8_t* row, int64_t nblk, const Q8Blocks* const* x, float* out) {
  __m256 acc[R];
  float corr[R];
  for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps(), corr[r] = 0.0f;
  for (int64_t b = 0; b < nblk; ++b) {
    const uint8_t* p = row + b * kBlockBytes;
    const __m256i nib = load_nibbles(p + 2);
    const float d = f16(p);
    for (int r = 0; r < R; ++r) {
      const float ds = d * x[r]->s[size_t(b)];
      const __m256i dp = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), nib, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[r]->q.data() + b * 32)));
      acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dp), _mm256_set1_ps(ds), acc[r]);
      corr[r] += float(8 * x[r]->sum[size_t(b)]) * ds;
    }
  }
  for (int r = 0; r < R; ++r) {
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[r]), _mm256_extractf128_ps(acc[r], 1));
    h = _mm_hadd_ps(h, h);
    h = _mm_hadd_ps(h, h);
    out[r] = _mm_cvtss_f32(h) - corr[r];
  }
}

void gemv_q4_multi(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks* const* x, int nx, float* const* y) {
  const int64_t nblk = k / 32, row_bytes = nblk * kBlockBytes;
  for (int64_t r = lo; r < hi; ++r) {
    const uint8_t* row = rows + r * row_bytes;
    int j = 0;
    float o[4];
    for (; j + 4 <= nx; j += 4) {
      row_dot_multi<4>(row, nblk, x + j, o);
      for (int i = 0; i < 4; ++i) y[j + i][r] = o[i];
    }
    for (; j < nx; ++j) y[j][r] = row_dot(row, nblk, *x[j]);
  }
}

void gemv_q4(const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, Pool& pool) {
  if (k % 32) throw std::invalid_argument("gemv_q4: k is not a multiple of 32");
  const int64_t nblk = k / 32;
  pool.for_each(n, 16, [=, &x](int64_t lo, int64_t hi) { rows_range(rows, nblk, x, y, lo, hi); });
}

}  // namespace eklow::cpu
