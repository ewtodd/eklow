// Q4_0 GEMVs on the host CPU, for the 26B's cold experts: the weights come out of RAM (fiona's
// two DDR5 channels) and the one activation row is quantized to int8 per 32-block, the same
// arithmetic the card's int8 decode path does (src/kernels/xmx.cpp). AVX-VNNI's
// _mm256_dpbusd_avx_epi32 (unsigned bytes x signed bytes) takes the nibbles as unsigned 0..15,
// and the (nibble - 8) is removed with the block's activation sum.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace eklow::cpu {

// One input row for gemv_q4: int8 values in 32-blocks, each block's scale and int32 activation
// sum (it removes the nibbles' -8 once per block, not 32 times). Values are in natural order: a
// Q4_0 block's low nibbles are values 0..15 and its high nibbles 16..31, so the unpacked
// nibble vector lines up with q with no shuffle.
struct Q8Blocks {
  std::vector<int8_t> q;
  std::vector<float> s;
  std::vector<int32_t> sum;
};

// Quantizes k floats per 32-block, the same rule as xmx::quantize_run: s = max|x| / 127,
// q = round(x / s), s = 0 gives q = 0.
Q8Blocks quantize(const float* x, int64_t k);

// y[r] = dot(x, rows[r]) for r < n. rows are k/32 Q4_0 blocks of 18 bytes (an fp16 scale d, then
// 16 bytes of nibbles: byte i's low nibble is value i, its high value i + 16; value =
// (nibble - 8) * d); each block's integer dot is scaled by d * s and summed in fp32.
//
// This overload splits the rows over `threads` std::threads created for the call and joined
// before it returns (tens of microseconds per call: fine for a test, not for the engine, which
// should create a Pool once and pass it to every call).
void gemv_q4(const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, int threads);

// A pool of idle workers. for_each runs f(lo, hi) over contiguous chunks of [0, n),
// claimed dynamically in grain-row steps (the i7-12700H mixes P and E cores, so a static even
// split leaves the fast cores waiting), by the workers and the caller; returns when all are done.
// One call at a time. A worker that has just finished a job spins for about spin_us before it sleeps
// on the condition variable, and the caller spins for the workers the same way: a futex wake costs
// 50-100 us, as much as a decode-time expert's whole GEMV, and decode calls come every few hundred
// microseconds. A job of one grain runs on the caller alone.
class Pool {
 public:
  explicit Pool(int threads, int spin_us = 400);  // total threads including the caller
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  int size() const { return int(workers_.size()) + 1; }
  void for_each(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& f);

 private:
  void run(int id);
  int spin_us_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable wake_, done_;
  const std::function<void(int64_t, int64_t)>* job_ = nullptr;
  int64_t n_ = 0, grain_ = 1;
  std::atomic<int64_t> next_{0};
  std::atomic<uint64_t> gen_{0};  // bumped per job so each worker runs each job exactly once
  std::atomic<int> pending_{0};
  std::atomic<bool> stop_{false};
};

void gemv_q4(const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, Pool& pool);

// Rows [lo, hi) of the matrix against nx inputs at once: y[j][r] = dot(x[j], rows[r]). Each weight
// row is loaded and unpacked once for all the inputs (in groups of four), which is what a prefill
// chunk's tokens sharing one cold expert need: its weights come out of RAM once, not once a token.
// `rows` points at row 0 of the matrix.
void gemv_q4_multi(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks* const* x, int nx, float* const* y);

}  // namespace eklow::cpu
