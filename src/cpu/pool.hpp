// The host CPU's side of the cold experts: a thread pool tuned for fiona's mixed cores, and the int8 activations the
// AVX-VNNI GEMVs take (qgemv.hpp: Q8_0 and Q5_K; nvfp4.hpp: NVFP4). The weights come out of RAM (fiona's two DDR4
// channels).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace eklow::cpu {

// One input row for the Q8_0 and Q5_K GEMVs: int8 values in 32-blocks in natural order, each block's scale and int32
// activation sum (the K-quants' minimum comes off with it).
struct Q8Blocks {
  std::vector<int8_t> q;
  std::vector<float> s;
  std::vector<int32_t> sum;
};

// Quantizes k floats per 32-block, the same rule as xmx::quantize_run: s = max|x| / 127,
// q = round(x / s), s = 0 gives q = 0. quantize_into reuses `r`'s storage.
Q8Blocks quantize(const float* x, int64_t k);
void quantize_into(const float* x, int64_t k, Q8Blocks& r);

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
  void set_spin_us(int us) { spin_us_.store(us, std::memory_order_relaxed); }
  void for_each(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& f);

 private:
  void run(int id);
  std::atomic<int> spin_us_;
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

}  // namespace eklow::cpu
