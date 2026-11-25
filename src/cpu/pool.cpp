#include "cpu/pool.hpp"

#include <pthread.h>
#include <sched.h>

#include <cstdlib>
#include <vector>

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace eklow::cpu {

void quantize_into(const float* x, int64_t k, Q8Blocks& r) {
  const int64_t nblk = k / 32;
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
}

Q8Blocks quantize(const float* x, int64_t k) {
  Q8Blocks r;
  quantize_into(x, k, r);
  return r;
}

namespace {
using SpinClock = std::chrono::steady_clock;
}

Pool::Pool(int threads, int spin_us) : spin_us_(spin_us) {
  // EKLOW_CPU_PIN: a comma-separated list of CPUs; worker i (1 on: the caller is 0 and stays where it is) runs on entry
  // i - 1, cycling (speed work: P-cores against E-cores).
  std::vector<int> pin;
  if (const char* e = std::getenv("EKLOW_CPU_PIN"))
    for (const char* c = e; *c;) {
      char* end = nullptr;
      const long v = std::strtol(c, &end, 10);
      if (end == c) break;
      pin.push_back(int(v));
      c = *end == ',' ? end + 1 : end;
    }
  for (int i = 1; i < std::max(threads, 1); ++i)
    workers_.emplace_back([this, i, pin] {
      if (!pin.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pin[size_t(i - 1) % pin.size()], &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
      }
      run(i);
    });
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
  if (n <= grain || workers_.empty()) {
    if (n > 0) f(0, n);
    return;
  }
  {
    std::lock_guard lk(mu_);
    job_ = &f;
    n_ = n;
    grain_ = grain;
    next_.store(0, std::memory_order_relaxed);
    pending_.store(int(workers_.size()), std::memory_order_relaxed);
    gen_.fetch_add(1, std::memory_order_release);
  }
  wake_.notify_all();
  for (int64_t lo; (lo = next_.fetch_add(grain)) < n;) f(lo, std::min(n, lo + grain));
  const auto t0 = SpinClock::now();
  while (pending_.load(std::memory_order_acquire) != 0 &&
         std::chrono::duration_cast<std::chrono::microseconds>(SpinClock::now() - t0).count() < spin_us_.load(std::memory_order_relaxed))
    _mm_pause();
  if (pending_.load(std::memory_order_acquire) != 0) {
    std::unique_lock lk(mu_);
    done_.wait(lk, [this] { return pending_.load(std::memory_order_acquire) == 0; });
  }
}

void Pool::run(int) {
  uint64_t seen = 0;
  for (;;) {
    const auto t0 = SpinClock::now();
    while (gen_.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_relaxed) &&
           std::chrono::duration_cast<std::chrono::microseconds>(SpinClock::now() - t0).count() < spin_us_.load(std::memory_order_relaxed))
      _mm_pause();
    const std::function<void(int64_t, int64_t)>* job;
    int64_t n, grain;
    {
      std::unique_lock lk(mu_);
      wake_.wait(lk, [&] { return stop_.load() || gen_.load() != seen; });
      if (stop_) return;
      seen = gen_.load();
      job = job_;
      n = n_;
      grain = grain_;
    }
    for (int64_t lo; (lo = next_.fetch_add(grain)) < n;) (*job)(lo, std::min(n, lo + grain));
    if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard lk(mu_);
      done_.notify_one();
    }
  }
}

}  // namespace eklow::cpu
