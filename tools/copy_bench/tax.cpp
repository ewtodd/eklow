// scratch: how DMA shapes affect a concurrent VRAM-streaming kernel
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <atomic>
#include "rt/copy_engine.hpp"
#include "rt/device.hpp"
using namespace eklow;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  sycl::queue q(rt::pick_device(), sycl::property_list{sycl::property::queue::in_order()});
  const size_t big_n = size_t(128) << 20;
  auto* big = sycl::malloc_device<float>(big_n, q);
  q.memset(big, 0, big_n * 4).wait();
  auto* sink = sycl::malloc_device<float>(1 << 16, q);
  auto bw = [&] {
    q.parallel_for(sycl::range<1>(1 << 16), [=](sycl::id<1> id) {
      float s = 0;
      for (size_t i = id[0]; i < big_n; i += (1 << 16)) s += big[i];
      sink[id[0]] = s;
    });
  };
  const size_t span = size_t(256) << 20;
  auto* pin = sycl::malloc_host<uint8_t>(span, q);
  std::memset(pin, 1, span);
  auto* dst = sycl::malloc_device<uint8_t>(span, q);
  rt::CopyEngine ce(q);
  auto measure = [&] {
    double t = now();
    int r = 0;
    while (now() - t < 0.3) { bw(); q.wait(); ++r; }
    return (now() - t) / r;
  };
  measure();
  double t0;
  double alone = measure();
  std::printf("kernel alone %.2f ms\n", alone * 1e3);
  struct V { const char* name; size_t chunk; bool h2d; };
  for (V v : {V{"h2d 3.3MB", 3345408, true}, V{"h2d 64KB", 65536, true}, V{"h2d 256KB", 262144, true}, V{"h2d 32MB", size_t(32) << 20, true}, V{"d2h 3.3MB", 3345408, false}, V{"d2d 3.3MB", 3345408, true}}) {
    const bool d2d = v.name[0] == 'd' && v.name[2] == 'd';
    const uint8_t* src_h = pin;
    std::atomic<bool> stop{false};
    double bytes = 0;
    std::thread th([&] {
      const double s0 = now();
      while (!stop) {
        for (int i = 0; i < 8; ++i) {
          const size_t off = (size_t(i) * v.chunk) % (span - v.chunk);
          if (d2d) ce.copy(dst + off, big + (off / 4) % (big_n - v.chunk / 4), v.chunk);
          else if (v.h2d) ce.copy(dst + off, src_h + off, v.chunk);
          else ce.copy(pin + off, dst + off, v.chunk);
          bytes += double(v.chunk);
        }
        ce.wait(ce.fence());
      }
      bytes /= (now() - s0);  // (whole thread life)
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    const double with = measure();
    stop = true;
    th.join();
    alone = measure();
    std::printf("%-12s dma %5.1f GB/s: kernel %.2f ms (x%.2f)\n", v.name, bytes / 1e9, with * 1e3, with / alone);
  }
}
