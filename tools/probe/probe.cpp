// eklow-probe: what the A770M offers this engine, measured rather than read from a spec sheet.
//
//   eklow-probe [--json FILE] [--skip-alloc]
//
// Records the device as Level Zero reports it, how much device memory can really be allocated and
// touched, host<->device copy rates (pinned and pageable, verified), kernel launch and sync
// latency, and how quickly a running kernel sees a flag the host writes into host memory (the
// mechanism the 26B's cold experts would use). Every kernel is bounded well under xe's 5 s job
// timeout. Run with ONEAPI_DEVICE_SELECTOR=level_zero:N to pick a device; by default the GPU with
// the most memory is used.
#include <level_zero/ze_api.h>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "rt/device.hpp"

using namespace eklow;
using Clock = std::chrono::steady_clock;

namespace {

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Stats {
  double min, median, p99, mean;
};

Stats stats(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  return {v.front(), v[v.size() / 2], v[std::min(v.size() - 1, size_t(v.size() * 0.99))], mean};
}

void write_stats(json::Writer& w, const char* key, const Stats& s, double scale) {
  w.key(key).begin_object().kv("min", s.min * scale).kv("median", s.median * scale).kv("p99", s.p99 * scale)
      .kv("mean", s.mean * scale).end_object();
}

const char* backend_name(sycl::backend b) {
  switch (b) {
    case sycl::backend::ext_oneapi_level_zero: return "level_zero";
    case sycl::backend::opencl: return "opencl";
    default: return "other";
  }
}

void device_info(json::Writer& w, const sycl::device& d) {
  namespace di = sycl::info::device;
  namespace xi = sycl::ext::intel::info::device;
  w.key("device").begin_object();
  w.kv("name", d.get_info<di::name>()).kv("vendor", d.get_info<di::vendor>())
      .kv("driver_version", d.get_info<di::driver_version>()).kv("backend", backend_name(d.get_backend()))
      .kv("compute_units", uint64_t(d.get_info<di::max_compute_units>()))
      .kv("max_work_group_size", uint64_t(d.get_info<di::max_work_group_size>()))
      .kv("global_mem_size", uint64_t(d.get_info<di::global_mem_size>()))
      .kv("max_mem_alloc_size", uint64_t(d.get_info<di::max_mem_alloc_size>()))
      .kv("local_mem_size", uint64_t(d.get_info<di::local_mem_size>()))
      .kv("max_clock_mhz", uint64_t(d.get_info<di::max_clock_frequency>()));
  w.key("sub_group_sizes").begin_array();
  for (auto s : d.get_info<di::sub_group_sizes>()) w.value(uint64_t(s));
  w.end_array();
  auto opt = [&](sycl::aspect a, const char* key, auto get) {
    if (d.has(a)) w.kv(key, uint64_t(get()));
    else w.key(key).null();
  };
  opt(sycl::aspect::ext_intel_gpu_eu_count, "eu_count", [&] { return d.get_info<xi::gpu_eu_count>(); });
  opt(sycl::aspect::ext_intel_gpu_eu_simd_width, "eu_simd_width", [&] { return d.get_info<xi::gpu_eu_simd_width>(); });
  opt(sycl::aspect::ext_intel_gpu_slices, "slices", [&] { return d.get_info<xi::gpu_slices>(); });
  opt(sycl::aspect::ext_intel_gpu_subslices_per_slice, "subslices_per_slice", [&] { return d.get_info<xi::gpu_subslices_per_slice>(); });
  opt(sycl::aspect::ext_intel_gpu_eu_count_per_subslice, "eus_per_subslice", [&] { return d.get_info<xi::gpu_eu_count_per_subslice>(); });
  opt(sycl::aspect::ext_intel_gpu_hw_threads_per_eu, "threads_per_eu", [&] { return d.get_info<xi::gpu_hw_threads_per_eu>(); });
  opt(sycl::aspect::ext_intel_memory_clock_rate, "memory_clock_mhz", [&] { return d.get_info<xi::memory_clock_rate>(); });
  opt(sycl::aspect::ext_intel_memory_bus_width, "memory_bus_width", [&] { return d.get_info<xi::memory_bus_width>(); });
  opt(sycl::aspect::ext_intel_max_mem_bandwidth, "max_mem_bandwidth", [&] { return d.get_info<xi::max_mem_bandwidth>(); });
  opt(sycl::aspect::ext_intel_free_memory, "free_memory", [&] { return d.get_info<xi::free_memory>(); });
  if (d.has(sycl::aspect::ext_intel_pci_address)) w.kv("pci_address", d.get_info<xi::pci_address>());
  if (d.has(sycl::aspect::ext_intel_device_id)) w.kv("device_id", uint64_t(d.get_info<xi::device_id>()));
  w.end_object();
}

// MemAvailable from /proc/meminfo, in bytes (0 when unreadable).
uint64_t mem_available() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  uint64_t kib = 0;
  while (f >> key) {
    if (key == "MemAvailable:") {
      f >> kib;
      return kib << 10;
    }
    f.ignore(1 << 10, '\n');
  }
  return 0;
}

// Allocate device memory in chunks, touching each with a memset so it is backed, up to the reported
// size. Never on an integrated GPU, whose device memory is the host's RAM (an earlier version ran
// fiona out of memory that way), and it stops as soon as the host's available memory falls by more
// than 512 MiB: xe can evict device allocations to system RAM under pressure.
void alloc_probe(json::Writer& w, sycl::queue& q) {
  if (rt::integrated(q.get_device())) {
    w.key("allocation").begin_object().kv("skipped", "integrated GPU").end_object();
    std::printf("allocation: skipped on an integrated GPU\n");
    return;
  }
  const uint64_t chunk = 256ull << 20;
  const uint64_t cap = q.get_device().get_info<sycl::info::device::global_mem_size>();
  const uint64_t host_before = mem_available();
  std::vector<void*> blocks;
  uint64_t total = 0;
  std::string stop = "reached the reported size";
  auto t0 = Clock::now();
  while (total + chunk <= cap) {
    if (host_before && mem_available() + (512ull << 20) < host_before) {
      stop = "host memory falling (spill to system RAM?)";
      break;
    }
    void* p = nullptr;
    try {
      p = sycl::malloc_device(chunk, q);
      if (!p) {
        stop = "malloc_device returned null";
        break;
      }
      q.memset(p, 0x5a, chunk).wait_and_throw();
    } catch (const sycl::exception& e) {
      if (p) sycl::free(p, q);
      stop = std::string("exception: ") + e.what();
      break;
    }
    blocks.push_back(p);
    total += chunk;
  }
  double secs = since(t0);
  // Spot-check that the last block holds what was written.
  bool ok = true;
  if (!blocks.empty()) {
    std::vector<uint8_t> back(4096);
    q.memcpy(back.data(), static_cast<uint8_t*>(blocks.back()) + chunk - back.size(), back.size()).wait();
    ok = std::all_of(back.begin(), back.end(), [](uint8_t b) { return b == 0x5a; });
  }
  for (void* p : blocks) sycl::free(p, q);
  w.key("allocation").begin_object().kv("chunk_bytes", chunk).kv("allocated_bytes", total)
      .kv("host_available_before", host_before).kv("host_available_after", mem_available())
      .kv("allocated_gib", total / 1073741824.0).kv("stopped_by", stop).kv("seconds", secs).kv("verified", ok)
      .end_object();
  std::printf("allocation: %.2f GiB in %llu MiB chunks (%s), verified %s\n", total / 1073741824.0,
              (unsigned long long)(chunk >> 20), stop.c_str(), ok ? "yes" : "NO");
}

// Copy rates for one direction and host memory kind; each size is copied until about 0.3 s has
// passed, and the fastest and median copies are reported.
void copy_probe(json::Writer& w, sycl::queue& q) {
  const std::vector<uint64_t> sizes = {4ull << 10, 64ull << 10, 1ull << 20, 16ull << 20, 256ull << 20, 1ull << 30};
  const uint64_t maxb = sizes.back();
  uint8_t* dev = sycl::malloc_device<uint8_t>(maxb, q);
  uint8_t* pinned = sycl::malloc_host<uint8_t>(maxb, q);
  std::vector<uint8_t> pageable(maxb);
  for (uint64_t i = 0; i < maxb; i += 4096) pinned[i] = pageable[i] = uint8_t(i >> 12);

  w.key("copies").begin_array();
  for (int kind = 0; kind < 2; ++kind) {
    uint8_t* host = kind == 0 ? pinned : pageable.data();
    for (int dir = 0; dir < 2; ++dir) {
      for (uint64_t n : sizes) {
        if (kind == 1 && n < (1ull << 20)) continue;  // pageable small copies say nothing new
        std::vector<double> t;
        auto start = Clock::now();
        while (since(start) < 0.3 || t.size() < 3) {
          auto t0 = Clock::now();
          if (dir == 0) q.memcpy(dev, host, n).wait();
          else q.memcpy(host, dev, n).wait();
          t.push_back(since(t0));
          if (t.size() >= 2000) break;
        }
        auto s = stats(t);
        const char* kname = kind == 0 ? "pinned" : "pageable";
        const char* dname = dir == 0 ? "h2d" : "d2h";
        w.begin_object().kv("host", kname).kv("direction", dname).kv("bytes", n).kv("repeats", uint64_t(t.size()))
            .kv("best_gbps", n / s.min / 1e9).kv("median_gbps", n / s.median / 1e9).kv("median_us", s.median * 1e6)
            .end_object();
        if (n >= (16ull << 20))
          std::printf("copy %-8s %s %6llu MiB: best %.2f GB/s, median %.2f GB/s\n", kname, dname,
                      (unsigned long long)(n >> 20), n / s.min / 1e9, n / s.median / 1e9);
      }
    }
  }
  w.end_array();

  // Round trip through the device, checked word for word.
  for (uint64_t i = 0; i < (256ull << 20) / 8; ++i) reinterpret_cast<uint64_t*>(pinned)[i] = i * 0x9e3779b97f4a7c15ull;
  q.memcpy(dev, pinned, 256ull << 20).wait();
  std::memset(pinned, 0, 256ull << 20);
  q.memcpy(pinned, dev, 256ull << 20).wait();
  bool ok = true;
  for (uint64_t i = 0; i < (256ull << 20) / 8 && ok; ++i) ok = reinterpret_cast<uint64_t*>(pinned)[i] == i * 0x9e3779b97f4a7c15ull;
  w.kv("copy_round_trip_verified", ok);
  std::printf("copy round trip of 256 MiB verified: %s\n", ok ? "yes" : "NO");
  sycl::free(dev, q);
  sycl::free(pinned, q);
}

// Kernel launch cost: one empty kernel submitted and waited for, and many submitted back to back
// with one wait at the end.
void launch_probe(json::Writer& w, sycl::queue& q) {
  uint32_t* sink = sycl::malloc_device<uint32_t>(1, q);
  for (int i = 0; i < 50; ++i) q.single_task([=] { sink[0] = 1; }).wait();

  std::vector<double> rt;
  for (int i = 0; i < 2000; ++i) {
    auto t0 = Clock::now();
    q.single_task([=] { sink[0] = 2; }).wait();
    rt.push_back(since(t0));
  }
  const int n = 20000;
  auto t0 = Clock::now();
  for (int i = 0; i < n; ++i) q.single_task([=] { sink[0] = 3; });
  q.wait();
  double per = since(t0) / n;

  // A kernel's own run time, from event profiling, for a sense of the device-side floor.
  sycl::queue pq(q.get_context(), q.get_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
  std::vector<double> dev_t;
  for (int i = 0; i < 200; ++i) {
    auto e = pq.single_task([=] { sink[0] = 4; });
    e.wait();
    dev_t.push_back((e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                     e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9);
  }
  sycl::free(sink, q);

  auto s = stats(rt), sd = stats(dev_t);
  w.key("launch").begin_object();
  write_stats(w, "submit_wait_us", s, 1e6);
  w.kv("back_to_back_us", per * 1e6);
  write_stats(w, "device_time_us", sd, 1e6);
  w.end_object();
  std::printf("launch: submit+wait median %.1f us (p99 %.1f), back-to-back %.2f us/kernel, device time %.2f us\n",
              s.median * 1e6, s.p99 * 1e6, per * 1e6, sd.median * 1e6);
}

// Host -> running kernel -> host: a kernel announces that it runs, spins on a flag in host memory,
// and acknowledges when it sees the flag; the host times from its write to the acknowledgement.
// A plain system-scope atomic load on sycl::malloc_host memory never saw the host's write on fiona
// (2026-10-24: 300 of 300 lost), so each way of reading the flag is tried and reported:
//   atomic        seq_cst system-scope atomic load
//   atomic+fence  the same, with a system-scope acquire fence after each poll
//   uncached      a plain load through an annotated_ptr asking for no L1/L3 caching
// on memory from sycl::malloc_host and from zeMemAllocHost with the uncached bias flag.
namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;
using uncached_ptr = syclex::annotated_ptr<
    uint32_t, decltype(syclex::properties(intelex::read_hint<intelex::cache_control<
                                              intelex::cache_mode::uncached, intelex::cache_level::L1, intelex::cache_level::L3>>))>;

enum class Poll { Atomic, AtomicFence, Uncached };

template <Poll P>
void wake_kernel(sycl::queue& q, uint32_t* cell) {
  q.single_task([=] {
    using ref = sycl::atomic_ref<uint32_t, sycl::memory_order::seq_cst, sycl::memory_scope::system,
                                 sycl::access::address_space::global_space>;
    ref(cell[0]).store(1);
    // Bounded: a poll that really crosses PCIe takes about a microsecond, so 1e6 polls give up
    // within a second or two, well before the job timeout; cached polls give up in milliseconds.
    uint64_t polls = 0;
    for (;;) {
      uint32_t v;
      if constexpr (P == Poll::Uncached) {
        uncached_ptr up(cell + 1);
        v = *up;
      } else {
        v = ref(cell[1]).load();
        if constexpr (P == Poll::AtomicFence) sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
      }
      if (v != 0 || ++polls >= 1000000ull) break;
    }
    if (polls >= 1000000ull) ref(cell[3]).store(1);
    ref(cell[2]).store(1);
  }).wait();
}

template <Poll P>
void wake_variant(json::Writer& w, sycl::queue& q, uint32_t* cell, const char* memory, const char* poll) {
  std::vector<double> rt;
  int lost = 0;
  for (int i = 0; i < 100; ++i) {
    std::memset(cell, 0, 4 * sizeof(uint32_t));
    std::thread gpu([&] { wake_kernel<P>(q, cell); });
    auto started = Clock::now();
    while (__atomic_load_n(&cell[0], __ATOMIC_SEQ_CST) == 0 && since(started) < 2.0) {
    }
    auto t0 = Clock::now();
    __atomic_store_n(&cell[1], 1u, __ATOMIC_SEQ_CST);
    while (__atomic_load_n(&cell[2], __ATOMIC_SEQ_CST) == 0 && since(t0) < 3.0) {
    }
    double dt = since(t0);
    gpu.join();
    if (cell[3] || !cell[2]) ++lost;
    else rt.push_back(dt);
  }
  w.begin_object().kv("memory", memory).kv("poll", poll).kv("samples", uint64_t(rt.size())).kv("lost", uint64_t(lost));
  if (!rt.empty()) write_stats(w, "round_trip_us", stats(rt), 1e6);
  w.end_object();
  if (rt.empty()) {
    std::printf("host signal %-14s %-12s: never seen (%d lost)\n", memory, poll, lost);
  } else {
    auto s = stats(rt);
    std::printf("host signal %-14s %-12s: median %.2f us, p99 %.2f us, %d of 100 lost\n", memory, poll, s.median * 1e6,
                s.p99 * 1e6, lost);
  }
}

void wake_probe(json::Writer& w, sycl::queue& q) {
  auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED};
  void* zmem = nullptr;
  if (zeMemAllocHost(zctx, &hd, 64, 64, &zmem) != ZE_RESULT_SUCCESS) zmem = nullptr;
  auto* smem = sycl::malloc_host<uint32_t>(16, q);

  w.key("host_signal").begin_array();
  for (auto [mem, name] : {std::pair{smem, "malloc_host"}, std::pair{static_cast<uint32_t*>(zmem), "ze_uncached"}}) {
    if (!mem) continue;
    wake_variant<Poll::Atomic>(w, q, mem, name, "atomic");
    wake_variant<Poll::AtomicFence>(w, q, mem, name, "atomic+fence");
    wake_variant<Poll::Uncached>(w, q, mem, name, "uncached");
  }
  w.end_array();
  if (zmem) zeMemFree(zctx, zmem);
  sycl::free(smem, q);
}

}  // namespace

int main(int argc, char** argv) {
  std::string json_path;
  bool skip_alloc = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else if (!std::strcmp(argv[i], "--skip-alloc")) skip_alloc = true;
    else {
      std::fprintf(stderr, "usage: eklow-probe [--json FILE] [--skip-alloc]\n");
      return 2;
    }
  }
  try {
    sycl::device dev = rt::pick_device();
    sycl::queue q(dev, sycl::property::queue::in_order());
    json::Writer w;
    w.begin_object();
    device_info(w, dev);
    std::printf("device: %s (%s), %.2f GiB reported\n", dev.get_info<sycl::info::device::name>().c_str(),
                backend_name(dev.get_backend()), dev.get_info<sycl::info::device::global_mem_size>() / 1073741824.0);
    w.key("env").begin_object();
    for (const char* k : {"ONEAPI_DEVICE_SELECTOR", "UR_L0_USE_IMMEDIATE_COMMANDLISTS", "UR_L0_ENABLE_ZESINIT_DEFAULT",
                          "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"}) {
      const char* v = std::getenv(k);
      if (v) w.kv(k, v);
      else w.key(k).null();
    }
    w.end_object();
    if (!skip_alloc) alloc_probe(w, q);
    copy_probe(w, q);
    launch_probe(w, q);
    wake_probe(w, q);
    w.end_object();
    if (!json_path.empty()) {
      std::ofstream(json_path) << w.str() << "\n";
      std::printf("wrote %s\n", json_path.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-probe: %s\n", e.what());
    return 1;
  }
  return 0;
}
