// eklow-copy-bench: host-to-card copy rates for a cold expert's 3.35 MB (M6 stage 3b), from the GGUF's
// pageable mapping, from pinned host memory and from the mapping registered for device copies, and whether
// a copy on a second queue overlaps the grouped prefill GEMMs.
//
//   eklow-copy-bench --model GGUF
#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kernels/xmx.hpp"
#include "model/gguf.hpp"
#include "rt/copy_engine.hpp"
#include "rt/device.hpp"

using namespace eklow;
using Clock = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--model")) model = argv[i + 1];
  gguf::File f = gguf::File::open(model);
  sycl::queue q(rt::pick_device(), sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
  sycl::queue q2(q.get_context(), q.get_device(), sycl::property_list{sycl::property::queue::in_order()});
  const gguf::Tensor* gu = f.tensor("blk.10.ffn_gate_up_exps.weight");
  const gguf::Tensor* dn = f.tensor("blk.10.ffn_down_exps.weight");
  const int64_t NE = 128, Fe = 704, E = 2816;
  const size_t gu_b = size_t(2 * Fe * (E / 32 * 18)), dn_b = size_t(E * (Fe / 32 * 18)), ex_b = gu_b + dn_b;
  std::printf("expert bytes: gate_up %zu down %zu total %zu\n", gu_b, dn_b, ex_b);
  const int n = 32;  // experts per measurement
  auto* card = sycl::malloc_device<uint8_t>(size_t(n) * ex_b, q);

  auto run = [&](const char* name, const uint8_t* gu_src, const uint8_t* dn_src, sycl::queue& qq) {
    // Whole experts, 2 memcpys each (gate_up and down are separate tensors); three rounds, best.
    double best = 1e9;
    for (int r = 0; r < 4; ++r) {
      const double t0 = now();
      for (int i = 0; i < n; ++i) {
        const int e = (i * 3 + r * 7) % NE;
        qq.memcpy(card + size_t(i) * ex_b, gu_src + size_t(e) * gu_b, gu_b);
        qq.memcpy(card + size_t(i) * ex_b + gu_b, dn_src + size_t(e) * dn_b, dn_b);
      }
      qq.wait();
      const double t = now() - t0;
      if (r > 0) best = std::min(best, t);
    }
    std::printf("%-34s %7.1f us/expert  %6.2f GB/s\n", name, best / n * 1e6, double(n) * double(ex_b) / best / 1e9);
  };

  // Page the mapping in first.
  {
    volatile uint64_t s = 0;
    for (const gguf::Tensor* t : {gu, dn})
      for (size_t o = 0; o < t->nbytes; o += 4096) s = s + t->data[o];
  }
  run("pageable (GGUF mapping)", gu->data, dn->data, q);
  // Pinned copy of the layer's experts (0.43 GB).
  auto* pg = sycl::malloc_host<uint8_t>(gu->nbytes, q);
  auto* pd = sycl::malloc_host<uint8_t>(dn->nbytes, q);
  std::memcpy(pg, gu->data, gu->nbytes);
  std::memcpy(pd, dn->data, dn->nbytes);
  run("pinned (malloc_host)", pg, pd, q);
  // The pageable buffer of plain malloc, for reference.
  auto* mg = static_cast<uint8_t*>(std::aligned_alloc(4096, (gu->nbytes + 4095) & ~size_t(4095)));
  auto* md = static_cast<uint8_t*>(std::aligned_alloc(4096, (dn->nbytes + 4095) & ~size_t(4095)));
  std::memcpy(mg, gu->data, gu->nbytes);
  std::memcpy(md, dn->data, dn->nbytes);
  run("pageable (malloc, copied)", mg, md, q);
#if 1
  try {
    namespace sx = sycl::ext::oneapi::experimental;
    sx::prepare_for_device_copy(const_cast<uint8_t*>(gu->data), gu->nbytes, q);
    sx::prepare_for_device_copy(const_cast<uint8_t*>(dn->data), dn->nbytes, q);
    run("mapping, prepare_for_device_copy", gu->data, dn->data, q);
    {
      rt::CopyEngine ce2(q);
      for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now();
        for (int i = 0; i < n; ++i) {
          ce2.copy(card + size_t(i) * ex_b, gu->data + size_t(i * 3) * gu_b, gu_b);
          ce2.copy(card + size_t(i) * ex_b + gu_b, dn->data + size_t(i * 3) * dn_b, dn_b);
        }
        const double t1 = now();
        ce2.wait(ce2.fence());
        std::printf("copy engine from the registered mapping: %d experts, issue %.0f us, done %.2f ms (%.0f us each)\n", n, (t1 - t0) * 1e6, (now() - t0) * 1e3, (now() - t0) / n * 1e6);
      }
    }
    sx::release_from_device_copy(const_cast<uint8_t*>(gu->data), q);
    sx::release_from_device_copy(const_cast<uint8_t*>(dn->data), q);
  } catch (const std::exception& e) {
    std::printf("prepare_for_device_copy: %s\n", e.what());
  }
#endif

  // Overlap with the grouped GEMMs: 128 experts of random weights, 128 tiles of 64 rows (a 1k chunk's 8k pairs).
  std::mt19937 rng(1);
  const int64_t tile = 64, ntiles = 128;
  std::vector<xmx::Weight> wg, wd;
  auto rnd_weight = [&](int64_t k, int64_t nn) {
    xmx::Weight w{k, nn, nullptr, nullptr};
    const size_t tb = size_t(nn / 8 * (k / 32) * 128), sb = size_t(nn / 8 * (k / 32) * 8);
    w.tiles = sycl::malloc_device<uint8_t>(tb, q);
    w.scales = sycl::malloc_device<sycl::half>(sb, q);
    std::vector<uint8_t> h(tb);
    for (auto& b : h) b = uint8_t(rng());
    q.memcpy(w.tiles, h.data(), tb);
    std::vector<sycl::half> hs(sb, sycl::half(0.01f));
    q.memcpy(w.scales, hs.data(), sb * 2).wait();
    return w;
  };
  for (int e = 0; e < NE; ++e) {
    wg.push_back(rnd_weight(E, 2 * Fe));
    wd.push_back(rnd_weight(Fe, E));
  }
  xmx::ExpertTable tg = xmx::make_expert_table(q, wg), td = xmx::make_expert_table(q, wd);
  std::vector<int32_t> th(4 + 4 * ntiles);
  th[0] = int32_t(ntiles);
  for (int j = 0; j < ntiles; ++j) {
    th[4 + 4 * j] = j;
    th[5 + 4 * j] = int32_t(j * tile);
    th[6 + 4 * j] = int32_t(tile);
  }
  auto* tl = sycl::malloc_device<int32_t>(th.size(), q);
  q.memcpy(tl, th.data(), th.size() * 4).wait();
  const size_t rows = size_t(ntiles * tile);
  auto* a = sycl::malloc_device<sycl::half>(rows * E, q);
  auto* hh = sycl::malloc_device<sycl::half>(rows * Fe, q);
  auto* y = sycl::malloc_device<float>(rows * E, q);
  q.memset(a, 0x3c, rows * E * 2).wait();
  auto gemm = [&] {
    xmx::matmul_geglu_grouped16(q, a, tg, tl, ntiles, tile, hh);
    xmx::matmul_grouped16(q, hh, td, tl, ntiles, tile, y);
  };
  auto time_gemm = [&](int reps) {
    gemm();
    q.wait();
    const double t0 = now();
    for (int i = 0; i < reps; ++i) gemm();
    q.wait();
    return (now() - t0) / reps;
  };
  const double g0 = time_gemm(20);
  std::printf("grouped GEMMs (128 tiles of 64 rows), alone: %.2f ms\n", g0 * 1e3);
  // Copies alone on q2, 32 experts pinned.
  auto copies = [&](sycl::queue& qq, const uint8_t* sg, const uint8_t* sd) {
    for (int i = 0; i < n; ++i) {
      qq.memcpy(card + size_t(i) * ex_b, sg + size_t(i) * gu_b, gu_b);
      qq.memcpy(card + size_t(i) * ex_b + gu_b, sd + size_t(i) * dn_b, dn_b);
    }
  };
  for (const char* what : {"pinned", "pageable"}) {
    const bool pin = what[1] == 'i';
    const uint8_t* sg = pin ? pg : gu->data;
    const uint8_t* sd = pin ? pd : dn->data;
    copies(q2, sg, sd);
    q2.wait();
    double t0 = now();
    copies(q2, sg, sd);
    q2.wait();
    const double c0 = now() - t0;
    // Both at once: GEMMs back to back on q while the copies run on q2; time each to its own completion.
    double best_g = 1e9, best_c = 1e9;
    for (int r = 0; r < 3; ++r) {
      q.wait();
      t0 = now();
      copies(q2, sg, sd);
      auto ce = q2.single_task([] {});
      int reps = 0;
      double tg_end = 0;
      // Keep the card busy with GEMMs until the copies are done.
      while (ce.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete || reps < 2) {
        gemm();
        q.wait();
        ++reps;
        tg_end = now() - t0;
      }
      q2.wait();
      const double tc = now() - t0;
      best_g = std::min(best_g, tg_end / reps);
      best_c = std::min(best_c, tc);
    }
    std::printf("%s: %d experts alone %.2f ms; with GEMMs: copies done at %.2f ms, GEMM %.2f ms each (alone %.2f)\n", what, n, c0 * 1e3,
                best_c * 1e3, best_g * 1e3, g0 * 1e3);
  }
  {
    // The card-side repack of one expert (raw bytes on the card to the tile layout), 32 experts back to back.
    xmx::Weight wgu{E, 2 * Fe, nullptr, nullptr}, wdn{Fe, E, nullptr, nullptr};
    wgu.tiles = sycl::malloc_device<uint8_t>(xmx::weight_tile_bytes(E, 2 * Fe), q);
    wgu.scales = sycl::malloc_device<sycl::half>(xmx::weight_scale_bytes(E, 2 * Fe) / 2, q);
    wdn.tiles = sycl::malloc_device<uint8_t>(xmx::weight_tile_bytes(Fe, E), q);
    wdn.scales = sycl::malloc_device<sycl::half>(xmx::weight_scale_bytes(Fe, E) / 2, q);
    auto rep = [&] {
      for (int i = 0; i < n; ++i) {
        xmx::repack_q4_0_into(q, ref::Weight{gguf::Type::Q4_0, E, 2 * Fe, card + size_t(i) * ex_b, uint64_t(E / 32 * 18)}, true, wgu);
        xmx::repack_q4_0_into(q, ref::Weight{gguf::Type::Q4_0, Fe, E, card + size_t(i) * ex_b + gu_b, uint64_t(Fe / 32 * 18)}, false, wdn);
      }
    };
    rep();
    q.wait();
    const double t0 = now();
    rep();
    q.wait();
    std::printf("card repack of an expert: %.1f us\n", (now() - t0) / n * 1e6);
  }
  // A plain VRAM-bound kernel for comparison: sum of 512 MB.
  auto* big = sycl::malloc_device<float>(size_t(128) << 20, q);
  q.memset(big, 0, size_t(512) << 20).wait();
  auto* sink = sycl::malloc_device<float>(1 << 16, q);
  auto bw = [&] {
    q.parallel_for(sycl::range<1>(1 << 16), [=](sycl::id<1> id) {
      float s = 0;
      for (size_t i = id[0]; i < (size_t(128) << 20); i += (1 << 16)) s += big[i];
      sink[id[0]] = s;
    });
  };
  {
    bw();
    q.wait();
    double t0 = now();
    for (int i = 0; i < 5; ++i) bw();
    q.wait();
    const double alone = (now() - t0) / 5;
    rt::CopyEngine ce(q);
    auto issue = [&] {
      for (int i = 0; i < 4 * n; ++i) {
        ce.copy(card + size_t(i % n) * ex_b, pg + size_t(i % NE) * gu_b, gu_b);
        ce.copy(card + size_t(i % n) * ex_b + gu_b, pd + size_t(i % NE) * dn_b, dn_b);
      }
      return ce.fence();
    };
    t0 = now();
    const int fid = issue();
    int reps = 0;
    while (!ce.done(fid)) {
      bw();
      q.wait();
      ++reps;
    }
    const double dt = now() - t0;
    {
      auto alu = [&] {
        q.parallel_for(sycl::range<1>(1 << 20), [=](sycl::id<1> id) {
          float x = float(id[0]) * 1e-7f, y = 1.0001f;
          for (int i = 0; i < 20000; ++i) x = x * y + 0.5f;
          sink[id[0] & 0xffff] = x;
        });
      };
      alu();
      q.wait();
      double u0 = now();
      for (int i = 0; i < 3; ++i) alu();
      q.wait();
      const double a_alone = (now() - u0) / 3;
      u0 = now();
      const int f2 = issue();
      int rr = 0;
      while (!ce.done(f2)) {
        alu();
        q.wait();
        ++rr;
      }
      std::printf("alu kernel: alone %.2f ms, during pinned copies %.2f ms\n", a_alone * 1e3, (now() - u0) / rr * 1e3);
    }
    {
      // A kernel on the second queue reading pinned host memory into VRAM (zero-copy), beside the bandwidth kernel on q.
      const size_t words = gu->nbytes / 16;
      auto pull = [&](int nwg) {
        auto* dst = reinterpret_cast<sycl::uint4*>(card);
        const sycl::uint4* s4 = reinterpret_cast<const sycl::uint4*>(pg);
        return q2.parallel_for(sycl::nd_range<1>(size_t(nwg) * 256, 256), [=](sycl::nd_item<1> it) {
          for (size_t i = it.get_global_id(0); i < words; i += size_t(nwg) * 256) dst[i] = s4[i];
        });
      };
      for (int nwg : {8, 32, 128}) {
        pull(nwg).wait();
        double u0 = now();
        for (int i = 0; i < 5; ++i) pull(nwg);
        q2.wait();
        const double tp = (now() - u0) / 5;
        u0 = now();
        for (int i = 0; i < 5; ++i) pull(nwg);
        int reps = 0;
        auto ev = q2.single_task([] {});
        while (ev.get_info<sycl::info::event::command_execution_status>() != sycl::info::event_command_status::complete) {
          bw();
          q.wait();
          ++reps;
        }
        std::printf("zero-copy pull, %d groups: %.1f us per %.2f MB (%.2f GB/s); bandwidth kernel beside it %.2f ms (alone 1.31)\n", nwg, tp * 1e6,
                    double(gu->nbytes) / 1e6, double(gu->nbytes) / tp / 1e9, (now() - u0) / std::max(reps, 1) * 1e3);
      }
    }
    std::printf("streaming read of 512 MB: alone %.2f ms (%.0f GB/s); during pinned copies %.2f ms\n", alone * 1e3, 0.5368 / alone, dt / reps * 1e3);
  }
  // The same on the copy engine (Level Zero, rt::CopyEngine).
  try {
    rt::CopyEngine ce(q);
    auto* card2 = sycl::malloc_device<uint8_t>(gu->nbytes + dn->nbytes, q);
    q.memset(card2, 1, gu->nbytes + dn->nbytes).wait();
    for (const char* what : {"pinned", "device"}) {
      const bool pin = what[0] == 'p';
      const uint8_t* sg = pin ? pg : card2;
      const uint8_t* sd = pin ? pd : card2 + gu->nbytes;
      auto issue = [&] {
        for (int i = 0; i < 4 * n; ++i) {
          ce.copy(card + size_t(i % n) * ex_b, sg + size_t(i % NE) * gu_b, gu_b);
          ce.copy(card + size_t(i % n) * ex_b + gu_b, sd + size_t(i % NE) * dn_b, dn_b);
        }
        return ce.fence();
      };
      ce.wait(issue());
      double t0 = now();
      ce.wait(issue());
      const double c0 = now() - t0;
      double best_g = 1e9, best_c = 1e9;
      for (int r = 0; r < 4; ++r) {
        q.wait();
        t0 = now();
        const int fid = issue();
        double t_copy = 0;
        std::thread th([&] {
          ce.wait(fid);
          t_copy = now() - t0;
        });
        // GEMM reps that ended while the copies were still running
        double sum = 0;
        int reps = 0;
        for (;;) {
          const double a0 = now();
          auto e0 = q.single_task([] {});
          gemm();
          auto e1 = q.single_task([] {});
          q.wait();
          const double a1 = now();
          const double dev = double(e1.get_profiling_info<sycl::info::event_profiling::command_end>() - e0.get_profiling_info<sycl::info::event_profiling::command_end>()) * 1e-6;
          if (r == 3) std::printf("  rep %d: %.2f..%.2f ms, card %.2f ms (copy %s)\n", reps, (a0 - t0) * 1e3, (a1 - t0) * 1e3, dev, t_copy ? "done" : "running");
          if (t_copy != 0) break;
          sum += a1 - a0;
          ++reps;
        }
        th.join();
        if (r > 0) {
          best_g = std::min(best_g, sum / std::max(reps, 1));
          best_c = std::min(best_c, t_copy);
        }
      }
      std::printf("copy engine, %s: 4x%d experts alone %.2f ms (%.1f us each); with GEMMs: copies done by %.2f ms, GEMM %.2f ms each (alone %.2f)\n",
                  what, n, c0 * 1e3, c0 / n * 1e6, best_c * 1e3, best_g * 1e3, g0 * 1e3);
    }
    {
      // The fence as a dependency of the queue: a kernel after it must not start before the copies are done.
      for (int rep = 0; rep < 3; ++rep) {
        const double t0 = now();
        for (int i = 0; i < n; ++i) {
          ce.copy(card + size_t(i) * ex_b, pg + size_t(i) * gu_b, gu_b);
          ce.copy(card + size_t(i) * ex_b + gu_b, pd + size_t(i) * dn_b, dn_b);
        }
        sycl::event ev = ce.event(ce.fence());
        const double t1 = now();
        q.submit([&](sycl::handler& h) {
          h.depends_on(ev);
          h.single_task([] {});
        });
        const double t2 = now();
        q.wait();
        std::printf("fence as a queue dependency: %d experts issued in %.0f us, submit %.0f us, kernel done at %.2f ms\n", n, (t1 - t0) * 1e6, (t2 - t1) * 1e6,
                    (now() - t0) * 1e3);
      }
    }
  } catch (const std::exception& e) {
    std::printf("copy engine: %s\n", e.what());
  }
  return 0;
}
