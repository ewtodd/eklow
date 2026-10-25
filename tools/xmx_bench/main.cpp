// eklow-xmx-bench: what the A770M's XMX units deliver for the inner loops eklow could use, and how
// fast its VRAM reads.
//
//   eklow-xmx-bench [--json FILE]
//
// 1. verify: one dpas per operand type against a CPU reference, which pins down the packed (VNNI)
//    layouts the kernels must produce. Mixed int8 x int4 is tried with two candidate B layouts and
//    the one that matches is reported.
// 2. peak: register-resident dpas loops (fp16, bf16, s8, s8 x s4, s4) with several independent
//    accumulators, no memory traffic.
// 3. Q4_0 inner loops: what a weight block costs on top of the dpas:
//      s8xs4+scale   int8 activations x 4-bit weights (Q4_0's q - 8 is its nibble with bit 3
//                    flipped), each K=32 dpas being one Q4_0 block, then int32 -> float and the
//                    per-row / per-column block scales;
//      fp16+dequant  nibbles unpacked to fp16 and scaled in registers, then reused by several
//                    fp16 dpas (one per 8 activation rows).
// 4. bandwidth: block loads streaming 1 GiB of VRAM.
// Kernels run 100-400 ms, far below xe's 5 s job timeout; each is launched once untimed (the clock
// ramps up) and then timed three times, keeping the fastest.
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "rt/device.hpp"

using namespace eklow;
namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using xmx::dpas_argument_type;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace {

constexpr int kM = 8;  // repeat count: activation rows per dpas
constexpr int kN = 8;  // execution size on DG2: weight columns per dpas

double kernel_seconds(const sycl::event& e) {
  return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
          e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-9;
}

// One untimed launch, then the fastest of three.
template <class F>
double best_seconds(F&& launch) {
  launch().wait();
  double best = 1e30;
  for (int i = 0; i < 3; ++i) {
    auto e = launch();
    e.wait();
    best = std::min(best, kernel_seconds(e));
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// verify

// B (K x N) in the VNNI layout dpas expects: groups of `ops` consecutive k for one column n are
// adjacent, i.e. element (k, n) sits at ((k / ops) * N + n) * ops + k % ops.
template <class T>
std::vector<T> vnni(const std::vector<T>& b, int K, int ops) {
  std::vector<T> out(b.size());
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < kN; ++n) out[((k / ops) * kN + n) * ops + k % ops] = b[k * kN + n];
  return out;
}

// Packs signed 4-bit values two per byte, the lower index in the low nibble.
std::vector<uint8_t> pack_s4(const std::vector<int>& v) {
  std::vector<uint8_t> out(v.size() / 2);
  for (size_t i = 0; i < v.size(); ++i) out[i / 2] |= uint8_t((v[i] & 0xf) << (4 * (i % 2)));
  return out;
}

struct VerifyResult {
  std::string kind;
  std::string layout;
  double max_err;
  bool ok;
};

template <class T>
T* upload(sycl::queue& q, const std::vector<T>& v) {
  T* p = sycl::malloc_device<T>(v.size(), q);
  q.memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  return p;
}

VerifyResult verify_fp16(sycl::queue& q, std::mt19937& rng) {
  constexpr int K = 16;
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> a(kM * K), b(K * kN);
  for (auto& x : a) x = float(sycl::half(u(rng)));
  for (auto& x : b) x = float(sycl::half(u(rng)));
  std::vector<sycl::half> ah(a.begin(), a.end()), bh(b.begin(), b.end());
  auto bv = vnni(bh, K, 2);
  auto* da = upload(q, ah);
  auto* db = upload(q, bv);
  auto* dc = sycl::malloc_device<float>(kM * kN, q);
  q.single_task([=]() SYCL_ESIMD_KERNEL {
     esimd::simd<sycl::half, kM * K> A = esimd::block_load<sycl::half, kM * K>(da);
     esimd::simd<sycl::half, K * kN> B = esimd::block_load<sycl::half, K * kN>(db);
     esimd::simd<float, kM * kN> C = 0;
     C = xmx::dpas<8, kM, float>(C, B, A);
     esimd::block_store<float, kM * kN>(dc, C);
   }).wait();
  std::vector<float> c(kM * kN);
  q.memcpy(c.data(), dc, c.size() * sizeof(float)).wait();
  double err = 0;
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n) {
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += double(a[m * K + k]) * b[k * kN + n];
      err = std::max(err, std::abs(ref - c[m * kN + n]));
    }
  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dc, q);
  return {"fp16", "vnni ops=2", err, err < 1e-3};
}

// Integer dpas: A is s8 (K=32) or s4 (K=64); B is s8 or s4. `bops` is the B layout's group size.
template <dpas_argument_type AP, dpas_argument_type BP>
VerifyResult verify_int(sycl::queue& q, std::mt19937& rng, int bops, const char* kind) {
  constexpr int abits = AP == dpas_argument_type::s4 ? 4 : 8;
  constexpr int bbits = BP == dpas_argument_type::s4 ? 4 : 8;
  constexpr int K = 8 * (32 / std::max(abits, bbits));
  constexpr int ABYTES = kM * K * abits / 8;
  constexpr int BBYTES = K * kN * bbits / 8;
  std::uniform_int_distribution<int> ua(abits == 4 ? -8 : -128, abits == 4 ? 7 : 127);
  std::uniform_int_distribution<int> ub(bbits == 4 ? -8 : -128, bbits == 4 ? 7 : 127);
  std::vector<int> a(kM * K), b(K * kN);
  for (auto& x : a) x = ua(rng);
  for (auto& x : b) x = ub(rng);

  std::vector<uint8_t> abytes, bbytes;
  if constexpr (abits == 4) abytes = pack_s4(a);
  else for (int x : a) abytes.push_back(uint8_t(int8_t(x)));
  std::vector<int> bv(b.size());
  for (int k = 0; k < K; ++k)
    for (int n = 0; n < kN; ++n) bv[((k / bops) * kN + n) * bops + k % bops] = b[k * kN + n];
  if constexpr (bbits == 4) bbytes = pack_s4(bv);
  else for (int x : bv) bbytes.push_back(uint8_t(int8_t(x)));

  auto* da = upload(q, abytes);
  auto* db = upload(q, bbytes);
  auto* dc = sycl::malloc_device<int>(kM * kN, q);
  q.single_task([=]() SYCL_ESIMD_KERNEL {
     esimd::simd<uint8_t, ABYTES> A = esimd::block_load<uint8_t, ABYTES>(da);
     esimd::simd<uint8_t, BBYTES> B = esimd::block_load<uint8_t, BBYTES>(db);
     esimd::simd<int, kM * kN> C = 0;
     C = xmx::dpas<8, kM, int, int, uint8_t, uint8_t, BP, AP>(C, B, A);
     esimd::block_store<int, kM * kN>(dc, C);
   }).wait();
  std::vector<int> c(kM * kN);
  q.memcpy(c.data(), dc, c.size() * sizeof(int)).wait();
  double err = 0;
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n) {
      long ref = 0;
      for (int k = 0; k < K; ++k) ref += long(a[m * K + k]) * b[k * kN + n];
      err = std::max(err, double(std::labs(ref - c[m * kN + n])));
    }
  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dc, q);
  return {kind, "vnni ops=" + std::to_string(bops), err, err == 0};
}

// ---------------------------------------------------------------------------------------------
// peak: U independent accumulators, ITERS dpas each, A and B in registers.

constexpr int kIters = 32768;
constexpr int kAcc = 8;

struct Peak {
  std::string kind;
  double seconds;
  double tops;  // tera-ops: 2 x M x N x K per dpas
};

template <class T, class CT, class BT, class AT, dpas_argument_type BP, dpas_argument_type AP, int K, int BLEN, int ALEN>
Peak run_peak(sycl::queue& q, const char* kind, size_t threads) {
  auto* src = sycl::malloc_device<uint8_t>(4096, q);
  q.memset(src, 0x3c, 4096).wait();  // 0x3c3c: a normal fp16 (1.06) and bf16
  auto* out = sycl::malloc_device<CT>(threads * kM * kN, q);
  auto launch = [&] { return q.parallel_for(sycl::nd_range<1>(threads, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::simd<BT, BLEN> B = esimd::block_load<BT, BLEN>(reinterpret_cast<const BT*>(src));
    esimd::simd<AT, ALEN> A = esimd::block_load<AT, ALEN>(reinterpret_cast<const AT*>(src + 1024));
    esimd::simd<CT, kM * kN> C[kAcc];
#pragma unroll
    for (int u = 0; u < kAcc; ++u) C[u] = 0;
    for (int i = 0; i < kIters; ++i) {
#pragma unroll
      for (int u = 0; u < kAcc; ++u) C[u] = xmx::dpas<8, kM, T, CT, BT, AT, BP, AP>(C[u], B, A);
    }
    esimd::simd<CT, kM * kN> s = C[0];
#pragma unroll
    for (int u = 1; u < kAcc; ++u) s += C[u];
    esimd::block_store<CT, kM * kN>(out + it.get_global_id(0) * kM * kN, s);
  }); };
  double secs = best_seconds(launch);
  sycl::free(src, q);
  sycl::free(out, q);
  double ops = double(threads) * kIters * kAcc * 2.0 * kM * kN * K;
  return {kind, secs, ops / secs / 1e12};
}

// ---------------------------------------------------------------------------------------------
// Q4_0 inner loops. Both compute an (8 * RB) x 8 output tile over many K=32 blocks, the blocks'
// data held in registers (no memory traffic), so they measure the compute cost per block.

constexpr int kRB = 4;  // activation row tiles that share one weight tile

// int8 activations x s4 weights: one K=32 dpas per block and row tile, then int32 -> float,
// times the activation row scales and the weight column scales.
Peak run_s8s4_scaled(sycl::queue& q, size_t threads) {
  auto* src = sycl::malloc_device<uint8_t>(4096, q);
  q.memset(src, 0x3c, 4096).wait();  // 0x3c3c: a normal fp16 (1.06) and bf16
  auto* out = sycl::malloc_device<float>(threads * kRB * kM * kN, q);
  auto launch = [&] { return q.parallel_for(sycl::nd_range<1>(threads, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::simd<uint8_t, 32 * kN / 2> W = esimd::block_load<uint8_t, 32 * kN / 2>(src);
    esimd::simd<uint8_t, kM * 32> X[kRB];
#pragma unroll
    for (int r = 0; r < kRB; ++r) X[r] = esimd::block_load<uint8_t, kM * 32>(src + 256 + 256 * r);
    esimd::simd<float, kN> wscale = esimd::block_load<float, kN>(reinterpret_cast<const float*>(src + 2048));
    esimd::simd<float, kM> xscale = esimd::block_load<float, kM>(reinterpret_cast<const float*>(src + 2560));
    esimd::simd<float, kM * kN> acc[kRB];
#pragma unroll
    for (int r = 0; r < kRB; ++r) acc[r] = 0;
    for (int i = 0; i < kIters; ++i) {
      // The scales change every block; perturb them so the multiply is not hoisted.
      wscale += 1e-7f;
#pragma unroll
      for (int r = 0; r < kRB; ++r) {
        esimd::simd<int, kM * kN> c =
            xmx::dpas<8, kM, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(W, X[r]);
        esimd::simd<float, kM * kN> cf = c;
#pragma unroll
        for (int m = 0; m < kM; ++m) {
          auto row = acc[r].select<kN, 1>(m * kN);
          float xs = xscale[m];
          row += cf.select<kN, 1>(m * kN) * wscale * xs;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < kRB; ++r) esimd::block_store<float, kM * kN>(out + (it.get_global_id(0) * kRB + r) * kM * kN, acc[r]);
  }); };
  double secs = best_seconds(launch);
  sycl::free(src, q);
  sycl::free(out, q);
  double ops = double(threads) * kIters * kRB * 2.0 * kM * kN * 32;
  return {"s8xs4+scale", secs, ops / secs / 1e12};
}

// fp16 activations x Q4_0 weights unpacked in registers: per block (K=32, two K=16 dpas) the 32 x 8
// nibbles become fp16 (q - 8) * d, then every activation row tile reuses them.
Peak run_fp16_dequant(sycl::queue& q, size_t threads) {
  auto* src = sycl::malloc_device<uint8_t>(4096, q);
  q.memset(src, 0x3c, 4096).wait();  // 0x3c3c: a normal fp16 (1.06) and bf16
  auto* out = sycl::malloc_device<float>(threads * kRB * kM * kN, q);
  auto launch = [&] { return q.parallel_for(sycl::nd_range<1>(threads, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    esimd::simd<uint32_t, 32 * kN / 8> Wq = esimd::block_load<uint32_t, 32 * kN / 8>(reinterpret_cast<const uint32_t*>(src));
    esimd::simd<sycl::half, kN> d = esimd::block_load<sycl::half, kN>(reinterpret_cast<const sycl::half*>(src + 2048));
    esimd::simd<sycl::half, kM * 16> X[kRB][2];
#pragma unroll
    for (int r = 0; r < kRB; ++r)
#pragma unroll
      for (int h = 0; h < 2; ++h)
        X[r][h] = esimd::block_load<sycl::half, kM * 16>(reinterpret_cast<const sycl::half*>(src + 256 * (2 * r + h)));
    esimd::simd<float, kM * kN> acc[kRB];
#pragma unroll
    for (int r = 0; r < kRB; ++r) acc[r] = 0;
    for (int i = 0; i < kIters; ++i) {
      Wq ^= uint32_t(i);  // a new block each iteration, so the unpack is not hoisted
      esimd::simd<sycl::half, 32 * kN> Wh;
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        esimd::simd<uint16_t, 32 * kN / 8> nib = (Wq >> (4 * s)) & 0xf;
        Wh.select<32 * kN / 8, 8>(s) = esimd::simd<sycl::half, 32 * kN / 8>(nib);
      }
      Wh = (Wh - sycl::half(8)) * d.replicate<32>();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        esimd::simd<sycl::half, 16 * kN> B = Wh.select<16 * kN, 1>(h * 16 * kN);
#pragma unroll
        for (int r = 0; r < kRB; ++r) acc[r] = xmx::dpas<8, kM, float>(acc[r], B, X[r][h]);
      }
    }
#pragma unroll
    for (int r = 0; r < kRB; ++r) esimd::block_store<float, kM * kN>(out + (it.get_global_id(0) * kRB + r) * kM * kN, acc[r]);
  }); };
  double secs = best_seconds(launch);
  sycl::free(src, q);
  sycl::free(out, q);
  double ops = double(threads) * kIters * kRB * 2.0 * kM * kN * 32;
  return {"fp16+dequant", secs, ops / secs / 1e12};
}

// ---------------------------------------------------------------------------------------------
// bandwidth: each thread streams a contiguous slice with 256-byte block loads.

// Threads read interleaved 256-byte blocks (thread t takes blocks t, t + threads, ...), LOADS of
// them in flight per iteration.
template <int LOADS>
double run_bandwidth(sycl::queue& q, uint32_t* buf, size_t bytes, size_t threads, uint32_t* out) {
  const size_t blocks = bytes / 256;
  double best = 0;
  for (int rep = 0; rep < 5; ++rep) {
    auto e = q.parallel_for(sycl::nd_range<1>(threads, 16), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      const size_t t = it.get_global_id(0);
      esimd::simd<uint32_t, 64> s[LOADS];
#pragma unroll
      for (int l = 0; l < LOADS; ++l) s[l] = 0;
      for (size_t b = t; b + (LOADS - 1) * threads < blocks; b += LOADS * threads) {
#pragma unroll
        for (int l = 0; l < LOADS; ++l) s[l] += esimd::block_load<uint32_t, 64>(buf + (b + l * threads) * 64);
      }
#pragma unroll
      for (int l = 1; l < LOADS; ++l) s[0] += s[l];
      esimd::block_store<uint32_t, 16>(out + t * 16, s[0].template select<16, 1>(0));
    });
    e.wait();
    best = std::max(best, bytes / kernel_seconds(e) / 1e9);
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else {
      std::fprintf(stderr, "usage: eklow-xmx-bench [--json FILE]\n");
      return 2;
    }
  }
  try {
    sycl::device dev = rt::pick_device();
    sycl::queue q(dev, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    const unsigned eus = dev.get_info<sycl::info::device::max_compute_units>();
    const unsigned mhz = dev.get_info<sycl::info::device::max_clock_frequency>();
    std::printf("device: %s, %u EUs, %u MHz max\n", dev.get_info<sycl::info::device::name>().c_str(), eus, mhz);
    json::Writer w;
    w.begin_object().kv("device", dev.get_info<sycl::info::device::name>()).kv("eus", uint64_t(eus)).kv("max_clock_mhz", uint64_t(mhz));

    std::mt19937 rng(7);
    std::vector<VerifyResult> checks = {
        verify_fp16(q, rng),
        verify_int<dpas_argument_type::s8, dpas_argument_type::s8>(q, rng, 4, "s8"),
        verify_int<dpas_argument_type::s8, dpas_argument_type::s4>(q, rng, 4, "s8xs4"),
        verify_int<dpas_argument_type::s8, dpas_argument_type::s4>(q, rng, 8, "s8xs4"),
        verify_int<dpas_argument_type::s4, dpas_argument_type::s4>(q, rng, 8, "s4"),
    };
    w.key("verify").begin_array();
    for (auto& c : checks) {
      w.begin_object().kv("kind", c.kind).kv("layout", c.layout).kv("max_err", c.max_err).kv("ok", c.ok).end_object();
      std::printf("verify %-6s %-12s max error %-10g %s\n", c.kind.c_str(), c.layout.c_str(), c.max_err, c.ok ? "ok" : "MISMATCH");
    }
    w.end_array();

    // Enough threads to fill every EU's 8 hardware threads four times over.
    const size_t threads = size_t(eus) * 8 * 4;
    std::vector<Peak> peaks;
    peaks.push_back(run_peak<float, float, sycl::half, sycl::half, dpas_argument_type::fp16, dpas_argument_type::fp16, 16, 128, 128>(q, "fp16", threads));
    peaks.push_back(run_peak<float, float, bf16, bf16, dpas_argument_type::bf16, dpas_argument_type::bf16, 16, 128, 128>(q, "bf16", threads));
    peaks.push_back(run_peak<int, int, uint8_t, uint8_t, dpas_argument_type::s8, dpas_argument_type::s8, 32, 256, 256>(q, "s8", threads));
    peaks.push_back(run_peak<int, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8, 32, 128, 256>(q, "s8xs4", threads));
    peaks.push_back(run_peak<int, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s4, 64, 256, 256>(q, "s4", threads));
    peaks.push_back(run_s8s4_scaled(q, threads));
    peaks.push_back(run_fp16_dequant(q, threads));
    w.key("throughput").begin_array();
    for (auto& p : peaks) {
      w.begin_object().kv("kind", p.kind).kv("seconds", p.seconds).kv("tops", p.tops).end_object();
      std::printf("%-14s %7.1f TOPS  (%.0f ms)\n", p.kind.c_str(), p.tops, p.seconds * 1e3);
    }
    w.end_array();

    const size_t bw_bytes = 1ull << 30;
    auto* buf = sycl::malloc_device<uint32_t>(bw_bytes / 4, q);
    q.memset(buf, 1, bw_bytes).wait();
    auto* sink = sycl::malloc_device<uint32_t>(size_t(eus) * 8 * 8 * 16, q);
    w.key("read_bandwidth").begin_array();
    double gbps = 0;
    for (size_t mult : {2, 4, 8}) {
      size_t th = size_t(eus) * 8 * mult;
      for (auto [loads, g] : {std::pair{2, run_bandwidth<2>(q, buf, bw_bytes, th, sink)},
                              std::pair{4, run_bandwidth<4>(q, buf, bw_bytes, th, sink)},
                              std::pair{8, run_bandwidth<8>(q, buf, bw_bytes, th, sink)}}) {
        w.begin_object().kv("threads", uint64_t(th)).kv("loads_in_flight", loads).kv("gbps", g).end_object();
        std::printf("VRAM read: %6zu threads, %d loads in flight: %.0f GB/s\n", th, loads, g);
        gbps = std::max(gbps, g);
      }
    }
    w.end_array();
    w.kv("read_bandwidth_gbps", gbps);
    sycl::free(buf, q);
    sycl::free(sink, q);
    w.end_object();
    if (!json_path.empty()) {
      std::ofstream(json_path) << w.str() << "\n";
      std::printf("wrote %s\n", json_path.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-xmx-bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
