// eklow-cpu-qgemv-test: the host CPU's GEMVs for the cold experts (src/cpu/qgemv.cpp, q4_gemv.cpp), checked and timed.
//
//   eklow-cpu-qgemv-test [--formats Q4_0,Q8_0,Q4_K,Q5_K,Q6_K,IQ4_XS] [--gguf FORMAT=PATH ...] [--check] [--bench] [--bw]
//                        [--threads LIST] [--json FILE]
//
// --check: 1, 8 and 16 inputs against rows of real tensors (a GGUF of each format, --gguf) and random valid blocks at the
//   expert shapes (gate|up 2048 -> 2 x 512, down 512 -> 2048): (1) an fp32 dot of ggml's own dequantization with the fp32
//   input (the int8 quantization of the input shows: ~1e-2) and (2) an exact emulation of the integer arithmetic on
//   the same int8 input (~1e-6).
// --bw: what the memory gives: threads streaming a large buffer, and random 1.2 MB reads across a larger pool (the cold
//   experts' access pattern).
// --bench: one cold expert's FFN (gate|up GEMV, SwiGLU, requantize, down GEMV) over a pool of 64 experts in order of a
//   random permutation, per format, thread count and number of inputs (1: decode; 8, 16: a verify block's tokens that
//   pick the same expert share its read), in GB/s of weight bytes read.
#include <immintrin.h>

#include <map>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/json.hpp"
#include "cpu/qgemv.hpp"
#include "model/gguf.hpp"
#include "../qfmt_test/refs.hpp"

using namespace eklow;
using cpu::QType;

namespace {

struct FormatInfo {
  const char* name;
  gguf::Type type;
  QType qtype;
};
const FormatInfo kFormats[] = {
    {"Q4_0", gguf::Type::Q4_0, QType::Q4_0}, {"Q8_0", gguf::Type::Q8_0, QType::Q8_0}, {"Q4_K", gguf::Type::Q4_K, QType::Q4_K},
    {"Q5_K", gguf::Type::Q5_K, QType::Q5_K}, {"Q6_K", gguf::Type::Q6_K, QType::Q6_K}, {"IQ4_XS", gguf::Type::IQ4_XS, QType::IQ4_XS},
};

std::mt19937 g_rng(5);
using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

std::vector<float> random_x(int64_t k) {
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> x(static_cast<size_t>(k));
  for (auto& v : x) v = nd(g_rng);
  for (int64_t c = 0; c < k; c += 487) x[size_t(c)] *= 30;
  return x;
}

refs::QRow qrow_of(const cpu::Q8Blocks& x) {
  refs::QRow r;
  r.q.assign(x.q.begin(), x.q.end());
  r.s = x.s;
  return r;
}

struct CheckResult {
  double fp32 = 0, emul = 0;
};

// y[j][r] for the nx inputs against the references at sampled rows.
CheckResult check_rows(const FormatInfo& f, const uint8_t* rows, int64_t k, int64_t n, int nx, cpu::Pool& pool) {
  std::vector<std::vector<float>> xs;
  std::vector<cpu::Q8Blocks> qs;
  for (int j = 0; j < nx; ++j) {
    xs.push_back(random_x(k));
    qs.push_back(cpu::quantize(xs.back().data(), k));
  }
  std::vector<std::vector<float>> ys(static_cast<size_t>(nx), std::vector<float>(static_cast<size_t>(n)));
  std::vector<const cpu::Q8Blocks*> qp;
  std::vector<float*> yp;
  for (int j = 0; j < nx; ++j) qp.push_back(&qs[size_t(j)]), yp.push_back(ys[size_t(j)].data());
  if (nx == 1) cpu::gemv_q(f.qtype, rows, k, n, qs[0], yp[0], pool);
  else pool.for_each(n, 16, [&](int64_t lo, int64_t hi) { cpu::gemv_q_multi(f.qtype, rows, k, lo, hi, qp.data(), nx, yp.data()); });
  const uint64_t rb = refs::row_bytes(f.type, k);
  std::vector<float> deq(static_cast<size_t>(k));
  double num = 0, den = 0, emax = 0, ersum = 0;
  int cnt = 0;
  for (int i = 0; i < 48; ++i) {
    const int64_t r = int64_t(g_rng() % uint32_t(n));
    refs::dequant_row(f.type, rows + uint64_t(r) * rb, k, deq.data());
    for (int j = 0; j < nx; ++j) {
      double ref = 0;
      for (int64_t c = 0; c < k; ++c) ref += double(xs[size_t(j)][size_t(c)]) * double(deq[size_t(c)]);
      const double got = ys[size_t(j)][size_t(r)];
      num += (got - ref) * (got - ref);
      den += ref * ref;
      const double em = refs::emulate(f.type, rows + uint64_t(r) * rb, k, qrow_of(qs[size_t(j)]));
      ersum += em * em;
      emax = std::max(emax, std::fabs(got - em));
      ++cnt;
    }
  }
  return {std::sqrt(num / std::max(den, 1e-30)), emax / std::max(std::sqrt(ersum / std::max(cnt, 1)), 1e-30)};
}

void run_check(const FormatInfo& f, const std::map<std::string, std::string>& ggufs, json::Writer& jw) {
  cpu::Pool pool(8);
  jw.key(std::string("check_") + f.name).begin_array();
  auto report = [&](const char* label, int64_t k, int64_t n, int nx, const CheckResult& c) {
    std::printf("  %-8s %-26s %5lldx%-6lld inputs %2d  vs fp32 %9.2e  vs emul %9.2e %s\n", f.name, label, (long long)k, (long long)n, nx, c.fp32, c.emul,
                (c.fp32 < 3e-2 && c.emul < 2e-5) ? "" : "  <-- OUT OF BAND");
    jw.begin_object().kv("format", f.name).kv("shape", label).kv("k", k).kv("n", n).kv("inputs", nx).kv("rel_vs_fp32", c.fp32).kv("rel_vs_emulation", c.emul).end_object();
  };
  auto it = ggufs.find(f.name);
  if (it != ggufs.end()) {
    auto file = gguf::File::open(it->second);
    for (const char* name : {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.5.attn_q.weight"}) {
      const gguf::Tensor* t = file.tensor(name);
      if (!t || t->type != f.type) continue;
      const int64_t k = int64_t(t->ne[0]), n = std::min<int64_t>(int64_t(t->ne[1]), 2048);
      for (int nx : {1, 8, 16}) report((std::string("real ") + name).c_str(), k, n, nx, check_rows(f, t->data, k, n, nx, pool));
    }
  }
  for (auto [label, k, n] : {std::tuple<const char*, int64_t, int64_t>{"expert gate|up", 2048, 1024}, {"expert down", 512, 2048}, {"attn_qkv", 2048, 8192}}) {
    if (k % cpu::qtype_block(f.qtype)) continue;
    const auto rows = refs::random_rows(f.type, n, k, g_rng);
    for (int nx : {1, 2, 5, 8, 16}) report(label, k, n, nx, check_rows(f, rows.data(), k, n, nx, pool));
  }
  jw.end_array();
}

// ----- memory bandwidth

// Threads each stream their slice of `bytes` with 256-byte steps (eight 32-byte loads), best of three.
double stream_gbps(const uint8_t* buf, size_t bytes, int threads) {
  double best = 1e30;
  for (int rep = 0; rep < 4; ++rep) {
    std::vector<std::thread> th;
    std::vector<float> sink(static_cast<size_t>(threads) * 16);
    const auto t0 = Clock::now();
    for (int i = 0; i < threads; ++i)
      th.emplace_back([&, i] {
        const size_t lo = bytes / size_t(threads) * size_t(i) / 256 * 256, hi = bytes / size_t(threads) * size_t(i + 1) / 256 * 256;
        __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (size_t p = lo; p + 128 <= hi; p += 128) {
          a0 = _mm256_add_epi32(a0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + p)));
          a1 = _mm256_add_epi32(a1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + p + 32)));
          a2 = _mm256_add_epi32(a2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + p + 64)));
          a3 = _mm256_add_epi32(a3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + p + 96)));
        }
        const __m256i s = _mm256_add_epi32(_mm256_add_epi32(a0, a1), _mm256_add_epi32(a2, a3));
        sink[size_t(i) * 16] = float(_mm256_extract_epi32(s, 0));
      });
    for (auto& t : th) t.join();
    if (rep) best = std::min(best, since(t0));
  }
  return double(bytes) / best / 1e9;
}

void run_bw(json::Writer& jw) {
  const size_t bytes = size_t(3) << 30;
  std::vector<uint8_t> buf(bytes);
  for (size_t i = 0; i < bytes; i += 4096) buf[i] = uint8_t(i);  // touch every page
  std::memset(buf.data(), 1, bytes);
  jw.key("memory_bandwidth").begin_array();
  std::printf("memory: threads streaming 3 GiB (sequential, per-thread slices)\n");
  for (int th : {1, 2, 4, 6, 8, 12, 16, 20}) {
    const double g = stream_gbps(buf.data(), bytes, th);
    std::printf("  %2d threads %7.1f GB/s\n", th, g);
    jw.begin_object().kv("threads", th).kv("gbps", g).end_object();
  }
  jw.end_array();
}

// ----- one expert's FFN

struct Expert {
  std::vector<uint8_t> gu, dn;
};

struct Ffn {
  const FormatInfo& f;
  int64_t k = 2048, fe = 512, n_gu = 1024, n_dn = 2048;
  // inputs / scratch for nx inputs
  std::vector<cpu::Q8Blocks> x;
  std::vector<std::vector<float>> gu, h, y;

  Ffn(const FormatInfo& fi, int nx) : f(fi), x(static_cast<size_t>(nx)), gu(static_cast<size_t>(nx)), h(static_cast<size_t>(nx)), y(static_cast<size_t>(nx)) {
    for (int j = 0; j < nx; ++j) {
      const auto xr = random_x(k);
      x[size_t(j)] = cpu::quantize(xr.data(), k);
      gu[size_t(j)].resize(size_t(n_gu));
      h[size_t(j)].resize(size_t(fe));
      y[size_t(j)].resize(size_t(n_dn));
    }
  }
  void run(const Expert& e, cpu::Pool& pool) {
    const int nx = int(x.size());
    if (nx == 1) {
      cpu::gemv_q(f.qtype, e.gu.data(), k, n_gu, x[0], gu[0].data(), pool);
      cpu::glu(gu[0].data(), fe, cpu::CpuGlu::Silu, h[0].data());
      const cpu::Q8Blocks hq = cpu::quantize(h[0].data(), fe);
      cpu::gemv_q(f.qtype, e.dn.data(), fe, n_dn, hq, y[0].data(), pool);
      return;
    }
    std::vector<const cpu::Q8Blocks*> xp;
    std::vector<float*> gp, yp;
    for (int j = 0; j < nx; ++j) xp.push_back(&x[size_t(j)]), gp.push_back(gu[size_t(j)].data()), yp.push_back(y[size_t(j)].data());
    pool.for_each(n_gu, 16, [&](int64_t lo, int64_t hi) { cpu::gemv_q_multi(f.qtype, e.gu.data(), k, lo, hi, xp.data(), nx, gp.data()); });
    std::vector<cpu::Q8Blocks> hq(static_cast<size_t>(nx));
    std::vector<const cpu::Q8Blocks*> hp;
    for (int j = 0; j < nx; ++j) {
      cpu::glu(gu[size_t(j)].data(), fe, cpu::CpuGlu::Silu, h[size_t(j)].data());
      hq[size_t(j)] = cpu::quantize(h[size_t(j)].data(), fe);
      hp.push_back(&hq[size_t(j)]);
    }
    pool.for_each(n_dn, 16, [&](int64_t lo, int64_t hi) { cpu::gemv_q_multi(f.qtype, e.dn.data(), fe, lo, hi, hp.data(), nx, yp.data()); });
  }
};

void run_bench(const FormatInfo& f, const std::vector<int>& thread_list, json::Writer& jw) {
  constexpr int kExperts = 64;
  std::vector<Expert> experts(kExperts);
  for (auto& e : experts) {
    e.gu = refs::random_rows(f.type, 1024, 2048, g_rng);
    e.dn = refs::random_rows(f.type, 2048, 512, g_rng);
  }
  const double bytes = double(experts[0].gu.size() + experts[0].dn.size());
  std::vector<int> order(kExperts * 4);
  for (size_t i = 0; i < order.size(); ++i) order[i] = int(g_rng() % kExperts);
  jw.key(std::string("bench_") + f.name).begin_array();
  for (int nx : {1, 8, 16}) {
    Ffn ffn(f, nx);
    for (int th : thread_list) {
      cpu::Pool pool(th, nx == 1 ? 400 : 50);
      for (int w = 0; w < 8; ++w) ffn.run(experts[size_t(order[size_t(w)])], pool);
      double best = 1e30;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        for (int i = 0; i < int(order.size()); ++i) ffn.run(experts[size_t(order[size_t(i)])], pool);
        best = std::min(best, since(t0) / double(order.size()));
      }
      std::printf("  %-8s inputs %2d threads %2d  one expert's FFN %8.1f us  %6.1f GB/s of weights (%.2f MB)  %6.1f us per input\n", f.name, nx, th,
                  best * 1e6, bytes / best / 1e9, bytes / 1e6, best * 1e6 / nx);
      jw.begin_object().kv("format", f.name).kv("inputs", nx).kv("threads", th).kv("us", best * 1e6).kv("weight_gbps", bytes / best / 1e9)
          .kv("expert_bytes", uint64_t(bytes)).end_object();
    }
  }
  jw.end_array();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> only;
  std::map<std::string, std::string> ggufs;
  std::vector<int> threads = {1, 4, 8, 12, 16, 20};
  std::string json_path;
  bool check = false, bench = false, bw = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--formats" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      for (std::string t; std::getline(ss, t, ',');) only.push_back(t);
    } else if (a == "--gguf" && i + 1 < argc) {
      const std::string v = argv[++i];
      ggufs[v.substr(0, v.find('='))] = v.substr(v.find('=') + 1);
    } else if (a == "--threads" && i + 1 < argc) {
      threads.clear();
      std::stringstream ss(argv[++i]);
      for (std::string t; std::getline(ss, t, ',');) threads.push_back(std::stoi(t));
    } else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
    else if (a == "--check") check = true;
    else if (a == "--bench") bench = true;
    else if (a == "--bw") bw = true;
    else {
      std::fprintf(stderr, "usage: eklow-cpu-qgemv-test [--formats LIST] [--gguf FORMAT=PATH] [--check] [--bench] [--bw] [--threads LIST] [--json FILE]\n");
      return 2;
    }
  }
  if (!check && !bench && !bw) check = true;
  try {
    json::Writer jw;
    jw.begin_object();
    if (bw) run_bw(jw);
    for (const FormatInfo& f : kFormats) {
      if (!only.empty() && std::find(only.begin(), only.end(), f.name) == only.end()) continue;
      std::printf("== %s\n", f.name);
      if (check) run_check(f, ggufs, jw);
      if (bench) run_bench(f, threads, jw);
    }
    jw.end_object();
    if (!json_path.empty()) std::ofstream(json_path) << jw.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-cpu-qgemv-test: %s\n", e.what());
    return 1;
  }
  return 0;
}
