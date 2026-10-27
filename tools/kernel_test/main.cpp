// eklow-kernel-test: the fast kernels against the reference kernels and an exact CPU emulation,
// on real Gemma 4 weights.
//
//   eklow-kernel-test --model GGUF [--moe-model GGUF26B] [--json FILE]
//
// For each weight and row count: the XMX matmul against ref::matmul (fp32; the difference is the
// int8 activation quantization, the same trade llama.cpp makes, or above xmx::kFp16Rows rows the
// fp16 rounding of activations and scaled weights), against a CPU emulation of that arithmetic at
// sampled outputs (should agree to fp32 rounding), and its speed. The two stacked engine shapes
// (gate|up through matmul_geglu, q|k|v through matmul) are checked against ref the same way,
// without the emulation.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "kernels/fast.hpp"
#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

float half_to_float(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(uint16_t(p[0] | p[1] << 8))); }

// The product at (row, col) on the CPU, in double, with the GPU's operands: the Q4_0 weights
// from the file, and int8 activations from the same quantization rule or, with fp16, the
// activations and the scaled weights each rounded to fp16 as the fp16 GEMM's are.
double emulate(const std::vector<float>& x, int64_t k, const gguf::Tensor& w, int64_t row, int64_t col, bool fp16) {
  const uint8_t* wrow = w.data + uint64_t(col) * (w.nbytes / w.ne[1]);
  double total = 0;
  for (int64_t b = 0; b < k / 32; ++b) {
    const float* xs = x.data() + row * k + b * 32;
    const uint8_t* blk = wrow + b * 18;
    if (fp16) {
      const sycl::half d = sycl::bit_cast<sycl::half>(uint16_t(blk[0] | blk[1] << 8));
      for (int j = 0; j < 32; ++j) {
        const int qw = (j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4) - 8;
        total += double(float(sycl::half(xs[j]))) * double(float(sycl::half(float(qw)) * d));
      }
      continue;
    }
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xs[j]));
    const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
    long dot = 0;
    for (int j = 0; j < 32; ++j) {
      const int qa = int(std::round(xs[j] * id));
      const int qw = (j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4) - 8;
      dot += long(qa) * qw;
    }
    total += double(dot) * double(d) * double(half_to_float(blk));
  }
  return total;
}


// The grouped one-row GEMV against xmx::matmul per pair, on the 26B's layer 0 experts: same
// arithmetic, so the relative L2 should be ~1e-6 (only the K split over threads differs).
void grouped_test(sycl::queue& q, const gguf::File& f, json::Writer& w) {
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0, 1);
  const gguf::Tensor* tg = f.tensor("blk.0.ffn_gate_up_exps.weight");
  const gguf::Tensor* td = f.tensor("blk.0.ffn_down_exps.weight");
  if (!tg || !td) throw std::runtime_error("no blk.0 expert tensors in the MoE model");
  w.key("grouped").begin_array();
  std::printf("%-18s %5s %9s %10s %9s %9s\n", "weights", "pairs", "distinct", "rel L2", "ms", "GB/s");
  for (const bool gate_up : {true, false}) {
    const gguf::Tensor* t = gate_up ? tg : td;
    const int64_t n_exp = int64_t(t->ne[2]), n = int64_t(t->ne[1]), k = int64_t(t->ne[0]);
    const uint64_t row_bytes = t->nbytes / uint64_t(n_exp) / uint64_t(n), exp_bytes = row_bytes * uint64_t(n);
    auto* staging = sycl::malloc_device<uint8_t>(exp_bytes, q);
    std::vector<xmx::Weight> experts;
    for (int64_t e = 0; e < n_exp; ++e) {
      q.memcpy(staging, t->data + uint64_t(e) * exp_bytes, exp_bytes).wait();
      experts.push_back(xmx::repack_q4_0(q, ref::Weight{gguf::Type::Q4_0, k, n, staging, row_bytes}, gate_up));
    }
    sycl::free(staging, q);
    xmx::ExpertTable table = xmx::make_expert_table(q, experts);
    for (int64_t P : {8, 16}) {
      const int64_t row_div = gate_up ? 8 : 1, rows = (P + row_div - 1) / row_div;
      xmx::Activations act = xmx::alloc_activations(q, 32, k), one = xmx::alloc_activations(q, 32, k);
      auto* xd = sycl::malloc_device<float>(size_t(rows * k), q);
      auto* yg = sycl::malloc_device<float>(size_t(P * n), q);
      auto* yr = sycl::malloc_device<float>(size_t(n), q);
      auto* idd = sycl::malloc_device<int32_t>(size_t(P), q);
      auto* idsets = sycl::malloc_device<int32_t>(size_t(32 * P), q);
      std::vector<float> x(size_t(rows * k) + 0);
      for (auto& v : x) v = nd(rng);
      for (int64_t r = 0; r < rows; ++r)
        for (int64_t c = 0; c < k; c += 487) x[size_t(r * k + c)] *= 30;
      std::vector<int32_t> ids(size_t(P) + 0);
      for (auto& id : ids) id = int32_t(rng() % uint32_t(n_exp));
      q.memcpy(xd, x.data(), x.size() * sizeof(float));
      q.memcpy(idd, ids.data(), ids.size() * 4).wait();
      xmx::quantize(q, xd, rows, act);
      xmx::matmul_grouped(q, act, table, idd, P, row_div, yg);
      q.wait();
      std::vector<float> hg(size_t(P * n) + 0), hr(size_t(n) + 0);
      q.memcpy(hg.data(), yg, hg.size() * 4).wait();
      double num = 0, den = 0;
      for (int64_t p = 0; p < P; ++p) {
        xmx::quantize(q, xd + (p / row_div) * k, 1, one);
        xmx::matmul(q, one, experts[size_t(ids[size_t(p)])], yr);
        q.memcpy(hr.data(), yr, hr.size() * 4).wait();
        for (int64_t i = 0; i < n; ++i) {
          const double d = double(hg[size_t(p * n + i)]) - hr[size_t(i)];
          num += d * d;
          den += double(hr[size_t(i)]) * hr[size_t(i)];
        }
      }
      std::vector<int32_t> all_ids(size_t(32 * P) + 0);
      int64_t distinct_sum = 0;
      for (int s = 0; s < 32; ++s) {
        for (int64_t p = 0; p < P; ++p) all_ids[size_t(s * P + p)] = int32_t(rng() % uint32_t(n_exp));
        std::vector<int32_t> u(all_ids.begin() + s * P, all_ids.begin() + (s + 1) * P);
        std::sort(u.begin(), u.end());
        distinct_sum += std::unique(u.begin(), u.end()) - u.begin();
      }
      q.memcpy(idsets, all_ids.data(), all_ids.size() * 4).wait();
      double best = 1e30;
      for (int rep = 0; rep < 4; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        // 32 different draws of ids, so the experts come from VRAM and not the 16 MB L2.
        for (int i = 0; i < 32; ++i) xmx::matmul_grouped(q, act, table, idsets + i * P, P, row_div, yg);
        q.wait();
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
        if (rep) best = std::min(best, dt);
      }


      const double distinct = double(distinct_sum) / 32;
      const double gbps = distinct * double(exp_bytes) / best / 1e9;
      std::printf("%-18s %5lld %9.1f %10.2e %9.4f %9.0f\n", gate_up ? "blk.0 gate_up exps" : "blk.0 down exps", (long long)P,
                  distinct, std::sqrt(num / den), best * 1e3, gbps);
      w.begin_object().kv("weight", gate_up ? "gate_up" : "down").kv("pairs", P).kv("distinct", distinct)
          .kv("rel_vs_matmul", std::sqrt(num / den)).kv("ms", best * 1e3).kv("gbps", gbps).end_object();
      if (gate_up) {
        // With the GeGLU and its int8 quantization after it, as the engine runs it.
        xmx::Activations glu = xmx::alloc_activations(q, 32, n / 2);
        auto* scr = sycl::malloc_device<float>(size_t(P * n), q);
        double gbest = 1e30;
        for (int rep = 0; rep < 4; ++rep) {
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < 32; ++i) xmx::matmul_geglu_grouped(q, act, table, idsets + i * P, P, row_div, glu, scr);
          q.wait();
          const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 32;
          if (rep) gbest = std::min(gbest, dt);
        }
        std::printf("%-18s %5lld %9.1f %10s %9.4f %9.0f\n", "  + GeGLU", (long long)P, distinct, "", gbest * 1e3,
                    distinct * double(exp_bytes) / gbest / 1e9);
        sycl::free(scr, q);
        xmx::free_activations(q, glu);
      }
      sycl::free(xd, q);
      sycl::free(yg, q);
      sycl::free(yr, q);
      sycl::free(idd, q);
      sycl::free(idsets, q);
      xmx::free_activations(q, act);
      xmx::free_activations(q, one);
    }
    xmx::free_expert_table(q, table);
    for (auto& e : experts) xmx::free_weight(q, e);
  }
  w.end_array();
}

// The grouped fp16 GEMMs (prefill's MoE) against xmx::matmul_geglu / xmx::matmul per expert, on the
// 26B's layer 0 experts, with T random tokens each routed to a random set of 8 experts. Same fp16
// arithmetic, so the relative L2 should be ~1e-6..1e-4 (a different K order only through the
// work-group's schedule). Also checks fast::moe_group's order on the host, and times the kernels.
void grouped16_test(sycl::queue& q, const gguf::File& f, json::Writer& w, int64_t tile) {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0, 1);
  const gguf::Tensor* tg = f.tensor("blk.0.ffn_gate_up_exps.weight");
  const gguf::Tensor* td = f.tensor("blk.0.ffn_down_exps.weight");
  if (!tg || !td) throw std::runtime_error("no blk.0 expert tensors in the MoE model");
  const int64_t T = 1024, NU = 8, P = T * NU, NE = int64_t(tg->ne[2]);
  const int64_t E = int64_t(tg->ne[0]), Fe = int64_t(tg->ne[1]) / 2;
  std::vector<xmx::Weight> gu, dn;
  for (const bool gate_up : {true, false}) {
    const gguf::Tensor* t = gate_up ? tg : td;
    const int64_t n = int64_t(t->ne[1]), k = int64_t(t->ne[0]);
    const uint64_t row_bytes = t->nbytes / uint64_t(NE) / uint64_t(n), exp_bytes = row_bytes * uint64_t(n);
    auto* staging = sycl::malloc_device<uint8_t>(exp_bytes, q);
    for (int64_t e = 0; e < NE; ++e) {
      q.memcpy(staging, t->data + uint64_t(e) * exp_bytes, exp_bytes).wait();
      (gate_up ? gu : dn).push_back(xmx::repack_q4_0(q, ref::Weight{gguf::Type::Q4_0, k, n, staging, row_bytes}, gate_up));
    }
    sycl::free(staging, q);
  }
  xmx::ExpertTable tab_gu = xmx::make_expert_table(q, gu), tab_dn = xmx::make_expert_table(q, dn);

  // Routing: 8 distinct experts per token.
  std::vector<int32_t> ids(size_t(P) + 0);
  std::vector<int32_t> perm(size_t(NE) + 0);
  for (int64_t t = 0; t < T; ++t) {
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int64_t j = 0; j < NU; ++j) ids[size_t(t * NU + j)] = perm[size_t(j)];
  }
  std::vector<float> x(size_t(T * E));
  for (auto& v : x) v = nd(rng);
  const int64_t max_tiles = (P + tile - 1) / tile + NE, rows = max_tiles * tile;
  auto* idd = sycl::malloc_device<int32_t>(size_t(P), q);
  auto* rowp = sycl::malloc_device<int32_t>(size_t(P), q);
  auto* scr = sycl::malloc_device<int32_t>(size_t((P / 256 + 2) * NE), q);
  auto* tiles = sycl::malloc_device<int32_t>(size_t(4 + 4 * max_tiles), q);
  auto* xd = sycl::malloc_device<float>(size_t(T * E), q);
  auto* a = sycl::malloc_device<sycl::half>(size_t(rows * E), q);
  auto* h = sycl::malloc_device<sycl::half>(size_t(rows * Fe), q);
  auto* y = sycl::malloc_device<float>(size_t(rows * E), q);
  q.memcpy(idd, ids.data(), ids.size() * 4);
  q.memcpy(xd, x.data(), x.size() * 4).wait();
  q.memset(a, 0, size_t(rows * E) * 2).wait();
  fast::moe_group(q, idd, P, NE, tile, scr, tiles, rowp);
  fast::moe_gather(q, xd, rowp, P, NU, E, a);
  xmx::matmul_geglu_grouped16(q, a, tab_gu, tiles, max_tiles, tile, h);
  xmx::matmul_grouped16(q, h, tab_dn, tiles, max_tiles, tile, y);
  q.wait();

  // Host check of the grouping: rows unique, segments tile-aligned and in expert order, each
  // expert's pairs in pair order.
  std::vector<int32_t> hrow(size_t(P) + 0), htiles(size_t(4 + 4 * max_tiles));
  q.memcpy(hrow.data(), rowp, hrow.size() * 4);
  q.memcpy(htiles.data(), tiles, htiles.size() * 4).wait();
  std::vector<std::vector<int64_t>> by_expert(size_t(NE) + 0);
  for (int64_t p = 0; p < P; ++p) by_expert[size_t(ids[size_t(p)])].push_back(p);
  bool order_ok = true;
  int64_t n_tiles = htiles[0], ti = 0;
  for (int64_t e = 0; e < NE; ++e) {
    const int64_t cnt = int64_t(by_expert[size_t(e)].size()), nt = (cnt + tile - 1) / tile;
    int64_t start = -1;
    for (int64_t j = 0; j < nt; ++j, ++ti) {
      order_ok &= ti < n_tiles && htiles[size_t(4 + 4 * ti)] == e && htiles[size_t(6 + 4 * ti)] == std::min(tile, cnt - j * tile);
      if (j == 0) start = htiles[size_t(5 + 4 * ti)];
      order_ok &= ti < n_tiles && htiles[size_t(5 + 4 * ti)] == start + j * tile;
    }
    for (int64_t i = 0; i < cnt; ++i) order_ok &= hrow[size_t(by_expert[size_t(e)][size_t(i)])] == start + i;
  }
  order_ok &= ti == n_tiles;

  // Per expert reference: its rows through the dense fp16 path.
  std::vector<sycl::half> hh(size_t(rows * Fe));
  std::vector<float> hy(size_t(rows * E));
  q.memcpy(hh.data(), h, hh.size() * 2);
  q.memcpy(hy.data(), y, hy.size() * 4).wait();
  double num_gu = 0, den_gu = 0, num_dn = 0, den_dn = 0;
  xmx::Activations ain = xmx::alloc_activations(q, 2048, E), ah = xmx::alloc_activations(q, 2048, Fe);
  auto* xe = sycl::malloc_device<float>(size_t(2048 * E), q);
  auto* ye = sycl::malloc_device<float>(size_t(2048 * E), q);
  std::vector<float> hye(size_t(2048 * E));
  for (int64_t e = 0; e < NE; ++e) {
    const auto& prs = by_expert[size_t(e)];
    if (prs.empty()) continue;
    const int64_t cnt = int64_t(prs.size()), m = std::max<int64_t>(cnt, xmx::kFp16Rows + 1);
    std::vector<float> xs(size_t(m * E), 0.0f);
    for (int64_t i = 0; i < cnt; ++i) std::copy_n(x.begin() + (prs[size_t(i)] / NU) * E, E, xs.begin() + i * E);
    q.memcpy(xe, xs.data(), xs.size() * 4).wait();
    xmx::quantize(q, xe, m, ain);
    xmx::matmul_geglu(q, ain, gu[size_t(e)], ah, nullptr);
    xmx::matmul(q, ah, dn[size_t(e)], ye);
    std::vector<sycl::half> rh(size_t(ah.m_pad * Fe));
    q.memcpy(rh.data(), ah.h, rh.size() * 2);
    q.memcpy(hye.data(), ye, size_t(m * E) * 4).wait();
    for (int64_t i = 0; i < cnt; ++i) {
      const int64_t row = hrow[size_t(prs[size_t(i)])];
      for (int64_t c = 0; c < Fe; ++c) {
        const double g = double(float(hh[size_t(xmx::act16_index(row, c, Fe / 16))])), r = double(float(rh[size_t(xmx::act16_index(i, c, Fe / 16))]));
        num_gu += (g - r) * (g - r);
        den_gu += r * r;
      }
      for (int64_t c = 0; c < E; ++c) {
        const double g = double(hy[size_t(row * E + c)]), r = double(hye[size_t(i * E + c)]);
        num_dn += (g - r) * (g - r);
        den_dn += r * r;
      }
    }
  }

  // Speed: the three grouped kernels, best of several runs; TFLOPS on the real pairs and on the
  // padded rows the tiles hold.
  int64_t used_rows = htiles[0] * tile;
  auto time = [&](auto&& fn) {
    double best = 1e30;
    for (int rep = 0; rep < 6; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < 10; ++i) fn();
      q.wait();
      const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 10;
      if (rep) best = std::min(best, dt);
    }
    return best;
  };
  const double t_gu = time([&] { xmx::matmul_geglu_grouped16(q, a, tab_gu, tiles, max_tiles, tile, h); });
  const double t_dn = time([&] { xmx::matmul_grouped16(q, h, tab_dn, tiles, max_tiles, tile, y); });
  const double t_grp = time([&] {
    fast::moe_group(q, idd, P, NE, tile, scr, tiles, rowp);
    fast::moe_gather(q, xd, rowp, P, NU, E, a);
  });
  const double fl_gu = 2.0 * double(P) * double(E) * double(2 * Fe), fl_dn = 2.0 * double(P) * double(Fe) * double(E);
  std::printf("grouped fp16 GEMMs, %lld tokens x 8, tile %lld rows: grouping %s, %lld tiles (%lld rows for %lld pairs)\n", (long long)T,
              (long long)tile, order_ok ? "ok" : "WRONG", (long long)htiles[0], (long long)used_rows, (long long)P);
  std::printf("%-14s %10s %9s %12s %12s\n", "kernel", "rel L2", "ms", "TFLOPS real", "TFLOPS padded");
  std::printf("%-14s %10.2e %9.4f %12.1f %12.1f\n", "gate_up+GeGLU", std::sqrt(num_gu / den_gu), t_gu * 1e3, fl_gu / t_gu / 1e12,
              fl_gu * double(used_rows) / double(P) / t_gu / 1e12);
  std::printf("%-14s %10.2e %9.4f %12.1f %12.1f\n", "down", std::sqrt(num_dn / den_dn), t_dn * 1e3, fl_dn / t_dn / 1e12,
              fl_dn * double(used_rows) / double(P) / t_dn / 1e12);
  std::printf("%-14s %10s %9.4f\n", "group+gather", "", t_grp * 1e3);
  w.key("grouped16").begin_object().kv("tile", tile).kv("tokens", T).kv("grouping_ok", order_ok).kv("tiles", int64_t(htiles[0]))
      .kv("rel_gate_up", std::sqrt(num_gu / den_gu)).kv("rel_down", std::sqrt(num_dn / den_dn)).kv("ms_gate_up", t_gu * 1e3)
      .kv("ms_down", t_dn * 1e3).kv("tflops_gate_up", fl_gu / t_gu / 1e12).kv("tflops_down", fl_dn / t_dn / 1e12).end_object();
  for (void* p : {static_cast<void*>(idd), static_cast<void*>(rowp), static_cast<void*>(scr), static_cast<void*>(tiles),
                  static_cast<void*>(xd), static_cast<void*>(a), static_cast<void*>(h), static_cast<void*>(y),
                  static_cast<void*>(xe), static_cast<void*>(ye)})
    sycl::free(p, q);
  xmx::free_activations(q, ain);
  xmx::free_activations(q, ah);
  xmx::free_expert_table(q, tab_gu);
  xmx::free_expert_table(q, tab_dn);
  for (auto& e : gu) xmx::free_weight(q, e);
  for (auto& e : dn) xmx::free_weight(q, e);
}
}  // namespace

int main(int argc, char** argv) {
  std::string model, json_path, moe_model;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else if (!std::strcmp(argv[i], "--moe-model") && i + 1 < argc) moe_model = argv[++i];
    else {
      std::fprintf(stderr, "usage: eklow-kernel-test --model GGUF [--moe-model GGUF26B] [--json FILE]\n");
      return 2;
    }
  }
  try {
    if (model.empty() && !moe_model.empty()) {
      // Only the grouped MoE GEMV: --moe-model alone.
      sycl::queue gq(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
      auto fm = gguf::File::open(moe_model);
      json::Writer gw;
      gw.begin_object();
      grouped_test(gq, fm, gw);
      for (const int64_t tile : {32, 64, 128}) grouped16_test(gq, fm, gw, tile);
      gw.end_object();
      if (!json_path.empty()) std::ofstream(json_path) << gw.str() << "\n";
      return 0;
    }
    auto f = gguf::File::open(model);
    sycl::queue q(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    json::Writer w;
    w.begin_object().kv("device", q.get_device().get_info<sycl::info::device::name>()).key("matmul").begin_array();
    std::printf("%-22s %5s %10s %10s %10s %9s %9s\n", "weight", "rows", "vs fp32", "vs emul", "ms", "TOPS", "GB/s");
    for (const char* name : {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.5.attn_q.weight", "blk.0.attn_output.weight"}) {
      const gguf::Tensor* t = f.tensor(name);
      if (!t) throw std::runtime_error(std::string("no tensor ") + name);
      const int64_t k = int64_t(t->ne[0]), n = int64_t(t->ne[1]);
      auto* raw = sycl::malloc_device<uint8_t>(t->nbytes, q);
      q.memcpy(raw, t->data, t->nbytes).wait();
      ref::Weight rw{t->type, k, n, raw, t->nbytes / uint64_t(n)};
      xmx::Weight xw = xmx::repack_q4_0(q, rw);
      const int64_t max_m = 2048;
      xmx::Activations act = xmx::alloc_activations(q, max_m, k);
      auto* xd = sycl::malloc_device<float>(size_t(max_m * k), q);
      auto* yr = sycl::malloc_device<float>(size_t(max_m * n), q);
      auto* yx = sycl::malloc_device<float>(size_t(max_m * n), q);
      for (int64_t m : {1, 2, 5, 8, 16, 32, 64, 256, 1024, 2048}) {
        std::vector<float> x(size_t(m * k));
        for (auto& v : x) v = nd(rng);
        // Gemma's hidden states carry a few large channels; give the input some.
        for (int64_t r = 0; r < m; ++r)
          for (int64_t c = 0; c < k; c += 487) x[size_t(r * k + c)] *= 30;
        q.memcpy(xd, x.data(), x.size() * sizeof(float)).wait();
        ref::matmul(q, rw, xd, m, yr);
        xmx::quantize(q, xd, m, act);
        xmx::matmul(q, act, xw, yx);
        q.wait();
        std::vector<float> hr(size_t(m * n)), hx(size_t(m * n));
        q.memcpy(hr.data(), yr, hr.size() * sizeof(float));
        q.memcpy(hx.data(), yx, hx.size() * sizeof(float)).wait();
        double num = 0, den = 0;
        for (size_t i = 0; i < hr.size(); ++i) {
          num += double(hx[i] - hr[i]) * (hx[i] - hr[i]);
          den += double(hr[i]) * hr[i];
        }
        double emul_err = 0;
        for (int s = 0; s < 16; ++s) {
          const int64_t row = int64_t(rng() % uint32_t(m)), col = int64_t(rng() % uint32_t(n));
          const double e = emulate(x, k, *t, row, col, m > xmx::kFp16Rows);
          emul_err = std::max(emul_err, std::fabs(double(hx[size_t(row * n + col)]) - e) / std::max(1e-6, std::fabs(e)));
        }
        // Timing: quantize + matmul, 20 calls enqueued back to back and one wait, so launch and
        // wait latency are amortized as they are in a forward pass; warm, best of 3.
        double best = 1e30;
        for (int rep = 0; rep < 4; ++rep) {
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < 20; ++i) {
            xmx::quantize(q, xd, m, act);
            xmx::matmul(q, act, xw, yx);
          }
          q.wait();
          double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
          if (rep) best = std::min(best, dt);
        }
        const double tops = 2.0 * double(m) * n * k / best / 1e12;
        const double gbps = double(t->nbytes) / best / 1e9;
        std::printf("%-22s %5lld %10.2e %10.2e %10.3f %9.1f %9.0f\n", name, (long long)m, std::sqrt(num / den), emul_err,
                    best * 1e3, tops, gbps);
        w.begin_object().kv("weight", name).kv("rows", m).kv("rel_vs_fp32", std::sqrt(num / den)).kv("rel_vs_emulation", emul_err)
            .kv("ms", best * 1e3).kv("tops", tops).kv("weight_gbps", gbps).end_object();
      }
      sycl::free(xd, q);
      sycl::free(yr, q);
      sycl::free(yx, q);
      sycl::free(raw, q);
      xmx::free_activations(q, act);
      xmx::free_weight(q, xw);
    }
    // The engine's shapes (src/engine/engine.cpp): matrices that read the same input are stacked
    // into one, gate and up repacked with interleave, through matmul_geglu and matmul as the
    // engine calls them.
    for (const auto& sc : std::vector<std::tuple<const char*, std::vector<const char*>, bool>>{
             {"blk.0.ffn_gate_up", {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"}, true},
             {"blk.0.attn_qkv", {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight"}, false}}) {
      const char* label = std::get<0>(sc);
      const std::vector<const char*>& names = std::get<1>(sc);
      const bool geglu = std::get<2>(sc);
      const gguf::Tensor* t0 = f.tensor(names[0]);
      if (!t0) throw std::runtime_error(std::string("no tensor ") + names[0]);
      const int64_t k = int64_t(t0->ne[0]);
      int64_t n = 0;
      uint64_t staged_bytes = 0;
      for (const char* name : names) {
        const gguf::Tensor* t = f.tensor(name);
        if (!t) throw std::runtime_error(std::string("no tensor ") + name);
        if (t->ne[0] != (uint64_t)k) throw std::runtime_error(std::string("stacked widths differ for ") + name);
        n += int64_t(t->ne[1]);
        staged_bytes += t->nbytes;
      }
      auto* staging = sycl::malloc_device<uint8_t>(staged_bytes, q);
      {
        uint64_t at = 0;
        for (const char* name : names) {
          const gguf::Tensor* t = f.tensor(name);
          q.memcpy(staging + at, t->data, t->nbytes).wait();
          at += t->nbytes;
        }
      }
      ref::Weight raw{gguf::Type::Q4_0, k, n, staging, uint64_t(t0->nbytes) / uint64_t(t0->ne[1])};
      xmx::Weight xw = xmx::repack_q4_0(q, raw, geglu);
      const int64_t max_m = 2048, nn = geglu ? n / 2 : n;
      xmx::Activations act = xmx::alloc_activations(q, max_m, k);
      xmx::Activations act_out;
      if (geglu) act_out = xmx::alloc_activations(q, max_m, nn);
      auto* xd = sycl::malloc_device<float>(size_t(max_m * k), q);
      auto* yr = sycl::malloc_device<float>(size_t(max_m * n), q);  // ref of the stacked product
      auto* yref = geglu ? sycl::malloc_device<float>(size_t(max_m * nn), q) : yr;
      auto* yx = sycl::malloc_device<float>(size_t(max_m * nn), q);
      for (int64_t m : {1, 2, 5, 8, 16, 32, 64, 256, 1024, 2048}) {
        std::vector<float> x(size_t(m * k));
        for (auto& v : x) v = nd(rng);
        for (int64_t r = 0; r < m; ++r)
          for (int64_t c = 0; c < k; c += 487) x[size_t(r * k + c)] *= 30;
        q.memcpy(xd, x.data(), x.size() * sizeof(float)).wait();
        ref::matmul(q, raw, xd, m, yr);
        if (geglu)
          // gelu_tanh(gate) * up, row by row: the gate and up halves sit in the same row of yr.
          q.parallel_for(sycl::range<2>(size_t(m), size_t(nn)), [=](sycl::id<2> id) {
            const float g = yr[id[0] * n + id[1]], u = yr[id[0] * n + nn + id[1]];
            const float k0 = 0.7978845608028654f, k1 = 0.044715f;
            yref[id[0] * nn + id[1]] = 0.5f * g * (1.0f + sycl::tanh(k0 * (g + k1 * g * g * g))) * u;
          });
        xmx::quantize(q, xd, m, act);
        if (geglu) xmx::matmul_geglu(q, act, xw, act_out, yr);
        else xmx::matmul(q, act, xw, yx);
        q.wait();
        if (geglu) {
          // Read the GeGLU result back to fp32 rows: fp16 tiles, or int8 tiles times their scale.
          if (act_out.fp16)
            q.parallel_for(sycl::range<2>(size_t(m), size_t(nn)), [=](sycl::id<2> id) {
              yx[id[0] * nn + id[1]] = float(act_out.h[xmx::act16_index(id[0], id[1], nn / 16)]);
            });
          else
            q.parallel_for(sycl::range<2>(size_t(m), size_t(nn)), [=](sycl::id<2> id) {
              yx[id[0] * nn + id[1]] = float(act_out.q[xmx::act_index(id[0], id[1], nn / 32)]) *
                                       act_out.s[xmx::act_scale_index(id[0], id[1], nn / 32)];
            });
          q.wait();
        }
        std::vector<float> hr(size_t(m * nn)), hx(size_t(m * nn));
        q.memcpy(hr.data(), yref, hr.size() * sizeof(float));
        q.memcpy(hx.data(), yx, hx.size() * sizeof(float)).wait();
        double num = 0, den = 0;
        for (size_t i = 0; i < hr.size(); ++i) {
          num += double(hx[i] - hr[i]) * (hx[i] - hr[i]);
          den += double(hr[i]) * hr[i];
        }
        // Timing: quantize + the matmul call the engine makes, 20 back to back, warm, best of 3.
        double best = 1e30;
        for (int rep = 0; rep < 4; ++rep) {
          auto t0 = std::chrono::steady_clock::now();
          for (int i = 0; i < 20; ++i) {
            xmx::quantize(q, xd, m, act);
            if (geglu) xmx::matmul_geglu(q, act, xw, act_out, yr);
            else xmx::matmul(q, act, xw, yx);
          }
          q.wait();
          double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
          if (rep) best = std::min(best, dt);
        }
        const double tops = 2.0 * double(m) * n * k / best / 1e12;
        const double gbps = double(staged_bytes) / best / 1e9;
        std::printf("%-22s %5lld %10.2e %10s %10.3f %9.1f %9.0f\n", label, (long long)m, std::sqrt(num / den), "-",
                    best * 1e3, tops, gbps);
        w.begin_object().kv("weight", label).kv("rows", m).kv("rel_vs_fp32", std::sqrt(num / den)).kv("rel_vs_emulation", 0.0)
            .kv("ms", best * 1e3).kv("tops", tops).kv("weight_gbps", gbps).end_object();
      }
      sycl::free(xd, q);
      sycl::free(yr, q);
      if (geglu) sycl::free(yref, q);
      sycl::free(yx, q);
      sycl::free(staging, q);
      xmx::free_activations(q, act);
      if (geglu) xmx::free_activations(q, act_out);
      xmx::free_weight(q, xw);
    }
    w.end_array().end_object();
    if (!json_path.empty()) std::ofstream(json_path) << w.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-kernel-test: %s\n", e.what());
    return 1;
  }
  return 0;
}
