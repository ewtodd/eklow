// eklow-cpu-gemv-test: checks and times the CPU Q4_0 expert GEMVs (src/cpu/q4_gemv.cpp) on
// layer 0 of the 26B's GGUF.
//
//   eklow-cpu-gemv-test --model GGUF
//
// Correctness: experts 0, 17 and 127, gate_up (k 2816, n 1408) and down (k 704, n 2816) with
// random inputs, against (1) an fp32 dot of the dequantized weights (the int8 activation
// quantization shows up, ~1e-2) and (2) a scalar emulation of the same integer arithmetic
// (~1e-6).
// Speed: one expert's gate_up + GeGLU + quantize + down, best of 20 after a warm-up, at
// 1/4/8/12/20 threads, then 8 different experts one after another at the best thread count.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <random>
#include <string>
#include <vector>

#include "cpu/q4_gemv.hpp"
#include "model/gguf.hpp"

using namespace eklow;

namespace {

int64_t meta_int(const gguf::File& f, const std::string& key) { return f.at(key).as_int(); }

float f16(const uint8_t* p) {
  uint16_t raw;
  std::memcpy(&raw, p, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(raw)));
}

// One Q4_0 row as fp32: value j = (nibble - 8) * d, byte i low nibble = value i, high = i + 16.
void dequant_row(const uint8_t* row, int64_t k, float* out) {
  for (int64_t b = 0; b < k / 32; ++b) {
    const uint8_t* p = row + b * 18;
    const float d = f16(p);
    for (int j = 0; j < 32; ++j) {
      const int nib = j < 16 ? (p[2 + j] & 0x0f) : (p[2 + j - 16] >> 4);
      out[b * 32 + j] = float(nib - 8) * d;
    }
  }
}

// The kernel's arithmetic in plain scalars: the integer dot is exact, the fp32 scaling per
// block.
void emul_row(const uint8_t* row, int64_t k, const cpu::Q8Blocks& x, float* out) {
  float acc = 0.0f;
  for (int64_t b = 0; b < k / 32; ++b) {
    const uint8_t* p = row + b * 18;
    int32_t dot = 0;
    for (int i = 0; i < 16; ++i) {
      dot += (p[2 + i] & 0x0f) * int(x.q[size_t(b) * 32 + i]);
      dot += (p[2 + i] >> 4) * int(x.q[size_t(b) * 32 + i + 16]);
    }
    acc += float(dot - 8 * x.sum[size_t(b)]) * f16(p) * x.s[size_t(b)];
  }
  *out = acc;
}

float rel_l2(const float* a, const float* b, int64_t n) {
  double da = 0, db = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double d = double(a[i]) - double(b[i]);
    da += d * d;
    db += double(b[i]) * double(b[i]);
  }
  return float(std::sqrt(da / db));
}

// GeGLU, the same tanh gelu as ref::gelu_mul.
void geglu(const float* gu, int64_t fe, float* out) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  for (int64_t j = 0; j < fe; ++j) {
    const float g = gu[j];
    out[j] = 0.5f * g * (1.0f + std::tanh(k0 * (g + k1 * g * g * g))) * gu[fe + j];
  }
}

using Clock = std::chrono::steady_clock;

// Best of `runs` measurements of `fn`, with `warm` untimed runs first; microseconds.
double best_of(std::function<void()> fn, int warm, int runs) {
  for (int i = 0; i < warm; ++i) fn();
  double best = 1e30;
  for (int i = 0; i < runs; ++i) {
    const auto t0 = Clock::now();
    fn();
    best = std::min(best, std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
  }
  return best;
}

int usage() {
  std::fprintf(stderr, "usage: eklow-cpu-gemv-test --model GGUF\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  for (int i = 1; i + 1 < argc; ++i)
    if (!std::strcmp(argv[i], "--model")) model = argv[i + 1];
  if (model.empty()) return usage();

  try {
    const gguf::File f = gguf::File::open(model);
    const int64_t E = meta_int(f, "gemma4.embedding_length");
    const int64_t Fe = meta_int(f, "gemma4.expert_feed_forward_length");
    const int64_t n_expert = meta_int(f, "gemma4.expert_count");
    const gguf::Tensor* t_gu = f.tensor("blk.0.ffn_gate_up_exps.weight");
    const gguf::Tensor* t_dn = f.tensor("blk.0.ffn_down_exps.weight");
    if (!t_gu || t_gu->type != gguf::Type::Q4_0 || t_gu->ne.size() != 3 || int64_t(t_gu->ne[0]) != E ||
        int64_t(t_gu->ne[1]) != 2 * Fe || int64_t(t_gu->ne[2]) != n_expert ||
        !t_dn || t_dn->type != gguf::Type::Q4_0 || t_dn->ne.size() != 3 || int64_t(t_dn->ne[0]) != Fe ||
        int64_t(t_dn->ne[1]) != E || int64_t(t_dn->ne[2]) != n_expert) {
      std::fprintf(stderr, "not a Gemma 4 MoE GGUF (layer 0 expert shapes)\n");
      return 1;
    }
    const uint8_t* gu = t_gu->data;  // expert e: rows [e*2*Fe, (e+1)*2*Fe), row E/32*18 bytes
    const uint8_t* dn = t_dn->data;  // expert e: rows [e*E, (e+1)*E), row Fe/32*18 bytes
    const int64_t gu_row = E / 32 * 18, dn_row = Fe / 32 * 18;
    const int64_t expert_bytes = 2 * Fe * gu_row + E * dn_row;

    std::printf("model: %s\nn_embd %lld, n_ff_exp %lld, n_expert %lld, layer-0 expert %lld B "
                "(gate_up %lld x %lld B, down %lld x %lld B)\n\n",
                f.path().c_str(), (long long)E, (long long)Fe, (long long)n_expert, (long long)expert_bytes,
                (long long)(2 * Fe), (long long)gu_row, (long long)E, (long long)dn_row);

    std::mt19937 rng(20261026);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    // Correctness: experts 0, 17 and 127, both matrices.
    std::printf("correctness (rel L2, random inputs):\n");
    std::printf("matrix     expert   n      vs fp32    vs emul\n");
    const int64_t experts[] = {0, 17, 127};
    const struct {
      const char* name;
      const uint8_t* rows;
      int64_t k, n, row;
    } mats[] = {
        {"gate_up", gu, E, 2 * Fe, gu_row},
        {"down", dn, Fe, E, dn_row},
    };
    std::vector<float> xrow, y, yref, yemul, wrow;
    for (const auto& m : mats) {
      for (int64_t e : experts) {
        xrow.assign(size_t(m.k), 0.0f);
        for (auto& v : xrow) v = noise(rng);
        const cpu::Q8Blocks xb = cpu::quantize(xrow.data(), m.k);
        y.resize(size_t(m.n));
        yref.resize(size_t(m.n));
        yemul.resize(size_t(m.n));
        cpu::gemv_q4(m.rows + e * m.n * m.row, m.k, m.n, xb, y.data(), 1);
        wrow.resize(size_t(m.k));
        for (int64_t r = 0; r < m.n; ++r) {
          dequant_row(m.rows + (e * m.n + r) * m.row, m.k, wrow.data());
          float acc = 0.0f;
          for (int64_t j = 0; j < m.k; ++j) acc += wrow[j] * xrow[j];
          yref[size_t(r)] = acc;
          emul_row(m.rows + (e * m.n + r) * m.row, m.k, xb, &yemul[size_t(r)]);
        }
        std::printf("%-10s %6lld %6lld  %.2e   %.2e\n", m.name, (long long)e, (long long)m.n,
                    rel_l2(y.data(), yref.data(), m.n), rel_l2(y.data(), yemul.data(), m.n));
      }
    }

    // Speed: one expert's gate_up + GeGLU + quantize + down.
    const int64_t Fe2 = 2 * Fe;
    std::vector<float> xin(size_t(E), 0.0f), gu_out(size_t(Fe2), 0.0f), g_out(size_t(Fe), 0.0f),
        dout(size_t(E), 0.0f);
    for (auto& v : xin) v = noise(rng);
    const cpu::Q8Blocks xin_q = cpu::quantize(xin.data(), E);
    auto run_expert = [&](int64_t e, int threads) {
      cpu::gemv_q4(gu + e * Fe2 * gu_row, E, Fe2, xin_q, gu_out.data(), threads);
      geglu(gu_out.data(), Fe, g_out.data());
      const cpu::Q8Blocks g_q = cpu::quantize(g_out.data(), Fe);
      cpu::gemv_q4(dn + e * E * dn_row, Fe, E, g_q, dout.data(), threads);
    };

    std::printf("\nspeed, expert 0, gate_up + GeGLU + quantize + down (%lld B, best of 20):\n",
                (long long)expert_bytes);
    std::printf("threads     us     GB/s   (fresh std::threads per call)\n");
    int best_threads = 1;
    double best_us = 1e30;
    for (int threads : {1, 4, 8, 12, 20}) {
      const double us = best_of([&] { run_expert(0, threads); }, 3, 20);
      if (us < best_us) {
        best_us = us;
        best_threads = threads;
      }
      std::printf("%7d %9.1f %8.1f\n", threads, us, expert_bytes / us * 1e-3);
    }

    // The same sweep on a persistent pool: std::thread creation per call costs tens of
    // microseconds, a large part of the GEMV itself.
    std::printf("workers     us     GB/s   (persistent pool)\n");
    int best_workers = 1;
    double best_pool_us = 1e30;
    for (int nworkers : {1, 4, 8, 12, 20}) {
      cpu::Pool pool(nworkers);
      auto run_pool = [&] {
        cpu::gemv_q4(gu, E, Fe2, xin_q, gu_out.data(), pool);
        geglu(gu_out.data(), Fe, g_out.data());
        const cpu::Q8Blocks g_q = cpu::quantize(g_out.data(), Fe);
        cpu::gemv_q4(dn, Fe, E, g_q, dout.data(), pool);
      };
      const double us = best_of(run_pool, 3, 20);
      if (us < best_pool_us) {
        best_pool_us = us;
        best_workers = nworkers;
      }
      std::printf("%7d %9.1f %8.1f\n", nworkers, us, expert_bytes / us * 1e-3);
    }

    // 8 different experts one after another (3.35 MB each, more than fits in any cache level):
    // the case the engine would actually hit for cold experts.
    const int64_t cold[] = {0, 16, 32, 48, 64, 80, 96, 112};
    auto run_cold = [&] {
      for (int64_t e : cold) run_expert(e, best_threads);
    };
    const double us8 = best_of(run_cold, 3, 20);
    std::printf("\n8 experts one after another (fresh threads %d): %8.1f us/expert  %6.1f GB/s\n",
                best_threads, us8 / 8, 8.0 * expert_bytes / us8 * 1e-3);
    {
      cpu::Pool pool(best_workers);
      auto run_cold_pool = [&] {
        for (int64_t e : cold) {
          cpu::gemv_q4(gu + e * Fe2 * gu_row, E, Fe2, xin_q, gu_out.data(), pool);
          geglu(gu_out.data(), Fe, g_out.data());
          const cpu::Q8Blocks g_q = cpu::quantize(g_out.data(), Fe);
          cpu::gemv_q4(dn + e * E * dn_row, Fe, E, g_q, dout.data(), pool);
        }
      };
      const double us8p = best_of(run_cold_pool, 3, 20);
      std::printf("8 experts one after another (pool %d):      %8.1f us/expert  %6.1f GB/s\n", best_workers,
                  us8p / 8, 8.0 * expert_bytes / us8p * 1e-3);
    }
    // 64 experts (214 MB, well past the 24 MB L3) in a row, one pass per timing: what comes from
    // DRAM. Swept over the pool sizes.
    std::printf("\n64 experts in a row, cold from DRAM (persistent pool, per expert):\n");
    std::printf("workers     us     GB/s\n");
    for (int nworkers : {1, 2, 4, 6, 8, 12, 20}) {
      cpu::Pool pool(nworkers);
      auto run64 = [&] {
        for (int64_t e = 0; e < n_expert; e += 2) {
          cpu::gemv_q4(gu + e * Fe2 * gu_row, E, Fe2, xin_q, gu_out.data(), pool);
          geglu(gu_out.data(), Fe, g_out.data());
          const cpu::Q8Blocks g_q = cpu::quantize(g_out.data(), Fe);
          cpu::gemv_q4(dn + e * E * dn_row, Fe, E, g_q, dout.data(), pool);
        }
      };
      const double us = best_of(run64, 1, 5) / 64;
      std::printf("%7d %9.1f %8.1f\n", nworkers, us, expert_bytes / us * 1e-3);
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-cpu-gemv-test: %s\n", e.what());
    return 1;
  }
}
