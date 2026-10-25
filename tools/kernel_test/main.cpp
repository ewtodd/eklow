// eklow-kernel-test: the fast kernels against the reference kernels and an exact CPU emulation,
// on real Gemma 4 weights.
//
//   eklow-kernel-test --model GGUF [--json FILE]
//
// For each weight and row count: the XMX matmul against ref::matmul (fp32; the difference is the
// int8 activation quantization, the same trade llama.cpp makes), against a CPU emulation of the
// quantized arithmetic at sampled outputs (should agree to fp32 rounding), and its speed.
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

float half_to_float(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(uint16_t(p[0] | p[1] << 8))); }

// The quantized product at (row, col) on the CPU, in double: int8 activations from the same
// quantization rule, the Q4_0 weights from the file.
double emulate(const std::vector<float>& x, int64_t k, const gguf::Tensor& w, int64_t row, int64_t col) {
  const uint8_t* wrow = w.data + uint64_t(col) * (w.nbytes / w.ne[1]);
  double total = 0;
  for (int64_t b = 0; b < k / 32; ++b) {
    const float* xs = x.data() + row * k + b * 32;
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xs[j]));
    const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
    const uint8_t* blk = wrow + b * 18;
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

}  // namespace

int main(int argc, char** argv) {
  std::string model, json_path;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--model") && i + 1 < argc) model = argv[++i];
    else if (!std::strcmp(argv[i], "--json") && i + 1 < argc) json_path = argv[++i];
    else {
      std::fprintf(stderr, "usage: eklow-kernel-test --model GGUF [--json FILE]\n");
      return 2;
    }
  }
  try {
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
      for (int64_t m : {1, 5, 8, 16, 32, 64, 256, 2048}) {
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
          const double e = emulate(x, k, *t, row, col);
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
    w.end_array().end_object();
    if (!json_path.empty()) std::ofstream(json_path) << w.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-kernel-test: %s\n", e.what());
    return 1;
  }
  return 0;
}
