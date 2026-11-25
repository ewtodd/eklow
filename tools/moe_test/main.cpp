// eklow-moe-test: Inkling 2 Mini's experts on the card (kernels/moe.hpp) against the reference ops (ref::matmul_experts,
// ref::inkling::situ, ::moe_combine, fp32), on a real GGUF's layers, and their speed.
//
//   eklow-moe-test --gguf FILE [--layers 0,6,17,19,20,23] [--tokens 1,2,3,8,32,64,512,2048] [--reps N] [--seed N]
//                  [--no-check] [--no-time]
//
// Inputs are made like the model's: random rows, RMS-normed and multiplied by the layer's ffn_norm.weight, routed by the
// layer's own router (ref::matmul of ffn_gate_inp, ref::inkling::route with exp_probs_b, scale 8). Check: relative L2 of
// each token's output row against the reference (median and max over tokens), gate <= 2e-3 median and <= 1e-2 max; and a
// rebased copy of the layer (moe::rebase) giving the same output bit for bit. Speed: time per layer at each token count,
// the bytes of the distinct experts the step reads and their rate, tokens/s.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>
#include <thread>
#include <fstream>
#include <sys/mman.h>
#include <memory>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/moe.hpp"
#include "rt/copy_engine.hpp"

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "kernels/ref.hpp"
#include "kernels/ref_inkling.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

std::vector<int64_t> list(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, ',');)
    if (!p.empty()) out.push_back(std::atoll(p.c_str()));
  return out;
}

template <class T>
T* to_dev(sycl::queue& q, const T* h, size_t n) {
  T* d = sycl::malloc_device<T>(n, q);
  if (!d) throw std::runtime_error("out of device memory");
  q.memcpy(d, h, n * sizeof(T)).wait();
  return d;
}

const gguf::Tensor& need(const gguf::File& f, const std::string& n) {
  const gguf::Tensor* t = f.tensor(n);
  if (!t) throw std::runtime_error("missing tensor " + n);
  return *t;
}

struct Cfg {
  int64_t E = 2304, F = 1152, NE = 128, NU = 8;
  float beta = 4, lb = 25, w_scale = 8, eps = 1e-6f;
};

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  std::vector<int64_t> layers = {0, 6, 17, 19, 20, 23}, tokens = {1, 2, 3, 8, 32, 64, 512, 2048};
  int reps = 20;
  uint32_t seed = 1;
  bool check = true, timing = true;
  std::vector<int> hot_pct = {100, 75, 50, 25};
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--gguf") path = next();
    else if (a == "--layers") layers = list(next());
    else if (a == "--tokens") tokens = list(next());
    else if (a == "--reps") reps = std::atoi(next().c_str());
    else if (a == "--seed") seed = uint32_t(std::atoi(next().c_str()));
    else if (a == "--no-check") check = false;
    else if (a == "--no-time") timing = false;
    else if (a == "--hot") {
      hot_pct.clear();
      for (int64_t v : list(next())) hot_pct.push_back(int(v));
    }
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "usage: eklow-moe-test --gguf FILE [--layers L,..] [--tokens N,..] [--reps N] [--no-check] [--no-time]\n");
    return 2;
  }
  try {
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    gguf::File f = gguf::File::open(path);
    Cfg c;
    const int64_t max_tok = *std::max_element(tokens.begin(), tokens.end());
    moe::Scratch* scratch = moe::reserve(q, max_tok, c.NU, c.E, c.F);
    std::printf("scratch %.1f MiB (slab %lld tokens)\n", double(moe::scratch_bytes(*scratch)) / 1048576.0, (long long)moe::scratch_slab());
    bool all_ok = true;
    for (int64_t L : layers) {
      const std::string b = "blk." + std::to_string(L) + ".";
      const gguf::Tensor &tg = need(f, b + "ffn_gate_up_exps.weight"), &td = need(f, b + "ffn_down_exps.weight");
      const gguf::Tensor &tr = need(f, b + "ffn_gate_inp.weight"), &tn = need(f, b + "ffn_norm.weight");
      const gguf::Tensor* tb = f.tensor(b + "exp_probs_b.bias");
      const gguf::Tensor* sg = f.tensor(b + "ffn_gate_up_exps.scale");
      const gguf::Tensor* sd = f.tensor(b + "ffn_down_exps.scale");
      const uint64_t gstride = tg.nbytes / uint64_t(c.NE), dstride = td.nbytes / uint64_t(c.NE);
      const ref::Weight graw{tg.type, c.E, 2 * c.F, tg.data, gstride / uint64_t(2 * c.F)};
      const ref::Weight draw{td.type, c.F, c.E, td.data, dstride / uint64_t(c.E)};
      const auto t0 = std::chrono::steady_clock::now();
      moe::Layer layer = moe::upload(q, graw, gstride, sg ? reinterpret_cast<const float*>(sg->data) : nullptr, draw, dstride,
                                     sd ? reinterpret_cast<const float*>(sd->data) : nullptr, c.NE);
      const double up_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("\nlayer %lld: gate_up %s, down %s, %.1f MiB on the card (uploaded in %.1f s)\n", (long long)L,
                  gguf::type_name(tg.type), gguf::type_name(td.type), double(layer.bytes) / 1048576.0, up_s);

      // the reference's operands on the card
      const uint8_t* rg = to_dev(q, tg.data, tg.nbytes);
      const uint8_t* rd = to_dev(q, td.data, td.nbytes);
      const uint8_t* rr = to_dev(q, tr.data, tr.nbytes);
      const float* norm = to_dev(q, reinterpret_cast<const float*>(tn.data), size_t(c.E));
      const float* bias = tb ? to_dev(q, reinterpret_cast<const float*>(tb->data), size_t(c.NE)) : nullptr;
      const float* gs = sg ? to_dev(q, reinterpret_cast<const float*>(sg->data), size_t(c.NE)) : nullptr;
      const float* ds = sd ? to_dev(q, reinterpret_cast<const float*>(sd->data), size_t(c.NE)) : nullptr;
      const ref::Weight rgw{tg.type, c.E, 2 * c.F, rg, graw.row_bytes}, rdw{td.type, c.F, c.E, rd, draw.row_bytes},
          rrw{tr.type, c.E, c.NE, rr, tr.nbytes / uint64_t(c.NE)};

      const int64_t T = max_tok;
      std::mt19937 rng(seed + uint32_t(L));
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<float> hx(size_t(T * c.E));
      for (auto& v : hx) v = nd(rng);
      float* x0 = to_dev(q, hx.data(), hx.size());
      float* x = sycl::malloc_device<float>(size_t(T * c.E), q);
      ref::rmsnorm(q, x0, norm, T, c.E, c.eps, x);
      float* logits = sycl::malloc_device<float>(size_t(T * c.NE), q);
      ref::matmul(q, rrw, x, T, logits);
      int32_t* ids = sycl::malloc_device<int32_t>(size_t(T * c.NU), q);
      float* wts = sycl::malloc_device<float>(size_t(T * c.NU), q);
      ref::inkling::route(q, logits, bias, T, c.NE, c.NU, c.w_scale, ids, wts);
      float* out = sycl::malloc_device<float>(size_t(T * c.E), q);
      q.wait();
      std::vector<int32_t> hids(size_t(T * c.NU));
      q.memcpy(hids.data(), ids, hids.size() * 4).wait();
      // EKLOW_MOE_TEST_ROUTE: "even" spreads the pairs over the experts evenly (token t's slot j to expert
      // (8 t + j) mod n_expert, so every expert gets the same count), "few" puts them on experts 0-7 only.
      if (const char* rt = std::getenv("EKLOW_MOE_TEST_ROUTE")) {
        const std::string r = rt;
        for (int64_t t = 0; t < T; ++t)
          for (int64_t j = 0; j < c.NU; ++j)
            hids[size_t(t * c.NU + j)] = int32_t(r == "few" ? j : (t * c.NU + j) % c.NE);
        q.memcpy(ids, hids.data(), hids.size() * 4).wait();
      }

      for (int64_t n : tokens) {
        // ---- check
        if (check) {
          std::vector<float> ref_out(size_t(n * c.E));
          {
            const int64_t chunk = 256;
            float* gu = sycl::malloc_device<float>(size_t(chunk * c.NU * 2 * c.F), q);
            float* act = sycl::malloc_device<float>(size_t(chunk * c.NU * c.F), q);
            float* dn = sycl::malloc_device<float>(size_t(chunk * c.NU * c.E), q);
            float* ro = sycl::malloc_device<float>(size_t(n * c.E), q);
            for (int64_t t0c = 0; t0c < n; t0c += chunk) {
              const int64_t tc = std::min(chunk, n - t0c);
              ref::matmul_experts(q, rgw, gstride, x + t0c * c.E, false, ids + t0c * c.NU, tc, c.NU, gs, gu);
              ref::inkling::situ(q, gu, tc * c.NU, c.F, c.beta, c.lb, act);
              ref::matmul_experts(q, rdw, dstride, act, true, ids + t0c * c.NU, tc, c.NU, ds, dn);
              ref::inkling::moe_combine(q, dn, wts + t0c * c.NU, tc, c.NU, c.E, ro + t0c * c.E);
              q.wait();
            }
            q.memcpy(ref_out.data(), ro, ref_out.size() * 4).wait();
            sycl::free(gu, q);
            sycl::free(act, q);
            sycl::free(dn, q);
            sycl::free(ro, q);
          }
          moe::ffn(q, layer, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out);
          std::vector<float> got(size_t(n * c.E));
          q.memcpy(got.data(), out, got.size() * 4).wait();
          std::vector<double> err(size_t(n), 0.0);
          for (int64_t t = 0; t < n; ++t) {
            double d2 = 0, r2 = 0;
            for (int64_t j = 0; j < c.E; ++j) {
              const double r = ref_out[size_t(t * c.E + j)], g = got[size_t(t * c.E + j)];
              d2 += (g - r) * (g - r);
              r2 += r * r;
            }
            err[size_t(t)] = std::sqrt(d2 / std::max(r2, 1e-30));
          }
          std::vector<double> s = err;
          std::sort(s.begin(), s.end());
          const double med = s[s.size() / 2], mx = s.back();
          const bool ok = std::isfinite(mx) && med <= 2e-3 && mx <= 1e-2;
          all_ok &= ok;
          std::printf("  check %5lld tokens: rel L2 median %.2e max %.2e  %s\n", (long long)n, med, mx, ok ? "ok" : "FAIL");

          // the same on a rebased copy: bit for bit
          if (n == tokens.front() || n == 8 || n == max_tok) {
            std::vector<uint8_t> host(layer.bytes);
            q.memcpy(host.data(), layer.base, layer.bytes).wait();
            uint8_t* nb = sycl::malloc_device<uint8_t>(layer.bytes, q);
            q.memcpy(nb, host.data(), layer.bytes).wait();
            moe::Layer copy = layer;
            moe::rebase(q, copy, nb);
            float* out2 = sycl::malloc_device<float>(size_t(n * c.E), q);
            moe::ffn(q, copy, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out2);
            std::vector<float> got2(size_t(n * c.E));
            q.memcpy(got2.data(), out2, got2.size() * 4).wait();
            const bool same = std::memcmp(got.data(), got2.data(), got.size() * 4) == 0;
            all_ok &= same;
            std::printf("  rebase %4lld tokens: %s\n", (long long)n, same ? "identical" : "DIFFERS");
            sycl::free(nb, q);
            sycl::free(out2, q);
          }
        }
        // ---- speed
        if (timing) {
          moe::ffn(q, layer, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out);
          q.wait();
          moe::profile_print(false);
          // EKLOW_MOE_TEST_COPY=MB: host-to-card copies of that size kept running on the copy engine during the timing, to
          // measure what they cost the kernels (DECISIONS 2026-11-25, "the cold copies")
          std::unique_ptr<rt::CopyEngine> ce;
          std::thread pacer;
          std::atomic<bool> pacer_stop{false};
          uint8_t *ch = nullptr, *cd = nullptr;
          if (const char* cm = std::getenv("EKLOW_MOE_TEST_COPY")) {
            const size_t mb = size_t(std::atoll(cm)) << 20;
            ce = std::make_unique<rt::CopyEngine>(q);
            // EKLOW_MOE_TEST_COPY_SRC: "dev" card to card, "huge" host memory on 2 MB transparent huge pages imported into
            // Level Zero, else pinned host memory from the runtime
            const std::string src = std::getenv("EKLOW_MOE_TEST_COPY_SRC") ? std::getenv("EKLOW_MOE_TEST_COPY_SRC") : "";
            if (src == "dev") {
              ch = sycl::malloc_device<uint8_t>(mb, q);
              q.memset(ch, 1, mb).wait();
            } else if (src == "huge") {
              void* p = mmap(nullptr, mb + (2u << 20), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
              ch = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(p) + (2u << 20) - 1) & ~uintptr_t((2u << 20) - 1));
              madvise(ch, mb, MADV_HUGEPAGE);
              std::memset(ch, 1, mb);
              using Import = ze_result_t (*)(ze_driver_handle_t, void*, size_t);
              ze_driver_handle_t drv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device().get_platform());
              void* fn = nullptr;
              zeDriverGetExtensionFunctionAddress(drv, "zexDriverImportExternalPointer", &fn);
              if (!fn || reinterpret_cast<Import>(fn)(drv, ch, mb) != ZE_RESULT_SUCCESS) throw std::runtime_error("import failed");
              std::ifstream mi("/proc/meminfo");
              for (std::string l; std::getline(mi, l);)
                if (l.rfind("AnonHugePages", 0) == 0) std::printf("    %s\n", l.c_str());
            } else {
              ch = sycl::malloc_host<uint8_t>(mb, q);
              std::memset(ch, 1, mb);
            }
            cd = sycl::malloc_device<uint8_t>(mb, q);
            if (!std::getenv("EKLOW_MOE_TEST_COPY_RATE")) {
              for (int i = 0; i < 40; ++i) ce->copy(cd, ch, mb);
            } else {
              // EKLOW_MOE_TEST_COPY_RATE=GB/s: 8 MB copies appended at that rate by a host thread while the timing runs
              const double rate = std::atof(std::getenv("EKLOW_MOE_TEST_COPY_RATE"));
              pacer_stop = false;
              pacer = std::thread([&ce, ch, cd, mb, rate, &pacer_stop] {
                const size_t chunk = size_t(8) << 20;
                const auto t0 = std::chrono::steady_clock::now();
                size_t sent = 0;
                while (!pacer_stop.load()) {
                  const double due = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * rate * 1e9;
                  while (double(sent) < due) {
                    ce->copy(cd + (sent % mb) / chunk * chunk % mb, ch + (sent % mb) / chunk * chunk % mb, chunk);
                    sent += chunk;
                  }
                  std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
              });
            }
          }
          const auto s0 = std::chrono::steady_clock::now();
          const int r = n >= 512 ? std::max(2, reps / 4) : reps;
          for (int i = 0; i < r; ++i) moe::ffn(q, layer, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out);
          q.wait();
          const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count() / r;
          std::vector<char> used(size_t(c.NE), 0);
          for (int64_t i = 0; i < n * c.NU; ++i) used[size_t(hids[size_t(i)])] = 1;
          const int64_t distinct = std::count(used.begin(), used.end(), 1);
          const double bytes = double(distinct) * double(layer.gate_up.expert_bytes + layer.down.expert_bytes);
          std::printf("  time  %5lld tokens: %8.3f ms  %5lld experts %7.1f MB read, %6.1f GB/s  %9.0f tokens/s\n", (long long)n, ms,
                      (long long)distinct, bytes / 1e6, bytes / (ms * 1e-3) / 1e9, double(n) / (ms * 1e-3));
          if (std::getenv("EKLOW_MOE_PROFILE")) moe::profile_print();
          if (pacer.joinable()) {
            pacer_stop = true;
            pacer.join();
          }
          if (ce) {
            const auto c0 = std::chrono::steady_clock::now();
            ce->wait(ce->fence());
            std::printf("    copies drained %.1f ms after the timing\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count());
            if (!std::getenv("EKLOW_MOE_TEST_COPY_SRC") || std::string(std::getenv("EKLOW_MOE_TEST_COPY_SRC")) != "huge") sycl::free(ch, q);
            sycl::free(cd, q);
          }
        }
      }
      // ---- residency: only the busiest experts on the card, the rest's weighted sum from the reference ops
      if (!hot_pct.empty()) {
        std::vector<int> cnt(size_t(c.NE), 0), order(size_t(c.NE));
        for (int32_t id : hids) ++cnt[size_t(id)];
        for (int i = 0; i < c.NE; ++i) order[size_t(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return cnt[size_t(a)] > cnt[size_t(b)]; });
        std::vector<moe::Layer> hl;
        std::vector<std::vector<char>> is_hot;
        for (int pct : hot_pct) {
          std::vector<bool> hot(size_t(c.NE), false);
          std::vector<char> hc(size_t(c.NE), 0);
          for (int i = 0; i < c.NE * pct / 100; ++i) hot[size_t(order[size_t(i)])] = true, hc[size_t(order[size_t(i)])] = 1;
          hl.push_back(moe::upload(q, graw, gstride, sg ? reinterpret_cast<const float*>(sg->data) : nullptr, draw, dstride,
                                   sd ? reinterpret_cast<const float*>(sd->data) : nullptr, c.NE, &hot));
          is_hot.push_back(hc);
          std::printf("  hot %3d%%: %lld experts, %.1f MiB on the card\n", pct, (long long)hl.back().n_hot, double(hl.back().bytes) / 1048576.0);
        }
        std::vector<float> hw(size_t(T * c.NU));
        q.memcpy(hw.data(), wts, hw.size() * 4).wait();
        for (int64_t n : tokens) {
          const size_t nf = hot_pct.size();
          std::vector<float*> wm(nf), cold(nf);
          for (size_t f = 0; f < nf; ++f) {
            std::vector<float> m(size_t(n * c.NU));
            for (int64_t i = 0; i < n * c.NU; ++i) m[size_t(i)] = is_hot[f][size_t(hids[size_t(i)])] ? 0.0f : hw[size_t(i)];
            wm[f] = to_dev(q, m.data(), m.size());
            cold[f] = sycl::malloc_device<float>(size_t(n * c.E), q);
          }
          std::vector<float> ref_out(size_t(n * c.E));
          {
            const int64_t chunk = 256;
            float* gu = sycl::malloc_device<float>(size_t(chunk * c.NU * 2 * c.F), q);
            float* act = sycl::malloc_device<float>(size_t(chunk * c.NU * c.F), q);
            float* dn = sycl::malloc_device<float>(size_t(chunk * c.NU * c.E), q);
            float* ro = sycl::malloc_device<float>(size_t(n * c.E), q);
            for (int64_t t0c = 0; t0c < n; t0c += chunk) {
              const int64_t tc = std::min(chunk, n - t0c);
              ref::matmul_experts(q, rgw, gstride, x + t0c * c.E, false, ids + t0c * c.NU, tc, c.NU, gs, gu);
              ref::inkling::situ(q, gu, tc * c.NU, c.F, c.beta, c.lb, act);
              ref::matmul_experts(q, rdw, dstride, act, true, ids + t0c * c.NU, tc, c.NU, ds, dn);
              ref::inkling::moe_combine(q, dn, wts + t0c * c.NU, tc, c.NU, c.E, ro + t0c * c.E);
              for (size_t f = 0; f < nf; ++f) ref::inkling::moe_combine(q, dn, wm[f] + t0c * c.NU, tc, c.NU, c.E, cold[f] + t0c * c.E);
              q.wait();
            }
            q.memcpy(ref_out.data(), ro, ref_out.size() * 4).wait();
            sycl::free(gu, q);
            sycl::free(act, q);
            sycl::free(dn, q);
            sycl::free(ro, q);
          }
          for (size_t f = 0; f < nf; ++f) {
            const moe::Layer& L2 = hl[f];
            const float* cr = hot_pct[f] == 100 ? nullptr : cold[f];
            if (check) {
              moe::ffn(q, L2, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out, cr);
              std::vector<float> got(size_t(n * c.E));
              q.memcpy(got.data(), out, got.size() * 4).wait();
              std::vector<double> err(size_t(n), 0.0);
              for (int64_t t = 0; t < n; ++t) {
                double d2 = 0, r2 = 0;
                for (int64_t j = 0; j < c.E; ++j) {
                  const double r = ref_out[size_t(t * c.E + j)], g = got[size_t(t * c.E + j)];
                  d2 += (g - r) * (g - r);
                  r2 += r * r;
                }
                err[size_t(t)] = std::sqrt(d2 / std::max(r2, 1e-30));
              }
              std::sort(err.begin(), err.end());
              const bool ok = std::isfinite(err.back()) && err[err.size() / 2] <= 2e-3 && err.back() <= 1e-2;
              all_ok &= ok;
              std::printf("  hot %3d%% check %5lld tokens: rel L2 median %.2e max %.2e  %s\n", hot_pct[f], (long long)n, err[err.size() / 2], err.back(), ok ? "ok" : "FAIL");
            }
            if (timing) {
              moe::ffn(q, L2, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out, cr);
              q.wait();
              const auto s0 = std::chrono::steady_clock::now();
              const int r = n >= 512 ? std::max(2, reps / 4) : reps;
              for (int i = 0; i < r; ++i) moe::ffn(q, L2, x, n, ids, wts, c.NU, c.beta, c.lb, *scratch, out, cr);
              q.wait();
              const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count() / r;
              int64_t pairs_hot = 0;
              for (int64_t i = 0; i < n * c.NU; ++i) pairs_hot += is_hot[f][size_t(hids[size_t(i)])];
              std::printf("  hot %3d%% time  %5lld tokens: %8.3f ms  (%.0f%% of pairs on the card)\n", hot_pct[f], (long long)n, ms,
                          100.0 * double(pairs_hot) / double(n * c.NU));
            }
          }
          for (size_t f = 0; f < nf; ++f) {
            sycl::free(wm[f], q);
            sycl::free(cold[f], q);
          }
        }
        for (auto& l2 : hl) moe::free_layer(q, l2);
      }
      for (const void* p : {(const void*)rg, (const void*)rd, (const void*)rr, (const void*)norm, (const void*)bias, (const void*)gs, (const void*)ds})
        if (p) sycl::free(const_cast<void*>(p), q);
      sycl::free(x0, q);
      sycl::free(x, q);
      sycl::free(logits, q);
      sycl::free(ids, q);
      sycl::free(wts, q);
      sycl::free(out, q);
      moe::free_layer(q, layer);
    }
    moe::free_scratch(q, scratch);
    std::printf("\n%s\n", all_ok ? "ALL OK" : "FAILED");
    return all_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
