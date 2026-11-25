// eklow-inkling-fast-test: the engine's Inkling kernels (src/kernels/inkling_fast.*) against the reference ones
// (src/kernels/ref_inkling.*) and a host fp64 product, on random inputs at the shapes the engine uses.
//   - route: the same experts in the same order (ties put in on purpose: logits and biases from a few values), weights to
//     1e-5;
//   - depth_mix: 1 to 6 states, 1 or 3 queries, rows on both paths (a work-group a row up to 64, a sub-group a row above);
//   - matmul_f32: the work-group-a-row path (few outputs, long rows: the router, the sinks, the dynamic taps, the
//     controllers' down projection) and the sub-group path.
// Exit status 1 on any difference past its tolerance.
#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "kernels/inkling_fast.hpp"
#include "kernels/ref_inkling.hpp"
#include "rt/device.hpp"

using namespace eklow;
namespace ik = eklow::inkling;

namespace {

int failures = 0;

void report(const char* what, bool ok, const char* detail) {
  std::printf("%s %s: %s\n", ok ? "ok  " : "FAIL", what, detail);
  if (!ok) ++failures;
}

template <class T>
T* dev(sycl::queue& q, const std::vector<T>& h) {
  T* p = sycl::malloc_device<T>(h.size(), q);
  q.memcpy(p, h.data(), h.size() * sizeof(T)).wait();
  return p;
}

template <class T>
std::vector<T> host(sycl::queue& q, const T* p, size_t n) {
  std::vector<T> h(n);
  q.memcpy(h.data(), p, n * sizeof(T)).wait();
  return h;
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(double(a[i]) - double(b[i])));
    den = std::max(den, std::abs(double(b[i])));
  }
  return den > 0 ? num / den : num;
}

void test_route(sycl::queue& q, std::mt19937& rng) {
  const int64_t T = 300, NE = 128, NU = 8;
  for (int ties : {0, 1}) {
    std::vector<float> lg(static_cast<size_t>(T * NE)), bias(static_cast<size_t>(NE));
    std::uniform_real_distribution<float> u(-4, 4);
    std::uniform_int_distribution<int> few(-3, 3);
    for (auto& v : lg) v = ties ? float(few(rng)) * 0.5f : u(rng);
    for (auto& v : bias) v = ties ? float(few(rng)) * 0.01f : u(rng) * 0.05f;
    float* dl = dev(q, lg);
    float* db = dev(q, bias);
    int32_t* ia = sycl::malloc_device<int32_t>(size_t(T * NU), q);
    int32_t* ib = sycl::malloc_device<int32_t>(size_t(T * NU), q);
    float* wa = sycl::malloc_device<float>(size_t(T * NU), q);
    float* wb = sycl::malloc_device<float>(size_t(T * NU), q);
    for (int with_bias : {1, 0}) {
      ref::inkling::route(q, dl, with_bias ? db : nullptr, T, NE, NU, 8.0f, ia, wa);
      ik::route(q, dl, with_bias ? db : nullptr, T, NE, NU, 8.0f, ib, wb);
      q.wait();
      const auto ha = host(q, ia, size_t(T * NU)), hb = host(q, ib, size_t(T * NU));
      const auto fa = host(q, wa, size_t(T * NU)), fb = host(q, wb, size_t(T * NU));
      int64_t diff = 0;
      for (size_t i = 0; i < ha.size(); ++i) diff += ha[i] != hb[i];
      char d[160];
      std::snprintf(d, sizeof d, "%s logits, %s bias: %lld of %lld picks differ, weights max rel %.2e", ties ? "tied" : "random",
                    with_bias ? "with" : "no", (long long)diff, (long long)ha.size(), max_rel(fb, fa));
      report("route", diff == 0 && max_rel(fb, fa) < 1e-5, d);
    }
    for (void* p : {(void*)dl, (void*)db, (void*)ia, (void*)ib, (void*)wa, (void*)wb}) sycl::free(p, q);
  }
}

void test_depth_mix(sycl::queue& q, std::mt19937& rng) {
  const int64_t E = 2304;
  std::normal_distribution<float> nd(0, 1);
  for (int64_t rows : {int64_t(1), int64_t(7), int64_t(64), int64_t(200)})
    for (int n_states : {1, 3, 6})
      for (int n_query : {1, 3}) {
        std::vector<float*> sp;
        for (int s = 0; s < n_states; ++s) {
          std::vector<float> h(size_t(rows * E));
          for (auto& v : h) v = nd(rng) * (1.0f + float(s));
          sp.push_back(dev(q, h));
        }
        std::vector<float> qh(size_t(n_query * E));
        for (auto& v : qh) v = nd(rng) * 0.2f;
        float* dq = dev(q, qh);
        std::vector<const float*> ptrs(sp.begin(), sp.end());
        const float** table = sycl::malloc_device<const float*>(ptrs.size(), q);
        q.memcpy(table, ptrs.data(), ptrs.size() * sizeof(float*)).wait();
        float* oa = sycl::malloc_device<float>(size_t(n_query * rows * E), q);
        float* ob = sycl::malloc_device<float>(size_t(n_query * rows * E), q);
        ref::inkling::depth_mix(q, table, n_states, dq, n_query, rows, E, 1e-6f, oa);
        ik::States st;
        for (float* p : sp) st.p[st.n++] = p;
        float* scratch = sycl::malloc_device<float>(size_t(ik::kMixFewRows * ik::kMixScratch), q);
        for (bool few : {false, true}) {
          if (few && rows > ik::kMixFewRows) continue;
          ik::depth_mix(q, st, dq, n_query, rows, E, 1e-6f, ob, few ? scratch : nullptr);
          q.wait();
          const double e = max_rel(host(q, ob, size_t(n_query * rows * E)), host(q, oa, size_t(n_query * rows * E)));
          char d[120];
          std::snprintf(d, sizeof d, "%lld rows, %d states, %d queries%s: max rel %.2e", (long long)rows, n_states, n_query,
                        few ? ", split rows" : "", e);
          report("depth_mix", e < 1e-5, d);
        }
        sycl::free(scratch, q);
        for (float* p : sp) sycl::free(p, q);
        for (void* p : {(void*)dq, (void*)table, (void*)oa, (void*)ob}) sycl::free(p, q);
      }
}

void test_matmul(sycl::queue& q, std::mt19937& rng) {
  std::normal_distribution<float> nd(0, 1);
  struct Shape {
    int64_t n, k, T;
  };
  for (const Shape& sh : {Shape{24, 2304, 1}, Shape{32, 2304, 1}, Shape{64, 4608, 1}, Shape{128, 2304, 1}, Shape{128, 2304, 16},
                          Shape{64, 4608, 16}, Shape{2304, 64, 1}, Shape{2304, 64, 16}, Shape{24, 2304, 1024}}) {
    std::vector<float> w(static_cast<size_t>(sh.n * sh.k)), x(static_cast<size_t>(sh.T * sh.k));
    for (auto& v : w) v = nd(rng) * 0.05f;
    for (auto& v : x) v = nd(rng);
    float* dw = dev(q, w);
    float* dx = dev(q, x);
    float* dout = sycl::malloc_device<float>(size_t(sh.T * sh.n), q);
    ik::matmul_f32(q, dw, sh.n, sh.k, dx, sh.k, sh.T, dout, sh.n);
    q.wait();
    const auto got = host(q, dout, size_t(sh.T * sh.n));
    std::vector<float> want(got.size());
    for (int64_t t = 0; t < sh.T; ++t)
      for (int64_t j = 0; j < sh.n; ++j) {
        double a = 0;
        for (int64_t i = 0; i < sh.k; ++i) a += double(w[size_t(j * sh.k + i)]) * double(x[size_t(t * sh.k + i)]);
        want[size_t(t * sh.n + j)] = float(a);
      }
    const double e = max_rel(got, want);
    char d[120];
    std::snprintf(d, sizeof d, "n %lld, k %lld, %lld tokens: max rel %.2e", (long long)sh.n, (long long)sh.k, (long long)sh.T, e);
    report("matmul_f32", e < 1e-5, d);
    for (void* p : {(void*)dw, (void*)dx, (void*)dout}) sycl::free(p, q);
  }
}

void test_control(sycl::queue& q, std::mt19937& rng) {
  const int64_t E = 2304, R = 64;
  std::normal_distribution<float> nd(0, 1);
  for (int64_t T : {int64_t(1), int64_t(3), int64_t(16)}) {
    std::vector<float> cur(static_cast<size_t>(T * E)), msg(cur.size()), down(static_cast<size_t>(R * 2 * E)), gain(static_cast<size_t>(E * R)),
        forget(gain.size());
    for (auto& v : cur) v = nd(rng) * 3.0f;
    for (auto& v : msg) v = nd(rng) * 0.5f;
    for (auto& v : down) v = nd(rng) * 0.02f;
    for (auto& v : gain) v = nd(rng) * 0.1f;
    for (auto& v : forget) v = nd(rng) * 0.1f;
    float *dc = dev(q, cur), *dm = dev(q, msg), *dd = dev(q, down), *dg = dev(q, gain), *df = dev(q, forget);
    float* ci = sycl::malloc_device<float>(size_t(T * 2 * E), q);
    float* cd = sycl::malloc_device<float>(size_t(T * R), q);
    float* out = sycl::malloc_device<float>(size_t(T * E), q);
    // the reference: ref control_input, the projections on the host in fp64, ref control_mix
    ref::inkling::control_input(q, dc, dm, T, E, 1e-6f, ci);
    q.wait();
    const auto hci = host(q, ci, size_t(T * 2 * E));
    std::vector<float> hcd(static_cast<size_t>(T * R)), hg(static_cast<size_t>(T * E)), hf(hg.size());
    for (int64_t t = 0; t < T; ++t) {
      for (int64_t r = 0; r < R; ++r) {
        double a = 0;
        for (int64_t i = 0; i < 2 * E; ++i) a += double(down[size_t(r * 2 * E + i)]) * double(hci[size_t(t * 2 * E + i)]);
        hcd[size_t(t * R + r)] = float(a);
      }
      for (int64_t e = 0; e < E; ++e) {
        double g = 0, f = 0;
        for (int64_t r = 0; r < R; ++r)
          g += double(gain[size_t(e * R + r)]) * hcd[size_t(t * R + r)], f += double(forget[size_t(e * R + r)]) * hcd[size_t(t * R + r)];
        hg[size_t(t * E + e)] = float(g), hf[size_t(t * E + e)] = float(f);
      }
    }
    float *dhg = dev(q, hg), *dhf = dev(q, hf);
    float* want = sycl::malloc_device<float>(size_t(T * E), q);
    ref::inkling::control_mix(q, dc, dm, dhg, dhf, T * E, want);
    ik::control(q, dc, dm, dd, dg, df, T, E, R, 1e-6f, cd, out);
    q.wait();
    const double ecd = max_rel(host(q, cd, size_t(T * R)), hcd), e = max_rel(host(q, out, size_t(T * E)), host(q, want, size_t(T * E)));
    char d[120];
    std::snprintf(d, sizeof d, "%lld tokens: down max rel %.2e, out max rel %.2e", (long long)T, ecd, e);
    report("control", ecd < 1e-5 && e < 1e-5, d);
    for (void* p : {(void*)dc, (void*)dm, (void*)dd, (void*)dg, (void*)df, (void*)ci, (void*)cd, (void*)out, (void*)dhg, (void*)dhf, (void*)want})
      sycl::free(p, q);
  }
}

}  // namespace

int main() {
  try {
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(20261124);
    test_route(q, rng);
    test_depth_mix(q, rng);
    test_matmul(q, rng);
    test_control(q, rng);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-inkling-fast-test: %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", failures ? "FAILED" : "all ok");
  return failures ? 1 : 0;
}
