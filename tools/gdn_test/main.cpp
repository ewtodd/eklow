// eklow-gdn-test: the Gated DeltaNet kernels (kernels/gdn.hpp) against a CPU emulation in double that follows llama.cpp's
// op order (build_layer_attn_linear / build_delta_net_autoregressive), the speculative rollback bit for bit, and their
// speed.
//
//   eklow-gdn-test [--only accuracy|edge|rollback|time|snapshot] [--quick] [--layers N] [--reps N]
//
// accuracy: slots with sequences of different lengths (the longest 4,200 tokens) stepped together as the engine would, in
//   prefill chunks, verify-sized blocks and single decode tokens; every step's scan output o and activation rows y are
//   compared with the emulation, and the state after the last token (and the conv state, which must be exact).
// edge: the same with g near 0 and very negative, beta near 0 and 1, huge and zero rows.
// rollback: a verify block of B tokens then `accept` a of them gives the bit-identical outputs, state and conv state
//   as running only those a (every a for several B; chains of random blocks; partial commits; flushes with n = 0).
// time: per kernel and in all, for the step shapes of TASK.md, one layer and 30.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kernels/gdn.hpp"
#include "rt/device.hpp"

using namespace eklow;
using namespace eklow::gdn;

namespace {

constexpr int kD = kHead;
constexpr double kEps = 1e-6;

struct Model {
  std::vector<float> conv_w, ssm_a, dt, norm_w;
  float *d_conv_w = nullptr, *d_ssm_a = nullptr, *d_dt = nullptr, *d_norm_w = nullptr;
  Weights dev() const { return Weights{d_conv_w, d_ssm_a, d_dt, d_norm_w, float(kEps)}; }
};

template <class T>
T* upload(sycl::queue& q, const std::vector<T>& v) {
  T* p = sycl::malloc_device<T>(std::max<size_t>(v.size(), 1), q);
  q.memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  return p;
}

// Realistic magnitudes unless `edge`: ssm_a = -exp(A_log) with A_log in [0, 2.8], dt_bias around the softplus knee.
Model make_model(sycl::queue& q, std::mt19937& rng, bool edge) {
  Model m;
  std::normal_distribution<float> nd(0, 1);
  std::uniform_real_distribution<float> ud(0, 1);
  m.conv_w.resize(size_t(kChannels) * 4);
  for (auto& v : m.conv_w) v = 0.5f * nd(rng);
  m.ssm_a.resize(kValueHeads), m.dt.resize(kValueHeads), m.norm_w.resize(kD);
  const float edge_a[8] = {-1e-6f, -1e-3f, -0.1f, -1.0f, -4.0f, -16.0f, -100.0f, -1000.0f};
  for (int h = 0; h < kValueHeads; ++h) {
    m.ssm_a[h] = edge ? edge_a[h % 8] : -std::exp(2.8f * ud(rng));
    m.dt[h] = edge ? 3.0f * nd(rng) : 0.5f + 0.5f * nd(rng);
  }
  for (auto& v : m.norm_w) v = 1.0f + 0.2f * nd(rng);
  m.d_conv_w = upload(q, m.conv_w), m.d_ssm_a = upload(q, m.ssm_a), m.d_dt = upload(q, m.dt), m.d_norm_w = upload(q, m.norm_w);
  return m;
}

// Host copies of input rows.
struct Rows {
  std::vector<float> x, z, a, b;
  void resize(size_t n) { x.resize(n * kChannels), z.resize(n * kInner), a.resize(n * kValueHeads), b.resize(n * kValueHeads); }
  size_t n() const { return a.size() / kValueHeads; }
  // rows [from, from + n) of `src` to rows [at, at + n) of this
  void copy_from(const Rows& src, size_t from, size_t n, size_t at) {
    std::copy_n(src.x.begin() + std::ptrdiff_t(from * kChannels), n * kChannels, x.begin() + std::ptrdiff_t(at * kChannels));
    std::copy_n(src.z.begin() + std::ptrdiff_t(from * kInner), n * kInner, z.begin() + std::ptrdiff_t(at * kInner));
    std::copy_n(src.a.begin() + std::ptrdiff_t(from * kValueHeads), n * kValueHeads, a.begin() + std::ptrdiff_t(at * kValueHeads));
    std::copy_n(src.b.begin() + std::ptrdiff_t(from * kValueHeads), n * kValueHeads, b.begin() + std::ptrdiff_t(at * kValueHeads));
  }
};

void fill_rows(std::mt19937& rng, Rows& r, size_t from, size_t n, bool edge) {
  std::normal_distribution<float> nd(0, 1);
  std::uniform_real_distribution<float> ud(0, 1);
  for (size_t i = from * kChannels; i < (from + n) * kChannels; ++i) r.x[i] = 1.5f * nd(rng);
  for (size_t i = from * kInner; i < (from + n) * kInner; ++i) r.z[i] = 2.0f * nd(rng);
  for (size_t i = from * kValueHeads; i < (from + n) * kValueHeads; ++i) {
    r.a[i] = edge ? 4.0f * nd(rng) : nd(rng);
    r.b[i] = edge ? (ud(rng) < 0.3f ? (ud(rng) < 0.5f ? -30.0f : 30.0f) : 3.0f * nd(rng)) : nd(rng);
  }
  if (edge)  // huge rows now and then (conv output far from unit scale), and an all-zero row
    for (size_t t = from; t < from + n; ++t) {
      const float u = ud(rng);
      if (u < 0.02f) std::fill_n(r.x.begin() + std::ptrdiff_t(t * kChannels), kChannels, 0.0f);
      else if (u < 0.05f)
        for (int c = 0; c < kChannels; ++c) r.x[t * kChannels + size_t(c)] *= 30.0f;
    }
}

// The emulation of one slot's layer, in double.
struct RefSlot {
  std::vector<double> S;    // [32][128 v][128 k]
  std::vector<float> conv;  // [3][8192]
  RefSlot() : S(size_t(kValueHeads) * kD * kD, 0.0), conv(3 * kChannels, 0.0f) {}
};

// Runs n tokens (rows r0.. of `r`) through the emulation; o and y receive [n][4096] each.
void ref_run(const Model& m, RefSlot& s, const Rows& r, size_t r0, size_t n, std::vector<double>* o_out, std::vector<double>* y_out) {
  // Prepare all tokens, then walk each head's recurrence in a thread.
  std::vector<double> qkv(n * kChannels), eg(n * kValueHeads), beta(n * kValueHeads);
  auto xrow = [&](int64_t t, int c) -> float { return t >= 0 ? r.x[(r0 + size_t(t)) * kChannels + size_t(c)] : s.conv[size_t(t + 3) * kChannels + size_t(c)]; };
  for (size_t t = 0; t < n; ++t) {
    double* row = &qkv[t * kChannels];
    for (int c = 0; c < kChannels; ++c) {
      double sum = 0;
      for (int i = 0; i < 4; ++i) sum += double(xrow(int64_t(t) - 3 + i, c)) * double(m.conv_w[size_t(c) * 4 + size_t(i)]);
      row[c] = sum / (1.0 + std::exp(-sum));
    }
    for (int h = 0; h < 2 * kKeyHeads; ++h) {
      double ss = 0;
      for (int i = 0; i < kD; ++i) ss += row[h * kD + i] * row[h * kD + i];
      const double inv = 1.0 / std::sqrt(ss + kEps) * (h < kKeyHeads ? 1.0 / std::sqrt(double(kD)) : 1.0);
      for (int i = 0; i < kD; ++i) row[h * kD + i] *= inv;
    }
    for (int h = 0; h < kValueHeads; ++h) {
      const double sp = double(r.a[(r0 + t) * kValueHeads + size_t(h)]) + double(m.dt[size_t(h)]);
      const double g = double(m.ssm_a[size_t(h)]) * (sp > 20 ? sp : std::log(1.0 + std::exp(sp)));
      eg[t * kValueHeads + size_t(h)] = std::exp(g);
      beta[t * kValueHeads + size_t(h)] = 1.0 / (1.0 + std::exp(-double(r.b[(r0 + t) * kValueHeads + size_t(h)])));
    }
  }
  std::vector<double> o(n * kInner);
  auto head = [&](int h0, int h1) {
    for (int h = h0; h < h1; ++h) {
      double* S = &s.S[size_t(h) * kD * kD];
      const int kh = h % kKeyHeads;
      for (size_t t = 0; t < n; ++t) {
        const double* q = &qkv[t * kChannels + size_t(kh * kD)];
        const double* k = &qkv[t * kChannels + size_t(kQk + kh * kD)];
        const double* v = &qkv[t * kChannels + size_t(2 * kQk + h * kD)];
        const double e = eg[t * kValueHeads + size_t(h)], b = beta[t * kValueHeads + size_t(h)];
        for (int c = 0; c < kD; ++c) {
          double* col = S + c * kD;
          double kv = 0;
          for (int i = 0; i < kD; ++i) kv += (col[i] *= e) * k[i];
          const double d = (v[c] - kv) * b;
          double oc = 0;
          for (int i = 0; i < kD; ++i) oc += (col[i] += k[i] * d) * q[i];
          o[t * kInner + size_t(h * kD + c)] = oc;
        }
      }
    }
  };
  const int nt = int(std::max(1u, std::min(16u, std::thread::hardware_concurrency())));
  std::vector<std::thread> th;
  for (int i = 0; i < nt; ++i) th.emplace_back(head, i * kValueHeads / nt, (i + 1) * kValueHeads / nt);
  for (auto& t : th) t.join();
  if (y_out) {
    y_out->resize(n * kInner);
    for (size_t t = 0; t < n; ++t)
      for (int h = 0; h < kValueHeads; ++h) {
        double ss = 0;
        for (int i = 0; i < kD; ++i) ss += o[t * kInner + size_t(h * kD + i)] * o[t * kInner + size_t(h * kD + i)];
        const double inv = 1.0 / std::sqrt(ss / kD + kEps);
        for (int i = 0; i < kD; ++i) {
          const double z = r.z[(r0 + t) * kInner + size_t(h * kD + i)];
          (*y_out)[t * kInner + size_t(h * kD + i)] = o[t * kInner + size_t(h * kD + i)] * inv * double(m.norm_w[size_t(i)]) * (z / (1.0 + std::exp(-z)));
        }
      }
  }
  // Conv state: the last three raw rows.
  std::vector<float> nc(3 * kChannels);
  for (int j = 0; j < 3; ++j)
    for (int c = 0; c < kChannels; ++c) nc[size_t(j * kChannels + c)] = xrow(int64_t(n) - 3 + j, c);
  s.conv = nc;
  if (o_out) *o_out = std::move(o);
}

struct Err {
  double max_abs = 0, max_rel = 0, sum_sq = 0, ref_sq = 0, ref_max = 0;
  size_t n = 0;
  void add(double got, double ref, double floor_ = 0) {
    const double e = std::fabs(got - ref);
    max_abs = std::max(max_abs, e);
    sum_sq += e * e, ref_sq += ref * ref;
    ref_max = std::max(ref_max, std::fabs(ref));
    ++n;
    max_rel = std::max(max_rel, e / std::max(std::fabs(ref), floor_));
  }
  // max_rel: error over max(|ref|, floor), the floor 1e-3 of the step's rms
  double rms() const { return n ? std::sqrt(ref_sq / double(n)) : 0; }
  double rel_err_rms() const { return std::sqrt(sum_sq / double(std::max<size_t>(n, 1))) / std::max(rms(), 1e-30); }
  void print(const char* what) {
    std::printf("  %-26s n %-10zu max abs %.3e  max rel %.3e (floor 1e-3 step rms)  rms err / rms signal %.3e  (signal rms %.3e, max %.3e)\n", what, n, max_abs,
                max_rel, rel_err_rms(), rms(), ref_max);
  }
};

// Activation rows back to floats; the quantization step per element, for the error bound.
void dequant(sycl::queue& q, const xmx::Activations& a, int64_t n, std::vector<float>& y, std::vector<float>& step) {
  y.assign(size_t(n) * kInner, 0.0f), step.assign(size_t(n) * kInner, 0.0f);
  const int64_t kb = kInner / 32;
  if (a.fp16) {
    std::vector<sycl::half> h(size_t(a.m_pad) * kInner);
    q.memcpy(h.data(), a.h, h.size() * 2).wait();
    for (int64_t t = 0; t < n; ++t)
      for (int64_t k = 0; k < kInner; ++k) {
        const float v = float(h[size_t(xmx::act16_index(t, k, kInner / 16))]);
        y[size_t(t * kInner + k)] = v;
        step[size_t(t * kInner + k)] = std::fabs(v) * 0.00049f;  // fp16: 2^-11 relative
      }
  } else {
    std::vector<int8_t> qd(size_t(a.m_pad) * kInner);
    std::vector<float> sd(size_t(a.m_pad) * size_t(kb));
    q.memcpy(qd.data(), a.q, qd.size()).wait();
    q.memcpy(sd.data(), a.s, sd.size() * 4).wait();
    for (int64_t t = 0; t < n; ++t)
      for (int64_t k = 0; k < kInner; ++k) {
        const float s = sd[size_t(xmx::act_scale_index(t, k, kb))];
        y[size_t(t * kInner + k)] = float(qd[size_t(xmx::act_index(t, k, kb))]) * s;
        step[size_t(t * kInner + k)] = 0.5f * s;
      }
  }
}

// One test bench's device state.
struct Bench {
  sycl::queue q;
  Model m;
  Cache cache;
  Scratch scr;
  xmx::Activations act;
  float *dx = nullptr, *dz = nullptr, *da = nullptr, *db = nullptr;
  int64_t cap_tokens;
  Bench(int layers, int slots, int max_pending, int64_t tokens, std::mt19937& rng, bool edge)
      : q(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}), cap_tokens(tokens) {
    m = make_model(q, rng, edge);
    cache = alloc_cache(q, layers, slots, max_pending);
    scr = alloc_scratch(q, tokens);
    act = xmx::alloc_activations(q, tokens, kInner);
    dx = sycl::malloc_device<float>(size_t(tokens) * kChannels, q);
    dz = sycl::malloc_device<float>(size_t(tokens) * kInner, q);
    da = sycl::malloc_device<float>(size_t(tokens) * kValueHeads, q);
    db = sycl::malloc_device<float>(size_t(tokens) * kValueHeads, q);
  }
  ~Bench() {
    q.wait();
    sycl::free(dx, q), sycl::free(dz, q), sycl::free(da, q), sycl::free(db, q);
    free_scratch(q, scr);
    xmx::free_activations(q, act);
    free_cache(q, cache);
  }
  Inputs inputs() const { return Inputs{dx, dz, da, db}; }
  // Device rows laid out one range after another: ranges[i] of src[i].
  void upload_gather(const std::vector<const Rows*>& src, const std::vector<std::pair<size_t, size_t>>& ranges) {
    size_t at = 0;
    for (size_t i = 0; i < src.size(); ++i) {
      const Rows& r = *src[i];
      const size_t from = ranges[i].first, n = ranges[i].second;
      if (n) {
        q.memcpy(dx + at * kChannels, &r.x[from * kChannels], n * kChannels * 4);
        q.memcpy(dz + at * kInner, &r.z[from * kInner], n * kInner * 4);
        q.memcpy(da + at * kValueHeads, &r.a[from * kValueHeads], n * kValueHeads * 4);
        q.memcpy(db + at * kValueHeads, &r.b[from * kValueHeads], n * kValueHeads * 4);
      }
      at += n;
    }
  }
  void read_state(int slot, int layer, std::vector<float>& ssm, std::vector<float>& conv) {
    ssm.resize(size_t(cache.ssm_floats_per_layer())), conv.resize(size_t(cache.conv_floats_per_layer()));
    q.memcpy(ssm.data(), cache.ssm + (int64_t(slot) * cache.n_layers + layer) * cache.ssm_floats_per_layer(), ssm.size() * 4);
    q.memcpy(conv.data(), cache.conv + (int64_t(slot) * cache.n_layers + layer) * cache.conv_floats_per_layer(), conv.size() * 4);
    q.wait();
  }
  std::vector<float> read_o(int64_t n) {
    std::vector<float> o(size_t(n) * kInner);
    if (n) q.memcpy(o.data(), scr.o, o.size() * 4).wait();
    return o;
  }
};

bool g_fail = false;
void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("  FAIL: %s\n", what);
    g_fail = true;
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// accuracy and edge

void accuracy(bool edge, bool quick) {
  std::printf("%s\n", edge ? "edge cases (g near 0 and very negative, beta near 0 and 1, huge and zero rows)" : "accuracy against the double emulation");
  std::mt19937 rng(edge ? 7 : 1);
  const int layers = 3, layer = 1;
  std::vector<int> lens = quick ? std::vector<int>{600, 300, 37, 1} : std::vector<int>{4200, 2300, 37, 1};
  if (edge) lens = quick ? std::vector<int>{800, 200} : std::vector<int>{2500, 700};
  const int S = int(lens.size());
  const int64_t max_step = 1400;
  Bench b(layers, S, 16, 2048 + max_step * S, rng, edge);
  std::vector<Rows> data{size_t(S)};
  std::vector<RefSlot> ref{size_t(S)};
  for (int s = 0; s < S; ++s) {
    data[size_t(s)].resize(size_t(lens[size_t(s)]));
    fill_rows(rng, data[size_t(s)], 0, size_t(lens[size_t(s)]), edge);
  }
  std::vector<int> done(size_t(S), 0);
  std::vector<SlotBook> book(size_t(S), SlotBook{});
  Err e_o, e_y;
  double worst_ratio_y = 0;
  struct Worst {
    int t, c;
    double got, ref, step, o, ro;
  } worst{};
  int steps = 0;
  const Weights w = b.m.dev();
  std::uniform_real_distribution<float> ud(0, 1);
  for (;;) {
    std::vector<SlotStep> sl;
    std::vector<const Rows*> src;
    std::vector<std::pair<size_t, size_t>> ranges;
    std::vector<int> who;
    int tok = 0;
    for (int s = 0; s < S; ++s) {
      const int rem = lens[size_t(s)] - done[size_t(s)];
      if (rem == 0) continue;
      const float u = ud(rng);
      int n = u < 0.4f ? 1 : u < 0.65f ? 2 + int(ud(rng) * 15) : 17 + int(ud(rng) * float(max_step - 17));
      if (s == 0 && !edge && done[0] == 0) n = 2048;  // a full-size prefill chunk first
      n = std::min(n, rem);
      sl.push_back(book[size_t(s)].next(s, tok, n));
      src.push_back(&data[size_t(s)]);
      ranges.push_back({size_t(done[size_t(s)]), size_t(n)});
      who.push_back(s);
      tok += n;
    }
    if (sl.empty()) break;
    b.upload_gather(src, ranges);
    Plan p = make_plan(b.q, sl);
    step(b.q, w, b.cache, layer, p, b.inputs(), b.scr, b.act);
    b.q.wait();
    const std::vector<float> o = b.read_o(tok);
    std::vector<float> y, stp;
    dequant(b.q, b.act, tok, y, stp);
    for (size_t i = 0; i < who.size(); ++i) {
      const int s = who[i];
      std::vector<double> ro, ry;
      ref_run(b.m, ref[size_t(s)], data[size_t(s)], ranges[i].first, ranges[i].second, &ro, &ry);
      const size_t t0 = size_t(sl[i].tok0);
      auto rms_of = [](const std::vector<double>& v) {
        double s = 0;
        for (double x : v) s += x * x;
        return std::sqrt(s / double(std::max<size_t>(v.size(), 1)));
      };
      double invh[kValueHeads];
      const double fo = 1e-3 * rms_of(ro), fy = 1e-3 * rms_of(ry);
      for (size_t t = 0; t < ranges[i].second; ++t) {
        for (int h = 0; h < kValueHeads; ++h) {
          double ss = 0;
          for (int e = 0; e < kD; ++e) ss += ro[t * kInner + size_t(h * kD + e)] * ro[t * kInner + size_t(h * kD + e)];
          invh[h] = 1.0 / std::sqrt(ss / kD + kEps);
        }
        for (size_t c = 0; c < size_t(kInner); ++c) {
          const size_t k = (t0 + t) * kInner + c;
          e_o.add(o[k], ro[t * kInner + c], fo);
          const double ref_y = ry[t * kInner + c], ey = std::fabs(double(y[k]) - ref_y);
          // The quantization's own bound (half a step, or fp16's relative step) plus fp32 noise in o and in the scale.
          const double zz = data[size_t(s)].z[(ranges[i].first + t) * kInner + c];
          // o carries an absolute fp32 noise (measured max 6e-8 here); a head whose o is near zero is amplified by its norm.
          const double amp = 3e-7 * invh[c / kD] * std::fabs(double(b.m.norm_w[c % kD])) * std::fabs(zz / (1.0 + std::exp(-zz)));
          const double ratio = ey / (double(stp[k]) + 2e-4 * std::fabs(ref_y) + 1e-7 + amp);
          if (ratio > worst_ratio_y) {
            worst_ratio_y = ratio;
            worst = {int(t0 + t), int(c), double(y[k]), ref_y, double(stp[k]), double(o[k]), ro[t * kInner + c]};
          }
          e_y.add(y[k], ref_y, fy);
        }
      }
      done[size_t(s)] += int(ranges[i].second);
    }
    free_plan(b.q, p);
    ++steps;
  }
  std::printf("  %d steps\n", steps);
  e_o.print("scan output o");
  e_y.print("activations y (dequant)");
  std::printf("  worst y: row %d col %d got %.6g ref %.6g step %.3g o %.6g ro %.6g (%s)\n", worst.t, worst.c, worst.got, worst.ref, worst.step, worst.o, worst.ro, b.act.fp16 ? "fp16" : "int8");
  std::printf("  y worst error / (quantization bound + 2e-4 |y| + o noise through the norm): %.3f\n", worst_ratio_y);
  check(worst_ratio_y <= 1.01, "y beyond the quantization bound");
  // The state after the last token of each slot.
  Err e_s;
  bool conv_exact = true, untouched = true;
  for (int s = 0; s < S; ++s) {
    std::vector<float> ssm, conv;
    b.read_state(s, layer, ssm, conv);
    for (size_t i = 0; i < ssm.size(); ++i) e_s.add(ssm[i], ref[size_t(s)].S[i], 1e-3);
    conv_exact &= std::memcmp(conv.data(), ref[size_t(s)].conv.data(), conv.size() * 4) == 0;
    std::vector<float> s0, c0;
    b.read_state(s, 0, s0, c0);  // the other layers were never touched
    for (float v : s0) untouched &= v == 0.0f;
    for (float v : c0) untouched &= v == 0.0f;
  }
  e_s.print("state after the last token");
  std::printf("  conv state exact: %s; other layers untouched: %s\n", conv_exact ? "yes" : "NO", untouched ? "yes" : "NO");
  check(conv_exact, "conv state differs from the raw rows");
  check(untouched, "a layer that was not stepped changed");
  // The bound: o is a 128-term fp32 dot product over a state that the fp32 recurrence carries (and every token's
  // fp32 rounding of it); the rms of the error against the rms of the signal is held to 1e-4 (the double emulation
  // and fp32 differ by ~1e-6..1e-5 per operation, accumulated over 4,096 tokens that decay by exp(g)).
  const double bound = 1e-4;
  std::printf("  rms error / rms signal: o %.3e, state %.3e (bound %.0e)\n", e_o.rel_err_rms(), e_s.rel_err_rms(), bound);
  check(e_o.rel_err_rms() < bound, "o beyond the fp32 bound");
  check(e_s.rel_err_rms() < bound, "state beyond the fp32 bound");
}

// ---------------------------------------------------------------------------------------------------------------------
// rollback

bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * 4); }

// Two slots fed identical rows: slot 0 runs verify blocks and accepts a prefix of each, slot 1 runs just the accepted
// tokens. Compared bit for bit: the outputs of the accepted tokens, and the state.
struct Twin {
  Bench b;
  std::vector<SlotBook> book = std::vector<SlotBook>(2, SlotBook{});
  const int layer = 1;
  explicit Twin(std::mt19937& rng) : b(3, 2, 16, 128, rng, false) {
    reset_slot(b.q, b.cache, 0), reset_slot(b.q, b.cache, 1);
  }
  struct Out {
    std::vector<float> o0, o1, y0, y1;
  };
  // Slot 0 runs rows r0 of `rows` as step s0, slot 1 rows r1 as s1.
  Out step(const Rows& rows, SlotStep s0, SlotStep s1, std::pair<size_t, size_t> r0, std::pair<size_t, size_t> r1) {
    s0.tok0 = 0;
    s1.tok0 = int32_t(r0.second);
    Plan p = make_plan(b.q, {s0, s1});
    b.upload_gather({&rows, &rows}, {r0, r1});
    gdn::step(b.q, b.m.dev(), b.cache, layer, p, b.inputs(), b.scr, b.act);
    b.q.wait();
    const int64_t n0 = int64_t(r0.second), n1 = int64_t(r1.second);
    Out out;
    const std::vector<float> o = b.read_o(n0 + n1);
    out.o0.assign(o.begin(), o.begin() + n0 * kInner);
    out.o1.assign(o.begin() + n0 * kInner, o.end());
    if (n0 + n1) {
      std::vector<float> y, st;
      dequant(b.q, b.act, n0 + n1, y, st);
      out.y0.assign(y.begin(), y.begin() + n0 * kInner);
      out.y1.assign(y.begin() + n0 * kInner, y.end());
    }
    free_plan(b.q, p);
    return out;
  }
};

void rollback(bool quick) {
  std::printf("rollback: verify block + accept a == running just those a (bitwise)\n");
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> ud(0, 1);
  int cases = 0, bad = 0;
  auto compare_state = [&](Twin& tw, const char* what) {
    std::vector<float> s0, c0, s1, c1;
    tw.b.read_state(0, tw.layer, s0, c0);
    tw.b.read_state(1, tw.layer, s1, c1);
    ++cases;
    if (!same(s0, s1) || !same(c0, c1)) {
      ++bad;
      std::printf("  FAIL: state or conv state differs: %s\n", what);
    }
  };
  // The first n rows of a against all of b.
  auto compare = [&](const std::vector<float>& a, const std::vector<float>& b2, size_t n, const char* what) {
    ++cases;
    if (a.size() < n * kInner || b2.size() != n * kInner || std::memcmp(a.data(), b2.data(), n * kInner * 4)) {
      ++bad;
      std::printf("  FAIL: outputs differ: %s\n", what);
    }
  };
  // 1. every a for several B: prefix, verify block, accept a, then (a) a flush, (b) new tokens, (c) new tokens with a partial commit.
  const std::vector<int> Bs = quick ? std::vector<int>{2, 8, 16} : std::vector<int>{2, 3, 4, 8, 15, 16};
  for (int B : Bs)
    for (int a = 1; a <= B; ++a)
      for (int variant = 0; variant < 3; ++variant) {
        Twin tw(rng);
        Rows rows;
        const int P = 5 + int(ud(rng) * 40), N3 = 1 + int(ud(rng) * 12);
        rows.resize(size_t(P + B + N3));
        fill_rows(rng, rows, 0, rows.n(), false);
        auto& bk = tw.book;
        auto o = tw.step(rows, bk[0].next(0, 0, P), bk[1].next(1, 0, P), {0, size_t(P)}, {0, size_t(P)});
        compare(o.o0, o.o1, size_t(P), "prefix");
        // verify block B on slot 0 (nothing committed); slot 1 runs the first a tokens plain
        o = tw.step(rows, bk[0].next(0, 0, B, 0, 0), bk[1].next(1, 0, a), {size_t(P), size_t(B)}, {size_t(P), size_t(a)});
        compare(o.o0, o.o1, size_t(a), "verify outputs vs plain");
        compare(o.y0, o.y1, size_t(a), "verify y vs plain");
        if (variant == 0) {
          tw.step(rows, bk[0].next(0, 0, 0, a, 0), bk[1].next(1, 0, 0), {0, 0}, {0, 0});
          compare_state(tw, "flush after verify");
        } else if (variant == 1) {
          o = tw.step(rows, bk[0].next(0, 0, N3, a), bk[1].next(1, 0, N3), {size_t(P + B), size_t(N3)}, {size_t(P + B), size_t(N3)});
          compare(o.o0, o.o1, size_t(N3), "tokens after the accepted ones");
          compare(o.y0, o.y1, size_t(N3), "y after the accepted ones");
          compare_state(tw, "accept then new tokens");
        } else {
          // N3 tokens of which only c are committed, the rest left pending; then accept those and flush
          const int c = int(ud(rng) * float(N3 + 1)) % (N3 + 1);
          o = tw.step(rows, bk[0].next(0, 0, N3, a, c), bk[1].next(1, 0, N3), {size_t(P + B), size_t(N3)}, {size_t(P + B), size_t(N3)});
          compare(o.o0, o.o1, size_t(N3), "partial commit outputs");
          tw.step(rows, bk[0].next(0, 0, 0, N3 - c, 0), bk[1].next(1, 0, 0), {0, 0}, {0, 0});
          compare_state(tw, "partial commit then flush");
        }
      }
  // 2. chains of random verify blocks whose rejected tokens are garbage, against a twin that runs the accepted ones.
  const int rounds = quick ? 40 : 200;
  for (int chain = 0; chain < (quick ? 2 : 4); ++chain) {
    Twin tw(rng);
    Rows truth;  // the accepted stream
    truth.resize(size_t(rounds) * 16 + 64);
    fill_rows(rng, truth, 0, truth.n(), false);
    size_t pos = 0;
    for (int r = 0; r < rounds; ++r) {
      const int B = 2 + int(ud(rng) * 15) % 15, a = 1 + int(ud(rng) * float(B)) % B;
      // rows 0..B-1: the block (its first a rows the true ones, the rest garbage), rows B..B+a-1: the true ones again for slot 1
      Rows blk;
      blk.resize(size_t(B + a));
      blk.copy_from(truth, pos, size_t(a), 0);
      fill_rows(rng, blk, size_t(a), size_t(B - a), false);
      blk.copy_from(truth, pos, size_t(a), size_t(B));
      auto o = tw.step(blk, tw.book[0].next(0, 0, B, -1, 0), tw.book[1].next(1, 0, a), {0, size_t(B)}, {size_t(B), size_t(a)});
      compare(o.o0, o.o1, size_t(a), "chain: accepted outputs");
      tw.book[0].pending = a;  // only a of the B pending tokens were accepted: the next step's accept
      pos += size_t(a);
    }
    tw.step(truth, tw.book[0].next(0, 0, 0, -1, 0), tw.book[1].next(1, 0, 0), {0, 0}, {0, 0});
    compare_state(tw, "chain end");
  }
  std::printf("  %d comparisons, %d differ\n", cases, bad);
  check(bad == 0, "rollback is not bit-identical");
}

// ---------------------------------------------------------------------------------------------------------------------
// time

void fill_dev(sycl::queue& q, float* p, int64_t n, uint32_t seed, float scale) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
    uint32_t h = uint32_t(id[0]) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
    uint32_t h2 = h * 747796405u + 2891336453u;
    h2 = ((h2 >> ((h2 >> 28) + 4)) ^ h2) * 277803737u;
    // sum of two uniforms: bell-ish, zero mean
    p[id[0]] = scale * (float(h >> 8) * (1.0f / 8388608.0f) + float(h2 >> 8) * (1.0f / 8388608.0f) - 1.0f);
  }).wait();
}

double ns(const sycl::event& e) {
  return double(e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>());
}

void timing(int layers, int reps) {
  std::printf("time: GPU time per step in us; one layer (the same layer repeated), then %d layers (distinct state)\n", layers);
  std::printf("  %-8s %7s %7s %7s %8s | %8s %8s %8s %9s %9s\n", "shape", "prep", "scan", "norm", "layer", "prep", "scan", "norm", "sum", "wall");
  std::mt19937 rng(5);
  struct Cfg {
    const char* name;
    std::vector<int> ns;
  };
  const std::vector<Cfg> cfgs = {{"1x1", {1}}, {"2x1", {1, 1}}, {"1x8", {8}}, {"1x16", {16}}, {"1x2048", {2048}}, {"2x2048", {2048, 2048}}};
  Bench b(layers, 2, 16, 4096, rng, false);
  fill_dev(b.q, b.dx, 4096 * int64_t(kChannels), 1, 3.0f);
  fill_dev(b.q, b.dz, 4096 * int64_t(kInner), 2, 4.0f);
  fill_dev(b.q, b.da, 4096 * int64_t(kValueHeads), 3, 2.0f);
  fill_dev(b.q, b.db, 4096 * int64_t(kValueHeads), 4, 2.0f);
  const Weights w = b.m.dev();
  for (const Cfg& c : cfgs) {
    std::vector<SlotStep> sl;
    int tok = 0;
    for (size_t s = 0; s < c.ns.size(); ++s) {
      sl.push_back(SlotStep{int32_t(s), tok, c.ns[s], 0, c.ns[s], {0, 0, 0}});
      tok += c.ns[s];
    }
    Plan p = make_plan(b.q, sl);
    auto med = [](std::vector<double>& v) {
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    std::vector<double> t1[3], t30[3], tw;
    const int r = tok >= 1000 ? std::max(3, reps / 4) : reps;
    for (int rep = 0; rep < r + 1; ++rep) {
      double a1[3] = {};
      for (int i = 0; i < 4; ++i) {  // one layer repeated
        auto e0 = prepare(b.q, w, b.cache, 0, p, b.inputs(), b.scr);
        auto e1 = scan(b.q, b.cache, 0, p, b.scr);
        auto e2 = gated_norm(b.q, w, b.cache, 0, p, b.inputs(), b.scr, b.act);
        b.q.wait();
        a1[0] = ns(e0), a1[1] = ns(e1), a1[2] = ns(e2);
      }
      std::vector<sycl::event> ev;
      auto m0 = b.q.single_task([] {});
      for (int l = 0; l < layers; ++l) {
        ev.push_back(prepare(b.q, w, b.cache, l, p, b.inputs(), b.scr));
        ev.push_back(scan(b.q, b.cache, l, p, b.scr));
        ev.push_back(gated_norm(b.q, w, b.cache, l, p, b.inputs(), b.scr, b.act));
      }
      auto m1 = b.q.single_task([] {});
      b.q.wait();
      if (rep == 0) continue;
      double a30[3] = {};
      for (int l = 0; l < layers; ++l)
        for (int k = 0; k < 3; ++k) a30[k] += ns(ev[size_t(l * 3 + k)]);
      for (int k = 0; k < 3; ++k) t1[k].push_back(a1[k]), t30[k].push_back(a30[k]);
      tw.push_back(double(m1.get_profiling_info<sycl::info::event_profiling::command_start>() - m0.get_profiling_info<sycl::info::event_profiling::command_end>()));
    }
    const double s1 = med(t1[0]) + med(t1[1]) + med(t1[2]), s30 = med(t30[0]) + med(t30[1]) + med(t30[2]);
    std::printf("  %-8s %7.1f %7.1f %7.1f %8.1f | %8.1f %8.1f %8.1f %9.1f %9.1f\n", c.name, med(t1[0]) / 1e3, med(t1[1]) / 1e3, med(t1[2]) / 1e3, s1 / 1e3,
                med(t30[0]) / 1e3, med(t30[1]) / 1e3, med(t30[2]) / 1e3, s30 / 1e3, med(tw) / 1e3);
    free_plan(b.q, p);
  }
}

// Snapshot copies of a slot's whole state.
void snapshot(int layers) {
  std::printf("snapshot: a slot's committed state (%d layers) to and from host memory\n", layers);
  std::mt19937 rng(9);
  Bench b(layers, 2, 16, 64, rng, false);
  Rows rows;
  rows.resize(40);
  fill_rows(rng, rows, 0, 40, false);
  b.upload_gather({&rows}, {{0, 40}});
  const Weights w = b.m.dev();
  Plan p = make_plan(b.q, {SlotStep{0, 0, 40, 0, 40, {0, 0, 0}}});
  for (int l = 0; l < layers; ++l) step(b.q, w, b.cache, l, p, b.inputs(), b.scr, b.act);
  b.q.wait();
  const size_t bytes = size_t(b.cache.snapshot_bytes());
  for (int pinned = 0; pinned < 2; ++pinned) {
    void* host = pinned ? sycl::malloc_host(bytes, b.q) : std::malloc(bytes);
    double d2h = 1e30, h2d = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      snapshot_to_host(b.q, b.cache, 0, host).wait();
      auto t1 = std::chrono::steady_clock::now();
      restore_from_host(b.q, b.cache, 1, host).wait();
      auto t2 = std::chrono::steady_clock::now();
      d2h = std::min(d2h, std::chrono::duration<double, std::milli>(t1 - t0).count());
      h2d = std::min(h2d, std::chrono::duration<double, std::milli>(t2 - t1).count());
    }
    bool ok = true;  // slot 1 now holds slot 0's state, layer by layer
    for (int l = 0; l < layers; l += std::max(1, layers / 4)) {
      std::vector<float> s0, c0, s1, c1;
      b.read_state(0, l, s0, c0), b.read_state(1, l, s1, c1);
      ok &= same(s0, s1) && same(c0, c1);
    }
    std::printf("  %s host memory, %.1f MiB: to host %.2f ms (%.1f GB/s), to card %.2f ms (%.1f GB/s), round trip %s\n", pinned ? "pinned  " : "pageable",
                double(bytes) / 1048576.0, d2h, double(bytes) / d2h / 1e6, h2d, double(bytes) / h2d / 1e6, ok ? "exact" : "DIFFERS");
    check(ok, "snapshot round trip");
    if (pinned) sycl::free(host, b.q);
    else std::free(host);
  }
  free_plan(b.q, p);
}

}  // namespace

int main(int argc, char** argv) {
  std::string only;
  bool quick = false;
  int layers = 30, reps = 20;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--quick") quick = true;
    else if (a == "--layers" && i + 1 < argc) layers = std::atoi(argv[++i]);
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: eklow-gdn-test [--only accuracy|edge|rollback|time|snapshot] [--quick] [--layers N] [--reps N]\n");
      return 2;
    }
  }
  try {
    if (only.empty() || only == "accuracy") accuracy(false, quick);
    if (only.empty() || only == "edge") accuracy(true, quick);
    if (only.empty() || only == "rollback") rollback(quick);
    if (only.empty() || only == "snapshot") snapshot(layers);
    if (only.empty() || only == "time") timing(layers, reps);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", g_fail ? "FAILED" : "all ok");
  return g_fail ? 1 : 0;
}
