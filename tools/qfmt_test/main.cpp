// eklow-qfmt-test: the weight formats' kernels (xmx.hpp, xmx_fmt.hpp) on the card, checked and timed.
//
//   eklow-qfmt-test [--formats Q8_0,Q4_K,Q5_K,Q6_K,IQ4_XS,Q4_0] [--gguf Q4_K=PATH ...] [--check] [--bench]
//                   [--gemm] [--rows 1,2,8,16,32] [--json FILE]
//
// --check: for each format, tensors of the given GGUFs (real ones: --gguf FORMAT=PATH, each tensor in the format's
//   type) and random valid blocks at Bolt's shapes: the decode matmul against (1) an fp32 dot of ggml's own
//   dequantization with the fp32 activations (the int8 activation quantization shows: ~1e-2) and (2) an exact CPU
//   emulation of the quantized arithmetic (~1e-6); the SwiGLU and GeGLU fusions likewise.
// --bench: GB/s of weight read per format, shape and row count (decode), against the 399 GB/s the card reads;
//   each call reads a different copy of the matrix (a pool larger than the 16 MB L2).
// --gemm: the fp16 GEMM's TFLOPS at 2,048 and 4,096 rows (check and bench both).
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "kernels/fast.hpp"
#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"
#include "refs.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

struct FormatInfo {
  const char* name;
  gguf::Type type;
  xmx::WeightFormat format;
};
const FormatInfo kFormats[] = {
    {"Q4_0", gguf::Type::Q4_0, xmx::WeightFormat::Q4_0},    {"Q8_0", gguf::Type::Q8_0, xmx::WeightFormat::Q8_0I},
    {"Q4_K", gguf::Type::Q4_K, xmx::WeightFormat::Q4_K},    {"Q5_K", gguf::Type::Q5_K, xmx::WeightFormat::Q5_K},
    {"Q6_K", gguf::Type::Q6_K, xmx::WeightFormat::Q6_K},    {"IQ4_XS", gguf::Type::IQ4_XS, xmx::WeightFormat::IQ4_XS},
};

// Bolt's matrices (Gemma 4 E4B, hidden 2560): K, N.
struct Shape {
  const char* name;
  int64_t k, n;
};
const Shape kShapes[] = {
    {"ffn gate|up", 2560, 20480},       {"ffn_down", 10240, 2560},          {"attn_q (sliding)", 2560, 2048},
    {"attn_k / attn_v (sliding)", 2560, 512}, {"attn_q (global)", 2560, 4096}, {"attn_k / attn_v (global)", 2560, 1024},
    {"attn_output (global)", 4096, 2560}, {"head", 2560, 262144},
};

std::vector<int64_t> parse_list(const std::string& s) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stoll(t));
  return v;
}

std::mt19937 g_rng(3);

// A matrix on the card: raw ggml bytes and the repacked weight. Q4_0 goes through repack_q4_0.
struct Mat {
  std::vector<uint8_t> host;  // the raw rows (empty when taken straight from a mapping: see `raw`)
  const uint8_t* raw = nullptr;
  int64_t k = 0, n = 0;
  uint64_t rb = 0;
  xmx::Weight w;
  uint64_t bytes = 0;  // repacked bytes (tiles + records)
};

xmx::Weight repack_any(sycl::queue& q, const FormatInfo& f, const ref::Weight& rw, bool interleave) {
  if (f.format == xmx::WeightFormat::Q4_0) return xmx::repack_q4_0(q, rw, interleave);
  return xmx::repack_format(q, rw, f.format, interleave);
}
uint64_t weight_bytes(const FormatInfo& f, int64_t k, int64_t n) {
  if (f.format == xmx::WeightFormat::Q4_0) return xmx::weight_tile_bytes(k, n) + xmx::weight_scale_bytes(k, n);
  return xmx::weight_tile_bytes(f.format, k, n) + xmx::weight_meta_bytes(f.format, k, n);
}

Mat make_mat(sycl::queue& q, const FormatInfo& f, int64_t n, int64_t k, bool interleave, const uint8_t* given = nullptr) {
  Mat m;
  m.k = k;
  m.n = n;
  m.rb = refs::row_bytes(f.type, k);
  if (given) {
    m.raw = given;
  } else {
    m.host = refs::random_rows(f.type, n, k, g_rng);
    m.raw = m.host.data();
  }
  auto* dev = sycl::malloc_device<uint8_t>(size_t(m.rb * uint64_t(n)), q);
  q.memcpy(dev, m.raw, size_t(m.rb * uint64_t(n))).wait();
  m.w = repack_any(q, f, ref::Weight{f.type, k, n, dev, m.rb}, interleave);
  m.bytes = weight_bytes(f, k, n);
  sycl::free(dev, q);
  return m;
}

// Copies of a repacked weight (a pool larger than L2, so that every call reads from VRAM).
std::vector<xmx::Weight> make_pool(sycl::queue& q, const FormatInfo& f, const xmx::Weight& w, uint64_t bytes, int64_t target_bytes) {
  const int64_t copies = std::clamp<int64_t>(target_bytes / int64_t(bytes) + 1, 1, 64);
  const uint64_t tb = f.format == xmx::WeightFormat::Q4_0 ? xmx::weight_tile_bytes(w.k, w.n) : xmx::weight_tile_bytes(f.format, w.k, w.n);
  const uint64_t sb = bytes - tb;
  std::vector<xmx::Weight> pool{w};
  for (int64_t c = 1; c < copies; ++c) {
    xmx::Weight cw = w;
    cw.tiles = sycl::malloc_device<uint8_t>(size_t(tb), q);
    cw.scales = reinterpret_cast<sycl::half*>(sycl::malloc_device<uint8_t>(size_t(sb), q));
    if (!cw.tiles || !cw.scales) break;
    q.memcpy(cw.tiles, w.tiles, size_t(tb));
    q.memcpy(cw.scales, w.scales, size_t(sb));
    pool.push_back(cw);
  }
  q.wait();
  return pool;
}

template <class F>
double best_time(sycl::queue& q, int calls, F&& fn) {
  double best = 1e30;
  for (int rep = 0; rep < 5; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < calls; ++i) fn(i);
    q.wait();
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / calls;
    if (rep) best = std::min(best, dt);
  }
  return best;
}

std::vector<float> random_x(int64_t m, int64_t k) {
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> x(static_cast<size_t>(m * k));
  for (auto& v : x) v = nd(g_rng);
  for (int64_t r = 0; r < m; ++r)
    for (int64_t c = 0; c < k; c += 487) x[size_t(r * k + c)] *= 30;  // a few large channels, as hidden states have
  return x;
}

struct Stat {
  double fp32 = 0, emul = 0;
  double eps = 0;  // the worst emulation error in units of fp32 epsilon x sum |terms| (what float accumulation alone can cost)
};

// out[m][n] vs the references on sampled columns (and at most 48 sampled rows): fp32 dot of ggml's dequantization with the
// fp32 activations, and the exact emulation of the arithmetic the form does: int8 activations (up to 32 rows) or the fp16
// GEMM's (fp16 activations, the dequantization's fp16 weights, fp32 accumulation).
Stat compare(const FormatInfo& f, const Mat& mat, const std::vector<float>& x, int64_t m, const std::vector<float>& y, int cols,
             const std::vector<refs::QRow>* dev_q = nullptr) {
  const int64_t k = mat.k, n = mat.n;
  const bool fp16 = m > xmx::kFp16Rows;
  std::vector<int64_t> pick, rpick;
  for (int i = 0; i < cols; ++i) pick.push_back(int64_t(g_rng() % uint32_t(n)));
  for (int64_t r = 0; r < std::min<int64_t>(m, 48); ++r) rpick.push_back(m <= 48 ? r : int64_t(g_rng() % uint32_t(m)));
  std::vector<std::vector<float>> deq(pick.size(), std::vector<float>(static_cast<size_t>(k))), w16 = deq;
  for (size_t i = 0; i < pick.size(); ++i) {
    refs::dequant_row(f.type, mat.raw + uint64_t(pick[i]) * mat.rb, k, deq[i].data());
    refs::weights16_row(f.type, mat.raw + uint64_t(pick[i]) * mat.rb, k, w16[i].data());
  }
  double num = 0, den = 0, emax = 0, ersum = 0, epsmax = 0;
  int cnt = 0;
  for (int64_t r : rpick) {
    refs::QRow qr;
    if (!fp16) qr = dev_q ? (*dev_q)[size_t(r)] : refs::quantize_row(x.data() + r * k, k);
    for (size_t i = 0; i < pick.size(); ++i) {
      double ref = 0, em = 0, cond = 0;
      for (int64_t c = 0; c < k; ++c) {
        ref += double(x[size_t(r * k + c)]) * double(deq[i][size_t(c)]);
        cond += std::fabs(double(x[size_t(r * k + c)]) * double(deq[i][size_t(c)]));
      }
      const double got = y[size_t(r * n + pick[i])];
      num += (got - ref) * (got - ref);
      den += ref * ref;
      if (fp16) {
        for (int64_t c = 0; c < k; ++c) em += double(float(sycl::half(x[size_t(r * k + c)]))) * double(w16[i][size_t(c)]);
      } else {
        em = refs::emulate(f.type, mat.raw + uint64_t(pick[i]) * mat.rb, k, qr);
      }
      ersum += em * em;
      emax = std::max(emax, std::fabs(got - em));
      epsmax = std::max(epsmax, std::fabs(got - em) / (1.2e-7 * cond));
      ++cnt;
    }
  }
  const double rms = std::sqrt(ersum / std::max(cnt, 1));
  return {std::sqrt(num / std::max(den, 1e-30)), emax / std::max(rms, 1e-30), epsmax};
}

void check_matmul(sycl::queue& q, const FormatInfo& f, const char* label, Mat& mat, const std::vector<int64_t>& rows, json::Writer& jw,
                  bool fp16_rows = false) {
  const int64_t k = mat.k, n = mat.n;
  const int64_t max_m = *std::max_element(rows.begin(), rows.end());
  xmx::Activations act = xmx::alloc_activations(q, max_m, k);
  auto* xd = sycl::malloc_device<float>(size_t(max_m * k), q);
  auto* yd = sycl::malloc_device<float>(size_t(max_m * n), q);
  for (int64_t m : rows) {
    if ((m > xmx::kFp16Rows) != fp16_rows) continue;  // int8 GEMVs up to 32 rows, the fp16 GEMM above
    const auto x = random_x(m, k);
    q.memcpy(xd, x.data(), x.size() * 4).wait();
    xmx::quantize(q, xd, m, act);
    xmx::matmul(q, act, mat.w, yd);
    q.wait();
    std::vector<float> y(static_cast<size_t>(m * n));
    q.memcpy(y.data(), yd, y.size() * 4).wait();
    // The emulation takes the card's own int8 activations (a quantization tie can round either way by an ulp of the reciprocal).
    std::vector<refs::QRow> dq;
    if (m <= xmx::kFp16Rows) {
      const int64_t kb = k / 32;
      std::vector<int8_t> hq(static_cast<size_t>(act.m_pad * k));
      std::vector<float> hs(static_cast<size_t>(act.m_pad * kb));
      q.memcpy(hq.data(), act.q, hq.size());
      q.memcpy(hs.data(), act.s, hs.size() * 4).wait();
      for (int64_t r = 0; r < m; ++r) {
        refs::QRow row;
        row.q.resize(static_cast<size_t>(k));
        row.s.resize(static_cast<size_t>(kb));
        for (int64_t c = 0; c < k; ++c) row.q[size_t(c)] = hq[size_t(xmx::act_index(r, c, kb))];
        for (int64_t b = 0; b < kb; ++b) row.s[size_t(b)] = hs[size_t(xmx::act_scale_index(r, b * 32, kb))];
        dq.push_back(std::move(row));
      }
    }
    const Stat s = compare(f, mat, x, m, y, 24, m <= xmx::kFp16Rows ? &dq : nullptr);
    const bool ok = (fp16_rows ? s.fp32 < 2e-3 : s.fp32 < 3e-2) && (s.emul < 2e-5 || s.eps < 64);
    std::printf("  %-8s %-26s %4lldx%-7lld rows %3lld  vs fp32 %9.2e  vs emul %9.2e (%4.1f eps) %s\n", f.name, label, (long long)k, (long long)n,
                (long long)m, s.fp32, s.emul, s.eps, ok ? "" : "  <-- OUT OF BAND");
    jw.begin_object().kv("format", f.name).kv("shape", label).kv("k", k).kv("n", n).kv("rows", m).kv("rel_vs_fp32", s.fp32)
        .kv("rel_vs_emulation", s.emul).kv("emulation_error_in_eps", s.eps).end_object();
  }
  sycl::free(xd, q);
  sycl::free(yd, q);
  xmx::free_activations(q, act);
}

double silu(double g) { return g / (1.0 + std::exp(-g)); }
double gelu(double g) { return 0.5 * g * (1.0 + std::tanh(0.7978845608028654 * (g + 0.044715 * g * g * g))); }

// matmul_geglu on a stacked gate|up (interleaved repack) against the fp32 GLU of its dequantized halves.
void check_glu(sycl::queue& q, const FormatInfo& f, int64_t k, int64_t n2, xmx::Glu act, const std::vector<int64_t>& rows, json::Writer& jw,
               bool fp16_rows = false) {
  Mat mat = make_mat(q, f, n2, k, true);
  const int64_t n = n2 / 2;
  const int64_t max_m = *std::max_element(rows.begin(), rows.end());
  xmx::Activations in = xmx::alloc_activations(q, max_m, k), out = xmx::alloc_activations(q, max_m, n);
  auto* xd = sycl::malloc_device<float>(size_t(max_m * k), q);
  auto* scr = sycl::malloc_device<float>(size_t(max_m * n2), q);
  auto* yd = sycl::malloc_device<float>(size_t(max_m * n), q);
  std::vector<std::vector<float>> deq(static_cast<size_t>(n2), std::vector<float>(static_cast<size_t>(k)));
  for (int64_t r = 0; r < n2; ++r) refs::dequant_row(f.type, mat.raw + uint64_t(r) * mat.rb, k, deq[size_t(r)].data());
  for (int64_t m : rows) {
    if ((m > xmx::kFp16Rows) != fp16_rows) continue;
    const auto x = random_x(m, k);
    q.memcpy(xd, x.data(), x.size() * 4).wait();
    xmx::quantize(q, xd, m, in);
    xmx::matmul_geglu(q, in, mat.w, out, scr, act);
    if (out.fp16)
      q.parallel_for(sycl::range<2>(size_t(m), size_t(n)), [=, o = out](sycl::id<2> id) {
        yd[id[0] * n + id[1]] = float(o.h[xmx::act16_index(int64_t(id[0]), int64_t(id[1]), n / 16)]);
      });
    else
      q.parallel_for(sycl::range<2>(size_t(m), size_t(n)), [=, o = out](sycl::id<2> id) {
        yd[id[0] * n + id[1]] = float(o.q[xmx::act_index(int64_t(id[0]), int64_t(id[1]), n / 32)]) * o.s[xmx::act_scale_index(int64_t(id[0]), int64_t(id[1]), n / 32)];
      });
    q.wait();
    std::vector<float> y(static_cast<size_t>(m * n));
    q.memcpy(y.data(), yd, y.size() * 4).wait();
    double num = 0, den = 0;
    for (int64_t r = 0; r < m; ++r)
      for (int i = 0; i < 96; ++i) {
        const int64_t c = int64_t(g_rng() % uint32_t(n));
        double g = 0, u = 0;
        for (int64_t j = 0; j < k; ++j) g += double(x[size_t(r * k + j)]) * deq[size_t(c)][size_t(j)], u += double(x[size_t(r * k + j)]) * deq[size_t(n + c)][size_t(j)];
        const double ref = (act == xmx::Glu::Silu ? silu(g) : gelu(g)) * u, got = y[size_t(r * n + c)];
        num += (got - ref) * (got - ref);
        den += ref * ref;
      }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    std::printf("  %-8s %-8s gate|up %4lldx%-6lld rows %3lld  vs fp32 %9.2e %s\n", f.name, act == xmx::Glu::Silu ? "SwiGLU" : "GeGLU", (long long)k,
                (long long)n2, (long long)m, rel, rel < (fp16_rows ? 3e-3 : 5e-2) ? "" : "  <-- OUT OF BAND");
    jw.begin_object().kv("format", f.name).kv("glu", act == xmx::Glu::Silu ? "silu" : "gelu").kv("rows", m).kv("rel_vs_fp32", rel).end_object();
  }
  sycl::free(xd, q);
  sycl::free(scr, q);
  sycl::free(yd, q);
  xmx::free_activations(q, in);
  xmx::free_activations(q, out);
  xmx::free_weight(q, mat.w);
}

// rows_of views (a fused matrix sliced into its parts) against the same columns of the whole product.
void check_rows_of(sycl::queue& q, const FormatInfo& f, json::Writer& jw) {
  const int64_t k = 2048, n = 2048, n0 = 512, nv = 768;
  Mat mat = make_mat(q, f, n, k, false);
  xmx::Weight view = xmx::rows_of(mat.w, n0, nv);
  xmx::Activations act = xmx::alloc_activations(q, 2048, k);
  auto* xd = sycl::malloc_device<float>(size_t(2048 * k), q);
  auto* yf = sycl::malloc_device<float>(size_t(2048 * n), q);
  auto* yv = sycl::malloc_device<float>(size_t(2048 * nv), q);
  for (int64_t m : {1, 8, 256}) {
    const auto x = random_x(m, k);
    q.memcpy(xd, x.data(), x.size() * 4).wait();
    xmx::quantize(q, xd, m, act);
    xmx::matmul(q, act, mat.w, yf);
    xmx::matmul(q, act, view, yv);
    q.wait();
    std::vector<float> a(static_cast<size_t>(m * n)), b(static_cast<size_t>(m * nv));
    q.memcpy(a.data(), yf, a.size() * 4);
    q.memcpy(b.data(), yv, b.size() * 4).wait();
    double num = 0, den = 0;
    for (int64_t r = 0; r < m; ++r)
      for (int64_t c = 0; c < nv; ++c) {
        const double d = double(b[size_t(r * nv + c)]) - double(a[size_t(r * n + n0 + c)]);
        num += d * d;
        den += double(a[size_t(r * n + n0 + c)]) * double(a[size_t(r * n + n0 + c)]);
      }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    std::printf("  %-8s rows_of(%lld, %lld) of %lldx%lld  rows %3lld  vs the whole %9.2e %s\n", f.name, (long long)n0, (long long)nv, (long long)k, (long long)n,
                (long long)m, rel, rel < 1e-5 ? "" : "  <-- OUT OF BAND");
    jw.begin_object().kv("format", f.name).kv("rows_of", n0).kv("rows", m).kv("rel_vs_whole", rel).end_object();
  }
  sycl::free(xd, q);
  sycl::free(yf, q);
  sycl::free(yv, q);
  xmx::free_activations(q, act);
  xmx::free_weight(q, mat.w);
}

void bench_decode(sycl::queue& q, const FormatInfo& f, const std::vector<int64_t>& rows, json::Writer& jw) {
  for (const Shape& s : kShapes) {
    if (s.k % refs::block_k(f.type)) continue;
    const bool head = s.n > 100000;
    Mat mat = make_mat(q, f, s.n, s.k, false);
    auto pool = make_pool(q, f, mat.w, mat.bytes, 96ll << 20);
    const int64_t max_m = *std::max_element(rows.begin(), rows.end());
    xmx::Activations act = xmx::alloc_activations(q, max_m, s.k);
    auto* xd = sycl::malloc_device<float>(size_t(max_m * s.k), q);
    auto* yd = sycl::malloc_device<float>(size_t(max_m * s.n), q);
    for (int64_t m : rows) {
      const auto x = random_x(m, s.k);
      q.memcpy(xd, x.data(), x.size() * 4).wait();
      xmx::quantize(q, xd, m, act);
      const int calls = std::max<int>(8, int(pool.size()) * 2);
      const double t = best_time(q, calls, [&](int i) { xmx::matmul(q, act, pool[size_t(i) % pool.size()], yd); });
      const double gbps = double(mat.bytes) / t / 1e9;
      std::printf("  %-8s %-24s %4lldx%-7lld rows %3lld  %9.4f ms %8.0f GB/s  (%.3f bits a weight)\n", f.name, s.name, (long long)s.k, (long long)s.n,
                  (long long)m, t * 1e3, gbps, double(mat.bytes) * 8 / (double(s.k) * double(s.n)));
      jw.begin_object().kv("format", f.name).kv("shape", s.name).kv("k", s.k).kv("n", s.n).kv("rows", m).kv("ms", t * 1e3).kv("weight_gbps", gbps)
          .kv("bytes", mat.bytes).end_object();
    }
    (void)head;
    for (size_t c = 1; c < pool.size(); ++c) xmx::free_weight(q, pool[c]);
    xmx::free_weight(q, mat.w);
    sycl::free(xd, q);
    sycl::free(yd, q);
    xmx::free_activations(q, act);
  }
}

// The existing Q6_K head (fast::matmul_q6: fp32 activations, one sub-group a column) on the head's shape, for comparison with
// the dpas GEMV on int8 activations.
void bench_q6_head(sycl::queue& q, const FormatInfo& f, const std::vector<int64_t>& rows, json::Writer& jw) {
  const int64_t k = 2048, n = 248320;
  const auto host = refs::random_rows(f.type, n, k, g_rng);
  auto* dev = sycl::malloc_device<uint8_t>(host.size(), q);
  q.memcpy(dev, host.data(), host.size()).wait();
  fast::Q6 w = fast::repack_q6k(q, ref::Weight{f.type, k, n, dev, refs::row_bytes(f.type, k)});
  sycl::free(dev, q);
  const int64_t max_m = *std::max_element(rows.begin(), rows.end());
  auto* xd = sycl::malloc_device<float>(size_t(max_m * k), q);
  auto* yd = sycl::malloc_device<float>(size_t(max_m * n), q);
  for (int64_t m : rows) {
    const auto x = random_x(m, k);
    q.memcpy(xd, x.data(), x.size() * 4).wait();
    const double t = best_time(q, 8, [&](int) { fast::matmul_q6(q, w, xd, m, yd); });
    const double gbps = double(host.size()) / t / 1e9;
    std::printf("  %-8s head, fast::matmul_q6 (fp32 x) %4lldx%-7lld rows %3lld  %9.4f ms %8.0f GB/s\n", f.name, (long long)k, (long long)n, (long long)m, t * 1e3, gbps);
    jw.begin_object().kv("format", f.name).kv("shape", "head (fast::matmul_q6)").kv("rows", m).kv("ms", t * 1e3).kv("weight_gbps", gbps).end_object();
  }
  fast::free_q6(q, w);
  sycl::free(xd, q);
  sycl::free(yd, q);
}

// The fp16 GEMM on Bolt's shapes at 2,048 and 4,096 rows, plain and with the GeGLU epilogue (gate|up interleaved).
void bench_gemm(sycl::queue& q, const FormatInfo& f, const std::vector<int64_t>& rows, json::Writer& jw) {
  for (const Shape& s : kShapes) {
    if (s.n > 100000 || s.k % refs::block_k(f.type)) continue;
    const bool glu = std::string(s.name) == "ffn gate|up";
    for (int variant = 0; variant < (glu ? 2 : 1); ++variant) {
      Mat mat = make_mat(q, f, s.n, s.k, variant == 1);
      const int64_t max_m = *std::max_element(rows.begin(), rows.end());
      xmx::Activations act = xmx::alloc_activations(q, max_m, s.k), out = xmx::alloc_activations(q, max_m, s.n / 2);
      auto* xd = sycl::malloc_device<float>(size_t(max_m * s.k), q);
      auto* yd = sycl::malloc_device<float>(size_t(max_m * s.n), q);
      for (int64_t m : rows) {
        const auto x = random_x(m, s.k);
        q.memcpy(xd, x.data(), x.size() * 4).wait();
        xmx::quantize(q, xd, m, act);
        const double t = best_time(q, 6, [&](int) {
          if (variant == 1) xmx::matmul_geglu(q, act, mat.w, out, nullptr, xmx::Glu::Gelu);
          else xmx::matmul(q, act, mat.w, yd);
        });
        const double tf = 2.0 * double(m) * double(s.n) * double(s.k) / t / 1e12;
        std::printf("  %-8s %-24s %4lldx%-7lld rows %4lld  %9.3f ms %7.1f TFLOPS %s\n", f.name, variant ? "ffn gate|up + GeGLU" : s.name, (long long)s.k,
                    (long long)s.n, (long long)m, t * 1e3, tf, variant ? "(GLU epilogue, TFLOPS counted on the stacked matrix)" : "");
        jw.begin_object().kv("format", f.name).kv("shape", variant ? "ffn gate|up + GeGLU" : s.name).kv("k", s.k).kv("n", s.n).kv("rows", m)
            .kv("ms", t * 1e3).kv("tflops", tf).end_object();
      }
      sycl::free(xd, q);
      sycl::free(yd, q);
      xmx::free_activations(q, act);
      xmx::free_activations(q, out);
      xmx::free_weight(q, mat.w);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> only;
  std::map<std::string, std::string> ggufs;
  std::vector<int64_t> rows = {1, 2, 8, 16, 32};
  std::string json_path;
  bool check = false, bench = false, gemm = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--formats" && i + 1 < argc) {
      std::stringstream ss(argv[++i]);
      for (std::string t; std::getline(ss, t, ',');) only.push_back(t);
    } else if (a == "--gguf" && i + 1 < argc) {
      const std::string v = argv[++i];
      ggufs[v.substr(0, v.find('='))] = v.substr(v.find('=') + 1);
    } else if (a == "--rows" && i + 1 < argc) rows = parse_list(argv[++i]);
    else if (a == "--json" && i + 1 < argc) json_path = argv[++i];
    else if (a == "--check") check = true;
    else if (a == "--bench") bench = true;
    else if (a == "--gemm") gemm = true;
    else {
      std::fprintf(stderr, "usage: eklow-qfmt-test [--formats LIST] [--gguf FORMAT=PATH] [--check] [--bench] [--gemm] [--rows LIST] [--json FILE]\n");
      return 2;
    }
  }
  if (!check && !bench && !gemm) check = true;
  try {
    sycl::queue q(rt::pick_device(), {sycl::property::queue::in_order()});
    json::Writer jw;
    jw.begin_object().kv("device", q.get_device().get_info<sycl::info::device::name>());
    for (const FormatInfo& f : kFormats) {
      if (!only.empty() && std::find(only.begin(), only.end(), f.name) == only.end()) continue;
      std::printf("== %s (%s)\n", f.name, xmx::format_name(f.format));
      if (check) {
        jw.key(std::string("check_") + f.name).begin_array();
        auto it = ggufs.find(f.name);
        if (it != ggufs.end()) {
          auto file = gguf::File::open(it->second);
          for (const char* name : {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.5.attn_q.weight", "blk.0.attn_output.weight"}) {
            const gguf::Tensor* t = file.tensor(name);
            if (!t || t->type != f.type) {
              std::printf("  %s: no %s tensor in %s (type %s)\n", name, f.name, it->second.c_str(), t ? gguf::type_name(t->type) : "absent");
              continue;
            }
            Mat mat = make_mat(q, f, int64_t(t->ne[1]), int64_t(t->ne[0]), false, t->data);
            check_matmul(q, f, (std::string("real ") + name).c_str(), mat, rows, jw);
            xmx::free_weight(q, mat.w);
          }
        }
        for (const Shape& s : kShapes) {
          if (s.n > 100000 || s.k % refs::block_k(f.type)) continue;
          Mat mat = make_mat(q, f, s.n, s.k, false);
          check_matmul(q, f, s.name, mat, rows, jw);
          xmx::free_weight(q, mat.w);
        }
        for (xmx::Glu act : {xmx::Glu::Silu, xmx::Glu::Gelu}) check_glu(q, f, 2560, 2048, act, rows, jw);
        check_rows_of(q, f, jw);
        jw.end_array();
      }
      if (gemm) {
        jw.key(std::string("gemm_check_") + f.name).begin_array();
        const std::vector<int64_t> big = {64, 256, 2048};
        for (const Shape& s : kShapes) {
          if (s.n > 100000 || s.k % refs::block_k(f.type)) continue;
          Mat mat = make_mat(q, f, s.n, s.k, false);
          check_matmul(q, f, s.name, mat, big, jw, true);
          xmx::free_weight(q, mat.w);
        }
        for (xmx::Glu act : {xmx::Glu::Silu, xmx::Glu::Gelu}) check_glu(q, f, 2560, 2048, act, big, jw, true);
        jw.end_array();
        jw.key(std::string("gemm_") + f.name).begin_array();
        bench_gemm(q, f, {2048, 4096}, jw);
        jw.end_array();
      }
      if (bench) {
        jw.key(std::string("decode_") + f.name).begin_array();
        bench_decode(q, f, rows, jw);
        if (f.format == xmx::WeightFormat::Q6_K) bench_q6_head(q, f, rows, jw);
        jw.end_array();
      }
    }
    jw.end_object();
    if (!json_path.empty()) std::ofstream(json_path) << jw.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-qfmt-test: %s\n", e.what());
    return 1;
  }
  return 0;
}
