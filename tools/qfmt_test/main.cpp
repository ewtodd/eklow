// eklow-qfmt-test: the Qwen weight formats' kernels (xmx.hpp, xmx_fmt.hpp) on the card, checked and timed.
//
//   eklow-qfmt-test [--formats Q8_0,Q4_K,Q5_K,Q6_K,IQ4_XS,Q4_0] [--gguf Q4_K=PATH ...] [--check] [--bench]
//                   [--gemm] [--rows 1,2,8,16,32] [--json FILE]
//
// --check: for each format, tensors of the given GGUFs (real ones: --gguf FORMAT=PATH, each tensor in the format's
//   type) and random valid blocks at the Qwen shapes: the decode matmul against (1) an fp32 dot of ggml's own
//   dequantization with the fp32 activations (the int8 activation quantization shows: ~1e-2) and (2) an exact CPU
//   emulation of the quantized arithmetic (~1e-6); the SwiGLU and GeGLU fusions and the grouped GEMVs likewise.
// --bench: GB/s of weight read per format, Qwen shape and row count (decode), against the 399 GB/s the card reads;
//   each call reads a different copy of the matrix (a pool larger than the 16 MB L2).
// --gemm: the fp16 GEMM's TFLOPS at 2,048 and 4,096 rows and the grouped expert GEMMs (check and bench both).
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

// Qwen3.6-35B-A3B's matrices (hidden 2048): K, N.
struct Shape {
  const char* name;
  int64_t k, n;
};
const Shape kQwen[] = {
    {"expert gate|up", 2048, 1024}, {"expert down", 512, 2048},   {"attn_qkv / attn_q", 2048, 8192}, {"attn_gate", 2048, 4096},
    {"ssm_out / attn_output", 4096, 2048}, {"attn_k / attn_v", 2048, 512}, {"head", 2048, 248320},
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

// The grouped GEMVs against xmx::matmul per pair on the same experts (same arithmetic: ~1e-6), gate_up with SwiGLU.
void check_grouped(sycl::queue& q, const FormatInfo& f, json::Writer& jw) {
  const int64_t k = 2048, n2 = 1024, nd = 2048, kd = 512, NE = 8;
  std::vector<Mat> gu, dn;
  std::vector<xmx::Weight> guw, dnw;
  for (int64_t e = 0; e < NE; ++e) {
    gu.push_back(make_mat(q, f, n2, k, true));
    dn.push_back(make_mat(q, f, nd, kd, false));
    guw.push_back(gu.back().w);
    dnw.push_back(dn.back().w);
  }
  xmx::ExpertTable tg = xmx::make_expert_table(q, guw), td = xmx::make_expert_table(q, dnw);
  for (int64_t P : {8, 16, 32}) {
    const int64_t rows = (P + 7) / 8;
    xmx::Activations act = xmx::alloc_activations(q, 32, k), one = xmx::alloc_activations(q, 32, k);
    xmx::Activations actd = xmx::alloc_activations(q, 32, kd), oned = xmx::alloc_activations(q, 32, kd);
    auto* xd = sycl::malloc_device<float>(size_t(32 * k), q);
    auto* xdd = sycl::malloc_device<float>(size_t(32 * kd), q);
    auto* yg = sycl::malloc_device<float>(size_t(P * n2), q);
    auto* yr = sycl::malloc_device<float>(size_t(n2), q);
    auto* ygd = sycl::malloc_device<float>(size_t(P * nd), q);
    auto* yrd = sycl::malloc_device<float>(size_t(nd), q);
    auto* idd = sycl::malloc_device<int32_t>(size_t(P), q);
    std::vector<int32_t> ids(static_cast<size_t>(P));
    for (auto& i : ids) i = int32_t(g_rng() % uint32_t(NE));
    q.memcpy(idd, ids.data(), ids.size() * 4);
    const auto x = random_x(rows, k), xdn = random_x(P, kd);
    q.memcpy(xd, x.data(), x.size() * 4);
    q.memcpy(xdd, xdn.data(), xdn.size() * 4).wait();
    xmx::quantize(q, xd, rows, act);
    xmx::quantize(q, xdd, P, actd);
    xmx::matmul_grouped(q, act, tg, idd, P, 8, yg);
    xmx::matmul_grouped(q, actd, td, idd, P, 1, ygd);
    q.wait();
    std::vector<float> hg(static_cast<size_t>(P * n2)), hgd(static_cast<size_t>(P * nd)), hr(static_cast<size_t>(n2)), hrd(static_cast<size_t>(nd));
    q.memcpy(hg.data(), yg, hg.size() * 4);
    q.memcpy(hgd.data(), ygd, hgd.size() * 4).wait();
    double num = 0, den = 0, numd = 0, dend = 0;
    for (int64_t p = 0; p < P; ++p) {
      xmx::quantize(q, xd + (p / 8) * k, 1, one);
      xmx::matmul(q, one, gu[size_t(ids[size_t(p)])].w, yr);
      xmx::quantize(q, xdd + p * kd, 1, oned);
      xmx::matmul(q, oned, dn[size_t(ids[size_t(p)])].w, yrd);
      q.memcpy(hr.data(), yr, hr.size() * 4);
      q.memcpy(hrd.data(), yrd, hrd.size() * 4).wait();
      for (int64_t i = 0; i < n2; ++i) num += double(hg[size_t(p * n2 + i)] - hr[size_t(i)]) * (hg[size_t(p * n2 + i)] - hr[size_t(i)]), den += double(hr[size_t(i)]) * hr[size_t(i)];
      for (int64_t i = 0; i < nd; ++i) numd += double(hgd[size_t(p * nd + i)] - hrd[size_t(i)]) * (hgd[size_t(p * nd + i)] - hrd[size_t(i)]), dend += double(hrd[size_t(i)]) * hrd[size_t(i)];
    }
    // The fused SwiGLU on the grouped output: the same GLU kernel as the single one's, on the same bytes.
    xmx::Activations glu = xmx::alloc_activations(q, 32, n2 / 2);
    auto* scr = sycl::malloc_device<float>(size_t(P * n2), q);
    xmx::matmul_geglu_grouped(q, act, tg, idd, P, 8, glu, scr, xmx::Glu::Silu);
    q.wait();
    std::vector<int8_t> hq(static_cast<size_t>(glu.m_pad * (n2 / 2)));
    std::vector<float> hs(static_cast<size_t>(glu.m_pad * (n2 / 2) / 32));
    q.memcpy(hq.data(), glu.q, hq.size());
    q.memcpy(hs.data(), glu.s, hs.size() * 4).wait();
    double numg = 0, deng = 0;
    for (int64_t p = 0; p < P; ++p)
      for (int i = 0; i < 64; ++i) {
        const int64_t c = int64_t(g_rng() % uint32_t(n2 / 2));
        // the interleaved halves: tile 2j gate, 2j+1 up of 8 columns -> hg holds [gate | up] in the repack's column order
        const int64_t tile = c / 8, col = tile * 16 + c % 8;
        const double gate = hg[size_t(p * n2 + col)], up = hg[size_t(p * n2 + col + 8)];
        const double ref = silu(gate) * up;
        const double got = double(hq[size_t(xmx::act_index(p, c, n2 / 2 / 32))]) * double(hs[size_t(xmx::act_scale_index(p, c, n2 / 2 / 32))]);
        numg += (got - ref) * (got - ref);
        deng += ref * ref;
      }
    const double r1 = std::sqrt(num / den), r2 = std::sqrt(numd / dend), r3 = std::sqrt(numg / std::max(deng, 1e-30));
    std::printf("  %-8s grouped P=%2lld  gate_up vs matmul %9.2e  down %9.2e  SwiGLU(int8) vs fp32 %9.2e %s\n", f.name, (long long)P, r1, r2, r3,
                (r1 < 1e-5 && r2 < 1e-5 && r3 < 3e-2) ? "" : "  <-- OUT OF BAND");
    jw.begin_object().kv("format", f.name).kv("pairs", P).kv("gate_up_vs_matmul", r1).kv("down_vs_matmul", r2).kv("swiglu_vs_fp32", r3).end_object();
    for (void* p : {static_cast<void*>(xd), static_cast<void*>(xdd), static_cast<void*>(yg), static_cast<void*>(yr), static_cast<void*>(ygd),
                    static_cast<void*>(yrd), static_cast<void*>(idd), static_cast<void*>(scr)})
      sycl::free(p, q);
    xmx::free_activations(q, act);
    xmx::free_activations(q, one);
    xmx::free_activations(q, actd);
    xmx::free_activations(q, oned);
    xmx::free_activations(q, glu);
  }
  xmx::free_expert_table(q, tg);
  xmx::free_expert_table(q, td);
  for (auto& m : gu) xmx::free_weight(q, m.w);
  for (auto& m : dn) xmx::free_weight(q, m.w);
}

void bench_decode(sycl::queue& q, const FormatInfo& f, const std::vector<int64_t>& rows, json::Writer& jw) {
  for (const Shape& s : kQwen) {
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

// The grouped decode GEMVs over 256 experts: the engine's gate_up + SwiGLU and down, P pairs of 8 distinct-ish experts.
void bench_grouped(sycl::queue& q, const FormatInfo& f, json::Writer& jw) {
  const int64_t NE = 256;
  for (const bool gate_up : {true, false}) {
    const int64_t k = gate_up ? 2048 : 512, n = gate_up ? 1024 : 2048;
    Mat mat = make_mat(q, f, n, k, gate_up);
    std::vector<xmx::Weight> experts;
    const uint64_t tb = f.format == xmx::WeightFormat::Q4_0 ? xmx::weight_tile_bytes(k, n) : xmx::weight_tile_bytes(f.format, k, n);
    for (int64_t e = 0; e < NE; ++e) {
      xmx::Weight w = mat.w;
      if (e) {
        w.tiles = sycl::malloc_device<uint8_t>(size_t(tb), q);
        w.scales = reinterpret_cast<sycl::half*>(sycl::malloc_device<uint8_t>(size_t(mat.bytes - tb), q));
        if (!w.tiles || !w.scales) throw std::runtime_error("out of device memory for the expert pool");
        q.memcpy(w.tiles, mat.w.tiles, size_t(tb));
        q.memcpy(w.scales, mat.w.scales, size_t(mat.bytes - tb));
      }
      experts.push_back(w);
    }
    q.wait();
    xmx::ExpertTable table = xmx::make_expert_table(q, experts);
    for (int64_t P : {8, 16, 32}) {
      const int64_t row_div = gate_up ? 8 : 1, rows = (P + row_div - 1) / row_div;
      xmx::Activations act = xmx::alloc_activations(q, 32, k);
      auto* xd = sycl::malloc_device<float>(size_t(32 * k), q);
      auto* yg = sycl::malloc_device<float>(size_t(P * n), q);
      auto* idsets = sycl::malloc_device<int32_t>(size_t(32 * P), q);
      std::vector<int32_t> all(static_cast<size_t>(32 * P));
      int64_t distinct = 0;
      for (int s = 0; s < 32; ++s) {
        for (int64_t p = 0; p < P; ++p) all[size_t(s * P + p)] = int32_t(g_rng() % uint32_t(NE));
        std::vector<int32_t> u(all.begin() + s * P, all.begin() + (s + 1) * P);
        std::sort(u.begin(), u.end());
        distinct += std::unique(u.begin(), u.end()) - u.begin();
      }
      q.memcpy(idsets, all.data(), all.size() * 4);
      const auto x = random_x(rows, k);
      q.memcpy(xd, x.data(), x.size() * 4).wait();
      xmx::quantize(q, xd, rows, act);
      const double t = best_time(q, 32, [&](int i) { xmx::matmul_grouped(q, act, table, idsets + i * P, P, row_div, yg); });
      const double gbps = double(distinct) / 32 * double(mat.bytes) / t / 1e9;
      std::printf("  %-8s grouped %-12s P=%2lld  distinct %5.1f  %9.4f ms %8.0f GB/s\n", f.name, gate_up ? "gate|up" : "down", (long long)P,
                  double(distinct) / 32, t * 1e3, gbps);
      jw.begin_object().kv("format", f.name).kv("grouped", gate_up ? "gate_up" : "down").kv("pairs", P).kv("distinct", double(distinct) / 32)
          .kv("ms", t * 1e3).kv("weight_gbps", gbps).end_object();
      if (gate_up) {
        xmx::Activations glu = xmx::alloc_activations(q, 32, n / 2);
        auto* scr = sycl::malloc_device<float>(size_t(P * n), q);
        const double tg = best_time(q, 32, [&](int i) { xmx::matmul_geglu_grouped(q, act, table, idsets + i * P, P, row_div, glu, scr, xmx::Glu::Silu); });
        std::printf("  %-8s grouped %-12s P=%2lld  distinct %5.1f  %9.4f ms %8.0f GB/s\n", f.name, "  + SwiGLU", (long long)P, double(distinct) / 32,
                    tg * 1e3, double(distinct) / 32 * double(mat.bytes) / tg / 1e9);
        jw.begin_object().kv("format", f.name).kv("grouped", "gate_up_swiglu").kv("pairs", P).kv("ms", tg * 1e3)
            .kv("weight_gbps", double(distinct) / 32 * double(mat.bytes) / tg / 1e9).end_object();
        sycl::free(scr, q);
        xmx::free_activations(q, glu);
      }
      sycl::free(xd, q);
      sycl::free(yg, q);
      sycl::free(idsets, q);
      xmx::free_activations(q, act);
    }
    xmx::free_expert_table(q, table);
    for (int64_t e = 1; e < NE; ++e) xmx::free_weight(q, experts[size_t(e)]);
    xmx::free_weight(q, mat.w);
  }
}


// The fp16 GEMM on the Qwen shapes at 2,048 and 4,096 rows, plain and with the SwiGLU epilogue (gate|up interleaved).
void bench_gemm(sycl::queue& q, const FormatInfo& f, const std::vector<int64_t>& rows, json::Writer& jw) {
  for (const Shape& s : kQwen) {
    if (s.n > 100000 || s.k % refs::block_k(f.type)) continue;
    const bool glu = std::string(s.name) == "expert gate|up";
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
          if (variant == 1) xmx::matmul_geglu(q, act, mat.w, out, nullptr, xmx::Glu::Silu);
          else xmx::matmul(q, act, mat.w, yd);
        });
        const double tf = 2.0 * double(m) * double(s.n) * double(s.k) / t / 1e12;
        std::printf("  %-8s %-24s %4lldx%-7lld rows %4lld  %9.3f ms %7.1f TFLOPS %s\n", f.name, variant ? "expert gate|up + SwiGLU" : s.name, (long long)s.k,
                    (long long)s.n, (long long)m, t * 1e3, tf, variant ? "(GLU epilogue, TFLOPS counted on the stacked matrix)" : "");
        jw.begin_object().kv("format", f.name).kv("shape", variant ? "expert gate|up + SwiGLU" : s.name).kv("k", s.k).kv("n", s.n).kv("rows", m)
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

// The grouped fp16 GEMMs (prefill's MoE) on 256 experts, 2,048 tokens each routed to 8 of them (Qwen's: gate|up 2048 -> 2 x 512 with
// SwiGLU, down 512 -> 2048): against the dense fp16 path per expert (same arithmetic: ~1e-6..1e-4) and timed.
void grouped16(sycl::queue& q, const FormatInfo& f, int64_t tile, bool check, json::Writer& jw) {
  const int64_t T = 2048, NU = 8, P = T * NU, NE = 256, E = 2048, Fe = 512;
  std::vector<Mat> gum, dnm;
  std::vector<xmx::Weight> gu, dn;
  for (int64_t e = 0; e < NE; ++e) {
    gum.push_back(make_mat(q, f, 2 * Fe, E, true));
    dnm.push_back(make_mat(q, f, E, Fe, false));
    gu.push_back(gum.back().w);
    dn.push_back(dnm.back().w);
    if (!check) {  // timing only: the host copies are not needed
      gum.back().host.clear();
      dnm.back().host.clear();
    }
  }
  xmx::ExpertTable tab_gu = xmx::make_expert_table(q, gu), tab_dn = xmx::make_expert_table(q, dn);
  std::vector<int32_t> ids(static_cast<size_t>(P)), perm(static_cast<size_t>(NE));
  for (int64_t t = 0; t < T; ++t) {
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), g_rng);
    for (int64_t j = 0; j < NU; ++j) ids[size_t(t * NU + j)] = perm[size_t(j)];
  }
  const auto x = random_x(T, E);
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
  xmx::matmul_geglu_grouped16(q, a, tab_gu, tiles, max_tiles, tile, h, xmx::Glu::Silu);
  xmx::matmul_grouped16(q, h, tab_dn, tiles, max_tiles, tile, y);
  q.wait();
  double rel_gu = 0, rel_dn = 0;
  if (check) {
    std::vector<int32_t> hrow(static_cast<size_t>(P)), htiles(static_cast<size_t>(4 + 4 * max_tiles));
    q.memcpy(hrow.data(), rowp, hrow.size() * 4);
    q.memcpy(htiles.data(), tiles, htiles.size() * 4).wait();
    std::vector<std::vector<int64_t>> by_expert(static_cast<size_t>(NE));
    for (int64_t p = 0; p < P; ++p) by_expert[size_t(ids[size_t(p)])].push_back(p);
    std::vector<sycl::half> hh(static_cast<size_t>(rows * Fe));
    std::vector<float> hy(static_cast<size_t>(rows * E));
    q.memcpy(hh.data(), h, hh.size() * 2);
    q.memcpy(hy.data(), y, hy.size() * 4).wait();
    xmx::Activations ain = xmx::alloc_activations(q, 4096, E), ah = xmx::alloc_activations(q, 4096, Fe);
    auto* xe = sycl::malloc_device<float>(size_t(4096 * E), q);
    auto* ye = sycl::malloc_device<float>(size_t(4096 * E), q);
    std::vector<float> hye(static_cast<size_t>(4096 * E));
    double ngu = 0, dgu = 0, ndn = 0, ddn = 0;
    for (int64_t e = 0; e < NE; e += 7) {  // a spread of experts
      const auto& prs = by_expert[size_t(e)];
      if (prs.empty()) continue;
      const int64_t cnt = int64_t(prs.size()), m = std::max<int64_t>(cnt, xmx::kFp16Rows + 1);
      std::vector<float> xs(static_cast<size_t>(m * E), 0.0f);
      for (int64_t i = 0; i < cnt; ++i) std::copy_n(x.begin() + (prs[size_t(i)] / NU) * E, E, xs.begin() + i * E);
      q.memcpy(xe, xs.data(), xs.size() * 4).wait();
      xmx::quantize(q, xe, m, ain);
      xmx::matmul_geglu(q, ain, gu[size_t(e)], ah, nullptr, xmx::Glu::Silu);
      xmx::matmul(q, ah, dn[size_t(e)], ye);
      std::vector<sycl::half> rh(static_cast<size_t>(ah.m_pad * Fe));
      q.memcpy(rh.data(), ah.h, rh.size() * 2);
      q.memcpy(hye.data(), ye, size_t(m * E) * 4).wait();
      for (int64_t i = 0; i < cnt; ++i) {
        const int64_t row = hrow[size_t(prs[size_t(i)])];
        for (int64_t c = 0; c < Fe; ++c) {
          const double g = double(float(hh[size_t(xmx::act16_index(row, c, Fe / 16))])), r = double(float(rh[size_t(xmx::act16_index(i, c, Fe / 16))]));
          ngu += (g - r) * (g - r);
          dgu += r * r;
        }
        for (int64_t c = 0; c < E; ++c) {
          const double g = double(hy[size_t(row * E + c)]), r = double(hye[size_t(i * E + c)]);
          ndn += (g - r) * (g - r);
          ddn += r * r;
        }
      }
    }
    rel_gu = std::sqrt(ngu / std::max(dgu, 1e-30));
    rel_dn = std::sqrt(ndn / std::max(ddn, 1e-30));
    sycl::free(xe, q);
    sycl::free(ye, q);
    xmx::free_activations(q, ain);
    xmx::free_activations(q, ah);
  }
  std::vector<int32_t> htl(static_cast<size_t>(4));
  q.memcpy(htl.data(), tiles, 16).wait();
  const int64_t used_rows = htl[0] * tile;
  const double t_gu = best_time(q, 10, [&](int) { xmx::matmul_geglu_grouped16(q, a, tab_gu, tiles, max_tiles, tile, h, xmx::Glu::Silu); });
  const double t_dn = best_time(q, 10, [&](int) { xmx::matmul_grouped16(q, h, tab_dn, tiles, max_tiles, tile, y); });
  const double fl_gu = 2.0 * double(P) * double(E) * double(2 * Fe), fl_dn = 2.0 * double(P) * double(Fe) * double(E);
  std::printf("  %-8s grouped16 tile %3lld  gate|up+SwiGLU %8.3f ms %6.1f TFLOPS (%6.1f padded)   down %8.3f ms %6.1f TFLOPS (%6.1f padded)", f.name,
              (long long)tile, t_gu * 1e3, fl_gu / t_gu / 1e12, fl_gu * double(used_rows) / double(P) / t_gu / 1e12, t_dn * 1e3, fl_dn / t_dn / 1e12,
              fl_dn * double(used_rows) / double(P) / t_dn / 1e12);
  if (check) std::printf("   vs dense: gate|up %.2e  down %.2e %s", rel_gu, rel_dn, (rel_gu < 1e-3 && rel_dn < 1e-3) ? "" : "  <-- OUT OF BAND");
  std::printf("\n");
  jw.begin_object().kv("format", f.name).kv("tile", tile).kv("ms_gate_up", t_gu * 1e3).kv("ms_down", t_dn * 1e3).kv("tflops_gate_up", fl_gu / t_gu / 1e12)
      .kv("tflops_down", fl_dn / t_dn / 1e12).kv("tflops_gate_up_padded", fl_gu * double(used_rows) / double(P) / t_gu / 1e12)
      .kv("tflops_down_padded", fl_dn * double(used_rows) / double(P) / t_dn / 1e12).kv("rel_gate_up", rel_gu).kv("rel_down", rel_dn).end_object();
  for (void* p : {static_cast<void*>(idd), static_cast<void*>(rowp), static_cast<void*>(scr), static_cast<void*>(tiles), static_cast<void*>(xd),
                  static_cast<void*>(a), static_cast<void*>(h), static_cast<void*>(y)})
    sycl::free(p, q);
  xmx::free_expert_table(q, tab_gu);
  xmx::free_expert_table(q, tab_dn);
  for (auto& w : gu) xmx::free_weight(q, w);
  for (auto& w : dn) xmx::free_weight(q, w);
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
        for (const Shape& s : kQwen) {
          if (s.n > 100000 || s.k % refs::block_k(f.type)) continue;
          Mat mat = make_mat(q, f, s.n, s.k, false);
          check_matmul(q, f, s.name, mat, rows, jw);
          xmx::free_weight(q, mat.w);
        }
        for (xmx::Glu act : {xmx::Glu::Silu, xmx::Glu::Gelu}) check_glu(q, f, 2048, 1024, act, rows, jw);
        check_grouped(q, f, jw);
        check_rows_of(q, f, jw);
        jw.end_array();
      }
      if (gemm) {
        jw.key(std::string("gemm_check_") + f.name).begin_array();
        const std::vector<int64_t> big = {64, 256, 2048};
        for (const Shape& s : kQwen) {
          if (s.n > 100000 || s.k % refs::block_k(f.type)) continue;
          Mat mat = make_mat(q, f, s.n, s.k, false);
          check_matmul(q, f, s.name, mat, big, jw, true);
          xmx::free_weight(q, mat.w);
        }
        for (xmx::Glu act : {xmx::Glu::Silu, xmx::Glu::Gelu}) check_glu(q, f, 2048, 1024, act, big, jw, true);
        jw.end_array();
        jw.key(std::string("gemm_") + f.name).begin_array();
        bench_gemm(q, f, {2048, 4096}, jw);
        for (int64_t tile : {64, 128}) grouped16(q, f, tile, tile == 64, jw);
        jw.end_array();
      }
      if (bench) {
        jw.key(std::string("decode_") + f.name).begin_array();
        bench_decode(q, f, rows, jw);
        if (f.format == xmx::WeightFormat::Q6_K) bench_q6_head(q, f, rows, jw);
        bench_grouped(q, f, jw);
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
