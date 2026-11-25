// eklow-cpu-expert-test: one Inkling 2 Mini expert's FFN on the host CPU (src/cpu/expert.hpp) against a plain fp32
// reference written here with the reference's dequantization, and its speed.
//
//   eklow-cpu-expert-test --gguf FILE [--layers 0,6,19,20,23] [--experts 3,77,100] [--threads N] [--reps N] [--seed N]
//                         [--no-check] [--no-time] [--nx 16,64,256]
//
// Check: for each expert, nx = 1 and 8 inputs shaped like the model's (random rows, RMS-normed, times the layer's
// ffn_norm.weight); relative L2 per row against (a) the fp32 reference, whose gap is the int8 activations' (the floor),
// and (b) the same reference on activations rounded to int8 the way the kernels round them (x per 16 or 32, situ's
// output per 16 or 32), which leaves only arithmetic order. Speed (medians): one expert at nx = 1 and the weights' GB/s
// (experts rotate over the layer's 128 so that the weights come out of RAM, not the cache), 8 experts one after another and
// in one parallel pass, and nx = 16, 64, 256 against one expert.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <immintrin.h>

#include "cpu/expert.hpp"
#include "cpu/nvfp4.hpp"
#include "model/gguf.hpp"

using namespace eklow;
using eklow::cpu::QType;

namespace {

std::vector<int64_t> list(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, ',');)
    if (!p.empty()) out.push_back(std::atoll(p.c_str()));
  return out;
}

const gguf::Tensor& need(const gguf::File& f, const std::string& n) {
  const gguf::Tensor* t = f.tensor(n);
  if (!t) throw std::runtime_error("missing tensor " + n);
  return *t;
}

float half_at(const uint8_t* p) {
  uint16_t raw;
  std::memcpy(&raw, p, 2);
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(raw)));
}

// ---- the reference's dequantization (src/kernels/ref.cpp), one row of k weights into floats
void scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

void decode_q5_k(const uint8_t* blk, float* y) {
  const float d = half_at(blk), mn = half_at(blk + 2);
  const uint8_t* sc = blk + 4;
  const uint8_t* qh = blk + 16;
  const uint8_t* ql = blk + 48;
  int is = 0;
  uint8_t u1 = 1, u2 = 2;
  for (int j = 0; j < 256; j += 64) {
    uint8_t s, m;
    scale_min_k4(is, sc, s, m);
    const float d1 = d * s, m1 = mn * m;
    scale_min_k4(is + 1, sc, s, m);
    const float d2 = d * s, m2 = mn * m;
    for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
    for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
    ql += 32;
    is += 2;
    u1 <<= 2;
    u2 <<= 2;
  }
}

float ue4m3_half(uint8_t x) {
  if (x == 0 || x == 0x7f) return 0.0f;
  const int e = (x >> 3) & 15, m = x & 7;
  const float raw = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
  return raw * 0.5f;
}

float e2m1_doubled(uint8_t c) {
  constexpr int8_t k[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return float(k[c & 15]);
}

void dequant_row(gguf::Type t, const uint8_t* row, int64_t k, float* y) {
  if (t == gguf::Type::Q8_0) {
    for (int64_t b = 0; b < k / 32; ++b) {
      const uint8_t* blk = row + b * 34;
      const float d = half_at(blk);
      for (int j = 0; j < 32; ++j) y[b * 32 + j] = d * float(reinterpret_cast<const int8_t*>(blk + 2)[j]);
    }
  } else if (t == gguf::Type::Q5_K) {
    for (int64_t b = 0; b < k / 256; ++b) decode_q5_k(row + b * 176, y + b * 256);
  } else if (t == gguf::Type::NVFP4) {
    for (int64_t b = 0; b < k / 64; ++b) {
      const uint8_t* blk = row + b * 36;
      for (int s = 0; s < 4; ++s)
        for (int j = 0; j < 8; ++j) {
          const uint8_t c = blk[4 + s * 8 + j];
          y[b * 64 + s * 16 + j] = ue4m3_half(blk[s]) * e2m1_doubled(c);
          y[b * 64 + s * 16 + j + 8] = ue4m3_half(blk[s]) * e2m1_doubled(c >> 4);
        }
    }
  } else {
    throw std::runtime_error(std::string("no dequantization for ") + gguf::type_name(t));
  }
}

QType qtype(gguf::Type t) {
  if (t == gguf::Type::Q8_0) return QType::Q8_0;
  if (t == gguf::Type::Q5_K) return QType::Q5_K;
  if (t == gguf::Type::NVFP4) return QType::NVFP4;
  throw std::runtime_error(std::string("unsupported expert type ") + gguf::type_name(t));
}

// Activations rounded to int8 as the kernels do (blocks of `blk`: s = max|x| / 127, q = round(x / s)), back to float.
void fake_quant(const float* x, int64_t k, int blk, float* out) {
  for (int64_t b = 0; b < k / blk; ++b) {
    float amax = 0;
    for (int i = 0; i < blk; ++i) amax = std::max(amax, std::fabs(x[b * blk + i]));
    const float s = amax / 127.0f, inv = s != 0.0f ? 1.0f / s : 0.0f;
    for (int i = 0; i < blk; ++i) out[b * blk + i] = float(std::lrintf(x[b * blk + i] * inv)) * s;
  }
}

struct Dense {  // an expert's matrices dequantized
  std::vector<float> gu, dn;
};

// The reference's expert: y = dn_scale * down(situ(gu_scale * gate_up(x))); `quant` rounds the activations to int8.
void ref_expert(const Dense& d, int64_t E, int64_t F, const float* x, bool quant, int blk_gu, int blk_dn, float gs, float ds,
                float beta, float lb, float* y) {
  std::vector<float> xq(size_t(E), 0.0f), h(size_t(F), 0.0f), hq(size_t(F), 0.0f);
  const float* xin = x;
  if (quant) {
    fake_quant(x, E, blk_gu, xq.data());
    xin = xq.data();
  }
  for (int64_t i = 0; i < F; ++i) {
    double g = 0, u = 0;
    const float *rg = &d.gu[size_t(i * E)], *ru = &d.gu[size_t((F + i) * E)];
    for (int64_t j = 0; j < E; ++j) g += double(rg[j]) * xin[j], u += double(ru[j]) * xin[j];
    const float gf = float(g) * gs, uf = float(u) * gs;
    const float a = beta * std::tanh(gf / beta) * (1.0f / (1.0f + std::exp(-gf)));
    h[size_t(i)] = a * (lb > 0 ? lb * std::tanh(uf / lb) : uf);
  }
  const float* hin = h.data();
  if (quant) {
    fake_quant(h.data(), F, blk_dn, hq.data());
    hin = hq.data();
  }
  for (int64_t i = 0; i < E; ++i) {
    double s = 0;
    const float* r = &d.dn[size_t(i * F)];
    for (int64_t j = 0; j < F; ++j) s += double(r[j]) * hin[j];
    y[i] = float(s) * ds;
  }
}

double rel_l2(const float* a, const float* b, int64_t n) {
  double d2 = 0, r2 = 0;
  for (int64_t i = 0; i < n; ++i) d2 += (double(a[i]) - b[i]) * (double(a[i]) - b[i]), r2 += double(b[i]) * b[i];
  return std::sqrt(d2 / std::max(r2, 1e-30));
}

double now_us() {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  std::vector<int64_t> layers = {0, 6, 19, 20, 23}, experts = {3, 77, 100}, nxs = {16, 64, 256};
  int reps = 60, threads = 16;
  uint32_t seed = 1;
  bool check = true, timing = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--gguf") path = next();
    else if (a == "--layers") layers = list(next());
    else if (a == "--experts") experts = list(next());
    else if (a == "--nx") nxs = list(next());
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--reps") reps = std::atoi(next().c_str());
    else if (a == "--seed") seed = uint32_t(std::atoi(next().c_str()));
    else if (a == "--no-check") check = false;
    else if (a == "--no-time") timing = false;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  if (path.empty()) {
    std::fprintf(stderr, "usage: eklow-cpu-expert-test --gguf FILE [--layers L,..] [--experts E,..] [--threads N] [--reps N] [--nx N,..]\n");
    return 2;
  }
  try {
    gguf::File f = gguf::File::open(path);
    cpu::Pool pool(threads);
    const float beta = 4.0f, lb = 25.0f;
    bool all_ok = true;
    std::printf("threads %d\n", threads);
    if (check) {
      // NVFP4 on random bytes (every code, every scale byte including 0, denormals and 0x7f), activations with outliers:
      // the GEMV against the dequantized rows and the int8-rounded activations, at nx = 1, 3, 4 and 5.
      const int64_t K = 2304, N = 96;
      std::mt19937 r2(seed);
      std::vector<uint8_t> w(size_t(N) * size_t(K / 64) * 36);
      for (auto& b : w) b = uint8_t(r2());
      std::vector<float> xr(size_t(5) * size_t(K));
      std::normal_distribution<float> nd2(0.0f, 1.0f);
      for (auto& v : xr) v = nd2(r2) * (r2() % 50 == 0 ? 40.0f : 1.0f);
      std::vector<cpu::NvBlocks> xq(5);
      std::vector<const cpu::NvBlocks*> xqp(5, nullptr);
      std::vector<float> y(size_t(5) * size_t(N), 0.0f), row(size_t(K), 0.0f), xf(size_t(K), 0.0f);
      std::vector<float*> yp(5, nullptr);
      for (int j = 0; j < 5; ++j) {
        cpu::nv_quantize_into(&xr[size_t(j) * size_t(K)], K, xq[size_t(j)]);
        xqp[size_t(j)] = &xq[size_t(j)];
        yp[size_t(j)] = &y[size_t(j) * size_t(N)];
      }
      double worst = 0;
      for (int nx : {1, 3, 4, 5}) {
        cpu::gemv_nvfp4_multi(w.data(), K, 0, N, xqp.data(), nx, yp.data());
        for (int j = 0; j < nx; ++j) {
          fake_quant(&xr[size_t(j) * size_t(K)], K, 16, xf.data());
          std::vector<float> rf(size_t(N), 0.0f), got(yp[size_t(j)], yp[size_t(j)] + N);
          for (int64_t i = 0; i < N; ++i) {
            dequant_row(gguf::Type::NVFP4, &w[size_t(i) * size_t(K / 64) * 36], K, row.data());
            double s = 0;
            for (int64_t t = 0; t < K; ++t) s += double(row[size_t(t)]) * xf[size_t(t)];
            rf[size_t(i)] = float(s);
          }
          worst = std::max(worst, rel_l2(got.data(), rf.data(), N));
        }
      }
      const bool ok = worst <= 1e-5;
      all_ok &= ok;
      std::printf("random NVFP4 bytes, %lld rows: max rel L2 against the dequantized reference %.2e  %s\n", (long long)N, worst, ok ? "ok" : "FAIL");
    }
    for (int64_t L : layers) {
      const std::string b = "blk." + std::to_string(L) + ".";
      const gguf::Tensor &tg = need(f, b + "ffn_gate_up_exps.weight"), &td = need(f, b + "ffn_down_exps.weight");
      const gguf::Tensor& tn = need(f, b + "ffn_norm.weight");
      const gguf::Tensor* sg = f.tensor(b + "ffn_gate_up_exps.scale");
      const gguf::Tensor* sd = f.tensor(b + "ffn_down_exps.scale");
      const int64_t E = int64_t(tg.ne[0]), F = int64_t(tg.ne[1]) / 2, NE = int64_t(tg.ne[2]);
      const uint64_t gstride = tg.nbytes / uint64_t(NE), dstride = td.nbytes / uint64_t(NE);
      const QType gt = qtype(tg.type), dt = qtype(td.type);
      const int bg = gt == QType::NVFP4 ? 16 : 32, bd = dt == QType::NVFP4 ? 16 : 32;
      std::printf("\nlayer %lld: gate_up %s, down %s, n_embd %lld, n_ff %lld, %lld experts of %.2f MB\n", (long long)L,
                  gguf::type_name(tg.type), gguf::type_name(td.type), (long long)E, (long long)F, (long long)NE,
                  double(gstride + dstride) / 1e6);
      const double ebytes = double(gstride + dstride);
      auto ref_of = [&](int64_t ex) {
        cpu::ExpertRef r;
        r.gu_type = gt;
        r.dn_type = dt;
        r.gu = tg.data + uint64_t(ex) * gstride;
        r.dn = td.data + uint64_t(ex) * dstride;
        r.gu_scale = sg ? reinterpret_cast<const float*>(sg->data)[ex] : 1.0f;
        r.dn_scale = sd ? reinterpret_cast<const float*>(sd->data)[ex] : 1.0f;
        return r;
      };

      // inputs like the model's
      const int NXMAX = int(std::max<int64_t>(256, *std::max_element(nxs.begin(), nxs.end())));
      std::mt19937 rng(seed + uint32_t(L));
      std::normal_distribution<float> nd(0.0f, 1.0f);
      std::vector<float> xs(size_t(NXMAX) * size_t(E));
      const float* norm = reinterpret_cast<const float*>(tn.data);
      for (int j = 0; j < NXMAX; ++j) {
        float* x = &xs[size_t(j) * size_t(E)];
        double ss = 0;
        for (int64_t i = 0; i < E; ++i) ss += double(x[i] = nd(rng)) * x[i];
        const float inv = 1.0f / std::sqrt(float(ss / double(E)) + 1e-6f);
        for (int64_t i = 0; i < E; ++i) x[i] = x[i] * inv * norm[i];
      }

      if (check) {
        const int NC = 8;
        for (int64_t ex : experts) {
          Dense d;
          const cpu::ExpertRef r = ref_of(ex);
          d.gu.resize(size_t(2 * F * E));
          d.dn.resize(size_t(E * F));
          for (int64_t i = 0; i < 2 * F; ++i) dequant_row(tg.type, r.gu + uint64_t(i) * (gstride / uint64_t(2 * F)), E, &d.gu[size_t(i * E)]);
          for (int64_t i = 0; i < E; ++i) dequant_row(td.type, r.dn + uint64_t(i) * (dstride / uint64_t(E)), F, &d.dn[size_t(i * F)]);
          std::vector<float> got(size_t(NC) * size_t(E)), rf(size_t(E), 0.0f), rq(size_t(E), 0.0f);
          std::vector<const float*> xp(NC, nullptr);
          std::vector<float*> yp(NC, nullptr);
          for (int j = 0; j < NC; ++j) xp[size_t(j)] = &xs[size_t(j) * size_t(E)], yp[size_t(j)] = &got[size_t(j) * size_t(E)];
          for (int nx : {1, NC}) {
            cpu::expert_ffn(r, E, F, xp.data(), nx, yp.data(), beta, lb, pool);
            std::vector<double> e32, eq;
            for (int j = 0; j < nx; ++j) {
              ref_expert(d, E, F, xp[size_t(j)], false, bg, bd, r.gu_scale, r.dn_scale, beta, lb, rf.data());
              ref_expert(d, E, F, xp[size_t(j)], true, bg, bd, r.gu_scale, r.dn_scale, beta, lb, rq.data());
              e32.push_back(rel_l2(yp[size_t(j)], rf.data(), E));
              eq.push_back(rel_l2(yp[size_t(j)], rq.data(), E));
            }
            const double m32 = *std::max_element(e32.begin(), e32.end()), mq = *std::max_element(eq.begin(), eq.end());
            const bool ok = std::isfinite(m32) && std::isfinite(mq) && m32 <= 2e-2 && mq <= 1e-4;
            all_ok &= ok;
            std::printf("  check expert %3lld nx %d: max rel L2 vs fp32 %.2e (median %.2e), vs int8-rounded reference %.2e  %s\n",
                        (long long)ex, nx, m32, median(e32), mq, ok ? "ok" : "FAIL");
          }
        }
        // the multi-expert pass equals the one-at-a-time calls bit for bit
        {
          std::vector<cpu::ExpertRef> es;
          for (int64_t ex : experts) es.push_back(ref_of(ex));
          const int ne = int(es.size());
          std::vector<float> a(size_t(ne) * size_t(E)), c(size_t(ne) * size_t(E));
          std::vector<float*> ya, yc;
          for (int i = 0; i < ne; ++i) ya.push_back(&a[size_t(i) * size_t(E)]), yc.push_back(&c[size_t(i) * size_t(E)]);
          const float* xp1[1] = {xs.data()};
          cpu::experts_ffn(es.data(), ne, E, F, xp1, 1, ya.data(), beta, lb, pool);
          for (int i = 0; i < ne; ++i) cpu::expert_ffn(es[size_t(i)], E, F, xp1, 1, &yc[size_t(i)], beta, lb, pool);
          const bool same = std::memcmp(a.data(), c.data(), a.size() * 4) == 0;
          all_ok &= same;
          std::printf("  experts_ffn of %d experts against one call each: %s\n", ne, same ? "identical" : "DIFFERS");
        }
      }

      if (timing) {
        // warm the layer's pages (the mapping), then rotate over the experts so the weights come from RAM
        {
          volatile uint8_t sink = 0;
          for (uint64_t o = 0; o < tg.nbytes; o += 4096) sink = sink + tg.data[o];
          for (uint64_t o = 0; o < td.nbytes; o += 4096) sink = sink + td.data[o];
        }
        std::vector<float> out(size_t(8) * size_t(NXMAX) * size_t(E));
        auto run = [&](int nx, int ne, bool together, int rep) {
          std::vector<cpu::ExpertRef> es;
          for (int i = 0; i < ne; ++i) es.push_back(ref_of(std::getenv("EKLOW_TEST_HOT") ? int64_t(i) : (int64_t(rep) * 37 + i * 13 + 5) % NE));
          std::vector<const float*> xp(size_t(nx), nullptr);
          for (int j = 0; j < nx; ++j) xp[size_t(j)] = &xs[size_t(j) * size_t(E)];
          std::vector<float*> yp(size_t(ne) * size_t(nx), nullptr);
          for (size_t i = 0; i < yp.size(); ++i) yp[i] = &out[i * size_t(E)];
          const double t0 = now_us();
          if (together) cpu::experts_ffn(es.data(), ne, E, F, xp.data(), nx, yp.data(), beta, lb, pool);
          else
            for (int i = 0; i < ne; ++i) cpu::expert_ffn(es[size_t(i)], E, F, xp.data(), nx, &yp[size_t(i) * size_t(nx)], beta, lb, pool);
          return now_us() - t0;
        };
        auto bench = [&](const char* name, int nx, int ne, bool together) {
          for (int i = 0; i < 4; ++i) run(nx, ne, together, 1000 + i);
          std::vector<double> t;
          const int n = nx >= 64 ? std::max(10, reps / 4) : reps;
          for (int i = 0; i < n; ++i) t.push_back(run(nx, ne, together, i));
          const double us = median(t), best = *std::min_element(t.begin(), t.end());
          std::printf("  time %-26s nx %3d x %d expert%s: median %8.1f us (best %.1f)  %6.1f GB/s of weights  %7.2f us/token\n", name, nx,
                      ne, ne > 1 ? "s" : "", us, best, ebytes * ne / (us * 1e-6) / 1e9, us / nx / ne);
        };
        bench("one expert", 1, 1, false);
        bench("8 experts one after another", 1, 8, false);
        bench("8 experts in one pass", 1, 8, true);
        bench("2 experts in one pass", 1, 2, true);
        for (int ne : {1, 2, 8}) bench("two tokens, one pass", 2, ne, true);
        for (int64_t nx : nxs) bench("one expert, many inputs", int(nx), 1, false);
      }
    }
    std::printf("\n%s\n", all_ok ? "ALL OK" : "FAILED");
    return all_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
