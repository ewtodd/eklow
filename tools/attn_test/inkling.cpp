// eklow-attn-test --inkling: Inkling 2 Mini's biased attention (attn::attend with a Bias, head 128, 24 query heads over 8
// K/V heads) against ref::inkling::attention's arithmetic, and its speed against the plain kernel of the same shape.
//
//   eklow-attn-test --inkling [--gguf FILE] [--len N] [--chunk N] [--samples N] [--time-ctx N,...] [--pf N] [--only sliding|full]
//                   [--no-check] [--no-speed] [--attribute]
//
// Correctness: two slots hold different random sequences of --len positions (default 33,000) in the model's shapes: Q and K
// clipped to norms up to 22.6 and drawn from a few shared directions, so that a fraction of the keys scores near 40 against a
// query (attention is sharp, as the model's), R of norm up to 8 x 1.14, V RMS-normed, the sinks a few logits, the table the
// GGUF's (blk.0 for the sliding layers, blk.5 for the full ones). They are fed in steps of --chunk tokens (2,048) mixed with
// chunks of 1, 7 and 300, then decode steps of one and of two tokens a slot, through attn::write and the Bias overload of
// attend, the sliding layers' rings wrapping past the window many times. Each step's rows (every token of a short step, else
// --samples of them, always the first and last) are compared with a copy of ref::inkling::attention's loop run on the card over
// the sequence's contiguous fp16-rounded K/V (checked against ref::inkling::attention itself at the start): relative L2 per
// (token, head) row,  --attribute also compares with the reference run on the kernel's own fp16 operands (Q x scale, R, the
// table), which shows how much of the difference is the operands' rounding and how much the kernel's arithmetic.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/ref_inkling.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

constexpr int64_t kH = 24, kHkv = 8, kD = 128, kRel = 16, kGroup = 3;
constexpr float kScale = 0.08838834764831845f;  // 1 / sqrt(128)

struct Layer {
  const char* name;
  int64_t window;
  const char* table;
};
constexpr Layer kLayers[] = {{"sliding", 1024, "blk.0.attn_rel_bias.weight"}, {"full", 0, "blk.5.attn_rel_bias.weight"}};

template <class T>
T* upload(sycl::queue& q, const std::vector<T>& v) {
  T* p = sycl::malloc_device<T>(v.size(), q);
  q.memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  return p;
}

float half_round(float x) { return float(sycl::half(x)); }

// A slot's whole sequence, as the model's layer would hand it to attention.
struct Seq {
  std::vector<float> Q, K, V, R, S;  // [t][24][128], [t][8][128], [t][8][128], [t][24][16], [t][24]
};

// A row of norm `norm`, in the direction of dirs[c] with some noise of its own.
void directed(std::mt19937& rng, const std::vector<float>& dirs, int c, float noise, float norm, float* x) {
  std::normal_distribution<float> nd(0, 1);
  double n2 = 0;
  for (int64_t d = 0; d < kD; ++d) x[d] = dirs[size_t(c * kD + d)] + noise * nd(rng) / std::sqrt(float(kD));
  for (int64_t d = 0; d < kD; ++d) n2 += double(x[d]) * x[d];
  const float s = norm / float(std::sqrt(n2));
  for (int64_t d = 0; d < kD; ++d) x[d] *= s;
}

Seq make_seq(std::mt19937& rng, int64_t len, const std::vector<float>& dirs) {
  Seq s;
  std::uniform_real_distribution<float> u(0, 1);
  std::uniform_int_distribution<int> cl(0, int(dirs.size() / kD) - 1);
  std::normal_distribution<float> nd(0, 1);
  s.Q.resize(size_t(len * kH * kD)), s.K.resize(size_t(len * kHkv * kD)), s.V.resize(s.K.size());
  s.R.resize(size_t(len * kH * kRel)), s.S.resize(size_t(len * kH));
  for (int64_t r = 0; r < len * kH; ++r) directed(rng, dirs, cl(rng), 0.05f + 0.55f * u(rng), 22.6f * (0.85f + 0.15f * u(rng)), &s.Q[size_t(r * kD)]);
  for (int64_t r = 0; r < len * kHkv; ++r) {
    directed(rng, dirs, cl(rng), 0.05f + 0.55f * u(rng), 22.6f * (0.9f + 0.1f * u(rng)), &s.K[size_t(r * kD)]);
    // V: RMS-normed
    float* v = &s.V[size_t(r * kD)];
    double ss = 0;
    for (int64_t d = 0; d < kD; ++d) ss += double(v[d] = nd(rng)) * v[d];
    const float inv = 1.0f / float(std::sqrt(ss / kD + 1e-6));
    for (int64_t d = 0; d < kD; ++d) v[d] *= inv;
  }
  for (int64_t r = 0; r < len * kH; ++r) {
    float* x = &s.R[size_t(r * kRel)];
    double n2 = 0;
    for (int64_t k = 0; k < kRel; ++k) n2 += double(x[k] = nd(rng)) * x[k];
    const float sc = 8.0f * 1.14f * (0.2f + 0.8f * u(rng)) / float(std::sqrt(n2));
    for (int64_t k = 0; k < kRel; ++k) x[k] *= sc;
    s.S[size_t(r)] = 3.0f + 4.0f * nd(rng);
  }
  // The cache holds K and V in fp16: the reference sees what the kernel reads.
  for (auto& x : s.K) x = half_round(x);
  for (auto& x : s.V) x = half_round(x);
  return s;
}

struct Dev {  // a sequence on the card
  float *Q, *K, *V, *R, *S;
};
Dev to_dev(sycl::queue& q, const Seq& s) { return {upload(q, s.Q), upload(q, s.K), upload(q, s.V), upload(q, s.R), upload(q, s.S)}; }
void free_dev(sycl::queue& q, Dev& d) {
  for (float* p : {d.Q, d.K, d.V, d.R, d.S}) sycl::free(p, q);
}

// ref::inkling::attention's loop for the tokens ts (n of them) of one sequence, a work-item a (token, head); maxs: the largest score of
// each row.
void ref_rows(sycl::queue& q, const Dev& d, const float* table, int64_t extent, int64_t window, const int32_t* ts, int64_t n, float* out,
              float* maxs) {
  {
    q.parallel_for(sycl::range<1>(size_t(n * kH)), [=](sycl::id<1> id) {
       const int64_t i = int64_t(id[0]) / kH, t = ts[i], h = int64_t(id[0]) % kH, hk = h / kGroup;
       const float* qv = d.Q + (t * kH + h) * kD;
       const float* rv = d.R + (t * kH + h) * kRel;
       float* o = out + (i * kH + h) * kD;
       for (int64_t e = 0; e < kD; ++e) o[e] = 0;
       float m = d.S[t * kH + h], l = 1.0f, smax = -1e30f;
       const int64_t j0 = window > 0 ? sycl::max(int64_t(0), t - window + 1) : 0;
       for (int64_t j = j0; j <= t; ++j) {
         const float* kv = d.K + (j * kHkv + hk) * kD;
         float s = 0;
         for (int64_t e = 0; e < kD; ++e) s += qv[e] * kv[e];
         s *= kScale;
         const int64_t dist = t - j;
         if (dist < extent) {
           const float* tb = table + dist * kRel;
           float b = 0;
           for (int64_t k = 0; k < kRel; ++k) b += rv[k] * tb[k];
           s += b;
         }
         smax = sycl::fmax(smax, s);
         const float* vv = d.V + (j * kHkv + hk) * kD;
         if (s > m) {
           const float c = sycl::exp(m - s);
           l = l * c + 1.0f;
           for (int64_t e = 0; e < kD; ++e) o[e] = o[e] * c + vv[e];
           m = s;
         } else {
           const float p = sycl::exp(s - m);
           l += p;
           for (int64_t e = 0; e < kD; ++e) o[e] += p * vv[e];
         }
       }
       for (int64_t e = 0; e < kD; ++e) o[e] /= l;
       maxs[i * kH + h] = smax;
     });
  }
}

struct Stats {
  std::vector<double> rel;
  void add(double r) { rel.push_back(r); }
  double q(double f) {
    if (rel.empty()) return 0;
    std::sort(rel.begin(), rel.end());
    return rel[std::min(rel.size() - 1, size_t(f * double(rel.size())))];
  }
};

struct StepPlan {
  int64_t a, b;
};

}  // namespace

int inkling_main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::string gguf_path = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/models/inkling/Inkling-2-Mini-UD-Q4_K_M.gguf", only;
  int64_t len = 33000, chunk = 2048, samples = 64, pf = 2048;
  std::vector<int64_t> time_ctx = {8192, 32768};
  bool no_check = false, no_speed = false, attribute = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--inkling") continue;
    else if (a == "--gguf") gguf_path = next();
    else if (a == "--len") len = std::atoll(next().c_str());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--samples") samples = std::atoll(next().c_str());
    else if (a == "--pf") pf = std::atoll(next().c_str());
    else if (a == "--time-ctx") {
      time_ctx.clear();
      std::stringstream ss(next());
      for (std::string p; std::getline(ss, p, ',');)
        if (!p.empty()) time_ctx.push_back(std::atoll(p.c_str()));
    } else if (a == "--only") only = next();
    else if (a == "--no-check") no_check = true;
    else if (a == "--no-speed") no_speed = true;
    else if (a == "--attribute") attribute = true;
    else {
      std::fprintf(stderr, "usage: eklow-attn-test --inkling [--gguf FILE] [--len N] [--chunk N] [--samples N] [--time-ctx N,...] [--pf N] [--only sliding|full] [--no-check] [--no-speed] [--attribute]\n");
      return 2;
    }
  }
  try {
    sycl::queue q(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    std::printf("device %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    gguf::File file = gguf::File::open(gguf_path);
    std::mt19937 rng(23);
    bool ok = true;
    // Directions shared by queries and keys: a query scores ~40 against the keys of its direction.
    std::vector<float> dirs(16 * kD);
    {
      std::normal_distribution<float> nd(0, 1);
      for (int c = 0; c < 16; ++c) {
        double n2 = 0;
        for (int64_t d = 0; d < kD; ++d) n2 += double(dirs[size_t(c * kD + d)] = nd(rng)) * dirs[size_t(c * kD + d)];
        for (int64_t d = 0; d < kD; ++d) dirs[size_t(c * kD + d)] /= float(std::sqrt(n2));
      }
    }
    for (const Layer& ly : kLayers) {
      if (!only.empty() && only != ly.name) continue;
      const gguf::Tensor* tt = file.tensor(ly.table);
      if (!tt || tt->type != gguf::Type::F32 || tt->ne.size() != 2 || tt->ne[0] != kRel) throw std::runtime_error(std::string("unexpected ") + ly.table);
      const int64_t extent = int64_t(tt->ne[1]);
      std::vector<float> table(size_t(extent * kRel));
      std::memcpy(table.data(), tt->data, table.size() * 4);
      double tmax = 0, tn = 0;
      for (float x : table) tmax = std::max(tmax, double(std::abs(x))), tn += double(x) * x;
      std::printf("%s: table %s, extent %lld, max |x| %.3g, rms %.3g\n", ly.name, ly.table, (long long)extent, tmax, std::sqrt(tn / double(table.size())));
      float* dtable = upload(q, table);
      sycl::half* dtiles_table = sycl::malloc_device<sycl::half>(attn::bias_table_halves(extent), q);
      attn::bias_table(q, dtable, extent, dtiles_table);
      q.wait();

      if (!no_check) {
        // The copy of the reference's loop against ref::inkling::attention itself.
        {
          const int64_t n = 2600;
          Seq s = make_seq(rng, n, dirs);
          Dev d = to_dev(q, s);
          float *a = sycl::malloc_device<float>(size_t(n * kH * kD), q), *b = sycl::malloc_device<float>(size_t(n * kH * kD), q), *mx = sycl::malloc_device<float>(size_t(n * kH), q);
          std::vector<int32_t> ts(static_cast<size_t>(n));
          std::iota(ts.begin(), ts.end(), 0);
          int32_t* dts = upload(q, ts);
          ref::inkling::attention(q, d.Q, d.K, d.V, d.R, dtable, d.S, n, kH, kHkv, kD, kRel, extent, ly.window, kScale, a);
          ref_rows(q, d, dtable, extent, ly.window, dts, n, b, mx);
          std::vector<float> ha(size_t(n * kH * kD)), hb(ha.size());
          q.memcpy(ha.data(), a, ha.size() * 4);
          q.memcpy(hb.data(), b, hb.size() * 4).wait();
          double md = 0;
          for (size_t i = 0; i < ha.size(); ++i) md = std::max(md, double(std::abs(ha[i] - hb[i])));
          std::printf("%s: the reference's loop against ref::inkling::attention (%lld tokens): max abs %.3g\n", ly.name, (long long)n, md);
          if (md > 1e-5) ok = false;
          free_dev(q, d);
          for (void* p : {static_cast<void*>(a), static_cast<void*>(b), static_cast<void*>(mx), static_cast<void*>(dts)}) sycl::free(p, q);
        }

        constexpr int kSlots = 2;
        std::vector<Seq> seq;
        std::vector<Dev> dev;
        for (int s = 0; s < kSlots; ++s) seq.push_back(make_seq(rng, len, dirs)), dev.push_back(to_dev(q, seq.back()));
        // Variants of the reference on the kernel's fp16 operands, for --attribute: Q x scale rounded (Qr), and also R and the table.
        std::vector<Dev> dev_q, dev_qr;
        float* dtable_r = nullptr;
        if (attribute) {
          std::vector<float> tr = table;
          for (auto& x : tr) x = half_round(x);
          dtable_r = upload(q, tr);
          for (int s = 0; s < kSlots; ++s) {
            Seq a = seq[size_t(s)];
            for (auto& x : a.Q) x = half_round(x * kScale) / kScale;
            dev_q.push_back(to_dev(q, a));
            for (auto& x : a.R) x = half_round(x);
            dev_qr.push_back(to_dev(q, a));
          }
        }

        const int64_t n_dec1 = 16, n_dec2 = 8, pre = len - n_dec1 - 2 * n_dec2;
        struct Step {
          int64_t a, b;
          const char* cls;
        };
        std::vector<Step> sched;
        {
          const int64_t sizes[] = {chunk, chunk, 1, 7, 300};
          int64_t a = 0;
          for (size_t i = 0; a < pre; ++i) {
            const int64_t n = std::min(sizes[i % 5], pre - a);
            sched.push_back({a, a + n, n >= 300 ? "prefill" : "short"});
            a += n;
          }
          for (int64_t i = 0; i < n_dec1; ++i) sched.push_back({pre + i, pre + i + 1, "decode1"});
          for (int64_t i = 0; i < n_dec2; ++i) sched.push_back({pre + n_dec1 + 2 * i, pre + n_dec1 + 2 * i + 2, "decode2"});
        }
        attn::Cache cache{};
        std::vector<int32_t> table_host;
        int32_t* table_dev = nullptr;
        {
          if (ly.window > 0) {
            cache.pages = (ly.window + chunk + attn::kPage - 1) / attn::kPage;
          } else {
            cache.pages = (len + attn::kPage - 1) / attn::kPage;
            table_host.resize(size_t(kSlots * cache.pages));
            for (int64_t s = 0; s < kSlots; ++s)
              for (int64_t p = 0; p < cache.pages; ++p) table_host[size_t(s * cache.pages + p)] = int32_t(p * kSlots + (kSlots - 1 - s));
            table_dev = upload(q, table_host);
          }
          const size_t elems = size_t(kSlots * cache.pages * kHkv * attn::kPage * kD);
          cache.k = sycl::malloc_device(elems * 2, q);
          cache.v = sycl::malloc_device(elems * 2, q);
          cache.table = table_dev;
          cache.window = ly.window;
          cache.n_head_kv = kHkv;
          cache.head_dim = kD;
        }
        attn::Scratch scratch;
        const int64_t max_tok = kSlots * chunk;
        float *dq = sycl::malloc_device<float>(size_t(max_tok * kH * kD), q), *dk = sycl::malloc_device<float>(size_t(max_tok * kHkv * kD), q),
              *dv = sycl::malloc_device<float>(size_t(max_tok * kHkv * kD), q), *dout = sycl::malloc_device<float>(size_t(max_tok * kH * kD), q),
              *dr = sycl::malloc_device<float>(size_t(max_tok * kH * kRel), q), *dsink = sycl::malloc_device<float>(size_t(max_tok * kH), q);
        int32_t *dslot = sycl::malloc_device<int32_t>(size_t(max_tok), q), *dpos = sycl::malloc_device<int32_t>(size_t(max_tok), q);
        int32_t* dtl = sycl::malloc_device<int32_t>(size_t(6 * kHkv * (max_tok * kH / kHkv / 8 + kSlots)), q);
        const int64_t max_rows = kSlots * samples + 64;
        int32_t* dts = sycl::malloc_device<int32_t>(size_t(std::max<int64_t>(max_rows, max_tok)), q);
        float *rout = sycl::malloc_device<float>(size_t(std::max<int64_t>(max_rows, max_tok) * kH * kD), q), *rmax = sycl::malloc_device<float>(size_t(std::max<int64_t>(max_rows, max_tok) * kH), q);
        std::map<std::string, Stats> by_class, by_class_q, by_class_qr;
        Stats all;
        double worst = 0;
        char worst_at[400] = "";
        double max_score = 0;
        int64_t sharp_rows = 0, rows_n = 0;
        std::mt19937 pick(5);
        int step_no = 0;
        for (const Step& sp : sched) {
          if (step_no % 10 == 0) std::fprintf(stderr, "  %s step %d of %zu\n", ly.name, step_no, sched.size());
          const int64_t n = sp.b - sp.a, T = kSlots * n;
          attn::TileList tl(kD);
          std::vector<int32_t> slot, pos;
          std::vector<float> Q, K, V, R, S;
          for (int s = 0; s < kSlots; ++s) {
            const int32_t base = int32_t(pos.size());
            for (int64_t t = sp.a; t < sp.b; ++t) slot.push_back(s), pos.push_back(int32_t(t));
            const Seq& sq = seq[size_t(s)];
            Q.insert(Q.end(), sq.Q.begin() + sp.a * kH * kD, sq.Q.begin() + sp.b * kH * kD);
            K.insert(K.end(), sq.K.begin() + sp.a * kHkv * kD, sq.K.begin() + sp.b * kHkv * kD);
            V.insert(V.end(), sq.V.begin() + sp.a * kHkv * kD, sq.V.begin() + sp.b * kHkv * kD);
            R.insert(R.end(), sq.R.begin() + sp.a * kH * kRel, sq.R.begin() + sp.b * kH * kRel);
            S.insert(S.end(), sq.S.begin() + sp.a * kH, sq.S.begin() + sp.b * kH);
            tl.add_chunk(base, n, sp.b - 1, kGroup, kHkv, ly.window);
          }
          q.memcpy(dq, Q.data(), Q.size() * 4);
          q.memcpy(dk, K.data(), K.size() * 4);
          q.memcpy(dv, V.data(), V.size() * 4);
          q.memcpy(dr, R.data(), R.size() * 4);
          q.memcpy(dsink, S.data(), S.size() * 4);
          q.memcpy(dslot, slot.data(), size_t(T) * 4);
          q.memcpy(dpos, pos.data(), size_t(T) * 4);
          const std::vector<int32_t> packed = tl.packed();
          q.memcpy(dtl, packed.data(), packed.size() * 4);
          const attn::Tokens tok{dslot, dpos, T};
          attn::write(q, dk, dv, tok, cache);
          attn::Bias bias;
          bias.R = dr, bias.table = dtable, bias.n_rel = kRel, bias.extent = extent, bias.sink = dsink;
          if (step_no++ % 2) bias.table_tiles = dtiles_table;  // the engine's prepared table on every other step
          attn::attend(q, dq, tok, tl, dtl, cache, kH, kScale, bias, dout, scratch);
          std::vector<float> got(size_t(T * kH * kD));
          q.memcpy(got.data(), dout, got.size() * 4).wait();
          // The rows to check: every token of a short step, else the first, the last and random ones.
          for (int s = 0; s < kSlots; ++s) {
            std::vector<int32_t> ts;
            if (n <= samples) {
              for (int64_t t = sp.a; t < sp.b; ++t) ts.push_back(int32_t(t));
            } else {
              ts = {int32_t(sp.a), int32_t(sp.b - 1)};
              std::uniform_int_distribution<int64_t> ud(sp.a, sp.b - 1);
              while (int64_t(ts.size()) < samples) ts.push_back(int32_t(ud(pick)));
            }
            q.memcpy(dts, ts.data(), ts.size() * 4);
            std::vector<float> want(ts.size() * kH * kD), wmax(ts.size() * kH), wq(want.size()), wqr(want.size());
            ref_rows(q, dev[size_t(s)], dtable, extent, ly.window, dts, int64_t(ts.size()), rout, rmax);
            q.memcpy(want.data(), rout, want.size() * 4);
            q.memcpy(wmax.data(), rmax, wmax.size() * 4).wait();
            if (attribute) {
              ref_rows(q, dev_q[size_t(s)], dtable, extent, ly.window, dts, int64_t(ts.size()), rout, rmax);
              q.memcpy(wq.data(), rout, wq.size() * 4).wait();
              ref_rows(q, dev_qr[size_t(s)], dtable_r, extent, ly.window, dts, int64_t(ts.size()), rout, rmax);
              q.memcpy(wqr.data(), rout, wqr.size() * 4).wait();
            }
            for (size_t i = 0; i < ts.size(); ++i)
              for (int64_t h = 0; h < kH; ++h) {
                const float* g = &got[size_t(((int64_t(s) * n + (ts[i] - sp.a)) * kH + h) * kD)];
                auto rel = [&](const std::vector<float>& w) {
                  double e2 = 0, r2 = 0;
                  const float* r = &w[(i * kH + size_t(h)) * kD];
                  for (int64_t e = 0; e < kD; ++e) e2 += double(g[e] - r[e]) * (g[e] - r[e]), r2 += double(r[e]) * r[e];
                  return std::sqrt(e2) / (std::sqrt(r2) + 1e-3);  // a floor: a row the sink swallowed has ~0 output
                };
                const double r = rel(want);
                if (r > worst) worst = r, std::snprintf(worst_at, sizeof worst_at, "slot %d token %d head %lld, step [%lld, %lld) of the chunk of %lld; got %.4g %.4g %.4g, want %.4g %.4g %.4g, max score %.3f, sink %.3f", s, int(ts[i]), (long long)h, (long long)sp.a, (long long)sp.b, (long long)n, g[0], g[1], g[2], want[(i * kH + size_t(h)) * kD], want[(i * kH + size_t(h)) * kD + 1], want[(i * kH + size_t(h)) * kD + 2], wmax[i * kH + size_t(h)], seq[size_t(s)].S[size_t(ts[i] * kH + h)]);
                by_class[sp.cls].add(r), all.add(r);
                if (attribute) by_class_q[sp.cls].add(rel(wq)), by_class_qr[sp.cls].add(rel(wqr));
                const double ms = wmax[i * kH + size_t(h)];
                max_score = std::max(max_score, ms);
                sharp_rows += ms > 40, ++rows_n;
              }
          }
        }
        std::printf("%s: %zu steps, 2 slots, %lld positions, %lld rows checked; largest score %.1f, %.1f%% of rows reach 40\n", ly.name, sched.size(),
                    (long long)len, (long long)rows_n, max_score, 100.0 * double(sharp_rows) / double(rows_n));
        for (auto& [cls, st] : by_class)
          std::printf("  %-8s %7zu rows: rel L2 median %.2e, p99 %.2e, max %.2e%s\n", cls.c_str(), st.rel.size(), st.q(0.5), st.q(0.99), st.q(1.0), "");
        if (attribute)
          for (auto& [cls, st] : by_class_q)
            std::printf("  vs ref on fp16 Q: %-8s median %.2e, max %.2e; on fp16 Q, R and table: median %.2e, max %.2e\n", cls.c_str(), st.q(0.5), st.q(1.0),
                        by_class_qr[cls].q(0.5), by_class_qr[cls].q(1.0));
        std::printf("  worst row: %.3g at %s\n", worst, worst_at);
        const bool pass = all.q(0.5) <= 2e-3 && all.q(1.0) <= 1e-2;
        std::printf("%s: median %.2e, max %.2e %s\n", ly.name, all.q(0.5), all.q(1.0), pass ? "ok" : "FAIL");
        ok = ok && pass;
        for (void* p : {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv), static_cast<void*>(dout), static_cast<void*>(dr),
                        static_cast<void*>(dsink), static_cast<void*>(dslot), static_cast<void*>(dpos), static_cast<void*>(dtl), static_cast<void*>(dts),
                        static_cast<void*>(rout), static_cast<void*>(rmax), cache.k, cache.v, static_cast<void*>(table_dev), static_cast<void*>(scratch.q),
                        static_cast<void*>(scratch.part), static_cast<void*>(scratch.part2), static_cast<void*>(scratch.rt), static_cast<void*>(scratch.tv)})
          if (p) sycl::free(p, q);
        for (auto& d : dev) free_dev(q, d);
        for (auto& d : dev_q) free_dev(q, d);
        for (auto& d : dev_qr) free_dev(q, d);
        if (dtable_r) sycl::free(dtable_r, q);
      }

      if (!no_speed)
        for (int64_t ctx : time_ctx) {
          // One slot's cache filled to ctx; one decode token and one chunk of pf tokens ending there, biased and plain.
          const int64_t chunk_pf = std::min(pf, ctx);
          attn::Cache cache{};
          std::vector<int32_t> th;
          int32_t* td = nullptr;
          if (ly.window > 0) cache.pages = (ly.window + chunk_pf + attn::kPage - 1) / attn::kPage;
          else {
            cache.pages = (ctx + 1 + attn::kPage - 1) / attn::kPage;
            th.resize(size_t(cache.pages));
            std::iota(th.begin(), th.end(), 0);
            td = upload(q, th);
          }
          const size_t elems = size_t(cache.pages * kHkv * attn::kPage * kD);
          cache.k = sycl::malloc_device(elems * 2, q), cache.v = sycl::malloc_device(elems * 2, q);
          cache.table = td, cache.window = ly.window, cache.n_head_kv = kHkv, cache.head_dim = kD;
          const int64_t nt = chunk_pf;
          std::mt19937 r2(3);
          std::vector<float> hq(size_t(nt * kH * kD)), hk(size_t(nt * kHkv * kD)), hv(hk.size());
          {
            Seq s = make_seq(r2, nt, dirs);
            hq = s.Q, hk = s.K, hv = s.V;
          }
          Seq sr = make_seq(r2, nt, dirs);
          float *dq = upload(q, hq), *dk = upload(q, hk), *dv = upload(q, hv), *dr = upload(q, sr.R), *dsink = upload(q, sr.S);
          float* dout = sycl::malloc_device<float>(size_t(nt * kH * kD), q);
          int32_t *dslot = sycl::malloc_device<int32_t>(size_t(nt), q), *dpos = sycl::malloc_device<int32_t>(size_t(nt), q);
          int32_t* dtl = sycl::malloc_device<int32_t>(size_t(6 * kHkv * (nt * kH / kHkv / 8 + 2)), q);
          attn::Scratch scratch;
          attn::TileList tl(kD);
          auto prepare = [&](int64_t a, int64_t b) {
            std::vector<int32_t> pos, slot(size_t(b - a), 0);
            tl.clear();
            tl.add_chunk(0, b - a, b - 1, kGroup, kHkv, ly.window);
            for (int64_t t = a; t < b; ++t) pos.push_back(int32_t(t));
            q.memcpy(dpos, pos.data(), pos.size() * 4);
            q.memcpy(dslot, slot.data(), slot.size() * 4);
            const std::vector<int32_t> packed = tl.packed();
            q.memcpy(dtl, packed.data(), packed.size() * 4).wait();
            return attn::Tokens{dslot, dpos, b - a};
          };
          // Fill the cache: the same random rows at every position (the speed does not depend on them).
          for (int64_t a = 0; a <= ctx; a += nt) attn::write(q, dk, dv, prepare(a, std::min(ctx + 1, a + nt)), cache);
          q.wait();
          attn::Bias bias;
          bias.R = dr, bias.table = dtable, bias.n_rel = kRel, bias.extent = extent, bias.sink = dsink, bias.table_tiles = dtiles_table;
          // EKLOW_ATTN_ROTATE=N: N copies of the cache, the calls going round them, so that each reads its keys from memory
          // rather than L2 (the engine's 40 sliding layer applications), for the decode timings
          const int n_rot = std::getenv("EKLOW_ATTN_ROTATE") ? std::atoi(std::getenv("EKLOW_ATTN_ROTATE")) : 1;
          std::vector<attn::Cache> caches(1, cache);
          for (int i = 1; i < n_rot; ++i) {
            attn::Cache c2 = cache;
            c2.k = sycl::malloc_device(elems * 2, q), c2.v = sycl::malloc_device(elems * 2, q);
            q.memcpy(c2.k, cache.k, elems * 2);
            q.memcpy(c2.v, cache.v, elems * 2);
            caches.push_back(c2);
          }
          q.wait();
          int turn = 0;
          auto timed = [&](int64_t a, int64_t b, int reps, bool biased, bool prepared = true) {
            const attn::Tokens tok = prepare(a, b);
            attn::Bias bi = bias;
            if (!prepared) bi.table_tiles = nullptr;
            auto go = [&] {
              const attn::Cache& cc = caches[size_t((b - a) <= 2 ? turn++ % n_rot : 0)];
              if (biased) attn::attend(q, dq, tok, tl, dtl, cc, kH, kScale, bi, dout, scratch);
              else attn::attend(q, dq, tok, tl, dtl, cc, kH, kScale, dout, scratch);
            };
            go();
            q.wait();
            if (std::getenv("EKLOW_ATTN_KERNELS") && reps > 10) {  // each kernel of one call: device time, and the gap before it
              std::vector<sycl::event> log;
              attn::g_event_log = &log;
              for (int i = 0; i < 2; ++i) {
                log.clear();
                go();
                q.wait();
              }
              attn::g_event_log = nullptr;
              using namespace sycl::info;
              std::printf("    %s %lld..%lld:", biased ? "biased" : "plain", (long long)a, (long long)b);
              uint64_t prev = 0;
              for (auto& e : log) {
                const uint64_t st = e.get_profiling_info<event_profiling::command_start>(), en = e.get_profiling_info<event_profiling::command_end>();
                std::printf(" [%.1f us%s%.1f]", double(en - st) * 1e-3, prev ? ", gap " : "", prev ? double(st - prev) * 1e-3 : 0.0);
                prev = en;
              }
              std::printf("\n");
            }
            double best = 1e30;  // the best of three runs
            for (int rep3 = 0; rep3 < 3; ++rep3) {
              sycl::event e0 = q.single_task([] {});
              for (int r = 0; r < reps; ++r) go();
              sycl::event e1 = q.single_task([] {});
              e1.wait();
              using namespace sycl::info;
              best = std::min(best, double(e1.get_profiling_info<event_profiling::command_start>() - e0.get_profiling_info<event_profiling::command_end>()) * 1e-9 / reps);
            }
            return best;
          };
          const double d_b = timed(ctx, ctx + 1, 100, true), d_p = timed(ctx, ctx + 1, 100, false), d_u = timed(ctx, ctx + 1, 100, true, false);
          const double d2_b = timed(ctx - 1, ctx + 1, 100, true), d2_p = timed(ctx - 1, ctx + 1, 100, false);
          const double p_b = timed(ctx - nt + 1, ctx + 1, 5, true), p_p = timed(ctx - nt + 1, ctx + 1, 5, false);
          double flops = 0;
          for (int64_t t = ctx - nt + 1; t <= ctx; ++t) flops += 4.0 * double(kH * kD) * double(ly.window > 0 ? std::min(ly.window, t + 1) : t + 1);
          std::printf("%-7s ctx %6lld: decode 1 token biased %7.1f us, plain %7.1f us (table converted per call %7.1f us); 2 tokens %7.1f / %7.1f us; prefill %lld biased %7.2f ms (%5.2f TFLOPS), plain %7.2f ms (%5.2f TFLOPS)\n",
                      ly.name, (long long)ctx, d_b * 1e6, d_p * 1e6, d_u * 1e6, d2_b * 1e6, d2_p * 1e6, (long long)nt, p_b * 1e3, flops / p_b / 1e12, p_p * 1e3,
                      flops / p_p / 1e12);
          for (size_t i = 1; i < caches.size(); ++i) sycl::free(caches[i].k, q), sycl::free(caches[i].v, q);
          for (void* p : {static_cast<void*>(dq), static_cast<void*>(dk), static_cast<void*>(dv), static_cast<void*>(dr), static_cast<void*>(dsink),
                          static_cast<void*>(dout), static_cast<void*>(dslot), static_cast<void*>(dpos), static_cast<void*>(dtl), cache.k, cache.v,
                          static_cast<void*>(td), static_cast<void*>(scratch.q), static_cast<void*>(scratch.part), static_cast<void*>(scratch.part2),
                          static_cast<void*>(scratch.rt), static_cast<void*>(scratch.tv)})
            if (p) sycl::free(p, q);
        }
      sycl::free(dtable, q);
      sycl::free(dtiles_table, q);
    }
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-attn-test --inkling: %s\n", e.what());
    return 1;
  }
}
