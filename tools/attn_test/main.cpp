// eklow-attn-test: the cache attention (kernels/attention.hpp) against ref::attention on random
// Q/K/V, driven as the engine drives it, and its speed at decode and prefill shapes.
//
//   eklow-attn-test [--len N] [--chunk N] [--decode N] [--time-ctx N,...] [--json FILE]
//
// Correctness: two slots hold different random sequences of --len tokens, fed in the same steps as
// prefill chunks of --chunk and then --decode single tokens; every output row is compared with the
// reference over the whole sequence, for Gemma 4's sliding (head 256, 8 K/V heads, ring, window
// 1024) and global (head 512, one K/V head, page table) layers, with fp16 caches and int8 ones
// (sliding8, global8). Speed: the cache filled to each
// --time-ctx, one decode token and one 1024-token prefill chunk ending there, timed on the card.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "kernels/attention.hpp"
#include "kernels/ref.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

constexpr int64_t kHeads = 16;

struct Kind {
  const char* name;
  int64_t head_dim, n_head_kv, window;
  bool int8;
};
constexpr Kind kKinds[] = {{"sliding", 256, 8, 1024, false}, {"global", 512, 1, 0, false},
                           {"sliding8", 256, 8, 1024, true}, {"global8", 512, 1, 0, true}};

std::vector<int64_t> list(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, ',');)
    if (!p.empty()) out.push_back(std::atoll(p.c_str()));
  return out;
}

template <class T>
T* upload(sycl::queue& q, const std::vector<T>& v) {
  T* p = sycl::malloc_device<T>(v.size(), q);
  q.memcpy(p, v.data(), v.size() * sizeof(T)).wait();
  return p;
}

std::vector<float> randn(std::mt19937& rng, size_t n) {
  std::normal_distribution<float> nd(0, 1);
  std::vector<float> v(n);
  for (auto& x : v) x = nd(rng);
  return v;
}

// One layer's cache for `slots` sequences of up to `len` positions. Global pages are handed out
// interleaved between the slots, so a wrong table entry reads another sequence.
struct TestCache {
  attn::Cache c{};
  std::vector<int32_t> table;
  int32_t* table_dev = nullptr;
  TestCache(sycl::queue& q, const Kind& k, int slots, int64_t len, int64_t chunk) {
    int64_t pages;
    if (k.window > 0) {
      c.pages = (k.window + chunk + attn::kPage - 1) / attn::kPage;
      pages = slots * c.pages;
    } else {
      c.pages = (len + attn::kPage - 1) / attn::kPage;
      pages = slots * c.pages;
      table.resize(size_t(pages));
      for (int64_t s = 0; s < slots; ++s)
        for (int64_t p = 0; p < c.pages; ++p) table[size_t(s * c.pages + p)] = int32_t(p * slots + (slots - 1 - s));
      table_dev = upload(q, table);
    }
    const size_t elems = size_t(pages * k.n_head_kv * attn::kPage * k.head_dim);
    c.k = sycl::malloc_device(elems * (k.int8 ? 1 : 2), q);
    c.v = sycl::malloc_device(elems * (k.int8 ? 1 : 2), q);
    if (k.int8) {
      c.ks = sycl::malloc_device<sycl::half>(elems / 16, q);
      c.vs = sycl::malloc_device<sycl::half>(elems / 16, q);
    }
    c.table = table_dev;
    c.window = k.window;
    c.n_head_kv = k.n_head_kv;
    c.head_dim = k.head_dim;
  }
  void free(sycl::queue& q) {
    sycl::free(c.k, q);
    sycl::free(c.v, q);
    if (c.ks) sycl::free(c.ks, q);
    if (c.vs) sycl::free(c.vs, q);
    if (table_dev) sycl::free(table_dev, q);
  }
};

// One step: chunk [a, b) of every slot's sequence. Q/K/V of slot s are rows of qs[s], ks[s], vs[s].
struct Step {
  explicit Step(int64_t head_dim) : tiles(head_dim) {}
  std::vector<int32_t> slot, pos;
  attn::TileList tiles;
  std::vector<float> Q, K, V;
};

Step make_step(const std::vector<std::pair<int64_t, int64_t>>& ranges, const std::vector<std::vector<float>>& qs,
               const std::vector<std::vector<float>>& ks, const std::vector<std::vector<float>>& vs, const Kind& k) {
  Step st(k.head_dim);
  const int64_t D = k.head_dim, Hkv = k.n_head_kv;
  for (size_t s = 0; s < ranges.size(); ++s) {
    auto [a, b] = ranges[s];
    const int32_t base = int32_t(st.pos.size());
    for (int64_t t = a; t < b; ++t) {
      st.slot.push_back(int32_t(s));
      st.pos.push_back(int32_t(t));
    }
    st.Q.insert(st.Q.end(), qs[s].begin() + a * kHeads * D, qs[s].begin() + b * kHeads * D);
    st.K.insert(st.K.end(), ks[s].begin() + a * Hkv * D, ks[s].begin() + b * Hkv * D);
    st.V.insert(st.V.end(), vs[s].begin() + a * Hkv * D, vs[s].begin() + b * Hkv * D);
    st.tiles.add_chunk(base, b - a, b - 1, kHeads / Hkv, Hkv, k.window);
  }
  return st;
}

// GPU time of `f`'s work, from empty marker kernels before and after it.
template <class F>
double gpu_seconds(sycl::queue& q, F&& f) {
  sycl::event e0 = q.single_task([] {});
  f();
  sycl::event e1 = q.single_task([] {});
  e1.wait();
  using namespace sycl::info;
  return double(e1.get_profiling_info<event_profiling::command_start>() - e0.get_profiling_info<event_profiling::command_end>()) * 1e-9;
}

}  // namespace

int main(int argc, char** argv) {
  int64_t len = 3000, chunk = 512, decode = 40;
  std::vector<int64_t> time_ctx = {4096, 16384};
  std::string json_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--len") len = std::atoll(next().c_str());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--decode") decode = std::atoll(next().c_str());
    else if (a == "--time-ctx") time_ctx = list(next());
    else if (a == "--json") json_path = next();
    else {
      std::fprintf(stderr, "usage: eklow-attn-test [--len N] [--chunk N] [--decode N] [--time-ctx N,...] [--json FILE]\n");
      return 2;
    }
  }
  try {
    sycl::queue q(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()});
    std::mt19937 rng(11);
    json::Writer w;
    w.begin_object().kv("device", q.get_device().get_info<sycl::info::device::name>()).kv("len", len).kv("chunk", chunk);
    w.key("check").begin_array();
    bool ok = true;
    for (const Kind& k : kKinds) {
      const int64_t D = k.head_dim, Hkv = k.n_head_kv;
      const float scale = 2.0f / std::sqrt(float(D));
      constexpr int kSlots = 2;
      std::vector<std::vector<float>> qs, ks, vs, ref(kSlots);
      for (int s = 0; s < kSlots; ++s) {
        qs.push_back(randn(rng, size_t(len * kHeads * D)));
        ks.push_back(randn(rng, size_t(len * Hkv * D)));
        vs.push_back(randn(rng, size_t(len * Hkv * D)));
        // The reference sees the fp16-rounded K/V and Q the cache path sees.
        for (auto* v : {&qs[size_t(s)], &ks[size_t(s)], &vs[size_t(s)]})
          for (auto& x : *v) x = float(sycl::half(x));
        float *dq = upload(q, qs[size_t(s)]), *dk = upload(q, ks[size_t(s)]), *dv = upload(q, vs[size_t(s)]);
        float* dout = sycl::malloc_device<float>(size_t(len * kHeads * D), q);
        ref::attention(q, dq, dk, dv, len, kHeads, Hkv, D, k.window, scale, dout);
        ref[size_t(s)].resize(size_t(len * kHeads * D));
        q.memcpy(ref[size_t(s)].data(), dout, ref[size_t(s)].size() * sizeof(float)).wait();
        for (float* p : {dq, dk, dv, dout}) sycl::free(p, q);
      }

      TestCache cache(q, k, kSlots, len, chunk);
      attn::Scratch scratch;
      const int64_t max_tok = kSlots * chunk;
      float *dq = sycl::malloc_device<float>(size_t(max_tok * kHeads * D), q), *dk = sycl::malloc_device<float>(size_t(max_tok * Hkv * D), q),
            *dv = sycl::malloc_device<float>(size_t(max_tok * Hkv * D), q), *dout = sycl::malloc_device<float>(size_t(max_tok * kHeads * D), q);
      int32_t *dslot = sycl::malloc_device<int32_t>(size_t(max_tok), q), *dpos = sycl::malloc_device<int32_t>(size_t(max_tok), q);
      int32_t* dtiles = sycl::malloc_device<int32_t>(size_t(6 * Hkv * (max_tok * kHeads / Hkv / 8 + kSlots)), q);
      double err2 = 0, ref2 = 0, max_abs = 0;
      std::vector<std::pair<int64_t, int64_t>> sched;
      for (int64_t a = 0; a < len - decode; a += chunk) sched.push_back({a, std::min(len - decode, a + chunk)});
      for (int64_t a = len - decode; a < len; ++a) sched.push_back({a, a + 1});
      for (auto [a, b] : sched) {
        Step st = make_step({{a, b}, {a, b}}, qs, ks, vs, k);
        const int64_t T = int64_t(st.pos.size());
        q.memcpy(dq, st.Q.data(), st.Q.size() * 4);
        q.memcpy(dk, st.K.data(), st.K.size() * 4);
        q.memcpy(dv, st.V.data(), st.V.size() * 4);
        q.memcpy(dslot, st.slot.data(), size_t(T) * 4);
        q.memcpy(dpos, st.pos.data(), size_t(T) * 4);
        const std::vector<int32_t> packed = st.tiles.packed();
        q.memcpy(dtiles, packed.data(), packed.size() * 4);
        const attn::Tokens tok{dslot, dpos, T};
        attn::write(q, dk, dv, tok, cache.c);
        attn::attend(q, dq, tok, st.tiles, dtiles, cache.c, kHeads, scale, dout, scratch);
        std::vector<float> got(size_t(T * kHeads * D));
        q.memcpy(got.data(), dout, got.size() * 4).wait();
        for (int64_t i = 0; i < T; ++i) {
          const float* r = ref[size_t(st.slot[size_t(i)])].data() + st.pos[size_t(i)] * kHeads * D;
          for (int64_t e = 0; e < kHeads * D; ++e) {
            const double d = double(got[size_t(i * kHeads * D + e)]) - r[e];
            err2 += d * d;
            ref2 += double(r[e]) * r[e];
            max_abs = std::max(max_abs, std::abs(d));
          }
        }
      }
      const double rel = std::sqrt(err2 / ref2);
      // fp16 probabilities and fp32 accumulation: a few 1e-4 relative; int8 K/V a few 1e-3.
      const bool pass = rel < (k.int8 ? 2e-2 : 2e-3);
      ok = ok && pass;
      std::printf("%-8s %zu steps, 2 slots: rel L2 %.3g, max abs %.3g %s\n", k.name, sched.size(), rel, max_abs, pass ? "ok" : "FAIL");
      w.begin_object().kv("kind", k.name).kv("rel_l2", rel).kv("max_abs", max_abs).kv("pass", pass).end_object();
      for (float* p : {dq, dk, dv, dout}) sycl::free(p, q);
      for (int32_t* p : {dslot, dpos, dtiles}) sycl::free(p, q);
      if (scratch.q) sycl::free(scratch.q, q);
      if (scratch.part) sycl::free(scratch.part, q);
      if (scratch.part2) sycl::free(scratch.part2, q);
      cache.free(q);
    }
    w.end_array().key("speed").begin_array();

    for (const Kind& k : kKinds)
      for (int64_t ctx : time_ctx) {
        const int64_t D = k.head_dim, Hkv = k.n_head_kv, pf = std::min<int64_t>(1024, ctx);
        TestCache cache(q, k, 1, ctx + 1, pf);
        attn::Scratch scratch;
        float *dq = sycl::malloc_device<float>(size_t(pf * kHeads * D), q), *dk = sycl::malloc_device<float>(size_t(pf * Hkv * D), q),
              *dv = sycl::malloc_device<float>(size_t(pf * Hkv * D), q), *dout = sycl::malloc_device<float>(size_t(pf * kHeads * D), q);
        int32_t *dslot = sycl::malloc_device<int32_t>(size_t(pf), q), *dpos = sycl::malloc_device<int32_t>(size_t(pf), q);
        int32_t* dtiles = sycl::malloc_device<int32_t>(size_t(6 * Hkv * (pf * kHeads / Hkv / 8 + 1)), q);
        auto fill = [&](std::vector<float>& v) {
          v = randn(rng, v.size());
        };
        std::vector<float> hq(size_t(pf * kHeads * D)), hk(size_t(pf * Hkv * D)), hv(hk.size());
        fill(hq), fill(hk), fill(hv);
        q.memcpy(dq, hq.data(), hq.size() * 4);
        q.memcpy(dk, hk.data(), hk.size() * 4);
        q.memcpy(dv, hv.data(), hv.size() * 4);
        q.memset(dslot, 0, size_t(pf) * 4).wait();
        // Fill positions 0..ctx with the same random rows (the speed does not depend on them).
        attn::TileList tl(D);
        auto prepare = [&](int64_t a, int64_t b) {
          std::vector<int32_t> pos;
          for (int64_t t = a; t < b; ++t) pos.push_back(int32_t(t));
          tl.clear();
          tl.add_chunk(0, b - a, b - 1, kHeads / Hkv, Hkv, k.window);
          q.memcpy(dpos, pos.data(), pos.size() * 4);
          const std::vector<int32_t> packed = tl.packed();
          q.memcpy(dtiles, packed.data(), packed.size() * 4).wait();
          return attn::Tokens{dslot, dpos, b - a};
        };
        for (int64_t a = 0; a <= ctx; a += pf) attn::write(q, dk, dv, prepare(a, std::min(ctx + 1, a + pf)), cache.c);
        q.wait();
        auto timed = [&](int64_t a, int64_t b, int reps) {
          const attn::Tokens tok = prepare(a, b);
          auto go = [&] { attn::attend(q, dq, tok, tl, dtiles, cache.c, kHeads, 1.0f / std::sqrt(float(D)), dout, scratch); };
          go();  // warm-up, scratch sizes
          q.wait();
          return gpu_seconds(q, [&] {
                   for (int r = 0; r < reps; ++r) go();
                 }) / reps;
        };
        const double t_dec = timed(ctx, ctx + 1, 50);
        const double t_pf = timed(ctx - pf, ctx, 5);
        double flops = 0;
        for (int64_t t = ctx - pf; t < ctx; ++t) flops += 4.0 * double(kHeads * D) * double(k.window > 0 ? std::min(k.window, t + 1) : t + 1);
        const int64_t keys = k.window > 0 ? std::min(k.window, ctx + 1) : ctx + 1;
        const double dec_bytes = double(keys * Hkv * D * 2 * 2);
        std::printf("%-8s ctx %6lld: decode %8.1f us (%6.1f GB/s of K/V), prefill %lld %8.2f ms (%5.2f TFLOPS)\n", k.name,
                    (long long)ctx, t_dec * 1e6, dec_bytes / t_dec / 1e9, (long long)pf, t_pf * 1e3, flops / t_pf / 1e12);
        w.begin_object().kv("kind", k.name).kv("ctx", ctx).kv("decode_us", t_dec * 1e6).kv("prefill_chunk", pf);
        w.kv("prefill_ms", t_pf * 1e3).kv("prefill_tflops", flops / t_pf / 1e12).end_object();
        for (float* p : {dq, dk, dv, dout}) sycl::free(p, q);
        for (int32_t* p : {dslot, dpos, dtiles}) sycl::free(p, q);
        if (scratch.q) sycl::free(scratch.q, q);
        if (scratch.part) sycl::free(scratch.part, q);
        if (scratch.part2) sycl::free(scratch.part2, q);
        cache.free(q);
      }
    w.end_array().kv("pass", ok).end_object();
    if (!json_path.empty()) std::ofstream(json_path) << w.str() << "\n";
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-attn-test: %s\n", e.what());
    return 1;
  }
}
