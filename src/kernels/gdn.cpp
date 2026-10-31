#include "kernels/gdn.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace eklow::gdn {

namespace esimd = sycl::ext::intel::esimd;

SlotStep SlotBook::next(int32_t slot, int32_t tok0, int32_t n, int32_t accept, int32_t commit, int32_t max_pending) {
  if (accept < 0) accept = pending;
  if (commit < 0) commit = n;
  if (accept > pending || commit > n || n < 0 || n - commit > max_pending)
    throw std::runtime_error("gdn::SlotBook: accept " + std::to_string(accept) + " of " + std::to_string(pending) + " pending, commit " +
                             std::to_string(commit) + " of " + std::to_string(n) + " (at most " + std::to_string(max_pending) + " pending)");
  pending = n - commit;
  return SlotStep{slot, tok0, n, accept, commit, {0, 0, 0}};
}

Plan make_plan(sycl::queue& q, const std::vector<SlotStep>& slots) {
  Plan p;
  p.n_slots = int32_t(slots.size());
  for (const SlotStep& s : slots) {
    p.n_tokens += s.n;
    p.max_n = std::max(p.max_n, s.n);
  }
  auto* d = sycl::malloc_device<SlotStep>(std::max<size_t>(slots.size(), 1), q);
  if (!d) throw std::runtime_error("gdn::make_plan: out of device memory");
  q.memcpy(d, slots.data(), slots.size() * sizeof(SlotStep)).wait();
  p.dev = d;
  return p;
}

void free_plan(sycl::queue& q, Plan& p) {
  sycl::free(const_cast<SlotStep*>(p.dev), q);
  p = Plan{};
}

Cache alloc_cache(sycl::queue& q, int n_layers, int n_slots, int max_pending) {
  if (max_pending < 0 || max_pending > kMaxPendingLimit) throw std::runtime_error("gdn::alloc_cache: max_pending out of range");
  Cache c;
  c.n_layers = n_layers, c.n_slots = n_slots, c.max_pending = max_pending;
  const int64_t sl = int64_t(n_slots) * n_layers;
  auto alloc = [&](int64_t floats) {
    auto* p = sycl::malloc_device<float>(size_t(std::max<int64_t>(floats, 1)), q);
    if (!p) throw std::runtime_error("gdn::alloc_cache: out of device memory");
    return p;
  };
  c.ssm = alloc(sl * c.ssm_floats_per_layer());
  c.conv = alloc(sl * c.conv_floats_per_layer());
  c.pend_in = alloc(sl * max_pending * kChannels);
  c.pend_gb = alloc(sl * max_pending * 64);
  c.pend_x = alloc(sl * max_pending * kChannels);
  q.memset(c.ssm, 0, size_t(sl * c.ssm_floats_per_layer()) * 4);
  q.memset(c.conv, 0, size_t(sl * c.conv_floats_per_layer()) * 4);
  q.wait();
  return c;
}

void free_cache(sycl::queue& q, Cache& c) {
  sycl::free(c.ssm, q);
  sycl::free(c.conv, q);
  sycl::free(c.pend_in, q);
  sycl::free(c.pend_gb, q);
  sycl::free(c.pend_x, q);
  c = Cache{};
}

void reset_slot(sycl::queue& q, const Cache& c, int slot) {
  q.memset(c.ssm + int64_t(slot) * c.n_layers * c.ssm_floats_per_layer(), 0, size_t(c.ssm_bytes()));
  q.memset(c.conv + int64_t(slot) * c.n_layers * c.conv_floats_per_layer(), 0, size_t(c.conv_bytes()));
}

sycl::event snapshot_to_host(sycl::queue& q, const Cache& c, int slot, void* host) {
  q.memcpy(host, c.ssm + int64_t(slot) * c.n_layers * c.ssm_floats_per_layer(), size_t(c.ssm_bytes()));
  return q.memcpy(static_cast<char*>(host) + c.ssm_bytes(), c.conv + int64_t(slot) * c.n_layers * c.conv_floats_per_layer(),
                  size_t(c.conv_bytes()));
}

sycl::event restore_from_host(sycl::queue& q, const Cache& c, int slot, const void* host) {
  q.memcpy(c.ssm + int64_t(slot) * c.n_layers * c.ssm_floats_per_layer(), host, size_t(c.ssm_bytes()));
  return q.memcpy(c.conv + int64_t(slot) * c.n_layers * c.conv_floats_per_layer(), static_cast<const char*>(host) + c.ssm_bytes(),
                  size_t(c.conv_bytes()));
}

Scratch alloc_scratch(sycl::queue& q, int64_t max_tokens) {
  const int64_t per = kChannels + 64 + kInner;
  auto* p = sycl::malloc_device<float>(size_t(max_tokens * per), q);
  if (!p) throw std::runtime_error("gdn::alloc_scratch: out of device memory");
  return Scratch{p, p + max_tokens * kChannels, p + max_tokens * (kChannels + 64)};
}

void free_scratch(sycl::queue& q, Scratch& s) {
  sycl::free(s.in, q);
  s = Scratch{};
}


namespace {

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

// Tokens a prepare thread walks (its window of conv inputs slides along them).
constexpr int kTile = 8;

}  // namespace

float* repack_conv_w(sycl::queue& q, const float* raw) {
  auto* out = sycl::malloc_device<float>(size_t(kChannels) * kConvTaps, q);
  if (!out) throw std::runtime_error("gdn::repack_conv_w: out of device memory");
  q.parallel_for(sycl::range<1>(size_t(kChannels) * kConvTaps), [=](sycl::id<1> id) {
    const int32_t i = int32_t(id[0]), c = i / kConvTaps, t = i % kConvTaps;
    out[(c / kHead * kConvTaps + t) * kHead + c % kHead] = raw[i];
  }).wait();
  return out;
}

namespace {

// One thread of prepare: 128 channels (one head's q, k or v) over up to kTile tokens of one slot, the window of three
// earlier rows sliding in registers; thread 64 of each tile does g and beta.
struct Prepare {
  const SlotStep* slots;
  const float *x, *a, *b, *conv_w, *ssm_a, *dt;
  int64_t ldx, lda, ldb;
  float* conv;  // this layer, slot 0
  int64_t conv_slot;
  const float* pend_x;
  int64_t pend_slot;
  float *out, *gb;
  float eps;

  static esimd::simd<float, 128> row128(const float* p) SYCL_ESIMD_FUNCTION { return esimd::block_load<float, 128>(p); }

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    const SlotStep st = slots[it.get_group(0)];
    const int32_t j0 = int32_t(it.get_group(1)) * kTile, hg = int32_t(it.get_group(2));
    const int32_t j1 = std::min(j0 + kTile, st.n);
    if (hg == 64) {
      // g and beta of the tile's tokens, 32 heads at a time.
      const esimd::simd<float, 32> sa = esimd::block_load<float, 32>(ssm_a), db = esimd::block_load<float, 32>(dt);
      for (int32_t j = j0; j < j1; ++j) {
        const int64_t row = st.tok0 + j;
        const esimd::simd<float, 32> sp = esimd::block_load<float, 32>(a + row * lda) + db;
        esimd::simd<float, 32> soft = esimd::log(1.0f + esimd::exp(sp));
        soft.merge(sp, sp > 20.0f);
        const esimd::simd<float, 32> g = sa * soft, bv = esimd::block_load<float, 32>(b + row * ldb);
        esimd::block_store<float, 32>(gb + row * 64, esimd::exp(g));
        esimd::block_store<float, 32>(gb + row * 64 + 32, 1.0f / (1.0f + esimd::exp(-bv)));
      }
      return;
    }
    const int32_t ch0 = hg * kHead;
    const float* wp = conv_w + int64_t(hg) * kConvTaps * kHead;
    float* cs = conv + int64_t(st.slot) * conv_slot;
    const float* px = pend_x + int64_t(st.slot) * pend_slot;
    // The window: the three rows before this tile's first token. Row e of the history [conv state ++ accepted pending rows].
    esimd::simd<float, 128> h0, h1, h2;
    {
      const float* src[3];
      for (int r = 0; r < 3; ++r) {
        const int32_t pos = j0 - 3 + r, e = pos + 3 + st.accept;
        src[r] = pos >= 0 ? x + int64_t(st.tok0 + pos) * ldx + ch0 : e < 3 ? cs + int64_t(e) * kChannels + ch0 : px + int64_t(e - 3) * kChannels + ch0;
      }
      h0 = row128(src[0]), h1 = row128(src[1]), h2 = row128(src[2]);
    }
    if (j0 == 0) {
      // The slot's conv state through its accepted and committed tokens: the last three rows of
      // [state ++ accepted pending rows ++ committed new rows]. Only this tile reads or writes the state, and a new row r
      // only ever comes from an old row at or after r.
      const int32_t len = 3 + st.accept + st.commit;
      for (int r = 0; r < 3; ++r) {
        const int32_t e = len - 3 + r;
        if (e == r) continue;
        const float* src = e < 3 ? cs + int64_t(e) * kChannels + ch0
                           : e < 3 + st.accept ? px + int64_t(e - 3) * kChannels + ch0
                                                : x + int64_t(st.tok0 + e - 3 - st.accept) * ldx + ch0;
        esimd::block_store<float, 128>(cs + int64_t(r) * kChannels + ch0, row128(src));
      }
    }
    for (int32_t j = j0; j < j1; ++j) {
      const int64_t row = st.tok0 + j;
      const esimd::simd<float, 128> xn = row128(x + row * ldx + ch0);
      esimd::simd<float, 128> s = h0 * row128(wp);
      s += h1 * row128(wp + kHead);
      s += h2 * row128(wp + 2 * kHead);
      s += xn * row128(wp + 3 * kHead);
      h0 = h1, h1 = h2, h2 = xn;
      esimd::simd<float, 128> v = s / (1.0f + esimd::exp(-s));
      if (hg < 2 * kKeyHeads) {
        const float ss = esimd::reduce<float>(v * v, std::plus<>());
        float inv = esimd::inv(esimd::sqrt(esimd::simd<float, 1>(ss + eps)))[0];
        if (hg < kKeyHeads) inv *= 0.08838834764831845f;  // 1/sqrt(128)
        v *= inv;
      }
      esimd::block_store<float, 128>(out + row * kChannels + ch0, v);
    }
  }
};

}  // namespace

sycl::event prepare(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s) {
  if (p.n_slots == 0) return {};
  const int32_t tiles = std::max(1, (p.max_n + kTile - 1) / kTile);
  const int64_t nl = c.n_layers;
  Prepare k{p.dev,
            in.x,
            in.a,
            in.b,
            w.conv_w,
            w.ssm_a,
            w.dt_bias,
            in.ldx,
            in.lda,
            in.ldb,
            c.conv + int64_t(layer) * c.conv_floats_per_layer(),
            nl * c.conv_floats_per_layer(),
            c.pend_x + int64_t(layer) * c.max_pending * kChannels,
            nl * c.max_pending * kChannels,
            s.in,
            s.gb,
            w.eps};
  return q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(p.n_slots), size_t(tiles), 65), sycl::range<3>(1, 1, 1)), k);
}

namespace {

// 128 products summed into 16 lanes, in kDotChains independent chains (a single chain of eight dependent multiply-adds
// ran serially through one accumulator register).
constexpr int kDotChains = 2;
constexpr int kGroup = 2;
inline esimd::simd<float, 16> dot16(esimd::simd<float, 128> a, esimd::simd<float, 128> b) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, 16> c[kDotChains];
#pragma unroll
  for (int j = 0; j < kDotChains; ++j) c[j] = a.template select<16, 1>(16 * j) * b.template select<16, 1>(16 * j);
#pragma unroll
  for (int i = kDotChains; i < 8; ++i) c[i % kDotChains] += a.template select<16, 1>(16 * i) * b.template select<16, 1>(16 * i);
  if constexpr (kDotChains == 1) return c[0];
  else if constexpr (kDotChains == 2) return c[0] + c[1];
  else return (c[0] + c[1]) + (c[2] + c[3]);
}

// M batches of 16 lanes to M sums (log steps on all batches at once).
template <int M, int W>
esimd::simd<float, M> hsum(esimd::simd<float, M * W> a) SYCL_ESIMD_FUNCTION {
  if constexpr (W == 1) {
    return a;
  } else {
    auto v = a.template bit_cast_view<float, M, W>();
    esimd::simd<float, M * (W / 2)> b = v.template select<M, 1, W / 2, 1>(0, 0) + v.template select<M, 1, W / 2, 1>(0, W / 2);
    return hsum<M, W / 2>(b);
  }
}

// One thread of the scan: C columns of each of NH value heads (h and h + 16, which share a key head) of one slot, the
// state (M = C NH columns of 128 floats) in registers, walking the slot's tokens. The 128 rows of a column are eight
// SIMD16 chunks; per column and token the two dots (S^T k and S^T q, both on the state before this token's update) are
// 16 fused multiply-adds, the update 16 more, with the horizontal sums of all columns done together:
//   kv = eg (S^T k);  d = beta (v - kv);  o = eg (S^T q) + d (k . q);  S = eg S + k d^T
// which is the recurrence of llama.cpp with the decay folded in (S *= exp(g) first, then kv, then S += k d^T, then
// o = S^T q), in fp32 rounding.
template <int C, int NH>
struct Scan {
  static constexpr int M = C * NH;
  const SlotStep* slots;
  float* ssm;            // this layer, slot 0
  int64_t ssm_slot;      // floats between slots
  const float *in, *gb;  // scratch: [t][8192], [t][64]
  float* o;              // [t][4096]
  const float *pend_in, *pend_gb;  // this layer, slot 0
  int64_t pend_in_slot, pend_gb_slot;

  struct Tok {
    esimd::simd<float, 128> k, q;
    esimd::simd<float, M> v, eg, beta;
  };

  // A token's operands: qp, kp the row's q and k heads (128 floats), vp the first value head's columns (heads NH apart by
  // 16 x 128), gp the row's exp(g) | beta at the first head.
  static Tok load(const float* qp, const float* kp, const float* vp, const float* gp) SYCL_ESIMD_FUNCTION {
    Tok t;
    t.k = esimd::block_load<float, 128>(kp);
    t.q = esimd::block_load<float, 128>(qp);
#pragma unroll
    for (int i = 0; i < NH; ++i) {
      t.v.template select<C, 1>(i * C) = esimd::block_load<float, C>(vp + i * kKeyHeads * kHead);
      t.eg.template select<C, 1>(i * C) = gp[i * kKeyHeads];
      t.beta.template select<C, 1>(i * C) = gp[32 + i * kKeyHeads];
    }
    return t;
  }

  static void token(esimd::simd<float, 128> (&S)[M], Tok t, esimd::simd<float, M>& ov) SYCL_ESIMD_FUNCTION {
    // Columns in groups of kGroup, each group's dots and update done before the next group's, so that the accumulators of
    // only one group are live next to the state (M = 4 columns fill 64 of the 128 registers, k and q 32 more).
    constexpr int G = M < kGroup ? M : kGroup, NA = 2 * G + 1;  // S^T k and S^T q of each column, then k . q
    float kq = 0;
#pragma unroll
    for (int g0 = 0; g0 < M; g0 += G) {
      esimd::simd<float, 16 * NA> acc;
#pragma unroll
      for (int c = 0; c < G; ++c) {
        acc.template select<16, 1>(16 * c) = dot16(S[g0 + c], t.k);
        acc.template select<16, 1>(16 * (G + c)) = dot16(S[g0 + c], t.q);
      }
      acc.template select<16, 1>(16 * 2 * G) = dot16(t.k, t.q);
      esimd::simd<float, NA> sums = hsum<NA, 16>(acc);
      kq = sums[2 * G];
      const esimd::simd<float, G> eg = t.eg.template select<G, 1>(g0);
      const esimd::simd<float, G> kv = eg * sums.template select<G, 1>(0);
      const esimd::simd<float, G> d = (t.v.template select<G, 1>(g0) - kv) * t.beta.template select<G, 1>(g0);
      ov.template select<G, 1>(g0) = eg * sums.template select<G, 1>(G) + d * kq;
#pragma unroll
      for (int c = 0; c < G; ++c) {
        const float ec = eg[c], dc = d[c];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          S[g0 + c].template select<16, 1>(16 * i) = S[g0 + c].template select<16, 1>(16 * i) * ec + t.k.template select<16, 1>(16 * i) * dc;
      }
    }
  }


  static void store(float* const (&sp)[M], esimd::simd<float, 128> (&S)[M]) SYCL_ESIMD_FUNCTION {
#pragma unroll
    for (int c = 0; c < M; ++c) esimd::block_store<float, 128>(sp[c], S[c]);
  }

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    const SlotStep st = slots[it.get_group(0)];
    if (st.n == 0 && st.accept == 0) return;
    const int32_t h0 = int32_t(it.get_group(1)), col0 = int32_t(it.get_global_id(2)) * C, kh = h0 % kKeyHeads;
    float* sp[M];
    esimd::simd<float, 128> S[M];
#pragma unroll
    for (int i = 0; i < NH; ++i)
#pragma unroll
      for (int c = 0; c < C; ++c) {
        sp[i * C + c] = ssm + int64_t(st.slot) * ssm_slot + (int64_t(h0 + i * kKeyHeads) * kHead + col0 + c) * kHead;
        S[i * C + c] = esimd::block_load<float, 128>(sp[i * C + c]);
      }
    esimd::simd<float, M> ov = 0;
    // One loop over the tokens accepted from the last step's pending ones (replayed from their saved inputs, which hold q
    // too, so that the same code runs them as runs new tokens: the state after them is then bit-identical to having run
    // only those) and then the new ones. The state is stored once, after the last accepted-or-committed token. The rows
    // are walked by pointers advanced a row at a time (64-bit address arithmetic costs more than the loads).
    const int32_t total = st.accept + st.n, store_at = st.accept + st.commit;
    const int64_t new0 = int64_t(st.tok0);
    const float* r = st.accept > 0 ? pend_in + int64_t(st.slot) * pend_in_slot : in + new0 * kChannels;
    const float* g = st.accept > 0 ? pend_gb + int64_t(st.slot) * pend_gb_slot : gb + new0 * 64;
    float* op = o + new0 * kInner + int64_t(h0) * kHead + col0;
    const int32_t qoff = kh * kHead, koff = kQk + kh * kHead, voff = 2 * kQk + h0 * kHead + col0;
    for (int32_t i = 0; i < total; ++i) {
      if (i == st.accept && i > 0) r = in + new0 * kChannels, g = gb + new0 * 64;
      token(S, load(r + qoff, r + koff, r + voff, g + h0), ov);
      r += kChannels, g += 64;
      if (i >= st.accept) {
#pragma unroll
        for (int h = 0; h < NH; ++h) esimd::block_store<float, C>(op + h * kKeyHeads * kHead, esimd::simd<float, C>(ov.template select<C, 1>(h * C)));
        op += kInner;
      }
      if (i + 1 == store_at) store(sp, S);
    }
  }
};

// Two columns of one value head a thread: 2,048 threads for a slot. Measured (DECISIONS-style notes in REPORT.md): four columns
// halve the key and query loads, which are what limits the scan (the card's load ports deliver 64 B a clock a core, and every
// thread takes its own copy of the 1 KB of k and q a token), but leave a lone slot with 1,024 threads and too little to hide
// latency with (2.6 ms against 2.2 for 2,048 tokens), though two slots are faster that way (2.9 against 3.2 ms for 2 x 2,048).
constexpr int kScanCols = 2;

template <int C, int NH>
sycl::event launch_scan(sycl::queue& q, const Cache& c, int layer, const Plan& p, const Scratch& s) {
  constexpr int kPer = kHead / C;
  const int64_t nl = c.n_layers;
  Scan<C, NH> k{p.dev,
                      c.ssm + int64_t(layer) * c.ssm_floats_per_layer(),
                      nl * c.ssm_floats_per_layer(),
                      s.in,
                      s.gb,
                      s.o,
                      c.pend_in + int64_t(layer) * c.max_pending * kChannels,
                      c.pend_gb + int64_t(layer) * c.max_pending * 64,
                      nl * c.max_pending * kChannels,
                      nl * c.max_pending * 64};
  constexpr int kLocal = kPer < 8 ? kPer : 8;
  return q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(p.n_slots), kValueHeads / NH, kPer), sycl::range<3>(1, 1, kLocal)), k);
}

}  // namespace

sycl::event scan(sycl::queue& q, const Cache& c, int layer, const Plan& p, const Scratch& s) {
  if (p.n_slots == 0) return {};
  return launch_scan<kScanCols, 1>(q, c, layer, p, s);
}

sycl::event gated_norm(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s,
                xmx::Activations& out) {
  if (p.n_slots == 0 || p.n_tokens == 0) return {};
  const int32_t T = p.n_tokens, n_slots = p.n_slots;
  xmx::set_rows(out, T, kInner);
  int8_t* aq = nullptr;
  float* as = nullptr;
  sycl::half* ah = nullptr;
  if (out.fp16) ah = out.h;
  else aq = out.q, as = out.s;
  const SlotStep* slots = p.dev;
  const int64_t nl = c.n_layers;
  float* pend_in = c.pend_in + int64_t(layer) * c.max_pending * kChannels;
  float* pend_gb = c.pend_gb + int64_t(layer) * c.max_pending * 64;
  float* pend_x = c.pend_x + int64_t(layer) * c.max_pending * kChannels;
  const int64_t pi_slot = nl * c.max_pending * kChannels, pg_slot = nl * c.max_pending * 64;
  const float *o = s.o, *z = in.z, *x = in.x, *norm_w = w.norm_w, *sin = s.in, *sgb = s.gb;
  const int64_t ldz = in.ldz, ldx = in.ldx;
  const float eps = w.eps;
  constexpr int kb = kInner / 32;
  // One sub-group per (token, head): 16 lanes of 8 elements; a work-group is 8 heads of a token.
  return q.parallel_for(sycl::nd_range<1>(size_t(T) * 4 * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t g = int32_t(it.get_group(0)), t = g / 4;
    auto sg = it.get_sub_group();
    const int32_t h = (g % 4) * 8 + int32_t(sg.get_group_linear_id()), l = int32_t(sg.get_local_id()[0]), c0 = h * kHead + l * 8;
    float ov[8], ss = 0;
    for (int i = 0; i < 8; ++i) {
      ov[i] = o[int64_t(t) * kInner + c0 + i];
      ss += ov[i] * ov[i];
    }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(kHead) + eps);
    float y[8], amax = 0;
    for (int i = 0; i < 8; ++i) {
      y[i] = ov[i] * inv * norm_w[l * 8 + i] * silu(z[int64_t(t) * ldz + c0 + i]);
      amax = sycl::fmax(amax, sycl::fabs(y[i]));
    }
    if (ah) {
      for (int i = 0; i < 8; ++i) ah[xmx::act16_index(t, c0 + i, kInner / 16)] = sycl::half(y[i]);
    } else {
      amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
      amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
      const float d = xmx::quantize_run(y, amax, aq + xmx::act_index(t, c0, kb));
      if (c0 % 32 == 0) as[xmx::act_scale_index(t, c0, kb)] = d;
    }
    // Tokens left pending: their post-conv inputs, g, beta and raw row go to the slot's pending buffers.
    SlotStep st{};
    for (int32_t i = 0; i < n_slots; ++i) {
      st = slots[i];
      if (t >= st.tok0 && t < st.tok0 + st.n) break;
    }
    const int32_t j = t - st.tok0;
    if (j < st.commit) return;
    const int64_t pj = int64_t(j - st.commit);
    float* pi = pend_in + int64_t(st.slot) * pi_slot + pj * kChannels;
    const float* si = sin + int64_t(t) * kChannels;
    for (int i = 0; i < 8; ++i) pi[2 * kQk + c0 + i] = si[2 * kQk + c0 + i];
    if (h < kKeyHeads)
      for (int i = 0; i < 8; ++i) pi[c0 + i] = si[c0 + i], pi[kQk + c0 + i] = si[kQk + c0 + i];
    if (h == 0 && l < 8) {
      float* pg = pend_gb + int64_t(st.slot) * pg_slot + pj * 64;
      for (int i = 0; i < 8; ++i) pg[l * 8 + i] = sgb[int64_t(t) * 64 + l * 8 + i];
    }
    float* px = pend_x + int64_t(st.slot) * pi_slot + pj * kChannels;
    for (int i = 0; i < 16; ++i) px[h * 256 + l * 16 + i] = x[int64_t(t) * ldx + h * 256 + l * 16 + i];
  });
}

sycl::event step(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s,
          xmx::Activations& out) {
  prepare(q, w, c, layer, p, in, s);
  scan(q, c, layer, p, s);
  return gated_norm(q, w, c, layer, p, in, s, out);
}

}  // namespace eklow::gdn
