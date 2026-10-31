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
  return SlotStep{slot, tok0, n, accept, commit, 0, {0, 0}};
}

Plan make_plan(sycl::queue& q, const std::vector<SlotStep>& slots) {
  Plan p;
  p.n_slots = int32_t(slots.size());
  std::vector<SlotStep> s = slots;
  int32_t shift = 0;
  for (SlotStep& st : s) {
    if (st.accept < 0 || st.n < 0 || st.commit < 0 || st.commit > st.n) throw std::runtime_error("gdn::make_plan: bad slot step");
    shift += st.accept;
    st.shift = shift;
    p.n_tokens += st.n;
    p.max_n = std::max(p.max_n, st.n);
  }
  p.n_rows = p.n_tokens + shift;
  auto* d = sycl::malloc_device<SlotStep>(std::max<size_t>(s.size(), 1), q);
  if (!d) throw std::runtime_error("gdn::make_plan: out of device memory");
  q.memcpy(d, s.data(), s.size() * sizeof(SlotStep)).wait();
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
  c.pend_in = alloc(sl * kValueHeads * max_pending * kHeadChannels);
  c.pend_gb = alloc(sl * max_pending * 64);
  c.pend_x = alloc(sl * kValueHeads * max_pending * kHeadChannels);
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

Scratch alloc_scratch(sycl::queue& q, int64_t max_rows) {
  const int64_t per = kChannels + 64 + kInner;
  auto* p = sycl::malloc_device<float>(size_t(max_rows * per), q);
  if (!p) throw std::runtime_error("gdn::alloc_scratch: out of device memory");
  return Scratch{p, p + max_rows * kChannels, p + max_rows * (kChannels + 64)};
}

void free_scratch(sycl::queue& q, Scratch& s) {
  sycl::free(s.in, q);
  s = Scratch{};
}

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

constexpr int kHC = kHeadChannels;

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

// Tokens a prepare thread walks (its window of conv inputs slides along them).
constexpr int kTile = 8;

inline esimd::simd<float, 128> row128(const float* p) SYCL_ESIMD_FUNCTION { return esimd::block_load<float, 128>(p); }

// The convolution over four rows (taps in tap-major rows at wp), silu and, for q and k (hg < 32), the l2 norm, q scaled
// by 1/sqrt(128): one head's 128 channels of one token.
inline esimd::simd<float, 128> conv_unit(esimd::simd<float, 128> h0, esimd::simd<float, 128> h1, esimd::simd<float, 128> h2,
                                         esimd::simd<float, 128> xn, const float* wp, int32_t hg, float eps) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, 128> s = h0 * row128(wp);
  s += h1 * row128(wp + kHead);
  s += h2 * row128(wp + 2 * kHead);
  s += xn * row128(wp + 3 * kHead);
  esimd::simd<float, 128> v = s / (1.0f + esimd::exp(-s));
  if (hg < 2 * kKeyHeads) {
    const float ss = esimd::reduce<float>(v * v, std::plus<>());
    float inv = esimd::inv(esimd::sqrt(esimd::simd<float, 1>(ss + eps)))[0];
    if (hg < kKeyHeads) inv *= 0.08838834764831845f;  // 1/sqrt(128)
    v *= inv;
  }
  return v;
}

// exp(g) and beta of up to 16 tokens of one head: a, b the head's inputs of the first token (row strides lda, ldb), valid lanes
// < n.
inline void gate_beta(const float* a, const float* b, int64_t lda, int64_t ldb, float ssm_a, float dt, int32_t n, esimd::simd<float, 16>& eg,
                      esimd::simd<float, 16>& beta) SYCL_ESIMD_FUNCTION {
  const esimd::simd<uint32_t, 16> idx(0, 1);
  const esimd::simd_mask<16> valid = idx < uint32_t(n);
  const esimd::simd<float, 16> av = esimd::gather<float, 16>(a, idx * uint32_t(lda * 4), valid, esimd::simd<float, 16>(0.0f));
  const esimd::simd<float, 16> bv = esimd::gather<float, 16>(b, idx * uint32_t(ldb * 4), valid, esimd::simd<float, 16>(0.0f));
  const esimd::simd<float, 16> sp = av + dt;
  esimd::simd<float, 16> soft = esimd::log(1.0f + esimd::exp(sp));
  soft.merge(sp, sp > 20.0f);
  eg = esimd::exp(ssm_a * soft);
  beta = 1.0f / (1.0f + esimd::exp(-bv));
}

// One thread of prepare: 128 channels (one head's q, k or v) over up to kTile tokens of one slot, the window of three
// earlier rows sliding in registers; thread 64 of each tile does g and beta.
struct Prepare {
  const SlotStep* slots;
  const float *x, *a, *b, *conv_w, *ssm_a, *dt;
  int64_t ldx, lda, ldb;
  float* conv;  // this layer, slot 0
  int64_t conv_slot;
  const float* pend_x;
  const float *pend_in, *pend_gb;
  int64_t pend_slot, pend_gb_slot;
  int32_t maxp;
  float *out, *gb;
  float eps;

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    const SlotStep st = slots[it.get_group(0)];
    const int32_t j0 = int32_t(it.get_group(1)) * kTile, hg = int32_t(it.get_group(2));
    const int32_t j1 = std::min(j0 + kTile, st.n);
    const int64_t sb = int64_t(st.tok0) + st.shift - st.accept;  // the slot's first scratch row, its replayed tokens'
    if (hg == 64) {
      // g and beta of the tile's tokens, a tile at a time (16 tokens of one head per lane group).
      if (j0 == 0)  // the replayed tokens' saved exp(g) and beta
        for (int32_t p = 0; p < st.accept; ++p)
          esimd::block_store<float, 64>(gb + (sb + p) * 64, esimd::block_load<float, 64>(pend_gb + int64_t(st.slot) * pend_gb_slot + int64_t(p) * 64));
      for (int32_t j = j0; j < j1; ++j) {
        const int64_t row = st.tok0 + j, srow = row + st.shift;
        const esimd::simd<float, 32> sa = esimd::block_load<float, 32>(ssm_a), db = esimd::block_load<float, 32>(dt);
        const esimd::simd<float, 32> sp = esimd::block_load<float, 32>(a + row * lda) + db;
        esimd::simd<float, 32> soft = esimd::log(1.0f + esimd::exp(sp));
        soft.merge(sp, sp > 20.0f);
        const esimd::simd<float, 32> bv = esimd::block_load<float, 32>(b + row * ldb);
        esimd::block_store<float, 32>(gb + srow * 64, esimd::exp(sa * soft));
        esimd::block_store<float, 32>(gb + srow * 64 + 32, 1.0f / (1.0f + esimd::exp(-bv)));
      }
      return;
    }
    // The unit's head: q and k units belong to key head hg % 16 (value heads hh and hh + 16 each keep a copy of those
    // channels), v units to value head hg - 32.
    const int32_t part = hg < kKeyHeads ? 0 : hg < 2 * kKeyHeads ? 1 : 2, hh = part == 2 ? hg - 32 : hg % kKeyHeads, ch0 = hg * kHead;
    const float* wp = conv_w + int64_t(hg) * kConvTaps * kHead;
    float* cs = conv + int64_t(st.slot) * conv_slot + int64_t(hh) * 3 * kHC + part * kHead;
    const float* px = pend_x + int64_t(st.slot) * pend_slot + int64_t(hh) * maxp * kHC + part * kHead;
    // The window: the three rows before this tile's first token. Row e of the history [conv state ++ accepted pending rows].
    esimd::simd<float, 128> h0, h1, h2;
    {
      const float* src[3];
      for (int r = 0; r < 3; ++r) {
        const int32_t pos = j0 - 3 + r, e = pos + 3 + st.accept;
        src[r] = pos >= 0 ? x + int64_t(st.tok0 + pos) * ldx + ch0 : e < 3 ? cs + int64_t(e) * kHC : px + int64_t(e - 3) * kHC;
      }
      h0 = row128(src[0]), h1 = row128(src[1]), h2 = row128(src[2]);
    }
    if (j0 == 0) {
      // The slot's conv state through its accepted and committed tokens: the last three rows of
      // [state ++ accepted pending rows ++ committed new rows]. Only this tile reads or writes these channels' state, and a
      // new row r only ever comes from an old row at or after r.
      const int32_t len = 3 + st.accept + st.commit;
      for (int r = 0; r < 3; ++r) {
        const int32_t e = len - 3 + r;
        if (e == r) continue;
        const float* src = e < 3 ? cs + int64_t(e) * kHC
                           : e < 3 + st.accept ? px + int64_t(e - 3) * kHC
                                                : x + int64_t(st.tok0 + e - 3 - st.accept) * ldx + ch0;
        const esimd::simd<float, 128> v = row128(src);
        esimd::block_store<float, 128>(cs + int64_t(r) * kHC, v);
        if (part < 2) esimd::block_store<float, 128>(cs + int64_t(kKeyHeads) * 3 * kHC + int64_t(r) * kHC, v);
      }
    }
    if (j0 == 0) {  // the replayed tokens' saved inputs, in front of the slot's new rows
      const float* pi = pend_in + int64_t(st.slot) * pend_slot + int64_t(hh) * maxp * kHC + part * kHead;
      for (int32_t p = 0; p < st.accept; ++p) esimd::block_store<float, 128>(out + (sb + p) * kChannels + ch0, row128(pi + int64_t(p) * kHC));
    }
    for (int32_t j = j0; j < j1; ++j) {
      const int64_t row = st.tok0 + j;
      const esimd::simd<float, 128> xn = row128(x + row * ldx + ch0);
      esimd::block_store<float, 128>(out + (row + st.shift) * kChannels + ch0, conv_unit(h0, h1, h2, xn, wp, hg, eps));
      h0 = h1, h1 = h2, h2 = xn;
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
            c.pend_x + int64_t(layer) * kValueHeads * c.max_pending * kHC,
            c.pend_in + int64_t(layer) * kValueHeads * c.max_pending * kHC,
            c.pend_gb + int64_t(layer) * c.max_pending * 64,
            nl * kValueHeads * c.max_pending * kHC,
            nl * c.max_pending * 64,
            c.max_pending,
            s.in,
            s.gb,
            w.eps};
  return q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(p.n_slots), size_t(tiles), 65), sycl::range<3>(1, 1, 1)), k);
}

namespace {

// M batches of W lanes to M sums (log steps on all batches at once).
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

// 128 products summed into 16 lanes, in two independent chains (a single chain of eight dependent multiply-adds ran
// serially through one accumulator register).
inline esimd::simd<float, 16> dot16(esimd::simd<float, 128> a, esimd::simd<float, 128> b) SYCL_ESIMD_FUNCTION {
  esimd::simd<float, 16> c0 = a.template select<16, 1>(0) * b.template select<16, 1>(0);
  esimd::simd<float, 16> c1 = a.template select<16, 1>(16) * b.template select<16, 1>(16);
#pragma unroll
  for (int i = 2; i < 8; i += 2) {
    c0 += a.template select<16, 1>(16 * i) * b.template select<16, 1>(16 * i);
    c1 += a.template select<16, 1>(16 * (i + 1)) * b.template select<16, 1>(16 * (i + 1));
  }
  return c0 + c1;
}

// C columns of one value head's state in registers (C x 128 floats): the recurrence for one token. The 128 rows of a column
// are eight SIMD16 chunks; per column the two dots (S^T k and S^T q, both on the state before this token's update) are 16
// fused multiply-adds, the update 16 more, with the horizontal sums of all columns done together:
//   kv = eg (S^T k);  d = beta (v - kv);  o = eg (S^T q) + d (k . q);  S = eg S + k d^T
// which is the recurrence of llama.cpp with the decay folded in (S *= exp(g) first, then kv, then S += k d^T, then
// o = S^T q), in fp32 rounding. Used by the scan and by the fused kernel, once in each, with the operands from wherever.
template <int C>
struct Cols {
  struct Tok {
    esimd::simd<float, 128> k, q;
    esimd::simd<float, C> v, eg, beta;
  };

  static void token(esimd::simd<float, 128> (&S)[C], Tok t, esimd::simd<float, C>& ov) SYCL_ESIMD_FUNCTION {
    constexpr int NA = 2 * C + 1;  // S^T k and S^T q of each column, then k . q
    esimd::simd<float, 16 * NA> acc;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      acc.template select<16, 1>(16 * c) = dot16(S[c], t.k);
      acc.template select<16, 1>(16 * (C + c)) = dot16(S[c], t.q);
    }
    acc.template select<16, 1>(16 * 2 * C) = dot16(t.k, t.q);
    esimd::simd<float, NA> sums = hsum<NA, 16>(acc);
    const esimd::simd<float, C> kv = t.eg * sums.template select<C, 1>(0);
    const esimd::simd<float, C> d = (t.v - kv) * t.beta;
    const float kq = sums[2 * C];
    ov = t.eg * sums.template select<C, 1>(C) + d * kq;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const float ec = t.eg[c], dc = d[c];
#pragma unroll
      for (int i = 0; i < 8; ++i)
        S[c].template select<16, 1>(16 * i) = S[c].template select<16, 1>(16 * i) * ec + t.k.template select<16, 1>(16 * i) * dc;
    }
  }

  static void load_state(float* const (&sp)[C], esimd::simd<float, 128> (&S)[C]) SYCL_ESIMD_FUNCTION {
#pragma unroll
    for (int c = 0; c < C; ++c) S[c] = esimd::block_load<float, 128>(sp[c]);
  }
  static void store_state(float* const (&sp)[C], esimd::simd<float, 128> (&S)[C]) SYCL_ESIMD_FUNCTION {
#pragma unroll
    for (int c = 0; c < C; ++c) esimd::block_store<float, 128>(sp[c], S[c]);
  }
};

// Two columns of one value head a thread: 2,048 threads for a slot. Measured (REPORT.md): four columns halve the key and
// query loads, which are what limits the scan (the card's load ports deliver 64 B a clock a core, and every thread takes its
// own copy of the 1 KB of k and q a token), but leave a lone slot with 1,024 threads and too little to hide latency with
// (2.6 ms against 2.2 for 2,048 tokens), though two slots are faster that way (2.9 against 3.2 ms for 2 x 2,048).
constexpr int kScanCols = 2;

// One thread of the scan: C columns of one value head of one slot, walking the slot's tokens.
struct Scan {
  using K = Cols<kScanCols>;
  static constexpr int C = kScanCols;
  const SlotStep* slots;
  float* ssm;            // this layer, slot 0
  int64_t ssm_slot;      // floats between slots
  const float *in, *gb;  // scratch: [t][8192], [t][64]
  float* o;              // [t][4096]

  // A token's operands: qp, kp the row's q and k heads (128 floats), vp the head's columns, gp the head's exp(g) (beta 32 on).
  static K::Tok load(const float* qp, const float* kp, const float* vp, const float* gp) SYCL_ESIMD_FUNCTION {
    K::Tok t;
    t.k = esimd::block_load<float, 128>(kp);
    t.q = esimd::block_load<float, 128>(qp);
    t.v = esimd::block_load<float, C>(vp);
    t.eg = gp[0];
    t.beta = gp[32];
    return t;
  }

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    const SlotStep st = slots[it.get_group(0)];
    if (st.n == 0 && st.accept == 0) return;
    const int32_t h = int32_t(it.get_group(1)), col0 = int32_t(it.get_global_id(2)) * C, kh = h % kKeyHeads;
    float* sp[C];
    esimd::simd<float, 128> S[C];
#pragma unroll
    for (int c = 0; c < C; ++c) sp[c] = ssm + int64_t(st.slot) * ssm_slot + (int64_t(h) * kHead + col0 + c) * kHead;
    K::load_state(sp, S);
    esimd::simd<float, C> ov = 0;
    // One loop over the slot's scratch rows: the tokens accepted from the last step's pending ones (prepare copied their saved
    // inputs there, q included, so the same code runs them as runs new tokens and the state after them is bit-identical to
    // having run only those), then the new ones. The state is stored once, after the last accepted-or-committed token. The
    // rows are walked by pointers advanced a row at a time (64-bit address arithmetic costs more than the loads).
    const int64_t sb = int64_t(st.tok0) + st.shift - st.accept;
    const int32_t total = st.accept + st.n, store_at = st.accept + st.commit;
    const float* r = in + sb * kChannels;
    const float* gp = gb + sb * 64 + h;
    float* op = o + (sb + st.accept) * kInner + int64_t(h) * kHead + col0;
    const int32_t qo = kh * kHead, ko = kQk + kh * kHead, vo = 2 * kQk + h * kHead + col0;
    for (int32_t i = 0; i < total; ++i, r += kChannels, gp += 64) {
      K::token(S, load(r + qo, r + ko, r + vo, gp), ov);
      if (i >= st.accept) {
        esimd::block_store<float, C>(op, ov);
        op += kInner;
      }
      if (i + 1 == store_at) K::store_state(sp, S);
    }
  }
};

}  // namespace

sycl::event scan(sycl::queue& q, const Cache& c, int layer, const Plan& p, const Scratch& s) {
  if (p.n_slots == 0) return {};
  constexpr int kPer = kHead / kScanCols;
  const int64_t nl = c.n_layers;
  Scan k{p.dev,
         c.ssm + int64_t(layer) * c.ssm_floats_per_layer(),
         nl * c.ssm_floats_per_layer(),
         s.in,
         s.gb,
         s.o};
  constexpr int kLocal = kPer < 8 ? kPer : 8;
  return q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(p.n_slots), kValueHeads, kPer), sycl::range<3>(1, 1, kLocal)), k);
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
  const int32_t maxp = c.max_pending;
  float* pend_in = c.pend_in + int64_t(layer) * kValueHeads * maxp * kHC;
  float* pend_gb = c.pend_gb + int64_t(layer) * maxp * 64;
  float* pend_x = c.pend_x + int64_t(layer) * kValueHeads * maxp * kHC;
  const int64_t pi_slot = nl * kValueHeads * maxp * kHC, pg_slot = nl * maxp * 64;
  const float *o = s.o, *z = in.z, *x = in.x, *norm_w = w.norm_w, *sin = s.in, *sgb = s.gb;
  const int64_t ldz = in.ldz, ldx = in.ldx;
  const float eps = w.eps;
  constexpr int kb = kInner / 32;
  // One sub-group per (token, head): 16 lanes of 8 elements; a work-group is 8 heads of a token.
  return q.parallel_for(sycl::nd_range<1>(size_t(T) * 4 * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
    const int32_t g = int32_t(it.get_group(0)), t = g / 4;
    auto sg = it.get_sub_group();
    SlotStep st{};
    for (int32_t i = 0; i < n_slots; ++i) {
      st = slots[i];
      if (t >= st.tok0 && t < st.tok0 + st.n) break;
    }
    const int64_t srow = int64_t(t) + st.shift;  // the token's scratch row
    const int32_t h = (g % 4) * 8 + int32_t(sg.get_group_linear_id()), l = int32_t(sg.get_local_id()[0]), c0 = h * kHead + l * 8;
    float ov[8], ss = 0;
    for (int i = 0; i < 8; ++i) {
      ov[i] = o[srow * kInner + c0 + i];
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
    // Tokens left pending: head h's inputs (q and k of its key head, its v), g, beta and the raw channels it reads go to the
    // slot's pending buffers.
    const int32_t j = t - st.tok0;
    if (j < st.commit) return;
    const int64_t pj = int64_t(j - st.commit), kh = h % kKeyHeads;
    float* pi = pend_in + int64_t(st.slot) * pi_slot + (int64_t(h) * maxp + pj) * kHC;
    float* px = pend_x + int64_t(st.slot) * pi_slot + (int64_t(h) * maxp + pj) * kHC;
    const float* si = sin + srow * kChannels;
    const float* xr = x + int64_t(t) * ldx;
    const int32_t qo = kh * kHead + l * 8, ko = kQk + kh * kHead + l * 8, vo = 2 * kQk + h * kHead + l * 8;
    for (int i = 0; i < 8; ++i) {
      pi[l * 8 + i] = si[qo + i], pi[kHead + l * 8 + i] = si[ko + i], pi[2 * kHead + l * 8 + i] = si[vo + i];
      px[l * 8 + i] = xr[qo + i], px[kHead + l * 8 + i] = xr[ko + i], px[2 * kHead + l * 8 + i] = xr[vo + i];
    }
    if (l == 0) {
      float* pg = pend_gb + int64_t(st.slot) * pg_slot + pj * 64;
      pg[h] = sgb[srow * 64 + h], pg[32 + h] = sgb[srow * 64 + 32 + h];
    }
  });
}

namespace {

// The whole layer for a step of at most kFusedMaxTokens new tokens a slot, one work-group of 64 threads per (slot, value
// head h); the three stages meet in shared local memory:
//   1. convolution, silu, l2 norm for the head's three units of 128 channels (q and k of its key head, its v) of each token,
//      a thread to a unit, and exp(g) and beta of the head;
//   2. the scan, two columns of the head's state a thread, reading the new tokens' q, k, v from shared memory (the replayed
//      pending tokens' from their saved copies) and leaving o there;
//   3. the gated norm, a thread to a (token, 32-column block), quantized into the activations, and the tokens left pending
//      saved.
// The conv state, pending rows and ssm state are private to the head (heads h and h + 16 each keep their own copy of their
// shared key head's channels), so no work-group reads what another writes.
struct Fused {
  using K = Cols<kScanCols>;
  static constexpr int C = kScanCols, kThreads = kHead / kScanCols, kMaxTok = kFusedMaxTokens;
  static constexpr int kQkvOff = 0, kOOff = kMaxTok * kHC * 4, kSlm = kOOff + kMaxTok * kHead * 4;  // 32 KB: four work-groups fit a core

  const SlotStep* slots;
  const float *x, *z, *a, *b, *conv_w, *ssm_a, *dt, *norm_w;
  int64_t ldx, ldz, lda, ldb;
  float* ssm;
  int64_t ssm_slot;
  float* conv;
  int64_t conv_slot;
  float *pend_in, *pend_x, *pend_gb;
  int64_t pend_slot, pend_gb_slot;
  int32_t maxp;
  int8_t* aq;
  float* as;
  sycl::half* ah;
  float eps;
  float* debug_o;  // when given, o [t][4096] as the split scan writes it (the test checks it)

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlm>();
    const SlotStep st = slots[it.get_group(0)];
    if (st.n == 0 && st.accept == 0) return;  // the same for the whole work-group
    const int32_t h = int32_t(it.get_group(1)), tid = int32_t(it.get_local_id(2)), kh = h % kKeyHeads, n = st.n;
    const int64_t tok0 = st.tok0;
    float* cs = conv + int64_t(st.slot) * conv_slot + int64_t(h) * 3 * kHC;
    float* pxh = pend_x + int64_t(st.slot) * pend_slot + int64_t(h) * maxp * kHC;
    float* pih = pend_in + int64_t(st.slot) * pend_slot + int64_t(h) * maxp * kHC;
    float* pg = pend_gb + int64_t(st.slot) * pend_gb_slot;
    float* sp[C];
    const int32_t col0 = tid * C;
    for (int c = 0; c < C; ++c) sp[c] = ssm + int64_t(st.slot) * ssm_slot + (int64_t(h) * kHead + col0 + c) * kHead;
    // A part's channels of x: q and k of the key head, v of the head.
    auto xch = [&](int32_t part) SYCL_ESIMD_FUNCTION { return part == 0 ? kh * kHead : part == 1 ? kQk + kh * kHead : 2 * kQk + h * kHead; };

    // exp(g) and beta of the head for each token, in registers (every thread wants them).
    esimd::simd<float, 16> eg_all, beta_all;
    gate_beta(a + tok0 * lda + h, b + tok0 * ldb + h, lda, ldb, ssm_a[h], dt[h], n, eg_all, beta_all);

    // 1. A unit of each token: its three parts' conv outputs into shared memory.
    if (tid < 3 * n) {
      const int32_t j = tid / 3, part = tid % 3, hg = part == 0 ? kh : part == 1 ? kKeyHeads + kh : 2 * kKeyHeads + h;
      const int64_t ch0 = int64_t(hg) * kHead;
      esimd::simd<float, 128> hist[3];
      for (int r = 0; r < 3; ++r) {
        const int32_t pos = j - 3 + r, e = pos + 3 + st.accept;
        const float* src = pos >= 0 ? x + (tok0 + pos) * ldx + ch0 : e < 3 ? cs + int64_t(e) * kHC + part * kHead : pxh + int64_t(e - 3) * kHC + part * kHead;
        hist[r] = row128(src);
      }
      const esimd::simd<float, 128> xn = row128(x + (tok0 + j) * ldx + ch0);
      esimd::slm_block_store<float, 128>(kQkvOff + (j * kHC + part * kHead) * 4,
                                         conv_unit(hist[0], hist[1], hist[2], xn, conv_w + int64_t(hg) * kConvTaps * kHead, hg, eps));
    }
    esimd::barrier();

    // The conv state through the accepted and committed tokens (three threads, a part each, after everyone's reads of the old
    // rows): the last three rows of [state ++ accepted pending rows ++ committed new rows].
    if (tid < 3) {
      const int32_t part = tid, len = 3 + st.accept + st.commit;
      esimd::simd<float, 128> nr[3];
      for (int r = 0; r < 3; ++r) {
        const int32_t e = len - 3 + r;
        if (e == r) continue;
        const float* src = e < 3 ? cs + int64_t(e) * kHC + part * kHead
                           : e < 3 + st.accept ? pxh + int64_t(e - 3) * kHC + part * kHead
                                                : x + (tok0 + e - 3 - st.accept) * ldx + xch(part);
        nr[r] = row128(src);
      }
      for (int r = 0; r < 3; ++r)
        if (len - 3 + r != r) esimd::block_store<float, 128>(cs + int64_t(r) * kHC + part * kHead, nr[r]);
    }

    // 2. The scan: the accepted pending tokens replayed, then the new ones, one loop and one token function.
    {
      esimd::simd<float, 128> S[C];
      K::load_state(sp, S);
      esimd::simd<float, C> ov = 0;
      const int32_t total = st.accept + n, store_at = st.accept + st.commit;
      for (int32_t i = 0; i < total; ++i) {
        K::Tok t;
        if (i < st.accept) {
          const float* r = pih + int64_t(i) * kHC;
          t.q = esimd::block_load<float, 128>(r);
          t.k = esimd::block_load<float, 128>(r + kHead);
          t.v = esimd::block_load<float, C>(r + 2 * kHead + col0);
          t.eg = pg[i * 64 + h];
          t.beta = pg[i * 64 + 32 + h];
        } else {
          const int32_t j = i - st.accept;
          t.q = esimd::slm_block_load<float, 128>(kQkvOff + j * kHC * 4);
          t.k = esimd::slm_block_load<float, 128>(kQkvOff + (j * kHC + kHead) * 4);
          t.v = esimd::slm_block_load<float, C>(kQkvOff + (j * kHC + 2 * kHead + col0) * 4);
          const float egj = eg_all[j], bj = beta_all[j];
          t.eg = egj;
          t.beta = bj;
        }
        K::token(S, t, ov);
        if (i >= st.accept) esimd::slm_block_store<float, C>(kOOff + ((i - st.accept) * kHead + col0) * 4, ov);
        if (debug_o && i >= st.accept) esimd::block_store<float, C>(debug_o + (tok0 + st.shift + i - st.accept) * kInner + int64_t(h) * kHead + col0, ov);
        if (i + 1 == store_at) K::store_state(sp, S);
      }
    }
    esimd::barrier();

    // 3. The gated norm: thread (token j, block b) quantizes columns 32 b.. of the head's output for that token.
    if (tid < 4 * n) {
      const int32_t j = tid / 4, b32 = tid % 4;
      const int64_t row = tok0 + j;
      esimd::simd<float, 128> ov = esimd::slm_block_load<float, 128>(kOOff + j * kHead * 4);
      const float ss = esimd::reduce<float>(ov * ov, std::plus<>());
      const float inv = esimd::inv(esimd::sqrt(esimd::simd<float, 1>(ss / float(kHead) + eps)))[0];
      const esimd::simd<float, 32> zv = esimd::block_load<float, 32>(z + row * ldz + h * kHead + b32 * 32);
      const esimd::simd<float, 32> nw = esimd::block_load<float, 32>(norm_w + b32 * 32);
      esimd::simd<float, 32> y = ov.select<32, 1>(b32 * 32);
      y = y * inv * nw * (zv / (1.0f + esimd::exp(-zv)));
      const int32_t k0 = h * kHead + b32 * 32;
      if (ah) {
        esimd::simd<sycl::half, 32> hv(y);
        esimd::block_store<sycl::half, 16>(ah + xmx::act16_index(row, k0, kInner / 16), hv.select<16, 1>(0));
        esimd::block_store<sycl::half, 16>(ah + xmx::act16_index(row, k0 + 16, kInner / 16), hv.select<16, 1>(16));
      } else {
        const float amax = esimd::hmax<float>(esimd::abs(y));
        const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
        esimd::simd<float, 32> t = y * id;
        t += 0.5f;
        t.merge(t - 1.0f, y < 0.0f);  // x +- 0.5, then truncated: round half away from zero
        const esimd::simd<float, 32> r = esimd::rndz<float, 32>(t);
        esimd::block_store<int8_t, 32>(aq + xmx::act_index(row, k0, kInner / 32), esimd::simd<int8_t, 32>(r));
        as[xmx::act_scale_index(row, k0, kInner / 32)] = d;
      }
    }

    // The tokens left pending: the head's post-conv inputs (from shared memory), raw channels and gate.
    const int32_t np = n - st.commit;
    for (int32_t u = tid; u < 6 * np; u += kThreads) {
      const int32_t p = u / 6, kind = u % 6, part = kind % 3, j = st.commit + p;
      if (kind < 3) esimd::block_store<float, 128>(pih + int64_t(p) * kHC + part * kHead, esimd::slm_block_load<float, 128>(kQkvOff + (j * kHC + part * kHead) * 4));
      else esimd::block_store<float, 128>(pxh + int64_t(p) * kHC + part * kHead, row128(x + (tok0 + j) * ldx + xch(part)));
    }
    if (tid == kThreads - 1)
      for (int32_t p = 0; p < np; ++p) {
        pg[p * 64 + h] = eg_all[st.commit + p];
        pg[p * 64 + 32 + h] = beta_all[st.commit + p];
      }
  }
};

}  // namespace

sycl::event step_fused(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, xmx::Activations& out, float* debug_o) {
  if (p.n_slots == 0) return {};
  if (p.max_n > kFusedMaxTokens) throw std::runtime_error("gdn::step_fused: more than kFusedMaxTokens new tokens in a slot");
  xmx::set_rows(out, std::max(p.n_tokens, 1), kInner);
  const int64_t nl = c.n_layers;
  Fused k{p.dev,
          in.x,
          in.z,
          in.a,
          in.b,
          w.conv_w,
          w.ssm_a,
          w.dt_bias,
          w.norm_w,
          in.ldx,
          in.ldz,
          in.lda,
          in.ldb,
          c.ssm + int64_t(layer) * c.ssm_floats_per_layer(),
          nl * c.ssm_floats_per_layer(),
          c.conv + int64_t(layer) * c.conv_floats_per_layer(),
          nl * c.conv_floats_per_layer(),
          c.pend_in + int64_t(layer) * kValueHeads * c.max_pending * kHC,
          c.pend_x + int64_t(layer) * kValueHeads * c.max_pending * kHC,
          c.pend_gb + int64_t(layer) * c.max_pending * 64,
          nl * kValueHeads * c.max_pending * kHC,
          nl * c.max_pending * 64,
          c.max_pending,
          out.fp16 ? nullptr : out.q,
          out.fp16 ? nullptr : out.s,
          out.fp16 ? out.h : nullptr,
          w.eps,
          debug_o};
  return q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(p.n_slots), kValueHeads, Fused::kThreads), sycl::range<3>(1, 1, Fused::kThreads)), k);
}

sycl::event step(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s,
                 xmx::Activations& out, Path path) {
  if (path == Path::Fused || (path == Path::Auto && p.max_n <= kFusedMaxTokens)) return step_fused(q, w, c, layer, p, in, out);
  prepare(q, w, c, layer, p, in, s);
  scan(q, c, layer, p, s);
  return gated_norm(q, w, c, layer, p, in, s, out);
}

}  // namespace eklow::gdn
