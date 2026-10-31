#include "kernels/gdn.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
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

// Tokens a prepare work-group walks (its window of conv inputs slides along them).
constexpr int kTile = 16;

}  // namespace

sycl::event prepare(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s) {
  if (p.n_slots == 0) return {};
  const int32_t tiles = std::max(1, (p.max_n + kTile - 1) / kTile);
  const SlotStep* slots = p.dev;
  const int64_t sl_stride = int64_t(c.n_layers);
  float* conv = c.conv + int64_t(layer) * c.conv_floats_per_layer();
  const int64_t conv_slot = sl_stride * c.conv_floats_per_layer();
  const float* pend_x = c.pend_x + int64_t(layer) * c.max_pending * kChannels;
  const int64_t pend_slot = sl_stride * c.max_pending * kChannels;
  const float *x = in.x, *a = in.a, *b = in.b, *conv_w = w.conv_w, *ssm_a = w.ssm_a, *dt = w.dt_bias;
  const int64_t ldx = in.ldx, lda = in.lda, ldb = in.ldb;
  const float eps = w.eps;
  float* out = s.in;
  float* gb = s.gb;
  return q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(p.n_slots), size_t(tiles), 65 * 16), sycl::range<3>(1, 1, 16)),
                 [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
    const SlotStep st = slots[it.get_group(0)];
    const int32_t j0 = int32_t(it.get_group(1)) * kTile, hg = int32_t(it.get_group(2)), l = int32_t(it.get_local_id(2));
    const int32_t j1 = std::min(j0 + kTile, st.n);
    if (hg == 64) {
      // g and beta: head l and l + 16 of each token.
      for (int32_t j = j0; j < j1; ++j) {
        const int64_t row = st.tok0 + j;
        for (int32_t hh = l; hh < kValueHeads; hh += 16) {
          const float sp = a[row * lda + hh] + dt[hh];
          const float g = ssm_a[hh] * (sp > 20.0f ? sp : sycl::log(1.0f + sycl::exp(sp)));
          gb[row * 64 + hh] = sycl::exp(g);
          gb[row * 64 + 32 + hh] = 1.0f / (1.0f + sycl::exp(-b[row * ldb + hh]));
        }
      }
      return;
    }
    const int32_t ch0 = hg * kHead + l * 8;
    const float* cw = conv_w + int64_t(ch0) * 4;
    float wt[8][4];
    for (int i = 0; i < 8; ++i)
      for (int t = 0; t < 4; ++t) wt[i][t] = cw[i * 4 + t];
    float* cs = conv + int64_t(st.slot) * conv_slot;
    const float* px = pend_x + int64_t(st.slot) * pend_slot;
    // The window: the three rows before this tile's first token. Row e of the history [conv state ++ accepted pending rows].
    float h[3][8];
    for (int r = 0; r < 3; ++r) {
      const int32_t pos = j0 - 3 + r;
      for (int i = 0; i < 8; ++i) {
        if (pos >= 0) h[r][i] = x[int64_t(st.tok0 + pos) * ldx + ch0 + i];
        else {
          const int32_t e = pos + 3 + st.accept;
          h[r][i] = e < 3 ? cs[int64_t(e) * kChannels + ch0 + i] : px[int64_t(e - 3) * kChannels + ch0 + i];
        }
      }
    }
    if (j0 == 0) {
      // The slot's conv state through its accepted and committed tokens: the last three rows of
      // [state ++ accepted pending rows ++ committed new rows]. Only this tile reads or writes the state.
      const int32_t len = 3 + st.accept + st.commit;
      float nr[3][8];
      for (int r = 0; r < 3; ++r) {
        const int32_t e = len - 3 + r;
        for (int i = 0; i < 8; ++i)
          nr[r][i] = e < 3                ? cs[int64_t(e) * kChannels + ch0 + i]
                     : e < 3 + st.accept ? px[int64_t(e - 3) * kChannels + ch0 + i]
                                          : x[int64_t(st.tok0 + e - 3 - st.accept) * ldx + ch0 + i];
      }
      for (int r = 0; r < 3; ++r)
        for (int i = 0; i < 8; ++i) cs[int64_t(r) * kChannels + ch0 + i] = nr[r][i];
    }
    for (int32_t j = j0; j < j1; ++j) {
      const int64_t row = st.tok0 + j;
      float v[8], ss = 0;
      for (int i = 0; i < 8; ++i) {
        const float xn = x[row * ldx + ch0 + i];
        float sum = 0;
        sum += h[0][i] * wt[i][0];
        sum += h[1][i] * wt[i][1];
        sum += h[2][i] * wt[i][2];
        sum += xn * wt[i][3];
        h[0][i] = h[1][i], h[1][i] = h[2][i], h[2][i] = xn;
        v[i] = silu(sum);
        ss += v[i] * v[i];
      }
      if (hg < 2 * kKeyHeads) {
        ss = sycl::reduce_over_group(it.get_sub_group(), ss, sycl::plus<float>());
        float inv = 1.0f / sycl::sqrt(ss + eps);
        if (hg < kKeyHeads) inv *= 0.08838834764831845f;  // 1/sqrt(128)
        for (int i = 0; i < 8; ++i) v[i] *= inv;
      }
      for (int i = 0; i < 8; ++i) out[row * kChannels + ch0 + i] = v[i];
    }
  });
}

namespace {

// One thread of the scan: C columns of one value head's state (C x 128 floats) in registers, walking a slot's tokens.
// The 128 rows of a column are eight SIMD16 chunks; a dot product is eight fused multiply-adds into one chunk and a
// horizontal sum of its 16 lanes, the update eight more, so a column costs about 40 instructions a token.
template <int C>
struct Scan {
  const SlotStep* slots;
  float* ssm;            // this layer, slot 0
  int64_t ssm_slot;      // floats between slots
  const float *in, *gb;  // scratch: [t][8192], [t][64]
  float* o;              // [t][4096]
  const float *pend_in, *pend_gb;  // this layer, slot 0
  int64_t pend_in_slot, pend_gb_slot;

  // S *= exp(g); d = beta (v - S^T k); S += k d^T; and o = S^T q when Out. kv and o are per column.
  template <bool Out>
  static void token(esimd::simd<float, 128> (&S)[C], const float* kp, const float* qp, const float* vp, float eg, float beta,
                    esimd::simd<float, C>& ov) SYCL_ESIMD_FUNCTION {
    esimd::simd<float, 128> k = esimd::block_load<float, 128>(kp);
    const esimd::simd<float, C> v = esimd::block_load<float, C>(vp);
    esimd::simd<float, C> kv;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      S[c] *= eg;
      esimd::simd<float, 16> acc = S[c].template select<16, 1>(0) * k.template select<16, 1>(0);
#pragma unroll
      for (int i = 1; i < 8; ++i) acc += S[c].template select<16, 1>(16 * i) * k.template select<16, 1>(16 * i);
      kv[c] = esimd::reduce<float>(acc, std::plus<>());
    }
    const esimd::simd<float, C> d = (v - kv) * beta;
#pragma unroll
    for (int c = 0; c < C; ++c) {
      const float dc = d[c];
#pragma unroll
      for (int i = 0; i < 8; ++i) S[c].template select<16, 1>(16 * i) += k.template select<16, 1>(16 * i) * dc;
    }
    if constexpr (Out) {
      esimd::simd<float, 128> qv = esimd::block_load<float, 128>(qp);
#pragma unroll
      for (int c = 0; c < C; ++c) {
        esimd::simd<float, 16> acc = S[c].template select<16, 1>(0) * qv.template select<16, 1>(0);
#pragma unroll
        for (int i = 1; i < 8; ++i) acc += S[c].template select<16, 1>(16 * i) * qv.template select<16, 1>(16 * i);
        ov[c] = esimd::reduce<float>(acc, std::plus<>());
      }
    }
  }

  static void store(float* sp, esimd::simd<float, 128> (&S)[C]) SYCL_ESIMD_FUNCTION {
#pragma unroll
    for (int c = 0; c < C; ++c) esimd::block_store<float, 128>(sp + c * kHead, S[c]);
  }

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    const SlotStep st = slots[it.get_group(0)];
    if (st.n == 0 && st.accept == 0) return;
    const int32_t h = int32_t(it.get_group(1)), col0 = int32_t(it.get_global_id(2)) * C, kh = h % kKeyHeads;
    float* sp = ssm + int64_t(st.slot) * ssm_slot + (int64_t(h) * kHead + col0) * kHead;
    esimd::simd<float, 128> S[C];
#pragma unroll
    for (int c = 0; c < C; ++c) S[c] = esimd::block_load<float, 128>(sp + c * kHead);
    esimd::simd<float, C> ov = 0;
    // The tokens accepted from the last step's pending ones, replayed from their saved inputs.
    const float* pi = pend_in + int64_t(st.slot) * pend_in_slot;
    const float* pg = pend_gb + int64_t(st.slot) * pend_gb_slot;
    for (int32_t j = 0; j < st.accept; ++j) {
      const float* r = pi + int64_t(j) * kChannels;
      token<false>(S, r + kQk + kh * kHead, nullptr, r + 2 * kQk + h * kHead + col0, pg[j * 64 + h], pg[j * 64 + 32 + h], ov);
    }
    if (st.commit == 0 && st.accept > 0) store(sp, S);
    for (int32_t j = 0; j < st.n; ++j) {
      const int64_t row = st.tok0 + j;
      const float* r = in + row * kChannels;
      token<true>(S, r + kQk + kh * kHead, r + kh * kHead, r + 2 * kQk + h * kHead + col0, gb[row * 64 + h], gb[row * 64 + 32 + h], ov);
      esimd::block_store<float, C>(o + row * kInner + h * kHead + col0, ov);
      if (j + 1 == st.commit) store(sp, S);
    }
  }
};

constexpr int kScanCols = 4;

}  // namespace

sycl::event scan(sycl::queue& q, const Cache& c, int layer, const Plan& p, const Scratch& s) {
  if (p.n_slots == 0) return {};
  constexpr int C = kScanCols, kPer = kHead / C;
  const int64_t nl = c.n_layers;
  Scan<C> k{p.dev,
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
