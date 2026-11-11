#include "kernels/gdn_seq.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <stdexcept>

namespace eklow::gdn_seq {

namespace esimd = sycl::ext::intel::esimd;

namespace {

float* alloc_floats(sycl::queue& q, int64_t n, const char* what) {
  auto* p = sycl::malloc_device<float>(size_t(std::max<int64_t>(n, 1)), q);
  if (!p) throw std::runtime_error(std::string("gdn_seq: out of device memory for ") + what);
  return p;
}

// M batches of W lanes to M sums.
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

// The scan: a thread holds C columns of one head's state and walks the chunk's tokens. The recurrence is kernels/gdn.cpp's
// (Cols::token) with the decay folded in: kv = eg (S^T k); d = beta (v - kv); o = eg (S^T q) + d (k . q); S = eg S + k d^T.
constexpr int kCols = 2;

struct Scan {
  float* ssm;            // slot 0, this layer: [H][128][128]
  int64_t slot_floats;   // floats between slots
  const Run* runs;
  const float *in, *gb;
  float* o;
  int32_t H;

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    constexpr int C = kCols, NA = 2 * C + 1;
    const int32_t g0 = int32_t(it.get_global_id(0)), r = g0 / H, h = g0 % H, col0 = int32_t(it.get_global_id(1)) * C;
    const Run run = runs[r];
    const int32_t W = 3 * H * kHead, HW = H * kHead;
    float* sp = ssm + int64_t(run.slot) * slot_floats;
    esimd::simd<float, 128> S[C];
#pragma unroll
    for (int c = 0; c < C; ++c) S[c] = esimd::block_load<float, 128>(sp + (int64_t(h) * kHead + col0 + c) * kHead);
    const float* rp = in + int64_t(run.row0) * W;
    const float* g = gb + int64_t(run.row0) * 2 * H;
    float* op = o + int64_t(run.row0) * HW + int64_t(h) * kHead + col0;
    // Each token's operands are loaded a token ahead, so that their latency hides behind the previous token's arithmetic
    // (the recurrence leaves nothing else to overlap them with).
    esimd::simd<float, 128> qn, kn;
    esimd::simd<float, C> vn;
    float egn = 0, betan = 0;
    auto fetch = [&](const float* r, const float* gg) SYCL_ESIMD_FUNCTION {
      qn = esimd::block_load<float, 128>(r + h * kHead);
      kn = esimd::block_load<float, 128>(r + HW + h * kHead);
      vn = esimd::block_load<float, C>(r + 2 * HW + h * kHead + col0);
      egn = gg[h], betan = gg[H + h];
    };
    if (run.n > 0) fetch(rp, g);
    for (int32_t t = 0; t < run.n; ++t, rp += W, g += 2 * H, op += HW) {
      esimd::simd<float, 128> q = qn, k = kn;
      esimd::simd<float, C> v = vn;
      const float eg = egn, beta = betan;
      if (t + 1 < run.n) fetch(rp + W, g + 2 * H);
      esimd::simd<float, 16 * NA> acc;
#pragma unroll
      for (int c = 0; c < C; ++c) {
        acc.template select<16, 1>(16 * c) = dot16(S[c], k);
        acc.template select<16, 1>(16 * (C + c)) = dot16(S[c], q);
      }
      acc.template select<16, 1>(16 * 2 * C) = dot16(k, q);
      esimd::simd<float, NA> sums = hsum<NA, 16>(acc);
      esimd::simd<float, C> kv = eg * sums.template select<C, 1>(0);
      esimd::simd<float, C> d = (v - kv) * beta;
      const esimd::simd<float, C> ov = eg * sums.template select<C, 1>(C) + d * float(sums[2 * C]);
      esimd::block_store<float, C>(op, ov);
#pragma unroll
      for (int c = 0; c < C; ++c) {
        const float dc = d[c];
#pragma unroll
        for (int i = 0; i < 8; ++i)
          S[c].template select<16, 1>(16 * i) = S[c].template select<16, 1>(16 * i) * eg + k.template select<16, 1>(16 * i) * dc;
      }
    }
#pragma unroll
    for (int c = 0; c < C; ++c) esimd::block_store<float, 128>(sp + (int64_t(h) * kHead + col0 + c) * kHead, S[c]);
  }
};

}  // namespace

State alloc_state(sycl::queue& q, int n_layers, int n_heads, int n_slots) {
  State s;
  s.n_layers = n_layers, s.n_heads = n_heads, s.n_slots = n_slots;
  s.ssm = alloc_floats(q, s.ssm_floats() * n_slots, "the state");
  s.conv = alloc_floats(q, s.conv_floats() * n_slots, "the conv state");
  for (int i = 0; i < n_slots; ++i) reset(q, s, i);
  return s;
}

void free_state(sycl::queue& q, State& s) {
  if (s.ssm) sycl::free(s.ssm, q);
  if (s.conv) sycl::free(s.conv, q);
  s = State{};
}

void reset(sycl::queue& q, const State& s, int slot) {
  q.memset(s.ssm + slot * s.ssm_floats(), 0, size_t(s.ssm_floats()) * 4);
  q.memset(s.conv + slot * s.conv_floats(), 0, size_t(s.conv_floats()) * 4);
}

void copy(sycl::queue& q, const State& src, int src_slot, const State& dst, int dst_slot) {
  if (src.n_layers != dst.n_layers || src.n_heads != dst.n_heads) throw std::runtime_error("gdn_seq::copy: states differ in shape");
  q.memcpy(dst.ssm + dst_slot * dst.ssm_floats(), src.ssm + src_slot * src.ssm_floats(), size_t(src.ssm_floats()) * 4);
  q.memcpy(dst.conv + dst_slot * dst.conv_floats(), src.conv + src_slot * src.conv_floats(), size_t(src.conv_floats()) * 4);
}

Scratch alloc_scratch(sycl::queue& q, int64_t max_rows, int n_heads) {
  Scratch s;
  s.in = alloc_floats(q, max_rows * 3 * n_heads * kHead, "the chunk's inputs");
  s.gb = alloc_floats(q, max_rows * 2 * n_heads, "the chunk's gates");
  s.o = alloc_floats(q, max_rows * n_heads * kHead, "the chunk's outputs");
  return s;
}

void free_scratch(sycl::queue& q, Scratch& s) {
  for (float* p : {s.in, s.gb, s.o})
    if (p) sycl::free(p, q);
  s = Scratch{};
}

void layer(sycl::queue& q, const Weights& w, const State& s, int layer, const Run* runs, int32_t n_runs, const int32_t* run_of_row,
           int64_t rows, const float* x, const float* z, const float* ab, const Scratch& sc, xmx::Activations& out) {
  if (rows <= 0 || n_runs <= 0) return;
  const int32_t H = s.n_heads, W = 3 * H * kHead, HW = H * kHead, N = int32_t(rows);
  // This layer's conv state of slot 0, and the floats between slots.
  float* cs0 = s.conv + int64_t(layer) * 3 * W;
  const int64_t cs_slot = s.conv_floats();
  const float *cw = w.conv_w, *ssm_a = w.ssm_a, *dt = w.dt_bias;
  float *in = sc.in, *gb = sc.gb;
  // Conv, silu and the l2 norms: a work-group per (row, 128-channel slice), q's H slices, then k's, then v's; the v slices'
  // first work-item also writes the head's exp(g) and beta. A row reads earlier rows of its own run, or its sequence's
  // conv state before the run's first row.
  q.parallel_for(sycl::nd_range<1>(size_t(N) * 3 * H * kHead, kHead), [=](sycl::nd_item<1> it) {
    const int32_t g = int32_t(it.get_group(0)), t = g / (3 * H), slice = g % (3 * H), i = int32_t(it.get_local_id(0));
    const Run run = runs[run_of_row[t]];
    const int32_t c = slice * kHead + i, local = t - run.row0;
    const float* cs = cs0 + int64_t(run.slot) * cs_slot;
    float acc = 0;
#pragma unroll
    for (int j = 0; j < kTaps; ++j) {
      const int32_t lr = local - (kTaps - 1) + j;
      const float v = lr >= 0 ? x[int64_t(run.row0 + lr) * W + c] : cs[(3 + lr) * W + c];
      acc += cw[c * kTaps + j] * v;
    }
    float y = acc / (1.0f + sycl::exp(-acc));
    if (slice < 2 * H) {
      const float ss = sycl::reduce_over_group(it.get_group(), y * y, sycl::plus<float>());
      y *= 1.0f / sycl::sqrt(ss + 1e-6f);
      if (slice < H) y *= 0.08838834764831845f;  // 1/sqrt(128)
    } else if (i == 0) {
      const int32_t h = slice - 2 * H;
      const float a = ab[int64_t(t) * 2 * H + h] + dt[h], b = ab[int64_t(t) * 2 * H + H + h];
      const float sp = a > 20.0f ? a : sycl::log1p(sycl::exp(a));
      gb[int64_t(t) * 2 * H + h] = sycl::exp(ssm_a[h] * sp);
      gb[int64_t(t) * 2 * H + H + h] = 1.0f / (1.0f + sycl::exp(-b));
    }
    in[int64_t(t) * W + c] = y;
  });
  // Each run's conv state: its last three raw rows (with fewer rows than taps, the older state shifted down). Each work-item
  // reads indices above the ones it has written, so the shift is in place.
  q.parallel_for(sycl::range<2>(size_t(n_runs), size_t(W)), [=](sycl::id<2> id) {
    const Run run = runs[id[0]];
    const int32_t c = int32_t(id[1]);
    float* cs = cs0 + int64_t(run.slot) * cs_slot;
    for (int32_t i = 0; i < 3; ++i) {
      const int32_t lr = run.n - 3 + i;
      cs[i * W + c] = lr >= 0 ? x[int64_t(run.row0 + lr) * W + c] : cs[(i + run.n) * W + c];
    }
  });
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(n_runs) * H, size_t(kHead / kCols)), sycl::range<2>(1, 8)),
                 Scan{s.ssm + int64_t(layer) * H * kHead * kHead, s.ssm_floats(), runs, in, gb, sc.o, H});
  // y = rmsnorm(o_h) * norm_w * silu(z_h), in the fp16 activation layout.
  xmx::set_rows(out, rows, HW, true);
  sycl::half* ah = out.h;
  const float *o = sc.o, *nw = w.norm_w;
  const float eps = w.eps;
  q.parallel_for(sycl::nd_range<1>(size_t(N) * H * kHead, kHead), [=](sycl::nd_item<1> it) {
    const int32_t g = int32_t(it.get_group(0)), t = g / H, h = g % H, i = int32_t(it.get_local_id(0));
    const int64_t at = int64_t(t) * HW + h * kHead + i;
    const float v = o[at];
    const float ss = sycl::reduce_over_group(it.get_group(), v * v, sycl::plus<float>());
    const float zv = z[at];
    const float y = v / sycl::sqrt(ss / float(kHead) + eps) * nw[i] * (zv / (1.0f + sycl::exp(-zv)));
    ah[xmx::act16_index(t, h * kHead + i, HW / 16)] = sycl::half(y);
  });
}

}  // namespace eklow::gdn_seq
