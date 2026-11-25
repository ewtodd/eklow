// Quantize + decode GEMV in one launch, over one or several Q8_0I matrices that share K (DECISIONS 2026-11-25, "Inkling's
// decode GEMVs"). A thread quantizes the fp32 activations of its own K slice to int8 per 32-block as xmx::quantize does
// (same scale, same rounding: the results are bit-identical to quantize + matmul) and feeds the dpas from registers, so
// there is no quantize launch and no int8 round trip; it covers NG adjacent 8-column groups, so one quantization serves
// NG groups and NG chunks of weights are in flight; the chunk after the one being multiplied is loaded before it.
#pragma once

#include <algorithm>
#include <stdexcept>

#include "kernels/xmx_fmt.hpp"

namespace eklow::xmx::fmt {

namespace fused {

constexpr int kMaxJobs = 4;

struct JobDev {
  const uint8_t* bt;   // tiles
  const uint8_t* bm;   // fp16 scales
  const float* x;      // fp32 activations [m][k]
  float* out;          // [m][n]
  int32_t n;           // columns
  int32_t slot_end;    // first slot of the next job (a slot is NG column groups)
};

// round half away from zero, as sycl::round (the scalar quantize's rule), on 32 lanes
SYCL_ESIMD_FUNCTION inline esimd::simd<uint8_t, 32> quant32(const esimd::simd<float, 32>& x, float& d_out) {
  const float amax = esimd::hmax<float>(esimd::abs(x));
  const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
  const esimd::simd<float, 32> v = x * id;
  esimd::simd<int, 32> ti = v;  // truncates toward zero
  const esimd::simd<float, 32> diff = v - esimd::simd<float, 32>(ti);
  ti.merge(ti + 1, diff >= 0.5f);
  ti.merge(ti - 1, diff <= -0.5f);
  d_out = d;
  esimd::simd<int8_t, 32> q8 = ti;
  return q8.template bit_cast_view<uint8_t>();
}

template <int R, int NG, int CH>
struct Stage {
  esimd::simd<uint8_t, CH * 256> t[NG];
  esimd::simd<sycl::half, 8 * CH> s[NG];
  esimd::simd<float, 32> x[CH][R];
};

template <int R, int NG, int CH>
SYCL_ESIMD_FUNCTION inline void load_stage(Stage<R, NG, CH>& st, const JobDev& j, int64_t g0, int64_t kb, int64_t k, int64_t c, int m) {
#pragma unroll
  for (int u = 0; u < NG; ++u) {
    st.t[u] = esimd::block_load<uint8_t, CH * 256>(j.bt + ((g0 + u) * kb + c * CH) * 256);
    st.s[u] = esimd::block_load<sycl::half, 8 * CH>(reinterpret_cast<const sycl::half*>(j.bm) + ((g0 + u) * kb + c * CH) * 8);
  }
#pragma unroll
  for (int b = 0; b < CH; ++b)
#pragma unroll
    for (int r = 0; r < R; ++r)
      if (r < m) st.x[b][r] = esimd::block_load<float, 32>(j.x + r * k + (c * CH + b) * 32);
}

template <int R, int NG, int CH, int U>
SYCL_ESIMD_FUNCTION inline void compute_stage(Stage<R, NG, CH>& st, Acc<R>* acc, int m) {
  if constexpr (U < CH) {
    // the activations of block U: R rows (rows from m on: zeros), R scales
    esimd::simd<uint8_t, 32 * R> A = 0;
    esimd::simd<float, R> sa = 0;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      if (r < m) {
        float d;
        A.template select<32, 1>(r * 32) = quant32(st.x[U][r], d);
        sa[r] = d;
      }
    }
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      esimd::simd<uint8_t, 256> B = st.t[g].template select<256, 1>(U * 256);
      esimd::simd<float, 8 * R> cf = dp<dpas_argument_type::s8, R, 256>(B, A);
      esimd::simd<float, 8> sv = st.s[g].template select<8, 1>(U * 8);
      axpy<R>(acc[g], cf, sv, sa);
    }
    compute_stage<R, NG, CH, U + 1>(st, acc, m);
  }
}

template <int R, int NG, int CH, int T>
void launch(sycl::queue& q, const JobDev (&jobs)[kMaxJobs], int nj, int64_t k, int m) {
  constexpr int slm_part = NG * 32 * R;  // bytes of one thread's partial sums
  const int64_t kb = k / 32;
  const int64_t slots = jobs[nj - 1].slot_end;
  const JobDev j0 = jobs[0], j1 = jobs[1], j2 = jobs[2], j3 = jobs[3];
  const sycl::event ev = q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(slots), size_t(T)), sycl::range<2>(1, size_t(T))), [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
    if constexpr (T > 1) esimd::slm_init<T * slm_part>();
    const int64_t slot = int64_t(it.get_global_id(0));
    const int t = int(it.get_local_id(1));
    // the job of this slot, and the slot's first column group inside it
    const JobDev* J = &j0;
    int64_t first = 0;
    if (slot >= j0.slot_end) { J = &j1; first = j0.slot_end; }
    if (nj > 2 && slot >= j1.slot_end) { J = &j2; first = j1.slot_end; }
    if (nj > 3 && slot >= j2.slot_end) { J = &j3; first = j2.slot_end; }
    const JobDev j = *J;
    const int64_t g0 = (slot - first) * NG;
    Acc<R> acc[NG];
#pragma unroll
    for (int u = 0; u < NG; ++u) acc[u] = 0;
    // chunks t, t + T, ...: the threads of a group read neighbouring chunks together
    const int64_t nch = kb / CH;
    if (t < nch) {
      const int64_t c0 = t, c1 = nch;
      Stage<R, NG, CH> A, B;
      load_stage<R, NG, CH>(A, j, g0, kb, k, c0, m);
      for (int64_t c = c0; c < c1; c += 2 * T) {
        if (c + T < c1) load_stage<R, NG, CH>(B, j, g0, kb, k, c + T, m);
        compute_stage<R, NG, CH, 0>(A, acc, m);
        if (c + T < c1) {
          if (c + 2 * T < c1) load_stage<R, NG, CH>(A, j, g0, kb, k, c + 2 * T, m);
          compute_stage<R, NG, CH, 0>(B, acc, m);
        }
      }
    }
    if constexpr (T > 1) {
#pragma unroll
      for (int u = 0; u < NG; ++u) esimd::slm_block_store<float, 8 * R>(t * slm_part + u * 32 * R, acc[u]);
      esimd::barrier();
      for (int L = t; L < NG * R; L += T) {
        const int u = L / R, r = L % R;
        if (r >= m) continue;
        esimd::simd<float, 8> sum = 0;
        for (int v = 0; v < T; ++v) sum += esimd::slm_block_load<float, 8>(v * slm_part + u * 32 * R + r * 32);
        esimd::block_store<float, 8>(j.out + int64_t(r) * j.n + (g0 + u) * 8, sum);
      }
    } else {
#pragma unroll
      for (int L = 0; L < NG * R; ++L)
        if (L % R < m) esimd::block_store<float, 8>(j.out + int64_t(L % R) * j.n + (g0 + L / R) * 8, acc[L / R].template select<8, 1>((L % R) * 8));
    }
  });
  if (g_event_log) g_event_log->push_back(ev);
}


}  // namespace fused

}  // namespace eklow::xmx::fmt
