// The decode GEMVs of the Qwen weight formats (layouts and tile decoders: xmx_fmt.hpp), instantiated per format in
// xmx_gemv_<format>.cpp so that they compile in parallel. The schedule is xmx.cpp's: one work-group of T threads per
// (8-column group, row group), K split over the threads and the partial sums merged in shared local memory; rows 1
// use dpas with repeat count 1, 2-32 rows 8-row tiles (RT of them a thread). Each thread walks its tiles in chunks
// (4 tiles of a 4-bit format, 2 of Q8_0I) loaded together, with the record of the chunk's scales.
#pragma once

#include <algorithm>
#include <stdexcept>

#include "kernels/xmx_fmt.hpp"

namespace eklow::xmx::fmt {

namespace detail {

constexpr int64_t kThreads1 = 4096, kThreadsN = 8192;  // as xmx.cpp's

template <int R>
SYCL_ESIMD_FUNCTION inline esimd::simd<float, R> load_scales(const float* p) {
  if constexpr (R == 1) {
    esimd::simd<float, 1> s;
    s[0] = p[0];
    return s;
  } else {
    return esimd::block_load<float, R>(p);
  }
}

// Tile U of the chunk (when it is inside [b0, b1)) against every row tile, then the next, at compile time. One row's
// activations (32 bytes) are loaded for the whole chunk before its first dpas (R == 1: the loads stay in flight
// together); 8-row tiles (256 bytes each, RT of them) are loaded where used, or the registers would not hold them.
template <WeightFormat F, int R, int RT, int U>
SYCL_ESIMD_FUNCTION inline void chunk_tiles(typename Dec<F>::Chunk& ch, int64_t bc, int64_t b0, int64_t b1, const uint8_t* ab, int64_t a_rt,
                                            const float* sb, int64_t s_rt, esimd::simd<uint8_t, 32> (&A1)[Dec<F>::CH],
                                            esimd::simd<float, 1> (&S1)[Dec<F>::CH], Acc<R>* acc) {
  if constexpr (U < Dec<F>::CH) {
    if (bc + U >= b0 && bc + U < b1) {
#pragma unroll
      for (int r = 0; r < RT; ++r) {
        if constexpr (R == 1) {
          Dec<F>::template tile<U, R>(acc[r], ch, A1[U], S1[U]);
        } else {
          esimd::simd<uint8_t, 32 * R> A = esimd::block_load<uint8_t, 32 * R>(ab + r * a_rt + (bc + U) * 256);
          esimd::simd<float, R> sa = esimd::block_load<float, R>(sb + r * s_rt + (bc + U) * 8);
          Dec<F>::template tile<U, R>(acc[r], ch, A, sa);
        }
      }
    }
    chunk_tiles<F, R, RT, U + 1>(ch, bc, b0, b1, ab, a_rt, sb, s_rt, A1, S1, acc);
  }
}

// The tiles [b0, b1) of one column group (bt, bm: its tiles and records) against RT row tiles of activations:
// row tile r's tile b is at ab + r * a_rt + b * 256 (R rows of 32 bytes: the first row's at 1 row), its scales at
// sb + r * s_rt + b * 8.
template <WeightFormat F, int R, int RT>
SYCL_ESIMD_FUNCTION inline void run_tiles(const uint8_t* bt, const uint8_t* bm, int64_t b0, int64_t b1, const uint8_t* ab, int64_t a_rt,
                                          const float* sb, int64_t s_rt, Acc<R>* acc) {
  using D = Dec<F>;
  constexpr int CH = D::CH, TB = D::TB;
  for (int64_t c = b0 / CH; c * CH < b1; ++c) {
    const int64_t bc = c * CH;
    typename D::Chunk ch;
    if constexpr (D::REC_T == 1) {
      D::template load<0>(ch, bt + bc * TB, bm + bc * geo(F).rec_bytes);
    } else {
      const uint8_t* rec = bm + (bc / D::REC_T) * geo(F).rec_bytes;
      if ((bc % D::REC_T) == 0) D::template load<0>(ch, bt + bc * TB, rec);
      else D::template load<1>(ch, bt + bc * TB, rec);
    }
    esimd::simd<uint8_t, 32> A1[CH];
    esimd::simd<float, 1> S1[CH];
    if constexpr (R == 1) {
#pragma unroll
      for (int u = 0; u < CH; ++u) {
        A1[u] = esimd::block_load<uint8_t, 32>(ab + (bc + u) * 256);
        S1[u][0] = sb[(bc + u) * 8];
      }
    }
    chunk_tiles<F, R, RT, 0>(ch, bc, b0, b1, ab, a_rt, sb, s_rt, A1, S1, acc);
  }
}

template <WeightFormat F, int R, int RT, int T>
void gemv_t(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  using D = Dec<F>;
  constexpr int CH = D::CH, TB = D::TB;
  constexpr int64_t rec_t = D::REC_T, rec_b = geo(F).rec_bytes;
  constexpr int slm_base = F == WeightFormat::IQ4_XS ? kIqTableBytes : 0;
  constexpr int slm_part = RT * 32 * R;  // bytes of one thread's partial sums
  const int64_t kb = a.k / 32, ng = w.n / 8, m = a.m, n = w.n;
  const int64_t row_groups = R == 1 ? 1 : (a.m + RT * 8 - 1) / (RT * 8);
  const int64_t per = ((kb + T - 1) / T + CH - 1) / CH * CH;
  const uint8_t* aq = reinterpret_cast<const uint8_t*>(a.q);
  const float* as = a.s;
  const uint8_t* bt = w.tiles;
  const uint8_t* bm = reinterpret_cast<const uint8_t*>(w.scales);
  q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(ng), size_t(row_groups), size_t(T)), sycl::range<3>(1, 1, size_t(T))),
                 [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                   constexpr int slm = slm_base + (T > 1 ? T * slm_part : 0);
                   if constexpr (slm > 0) esimd::slm_init<slm>();
                   const int64_t g = int64_t(it.get_global_id(0)), rg = int64_t(it.get_global_id(1));
                   const int t = int(it.get_local_id(2));
                   if constexpr (F == WeightFormat::IQ4_XS) {
                     D::fill_table(t, T);
                     esimd::barrier();
                   }
                   Acc<R> acc[RT];
#pragma unroll
                   for (int r = 0; r < RT; ++r) acc[r] = 0;
                   const int64_t b0 = int64_t(t) * per, b1 = std::min(kb, b0 + per);
                   if (b0 < b1)
                     run_tiles<F, R, RT>(bt + g * kb * TB, bm + g * (kb / rec_t) * rec_b, b0, b1, aq + rg * RT * kb * 256, kb * 256, as + rg * RT * kb * 8,
                                         kb * 8, acc);
                   if constexpr (T > 1) {
#pragma unroll
                     for (int r = 0; r < RT; ++r) esimd::slm_block_store<float, 8 * R>(slm_base + t * slm_part + r * 32 * R, acc[r]);
                     esimd::barrier();
                     // Thread t adds the T partials of the group's rows t, t + T, ...
                     for (int L = t; L < RT * R; L += T) {
                       const int64_t row = rg * RT * R + L;
                       if (row >= m) break;
                       const int r = L / R, i = L % R;
                       esimd::simd<float, 8> sum = 0;
                       for (int u = 0; u < T; ++u) sum += esimd::slm_block_load<float, 8>(slm_base + u * slm_part + r * 32 * R + i * 32);
                       esimd::block_store<float, 8>(out + row * n + g * 8, sum);
                     }
                   } else {
#pragma unroll
                     for (int L = 0; L < RT * R; ++L) {
                       const int64_t row = rg * RT * R + L;
                       if (row >= m) break;
                       esimd::block_store<float, 8>(out + row * n + g * 8, acc[L / R].template select<8, 1>((L % R) * 8));
                     }
                   }
                 });
}

template <WeightFormat F, int R, int RT>
void gemv_tdisp(int T, sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  switch (T) {
    case 1: gemv_t<F, R, RT, 1>(q, a, w, out); break;
    case 2: gemv_t<F, R, RT, 2>(q, a, w, out); break;
    case 4: gemv_t<F, R, RT, 4>(q, a, w, out); break;
    case 8: gemv_t<F, R, RT, 8>(q, a, w, out); break;
    case 16: gemv_t<F, R, RT, 16>(q, a, w, out); break;
    default: gemv_t<F, R, RT, 32>(q, a, w, out); break;
  }
}

// T: the largest power of two under the thread target / column groups, at most 32, and no more than the matrix's
// chunks along K.
inline int pick_t(int64_t target, int64_t groups, int64_t kb, int ch) {
  int T = 1;
  while (T < 32 && 2 * T <= target / groups) T *= 2;
  while (T > 1 && int64_t(T) * ch > kb) T /= 2;
  return T;
}

}  // namespace detail

// Entry points, explicitly instantiated per format (xmx_gemv_<format>.cpp). `a` is int8 activations (a.m <= 32).
template <WeightFormat F>
void gemv(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  using namespace detail;
  using D = Dec<F>;
  if (a.k % (32 * D::CH)) throw std::runtime_error("xmx: decode GEMV needs K a multiple of 32 x the format's chunk");
  const int64_t kb = a.k / 32, ng = w.n / 8, row_tiles = (a.m + 7) / 8;
  const int T = pick_t(a.m == 1 ? kThreads1 : kThreadsN, ng, kb, D::CH);
  if (a.m == 1) gemv_tdisp<F, 1, 1>(T, q, a, w, out);
  else if (row_tiles <= 1) gemv_tdisp<F, 8, 1>(T, q, a, w, out);
  else if (row_tiles <= 2) gemv_tdisp<F, 8, 2>(T, q, a, w, out);
  else gemv_tdisp<F, 8, 4>(T, q, a, w, out);
}

}  // namespace eklow::xmx::fmt
