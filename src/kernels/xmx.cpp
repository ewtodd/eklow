#include "kernels/xmx.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <stdexcept>

namespace eklow::xmx {

namespace esimd = sycl::ext::intel::esimd;
namespace xmxd = sycl::ext::intel::esimd::xmx;
using xmxd::dpas_argument_type;

namespace {

constexpr int kTileBytes = 128;   // weight tile: 32 K x 8 N nibbles
constexpr int kATileBytes = 256;  // activation tile: 8 rows x 32 K bytes

// Tiles of one 8-column group run contiguously along K, so a decode thread's consecutive blocks are adjacent.
inline int64_t tile_at(int64_t b, int64_t g, int64_t kb) { return g * kb + b; }

}  // namespace

Weight repack_q4_0(sycl::queue& q, const ref::Weight& raw, bool interleave) {
  if (raw.type != gguf::Type::Q4_0) throw std::runtime_error("xmx: repack_q4_0 needs a Q4_0 matrix");
  if (raw.k % 32 || raw.n % 8) throw std::runtime_error("xmx: Q4_0 matrix shape not a multiple of 32 x 8");
  if (interleave && raw.n % 16) throw std::runtime_error("xmx: interleaved halves not a multiple of 8 rows");
  const int64_t half_rows = raw.n / 2;
  Weight w{raw.k, raw.n, nullptr, nullptr};
  const int64_t kb = raw.k / 32, ng = raw.n / 8;
  w.tiles = sycl::malloc_device<uint8_t>(size_t(ng * kb * kTileBytes), q);
  w.scales = sycl::malloc_device<sycl::half>(size_t(ng * kb * 8), q);
  if (!w.tiles || !w.scales) throw std::runtime_error("xmx: out of device memory repacking weights");
  const uint8_t* src = raw.data;
  const uint64_t row_bytes = raw.row_bytes;
  uint8_t* tiles = w.tiles;
  sycl::half* scales = w.scales;
  q.parallel_for(sycl::range<2>(size_t(ng), size_t(kb)), [=](sycl::id<2> id) {
     const int64_t g = int64_t(id[0]), b = int64_t(id[1]);
     uint32_t words[32] = {};
     for (int n = 0; n < 8; ++n) {
       const int64_t row = interleave ? (g % 2) * half_rows + (g / 2) * 8 + n : g * 8 + n;
       const uint8_t* blk = src + uint64_t(row) * row_bytes + uint64_t(b) * 18;
       scales[tile_at(b, g, kb) * 8 + n] = sycl::bit_cast<sycl::half>(uint16_t(blk[0] | blk[1] << 8));
       for (int k = 0; k < 32; ++k) {
         const uint32_t q4 = (k < 16 ? blk[2 + k] & 15 : blk[2 + k - 16] >> 4) ^ 8;
         const int idx = ((k / 8) * 8 + n) * 8 + k % 8;  // nibble index in the tile
         words[idx / 8] |= q4 << (4 * (idx % 8));
       }
     }
     uint32_t* dst = reinterpret_cast<uint32_t*>(tiles + tile_at(b, g, kb) * kTileBytes);
     for (int i = 0; i < 32; ++i) dst[i] = words[i];
   }).wait();
  return w;
}

void free_weight(sycl::queue& q, Weight& w) {
  if (w.tiles) sycl::free(w.tiles, q);
  if (w.scales) sycl::free(w.scales, q);
  w.tiles = nullptr;
  w.scales = nullptr;
}

Activations alloc_activations(sycl::queue& q, int64_t max_m, int64_t k) {
  if (k % 32) throw std::runtime_error("xmx: activation width not a multiple of 32");
  Activations a;
  a.k = k;
  a.capacity = (max_m + 255) / 256 * 256 * k;
  a.q = sycl::malloc_device<int8_t>(size_t(a.capacity), q);
  a.s = sycl::malloc_device<float>(size_t(a.capacity / 32), q);
  a.h = sycl::malloc_device<sycl::half>(size_t(a.capacity), q);
  if (!a.q || !a.s || !a.h) throw std::runtime_error("xmx: out of device memory for activations");
  return a;
}

void free_activations(sycl::queue& q, Activations& a) {
  if (a.q) sycl::free(a.q, q);
  if (a.s) sycl::free(a.s, q);
  if (a.h) sycl::free(a.h, q);
  a.q = nullptr;
  a.s = nullptr;
  a.h = nullptr;
}

void quantize(sycl::queue& q, const float* x, int64_t m, Activations& a) {
  set_rows(a, m, a.k);
  const int64_t m_pad = a.m_pad, kb = a.k / 32, k = a.k;
  if (a.fp16) {
    sycl::half* hd = a.h;
    q.parallel_for(sycl::range<2>(size_t(m), size_t(k)), [=](sycl::id<2> id) {
      const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
      hd[act16_index(r, c, k / 16)] = sycl::half(x[r * k + c]);
    });
    return;
  }
  int8_t* qd = a.q;
  float* sd = a.s;
  q.parallel_for(sycl::range<2>(size_t(m_pad), size_t(kb)), [=](sycl::id<2> id) {
     const int64_t r = int64_t(id[0]), b = int64_t(id[1]);
     int8_t* dst = qd + ((r / 8) * kb + b) * kATileBytes + (r % 8) * 32;
     float* sc = sd + ((r / 8) * kb + b) * 8 + r % 8;
     if (r >= m) {
       for (int j = 0; j < 32; ++j) dst[j] = 0;
       *sc = 0;
       return;
     }
     float v[32], amax = 0;
     for (int j = 0; j < 32; ++j) {
       v[j] = x[r * k + b * 32 + j];
       amax = sycl::fmax(amax, sycl::fabs(v[j]));
     }
     *sc = quantize_run(v, amax, dst);
   });
}

void set_rows(Activations& a, int64_t m, int64_t k) {
  if (k % 32) throw std::runtime_error("xmx: activation width not a multiple of 32");
  a.fp16 = m > kFp16Rows;
  const int64_t m_pad = a.fp16 ? (m + 255) / 256 * 256 : (m + 31) / 32 * 32;
  if (m_pad * k > a.capacity) throw std::runtime_error("xmx: activation buffer too small");
  a.m = m;
  a.m_pad = m_pad;
  a.k = k;
}

namespace {

// The fp16 GEMM: a work-group of WM x WN threads computes (16 WM) x (32 WN) of the output, each
// thread 16 x 32 (two 8-row tiles by four 8-column tiles). Per Q4_0 block the threads share out
// the group's weight tiles (dequantized to fp16 VNNI B operands, per K half) and activation tiles,
// writing them to one of two shared-memory buffers: each thread issues its global loads for block
// b+1 before block b's dpas and stores them after, so their latency hides behind the dpas.
// (DECISIONS 2026-10-24, "fp16 GEMM": 48-52 TFLOPS on the 12B's shapes, against 23-37 TOPS for
// the int8 GEMM, whose per-block scaling the vector units cannot keep up with.)
template <int WM, int WN>
struct Gemm16Kernel {
  static constexpr int NT = WM * WN, BT = WN * 4, AT = WM * 2;  // threads; weight / activation 8-tiles
  static constexpr int kBPieces = BT * 2, kAPieces = AT * 2;     // per block: (tile, K half) pieces
  static constexpr int BP = (kBPieces + NT - 1) / NT, AP = (kAPieces + NT - 1) / NT;
  static constexpr uint32_t kBBuf = kBPieces * 256, kBuf = kBBuf + kAPieces * 256;
  const sycl::half* a;
  const uint8_t* bt;
  const sycl::half* bs;
  float* out;
  int32_t kb, ng, m, n;
  sycl::half* glu = nullptr;  // GeGLU epilogue: fp16 activations of width n/2 instead of out

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<2 * kBuf>();
    const int tid = int(it.get_local_linear_id()), wi = tid / WN, wj = tid % WN;
    const int tm = int(it.get_group(0)), tn = int(it.get_group(1));
    esimd::simd<uint32_t, 16> D[BP];
    esimd::simd<sycl::half, 8> S[BP];
    esimd::simd<sycl::half, 128> Araw[AP];
    auto load = [&](int b) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < BP; ++i) {
        const int p = tid + i * NT, g = tn * BT + p / 2, h = p % 2;
        if (p >= kBPieces) break;
        D[i] = esimd::block_load<uint32_t, 16>(reinterpret_cast<const uint32_t*>(bt + tile_at(b, g, kb) * kTileBytes + h * 64));
        S[i] = esimd::block_load<sycl::half, 8>(bs + tile_at(b, g, kb) * 8);
      }
#pragma unroll
      for (int i = 0; i < AP; ++i) {
        const int p = tid + i * NT, rt = tm * AT + p / 2, h = p % 2;
        if (p >= kAPieces) break;
        Araw[i] = esimd::block_load<sycl::half, 128>(a + (int64_t(rt) * (2 * kb) + 2 * b + h) * 128);
      }
    };
    auto store = [&](uint32_t buf) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < BP; ++i) {
        const int p = tid + i * NT;
        if (p >= kBPieces) break;
        esimd::simd<sycl::half, 16> sc2;
        sc2.select<8, 2>(0) = S[i];
        sc2.select<8, 2>(1) = S[i];
        // One dword holds a column's 8 nibbles of one K octet, each the s4 value; flipping bit 3
        // gives back the Q4_0 nibble q (value q - 8). Each pair of nibbles becomes a VNNI word, q
        // as the fp16 1024 + q (exponent bits 0x6400 over the low mantissa).
        esimd::simd<uint32_t, 16> Dq = D[i] ^ 0x88888888u;
        esimd::simd<uint32_t, 64> W;
#pragma unroll
        for (int g2 = 0; g2 < 2; ++g2)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const esimd::simd<uint32_t, 8> t = esimd::simd<uint32_t, 8>(Dq.template select<8, 1>(g2 * 8)) >> (8 * q);
            W.select<8, 1>((g2 * 4 + q) * 8) = (t & 0xF) | ((t << 12) & 0xF0000) | 0x64006400u;
          }
        esimd::simd<sycl::half, 128> Wh = W.template bit_cast_view<sycl::half>();
        Wh = (Wh - sycl::half(1032)) * sc2.replicate<8>();
        esimd::slm_block_store<sycl::half, 128>(buf + p * 256, Wh);
      }
#pragma unroll
      for (int i = 0; i < AP; ++i)
        if (tid + i * NT < kAPieces) esimd::slm_block_store<sycl::half, 128>(buf + kBBuf + (tid + i * NT) * 256, Araw[i]);
    };
    esimd::simd<float, 64> acc[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[r][c] = 0;
    load(0);
    store(0);
    esimd::barrier();
    for (int b = 0; b < kb; ++b) {
      const uint32_t cur = (b & 1) * kBuf, nxt = kBuf - cur;
      if (b + 1 < kb) load(b + 1);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        esimd::simd<sycl::half, 128> A[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) A[r] = esimd::slm_block_load<sycl::half, 128>(cur + kBBuf + ((wi * 2 + r) * 2 + h) * 256);
        // The next B tile is loaded before the current dpas, so its SLM latency hides behind it.
        esimd::simd<sycl::half, 128> B = esimd::slm_block_load<sycl::half, 128>(cur + (wj * 4 * 2 + h) * 256);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          esimd::simd<sycl::half, 128> Bn;
          if (c + 1 < 4) Bn = esimd::slm_block_load<sycl::half, 128>(cur + ((wj * 4 + c + 1) * 2 + h) * 256);
#pragma unroll
          for (int r = 0; r < 2; ++r) acc[r][c] = xmxd::dpas<8, 8, float>(acc[r][c], B, A[r]);
          B = Bn;
        }
      }
      if (b + 1 < kb) store(nxt);
      esimd::barrier();
    }
    const int row_tile0 = (tm * WM + wi) * 2;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int row = (row_tile0 + r) * 8 + i;
        if (row >= m) continue;
        if (glu) {
          // Column tiles 0 and 2 are gate, 1 and 3 the matching up (an interleaved repack).
#pragma unroll
          for (int c = 0; c < 4; c += 2) {
            const esimd::simd<float, 8> g = acc[r][c].template select<8, 1>(i * 8), u = acc[r][c + 1].template select<8, 1>(i * 8);
            // tanh(x) = 1 - 2 / (exp(2x) + 1); ESIMD has no tanh.
            const esimd::simd<float, 8> x2 = 2.0f * 0.7978845608028654f * (g + 0.044715f * g * g * g);
            const esimd::simd<float, 8> h = 0.5f * g * (2.0f - 2.0f / (esimd::exp(x2) + 1.0f)) * u;
            const int col = ((tn * WN + wj) * 2 + c / 2) * 8;
            esimd::block_store<sycl::half, 8>(glu + act16_index(row, col, n / 32), esimd::simd<sycl::half, 8>(h));
          }
          continue;
        }
#pragma unroll
        for (int c = 0; c < 4; ++c)
          esimd::block_store<float, 8>(out + int64_t(row) * n + ((tn * WN + wj) * 4 + c) * 8, acc[r][c].template select<8, 1>(i * 8));
      }
  }
};

void gemm16(sycl::queue& q, const Activations& a, const Weight& w, float* out, sycl::half* glu = nullptr) {
  // At 1,024 rows the 256 x 128 tile leaves the narrow matrices (N 3,840) with 4 x 30
  // work-groups, under two waves of the 64 the card holds at once; the 128 x 128 tile doubles
  // the groups with the same per-thread work.
  constexpr int64_t kMinGroups = 256;
  if ((a.m_pad / 256) * (w.n / 128) < kMinGroups) {
    constexpr int WM = 8, WN = 4;
    if (a.m_pad % (16 * WM) || w.n % (32 * WN)) throw std::runtime_error("xmx: fp16 GEMM tile does not divide the problem");
    Gemm16Kernel<WM, WN> k{a.h, w.tiles, w.scales, out, int32_t(a.k / 32), int32_t(w.n / 8), int32_t(a.m), int32_t(w.n), glu};
    q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(a.m_pad / 16), size_t(w.n / 32)), sycl::range<2>(WM, WN)), k);
    return;
  }
  constexpr int WM = 16, WN = 4;
  if (a.m_pad % (16 * WM) || w.n % (32 * WN)) throw std::runtime_error("xmx: fp16 GEMM tile does not divide the problem");
  Gemm16Kernel<WM, WN> k{a.h, w.tiles, w.scales, out, int32_t(a.k / 32), int32_t(w.n / 8), int32_t(a.m), int32_t(w.n), glu};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(a.m_pad / 16), size_t(w.n / 32)), sycl::range<2>(WM, WN)), k);
}

// Column-group schedule for 2-32 rows (decode of several sequences): one work-group of T threads
// per (row group of up to 32 rows, 8-column group), the T threads splitting K and merging their
// partials in shared local memory, so no partial buffer and no second kernel. Each weight tile is
// read once per row group and applied to all its rows.
template <int RT, int T>
void gemv(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  const int64_t kb = a.k / 32, ng = w.n / 8, m = a.m, n = w.n;
  const int64_t row_groups = (a.m + RT * 8 - 1) / (RT * 8);
  const int64_t per = (kb + T - 1) / T;
  const int8_t* aq = a.q;
  const float* as = a.s;
  const uint8_t* bt = w.tiles;
  const sycl::half* bs = w.scales;
  q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(ng), size_t(row_groups), size_t(T)), sycl::range<3>(1, 1, size_t(T))),
                 [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                   if constexpr (T > 1) esimd::slm_init<T * RT * 256>();
                   const int64_t rg = int64_t(it.get_global_id(1)), g = int64_t(it.get_global_id(0));
                   const int t = int(it.get_local_id(2));
                   esimd::simd<float, 64> acc[RT];
#pragma unroll
                   for (int r = 0; r < RT; ++r) acc[r] = 0;
                   const int64_t b0 = int64_t(t) * per, b1 = std::min(kb, b0 + per);
                   for (int64_t b = b0; b < b1; ++b) {
                     const int64_t wt = tile_at(b, g, kb);
                     esimd::simd<uint8_t, kTileBytes> B = esimd::block_load<uint8_t, kTileBytes>(bt + wt * kTileBytes);
                     esimd::simd<sycl::half, 8> h = esimd::block_load<sycl::half, 8>(bs + wt * 8);
                     esimd::simd<float, 8> sb = h;
#pragma unroll
                     for (int r = 0; r < RT; ++r) {
                       const int64_t at = (rg * RT + r) * kb + b;
                       esimd::simd<uint8_t, kATileBytes> A =
                           esimd::block_load<uint8_t, kATileBytes>(reinterpret_cast<const uint8_t*>(aq) + at * kATileBytes);
                       esimd::simd<float, 8> sa = esimd::block_load<float, 8>(as + at * 8);
                       esimd::simd<int, 64> ci =
                           xmxd::dpas<8, 8, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B, A);
                       esimd::simd<float, 64> cf = ci;
#pragma unroll
                       for (int i = 0; i < 8; ++i) {
                         float sc = sa[i];
                         acc[r].template select<8, 1>(i * 8) += cf.template select<8, 1>(i * 8) * (sb * sc);
                       }
                     }
                   }
                   if constexpr (T > 1) {
                     #pragma unroll
                     for (int r = 0; r < RT; ++r) esimd::slm_block_store<float, 64>(t * RT * 256 + r * 256, acc[r]);
                     esimd::barrier();
                     // Thread t adds the T partials of the group's rows t, t + T, ... (L in rows).
                     for (int L = t; L < RT * 8; L += T) {
                       const int64_t row = rg * RT * 8 + L;
                       if (row >= m) break;
                       const int r = L / 8, i = L % 8;
                       esimd::simd<float, 8> sum = 0;
                       for (int u = 0; u < T; ++u) sum += esimd::slm_block_load<float, 8>(u * RT * 256 + r * 256 + i * 32);
                       esimd::block_store<float, 8>(out + row * n + g * 8, sum);
                     }
                   } else {
                     // T = 1: the thread owns all of K, so acc[r] is already the group's full partial.
                     for (int L = 0; L < RT * 8; ++L) {
                       const int64_t row = rg * RT * 8 + L;
                       if (row >= m) break;
                       const int r = L / 8, i = L % 8;
                       esimd::block_store<float, 8>(out + row * n + g * 8, acc[r].template select<8, 1>(i * 8));
                     }
                   }
                 });
}

// The flat grid of the old code ran about 8192 threads per GEMV whatever the shape; with one
// work-group of T threads per 8-column group a comparable count is kept only if T follows the
// group count, so matmul() derives it from the matrix's shape: at 4096 the widest matrix
// (gate|up, 3840 groups) drops to T = 1 and the narrowest (480 groups) rises to 8. One value
// cannot win for one row and several: 4096 suits the one-row GEMVs (ab.sh decode
// +2.8/+2.0/+2.2 vs +1.9/+2.0/+2.2 at 8192 and -1.9/-1.7/-1.8 at 16384), but the wide two-row
// GEMVs want 8192 (gate|up 2 rows: T = 2, 291 GB/s, vs T = 1 at 232). Measured on fiona
// 2026-10-26.
constexpr int64_t kThreads1 = 4096;  // one row (gemv1)
constexpr int64_t kThreadsN = 8192;  // 2-32 rows (gemv<RT, T>)

// The one-row GEMV (decode of a single sequence): K is split over the T threads of one work-group
// per 8-column group, the partials merged in shared local memory, so no partial buffer and no
// second kernel. dpas with repeat count 1 reads only row 0's 32 activation bytes per block instead
// of a whole 8-row tile. For T = 1 the thread owns all of K and stores straight to out.
template <int T>
void gemv1(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  const int64_t kb = a.k / 32, ng = w.n / 8, n = w.n;
  const int64_t per = (kb + T - 1) / T;
  const int8_t* aq = a.q;
  const float* as = a.s;
  const uint8_t* bt = w.tiles;
  const sycl::half* bs = w.scales;
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(ng), size_t(T)), sycl::range<2>(1, size_t(T))),
                 [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                   if constexpr (T > 1) esimd::slm_init<T * 32>();
                   const int64_t g = int64_t(it.get_global_id(0));
                   const int t = int(it.get_local_id(1));
                   esimd::simd<float, 8> acc = 0;
                   const int64_t b0 = int64_t(t) * per, b1 = std::min(kb, b0 + per);
                   int64_t b = b0;
                   // Four blocks per step: the group's tiles run contiguously along K, so the four
                   // come from one 512-byte load and their scales from one 64-byte load, and every
                   // load is issued before the first dpas, so each thread keeps the reads in flight.
                   for (; b + 4 <= b1; b += 4) {
                     esimd::simd<uint8_t, 4 * kTileBytes> B4 =
                         esimd::block_load<uint8_t, 4 * kTileBytes>(bt + tile_at(b, g, kb) * kTileBytes);
                     esimd::simd<sycl::half, 32> h4 = esimd::block_load<sycl::half, 32>(bs + tile_at(b, g, kb) * 8);
                     esimd::simd<uint8_t, 32> A[4];
                     float sa[4];
#pragma unroll
                     for (int u = 0; u < 4; ++u) {
                       A[u] = esimd::block_load<uint8_t, 32>(reinterpret_cast<const uint8_t*>(aq) + (b + u) * kATileBytes);
                       sa[u] = as[(b + u) * 8];
                     }
#pragma unroll
                     for (int u = 0; u < 4; ++u) {
                       const esimd::simd<uint8_t, kTileBytes> B = B4.template select<kTileBytes, 1>(u * kTileBytes);
                       const esimd::simd<float, 8> sb = h4.template select<8, 1>(u * 8);
                       esimd::simd<int, 8> ci =
                           xmxd::dpas<8, 1, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B, A[u]);
                       esimd::simd<float, 8> cf = ci;
                       acc += cf * (sb * sa[u]);
                     }
                   }
                   for (; b < b1; ++b) {
                     const int64_t wt = tile_at(b, g, kb);
                     esimd::simd<uint8_t, kTileBytes> B = esimd::block_load<uint8_t, kTileBytes>(bt + wt * kTileBytes);
                     esimd::simd<sycl::half, 8> h = esimd::block_load<sycl::half, 8>(bs + wt * 8);
                     esimd::simd<float, 8> sb = h;
                     esimd::simd<uint8_t, 32> A = esimd::block_load<uint8_t, 32>(reinterpret_cast<const uint8_t*>(aq) + b * kATileBytes);
                     const float sa = as[b * 8];
                     esimd::simd<int, 8> ci =
                         xmxd::dpas<8, 1, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B, A);
                     esimd::simd<float, 8> cf = ci;
                     acc += cf * (sb * sa);
                   }
                   if constexpr (T > 1) {
                     esimd::slm_block_store<float, 8>(t * 32, acc);
                     esimd::barrier();
                     if (t == 0) {
                       esimd::simd<float, 8> sum = 0;
                       for (int i = 0; i < T; ++i) sum += esimd::slm_block_load<float, 8>(i * 32);
                       esimd::block_store<float, 8>(out + g * 8, sum);
                     }
                   } else {
                     esimd::block_store<float, 8>(out + g * 8, acc);
                   }
                 });
}

// The dispatcher's switch over T, one template instantiation per T per row count.
void gemv1_d(int T, sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  switch (T) {
    case 1: gemv1<1>(q, a, w, out); break;
    case 2: gemv1<2>(q, a, w, out); break;
    case 4: gemv1<4>(q, a, w, out); break;
    case 8: gemv1<8>(q, a, w, out); break;
    case 16: gemv1<16>(q, a, w, out); break;
    default: gemv1<32>(q, a, w, out); break;
  }
}

template <int RT>
void gemv_d(int T, sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  switch (T) {
    case 1: gemv<RT, 1>(q, a, w, out); break;
    case 2: gemv<RT, 2>(q, a, w, out); break;
    case 4: gemv<RT, 4>(q, a, w, out); break;
    case 8: gemv<RT, 8>(q, a, w, out); break;
    case 16: gemv<RT, 16>(q, a, w, out); break;
    default: gemv<RT, 32>(q, a, w, out); break;
  }
}

constexpr int64_t kThreadsGrouped = 8192;  // work-items of a grouped GEMV; 4096-8192 measured best in the engine (2048-32768 tried)

void geglu_quant(sycl::queue& q, const float* scratch, int64_t m, int64_t n, Activations& out);

// The grouped one-row GEMV (decode MoE): gemv1 with the weight picked per pair from the expert
// tables and the activation row per pair from p / row_div. One work-group of T threads per
// (8-column group, pair), K split over the threads and merged in shared local memory.
template <int T>
void gemv1_grouped(sycl::queue& q, const Activations& a, const Activations* ad, const ExpertTable& w, const int32_t* ids, int64_t P,
                   int64_t pe, int row_div, float* out) {
  const int64_t kb = a.k / 32, ng = w.n / 8, n = w.n;
  const int64_t per = (kb + T - 1) / T;
  const int8_t* aq = a.q;
  const float* as = a.s;
  // Pairs from pe on are the dense FFN's thirds: weights table[count - 3 + (p - pe) % 3], activation row (p - pe) / 3 of ad.
  const int8_t* aq2 = ad ? ad->q : aq;
  const float* as2 = ad ? ad->s : as;
  const int e_dense = int(w.count) - 3;
  const uint8_t* const* tt = w.tiles;
  const sycl::half* const* ts = w.scales;
  q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(ng), size_t(P), size_t(T)), sycl::range<3>(1, 1, size_t(T))),
                 [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                   if constexpr (T > 1) esimd::slm_init<T * 32>();
                   const int64_t g = int64_t(it.get_global_id(0)), p = int64_t(it.get_global_id(1));
                   const int t = int(it.get_local_id(2));
                   const bool dn = p >= pe;
                   const int e = dn ? e_dense + int((p - pe) % 3) : ids[p];
                   const uint8_t* bt = tt[e];
                   const sycl::half* bs = ts[e];
                   const int64_t row = dn ? (p - pe) / 3 : p / row_div;
                   // Row `row`'s tile b starts at ((row / 8) * kb + b) * 256 + (row % 8) * 32.
                   const uint8_t* ab = reinterpret_cast<const uint8_t*>(dn ? aq2 : aq) + (row / 8) * kb * kATileBytes + (row % 8) * 32;
                   const float* sb_ = (dn ? as2 : as) + (row / 8) * kb * 8 + row % 8;
                   esimd::simd<float, 8> acc = 0;
                   const int64_t b0 = int64_t(t) * per, b1 = std::min(kb, b0 + per);
                   int64_t b = b0;
                   for (; b + 4 <= b1; b += 4) {
                     esimd::simd<uint8_t, 4 * kTileBytes> B4 =
                         esimd::block_load<uint8_t, 4 * kTileBytes>(bt + tile_at(b, g, kb) * kTileBytes);
                     esimd::simd<sycl::half, 32> h4 = esimd::block_load<sycl::half, 32>(bs + tile_at(b, g, kb) * 8);
                     esimd::simd<uint8_t, 32> A[4];
                     float sa[4];
#pragma unroll
                     for (int u = 0; u < 4; ++u) {
                       A[u] = esimd::block_load<uint8_t, 32>(ab + (b + u) * kATileBytes);
                       sa[u] = sb_[(b + u) * 8];
                     }
#pragma unroll
                     for (int u = 0; u < 4; ++u) {
                       const esimd::simd<uint8_t, kTileBytes> B = B4.template select<kTileBytes, 1>(u * kTileBytes);
                       const esimd::simd<float, 8> sb = h4.template select<8, 1>(u * 8);
                       esimd::simd<int, 8> ci =
                           xmxd::dpas<8, 1, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B, A[u]);
                       esimd::simd<float, 8> cf = ci;
                       acc += cf * (sb * sa[u]);
                     }
                   }
                   // K = 704 is 22 blocks: the tail takes what the four-block steps leave.
                   for (; b < b1; ++b) {
                     const int64_t wt = tile_at(b, g, kb);
                     esimd::simd<uint8_t, kTileBytes> B = esimd::block_load<uint8_t, kTileBytes>(bt + wt * kTileBytes);
                     esimd::simd<sycl::half, 8> h = esimd::block_load<sycl::half, 8>(bs + wt * 8);
                     esimd::simd<float, 8> sb = h;
                     esimd::simd<uint8_t, 32> A = esimd::block_load<uint8_t, 32>(ab + b * kATileBytes);
                     const float sa = sb_[b * 8];
                     esimd::simd<int, 8> ci =
                         xmxd::dpas<8, 1, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B, A);
                     esimd::simd<float, 8> cf = ci;
                     acc += cf * (sb * sa);
                   }
                   if constexpr (T > 1) {
                     esimd::slm_block_store<float, 8>(t * 32, acc);
                     esimd::barrier();
                     if (t == 0) {
                       esimd::simd<float, 8> sum = 0;
                       for (int i = 0; i < T; ++i) sum += esimd::slm_block_load<float, 8>(i * 32);
                       esimd::block_store<float, 8>(out + p * n + g * 8, sum);
                     }
                   } else {
                     esimd::block_store<float, 8>(out + p * n + g * 8, acc);
                   }
                 });
}

}  // namespace

void matmul(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  if (a.k != w.k) throw std::runtime_error("xmx: activation and weight widths differ");
  if (w.n % 32) throw std::runtime_error("xmx: output width not a multiple of 32");
  const int64_t row_tiles = (a.m + 7) / 8;
  if (a.fp16) {
    gemm16(q, a, w, out);
    return;
  }
  if (row_tiles > 4) throw std::runtime_error("xmx: int8 activations of more than 32 rows");
  // T = the largest power of two <= the row-count's thread target / ng, clamped to [1, 32].
  const int64_t ng = w.n / 8;
  const int64_t threads = a.m == 1 ? kThreads1 : kThreadsN;
  int T = 1;
  while (T < 32 && 2 * T <= threads / ng) T *= 2;
  if (a.m == 1) gemv1_d(T, q, a, w, out);
  else if (row_tiles <= 1) gemv_d<1>(T, q, a, w, out);
  else if (row_tiles <= 2) gemv_d<2>(T, q, a, w, out);
  else gemv_d<4>(T, q, a, w, out);
}

void matmul_geglu(sycl::queue& q, const Activations& a, const Weight& w, Activations& out, float* scratch) {
  const int64_t m = a.m, n = w.n / 2;
  // A work-group of 8 work-items covers 64 output columns; the shuffles below only combine the 4
  // of a 32-block, so any width a multiple of 64 works (the 26B's experts are 704 wide).
  if (n % 64) throw std::runtime_error("xmx: GeGLU width not a multiple of 64");
  set_rows(out, m, n);
  if (a.fp16) {
    if (!out.fp16) throw std::runtime_error("xmx: GeGLU input and output forms differ");
    return gemm16(q, a, w, nullptr, out.h);
  }
  matmul(q, a, w, scratch);
  geglu_quant(q, scratch, m, n, out);
}

namespace {

// GeGLU over m rows of interleaved gate|up (fp32, [m][2n]) into int8 `out` (n wide).
void geglu_quant(sycl::queue& q, const float* scratch, int64_t m, int64_t n, Activations& out) {
  // Eight columns per work-item, gate and up from the interleaved tiles; the four work-items of a
  // 32-block share its max through sub-group shuffles.
  int8_t* aq = out.q;
  float* as = out.s;
  const int32_t N = int32_t(n), kb = N / 32;
  const float* gu = scratch;
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(m), size_t(n / 8)), sycl::range<2>(1, 8)),
                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                   const int32_t r = int32_t(it.get_global_id(0)), c0 = int32_t(it.get_global_id(1)) * 8;
                   const float* src = gu + r * 2 * N + c0 * 2;
                   float v[8], amax = 0;
                   for (int i = 0; i < 8; ++i) {
                     const float g = src[i], u = src[8 + i];
                     v[i] = 0.5f * g * (1.0f + sycl::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
                     amax = sycl::fmax(amax, sycl::fabs(v[i]));
                   }
                   auto sg = it.get_sub_group();
                   amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
                   amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
                   const float d = quantize_run(v, amax, aq + act_index(r, c0, kb));
                   if (c0 % 32 == 0) as[act_scale_index(r, c0, kb)] = d;
                 });
}

}  // namespace

ExpertTable make_expert_table(sycl::queue& q, const std::vector<Weight>& experts) {
  if (experts.empty()) throw std::runtime_error("xmx: empty expert table");
  ExpertTable t;
  t.k = experts[0].k;
  t.n = experts[0].n;
  t.count = int64_t(experts.size());
  std::vector<const uint8_t*> tiles;
  std::vector<const sycl::half*> scales;
  for (const Weight& w : experts) {
    if (w.k != t.k || w.n != t.n) throw std::runtime_error("xmx: experts of one table differ in shape");
    tiles.push_back(w.tiles);
    scales.push_back(w.scales);
  }
  t.tiles = sycl::malloc_device<const uint8_t*>(experts.size(), q);
  t.scales = sycl::malloc_device<const sycl::half*>(experts.size(), q);
  if (!t.tiles || !t.scales) throw std::runtime_error("xmx: out of device memory for the expert table");
  q.memcpy(t.tiles, tiles.data(), tiles.size() * sizeof(void*));
  q.memcpy(t.scales, scales.data(), scales.size() * sizeof(void*)).wait();
  return t;
}

void free_expert_table(sycl::queue& q, ExpertTable& t) {
  if (t.tiles) sycl::free(t.tiles, q);
  if (t.scales) sycl::free(t.scales, q);
  t = ExpertTable{};
}

namespace {

void grouped(sycl::queue& q, const Activations& a, const Activations* ad, const ExpertTable& w, const int32_t* ids, int64_t P,
             int64_t pe, int64_t row_div, float* out) {
  if (a.k != w.k || (ad && ad->k != w.k)) throw std::runtime_error("xmx: activation and weight widths differ");
  if (a.fp16 || (ad && ad->fp16)) throw std::runtime_error("xmx: grouped GEMV needs int8 activations");
  if (w.n % 32) throw std::runtime_error("xmx: output width not a multiple of 32");
  // Same rule as matmul's one-row case, with the group count including the pairs.
  const int64_t groups = w.n / 8 * P;
  int T = 1;
  while (T < 32 && 2 * T <= kThreadsGrouped / groups) T *= 2;
  switch (T) {
    case 1: gemv1_grouped<1>(q, a, ad, w, ids, P, pe, int(row_div), out); break;
    case 2: gemv1_grouped<2>(q, a, ad, w, ids, P, pe, int(row_div), out); break;
    case 4: gemv1_grouped<4>(q, a, ad, w, ids, P, pe, int(row_div), out); break;
    case 8: gemv1_grouped<8>(q, a, ad, w, ids, P, pe, int(row_div), out); break;
    case 16: gemv1_grouped<16>(q, a, ad, w, ids, P, pe, int(row_div), out); break;
    default: gemv1_grouped<32>(q, a, ad, w, ids, P, pe, int(row_div), out); break;
  }
}

}  // namespace

void matmul_grouped(sycl::queue& q, const Activations& a, const ExpertTable& w, const int32_t* ids, int64_t P,
                    int64_t row_div, float* out) {
  grouped(q, a, nullptr, w, ids, P, P, row_div, out);
}

void matmul_geglu_grouped(sycl::queue& q, const Activations& a, const ExpertTable& w, const int32_t* ids, int64_t P,
                          int64_t row_div, Activations& out, float* scratch) {
  const int64_t n = w.n / 2;
  if (n % 64) throw std::runtime_error("xmx: GeGLU width not a multiple of 64");
  if (P > kFp16Rows) throw std::runtime_error("xmx: grouped GeGLU of more than 32 pairs");
  set_rows(out, P, n);
  matmul_grouped(q, a, w, ids, P, row_div, scratch);
  geglu_quant(q, scratch, P, n, out);
}

void matmul_geglu_grouped_dense(sycl::queue& q, const Activations& a, const Activations& ad, const ExpertTable& w, const int32_t* ids,
                                int64_t P, int64_t row_div, int64_t rows_d, Activations& out, Activations& out_d, float* scratch) {
  const int64_t n = w.n / 2, nd = 3 * n;
  if (n % 64) throw std::runtime_error("xmx: GeGLU width not a multiple of 64");
  if (P > kFp16Rows || rows_d > kFp16Rows) throw std::runtime_error("xmx: grouped GeGLU of more than 32 pairs");
  if (w.count < 3) throw std::runtime_error("xmx: expert table without the dense thirds");
  set_rows(out, P, n);
  set_rows(out_d, rows_d, nd);
  grouped(q, a, &ad, w, ids, P + 3 * rows_d, P, row_div, scratch);
  // One GeGLU launch for both: rows r < P are the experts' (n wide); the rest the dense FFN's (3 n wide,
  // its three thirds side by side in scratch at pair P + 3 t).
  int8_t *aq = out.q, *aqd = out_d.q;
  float *as = out.s, *asd = out_d.s;
  const int32_t N = int32_t(n), ND = int32_t(nd), Pe = int32_t(P);
  const float* gu = scratch;
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(P + rows_d), size_t(nd / 8)), sycl::range<2>(1, 8)),
                 [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(8)]] {
                   const int32_t r = int32_t(it.get_global_id(0)), c0 = int32_t(it.get_global_id(1)) * 8;
                   const bool dn = r >= Pe;
                   if (c0 >= (dn ? ND : N)) return;  // whole sub-groups of 8 items leave together
                   const int32_t Nr = dn ? ND : N;
                   const float* src = gu + (dn ? int64_t(Pe + 3 * (r - Pe)) * (2 * N) : int64_t(r) * 2 * N) + c0 * 2;
                   float v[8], amax = 0;
                   for (int i = 0; i < 8; ++i) {
                     const float g = src[i], u = src[8 + i];
                     v[i] = 0.5f * g * (1.0f + sycl::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u;
                     amax = sycl::fmax(amax, sycl::fabs(v[i]));
                   }
                   auto sg = it.get_sub_group();
                   amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
                   amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));
                   const int32_t row = dn ? r - Pe : r, kbr = Nr / 32;
                   const float d = quantize_run(v, amax, (dn ? aqd : aq) + act_index(row, c0, kbr));
                   if (c0 % 32 == 0) (dn ? asd : as)[act_scale_index(row, c0, kbr)] = d;
                 });
}

}  // namespace eklow::xmx
