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

// Partial sums of the K-split GEMV, reused across calls.
struct Scratch {
  sycl::queue* q = nullptr;
  float* p = nullptr;
  size_t floats = 0;
  float* get(sycl::queue& queue, size_t need) {
    if (need > floats || q != &queue) {
      if (p) {
        q->wait();  // kernels still in flight may read the old buffer
        sycl::free(p, *q);
      }
      q = &queue;
      floats = std::max(need, floats * 2);
      p = sycl::malloc_device<float>(floats, queue);
      if (!p) throw std::runtime_error("xmx: out of device memory for GEMV partials");
    }
    return p;
  }
} scratch;

int pick_local(int64_t n) {
  for (int l : {8, 4, 2})
    if (n % l == 0) return l;
  return 1;
}

}  // namespace

Weight repack_q4_0(sycl::queue& q, const ref::Weight& raw) {
  if (raw.type != gguf::Type::Q4_0) throw std::runtime_error("xmx: repack_q4_0 needs a Q4_0 matrix");
  if (raw.k % 32 || raw.n % 8) throw std::runtime_error("xmx: Q4_0 matrix shape not a multiple of 32 x 8");
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
       const uint8_t* blk = src + uint64_t(g * 8 + n) * row_bytes + uint64_t(b) * 18;
       scales[(b * ng + g) * 8 + n] = sycl::bit_cast<sycl::half>(uint16_t(blk[0] | blk[1] << 8));
       for (int k = 0; k < 32; ++k) {
         const uint32_t q4 = (k < 16 ? blk[2 + k] & 15 : blk[2 + k - 16] >> 4) ^ 8;
         const int idx = ((k / 8) * 8 + n) * 8 + k % 8;  // nibble index in the tile
         words[idx / 8] |= q4 << (4 * (idx % 8));
       }
     }
     uint32_t* dst = reinterpret_cast<uint32_t*>(tiles + (b * ng + g) * kTileBytes);
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
  a.capacity = (max_m + 31) / 32 * 32 * k;
  a.q = sycl::malloc_device<int8_t>(size_t(a.capacity), q);
  a.s = sycl::malloc_device<float>(size_t(a.capacity / 32), q);
  if (!a.q || !a.s) throw std::runtime_error("xmx: out of device memory for activations");
  return a;
}

void free_activations(sycl::queue& q, Activations& a) {
  if (a.q) sycl::free(a.q, q);
  if (a.s) sycl::free(a.s, q);
  a.q = nullptr;
  a.s = nullptr;
}

void quantize(sycl::queue& q, const float* x, int64_t m, Activations& a) {
  const int64_t m_pad = (m + 31) / 32 * 32, kb = a.k / 32, k = a.k;
  if (m_pad * k > a.capacity) throw std::runtime_error("xmx: activation buffer too small");
  a.m = m;
  a.m_pad = m_pad;
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
  const int64_t m_pad = (m + 31) / 32 * 32;
  if (m_pad * k > a.capacity) throw std::runtime_error("xmx: activation buffer too small");
  a.m = m;
  a.m_pad = m_pad;
  a.k = k;
}

namespace {

// Tiled GEMM: one thread per (8*RM) x (8*RN) block of the output, RM*RN dpas per Q4_0 block.
// 16 x 32 is the largest tile that fits DG2's 128 registers without spilling (IGC dump,
// 2026-10-24; the 256-register mode is not available on DG2).
template <int RM, int RN>
struct GemmKernel {
  const int8_t* aq;
  const float* as;
  const uint8_t* bt;
  const sycl::half* bs;
  float* out;  // the output, or with K split, partial sums [split][m_pad][n]
  int64_t kb, ng, m, m_pad, n, per;  // per: K blocks per split

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    const int64_t sp = int64_t(it.get_global_id(0)), ti = int64_t(it.get_global_id(1)), tj = int64_t(it.get_global_id(2));
    esimd::simd<float, 64> acc[RM][RN];
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
      for (int c = 0; c < RN; ++c) acc[r][c] = 0;
    const int64_t b0 = sp * per, b1 = std::min(kb, b0 + per);
    for (int64_t b = b0; b < b1; ++b) {
      esimd::simd<uint8_t, kATileBytes> A[RM];
      esimd::simd<float, 8> sa[RM];
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        const int64_t tile = (ti * RM + r) * kb + b;
        A[r] = esimd::block_load<uint8_t, kATileBytes>(reinterpret_cast<const uint8_t*>(aq) + tile * kATileBytes);
        sa[r] = esimd::block_load<float, 8>(as + tile * 8);
      }
      esimd::simd<uint8_t, kTileBytes> B[RN];
      esimd::simd<float, 8> sb[RN];
#pragma unroll
      for (int c = 0; c < RN; ++c) {
        const int64_t tile = b * ng + tj * RN + c;
        B[c] = esimd::block_load<uint8_t, kTileBytes>(bt + tile * kTileBytes);
        esimd::simd<sycl::half, 8> h = esimd::block_load<sycl::half, 8>(bs + tile * 8);
        sb[c] = h;
      }
#pragma unroll
      for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          esimd::simd<int, 64> ci =
              xmxd::dpas<8, 8, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B[c], A[r]);
          esimd::simd<float, 64> cf = ci;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float s = sa[r][i];
            acc[r][c].template select<8, 1>(i * 8) += cf.template select<8, 1>(i * 8) * (sb[c] * s);
          }
        }
    }
    float* dst = out + sp * m_pad * n;
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int64_t row = ti * RM * 8 + r * 8 + i;
        if (row < m)
#pragma unroll
          for (int c = 0; c < RN; ++c)
            esimd::block_store<float, 8>(dst + row * n + tj * RN * 8 + c * 8, acc[r][c].template select<8, 1>(i * 8));
      }
  }
};

template <int RM, int RN>
void gemm(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  const int64_t mt = a.m_pad / (8 * RM), nt = w.n / (8 * RN), kb = a.k / 32, m = a.m, n = w.n, m_pad = a.m_pad;
  if (a.m_pad % (8 * RM) || w.n % (8 * RN)) throw std::runtime_error("xmx: GEMM tile does not divide the problem");
  // Too few output tiles to fill the card (prefill of a short chunk): split K and add the partial
  // sums afterwards.
  const int64_t splits = std::clamp<int64_t>(8192 / (mt * nt), 1, kb / 8 > 0 ? kb / 8 : 1);
  const int64_t per = (kb + splits - 1) / splits;
  float* dst = splits == 1 ? out : scratch.get(q, size_t(splits * m_pad * n));
  GemmKernel<RM, RN> k{a.q, a.s, w.tiles, w.scales, dst, kb, w.n / 8, m, m_pad, n, per};
  q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(splits), size_t(mt), size_t(nt)), sycl::range<3>(1, 1, size_t(pick_local(nt)))), k);
  if (splits > 1)
    q.parallel_for(sycl::range<2>(size_t(m), size_t(n)), [=](sycl::id<2> id) {
       const int64_t row = int64_t(id[0]), col = int64_t(id[1]);
       float sum = 0;
       for (int64_t sp = 0; sp < splits; ++sp) sum += dst[(sp * m_pad + row) * n + col];
       out[row * n + col] = sum;
     });
}

// Column-group schedule for up to 256 rows (decode batches, short prefill chunks): one thread per
// (K slice, group of up to 64 rows, 8 columns). Each weight tile is read once per row group and
// applied to all its rows; partial sums over K slices are added afterwards.
template <int RT>
void gemv(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  const int64_t kb = a.k / 32, ng = w.n / 8, m = a.m, n = w.n, m_pad = a.m_pad;
  const int64_t row_groups = (a.m + RT * 8 - 1) / (RT * 8);
  const int64_t splits = std::clamp<int64_t>(8192 / (ng * row_groups), 1, kb);
  const int64_t per = (kb + splits - 1) / splits;
  float* part = scratch.get(q, size_t(splits * m_pad * n));
  const int8_t* aq = a.q;
  const float* as = a.s;
  const uint8_t* bt = w.tiles;
  const sycl::half* bs = w.scales;
  q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(splits), size_t(row_groups), size_t(ng)), sycl::range<3>(1, 1, size_t(pick_local(ng)))),
                 [=](sycl::nd_item<3> it) SYCL_ESIMD_KERNEL {
                   const int64_t s = int64_t(it.get_global_id(0)), rg = int64_t(it.get_global_id(1)), g = int64_t(it.get_global_id(2));
                   esimd::simd<float, 64> acc[RT];
#pragma unroll
                   for (int r = 0; r < RT; ++r) acc[r] = 0;
                   const int64_t b0 = s * per, b1 = std::min(kb, b0 + per);
                   for (int64_t b = b0; b < b1; ++b) {
                     const int64_t wt = b * ng + g;
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
#pragma unroll
                   for (int r = 0; r < RT; ++r)
#pragma unroll
                     for (int i = 0; i < 8; ++i) {
                       const int64_t row = (rg * RT + r) * 8 + i;
                       if (row < m)
                         esimd::block_store<float, 8>(part + (s * m_pad + row) * n + g * 8, acc[r].template select<8, 1>(i * 8));
                     }
                 });
  q.parallel_for(sycl::range<2>(size_t(m), size_t(n)), [=](sycl::id<2> id) {
     const int64_t row = int64_t(id[0]), col = int64_t(id[1]);
     float sum = 0;
     for (int64_t s = 0; s < splits; ++s) sum += part[(s * m_pad + row) * n + col];
     out[row * n + col] = sum;
   });
}

// The one-row GEMV (decode of a single sequence): dpas with repeat count 1 reads only row 0's 32
// activation bytes per block instead of a whole 8-row tile.
void gemv1(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  const int64_t kb = a.k / 32, ng = w.n / 8, n = w.n;
  const int64_t splits = std::clamp<int64_t>(8192 / ng, 1, kb);
  const int64_t per = (kb + splits - 1) / splits;
  float* part = scratch.get(q, size_t(splits * n));
  const int8_t* aq = a.q;
  const float* as = a.s;
  const uint8_t* bt = w.tiles;
  const sycl::half* bs = w.scales;
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(splits), size_t(ng)), sycl::range<2>(1, size_t(pick_local(ng)))),
                 [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
                   const int64_t s = int64_t(it.get_global_id(0)), g = int64_t(it.get_global_id(1));
                   esimd::simd<float, 8> acc = 0;
                   const int64_t b0 = s * per, b1 = std::min(kb, b0 + per);
                   int64_t b = b0;
                   // Four blocks per step, every load issued before the first dpas, so each
                   // thread keeps several weight reads in flight.
                   for (; b + 4 <= b1; b += 4) {
                     esimd::simd<uint8_t, kTileBytes> B[4];
                     esimd::simd<float, 8> sb[4];
                     esimd::simd<uint8_t, 32> A[4];
                     float sa[4];
#pragma unroll
                     for (int u = 0; u < 4; ++u) {
                       const int64_t wt = (b + u) * ng + g;
                       B[u] = esimd::block_load<uint8_t, kTileBytes>(bt + wt * kTileBytes);
                       esimd::simd<sycl::half, 8> h = esimd::block_load<sycl::half, 8>(bs + wt * 8);
                       sb[u] = h;
                       A[u] = esimd::block_load<uint8_t, 32>(reinterpret_cast<const uint8_t*>(aq) + (b + u) * kATileBytes);
                       sa[u] = as[(b + u) * 8];
                     }
#pragma unroll
                     for (int u = 0; u < 4; ++u) {
                       esimd::simd<int, 8> ci =
                           xmxd::dpas<8, 1, int, uint8_t, uint8_t, dpas_argument_type::s4, dpas_argument_type::s8>(B[u], A[u]);
                       esimd::simd<float, 8> cf = ci;
                       acc += cf * (sb[u] * sa[u]);
                     }
                   }
                   for (; b < b1; ++b) {
                     const int64_t wt = b * ng + g;
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
                   esimd::block_store<float, 8>(part + s * n + g * 8, acc);
                 });
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
     float sum = 0;
     for (int64_t s = 0; s < splits; ++s) sum += part[s * n + int64_t(id[0])];
     out[id[0]] = sum;
   });
}

}  // namespace

void matmul(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  if (a.k != w.k) throw std::runtime_error("xmx: activation and weight widths differ");
  if (w.n % 32) throw std::runtime_error("xmx: output width not a multiple of 32");
  const int64_t row_tiles = (a.m + 7) / 8;
  if (a.m == 1) gemv1(q, a, w, out);
  else if (row_tiles <= 1) gemv<1>(q, a, w, out);
  else if (row_tiles <= 2) gemv<2>(q, a, w, out);
  else if (row_tiles <= 4) gemv<4>(q, a, w, out);
  else if (row_tiles <= 32) gemv<8>(q, a, w, out);
  else gemm<2, 4>(q, a, w, out);
}

}  // namespace eklow::xmx
