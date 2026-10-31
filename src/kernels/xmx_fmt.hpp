// The Qwen weight formats on the matrix units (DECISIONS 2026-10-31): layouts, and the device-side tile decoders the
// decode GEMVs (xmx_fmt.cpp) and the fp16 GEMM (xmx.cpp) share.
//
// Every format keeps ggml's bytes per weight and its values; only the order changes, so that each thread's loads are
// contiguous. A matrix is tiles of 8 output columns x 32 K, [N/8][K/32] (column-group major, as Q4_0's), plus
// records in `Weight::scales`, [N/8][K/rec_tiles]: for Q4_0 and Q8_0I one record a tile (the fp16 scale of each
// column), for the others one per superblock (8 tiles = 256 K). The dpas B operands, by element type:
//   s8   256 bytes, dword (k/4)*8 + n holds k..k+3 of column n;
//   u4   128 bytes, dword (k/8)*8 + n holds k..k+7 (low nibble first);
//   u2    64 bytes, dword (k/16)*8 + n holds k..k+15 (2 bits each).
// (checked on the card by eklow-xmx-bench; all of s8 x {s8, u8, s4, u4, s2, u2} run at the s8 rate, 197.6 TOPS.)
//
//   Q8_0I   tile: s8 operand, the weights. record: fp16 d[8].                                       8.5 bits
//   Q4_K    tile: u4 operand of the 4-bit values. record (128 B): fp16 d[8], fp16 dmin[8], ggml's 12 scale/min bytes
//           of each column as [12][8].  w = d sc q - dmin m, q 0..15.                                4.5 bits
//   Q5_K    as Q4_K for the low 4 bits; the record (384 B) adds, per tile, 8 dwords (one a column) of high bits laid
//           out as the u2 operand's two dwords interleaved: bit 2 (k%16) + k/16 is bit 4 of the value of k. Decode
//           runs a u4 and a u2 dpas (the sum is q = low + 16 high), no unpack.                         5.5 bits
//   Q6_K    tile: u4 operand of the low 4 bits. record (656 B): int8 sc[16][8], per tile the s2 operand of the high 2
//           bits (xor 2: q - 32 = low + 16 s2), fp16 d[8]. 16-weight sub-blocks run as dpas on a half each (the B
//           operand's other half zero): 4 dpas a tile.                                               6.5625 bits
//   IQ4_XS  tile: byte w = (k4*8 + n)*2 + h holds the 4-bit codes of k = 4 k4 + 2h and the next; one lookup in a
//           256-entry table of int8 pairs (in shared local memory) gives the two weights as the s8 operand's word.
//           record (64 B): fp16 d[8], scales_h u16[8], scales_l [4][8].                                4.25 bits
#pragma once

#include <sycl/ext/intel/esimd.hpp>

#include "kernels/xmx.hpp"

namespace eklow::xmx::fmt {

namespace esimd = sycl::ext::intel::esimd;
namespace xmxd = sycl::ext::intel::esimd::xmx;
using xmxd::dpas_argument_type;

struct Geo {
  int tile_bytes;  // bytes of one tile in `tiles`
  int rec_tiles;   // tiles a record covers
  int rec_bytes;   // bytes of one record in `scales`
  int block_bytes; // ggml's bytes per superblock (Q8_0I: per 32-block)
};
constexpr Geo geo(WeightFormat f) {
  switch (f) {
    case WeightFormat::Q4_0: return {128, 1, 16, 18};
    case WeightFormat::Q8_0I: return {256, 1, 16, 34};
    case WeightFormat::Q4_K: return {128, 8, 128, 144};
    case WeightFormat::Q5_K: return {128, 8, 384, 176};
    case WeightFormat::Q6_K: return {128, 8, 656, 210};
    case WeightFormat::IQ4_XS: return {128, 8, 64, 136};
    default: return {0, 0, 0, 0};
  }
}
// Offsets inside a K-quant / IQ4_XS record.
constexpr int kQ4KDmin = 16, kQ4KScales = 32, kQ5KHigh = 128;
constexpr int kQ6KSc = 0, kQ6KHigh = 128, kQ6KD = 640;
constexpr int kIqScalesH = 16, kIqScalesL = 32;

// ggml's kvalues_iq4nl.
constexpr int8_t kIq4Values[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ------------------------------------------------------------------------------------------------------------
// Device-side decoders for the decode GEMVs. A chunk is CH tiles of one column group, loaded together (the loads
// of all of them go out before the first dpas); `tile<U, R>` multiplies tile U of the chunk by R activation rows
// (1: dpas with repeat count 1, the single-row GEMV; 8: an 8-row tile) and adds to the R x 8 accumulator.

template <int R>
using Acc = esimd::simd<float, 8 * R>;

template <dpas_argument_type BP, int R, int BN>
SYCL_ESIMD_FUNCTION inline esimd::simd<int, 8 * R> dp(const esimd::simd<uint8_t, BN>& B, const esimd::simd<uint8_t, 32 * R>& A) {
  return xmxd::dpas<8, R, int, uint8_t, uint8_t, BP, dpas_argument_type::s8>(B, A);
}

// acc row i += cf row i * (sv * sa[i]) for rows i < R, the columns' scale vector sv.
template <int R>
SYCL_ESIMD_FUNCTION inline void axpy(Acc<R>& acc, esimd::simd<float, 8 * R> cf, esimd::simd<float, 8> sv,
                                     esimd::simd<float, R> sa) {
#pragma unroll
  for (int i = 0; i < R; ++i) {
    const float s_i = sa[i];
    acc.template select<8, 1>(i * 8) += cf.template select<8, 1>(i * 8) * (sv * s_i);
  }
}

template <WeightFormat F>
struct Dec;

template <>
struct Dec<WeightFormat::Q8_0I> {
  static constexpr int CH = 2, TB = 256, REC_T = 1, REC_B = 16;
  struct Chunk {
    esimd::simd<uint8_t, CH * TB> t;
    esimd::simd<float, 8 * CH> s;
  };
  template <int H>
  SYCL_ESIMD_FUNCTION static void load(Chunk& c, const uint8_t* tp, const uint8_t* rp) {
    c.t = esimd::block_load<uint8_t, CH * TB>(tp);
    c.s = esimd::block_load<sycl::half, 8 * CH>(reinterpret_cast<const sycl::half*>(rp));
  }
  template <int U, int R>
  SYCL_ESIMD_FUNCTION static void tile(Acc<R>& acc, Chunk& c, const esimd::simd<uint8_t, 32 * R>& A, const esimd::simd<float, R>& sa) {
    esimd::simd<uint8_t, TB> B = c.t.template select<TB, 1>(U * TB);
    esimd::simd<float, 8 * R> cf = dp<dpas_argument_type::s8, R, TB>(B, A);
    axpy<R>(acc, cf, c.s.template select<8, 1>(U * 8), sa);
  }
};

// The min formats: acc row += sa * (ds * dot(q, a) - dm * sum(a)); the sum of the activation block comes from a
// second dpas against a tile of ones (the matrix units have room, the vector units less so).
struct MinFormat {
  static constexpr int CH = 4, TB = 128, REC_T = 8;
  struct Chunk {
    esimd::simd<uint8_t, CH * TB> t;
    esimd::simd<float, 8 * CH> ds, dm;
    esimd::simd<uint8_t, 128> x;  // Q5_K: the high bits of the chunk's tiles (CH x 32 bytes)
  };
  // The scales d sc and the mins dmin m of the chunk's four sub-blocks, from ggml's 12 packed bytes of each
  // column (rows [12][8]) and the fp16 d and dmin.
  template <int H>
  SYCL_ESIMD_FUNCTION static void scales(Chunk& c, const uint8_t* rp) {
    esimd::simd<float, 8> d = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rp));
    esimd::simd<float, 8> dmin = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rp + kQ4KDmin));
    esimd::simd<uint8_t, 96> S = esimd::block_load<uint8_t, 96>(rp + kQ4KScales);
    esimd::simd<uint16_t, 32> sc, mn;
    if constexpr (H == 0) {
      sc = S.template select<32, 1>(0) & 63;
      mn = S.template select<32, 1>(32) & 63;
    } else {
      esimd::simd<uint16_t, 32> s8 = S.template select<32, 1>(64), s0 = S.template select<32, 1>(0), s4 = S.template select<32, 1>(32);
      sc = (s8 & 15) | ((s0 >> 6) << 4);
      mn = (s8 >> 4) | ((s4 >> 6) << 4);
    }
    esimd::simd<float, 32> scf = sc, mnf = mn;
    c.ds = scf * d.template replicate<CH>();
    c.dm = mnf * dmin.template replicate<CH>();
  }
};

template <>
struct Dec<WeightFormat::Q4_K> : MinFormat {
  template <int H>
  SYCL_ESIMD_FUNCTION static void load(Chunk& c, const uint8_t* tp, const uint8_t* rp) {
    c.t = esimd::block_load<uint8_t, CH * TB>(tp);
    scales<H>(c, rp);
  }
  template <int U, int R>
  SYCL_ESIMD_FUNCTION static void tile(Acc<R>& acc, Chunk& c, const esimd::simd<uint8_t, 32 * R>& A, const esimd::simd<float, R>& sa) {
    esimd::simd<uint8_t, TB> B = c.t.template select<TB, 1>(U * TB);
    esimd::simd<uint8_t, TB> ones = 0x11;
    esimd::simd<float, 8 * R> cf = dp<dpas_argument_type::u4, R, TB>(B, A);
    esimd::simd<float, 8 * R> cs = dp<dpas_argument_type::u4, R, TB>(ones, A);
#pragma unroll
    for (int i = 0; i < R; ++i)
      acc.template select<8, 1>(i * 8) += (cf.template select<8, 1>(i * 8) * c.ds.template select<8, 1>(U * 8) -
                                           cs.template select<8, 1>(i * 8) * c.dm.template select<8, 1>(U * 8)) * sa[i];
  }
};

template <>
struct Dec<WeightFormat::Q5_K> : MinFormat {
  template <int H>
  SYCL_ESIMD_FUNCTION static void load(Chunk& c, const uint8_t* tp, const uint8_t* rp) {
    c.t = esimd::block_load<uint8_t, CH * TB>(tp);
    c.x = esimd::block_load<uint8_t, 128>(rp + kQ5KHigh + H * 128);
    scales<H>(c, rp);
  }
  template <int U, int R>
  SYCL_ESIMD_FUNCTION static void tile(Acc<R>& acc, Chunk& c, const esimd::simd<uint8_t, 32 * R>& A, const esimd::simd<float, R>& sa) {
    esimd::simd<uint8_t, TB> B = c.t.template select<TB, 1>(U * TB);
    esimd::simd<uint8_t, TB> ones = 0x11;
    // The u2 operand's two dwords of a column from its one word of interleaved high bits.
    esimd::simd<uint32_t, 8> hw = c.x.template select<32, 1>(U * 32).template bit_cast_view<uint32_t>();
    esimd::simd<uint32_t, 16> h2;
    h2.template select<8, 1>(0) = hw & 0x55555555u;
    h2.template select<8, 1>(8) = (hw >> 1) & 0x55555555u;
    esimd::simd<uint8_t, 64> B2 = h2.template bit_cast_view<uint8_t>();
    esimd::simd<int, 8 * R> ci = dp<dpas_argument_type::u4, R, TB>(B, A);
    ci += dp<dpas_argument_type::u2, R, 64>(B2, A) << 4;
    esimd::simd<float, 8 * R> cf = ci;
    esimd::simd<float, 8 * R> cs = dp<dpas_argument_type::u4, R, TB>(ones, A);
#pragma unroll
    for (int i = 0; i < R; ++i)
      acc.template select<8, 1>(i * 8) += (cf.template select<8, 1>(i * 8) * c.ds.template select<8, 1>(U * 8) -
                                           cs.template select<8, 1>(i * 8) * c.dm.template select<8, 1>(U * 8)) * sa[i];
  }
};

template <>
struct Dec<WeightFormat::Q6_K> {
  static constexpr int CH = 4, TB = 128, REC_T = 8;
  struct Chunk {
    esimd::simd<uint8_t, CH * TB> t;
    esimd::simd<uint8_t, CH * 64> x;   // the s2 operands
    esimd::simd<float, 16 * CH> s;     // per tile: d sc of the first 16 K, d sc of the second (8 columns each)
  };
  template <int H>
  SYCL_ESIMD_FUNCTION static void load(Chunk& c, const uint8_t* tp, const uint8_t* rp) {
    c.t = esimd::block_load<uint8_t, CH * TB>(tp);
    c.x = esimd::block_load<uint8_t, CH * 64>(rp + kQ6KHigh + H * CH * 64);
    esimd::simd<float, 8> d = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rp + kQ6KD));
    esimd::simd<int8_t, 64> sc = esimd::block_load<int8_t, 64>(reinterpret_cast<const int8_t*>(rp + kQ6KSc) + H * 64);
    esimd::simd<float, 64> scf = sc;
    c.s = scf * d.template replicate<8>();
  }
  template <int U, int R>
  SYCL_ESIMD_FUNCTION static void tile(Acc<R>& acc, Chunk& c, const esimd::simd<uint8_t, 32 * R>& A, const esimd::simd<float, R>& sa) {
    esimd::simd<uint8_t, TB> B4 = c.t.template select<TB, 1>(U * TB);
    esimd::simd<uint8_t, TB> B4l = 0, B4h = 0;
    B4l.template select<64, 1>(0) = B4.template select<64, 1>(0);
    B4h.template select<64, 1>(64) = B4.template select<64, 1>(64);
    esimd::simd<uint8_t, 64> B2l = 0, B2h = 0;
    B2l.template select<32, 1>(0) = c.x.template select<32, 1>(U * 64);
    B2h.template select<32, 1>(32) = c.x.template select<32, 1>(U * 64 + 32);
    esimd::simd<int, 8 * R> lo = dp<dpas_argument_type::u4, R, TB>(B4l, A);
    lo += dp<dpas_argument_type::s2, R, 64>(B2l, A) << 4;
    esimd::simd<int, 8 * R> hi = dp<dpas_argument_type::u4, R, TB>(B4h, A);
    hi += dp<dpas_argument_type::s2, R, 64>(B2h, A) << 4;
    esimd::simd<float, 8 * R> cl = lo, ch = hi;
    esimd::simd<float, 8> s0 = c.s.template select<8, 1>(U * 16), s1 = c.s.template select<8, 1>(U * 16 + 8);
#pragma unroll
    for (int i = 0; i < R; ++i)
      acc.template select<8, 1>(i * 8) += (cl.template select<8, 1>(i * 8) * s0 + ch.template select<8, 1>(i * 8) * s1) * sa[i];
  }
};

// IQ4_XS: the byte-pair table lives in shared local memory at offset 0 (kIqTableBytes).
constexpr int kIqTableBytes = 512;

template <>
struct Dec<WeightFormat::IQ4_XS> {
  static constexpr int CH = 4, TB = 128, REC_T = 8;
  struct Chunk {
    esimd::simd<uint8_t, CH * TB> t;
    esimd::simd<float, 8 * CH> s;
  };
  template <int H>
  SYCL_ESIMD_FUNCTION static void load(Chunk& c, const uint8_t* tp, const uint8_t* rp) {
    c.t = esimd::block_load<uint8_t, CH * TB>(tp);
    esimd::simd<float, 8> d = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rp));
    esimd::simd<uint16_t, 8> sh = esimd::block_load<uint16_t, 8>(reinterpret_cast<const uint16_t*>(rp + kIqScalesH));
    esimd::simd<uint8_t, 32> sl = esimd::block_load<uint8_t, 32>(rp + kIqScalesL);
#pragma unroll
    for (int u = 0; u < CH; ++u) {
      const int j = H * CH + u;  // the tile's sub-block
      esimd::simd<uint16_t, 8> l = sl.template select<8, 1>((j / 2) * 8);
      esimd::simd<uint16_t, 8> ls = ((l >> (4 * (j % 2))) & 15) | (((sh >> (2 * j)) & 3) << 4);
      esimd::simd<float, 8> lsf = ls;
      c.s.template select<8, 1>(u * 8) = d * (lsf - 32.0f);
    }
  }
  template <int U, int R>
  SYCL_ESIMD_FUNCTION static void tile(Acc<R>& acc, Chunk& c, const esimd::simd<uint8_t, 32 * R>& A, const esimd::simd<float, R>& sa) {
    esimd::simd<uint8_t, TB> P = c.t.template select<TB, 1>(U * TB);
    esimd::simd<uint32_t, TB> off = esimd::simd<uint32_t, TB>(P) << 1;
    esimd::simd<uint16_t, TB> w16;
#pragma unroll
    for (int i = 0; i < TB / 32; ++i) w16.template select<32, 1>(i * 32) = esimd::slm_gather<uint16_t, 32>(off.template select<32, 1>(i * 32));
    esimd::simd<uint8_t, 2 * TB> B = w16.template bit_cast_view<uint8_t>();
    esimd::simd<float, 8 * R> cf = dp<dpas_argument_type::s8, R, 2 * TB>(B, A);
    axpy<R>(acc, cf, c.s.template select<8, 1>(U * 8), sa);
  }
  // Fills the table of 256 words at `base`: entry b = int8 T[b & 15] | int8 T[b >> 4] << 8 (the two weights of one
  // packed byte). Thread t of `nt` fills the 256-byte blocks t, t + nt.
  SYCL_ESIMD_FUNCTION static void fill_table(int t, int nt, uint32_t base = 0) {
    esimd::simd<uint16_t, 16> lo16;
#pragma unroll
    for (int i = 0; i < 16; ++i) lo16[i] = uint16_t(uint8_t(kIq4Values[i]));
    esimd::simd<uint16_t, 128> lo = lo16.template replicate<8>();
    for (int blk = t; blk < 2; blk += nt) {
      esimd::simd<uint16_t, 128> v = lo;
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        // 16-entry groups m of block blk have the high nibble 8 blk + m.
        uint16_t hi = 0;
#pragma unroll
        for (int c = 0; c < 2; ++c) if (blk == c) hi = uint16_t(uint8_t(kIq4Values[8 * c + m])) << 8;
        v.template select<16, 1>(m * 16) |= hi;
      }
      esimd::slm_block_store<uint16_t, 128>(base + blk * 256, v);
    }
  }
};

}  // namespace eklow::xmx::fmt

namespace eklow::xmx::fmt {

// True for the formats with the layouts above and the decode GEMVs below (Q4_0's are xmx.cpp's own).
constexpr bool is_new_format(WeightFormat f) {
  return f == WeightFormat::Q8_0I || f == WeightFormat::Q4_K || f == WeightFormat::Q5_K || f == WeightFormat::Q6_K ||
         f == WeightFormat::IQ4_XS;
}

// ------------------------------------------------------------------------------------------------------------
// The fp16 GEMM's dequantization (xmx.cpp's Gemm16Kernel): a thread turns one "piece" (one tile's 16 K, 8 columns) of
// the format's bytes into the 128 fp16 of the B operand in VNNI order (element (k, n) of the piece at (k / 2) * 16
// + n * 2 + k % 2) and the scale of the tile's sub-block. Scales are combined in fp16 as the Q4_0 path does:
//   Q8_0I   int8 * d                          (one rounding)
//   Q4_K    q * fp16(d sc) - fp16(dmin m)       (two roundings; the scale and min are rounded to fp16)
//   Q5_K    as Q4_K with the 5-bit q
//   Q6_K    (q - 32) * fp16(d sc)
//   IQ4_XS  table value * fp16(d (ls - 32))
// A nibble q enters as the fp16 1024 + q (0x6400 over the mantissa), so that the offset comes off exactly.

// [s0, s0, s1, s1, ...] x 8: a column's scale at both of its elements of every K pair.
SYCL_ESIMD_FUNCTION inline esimd::simd<sycl::half, 128> vnni_cols(esimd::simd<sycl::half, 8> s) {
  esimd::simd<sycl::half, 16> s2;
  s2.select<8, 2>(0) = s;
  s2.select<8, 2>(1) = s;
  return s2.replicate<8>();
}

// 16 dwords (8 nibbles each, [kg][n]) to the 64 dwords of fp16 pairs 1024 + q, in VNNI order.
SYCL_ESIMD_FUNCTION inline esimd::simd<uint32_t, 64> nibbles_to_pairs(esimd::simd<uint32_t, 16> D) {
  esimd::simd<uint32_t, 64> W;
#pragma unroll
  for (int g2 = 0; g2 < 2; ++g2)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const esimd::simd<uint32_t, 8> t = esimd::simd<uint32_t, 8>(D.select<8, 1>(g2 * 8)) >> (8 * q);
      W.select<8, 1>((g2 * 4 + q) * 8) = (t & 0xF) | ((t << 12) & 0xF0000) | 0x64006400u;
    }
  return W;
}

// x shifted by s (right for s >= 0, left for s < 0).
SYCL_ESIMD_FUNCTION inline esimd::simd<uint32_t, 8> shift_by(esimd::simd<uint32_t, 8> x, int s) {
  if (s >= 0) return x >> s;
  return x << (-s);
}

template <WeightFormat F>
struct GemmDec;

template <>
struct GemmDec<WeightFormat::Q8_0I> {
  static constexpr int DW = 32, REC_T = 1;
  struct Meta {
    esimd::simd<sycl::half, 8> s;
  };
  SYCL_ESIMD_FUNCTION static void load_meta(Meta& m, const uint8_t* rec_g, int b, int h) {
    m.s = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rec_g + b * 16));
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(esimd::simd<uint32_t, DW> D, Meta& m, int b, int h, uint32_t slm) {
    // The s8 operand's dword (k4, n) holds k = 4 k4 .. 4 k4 + 3: its two words go to K pairs 2 k4 and 2 k4 + 1.
    esimd::simd<uint16_t, 64> src = D.bit_cast_view<uint16_t>(), dst;
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4)
#pragma unroll
      for (int a = 0; a < 2; ++a) dst.select<8, 1>((2 * k4 + a) * 8) = src.select<8, 2>(k4 * 16 + a);
    esimd::simd<int8_t, 128> b8 = dst.bit_cast_view<int8_t>();
    esimd::simd<sycl::half, 128> w = b8;
    return w * vnni_cols(m.s);
  }
};

// Q4_K and Q5_K: the record's 128 bytes (scales) and for Q5_K the tile's 8 dwords of high bits.
struct GemmMin {
  static constexpr int DW = 16, REC_T = 8;
  struct Meta {
    esimd::simd<uint8_t, 128> r;
    esimd::simd<uint32_t, 8> hw;  // Q5_K
  };
  // The tile's scale d sc and min dmin m as fp16 per column.
  SYCL_ESIMD_FUNCTION static void scales(Meta& m, int j, esimd::simd<sycl::half, 8>& ds, esimd::simd<sycl::half, 8>& dm) {
    esimd::simd<float, 8> d = m.r.select<16, 1>(0).bit_cast_view<sycl::half>();
    esimd::simd<float, 8> dmin = m.r.select<16, 1>(16).bit_cast_view<sycl::half>();
    esimd::simd<uint16_t, 8> sc, mn;
    if (j < 4) {
      sc = esimd::simd<uint16_t, 8>(m.r.select<8, 1>(32 + 8 * j)) & 63;
      mn = esimd::simd<uint16_t, 8>(m.r.select<8, 1>(32 + 8 * (j + 4))) & 63;
    } else {
      const esimd::simd<uint16_t, 8> s8 = m.r.select<8, 1>(32 + 8 * (j + 4)), s0 = m.r.select<8, 1>(32 + 8 * (j - 4)), s4 = m.r.select<8, 1>(32 + 8 * j);
      sc = (s8 & 15) | ((s0 >> 6) << 4);
      mn = (s8 >> 4) | ((s4 >> 6) << 4);
    }
    esimd::simd<float, 8> scf = sc, mnf = mn;
    ds = esimd::simd<sycl::half, 8>(d * scf);
    dm = esimd::simd<sycl::half, 8>(dmin * mnf);
  }
};

template <>
struct GemmDec<WeightFormat::Q4_K> : GemmMin {
  SYCL_ESIMD_FUNCTION static void load_meta(Meta& m, const uint8_t* rec_g, int b, int h) { m.r = esimd::block_load<uint8_t, 128>(rec_g + (b / 8) * 128); }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(esimd::simd<uint32_t, DW> D, Meta& m, int b, int h, uint32_t slm) {
    esimd::simd<sycl::half, 8> ds, dm;
    scales(m, b % 8, ds, dm);
    esimd::simd<uint32_t, 64> W = nibbles_to_pairs(D);
    esimd::simd<sycl::half, 128> x = W.bit_cast_view<sycl::half>();
    x = x - sycl::half(1024);
    x = x * vnni_cols(ds);  // two roundings, in two statements (no fused multiply-add: the emulation does the same)
    x = x - vnni_cols(dm);
    return x;
  }
};

template <>
struct GemmDec<WeightFormat::Q5_K> : GemmMin {
  SYCL_ESIMD_FUNCTION static void load_meta(Meta& m, const uint8_t* rec_g, int b, int h) {
    const uint8_t* rec = rec_g + (b / 8) * 384;
    m.r = esimd::block_load<uint8_t, 128>(rec);
    m.hw = esimd::block_load<uint32_t, 8>(reinterpret_cast<const uint32_t*>(rec + 128 + (b % 8) * 32));
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(esimd::simd<uint32_t, DW> D, Meta& m, int b, int h, uint32_t slm) {
    esimd::simd<sycl::half, 8> ds, dm;
    scales(m, b % 8, ds, dm);
    esimd::simd<uint32_t, 64> W = nibbles_to_pairs(D);
    // The piece's high bits: bit 2 j + h of a column's word is the high bit of k = 16 h + j; K pair kp holds j = 2 kp and
    // 2 kp + 1, which go to bits 4 and 20 of the pair's dword (bit 4 of the two fp16 mantissas).
    const esimd::simd<uint32_t, 8> hs = m.hw >> h;
#pragma unroll
    for (int kp = 0; kp < 8; ++kp)
      W.select<8, 1>(kp * 8) |= (shift_by(hs, 4 * kp - 4) & 0x10u) | (shift_by(hs, 4 * kp - 18) & 0x100000u);
    esimd::simd<sycl::half, 128> x = W.bit_cast_view<sycl::half>();
    x = x - sycl::half(1024);
    x = x * vnni_cols(ds);  // two roundings, in two statements (no fused multiply-add: the emulation does the same)
    x = x - vnni_cols(dm);
    return x;
  }
};

template <>
struct GemmDec<WeightFormat::Q6_K> {
  static constexpr int DW = 16, REC_T = 8;
  struct Meta {
    esimd::simd<int8_t, 16> sc;
    esimd::simd<sycl::half, 8> d;
    esimd::simd<uint32_t, 8> s2;
  };
  SYCL_ESIMD_FUNCTION static void load_meta(Meta& m, const uint8_t* rec_g, int b, int h) {
    const uint8_t* rec = rec_g + (b / 8) * 656;
    const int j = b % 8;
    m.sc = esimd::block_load<int8_t, 16>(reinterpret_cast<const int8_t*>(rec + 16 * j));
    m.d = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rec + kQ6KD));
    m.s2 = esimd::block_load<uint32_t, 8>(reinterpret_cast<const uint32_t*>(rec + kQ6KHigh + j * 64 + h * 32));
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(esimd::simd<uint32_t, DW> D, Meta& m, int b, int h, uint32_t slm) {
    // The piece is the tile's sub-block 2 j + h: its int8 scale is sc[8 h + n].
    esimd::simd<float, 8> scf = m.sc.select<8, 1>(8 * h), d = m.d;
    const esimd::simd<sycl::half, 8> ds = esimd::simd<sycl::half, 8>(d * scf);
    esimd::simd<uint32_t, 64> W = nibbles_to_pairs(D);
    // The stored field is (q >> 4) ^ 2; flipping its high bit back gives q >> 4 at bits 4..5 of each fp16.
    const esimd::simd<uint32_t, 8> hx = m.s2 ^ 0xAAAAAAAAu;
#pragma unroll
    for (int kp = 0; kp < 8; ++kp)
      W.select<8, 1>(kp * 8) |= (shift_by(hx, 4 * kp - 4) & 0x30u) | (shift_by(hx, 4 * kp - 18) & 0x300000u);
    esimd::simd<sycl::half, 128> x = W.bit_cast_view<sycl::half>();
    x = x - sycl::half(1056);
    return x * vnni_cols(ds);
  }
};

template <>
struct GemmDec<WeightFormat::IQ4_XS> {
  static constexpr int DW = 16, REC_T = 8;
  struct Meta {
    esimd::simd<sycl::half, 8> ds;
  };
  SYCL_ESIMD_FUNCTION static void load_meta(Meta& m, const uint8_t* rec_g, int b, int h) {
    const uint8_t* rec = rec_g + (b / 8) * 64;
    const int j = b % 8;
    esimd::simd<float, 8> d = esimd::block_load<sycl::half, 8>(reinterpret_cast<const sycl::half*>(rec));
    esimd::simd<uint16_t, 8> sh = esimd::block_load<uint16_t, 8>(reinterpret_cast<const uint16_t*>(rec + kIqScalesH));
    esimd::simd<uint8_t, 32> sl = esimd::block_load<uint8_t, 32>(rec + kIqScalesL);
    esimd::simd<uint16_t, 8> l = sl.select<8, 1>((j / 2) * 8);
    esimd::simd<uint16_t, 8> ls = ((l >> (4 * (j % 2))) & 15) | (((sh >> (2 * j)) & 3) << 4);
    esimd::simd<float, 8> lsf = ls;
    m.ds = esimd::simd<sycl::half, 8>(d * (lsf - 32.0f));
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(esimd::simd<uint32_t, DW> D, Meta& m, int b, int h, uint32_t slm) {
    esimd::simd<uint32_t, 64> off = esimd::simd<uint32_t, 64>(D.bit_cast_view<uint8_t>()) << 1;
    off += slm;
    esimd::simd<uint16_t, 64> src, dst;
    src.select<32, 1>(0) = esimd::slm_gather<uint16_t, 32>(off.select<32, 1>(0));
    src.select<32, 1>(32) = esimd::slm_gather<uint16_t, 32>(off.select<32, 1>(32));
    // Byte (k4 * 8 + n) * 2 + a of the piece holds the codes of k = 4 k4 + 2 a and the next: its word goes to K pair 2 k4 + a.
#pragma unroll
    for (int k4 = 0; k4 < 4; ++k4)
#pragma unroll
      for (int a = 0; a < 2; ++a) dst.select<8, 1>((2 * k4 + a) * 8) = src.select<8, 2>(k4 * 16 + a);
    esimd::simd<int8_t, 128> b8 = dst.bit_cast_view<int8_t>();
    esimd::simd<sycl::half, 128> w = b8;
    return w * vnni_cols(m.ds);
  }
};

// The Meta type of a format in the GEMM: the new formats' own, a placeholder for the rest.
template <WeightFormat F, bool New = is_new_format(F)>
struct GemmMetaOf {
  struct type {};
  static constexpr int DW = 16;
  static constexpr int REC_T = 1;
};
template <WeightFormat F>
struct GemmMetaOf<F, true> {
  using type = typename GemmDec<F>::Meta;
  static constexpr int DW = GemmDec<F>::DW;
  static constexpr int REC_T = GemmDec<F>::REC_T;
};

// The decode GEMVs on int8 activations (xmx_fmt_gemv.hpp, instantiated in xmx_gemv_<format>.cpp), and the runtime
// dispatch over them (xmx_fmt.cpp).
template <WeightFormat F>
void gemv(sycl::queue& q, const Activations& a, const Weight& w, float* out);
template <WeightFormat F>
void gemv_grouped(sycl::queue& q, const Activations& a, const Activations* ad, const ExpertTable& w, const int32_t* ids, int64_t P,
                  int64_t pe, int64_t row_div, float* out);
void gemv_any(sycl::queue& q, const Activations& a, const Weight& w, float* out);
void gemv_grouped_any(sycl::queue& q, const Activations& a, const Activations* ad, const ExpertTable& w, const int32_t* ids, int64_t P,
                      int64_t pe, int64_t row_div, float* out);

}  // namespace eklow::xmx::fmt
