// Inkling 2 Mini's experts on the A770M (moe.hpp; DECISIONS 2026-11-24, "Inkling M2: the experts").
//
// Layouts. A matrix is records of 8 output columns x 64 K ("a step"), [column unit][step][column group in unit] where a unit
// is `cg` column groups of 8 (gate_up: a gate group and the up group of the same rows, so one thread holds both for SITU;
// down: 2 groups, or 1 for Q8_0's 544-byte records). A record is `meta` bytes then the codes of 4 K16 pieces, each piece
// the 8 columns x 16 K of one dpas B operand:
//   NVFP4  (288 B, 4.5 bits)  meta: ue4m3 scale byte [piece][column] (ggml's 0x7f NaN stored as 0); piece: 16 dwords
//          [K octet][column], a dword holding 8 codes of a column, K pair q (K = 8 kg + 2q, 2q + 1) in nibbles q and 4 + q,
//          so that a K pair's two codes sit in the two fp16 lanes of one dword.
//   Q5_K   (384 B, 6 bits)    meta: per 32-K sub-block fp16 d*sc [column], fp16 dmin*m [column]; then the four pieces'
//          nibbles (16 dwords [K octet][column], K low nibble first), then the 5th bits, a u16 per column per piece.
//   Q8_0   (544 B, 8.5 bits)  meta: fp16 scale [32-K block][column]; piece: int8 in the fp16 B operand's order, word
//          (K pair, column).
// NVFP4 dequantizes without a table: a code's magnitude bits placed at fp16 bits 9..11 and the sign at 15, read as fp16, are
// the e2m1 value / 2^14 (the denormal 0.5 included; the dpas takes denormal inputs). The GEMVs feed those bits to the dpas
// as they are and multiply each K16 result by its group's scale * 2^14 in fp32 ("scale-after"); the GEMM, whose B tile
// serves many rows, multiplies the fp16 values by 2^7 and by the scale * 2^7 (at most 57,344) first: both exact in fp16.
#include "kernels/moe.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace eklow::moe {

namespace esimd = sycl::ext::intel::esimd;
namespace xmxd = sycl::ext::intel::esimd::xmx;

namespace {

constexpr int kStepK = 64;        // K a record covers
constexpr int kSlabTokens = 1024; // ffn runs longer steps in slabs of this many tokens (scratch size)
constexpr int kExperts = 128;     // the group kernel's work-group
constexpr int kGemvRows = 4;      // rows of a GEMV tile (at most: the dpas repeat count 1, 2 or 4)
constexpr int kGemmWM = 8;        // the GEMM tile: 16 * kGemmWM rows

inline int64_t env_int(const char* name, int64_t def) {
  const char* v = std::getenv(name);
  return v && *v ? std::atoll(v) : def;
}

// ---------------------------------------------------------------------------------------------------------------------
// Format geometry and the device decoders.

template <Format F>
struct Fd;
template <>
struct Fd<Format::NVFP4> {
  static constexpr int REC = 288;
};
template <>
struct Fd<Format::Q5_K> {
  static constexpr int REC = 384;
};
template <>
struct Fd<Format::Q8_0> {
  static constexpr int REC = 544;
};

constexpr int record_bytes(Format f) { return f == Format::NVFP4 ? 288 : f == Format::Q5_K ? 384 : 544; }

// [s0, s0, s1, s1, ...] x 8: a column's scale at both of its elements of every K pair.
SYCL_ESIMD_FUNCTION inline esimd::simd<sycl::half, 128> vnni_cols(esimd::simd<sycl::half, 8> s) {
  esimd::simd<sycl::half, 16> s2;
  s2.select<8, 2>(0) = s;
  s2.select<8, 2>(1) = s;
  return s2.replicate<8>();
}

// x shifted by s (right for s >= 0, left for s < 0).
SYCL_ESIMD_FUNCTION inline esimd::simd<uint32_t, 8> shift_by(esimd::simd<uint32_t, 8> x, int s) {
  if (s >= 0) return x >> s;
  return x << (-s);
}

// 16 dwords ([K octet][column], 8 nibbles, K low nibble first) as the fp16 1024 + q in VNNI order.
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

// Piece H (8 columns x 16 K, the dpas B operand in VNNI order: element (k, n) at (k / 2) * 16 + n * 2 + k % 2) of a record:
// `core` turns the piece's own bytes into it, `piece<H>` takes them from a whole record in registers, `Pc` + `load` +
// `dequant` from the record in memory (a thread dequantizing one piece of a work-group's step).
template <Format F>
struct Dec;

template <>
struct Dec<Format::NVFP4> {
  // sb: the piece's 8 scale bytes (ue4m3), cb: its 64 code bytes
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> core(esimd::simd<uint8_t, 8> sb, esimd::simd<uint8_t, 64> cb) {
    // The scales: ue4m3 (e, m) = (8 [e > 0] + m) 2^(max(e, 1) - 10), times 2^7.
    esimd::simd<uint32_t, 8> x = sb;
    esimd::simd<uint32_t, 8> e = (x >> 3) & 15, m = x & 7, ex = e, mant = m;
    ex.merge(esimd::simd<uint32_t, 8>(1), e == 0);
    mant.merge(m + 8, e != 0);
    esimd::simd<uint32_t, 8> pb = (ex + 117) << 23;
    esimd::simd<float, 8> p2 = pb.template bit_cast_view<float>();
    esimd::simd<float, 8> mf = mant;
    esimd::simd<float, 8> scf = mf * p2 * 128.0f;
    esimd::simd<sycl::half, 8> sc = scf;
    esimd::simd<uint32_t, 16> D = cb.template bit_cast_view<uint32_t>();
    esimd::simd<uint32_t, 64> W;
#pragma unroll
    for (int kg = 0; kg < 2; ++kg) {
      const esimd::simd<uint32_t, 8> d = D.template select<8, 1>(kg * 8);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        // nibbles q (low weight) and 4 + q (high weight): magnitude bits 4q..4q+2 to 9..11, the sign bit 4q+3 to 15
        const esimd::simd<uint32_t, 8> m_ = shift_by(d, 4 * q - 9) & 0x0E000E00u;
        const esimd::simd<uint32_t, 8> s_ = shift_by(d, 4 * q - 12) & 0x80008000u;
        W.template select<8, 1>((kg * 4 + q) * 8) = m_ | s_;
      }
    }
    esimd::simd<sycl::half, 128> w = W.template bit_cast_view<sycl::half>();
    w = w * sycl::half(128.0f);
    return w * vnni_cols(sc);
  }
  template <int H>
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> piece(esimd::simd<uint8_t, 288>& r) {
    return core(r.template select<8, 1>(8 * H), r.template select<64, 1>(32 + 64 * H));
  }
  struct Pc {
    esimd::simd<uint8_t, 8> m;
    esimd::simd<uint8_t, 64> c;
  };
  SYCL_ESIMD_FUNCTION static void load(Pc& p, const uint8_t* rec, int h) {
    p.m = esimd::block_load<uint8_t, 8>(rec + 8 * h);
    p.c = esimd::block_load<uint8_t, 64>(rec + 32 + 64 * h);
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(Pc& p) { return core(p.m, p.c); }
  // The scale-after form for the GEMVs: the code bits as fp16 are the e2m1 value / 2^14 (denormals the dpas takes as they
  // are), the scales of a record's 4 pieces x 8 columns (32 bytes) as fp32 ue4m3 * 2^14, applied to the dpas result.
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> bits(esimd::simd<uint8_t, 64> cb) {
    esimd::simd<uint32_t, 16> D = cb.template bit_cast_view<uint32_t>();
    esimd::simd<uint32_t, 64> W;
#pragma unroll
    for (int kg = 0; kg < 2; ++kg) {
      const esimd::simd<uint32_t, 8> d = D.template select<8, 1>(kg * 8);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const esimd::simd<uint32_t, 8> m_ = shift_by(d, 4 * q - 9) & 0x0E000E00u;
        const esimd::simd<uint32_t, 8> s_ = shift_by(d, 4 * q - 12) & 0x80008000u;
        W.template select<8, 1>((kg * 4 + q) * 8) = m_ | s_;
      }
    }
    return W.template bit_cast_view<sycl::half>();
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<float, 32> scales(esimd::simd<uint8_t, 32> sb) {
    esimd::simd<uint32_t, 32> x = sb;
    esimd::simd<uint32_t, 32> e = (x >> 3) & 15, m = x & 7, ex = e, mant = m;
    ex.merge(esimd::simd<uint32_t, 32>(1), e == 0);
    mant.merge(m + 8, e != 0);
    esimd::simd<uint32_t, 32> pb = (ex + 131) << 23;  // 2^(ex - 10 + 14)
    esimd::simd<float, 32> p2 = pb.template bit_cast_view<float>();
    esimd::simd<float, 32> mf = mant;
    return mf * p2;
  }
};

template <>
struct Dec<Format::Q5_K> {
  // dd: the sub-block's fp16 d*sc [8] and dmin*m [8], nb: the piece's nibbles, hb: its 5th bits
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> core(esimd::simd<uint8_t, 32> dd, esimd::simd<uint8_t, 64> nb,
                                                               esimd::simd<uint8_t, 16> hb) {
    esimd::simd<sycl::half, 16> dh = dd.template bit_cast_view<sycl::half>();
    esimd::simd<sycl::half, 8> ds = dh.template select<8, 1>(0), dm = dh.template select<8, 1>(8);
    esimd::simd<uint32_t, 16> D = nb.template bit_cast_view<uint32_t>();
    esimd::simd<uint32_t, 64> W = nibbles_to_pairs(D);
    esimd::simd<uint16_t, 8> h16 = hb.template bit_cast_view<uint16_t>();
    const esimd::simd<uint32_t, 8> hm = h16;
#pragma unroll
    for (int kp = 0; kp < 8; ++kp)  // K = 2 kp and 2 kp + 1: bits 2 kp and 2 kp + 1 to bits 4 and 20 of the pair's dword
      W.template select<8, 1>(kp * 8) |= (shift_by(hm, 2 * kp - 4) & 0x10u) | (shift_by(hm, 2 * kp - 19) & 0x100000u);
    esimd::simd<sycl::half, 128> x = W.template bit_cast_view<sycl::half>();
    x = x - sycl::half(1024);
    x = x * vnni_cols(ds);  // two roundings, in two statements, as the reference's emulation does
    x = x - vnni_cols(dm);
    return x;
  }
  template <int H>
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> piece(esimd::simd<uint8_t, 384>& r) {
    return core(r.template select<32, 1>((H / 2) * 32), r.template select<64, 1>(64 + 64 * H), r.template select<16, 1>(64 + 256 + 16 * H));
  }
  struct Pc {
    esimd::simd<uint8_t, 32> d;
    esimd::simd<uint8_t, 64> n;
    esimd::simd<uint8_t, 16> h;
  };
  SYCL_ESIMD_FUNCTION static void load(Pc& p, const uint8_t* rec, int h) {
    p.d = esimd::block_load<uint8_t, 32>(rec + (h / 2) * 32);
    p.n = esimd::block_load<uint8_t, 64>(rec + 64 + 64 * h);
    p.h = esimd::block_load<uint8_t, 16>(rec + 64 + 256 + 16 * h);
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(Pc& p) { return core(p.d, p.n, p.h); }
};

template <>
struct Dec<Format::Q8_0> {
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> core(esimd::simd<uint8_t, 16> sb, esimd::simd<uint8_t, 128> qb) {
    esimd::simd<sycl::half, 8> sc = sb.template bit_cast_view<sycl::half>();
    esimd::simd<int8_t, 128> q8 = qb.template bit_cast_view<int8_t>();
    esimd::simd<sycl::half, 128> w = q8;
    return w * vnni_cols(sc);
  }
  template <int H>
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> piece(esimd::simd<uint8_t, 544>& r) {
    return core(r.template select<16, 1>((H / 2) * 16), r.template select<128, 1>(32 + 128 * H));
  }
  struct Pc {
    esimd::simd<uint8_t, 16> s;
    esimd::simd<uint8_t, 128> q;
  };
  SYCL_ESIMD_FUNCTION static void load(Pc& p, const uint8_t* rec, int h) {
    p.s = esimd::block_load<uint8_t, 16>(rec + (h / 2) * 16);
    p.q = esimd::block_load<uint8_t, 128>(rec + 32 + 128 * h);
  }
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> dequant(Pc& p) { return core(p.s, p.q); }
};

// act16 layout (xmx's): row tiles of 8 rows x K16, 256 bytes each, row-major in the tile.
inline int64_t act16_index(int64_t row, int64_t k, int64_t kh) { return ((row / 8) * kh + k / 16) * 128 + (row % 8) * 16 + k % 16; }

// SITU: beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb) (ESIMD has no tanh: tanh x = 1 - 2 / (exp(2x) + 1)).
SYCL_ESIMD_FUNCTION inline esimd::simd<float, 8> situ8(esimd::simd<float, 8> g, esimd::simd<float, 8> u, float beta, float lb) {
  const esimd::simd<float, 8> tg = 1.0f - 2.0f / (esimd::exp(g * (2.0f / beta)) + 1.0f);
  const esimd::simd<float, 8> sg = 1.0f / (1.0f + esimd::exp(-g));
  esimd::simd<float, 8> lu = u;
  if (lb > 0) {
    const esimd::simd<float, 8> tu = 1.0f - 2.0f / (esimd::exp(u * (2.0f / lb)) + 1.0f);
    lu = tu * lb;
  }
  return beta * tg * sg * lu;
}

// ---------------------------------------------------------------------------------------------------------------------
// Grouping the (token, slot) pairs by expert.
//   tl[0] = number of tiles, tl[1] = rows used (segments start at multiples of 8); from tl[4], four ints a tile: expert,
//   first row, rows holding pairs. An expert's pairs fill its segment in token order (stable, so results do not depend on
//   scheduling); a segment is cut into tiles of T rows. rop[p] = pair p's row.
void group(sycl::queue& q, const int32_t* ids, const int32_t* slot_of, int64_t P, int64_t E, int T, int32_t* tl, int32_t* rop) {
  // 128 items: item i counts the pairs of its contiguous chunk per expert (hist[i][e], 16 bits), item e then turns column
  // e of the table into the chunks' starts within expert e and the expert's count, one item lays out the segments and
  // tiles, and item i places its chunk's pairs: stable, deterministic, no atomics.
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint16_t, 1> hist(sycl::range<1>(kExperts * kExperts), h);
    sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(kExperts), h), start(sycl::range<1>(kExperts), h);
    const int32_t p_n = int32_t(P), e_n = int32_t(E), chunk = int32_t((P + kExperts - 1) / kExperts);
    h.parallel_for(sycl::nd_range<1>(kExperts, kExperts), [=](sycl::nd_item<1> it) {
      const int lid = int(it.get_local_id(0));
      for (int e = 0; e < e_n; ++e) hist[lid * kExperts + e] = 0;
      const int p0 = lid * chunk, p1 = sycl::min(p0 + chunk, p_n);
      for (int p = p0; p < p1; ++p) {
        const int e = slot_of ? slot_of[ids[p]] : ids[p];
        if (e >= 0) hist[lid * kExperts + e] += 1;
      }
      sycl::group_barrier(it.get_group());
      if (lid < e_n) {
        int run = 0;
        for (int i = 0; i < kExperts; ++i) {
          const int c = hist[i * kExperts + lid];
          hist[i * kExperts + lid] = uint16_t(run);
          run += c;
        }
        cnt[lid] = run;
      }
      sycl::group_barrier(it.get_group());
      if (lid == 0) {
        int nt = 0, row = 0;
        for (int e = 0; e < e_n; ++e) {
          const int c = cnt[e];
          start[e] = row;
          for (int t0 = 0; t0 < c; t0 += T) {
            tl[4 + 4 * nt] = e;
            tl[5 + 4 * nt] = row + t0;
            tl[6 + 4 * nt] = sycl::min(T, c - t0);
            ++nt;
          }
          row += (c + 7) / 8 * 8;
        }
        tl[0] = nt;
        tl[1] = row;
      }
      sycl::group_barrier(it.get_group());
      for (int p = p0; p < p1; ++p) {
        const int e = slot_of ? slot_of[ids[p]] : ids[p];
        if (e < 0) {
          rop[p] = -1;
          continue;
        }
        rop[p] = start[e] + int(hist[lid * kExperts + e]);
        hist[lid * kExperts + e] += 1;
      }
    });
  });
}

// x[p / n_used][:] (fp32) of every pair p into its sorted row of `a` (act16 layout, width k).
struct Gather {
  const float* x;
  const int32_t* rop;
  sycl::half* a;
  int32_t n_used, k;
  void operator()(sycl::id<2> id) const SYCL_ESIMD_KERNEL {
    const int p = int(id[0]), j = int(id[1]);
    const int row = rop[p], kh = k / 16;
    if (row < 0) return;
    esimd::simd<float, 64> v = esimd::block_load<float, 64>(x + int64_t(p / n_used) * k + j * 64);
    esimd::simd<sycl::half, 64> hv = v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      esimd::block_store<sycl::half, 16>(a + act16_index(row, (j * 4 + i) * 16, kh), hv.template select<16, 1>(i * 16));
  }
};

// out[t][:] = sum_i w[t][i] y[rop[t n_used + i]][:], in the order i = 0 .. n_used - 1 (the reference's).
struct Combine {
  const float* y;
  const float* w;
  const int32_t* rop;
  float* out;
  const float* cold;  // or null
  int32_t n_used, n;
  void operator()(sycl::id<2> id) const SYCL_ESIMD_KERNEL {
    const int t = int(id[0]), j = int(id[1]);
    esimd::simd<float, 128> acc = 0;
    for (int i = 0; i < n_used; ++i) {
      const float wt = w[t * n_used + i];
      const int row = rop[t * n_used + i];
      if (row < 0) continue;
      acc += wt * esimd::block_load<float, 128>(y + int64_t(row) * n + j * 128);
    }
    if (cold) acc += esimd::block_load<float, 128>(cold + int64_t(t) * n + j * 128);
    esimd::block_store<float, 128>(out + int64_t(t) * n + j * 128, acc);
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// Decode: a grouped GEMV over row tiles of 8. A work-group is one tile and one column unit; its S threads split K and
// reduce through shared local memory. GU: the unit is a gate group and its up group, the epilogue SITU into fp16 rows of
// `out_h` (width n_ff); down: fp32 rows of `out_y`. Each thread streams its slice's records double-buffered in registers
// (the next step's loads go out before this step's dpas); the row count of the tile picks the dpas repeat count.
template <Format F, bool GU, int S, int CG, bool SA>
struct Gemv {
  static constexpr int REC = Fd<F>::REC;
  const sycl::half* a;
  const uint8_t* data;
  uint64_t expert_bytes;
  const int32_t* tl;
  const float* scale;  // per expert, or null
  sycl::half* out_h;
  float* out_y;
  int32_t kst, n_out;  // K / 64; down's N
  int32_t n_ff;        // GU: the output width
  float beta, lb;

  template <int RC>
  SYCL_ESIMD_FUNCTION void run(int u, int sl, int e, int row0, int valid) const {
    const int ns = kst / S, s0 = sl * ns;
    const uint8_t* eb = data + uint64_t(e) * expert_bytes;
    const sycl::half* ap = a + int64_t(row0 / 8) * (4 * int64_t(kst)) * 128 + (row0 % 8) * 16;
    esimd::simd<uint8_t, REC> bufA[CG], bufB[CG];
    esimd::simd<float, 8 * RC> acc[CG][4];
#pragma unroll
    for (int c = 0; c < CG; ++c)
#pragma unroll
      for (int h = 0; h < 4; ++h) acc[c][h] = 0;
    auto ld = [&](esimd::simd<uint8_t, REC>(&buf)[CG], int s) SYCL_ESIMD_FUNCTION {
      const uint8_t* p = eb + (int64_t(u) * kst + s) * CG * REC;
#pragma unroll
      for (int c = 0; c < CG; ++c) buf[c] = esimd::block_load<uint8_t, REC>(p + c * REC);
    };
    auto compute = [&](esimd::simd<uint8_t, REC>(&buf)[CG], int s) SYCL_ESIMD_FUNCTION {
      const sycl::half* as = ap + 4 * int64_t(s) * 128;
      if constexpr (SA) {
#pragma unroll
        for (int c = 0; c < CG; ++c) {
          esimd::simd<uint8_t, 32> sb = buf[c].template select<32, 1>(0);
          esimd::simd<float, 32> sc = Dec<F>::scales(sb);
          const esimd::simd<sycl::half, 16 * RC> A0 = esimd::block_load<sycl::half, 16 * RC>(as), A1 = esimd::block_load<sycl::half, 16 * RC>(as + 128),
                                                 A2 = esimd::block_load<sycl::half, 16 * RC>(as + 256), A3 = esimd::block_load<sycl::half, 16 * RC>(as + 384);
          esimd::simd<float, 8 * RC> r0 = xmxd::dpas<8, RC, float>(Dec<F>::bits(buf[c].template select<64, 1>(32)), A0);
          esimd::simd<float, 8 * RC> r1 = xmxd::dpas<8, RC, float>(Dec<F>::bits(buf[c].template select<64, 1>(96)), A1);
          esimd::simd<float, 8 * RC> r2 = xmxd::dpas<8, RC, float>(Dec<F>::bits(buf[c].template select<64, 1>(160)), A2);
          esimd::simd<float, 8 * RC> r3 = xmxd::dpas<8, RC, float>(Dec<F>::bits(buf[c].template select<64, 1>(224)), A3);
          { esimd::simd<float, 8> sv0 = sc.template select<8, 1>(0); acc[c][0] += r0 * sv0.template replicate<RC>(); }
          { esimd::simd<float, 8> sv1 = sc.template select<8, 1>(8); acc[c][1] += r1 * sv1.template replicate<RC>(); }
          { esimd::simd<float, 8> sv2 = sc.template select<8, 1>(16); acc[c][2] += r2 * sv2.template replicate<RC>(); }
          { esimd::simd<float, 8> sv3 = sc.template select<8, 1>(24); acc[c][3] += r3 * sv3.template replicate<RC>(); }
        }
        return;
      }
      // one K16 piece at a time: its activation tile once for the unit's column groups
      {
        const esimd::simd<sycl::half, 16 * RC> A = esimd::block_load<sycl::half, 16 * RC>(as);
#pragma unroll
        for (int c = 0; c < CG; ++c) acc[c][0] = xmxd::dpas<8, RC, float>(acc[c][0], Dec<F>::template piece<0>(buf[c]), A);
      }
      {
        const esimd::simd<sycl::half, 16 * RC> A = esimd::block_load<sycl::half, 16 * RC>(as + 128);
#pragma unroll
        for (int c = 0; c < CG; ++c) acc[c][1] = xmxd::dpas<8, RC, float>(acc[c][1], Dec<F>::template piece<1>(buf[c]), A);
      }
      {
        const esimd::simd<sycl::half, 16 * RC> A = esimd::block_load<sycl::half, 16 * RC>(as + 256);
#pragma unroll
        for (int c = 0; c < CG; ++c) acc[c][2] = xmxd::dpas<8, RC, float>(acc[c][2], Dec<F>::template piece<2>(buf[c]), A);
      }
      {
        const esimd::simd<sycl::half, 16 * RC> A = esimd::block_load<sycl::half, 16 * RC>(as + 384);
#pragma unroll
        for (int c = 0; c < CG; ++c) acc[c][3] = xmxd::dpas<8, RC, float>(acc[c][3], Dec<F>::template piece<3>(buf[c]), A);
      }
    };
    ld(bufA, s0);
    for (int i = 0; i < ns; i += 2) {
      if (i + 1 < ns) ld(bufB, s0 + i + 1);
      compute(bufA, s0 + i);
      if (i + 1 < ns) {
        if (i + 2 < ns) ld(bufA, s0 + i + 2);
        compute(bufB, s0 + i + 1);
      }
    }
    esimd::simd<float, 8 * RC> tot[CG];
#pragma unroll
    for (int c = 0; c < CG; ++c) tot[c] = acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    if constexpr (S > 1) {
#pragma unroll
      for (int c = 0; c < CG; ++c) esimd::slm_block_store<float, 8 * RC>((sl * CG + c) * 256, tot[c]);
      esimd::barrier();
      if (sl != 0) return;
#pragma unroll
      for (int c = 0; c < CG; ++c)
#pragma unroll
        for (int s = 1; s < S; ++s) tot[c] += esimd::slm_block_load<float, 8 * RC>((s * CG + c) * 256);
    }
    const float sc = scale ? scale[e] : 1.0f;
#pragma unroll
    for (int i = 0; i < RC; ++i) {
      if (i >= valid) break;
      const int row = row0 + i;
      if constexpr (GU) {
        const esimd::simd<float, 8> g = tot[0].template select<8, 1>(i * 8) * sc, up = tot[1].template select<8, 1>(i * 8) * sc;
        esimd::simd<sycl::half, 8> h = situ8(g, up, beta, lb);
        esimd::block_store<sycl::half, 8>(out_h + act16_index(row, u * 8, n_ff / 16), h);
      } else {
#pragma unroll
        for (int c = 0; c < CG; ++c)
          esimd::block_store<float, 8>(out_y + int64_t(row) * n_out + (u * CG + c) * 8, tot[c].template select<8, 1>(i * 8) * sc);
      }
    }
  }

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<S * CG * 256>();
    const int tm = int(it.get_group(0)), u = int(it.get_group(1)), sl = int(it.get_local_id(0));
    if (tm >= tl[0]) return;
    const int e = tl[4 + 4 * tm], row0 = tl[5 + 4 * tm], valid = tl[6 + 4 * tm];
    if (valid <= 1) run<1>(u, sl, e, row0, valid);
    else if (valid <= 2) run<2>(u, sl, e, row0, valid);
    else run<4>(u, sl, e, row0, valid);
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// Prefill: a grouped fp16 GEMM. A work-group of WM x WN threads takes one tile of 16 WM rows of one expert and 32 WN
// output columns (4 WN column groups); a thread computes 16 x 32. Per K step the threads dequantize the group's records
// to fp16 B operands (a thread a piece: 8 columns x 16 K) and copy the activation pieces into shared local memory (two buffers:
// the next step's global loads go out before this step's dpas), then every thread reads its operands from there. D > 1
// (loads D steps ahead) and NA > 1 (more accumulators) were measured slower: the registers spill.
template <Format F, bool GU, int WM, int WN, int CG, int NA, int D>
struct Gemm {
  static constexpr int NT = WM * WN, BT = WN * 4, AT = WM * 2, REC = Fd<F>::REC;
  static constexpr int kBPieces = BT * 4, kAPieces = AT * 4;
  static constexpr int BP = kBPieces / NT, AP = (kAPieces + NT - 1) / NT;  // pieces a thread dequantizes / copies a step
  static constexpr uint32_t kBBuf = kBPieces * 256, kBuf = kBBuf + kAPieces * 256;
  static_assert(kBPieces % NT == 0, "the pieces of a step split evenly over the threads");
  const sycl::half* a;
  const uint8_t* data;
  uint64_t expert_bytes;
  const int32_t* tl;
  const float* scale;
  sycl::half* out_h;
  float* out_y;
  int32_t kst, n_out, n_ff;
  float beta, lb;

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<2 * kBuf>();
    const int tid = int(it.get_local_linear_id()), wi = tid / WN, wj = tid % WN;
    const int tm = int(it.get_group(0)), tn = int(it.get_group(1));
    const bool live = tm < tl[0];
    const int e = live ? tl[4 + 4 * tm] : 0, row0 = live ? tl[5 + 4 * tm] : 0, valid = live ? tl[6 + 4 * tm] : 0;
    const int kbl = live ? int(kst) : 0, kh = 4 * kst;
    const uint8_t* eb = data + uint64_t(e) * expert_bytes;
    typename Dec<F>::Pc R[D][BP];
    esimd::simd<sycl::half, 128> Araw[AP];
    // thread t dequantizes pieces t + i NT of the step: column group p / 4 of the work-group, K16 piece p % 4
    auto load = [&](int slot, int b) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < BP; ++i) {
        const int p = tid + i * NT, g = tn * BT + p / 4;
        Dec<F>::load(R[slot][i], eb + ((int64_t(g / CG) * kst + b) * CG + g % CG) * REC, p % 4);
      }
    };
    auto load_a = [&](int b) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < AP; ++i) {
        const int p = tid + i * NT;
        if (p >= kAPieces) break;
        Araw[i] = esimd::block_load<sycl::half, 128>(a + (int64_t(row0 / 8 + p / 4) * kh + 4 * b + p % 4) * 128);
      }
    };
    auto store = [&](int slot, uint32_t buf) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < BP; ++i) esimd::slm_block_store<sycl::half, 128>(buf + (tid + i * NT) * 256, Dec<F>::dequant(R[slot][i]));
#pragma unroll
      for (int i = 0; i < AP; ++i)
        if (tid + i * NT < kAPieces) esimd::slm_block_store<sycl::half, 128>(buf + kBBuf + (tid + i * NT) * 256, Araw[i]);
    };
    esimd::simd<float, 64> acc[NA][2][4];
#pragma unroll
    for (int n = 0; n < NA; ++n)
#pragma unroll
      for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int c = 0; c < 4; ++c) acc[n][r][c] = 0;
    if (kbl > 0) {
#pragma unroll
      for (int j = 0; j < D; ++j)
        if (j < kbl) load(j, j);
      load_a(0);
      store(0, 0);
    }
    esimd::barrier();
    // step b computes from the shared buffer b % 2 while the registers of slot b % D hold step b's pieces, already stored;
    // loads for step b + D go into that slot, and step b + 1's pieces (slot (b + 1) % D) are dequantized into the other buffer
    for (int b0 = 0; b0 < kbl; b0 += D) {
#pragma unroll
      for (int j = 0; j < D; ++j) {
        const int b = b0 + j;
        if (b >= kbl) break;
        const uint32_t cur = (b & 1) * kBuf, nxt = kBuf - cur;
        if (b + D < kst) load(j, b + D);
        if (b + 1 < kst) load_a(b + 1);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          esimd::simd<sycl::half, 128> A[2];
#pragma unroll
          for (int r = 0; r < 2; ++r) A[r] = esimd::slm_block_load<sycl::half, 128>(cur + kBBuf + ((wi * 2 + r) * 4 + h) * 256);
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const esimd::simd<sycl::half, 128> B = esimd::slm_block_load<sycl::half, 128>(cur + ((wj * 4 + c) * 4 + h) * 256);
#pragma unroll
            for (int r = 0; r < 2; ++r)
              if ((wi * 2 + r) * 8 < valid) acc[h % NA][r][c] = xmxd::dpas<8, 8, float>(acc[h % NA][r][c], B, A[r]);
          }
        }
        if (b + 1 < kst) store((j + 1) % D, nxt);
        esimd::barrier();
      }
    }
    if (!live) return;
    const float sc = scale ? scale[e] : 1.0f;
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int n = 1; n < NA; ++n) acc[0][r][c] += acc[n][r][c];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int lr = (wi * 2 + r) * 8 + i;
        if (lr >= valid) continue;
        const int row = row0 + lr;
        if constexpr (GU) {
#pragma unroll
          for (int c = 0; c < 4; c += 2) {
            const esimd::simd<float, 8> g = acc[0][r][c].template select<8, 1>(i * 8) * sc,
                                        up = acc[0][r][c + 1].template select<8, 1>(i * 8) * sc;
            esimd::simd<sycl::half, 8> h = situ8(g, up, beta, lb);
            const int pu = (tn * BT + wj * 4 + c) / 2;
            esimd::block_store<sycl::half, 8>(out_h + act16_index(row, pu * 8, n_ff / 16), h);
          }
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c)
            esimd::block_store<float, 8>(out_y + int64_t(row) * n_out + (tn * BT + wj * 4 + c) * 8,
                                         acc[0][r][c].template select<8, 1>(i * 8) * sc);
        }
      }
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// Repack: one work-item a record, from ggml's blocks (rows of the expert on the card) to the layouts above.

inline sycl::half half_of(const uint8_t* p) { return sycl::bit_cast<sycl::half>(uint16_t(p[0] | (p[1] << 8))); }
inline void put_half(uint8_t* p, sycl::half h) {
  const uint16_t b = sycl::bit_cast<uint16_t>(h);
  p[0] = uint8_t(b);
  p[1] = uint8_t(b >> 8);
}

template <Format F>
inline void build_record(const uint8_t* raw, uint64_t row_bytes, int64_t row0, int s, uint8_t* o);

template <>
inline void build_record<Format::NVFP4>(const uint8_t* raw, uint64_t row_bytes, int64_t row0, int s, uint8_t* o) {
  for (int n = 0; n < 8; ++n) {
    const uint8_t* blk = raw + uint64_t(row0 + n) * row_bytes + uint64_t(s) * 36;
    for (int h = 0; h < 4; ++h) {
      o[h * 8 + n] = blk[h] == 0x7f ? 0 : blk[h];
      for (int kg = 0; kg < 2; ++kg) {
        uint32_t d = 0;
        for (int q = 0; q < 4; ++q)
          for (int j = 0; j < 2; ++j) {
            const int k = 8 * kg + 2 * q + j;
            const uint8_t byte = blk[4 + h * 8 + (k & 7)];
            const uint32_t code = k < 8 ? (byte & 15) : (byte >> 4);
            d |= code << (4 * (j ? 4 + q : q));
          }
        uint8_t* w = o + 32 + (h * 16 + kg * 8 + n) * 4;
        w[0] = uint8_t(d);
        w[1] = uint8_t(d >> 8);
        w[2] = uint8_t(d >> 16);
        w[3] = uint8_t(d >> 24);
      }
    }
  }
}

template <>
inline void build_record<Format::Q8_0>(const uint8_t* raw, uint64_t row_bytes, int64_t row0, int s, uint8_t* o) {
  for (int n = 0; n < 8; ++n)
    for (int bb = 0; bb < 2; ++bb) {
      const uint8_t* blk = raw + uint64_t(row0 + n) * row_bytes + uint64_t(2 * s + bb) * 34;
      o[(bb * 8 + n) * 2] = blk[0];
      o[(bb * 8 + n) * 2 + 1] = blk[1];
      for (int hh = 0; hh < 2; ++hh)
        for (int kp = 0; kp < 8; ++kp)
          for (int j = 0; j < 2; ++j) o[32 + (bb * 2 + hh) * 128 + (kp * 8 + n) * 2 + j] = blk[2 + hh * 16 + 2 * kp + j];
    }
}

template <>
inline void build_record<Format::Q5_K>(const uint8_t* raw, uint64_t row_bytes, int64_t row0, int s, uint8_t* o) {
  const int sbk = s / 4, j64 = s % 4;
  for (int n = 0; n < 8; ++n) {
    const uint8_t* blk = raw + uint64_t(row0 + n) * row_bytes + uint64_t(sbk) * 176;
    const float d = float(half_of(blk)), dmin = float(half_of(blk + 2));
    const uint8_t *sc = blk + 4, *qh = blk + 16, *qs = blk + 48;
    for (int bb = 0; bb < 2; ++bb) {
      const int j = 2 * j64 + bb;
      int scv, mv;
      if (j < 4) {
        scv = sc[j] & 63;
        mv = sc[j + 4] & 63;
      } else {
        scv = (sc[j + 4] & 15) | ((sc[j - 4] >> 6) << 4);
        mv = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
      }
      put_half(o + bb * 32 + n * 2, sycl::half(d * float(scv)));
      put_half(o + bb * 32 + 16 + n * 2, sycl::half(dmin * float(mv)));
      for (int l = 0; l < 32; ++l) {
        const uint8_t byte = qs[32 * j64 + l];
        const int q = (bb == 0 ? (byte & 15) : (byte >> 4)) | (((qh[l] >> j) & 1) << 4);
        const int kin = 32 * bb + l, h = kin / 16, kk = kin % 16;
        uint8_t* nib = o + 64 + h * 64 + ((kk / 8) * 8 + n) * 4 + (kk % 8) / 2;
        if (kk % 2 == 0) *nib = uint8_t((*nib & 0xF0) | (q & 15));
        else *nib = uint8_t((*nib & 0x0F) | ((q & 15) << 4));
        uint8_t* hi = o + 64 + 256 + h * 16 + n * 2 + kk / 8;
        const int hb = kk % 8;
        *hi = uint8_t((*hi & ~(1 << hb)) | ((q >> 4) << hb));
      }
    }
  }
}

// `count` experts from raw (device, expert x at raw + x * raw_stride) into dst (expert x at dst + x * dst_stride).
template <Format F>
void repack_group(sycl::queue& q, const uint8_t* raw, uint64_t raw_stride, uint64_t row_bytes, int64_t k, int64_t n, int cg,
                  bool interleave, uint8_t* dst, uint64_t dst_stride, int count) {
  const int kst = int(k / kStepK), nrec = int(n / 8) * kst, n_i = int(n);
  constexpr int REC = Fd<F>::REC;
  q.parallel_for(sycl::range<2>(size_t(count), size_t(nrec)), [=](sycl::id<2> id) {
    const int x = int(id[0]), rec = int(id[1]);
    const int c = rec % cg, t = rec / cg, s = t % kst, u = t / kst, cgi = u * cg + c;
    const int64_t row0 = interleave ? (cgi % 2 == 0 ? 8 * (cgi / 2) : n_i / 2 + 8 * (cgi / 2)) : 8 * int64_t(cgi);
    uint8_t* o = dst + uint64_t(x) * dst_stride + uint64_t(rec) * REC;
    for (int i = 0; i < REC; ++i) o[i] = 0;
    build_record<F>(raw + uint64_t(x) * raw_stride, row_bytes, row0, s, o);
  });
}

Format format_for(gguf::Type t, bool gate_up) {
  if (t == gguf::Type::NVFP4) return Format::NVFP4;
  if (t == gguf::Type::Q5_K && gate_up) return Format::Q5_K;
  if (t == gguf::Type::Q8_0 && !gate_up) return Format::Q8_0;
  throw std::runtime_error(std::string("moe: unsupported expert type ") + gguf::type_name(t) + (gate_up ? " for gate_up" : " for down"));
}

uint64_t align(uint64_t x, uint64_t a = 256) { return (x + a - 1) / a * a; }

}  // namespace

struct Scratch {
  int64_t slab = 0, n_used = 0, n_embd = 0, n_ff = 0, rows = 0, max_tiles = 0;
  int32_t* tl = nullptr;
  int32_t* rop = nullptr;
  sycl::half* a = nullptr;  // sorted activations, width n_embd, act16 layout
  sycl::half* h = nullptr;  // SITU output, width n_ff
  float* y = nullptr;       // down's outputs [row][n_embd]
  uint64_t bytes = 0;
};

void rebase(sycl::queue&, Layer& L, uint8_t* new_base) {
  L.base = new_base;
  L.gate_up_scale = L.gate_up_scale_offset ? reinterpret_cast<const float*>(new_base + L.gate_up_scale_offset) : nullptr;
  L.down_scale = L.down_scale_offset ? reinterpret_cast<const float*>(new_base + L.down_scale_offset) : nullptr;
  L.slot_of = L.slot_of_offset ? reinterpret_cast<const int32_t*>(new_base + L.slot_of_offset) : nullptr;
}

Layer upload(sycl::queue& q, const ref::Weight& gate_up_raw, uint64_t gate_up_stride, const float* gate_up_scale,
             const ref::Weight& down_raw, uint64_t down_stride, const float* down_scale, int64_t n_expert, const std::vector<bool>* hot) {
  Layer L;
  L.n_expert = n_expert;
  std::vector<int32_t> hot_ids, slot(size_t(n_expert), -1);
  for (int64_t e = 0; e < n_expert; ++e)
    if (!hot || (*hot)[size_t(e)]) {
      slot[size_t(e)] = int32_t(hot_ids.size());
      hot_ids.push_back(int32_t(e));
    }
  const bool all_hot = int64_t(hot_ids.size()) == n_expert;
  const int64_t n_hot = int64_t(hot_ids.size());
  L.n_hot = n_hot;
  L.n_embd = gate_up_raw.k;
  L.n_ff = gate_up_raw.n / 2;
  if (down_raw.k != L.n_ff || down_raw.n != L.n_embd) throw std::runtime_error("moe: down's shape does not match gate_up's");
  if (L.n_embd % kStepK || L.n_ff % kStepK) throw std::runtime_error("moe: K must be a multiple of 64");
  L.gate_up = Table{format_for(gate_up_raw.type, true), gate_up_raw.k, gate_up_raw.n, 2, 0, 0};
  L.down = Table{format_for(down_raw.type, false), down_raw.k, down_raw.n, down_raw.type == gguf::Type::Q8_0 ? 1 : 2, 0, 0};
  auto expert_bytes = [](const Table& t) { return uint64_t(t.n / 8) * uint64_t(t.k / kStepK) * uint64_t(record_bytes(t.format)); };
  L.gate_up.expert_bytes = expert_bytes(L.gate_up);
  L.down.expert_bytes = expert_bytes(L.down);
  uint64_t off = 0;
  L.gate_up.offset = off;
  off = align(off + uint64_t(n_hot) * L.gate_up.expert_bytes);
  L.down.offset = off;
  off = align(off + uint64_t(n_hot) * L.down.expert_bytes);
  if (gate_up_scale) {
    L.gate_up_scale_offset = off;
    off = align(off + uint64_t(n_hot) * 4);
  }
  if (down_scale) {
    L.down_scale_offset = off;
    off = align(off + uint64_t(n_hot) * 4);
  }
  if (!all_hot) {
    L.slot_of_offset = off;
    off = align(off + uint64_t(n_expert) * 4);
  }
  L.bytes = off;
  uint8_t* base = sycl::malloc_device<uint8_t>(size_t(std::max<uint64_t>(L.bytes, 256)), q);
  if (!base) throw std::runtime_error("moe: out of device memory for a layer's experts");
  rebase(q, L, base);
  auto put_scales = [&](const float* sc, uint64_t at) {
    std::vector<float> v(size_t(n_hot), 0.0f);
    for (int64_t i = 0; i < n_hot; ++i) v[size_t(i)] = sc[hot_ids[size_t(i)]];
    q.memcpy(L.base + at, v.data(), v.size() * 4).wait();
  };
  if (gate_up_scale) put_scales(gate_up_scale, L.gate_up_scale_offset);
  if (down_scale) put_scales(down_scale, L.down_scale_offset);
  if (!all_hot) q.memcpy(L.base + L.slot_of_offset, slot.data(), slot.size() * 4).wait();

  // Staging: a few experts at a time through the card, repacked there.
  auto one = [&](const ref::Weight& raw, uint64_t stride, const Table& t, bool interleave) {
    constexpr int G = 8;
    uint8_t* stage = sycl::malloc_device<uint8_t>(size_t(stride) * G, q);
    if (!stage) throw std::runtime_error("moe: out of device memory for staging");
    for (int64_t e0 = 0; e0 < n_hot; e0 += G) {
      const int cnt = int(std::min<int64_t>(G, n_hot - e0));
      for (int i = 0; i < cnt; ++i) q.memcpy(stage + uint64_t(i) * stride, raw.data + uint64_t(hot_ids[size_t(e0 + i)]) * stride, size_t(stride));
      q.wait();
      uint8_t* dst = L.base + t.offset + uint64_t(e0) * t.expert_bytes;
      switch (t.format) {
        case Format::NVFP4:
          repack_group<Format::NVFP4>(q, stage, stride, raw.row_bytes, t.k, t.n, t.cg, interleave, dst, t.expert_bytes, cnt);
          break;
        case Format::Q5_K:
          repack_group<Format::Q5_K>(q, stage, stride, raw.row_bytes, t.k, t.n, t.cg, interleave, dst, t.expert_bytes, cnt);
          break;
        case Format::Q8_0:
          repack_group<Format::Q8_0>(q, stage, stride, raw.row_bytes, t.k, t.n, t.cg, interleave, dst, t.expert_bytes, cnt);
          break;
      }
      q.wait();
    }
    sycl::free(stage, q);
  };
  one(gate_up_raw, gate_up_stride, L.gate_up, true);
  one(down_raw, down_stride, L.down, false);
  q.wait();
  return L;
}

void free_layer(sycl::queue& q, Layer& L) {
  q.wait();
  if (L.base) sycl::free(L.base, q);
  L = Layer{};
}

int64_t scratch_slab() { return env_int("EKLOW_MOE_SLAB", kSlabTokens); }

Scratch* reserve(sycl::queue& q, int64_t max_tokens, int64_t n_used, int64_t n_embd, int64_t n_ff) {
  auto* s = new Scratch;
  s->slab = std::min<int64_t>(std::max<int64_t>(max_tokens, 1), scratch_slab());
  s->n_used = n_used;
  s->n_embd = n_embd;
  s->n_ff = n_ff;
  const int64_t P = s->slab * n_used;
  s->rows = P + 8 * std::min<int64_t>(kExperts * 2, P) + 128;  // segments padded to 8 rows, and the last tile's slack
  s->max_tiles = P / kGemvRows + kExperts * 2 + 1;
  auto dev = [&](size_t bytes) {
    void* p = sycl::malloc_device(bytes, q);
    if (!p) throw std::runtime_error("moe: out of device memory for the scratch");
    s->bytes += bytes;
    return p;
  };
  s->tl = static_cast<int32_t*>(dev(size_t(4 + 4 * s->max_tiles) * 4));
  s->rop = static_cast<int32_t*>(dev(size_t(P) * 4));
  s->a = static_cast<sycl::half*>(dev(size_t(s->rows * n_embd) * 2));
  s->h = static_cast<sycl::half*>(dev(size_t(s->rows * n_ff) * 2));
  s->y = static_cast<float*>(dev(size_t(s->rows * n_embd) * 4));
  return s;
}

void free_scratch(sycl::queue& q, Scratch* s) {
  if (!s) return;
  q.wait();
  sycl::free(s->tl, q);
  sycl::free(s->rop, q);
  sycl::free(s->a, q);
  sycl::free(s->h, q);
  sycl::free(s->y, q);
  delete s;
}

uint64_t scratch_bytes(const Scratch& s) { return s.bytes; }

namespace {

template <Format F, int CG, int S, bool SA>
void gemv_gu_s(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles, float beta, float lb) {
  const Table& t = L.gate_up;
  const int64_t units = t.n / 8 / CG;
  Gemv<F, true, S, CG, SA> k{s.a, L.base + t.offset, t.expert_bytes, s.tl, L.gate_up_scale, s.h, nullptr, int32_t(t.k / kStepK), int32_t(t.n), int32_t(L.n_ff), beta, lb};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * S, size_t(units)), sycl::range<2>(S, 1)), k);
}

template <Format F, int CG, int S, bool SA>
void gemv_down_s(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles) {
  const Table& t = L.down;
  const int64_t units = t.n / 8 / CG;
  Gemv<F, false, S, CG, SA> k{s.h, L.base + t.offset, t.expert_bytes, s.tl, L.down_scale, nullptr, s.y, int32_t(t.k / kStepK), int32_t(t.n), 0, 0, 0};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * S, size_t(units)), sycl::range<2>(S, 1)), k);
}

// K splits of the GEMVs (threads a work-group): gate_up's K is 36 steps, down's 18 (Q8_0's records are bigger: 3 threads and
// one column group each). NVFP4 runs scale-after (Gemv::compute); the others dequantize with their scales first.
template <Format F, int CG>
void gemv_gu(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles, float beta, float lb) {
  gemv_gu_s<F, CG, 4, F == Format::NVFP4>(q, L, s, max_tiles, beta, lb);
}

template <Format F, int CG>
void gemv_down(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles) {
  gemv_down_s<F, CG, 3, F == Format::NVFP4>(q, L, s, max_tiles);
}

template <Format F, int CG, int WM, int NA, int D>
void gemm_gu(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles, float beta, float lb) {
  const Table& t = L.gate_up;
  constexpr int WN = 4;
  Gemm<F, true, WM, WN, CG, NA, D> k{s.a, L.base + t.offset, t.expert_bytes, s.tl, L.gate_up_scale, s.h, nullptr, int32_t(t.k / kStepK), int32_t(t.n), int32_t(L.n_ff), beta, lb};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * WM, size_t(t.n / 8 / 4 / WN) * WN), sycl::range<2>(WM, WN)), k);
}

template <Format F, int CG, int WM, int NA, int D>
void gemm_down(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles) {
  const Table& t = L.down;
  constexpr int WN = 4;
  Gemm<F, false, WM, WN, CG, NA, D> k{s.h, L.base + t.offset, t.expert_bytes, s.tl, L.down_scale, nullptr, s.y, int32_t(t.k / kStepK), int32_t(t.n), 0, 0, 0};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * WM, size_t(t.n / 8 / 4 / WN) * WN), sycl::range<2>(WM, WN)), k);
}

}  // namespace

namespace {

// EKLOW_MOE_PROFILE=1: every stage of ffn waits and is timed; profile_print() reports the sums.
double g_prof[6] = {};
const char* const kProfNames[6] = {"group", "gather", "gate_up", "down", "combine", "calls"};

template <int WM, int NA, int D>
void gemms(sycl::queue& q, const Layer& L, Scratch& s, int64_t tiles, float beta, float lb, bool prof) {
  auto stage = [&](int i, auto&& f) {
    if (!prof) return f();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    f();
    q.wait();
    g_prof[i] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  stage(2, [&] {
    if (L.gate_up.format == Format::NVFP4) gemm_gu<Format::NVFP4, 2, WM, NA, D>(q, L, s, tiles, beta, lb);
    else gemm_gu<Format::Q5_K, 2, WM, NA, D>(q, L, s, tiles, beta, lb);
  });
  stage(3, [&] {
    if (L.down.format == Format::NVFP4) gemm_down<Format::NVFP4, 2, WM, NA, D>(q, L, s, tiles);
    else gemm_down<Format::Q8_0, 1, WM, NA, D>(q, L, s, tiles);
  });
}

}  // namespace

void profile_print(bool print) {
  for (int i = 0; print && i < 6; ++i) std::printf("    %-8s %9.3f ms\n", kProfNames[i], i == 5 ? g_prof[i] : g_prof[i] / std::max(1.0, g_prof[5]));
  for (double& v : g_prof) v = 0;
}

void ffn(sycl::queue& q, const Layer& L, const float* x, int64_t n_tok, const int32_t* ids, const float* w, int64_t n_used,
         float situ_beta, float situ_linear_beta, Scratch& s, float* out, const float* cold_rows) {
  if (n_used != s.n_used || L.n_embd != s.n_embd || L.n_ff != s.n_ff) throw std::runtime_error("moe: scratch shape differs from the layer's");
  // Tokens above which the GEMM beats the GEMVs (measured on layers 6 and 20, DECISIONS 2026-11-24, "Inkling M2: the experts")
  const int64_t gemm_min = env_int("EKLOW_MOE_GEMM_MIN", L.gate_up.format == Format::NVFP4 ? 448 : 256);
  const int wm = kGemmWM;
  const bool prof = env_int("EKLOW_MOE_PROFILE", 0) != 0;
  auto stage = [&](int i, auto&& f) {
    if (!prof) return f();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    f();
    q.wait();
    g_prof[i] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  };
  for (int64_t t0 = 0; t0 < n_tok; t0 += s.slab) {
    const int64_t tc = std::min(s.slab, n_tok - t0), P = tc * n_used;
    const bool use_gemm = tc > gemm_min;
    const int T = use_gemm ? 16 * wm : kGemvRows;
    const int64_t tiles = std::min<int64_t>(P, (P + T - 1) / T + L.n_expert);
    stage(0, [&] { group(q, ids + t0 * n_used, L.slot_of, P, L.n_hot, T, s.tl, s.rop); });
    stage(1, [&] {
      q.parallel_for(sycl::range<2>(size_t(P), size_t(L.n_embd / 64)), Gather{x + t0 * L.n_embd, s.rop, s.a, int32_t(n_used), int32_t(L.n_embd)});
    });
    const Format fg = L.gate_up.format, fd = L.down.format;
    if (!use_gemm) {
      stage(2, [&] {
        if (fg == Format::NVFP4) gemv_gu<Format::NVFP4, 2>(q, L, s, tiles, situ_beta, situ_linear_beta);
        else gemv_gu<Format::Q5_K, 2>(q, L, s, tiles, situ_beta, situ_linear_beta);
      });
      stage(3, [&] {
        if (fd == Format::NVFP4) gemv_down<Format::NVFP4, 2>(q, L, s, tiles);
        else gemv_down<Format::Q8_0, 1>(q, L, s, tiles);
      });
    } else {
      gemms<kGemmWM, 1, 1>(q, L, s, tiles, situ_beta, situ_linear_beta, prof);
    }
    stage(4, [&] {
      q.parallel_for(sycl::range<2>(size_t(tc), size_t(L.n_embd / 128)), Combine{s.y, w + t0 * n_used, s.rop, out + t0 * L.n_embd, cold_rows ? cold_rows + t0 * L.n_embd : nullptr, int32_t(n_used), int32_t(L.n_embd)});
    });
  }
  g_prof[5] += 1;
}

}  // namespace eklow::moe
