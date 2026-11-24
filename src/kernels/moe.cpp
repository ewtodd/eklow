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
// NVFP4 dequantizes without a table: a code's magnitude bits placed at fp16 bits 9..11 and the sign at 15, read as
// fp16, are the e2m1 value / 2^14 (the denormal 0.5 included), so one exact multiply by 2^7 and one by the group's scale
// (ue4m3 * 2^7, at most 57,344) give the value exactly in fp16.
#include "kernels/moe.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace eklow::moe {

namespace esimd = sycl::ext::intel::esimd;
namespace xmxd = sycl::ext::intel::esimd::xmx;

namespace {

constexpr int kStepK = 64;        // K a record covers
constexpr int kSlabTokens = 512;  // ffn runs longer steps in slabs of this many tokens (scratch size)
constexpr int kExperts = 128;     // the group kernel's work-group

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

// Piece H (8 columns x 16 K, the dpas B operand in VNNI order: element (k, n) at (k / 2) * 16 + n * 2 + k % 2) of a record.
template <Format F>
struct Dec;

template <>
struct Dec<Format::NVFP4> {
  template <int H>
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> piece(esimd::simd<uint8_t, 288>& r) {
    // The scales: ue4m3 (e, m) = (8 [e > 0] + m) 2^(max(e, 1) - 10), times 2^7.
    esimd::simd<uint8_t, 8> xb = r.template select<8, 1>(8 * H);
    esimd::simd<uint32_t, 8> x = xb;
    esimd::simd<uint32_t, 8> e = (x >> 3) & 15, m = x & 7, ex = e, mant = m;
    ex.merge(esimd::simd<uint32_t, 8>(1), e == 0);
    mant.merge(m + 8, e != 0);
    esimd::simd<uint32_t, 8> pb = (ex + 117) << 23;
    esimd::simd<float, 8> p2 = pb.template bit_cast_view<float>();
    esimd::simd<float, 8> mf = mant;
    esimd::simd<float, 8> scf = mf * p2 * 128.0f;
    esimd::simd<sycl::half, 8> sc = scf;
    esimd::simd<uint8_t, 64> cb = r.template select<64, 1>(32 + 64 * H);
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
};

template <>
struct Dec<Format::Q5_K> {
  template <int H>
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> piece(esimd::simd<uint8_t, 384>& r) {
    esimd::simd<uint8_t, 16> dsb = r.template select<16, 1>((H / 2) * 32), dmb = r.template select<16, 1>((H / 2) * 32 + 16);
    esimd::simd<sycl::half, 8> ds = dsb.template bit_cast_view<sycl::half>(), dm = dmb.template bit_cast_view<sycl::half>();
    esimd::simd<uint8_t, 64> cb = r.template select<64, 1>(64 + 64 * H);
    esimd::simd<uint32_t, 16> D = cb.template bit_cast_view<uint32_t>();
    esimd::simd<uint32_t, 64> W = nibbles_to_pairs(D);
    esimd::simd<uint8_t, 16> hb = r.template select<16, 1>(64 + 256 + 16 * H);
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
};

template <>
struct Dec<Format::Q8_0> {
  template <int H>
  SYCL_ESIMD_FUNCTION static esimd::simd<sycl::half, 128> piece(esimd::simd<uint8_t, 544>& r) {
    esimd::simd<uint8_t, 16> sb = r.template select<16, 1>((H / 2) * 16);
    esimd::simd<sycl::half, 8> sc = sb.template bit_cast_view<sycl::half>();
    esimd::simd<uint8_t, 128> qb = r.template select<128, 1>(32 + 128 * H);
    esimd::simd<int8_t, 128> q8 = qb.template bit_cast_view<int8_t>();
    esimd::simd<sycl::half, 128> w = q8;
    return w * vnni_cols(sc);
  }
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
void group(sycl::queue& q, const int32_t* ids, int64_t P, int64_t E, int T, int32_t* tl, int32_t* rop) {
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int32_t, 1> cnt(sycl::range<1>(256), h), start(sycl::range<1>(256), h);
    const int32_t p_n = int32_t(P), e_n = int32_t(E);
    h.parallel_for(sycl::nd_range<1>(kExperts, kExperts), [=](sycl::nd_item<1> it) {
      const int lid = int(it.get_local_id(0));
      for (int e = lid; e < e_n; e += kExperts) {
        int c = 0;
        for (int p = 0; p < p_n; ++p) c += ids[p] == e;
        cnt[e] = c;
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
      for (int e = lid; e < e_n; e += kExperts) {
        int pos = start[e];
        for (int p = 0; p < p_n; ++p)
          if (ids[p] == e) rop[p] = pos++;
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
  int32_t n_used, n;
  void operator()(sycl::id<2> id) const SYCL_ESIMD_KERNEL {
    const int t = int(id[0]), j = int(id[1]);
    esimd::simd<float, 128> acc = 0;
    for (int i = 0; i < n_used; ++i) {
      const float wt = w[t * n_used + i];
      const int row = rop[t * n_used + i];
      acc += wt * esimd::block_load<float, 128>(y + int64_t(row) * n + j * 128);
    }
    esimd::block_store<float, 128>(out + int64_t(t) * n + j * 128, acc);
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// Decode: a grouped GEMV over row tiles of 8. A work-group is one tile and one column unit; its S threads split K and
// reduce through shared local memory. GU: the unit is a gate group and its up group, the epilogue SITU into fp16 rows of
// `out_h` (width n_ff); down: fp32 rows of `out_y`. Each thread streams its slice's records double-buffered in registers
// (the next step's loads go out before this step's dpas); the row count of the tile picks the dpas repeat count.
template <Format F, bool GU, int S, int CG>
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
    const sycl::half* ap = a + int64_t(row0 / 8) * (4 * int64_t(kst)) * 128;
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
      esimd::simd<sycl::half, 16 * RC> A[4];
#pragma unroll
      for (int h = 0; h < 4; ++h) A[h] = esimd::block_load<sycl::half, 16 * RC>(ap + (4 * int64_t(s) + h) * 128);
#pragma unroll
      for (int c = 0; c < CG; ++c) {
        acc[c][0] = xmxd::dpas<8, RC, float>(acc[c][0], Dec<F>::template piece<0>(buf[c]), A[0]);
        acc[c][1] = xmxd::dpas<8, RC, float>(acc[c][1], Dec<F>::template piece<1>(buf[c]), A[1]);
        acc[c][2] = xmxd::dpas<8, RC, float>(acc[c][2], Dec<F>::template piece<2>(buf[c]), A[2]);
        acc[c][3] = xmxd::dpas<8, RC, float>(acc[c][3], Dec<F>::template piece<3>(buf[c]), A[3]);
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
    else if (valid <= 4) run<4>(u, sl, e, row0, valid);
    else run<8>(u, sl, e, row0, valid);
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// Prefill: a grouped fp16 GEMM. A work-group of WM x WN threads takes one tile of 16 WM rows of one expert and 32 WN
// output columns (4 WN column groups); a thread computes 16 x 32. Per K step the threads dequantize the group's records
// to fp16 B operands and copy the activation pieces into shared local memory (two buffers: the next step's global loads
// go out before this step's dpas), then every thread reads its operands from there.
template <Format F, bool GU, int WM, int WN, int CG>
struct Gemm {
  static constexpr int NT = WM * WN, BT = WN * 4, AT = WM * 2, REC = Fd<F>::REC;
  static constexpr int kBPieces = BT * 4, kAPieces = AT * 4;
  static constexpr int BP = (BT + NT - 1) / NT, AP = (kAPieces + NT - 1) / NT;
  static constexpr uint32_t kBBuf = kBPieces * 256, kBuf = kBBuf + kAPieces * 256;
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
    esimd::simd<uint8_t, REC> R[BP];
    esimd::simd<sycl::half, 128> Araw[AP];
    auto load = [&](int b) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < BP; ++i) {
        const int g = tn * BT + tid + i * NT;
        if (tid + i * NT >= BT) break;
        R[i] = esimd::block_load<uint8_t, REC>(eb + ((int64_t(g / CG) * kst + b) * CG + g % CG) * REC);
      }
#pragma unroll
      for (int i = 0; i < AP; ++i) {
        const int p = tid + i * NT;
        if (p >= kAPieces) break;
        Araw[i] = esimd::block_load<sycl::half, 128>(a + (int64_t(row0 / 8 + p / 4) * kh + 4 * b + p % 4) * 128);
      }
    };
    auto store = [&](uint32_t buf) SYCL_ESIMD_FUNCTION {
#pragma unroll
      for (int i = 0; i < BP; ++i) {
        const int g = tid + i * NT;
        if (g >= BT) break;
        esimd::slm_block_store<sycl::half, 128>(buf + (g * 4 + 0) * 256, Dec<F>::template piece<0>(R[i]));
        esimd::slm_block_store<sycl::half, 128>(buf + (g * 4 + 1) * 256, Dec<F>::template piece<1>(R[i]));
        esimd::slm_block_store<sycl::half, 128>(buf + (g * 4 + 2) * 256, Dec<F>::template piece<2>(R[i]));
        esimd::slm_block_store<sycl::half, 128>(buf + (g * 4 + 3) * 256, Dec<F>::template piece<3>(R[i]));
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
    if (kbl > 0) {
      load(0);
      store(0);
    }
    esimd::barrier();
    for (int b = 0; b < kbl; ++b) {
      const uint32_t cur = (b & 1) * kBuf, nxt = kBuf - cur;
      if (b + 1 < kst) load(b + 1);
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
            if ((wi * 2 + r) * 8 < valid) acc[r][c] = xmxd::dpas<8, 8, float>(acc[r][c], B, A[r]);
        }
      }
      if (b + 1 < kst) store(nxt);
      esimd::barrier();
    }
    if (!live) return;
    const float sc = scale ? scale[e] : 1.0f;
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
            const esimd::simd<float, 8> g = acc[r][c].template select<8, 1>(i * 8) * sc,
                                        up = acc[r][c + 1].template select<8, 1>(i * 8) * sc;
            esimd::simd<sycl::half, 8> h = situ8(g, up, beta, lb);
            const int pu = (tn * BT + wj * 4 + c) / 2;
            esimd::block_store<sycl::half, 8>(out_h + act16_index(row, pu * 8, n_ff / 16), h);
          }
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c)
            esimd::block_store<float, 8>(out_y + int64_t(row) * n_out + (tn * BT + wj * 4 + c) * 8,
                                         acc[r][c].template select<8, 1>(i * 8) * sc);
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
}

Layer upload(sycl::queue& q, const ref::Weight& gate_up_raw, uint64_t gate_up_stride, const float* gate_up_scale,
             const ref::Weight& down_raw, uint64_t down_stride, const float* down_scale, int64_t n_expert) {
  Layer L;
  L.n_expert = n_expert;
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
  off = align(off + uint64_t(n_expert) * L.gate_up.expert_bytes);
  L.down.offset = off;
  off = align(off + uint64_t(n_expert) * L.down.expert_bytes);
  if (gate_up_scale) {
    L.gate_up_scale_offset = off;
    off = align(off + uint64_t(n_expert) * 4);
  }
  if (down_scale) {
    L.down_scale_offset = off;
    off = align(off + uint64_t(n_expert) * 4);
  }
  L.bytes = off;
  uint8_t* base = sycl::malloc_device<uint8_t>(size_t(L.bytes), q);
  if (!base) throw std::runtime_error("moe: out of device memory for a layer's experts");
  rebase(q, L, base);
  if (gate_up_scale) q.memcpy(L.base + L.gate_up_scale_offset, gate_up_scale, size_t(n_expert) * 4);
  if (down_scale) q.memcpy(L.base + L.down_scale_offset, down_scale, size_t(n_expert) * 4);

  // Staging: a few experts at a time through the card, repacked there.
  auto one = [&](const ref::Weight& raw, uint64_t stride, const Table& t, bool interleave) {
    constexpr int G = 8;
    uint8_t* stage = sycl::malloc_device<uint8_t>(size_t(stride) * G, q);
    if (!stage) throw std::runtime_error("moe: out of device memory for staging");
    for (int64_t e0 = 0; e0 < n_expert; e0 += G) {
      const int cnt = int(std::min<int64_t>(G, n_expert - e0));
      q.memcpy(stage, raw.data + uint64_t(e0) * stride, size_t(stride) * size_t(cnt)).wait();
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
  s->max_tiles = P / 8 + kExperts * 2 + 1;
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

template <Format F, int CG>
void gemv_gu(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles, float beta, float lb) {
  const Table& t = L.gate_up;
  constexpr int S = 4;
  const int64_t units = t.n / 8 / CG;
  Gemv<F, true, S, CG> k{s.a, L.base + t.offset, t.expert_bytes, s.tl, L.gate_up_scale, s.h, nullptr, int32_t(t.k / kStepK), int32_t(t.n), int32_t(L.n_ff), beta, lb};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * S, size_t(units)), sycl::range<2>(S, 1)), k);
}

template <Format F, int CG>
void gemv_down(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles) {
  const Table& t = L.down;
  constexpr int S = CG == 2 ? 2 : 3;
  const int64_t units = t.n / 8 / CG;
  Gemv<F, false, S, CG> k{s.h, L.base + t.offset, t.expert_bytes, s.tl, L.down_scale, nullptr, s.y, int32_t(t.k / kStepK), int32_t(t.n), 0, 0, 0};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * S, size_t(units)), sycl::range<2>(S, 1)), k);
}

template <Format F, int CG, int WM>
void gemm_gu(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles, float beta, float lb) {
  const Table& t = L.gate_up;
  constexpr int WN = 4;
  Gemm<F, true, WM, WN, CG> k{s.a, L.base + t.offset, t.expert_bytes, s.tl, L.gate_up_scale, s.h, nullptr, int32_t(t.k / kStepK), int32_t(t.n), int32_t(L.n_ff), beta, lb};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * WM, size_t(t.n / 8 / 4 / WN) * WN), sycl::range<2>(WM, WN)), k);
}

template <Format F, int CG, int WM>
void gemm_down(sycl::queue& q, const Layer& L, Scratch& s, int64_t max_tiles) {
  const Table& t = L.down;
  constexpr int WN = 4;
  Gemm<F, false, WM, WN, CG> k{s.h, L.base + t.offset, t.expert_bytes, s.tl, L.down_scale, nullptr, s.y, int32_t(t.k / kStepK), int32_t(t.n), 0, 0, 0};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(max_tiles) * WM, size_t(t.n / 8 / 4 / WN) * WN), sycl::range<2>(WM, WN)), k);
}

}  // namespace

void ffn(sycl::queue& q, const Layer& L, const float* x, int64_t n_tok, const int32_t* ids, const float* w, int64_t n_used,
         float situ_beta, float situ_linear_beta, Scratch& s, float* out) {
  if (n_used != s.n_used || L.n_embd != s.n_embd || L.n_ff != s.n_ff) throw std::runtime_error("moe: scratch shape differs from the layer's");
  const int64_t gemm_min = env_int("EKLOW_MOE_GEMM_MIN", 24);  // tokens above which the GEMM runs
  for (int64_t t0 = 0; t0 < n_tok; t0 += s.slab) {
    const int64_t tc = std::min(s.slab, n_tok - t0), P = tc * n_used;
    const bool use_gemm = tc > gemm_min;
    const int T = use_gemm ? 64 : 8;
    const int64_t tiles = std::min<int64_t>(P, (P + T - 1) / T + L.n_expert);
    group(q, ids + t0 * n_used, P, L.n_expert, T, s.tl, s.rop);
    q.parallel_for(sycl::range<2>(size_t(P), size_t(L.n_embd / 64)), Gather{x + t0 * L.n_embd, s.rop, s.a, int32_t(n_used), int32_t(L.n_embd)});
    const Format fg = L.gate_up.format, fd = L.down.format;
    if (!use_gemm) {
      if (fg == Format::NVFP4) gemv_gu<Format::NVFP4, 2>(q, L, s, tiles, situ_beta, situ_linear_beta);
      else gemv_gu<Format::Q5_K, 2>(q, L, s, tiles, situ_beta, situ_linear_beta);
      if (fd == Format::NVFP4) gemv_down<Format::NVFP4, 2>(q, L, s, tiles);
      else gemv_down<Format::Q8_0, 1>(q, L, s, tiles);
    } else {
      if (fg == Format::NVFP4) gemm_gu<Format::NVFP4, 2, 4>(q, L, s, tiles, situ_beta, situ_linear_beta);
      else gemm_gu<Format::Q5_K, 2, 4>(q, L, s, tiles, situ_beta, situ_linear_beta);
      if (fd == Format::NVFP4) gemm_down<Format::NVFP4, 2, 4>(q, L, s, tiles);
      else gemm_down<Format::Q8_0, 1, 4>(q, L, s, tiles);
    }
    q.parallel_for(sycl::range<2>(size_t(tc), size_t(L.n_embd / 128)), Combine{s.y, w + t0 * n_used, s.rop, out + t0 * L.n_embd, int32_t(n_used), int32_t(L.n_embd)});
  }
}

}  // namespace eklow::moe
