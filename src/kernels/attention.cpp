#include "kernels/attention.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace eklow::attn {

namespace esimd = sycl::ext::intel::esimd;
namespace xmxd = sycl::ext::intel::esimd::xmx;

// Tiles per prefill work-group: with 16 loaders, 64 threads, two of which fill an Xe-core's 128
// thread slots; the K/V block and the scores in shared local memory stay under 64 KB, of which an
// Xe-core holds two (it allocates power-of-two sizes from 128 KB; measured 2026-10-24).
TileList::TileList(int64_t head_dim) : per_group(head_dim >= 512 ? 6 : 12) {}

void TileList::add_chunk(int32_t base, int64_t n_tok, int64_t last_pos, int64_t group, int64_t n_head_kv, int64_t window) {
  const int64_t rows = n_tok * group;
  for (int64_t g = 0; g < n_head_kv; ++g) {
    const int64_t first = size();
    for (int64_t r0 = 0; r0 < rows; r0 += 8) desc.insert(desc.end(), {base, int32_t(rows), int32_t(r0), int32_t(g)});
    for (int64_t t = first; t < size(); t += per_group) groups.insert(groups.end(), {int32_t(t), int32_t(std::min(per_group, size() - t))});
  }
  // A tile's rows span at most 8 consecutive positions.
  max_keys = std::max(max_keys, window > 0 ? std::min(window + 7, last_pos + 1) : last_pos + 1);
}

std::vector<int32_t> TileList::packed() const {
  std::vector<int32_t> p(desc);
  p.insert(p.end(), groups.begin(), groups.end());
  return p;
}

namespace {

constexpr int kRows = 8, kKeys = 16, kSlice = 64;
// The running maximum starts finite so that a row with no key yet in a block gets weight
// exp(-inf - kNegBig) = 0 and correction exp(0) = 1 rather than NaN.
constexpr float kNegBig = -1e30f;

template <class T>
T* grow(sycl::queue& q, T*& p, size_t& have, size_t need) {
  if (need > have) {
    if (p) {
      q.wait();
      sycl::free(p, q);
    }
    have = std::max(need, have * 2);
    p = sycl::malloc_device<T>(have, q);
    if (!p) throw std::runtime_error("attention: out of device memory for scratch");
  }
  return p;
}

inline int32_t part_stride(int32_t d) { return kRows * d + 2 * kRows; }

// A K tile (8 positions x 16 dims) as an fp16 B operand: `at` is its element offset (fp16
// elements, or the int8 bytes in the same places), `sat` the offset of its 8 scales.
template <bool Q8>
inline esimd::simd<sycl::half, 128> k_tile(const void* base, const sycl::half* scales, int32_t at, int32_t sat) SYCL_ESIMD_FUNCTION {
  if constexpr (!Q8) {
    return esimd::block_load<sycl::half, 128>(static_cast<const sycl::half*>(base) + at);
  } else {
    const esimd::simd<int8_t, 128> q = esimd::block_load<int8_t, 128>(static_cast<const int8_t*>(base) + at);
    const esimd::simd<sycl::half, 8> sc = esimd::block_load<sycl::half, 8>(scales + sat);
    // Element (dim d, position n) is at (d/2)*16 + n*2 + d%2: the scale of n, twice, eight times.
    esimd::simd<sycl::half, 16> pat;
    pat.select<8, 2>(0) = sc;
    pat.select<8, 2>(1) = sc;
    return esimd::simd<sycl::half, 128>(q) * pat.replicate<8>();
  }
}

// A V tile (16 positions x 8 dims); `sat` the offset of its 16 positions' scales.
template <bool Q8>
inline esimd::simd<sycl::half, 128> v_tile(const void* base, const sycl::half* scales, int32_t at, int32_t sat) SYCL_ESIMD_FUNCTION {
  if constexpr (!Q8) {
    return esimd::block_load<sycl::half, 128>(static_cast<const sycl::half*>(base) + at);
  } else {
    const esimd::simd<int8_t, 128> q = esimd::block_load<int8_t, 128>(static_cast<const int8_t*>(base) + at);
    const esimd::simd<sycl::half, 16> sc = esimd::block_load<sycl::half, 16>(scales + sat);
    // Element (position k, dim n) is at (k/2)*16 + n*2 + k%2: row r of 16 alternates the scales
    // of positions 2r and 2r+1.
    esimd::simd<sycl::half, 128> pat;
#pragma unroll
    for (int r = 0; r < 8; ++r) pat.select<16, 1>(16 * r) = sc.template replicate_vs_w_hs<8, 0, 2, 1>(2 * r);
    return esimd::simd<sycl::half, 128>(q) * pat;
  }
}

// Bytes of one cache element.
template <bool Q8>
constexpr int kElem = Q8 ? 1 : 2;

// Index arithmetic below is in 32 bits with the query group size a template parameter: DG2 has no
// integer division, and 64-bit divisions by runtime values were library calls that cost more than
// a decode step's whole attention.
template <int D, int G, bool Q8>
struct Flash {
  static constexpr int P = D / kSlice;     // threads per tile
  static constexpr int RPT = kRows / P;    // rows each thread runs the softmax for
  static constexpr int DC = kSlice / 16;   // Q.K^T steps per slice
  static constexpr int DG = kSlice / 8;    // P.V output tiles per slice
  // Shared local memory: partial scores [P][2 key halves][8 rows][8 keys] fp32, the probability
  // tile [8 rows][16 keys] fp16, the correction per row, and at the end m and l per row.
  static constexpr uint32_t kScores = 0, kProbs = P * 512, kCorr = kProbs + 256, kM = kCorr + 32, kL = kM + 32,
                            kSlm = kL + 32;

  const sycl::half* qt;
  const int32_t* tiles;
  const int32_t *slot_of, *pos_of;
  const void *kc, *vc;
  const sycl::half *ks, *vs;
  const int32_t* table;
  int32_t pages, window, n_head_kv, n_head, splits;
  float* out;
  float* part;

  // The physical page (this K/V head's part of it) holding position `key`, as a page index.
  int32_t head_page(int32_t slot, int32_t kvh, int32_t key) const SYCL_ESIMD_FUNCTION {
    const int32_t logical = key / int32_t(kPage);
    const int32_t phys = table ? table[slot * pages + logical] : slot * pages + logical % pages;
    return phys * n_head_kv + kvh;
  }

  // Thread j's tiles of block b, as element offsets: K at ka + dc * 128 (keys 0-7) and
  // + (D/16 + dc) * 128 (keys 8-15) with scales at ksa + dc * 8 and + (D/16 + dc) * 8; V at
  // va + g * 128 with its scales at vsa + (g / 2) * 16.
  struct Offs {
    int32_t ka, va, ksa, vsa;
  };
  Offs tiles_of(int32_t b, int32_t slot, int32_t kvh, int j) const SYCL_ESIMD_FUNCTION {
    const int32_t key0 = b * kKeys, in_page = key0 % int32_t(kPage), hp = head_page(slot, kvh, key0);
    const int32_t at = hp * int32_t(kPage * D), sat = hp * int32_t(kPage * D / 16);
    return {at + ((in_page / 8) * (D / 16) + j * DC) * 128, at + ((in_page / 16) * (D / 8) + j * DG) * 128,
            sat + ((in_page / 8) * (D / 16) + j * DC) * 8, sat + ((in_page / 16) * (D / 16) + j * DG / 2) * 16};
  }

  // Each tile is consumed by a dpas right after its load, so loads straight from VRAM leave the
  // thread waiting on every one: the next block's tiles are prefetched into L1 a block ahead.
  void prefetch(int32_t b, int32_t slot, int32_t kvh, int j) const SYCL_ESIMD_FUNCTION {
    constexpr auto cached = esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>,
                                              esimd::cache_hint_L2<esimd::cache_hint::cached>};
    constexpr int W = 128 * kElem<Q8> / 4;  // dwords of a tile
    const Offs o = tiles_of(b, slot, kvh, j);
    const auto* k8 = static_cast<const uint8_t*>(kc);
    const auto* v8 = static_cast<const uint8_t*>(vc);
#pragma unroll
    for (int dc = 0; dc < DC; ++dc) {
      esimd::prefetch<uint32_t, W>(reinterpret_cast<const uint32_t*>(k8 + (o.ka + dc * 128) * kElem<Q8>), cached);
      esimd::prefetch<uint32_t, W>(reinterpret_cast<const uint32_t*>(k8 + (o.ka + (D / 16 + dc) * 128) * kElem<Q8>), cached);
    }
#pragma unroll
    for (int g = 0; g < DG; ++g) esimd::prefetch<uint32_t, W>(reinterpret_cast<const uint32_t*>(v8 + (o.va + g * 128) * kElem<Q8>), cached);
  }

  void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlm>();
    const int j = int(it.get_local_id(1));
    const int32_t tile = int32_t(it.get_group(1)), s = int32_t(it.get_global_id(0));
    const int32_t base = tiles[tile * 4], rows = tiles[tile * 4 + 1], r0 = tiles[tile * 4 + 2], kvh = tiles[tile * 4 + 3];
    const int32_t t_first = base + r0 / G, t_last = base + (std::min(r0 + kRows, rows) - 1) / G;
    const int32_t slot = slot_of[t_first];
    const int32_t lo = window > 0 ? std::max(0, pos_of[t_first] - window + 1) : 0;
    const int32_t blo = lo / kKeys, nblk = pos_of[t_last] / kKeys - blo + 1;
    const int32_t per = (nblk + splits - 1) / splits;
    const int32_t b0 = blo + s * per, b1 = std::min(blo + nblk, b0 + per);

    // This thread's softmax rows: each one's last visible key, first visible key minus one, and
    // running max and sum.
    int32_t hi[RPT], wlo[RPT];
    float m[RPT], l[RPT];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
      const int32_t r = r0 + j * RPT + i;
      hi[i] = r < rows ? pos_of[base + r / G] : -1;
      wlo[i] = window > 0 ? hi[i] - window : -1;
      m[i] = kNegBig;
      l[i] = 0;
    }
    esimd::simd<float, 64> O[DG];
#pragma unroll
    for (int g = 0; g < DG; ++g) O[g] = 0;
    const sycl::half* qs = qt + tile * (kRows * D) + j * DC * 128;
    if (b0 < b1) prefetch(b0, slot, kvh, j);
    for (int32_t b = b0; b < b1; ++b) {
      const int32_t key0 = b * kKeys;
      if (b + 1 < b1) prefetch(b + 1, slot, kvh, j);
      const Offs o = tiles_of(b, slot, kvh, j);
      esimd::simd<float, 64> S0 = 0, S1 = 0;
#pragma unroll
      for (int dc = 0; dc < DC; ++dc) {
        esimd::simd<sycl::half, 128> A = esimd::block_load<sycl::half, 128>(qs + dc * 128);
        esimd::simd<sycl::half, 128> B0 = k_tile<Q8>(kc, ks, o.ka + dc * 128, o.ksa + dc * 8);
        esimd::simd<sycl::half, 128> B1 = k_tile<Q8>(kc, ks, o.ka + (D / 16 + dc) * 128, o.ksa + (D / 16 + dc) * 8);
        S0 = xmxd::dpas<8, 8, float>(S0, B0, A);
        S1 = xmxd::dpas<8, 8, float>(S1, B1, A);
      }
      esimd::slm_block_store<float, 64>(kScores + j * 512, S0);
      esimd::slm_block_store<float, 64>(kScores + j * 512 + 256, S1);
      esimd::barrier();
      {
        esimd::simd<float, 8 * RPT> s0 = 0, s1 = 0;
#pragma unroll
        for (int t = 0; t < P; ++t) {
          s0 += esimd::slm_block_load<float, 8 * RPT>(kScores + t * 512 + j * RPT * 32);
          s1 += esimd::slm_block_load<float, 8 * RPT>(kScores + t * 512 + 256 + j * RPT * 32);
        }
        const esimd::simd<int, 16> key(key0, 1);
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
          esimd::simd<float, 16> sc;
          sc.select<8, 1>(0) = s0.template select<8, 1>(i * 8);
          sc.select<8, 1>(8) = s1.template select<8, 1>(i * 8);
          sc.merge(sc, esimd::simd<float, 16>(-std::numeric_limits<float>::infinity()), (key <= hi[i]) & (key > wlo[i]));
          const float mx = std::max(m[i], esimd::hmax<float>(sc));
          const esimd::simd<float, 16> p = esimd::exp(sc - mx);
          const float corr = esimd::exp(m[i] - mx);
          l[i] = l[i] * corr + esimd::reduce<float>(p, std::plus<>());
          m[i] = mx;
          esimd::slm_block_store<sycl::half, 16>(kProbs + (j * RPT + i) * 32, esimd::simd<sycl::half, 16>(p));
          esimd::slm_scalar_store<float>(kCorr + (j * RPT + i) * 4, corr);
        }
      }
      esimd::barrier();
      const esimd::simd<sycl::half, 128> Pa = esimd::slm_block_load<sycl::half, 128>(kProbs);
      const esimd::simd<float, 8> corr = esimd::slm_block_load<float, 8>(kCorr);
      esimd::simd<float, 64> cr;
#pragma unroll
      for (int i = 0; i < kRows; ++i) cr.select<8, 1>(i * 8) = corr[i];
#pragma unroll
      for (int g = 0; g < DG; ++g) {
        esimd::simd<sycl::half, 128> B = v_tile<Q8>(vc, vs, o.va + g * 128, o.vsa + (g / 2) * 16);
        O[g] = xmxd::dpas<8, 8, float>(O[g] * cr, B, Pa);
      }
    }
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
      esimd::slm_scalar_store<float>(kM + (j * RPT + i) * 4, m[i]);
      esimd::slm_scalar_store<float>(kL + (j * RPT + i) * 4, l[i]);
    }
    esimd::barrier();
    const esimd::simd<float, 8> L = esimd::slm_block_load<float, 8>(kL);
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int32_t r = r0 + i;
      if (r >= rows) break;
      if (splits == 1) {
        float* dst = out + ((base + r / G) * n_head + kvh * G + r % G) * D + j * kSlice;
        const float inv = 1.0f / L[i];
#pragma unroll
        for (int g = 0; g < DG; ++g) esimd::block_store<float, 8>(dst + g * 8, O[g].template select<8, 1>(i * 8) * inv);
      } else {
        float* dst = part + (tile * splits + s) * part_stride(D) + i * D + j * kSlice;
#pragma unroll
        for (int g = 0; g < DG; ++g) esimd::block_store<float, 8>(dst + g * 8, O[g].template select<8, 1>(i * 8));
      }
    }
    if (splits > 1 && j == 0) {
      float* ml = part + (tile * splits + s) * part_stride(D) + kRows * D;
      esimd::block_store<float, 8>(ml, esimd::slm_block_load<float, 8>(kM));
      esimd::block_store<float, 8>(ml + 8, L);
    }
  }
};

// The prefill kernel: a work-group per group of QT tiles of one chunk and K/V head. Its NC compute
// threads (P per tile, a 64-dimension slice each) run the tile loop of Flash over one shared copy
// of each 16-key block of K and V in shared local memory; L loader threads keep the next block's
// pieces in their registers and store them while the compute threads are past the point of
// reading the old ones. Per block, between the two barriers of the score exchange: K of block b+1
// (block b's Q.K^T is done) and V of block b (block b-1's P.V is done).
template <int D, int G, int QT, bool Q8>
struct FlashGroup {
  static constexpr int P = D / kSlice, NC = QT * P, L = 16, NT = NC + L;
  static constexpr int RPT = kRows / P, DC = kSlice / 16, DG = kSlice / 8;
  static constexpr int kBlockBytes = kKeys * D * 2;            // one block of K, or of V
  static constexpr int LP = kBlockBytes / 256 / L;             // 256-byte pieces of each per loader
  static constexpr int LT = D / 8 / L;                         // int8: tiles of each per loader
  static constexpr uint32_t kK = 0, kV = kBlockBytes, kScores = 2 * kBlockBytes, kProbs = kScores + NC * 512,
                            kCorr = kProbs + QT * 256, kM = kCorr + QT * 32, kL = kM + QT * 32, kSlm = kL + QT * 32;
  static_assert(kSlm <= 65536, "shared local memory");

  const sycl::half* qt;
  const int32_t* tiles;
  const int32_t* groups;
  const int32_t *slot_of, *pos_of;
  const void *kc, *vc;
  const sycl::half *ks, *vs;
  const int32_t* table;
  int32_t pages, window, n_head_kv, n_head;
  float* out;

  int32_t head_page(int32_t slot, int32_t kvh, int32_t key) const SYCL_ESIMD_FUNCTION {
    const int32_t logical = key / int32_t(kPage);
    const int32_t phys = table ? table[slot * pages + logical] : slot * pages + logical % pages;
    return phys * n_head_kv + kvh;
  }
  // Block b's K (or, with v, V) tiles of this K/V head, one contiguous run of D/8 tiles: their
  // element offset, and that of their scales.
  int32_t block(int32_t b, int32_t slot, int32_t kvh, bool v) const SYCL_ESIMD_FUNCTION {
    const int32_t key0 = b * kKeys, in_page = key0 % int32_t(kPage);
    return head_page(slot, kvh, key0) * int32_t(kPage * D) + (v ? (in_page / 16) * (D / 8) : (in_page / 8) * (D / 16)) * 128;
  }
  int32_t block_scales(int32_t b, int32_t slot, int32_t kvh, bool v) const SYCL_ESIMD_FUNCTION {
    const int32_t key0 = b * kKeys, in_page = key0 % int32_t(kPage);
    return head_page(slot, kvh, key0) * int32_t(kPage * D / 16) + (v ? (in_page / 16) * (D / 16) * 16 : (in_page / 8) * (D / 16) * 8);
  }

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlm>();
    const int tid = int(it.get_local_id(0));
    const int32_t grp = int32_t(it.get_group(0)), t0 = groups[2 * grp], nt = groups[2 * grp + 1];
    const int32_t base = tiles[t0 * 4], rows = tiles[t0 * 4 + 1], kvh = tiles[t0 * 4 + 3];
    const int32_t t_first = base + tiles[t0 * 4 + 2] / G;
    const int32_t t_last = base + (std::min(tiles[(t0 + nt - 1) * 4 + 2] + kRows, rows) - 1) / G;
    const int32_t slot = slot_of[t_first];
    const int32_t lo = window > 0 ? std::max(0, pos_of[t_first] - window + 1) : 0;
    const int32_t b0 = lo / kKeys, b1 = pos_of[t_last] / kKeys + 1;

    // Two loops with the same barriers: the loaders' registers and the compute threads' are never
    // live together, which a shared loop made the compiler spill.
    if (tid >= NC && Q8) {
      // int8: each loader holds LT tiles of K and of V with their scales, and dequantizes them
      // into the fp16 layout of shared memory as it stores them.
      const int l = tid - NC;
      esimd::simd<int8_t, 128> raw[2][LT];
      esimd::simd<sycl::half, 16> sc[2][LT];
      auto load = [&](int32_t b, bool v, int into) SYCL_ESIMD_FUNCTION {
        const int32_t at = block(b, slot, kvh, v), sat = block_scales(b, slot, kvh, v);
#pragma unroll
        for (int i = 0; i < LT; ++i) {
          const int t = l + i * L;
          raw[into][i] = esimd::block_load<int8_t, 128>(static_cast<const int8_t*>(v ? vc : kc) + at + t * 128);
          if (v) sc[into][i] = esimd::block_load<sycl::half, 16>(vs + sat + (t / 2) * 16);
          else sc[into][i].template select<8, 1>(0) = esimd::block_load<sycl::half, 8>(ks + sat + t * 8);
        }
      };
      auto store = [&](uint32_t region, int from, bool v) SYCL_ESIMD_FUNCTION {
#pragma unroll
        for (int i = 0; i < LT; ++i) {
          esimd::simd<sycl::half, 128> pat;
          if (v) {
#pragma unroll
            for (int r = 0; r < 8; ++r) pat.select<16, 1>(16 * r) = sc[from][i].template replicate_vs_w_hs<8, 0, 2, 1>(2 * r);
          } else {
            esimd::simd<sycl::half, 16> p16;
            p16.select<8, 2>(0) = sc[from][i].template select<8, 1>(0);
            p16.select<8, 2>(1) = sc[from][i].template select<8, 1>(0);
            pat = p16.replicate<8>();
          }
          esimd::slm_block_store<sycl::half, 128>(region + (l + i * L) * 256, esimd::simd<sycl::half, 128>(raw[from][i]) * pat);
        }
      };
      load(b0, false, 0);
      store(kK, 0, false);
      esimd::barrier();
      for (int32_t b = b0; b < b1; ++b) {
        load(b, true, 1);
        if (b + 1 < b1) load(b + 1, false, 0);
        esimd::barrier();
        store(kV, 1, true);
        if (b + 1 < b1) store(kK, 0, false);
        esimd::barrier();
      }
      esimd::barrier();
      return;
    }
    if (tid >= NC) {
      const int l = tid - NC;
      esimd::simd<uint32_t, 64> buf[2][LP];
      auto load = [&](int32_t b, bool v, int into) SYCL_ESIMD_FUNCTION {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(static_cast<const sycl::half*>(v ? vc : kc) + block(b, slot, kvh, v));
#pragma unroll
        for (int i = 0; i < LP; ++i) buf[into][i] = esimd::block_load<uint32_t, 64>(src + (l + i * L) * 64);
      };
      auto store = [&](uint32_t region, int from) SYCL_ESIMD_FUNCTION {
#pragma unroll
        for (int i = 0; i < LP; ++i) esimd::slm_block_store<uint32_t, 64>(region + (l + i * L) * 256, buf[from][i]);
      };
      load(b0, false, 0);
      store(kK, 0);
      esimd::barrier();
      for (int32_t b = b0; b < b1; ++b) {
        load(b, true, 1);
        if (b + 1 < b1) load(b + 1, false, 0);
        esimd::barrier();
        store(kV, 1);
        if (b + 1 < b1) store(kK, 0);
        esimd::barrier();
      }
      esimd::barrier();
      return;
    }

    const int qi = tid / P, j = tid % P;
    const bool active = qi < nt;
    const int32_t tile = t0 + (active ? qi : 0), r0 = tiles[tile * 4 + 2];
    int32_t hi[RPT], wlo[RPT];
    float m[RPT], l_[RPT];
#pragma unroll
    for (int i = 0; i < RPT; ++i) {
      const int32_t r = r0 + j * RPT + i;
      hi[i] = active && r < rows ? pos_of[base + r / G] : -1;
      wlo[i] = window > 0 ? hi[i] - window : -1;
      m[i] = kNegBig;
      l_[i] = 0;
    }
    esimd::simd<float, 64> O[DG];
#pragma unroll
    for (int g = 0; g < DG; ++g) O[g] = 0;
    const sycl::half* qs = qt + tile * (kRows * D) + j * DC * 128;
    const uint32_t sc_at = kScores + (qi * P + j) * 512, pr_at = kProbs + qi * 256, co_at = kCorr + qi * 32;
    esimd::barrier();
    for (int32_t b = b0; b < b1; ++b) {
      if (active) {
        esimd::simd<float, 64> S0 = 0, S1 = 0;
#pragma unroll
        for (int dc = 0; dc < DC; ++dc) {
          const esimd::simd<sycl::half, 128> A = esimd::block_load<sycl::half, 128>(qs + dc * 128);
          const esimd::simd<sycl::half, 128> B0 = esimd::slm_block_load<sycl::half, 128>(kK + (j * DC + dc) * 256);
          const esimd::simd<sycl::half, 128> B1 = esimd::slm_block_load<sycl::half, 128>(kK + (D / 16 + j * DC + dc) * 256);
          S0 = xmxd::dpas<8, 8, float>(S0, B0, A);
          S1 = xmxd::dpas<8, 8, float>(S1, B1, A);
        }
        esimd::slm_block_store<float, 64>(sc_at, S0);
        esimd::slm_block_store<float, 64>(sc_at + 256, S1);
      }
      esimd::barrier();
      if (active) {
        esimd::simd<float, 8 * RPT> s0 = 0, s1 = 0;
#pragma unroll
        for (int t = 0; t < P; ++t) {
          s0 += esimd::slm_block_load<float, 8 * RPT>(kScores + (qi * P + t) * 512 + j * RPT * 32);
          s1 += esimd::slm_block_load<float, 8 * RPT>(kScores + (qi * P + t) * 512 + 256 + j * RPT * 32);
        }
        const esimd::simd<int, 16> key(b * kKeys, 1);
#pragma unroll
        for (int i = 0; i < RPT; ++i) {
          esimd::simd<float, 16> sc;
          sc.select<8, 1>(0) = s0.template select<8, 1>(i * 8);
          sc.select<8, 1>(8) = s1.template select<8, 1>(i * 8);
          sc.merge(sc, esimd::simd<float, 16>(-std::numeric_limits<float>::infinity()), (key <= hi[i]) & (key > wlo[i]));
          const float mx = std::max(m[i], esimd::hmax<float>(sc));
          const esimd::simd<float, 16> p = esimd::exp(sc - mx);
          const float corr = esimd::exp(m[i] - mx);
          l_[i] = l_[i] * corr + esimd::reduce<float>(p, std::plus<>());
          m[i] = mx;
          esimd::slm_block_store<sycl::half, 16>(pr_at + (j * RPT + i) * 32, esimd::simd<sycl::half, 16>(p));
          esimd::slm_scalar_store<float>(co_at + (j * RPT + i) * 4, corr);
        }
      }
      esimd::barrier();
      if (active) {
        const esimd::simd<sycl::half, 128> Pa = esimd::slm_block_load<sycl::half, 128>(pr_at);
        const esimd::simd<float, 8> corr = esimd::slm_block_load<float, 8>(co_at);
        esimd::simd<float, 64> cr;
#pragma unroll
        for (int i = 0; i < kRows; ++i) cr.select<8, 1>(i * 8) = corr[i];
#pragma unroll
        for (int g = 0; g < DG; ++g) {
          const esimd::simd<sycl::half, 128> B = esimd::slm_block_load<sycl::half, 128>(kV + (j * DG + g) * 256);
          O[g] = xmxd::dpas<8, 8, float>(O[g] * cr, B, Pa);
        }
      }
    }
    if (active)
#pragma unroll
      for (int i = 0; i < RPT; ++i) esimd::slm_scalar_store<float>(kL + qi * 32 + (j * RPT + i) * 4, l_[i]);
    esimd::barrier();
    if (!active) return;
    const esimd::simd<float, 8> Lr = esimd::slm_block_load<float, 8>(kL + qi * 32);
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
      const int32_t r = r0 + i;
      if (r >= rows) break;
      float* dst = out + ((base + r / G) * n_head + kvh * G + r % G) * D + j * kSlice;
      const float inv = 1.0f / Lr[i];
#pragma unroll
      for (int g = 0; g < DG; ++g) esimd::block_store<float, 8>(dst + g * 8, O[g].template select<8, 1>(i * 8) * inv);
    }
  }
};


constexpr int64_t kPerSplitMax = 8;

template <int D, int G, bool Q8>
void run(sycl::queue& q, const float* Q, const Tokens& tok, const TileList& list, const int32_t* tiles, const Cache& c,
         int64_t n_head, float scale, float* out, Scratch& scratch) {
  constexpr int P = Flash<D, G, Q8>::P, QT = D >= 512 ? 6 : 12;
  const int64_t n_tiles = list.size(), max_keys = list.max_keys;
  if (list.per_group != QT) throw std::runtime_error("attention: tile list grouped for another head size");
  const int32_t H = int32_t(n_head);
  // Query tiles in fp16, the A operands of Q.K^T: [tile][D/16][8 rows][16].
  sycl::half* qt = grow(q, scratch.q, scratch.q_halves, size_t(n_tiles * kRows * D));
  q.parallel_for(sycl::range<3>(size_t(n_tiles), kRows, D), [=](sycl::id<3> id) {
    const int32_t tile = int32_t(id[0]), i = int32_t(id[1]), d = int32_t(id[2]);
    const int32_t base = tiles[tile * 4], rows = tiles[tile * 4 + 1], r = tiles[tile * 4 + 2] + i, kvh = tiles[tile * 4 + 3];
    float v = 0;
    if (r < rows) v = Q[((base + r / G) * H + kvh * G + r % G) * D + d] * scale;
    qt[tile * kRows * D + (d / 16) * 128 + i * 16 + d % 16] = sycl::half(v);
  });
  // Enough groups to fill the card (a prefill chunk): the group kernel, one pass over the keys.
  if (list.n_groups() * FlashGroup<D, G, QT, Q8>::NT >= 8192) {
    FlashGroup<D, G, QT, Q8> k{qt,   tiles, tiles + 4 * n_tiles, tok.slot, tok.pos, c.k, c.v, c.ks, c.vs, c.table, int32_t(c.pages),
                               int32_t(c.window), int32_t(c.n_head_kv), H, out};
    constexpr int NT = FlashGroup<D, G, QT, Q8>::NT;
    q.parallel_for(sycl::nd_range<1>(size_t(list.n_groups() * NT), NT), k);
    return;
  }
  // Enough threads to fill the card: when there are few tiles (decode), split each tile's keys.
  // Each block costs a thread several microseconds of load latency, so short splits win, but every
  // split adds partials to merge. Up to 4k keys the splits are as measured on 2026-10-24 (at least
  // 4 blocks, at most 32 splits); beyond, at most kPerSplitMax blocks a split, with the two-level
  // merge below (2026-10-26: global8 decode at 16k keys 470 -> 302 us; kPerSplitMax 4 gave 366).
  const int64_t max_blocks = max_keys / kKeys + 2;
  const int64_t per_split = std::clamp<int64_t>(max_blocks / 32, 4, kPerSplitMax);
  const int64_t splits = std::clamp<int64_t>((8192 + n_tiles * P - 1) / (n_tiles * P), 1, std::max<int64_t>(1, max_blocks / per_split));
  int32_t sp = int32_t(splits);
  float* part = splits > 1 ? grow(q, scratch.part, scratch.part_floats, size_t(n_tiles * splits * part_stride(D))) : nullptr;
  Flash<D, G, Q8> k{qt, tiles, tok.slot, tok.pos, c.k, c.v, c.ks, c.vs, c.table, int32_t(c.pages), int32_t(c.window),
                    int32_t(c.n_head_kv), H, sp, out, part};
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(splits), size_t(n_tiles * P)), sycl::range<2>(1, P)), k);
  if (splits == 1) return;
  // Each work-item of the merge below loops over every split, which at 128 splits was half of
  // the decode time: past 32 splits, first fold each group of 8 consecutive splits into one
  // partial in the same layout (the same rescaling as the merge), then merge those.
  if (splits > 32) {
    const int32_t sp2 = (sp + 7) / 8;
    float* part2 = grow(q, scratch.part2, scratch.part2_floats, size_t(n_tiles) * sp2 * part_stride(D));
    q.parallel_for(sycl::range<3>(size_t(n_tiles) * kRows, D, sp2), [=](sycl::id<3> id) {
      const int32_t tile = int32_t(id[0]) / kRows, i = int32_t(id[0]) % kRows, d = int32_t(id[1]), g = int32_t(id[2]);
      const int32_t rows = tiles[tile * 4 + 1], r = tiles[tile * 4 + 2] + i;
      if (r >= rows) return;
      const float* p = part + tile * sp * part_stride(D);
      float* o2 = part2 + (tile * sp2 + g) * part_stride(D);
      float mg = kNegBig, lg = 0, o = 0;
      for (int32_t s = g * 8; s < g * 8 + 8 && s < sp; ++s)
        if (p[s * part_stride(D) + kRows * D + kRows + i] > 0) mg = sycl::fmax(mg, p[s * part_stride(D) + kRows * D + i]);
      for (int32_t s = g * 8; s < g * 8 + 8 && s < sp; ++s) {
        const float* ps = p + s * part_stride(D);
        const float ls = ps[kRows * D + kRows + i];
        if (ls == 0) continue;
        const float w = sycl::exp(ps[kRows * D + i] - mg);
        lg += ls * w;
        o += ps[i * D + d] * w;
      }
      o2[i * D + d] = o;
      if (d == 0) o2[kRows * D + i] = mg, o2[kRows * D + kRows + i] = lg;
    });
    sp = sp2;
    part = part2;
  }
  q.parallel_for(sycl::range<3>(size_t(n_tiles), kRows, D), [=](sycl::id<3> id) {
    const int32_t tile = int32_t(id[0]), i = int32_t(id[1]), d = int32_t(id[2]);
    const int32_t base = tiles[tile * 4], rows = tiles[tile * 4 + 1], r = tiles[tile * 4 + 2] + i, kvh = tiles[tile * 4 + 3];
    if (r >= rows) return;
    const float* p = part + tile * sp * part_stride(D);
    float mx = kNegBig;
    for (int32_t s = 0; s < sp; ++s)
      if (p[s * part_stride(D) + kRows * D + kRows + i] > 0) mx = sycl::fmax(mx, p[s * part_stride(D) + kRows * D + i]);
    float lsum = 0, o = 0;
    for (int32_t s = 0; s < sp; ++s) {
      const float* ps = p + s * part_stride(D);
      const float ls = ps[kRows * D + kRows + i];
      if (ls == 0) continue;  // a split with no keys for this row
      const float w = sycl::exp(ps[kRows * D + i] - mx);
      lsum += ls * w;
      o += ps[i * D + d] * w;
    }
    out[((base + r / G) * H + kvh * G + r % G) * D + d] = o / lsum;
  });
}

}  // namespace

void attend(sycl::queue& q, const float* Q, const Tokens& tok, const TileList& list, const int32_t* packed, const Cache& c,
            int64_t n_head, float scale, float* out, Scratch& scratch) {
  if (list.size() == 0) return;
  const int64_t group = n_head / c.n_head_kv;
  if (c.head_dim == 256 && group == 2) {
    if (c.int8()) run<256, 2, true>(q, Q, tok, list, packed, c, n_head, scale, out, scratch);
    else run<256, 2, false>(q, Q, tok, list, packed, c, n_head, scale, out, scratch);
  } else if (c.head_dim == 512 && group == 16) {
    if (c.int8()) run<512, 16, true>(q, Q, tok, list, packed, c, n_head, scale, out, scratch);
    else run<512, 16, false>(q, Q, tok, list, packed, c, n_head, scale, out, scratch);
  }
  else throw std::runtime_error("attention: no kernel for head_dim " + std::to_string(c.head_dim) + ", group " + std::to_string(group));
}

void write(sycl::queue& q, const float* K, const float* V, const Tokens& tok, const Cache& cache) {
  const Cache c = cache;
  const Tokens t = tok;
  const int32_t D = int32_t(c.head_dim), Hkv = int32_t(c.n_head_kv), pages = int32_t(c.pages);
  // Where element (position n of its page, dim d) of head page hp goes, in elements.
  auto k_at = [=](int32_t hp, int32_t n, int32_t d) {
    return hp * int32_t(kPage) * D + ((n / 8) * (D / 16) + d / 16) * 128 + ((d % 16) / 2) * 16 + (n % 8) * 2 + d % 2;
  };
  auto v_at = [=](int32_t hp, int32_t n, int32_t d) {
    return hp * int32_t(kPage) * D + ((n / 16) * (D / 8) + d / 8) * 128 + ((n % 16) / 2) * 16 + (d % 8) * 2 + n % 2;
  };
  auto head_page = [=](int32_t i, int32_t h, int32_t& n) {
    const int32_t pos = t.pos[i], logical = pos / int32_t(kPage), slot = t.slot[i];
    n = pos % int32_t(kPage);
    const int32_t phys = c.table ? c.table[slot * pages + logical] : slot * pages + logical % pages;
    return phys * Hkv + h;
  };
  if (!c.int8()) {
    q.parallel_for(sycl::range<3>(size_t(t.n), size_t(Hkv), size_t(D)), [=](sycl::id<3> id) {
      const int32_t i = int32_t(id[0]), h = int32_t(id[1]), d = int32_t(id[2]);
      int32_t n;
      const int32_t hp = head_page(i, h, n), src = (i * Hkv + h) * D + d;
      static_cast<sycl::half*>(c.k)[k_at(hp, n, d)] = sycl::half(K[src]);
      static_cast<sycl::half*>(c.v)[v_at(hp, n, d)] = sycl::half(V[src]);
    });
    return;
  }
  // int8: a work-item per 16 dimensions, which share a scale.
  q.parallel_for(sycl::range<3>(size_t(t.n), size_t(Hkv), size_t(D / 16)), [=](sycl::id<3> id) {
    const int32_t i = int32_t(id[0]), h = int32_t(id[1]), d0 = int32_t(id[2]) * 16;
    int32_t n;
    const int32_t hp = head_page(i, h, n), src = (i * Hkv + h) * D + d0, sat = hp * int32_t(kPage) * (D / 16);
    for (int which = 0; which < 2; ++which) {
      const float* x = (which ? V : K) + src;
      float amax = 0;
      for (int e = 0; e < 16; ++e) amax = sycl::fmax(amax, sycl::fabs(x[e]));
      const float scale = amax / 127.0f, inv = scale != 0 ? 1.0f / scale : 0.0f;
      int8_t* dst = static_cast<int8_t*>(which ? c.v : c.k);
      for (int e = 0; e < 16; ++e) dst[which ? v_at(hp, n, d0 + e) : k_at(hp, n, d0 + e)] = int8_t(sycl::round(x[e] * inv));
      if (which) c.vs[sat + ((n / 16) * (D / 16) + d0 / 16) * 16 + n % 16] = sycl::half(scale);
      else c.ks[sat + ((n / 8) * (D / 16) + d0 / 16) * 8 + n % 8] = sycl::half(scale);
    }
  });
}

}  // namespace eklow::attn
