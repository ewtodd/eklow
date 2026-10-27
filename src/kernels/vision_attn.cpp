#include "kernels/vision_attn.hpp"

#include <sycl/ext/intel/esimd.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace eklow::vattn {

namespace esimd = sycl::ext::intel::esimd;
namespace xmxd = sycl::ext::intel::esimd::xmx;

namespace {

constexpr int kSegTiles = int(kSeg) / 128;
// The running maximum starts finite so that a row with no key yet in a block gets weight 0 rather than NaN.
constexpr float kNegBig = -1e30f;

// Per row (R of them): the 128-key tiles' maxima and sums and their factors [tile][row], the output's factor, the
// running maximum and sum [row], and the running output [row][D].
struct Stats {
  float *mt, *lt, *beta, *alpha, *mrun, *lrun, *orun;
  static size_t floats(int64_t R, int64_t D) { return size_t(R * (3 * kSegTiles + 3 + D)); }
  Stats(float* s, int64_t R) {
    mt = s, lt = mt + kSegTiles * R, beta = lt + kSegTiles * R, alpha = beta + kSegTiles * R;
    mrun = alpha + R, lrun = mrun + R, orun = lrun + R;
  }
};

template <class T>
T* grow(sycl::queue& q, T*& p, size_t& have, size_t need) {
  if (need > have) {
    if (p) {
      q.wait();
      sycl::free(p, q);
    }
    have = need;
    p = sycl::malloc_device<T>(have, q);
    if (!p) throw std::runtime_error("vision attention: out of device memory for scratch");
  }
  return p;
}

// S = Q K^T for 256 rows x 128 keys of a segment, work-groups of 16 x 4 threads (each 16 rows x 32 keys); 32-deep
// K blocks staged in shared local memory, the next block's loads issued before this block's dpas. The epilogue
// masks the keys at or past kend, takes each row's maximum over the 128 keys and its running maximum, and writes
// P = exp(S - m) (fp16, A layout) with that maximum and the row's sum over the 128 keys.
template <int D>
struct QK {
  static constexpr int WM = 16, WN = 4, KB = D / 32;
  // Per K block: B pieces (16 key tiles x 2 K halves) then A pieces (32 row tiles x 2 K halves), 256 bytes each;
  // two such buffers, then the epilogue's per-row partial maxima and sums [4 column threads][256 rows].
  static constexpr uint32_t kBBuf = 32 * 256, kBuf = kBBuf + 64 * 256, kRed = 2 * kBuf, kSlm = kRed + 2 * 4 * 256 * 4;
  const sycl::half *qt, *kt;
  sycl::half* p;
  float *mt, *lt;
  const float* mrun;
  int32_t R, seg0, kend, first, seg;
  int64_t op_stride, p_stride, stat_stride;  // per head of the launch: the Q, K, V tiles, P, the statistics

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlm>();
    const int wi = int(it.get_local_id(1)), wj = int(it.get_local_id(2)), tid = wi * WN + wj;
    const int64_t g = int64_t(it.get_group(0));
    const int32_t tm = int32_t(it.get_group(1)), tn = int32_t(it.get_group(2));
    const sycl::half* qt = this->qt + g * op_stride;
    const sycl::half* kt = this->kt + g * op_stride;
    sycl::half* p = this->p + g * p_stride;
    float* mt = this->mt + g * stat_stride;
    float* lt = this->lt + g * stat_stride;
    const float* mrun = this->mrun + g * stat_stride;
    // Every thread stages one A piece (row tile tid / 2, K half tid % 2) and threads below 32 one B piece (the
    // work-group's key tile tid / 2, K half tid % 2) of each block.
    const sycl::half* a_src = qt + (tm * 32 + tid / 2) * (8 * D) + (tid % 2) * 128;
    const int32_t key0 = seg0 + tn * 128 + (tid / 2) * 8;
    const bool b_ok = tid < 32 && key0 < kend;
    const sycl::half* b_src = kt + ((b_ok ? key0 / 8 : 0) * (D / 16) + tid % 2) * 128;
    esimd::simd<sycl::half, 128> Araw, Braw;
    auto load = [&](int kb) SYCL_ESIMD_FUNCTION {
      Araw = esimd::block_load<sycl::half, 128>(a_src + kb * 256);
      if (b_ok) Braw = esimd::block_load<sycl::half, 128>(b_src + kb * 256);
    };
    auto store = [&](uint32_t buf) SYCL_ESIMD_FUNCTION {
      esimd::slm_block_store<sycl::half, 128>(buf + kBBuf + tid * 256, Araw);
      if (tid < 32) esimd::slm_block_store<sycl::half, 128>(buf + tid * 256, b_ok ? Braw : esimd::simd<sycl::half, 128>(0));
    };
    esimd::simd<float, 64> acc[2][4];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[r][c] = 0;
    load(0);
    store(0);
    esimd::barrier();
    for (int kb = 0; kb < KB; ++kb) {
      const uint32_t cur = (kb & 1) * kBuf, nxt = kBuf - cur;
      if (kb + 1 < KB) load(kb + 1);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        esimd::simd<sycl::half, 128> A[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) A[r] = esimd::slm_block_load<sycl::half, 128>(cur + kBBuf + ((wi * 2 + r) * 2 + hh) * 256);
        esimd::simd<sycl::half, 128> B = esimd::slm_block_load<sycl::half, 128>(cur + (wj * 4 * 2 + hh) * 256);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          esimd::simd<sycl::half, 128> Bn;
          if (c + 1 < 4) Bn = esimd::slm_block_load<sycl::half, 128>(cur + ((wj * 4 + c + 1) * 2 + hh) * 256);
#pragma unroll
          for (int r = 0; r < 2; ++r) acc[r][c] = xmxd::dpas<8, 8, float>(acc[r][c], B, A[r]);
          B = Bn;
        }
      }
      if (kb + 1 < KB) store(nxt);
      esimd::barrier();
    }

    // Epilogue: this thread's rows are tm*256 + (wi*2 + r)*8 + i, its keys seg0 + tn*128 + (wj*4 + c)*8 + n.
    const int32_t key_t = seg0 + tn * 128 + wj * 32;
    esimd::simd<int32_t, 64> kpat;
#pragma unroll
    for (int i = 0; i < 8; ++i) kpat.select<8, 1>(i * 8) = esimd::simd<int32_t, 8>(0, 1);
    esimd::simd<float, 8> m[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const esimd::simd<int32_t, 64> key = kpat + (key_t + c * 8);
        acc[r][c].merge(esimd::simd<float, 64>(-std::numeric_limits<float>::infinity()), key >= kend);
      }
      esimd::simd<float, 64> t = esimd::max(esimd::max(acc[r][0], acc[r][1]), esimd::max(acc[r][2], acc[r][3]));
#pragma unroll
      for (int i = 0; i < 8; ++i) m[r][i] = esimd::hmax<float>(esimd::simd<float, 8>(t.template select<8, 1>(i * 8)));
      esimd::slm_block_store<float, 8>(kRed + (wj * 256 + (wi * 2 + r) * 8) * 4, m[r]);
    }
    esimd::barrier();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int32_t row0 = (tm * 32 + wi * 2 + r) * 8;
      m[r] = first ? esimd::simd<float, 8>(kNegBig) : esimd::block_load<float, 8>(mrun + row0);
#pragma unroll
      for (int w = 0; w < 4; ++w) m[r] = esimd::max(m[r], esimd::slm_block_load<float, 8>(kRed + (w * 256 + (wi * 2 + r) * 8) * 4));
      esimd::simd<float, 64> mb;
#pragma unroll
      for (int i = 0; i < 8; ++i) mb.select<8, 1>(i * 8) = float(m[r][i]);
      esimd::simd<float, 64> s = 0;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        acc[r][c] = esimd::exp(acc[r][c] - mb);
        s += acc[r][c];
      }
      esimd::simd<float, 8> l;
#pragma unroll
      for (int i = 0; i < 8; ++i) l[i] = esimd::reduce<float>(esimd::simd<float, 8>(s.template select<8, 1>(i * 8)), std::plus<>());
      esimd::slm_block_store<float, 8>(kRed + 4096 + (wj * 256 + (wi * 2 + r) * 8) * 4, l);
      // P in the A layout of P.V: per 8 rows, tiles of 16 keys.
#pragma unroll
      for (int q2 = 0; q2 < 2; ++q2) {
        esimd::simd<sycl::half, 128> pt;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          pt.select<8, 1>(i * 16) = esimd::simd<sycl::half, 8>(acc[r][2 * q2].template select<8, 1>(i * 8));
          pt.select<8, 1>(i * 16 + 8) = esimd::simd<sycl::half, 8>(acc[r][2 * q2 + 1].template select<8, 1>(i * 8));
        }
        esimd::block_store<sycl::half, 128>(p + ((tm * 32 + wi * 2 + r) * (seg / 16) + tn * 8 + wj * 2 + q2) * 128, pt);
      }
    }
    esimd::barrier();
    if (wj != 0) return;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int32_t row0 = (tm * 32 + wi * 2 + r) * 8;
      esimd::simd<float, 8> l = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w) l += esimd::slm_block_load<float, 8>(kRed + 4096 + (w * 256 + (wi * 2 + r) * 8) * 4);
      esimd::block_store<float, 8>(mt + tn * R + row0, m[r]);
      esimd::block_store<float, 8>(lt + tn * R + row0, l);
    }
  }
};

// O = alpha O + (beta P) V for 256 rows x all D dimensions a work-group (16 x 4 threads, each 16 rows x D/4 dims),
// 32-key blocks staged in shared local memory, the probabilities scaled by beta as they are staged.
template <int D>
struct PV {
  static constexpr int WN = 4, NT = 64, DT = D / 8, CT = DT / WN;  // CT: dimension tiles a thread holds
  static_assert(DT % WN == 0);
  // Per 32-key block: B pieces (DT dimension tiles x 2 key halves) then A pieces (32 row tiles x 2 key halves).
  static constexpr uint32_t kBBuf = DT * 2 * 256, kBuf = kBBuf + 64 * 256, kSlm = 2 * kBuf;
  const sycl::half *p, *vt;
  const float *beta, *alpha;
  float* orun;
  int32_t R, seg0, kend, nb, first, rows, seg;
  int64_t op_stride, p_stride, stat_stride;  // per head of the launch

  void operator()(sycl::nd_item<3> it) const SYCL_ESIMD_KERNEL {
    esimd::slm_init<kSlm>();
    const int wi = int(it.get_local_id(1)), wj = int(it.get_local_id(2)), tid = wi * WN + wj;
    const int64_t g = int64_t(it.get_group(0));
    const int32_t tm = int32_t(it.get_group(1));
    const sycl::half* p = this->p + g * p_stride;
    const sycl::half* vt = this->vt + g * op_stride;
    const float* beta = this->beta + g * stat_stride;
    const float* alpha = this->alpha + g * stat_stride;
    float* orun = this->orun + g * stat_stride;
    // Every thread stages one A piece (row tile tid / 2, keys 16 * (tid % 2) of the block) and threads below 2 * DT
    // one B piece (dimension tile tid / 2, keys 16 * (tid % 2) of the block).
    const int32_t a_tile = tm * 32 + tid / 2;
    const sycl::half* a_src = p + (a_tile * (seg / 16) + tid % 2) * 128;
    const float* a_beta = beta + a_tile * 8;
    esimd::simd<sycl::half, 128> Araw, Braw;
    esimd::simd<float, 8> Abeta;
    int32_t nv = 0;  // positions of the B piece before kend
    auto load = [&](int b) SYCL_ESIMD_FUNCTION {
      Araw = esimd::block_load<sycl::half, 128>(a_src + b * 256);
      Abeta = esimd::block_load<float, 8>(a_beta + (b / 4) * R);
      const int32_t key0 = seg0 + b * 32 + (tid % 2) * 16;
      nv = tid < 2 * DT ? std::clamp(kend - key0, 0, 16) : 0;
      if (nv > 0) Braw = esimd::block_load<sycl::half, 128>(vt + ((key0 / 16) * (D / 8) + tid / 2) * 128);
    };
    auto store = [&](uint32_t buf) SYCL_ESIMD_FUNCTION {
      esimd::simd<sycl::half, 128> sb;
#pragma unroll
      for (int i = 0; i < 8; ++i) sb.select<16, 1>(i * 16) = sycl::half(Abeta[i]);
      esimd::slm_block_store<sycl::half, 128>(buf + kBBuf + tid * 256, Araw * sb);
      if (tid < 2 * DT) {
        esimd::simd<sycl::half, 128> B = 0;
        if (nv > 0) {
          B = Braw;
          // Element (position k, dim n) is at (k/2)*16 + n*2 + k%2. Positions from kend on are padding, and their
          // probabilities zero; they are zeroed here so that no NaN there reaches the sum.
          if (nv < 16) {
            const esimd::simd<int16_t, 128> e(0, 1);
            B.merge(esimd::simd<sycl::half, 128>(0), (e / 16) * 2 + e % 2 >= int16_t(nv));
          }
        }
        esimd::slm_block_store<sycl::half, 128>(buf + tid * 256, B);
      }
    };
    esimd::simd<float, 64> acc[2][CT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int c = 0; c < CT; ++c) acc[r][c] = 0;
    load(0);
    store(0);
    esimd::barrier();
    for (int b = 0; b < nb; ++b) {
      const uint32_t cur = (b & 1) * kBuf, nxt = kBuf - cur;
      if (b + 1 < nb) load(b + 1);
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        esimd::simd<sycl::half, 128> A[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) A[r] = esimd::slm_block_load<sycl::half, 128>(cur + kBBuf + ((wi * 2 + r) * 2 + hh) * 256);
        esimd::simd<sycl::half, 128> B = esimd::slm_block_load<sycl::half, 128>(cur + (wj * CT * 2 + hh) * 256);
#pragma unroll
        for (int c = 0; c < CT; ++c) {
          esimd::simd<sycl::half, 128> Bn;
          if (c + 1 < CT) Bn = esimd::slm_block_load<sycl::half, 128>(cur + ((wj * CT + c + 1) * 2 + hh) * 256);
#pragma unroll
          for (int r = 0; r < 2; ++r) acc[r][c] = xmxd::dpas<8, 8, float>(acc[r][c], B, A[r]);
          B = Bn;
        }
      }
      if (b + 1 < nb) store(nxt);
      esimd::barrier();
    }
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int32_t row = (tm * 32 + wi * 2 + r) * 8 + i;
        if (row >= rows) break;
        const float a = first ? 0.0f : alpha[row];
#pragma unroll
        for (int c = 0; c < CT; ++c) {
          float* o = orun + row * D + (wj * CT + c) * 8;
          esimd::simd<float, 8> v = acc[r][c].template select<8, 1>(i * 8);
          if (!first) v += a * esimd::block_load<float, 8>(o);
          esimd::block_store<float, 8>(o, v);
        }
      }
  }
};

// Heads a launch and keys a segment. The probabilities of a segment take R * seg * 2 bytes a head, and one head's P.V
// has only R / 256 work-groups of 64 threads, too few to fill the card (4,096 threads) at the largest sizes; so heads go
// through together, as many as fit kPBudget, and the segment is halved (down to 512 keys) while that brings more work-groups
// a launch, up to kWgTarget.
constexpr int64_t kPBudget = int64_t(64) << 20, kWgTarget = 128;

struct Plan {
  int64_t G, seg;
};

Plan plan(int64_t R, int64_t n_head) {
  Plan best{1, kSeg};
  int64_t best_wgs = 0;
  for (int64_t seg = kSeg; seg >= 512; seg /= 2) {
    const int64_t G = std::clamp<int64_t>(kPBudget / (R * seg * 2), 1, n_head), wgs = G * (R / 256);
    if (wgs > best_wgs) best = {G, seg}, best_wgs = wgs;
    if (wgs >= kWgTarget) break;
  }
  return best;
}

// EKLOW_VATTN_PROFILE=1: wait after each kernel kind and print the sums at exit (the kernels then do not overlap).
struct Prof {
  bool on = std::getenv("EKLOW_VATTN_PROFILE") != nullptr;
  double ms[4] = {};
  std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
  void mark(sycl::queue& q, int kind) {
    if (!on) return;
    q.wait();
    const auto now = std::chrono::steady_clock::now();
    ms[kind] += std::chrono::duration<double, std::milli>(now - t).count();
    t = now;
  }
  ~Prof() {
    if (on) std::fprintf(stderr, "vattn: QK %.1f ms, combine %.1f ms, PV %.1f ms, out %.1f ms\n", ms[0], ms[1], ms[2], ms[3]);
  }
};
Prof g_prof;

template <int D>
void run(sycl::queue& q, const sycl::half* qt, const sycl::half* kt, const sycl::half* vt, int64_t n, int64_t n_head, int64_t d_real,
         Scratch& s, sycl::half* out, int64_t out_k) {
  const int64_t R64 = rows_padded(n);
  const Plan pl = plan(R64, n_head);
  const int64_t G = pl.G, SEG = pl.seg;
  const int32_t R = int32_t(R64), kend = int32_t(n), rows = kend;
  const int64_t op_stride = R64 * D, p_stride = R64 * SEG, stat_stride = int64_t(Stats::floats(R64, D));
  sycl::half* P = grow(q, s.p, s.p_halves, size_t(G * p_stride));
  const Stats st(grow(q, s.stat, s.stat_floats, size_t(G * stat_stride)), R64);
  for (int64_t h0 = 0; h0 < n_head; h0 += G) {
    const int64_t g = std::min(G, n_head - h0);
    const sycl::half *qh = qt + h0 * op_stride, *kh = kt + h0 * op_stride, *vh = vt + h0 * op_stride;
    for (int32_t s0 = 0; s0 < kend; s0 += int32_t(SEG)) {
      const int32_t kseg = std::min<int32_t>(int32_t(SEG), kend - s0), nk = (kseg + 127) / 128, nb = (kseg + 31) / 32, first = s0 == 0;
      QK<D> qk{qh, kh, P, st.mt, st.lt, st.mrun, R, s0, kend, first, int32_t(SEG), op_stride, p_stride, stat_stride};
      g_prof.mark(q, 3);
      q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(g), size_t(R / 16), size_t(nk * 4)), sycl::range<3>(1, 16, 4)), qk);
      g_prof.mark(q, 0);
      q.parallel_for(sycl::range<2>(size_t(g), size_t(R)), [=](sycl::id<2> id) {
        const int64_t off = int64_t(id[0]) * stat_stride;
        const int32_t row = int32_t(id[1]);
        const float m_old = first ? kNegBig : st.mrun[off + row], l_old = first ? 0.0f : st.lrun[off + row];
        float mn = m_old;
        for (int32_t t = 0; t < nk; ++t) mn = sycl::fmax(mn, st.mt[off + t * R + row]);
        const float a = sycl::exp(m_old - mn);
        float l = a * l_old;
        for (int32_t t = 0; t < nk; ++t) {
          const float b = sycl::exp(st.mt[off + t * R + row] - mn);
          st.beta[off + t * R + row] = b;
          l += b * st.lt[off + t * R + row];
        }
        st.mrun[off + row] = mn, st.lrun[off + row] = l, st.alpha[off + row] = a;
      });
      g_prof.mark(q, 1);
      PV<D> pv{P, vh, st.beta, st.alpha, st.orun, R, s0, kend, nb, first, rows, int32_t(SEG), op_stride, p_stride, stat_stride};
      q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(g), size_t(R / 16), 4), sycl::range<3>(1, 16, 4)), pv);
      g_prof.mark(q, 2);
    }
    // The heads' outputs in the activation tiles of the next GEMM.
    q.parallel_for(sycl::range<3>(size_t(g), size_t(rows), size_t(d_real)), [=](sycl::id<3> id) {
      const int64_t hh = int64_t(id[0]), row = int64_t(id[1]), d = int64_t(id[2]);
      const float* o = st.orun + hh * stat_stride;
      const float l = st.lrun[hh * stat_stride + row];
      out[xmx::act16_index(row, (h0 + hh) * d_real + d, out_k / 16)] = sycl::half(o[row * D + d] / l);
    });
  }
}

}  // namespace

uint64_t scratch_bytes(int64_t rows, int64_t D, int64_t n_head) {
  const int64_t R = rows_padded(rows);
  const Plan pl = plan(R, n_head);
  return uint64_t(pl.G) * (uint64_t(R) * pl.seg * sizeof(sycl::half) + Stats::floats(R, D) * sizeof(float));
}

void reserve(sycl::queue& q, Scratch& s, int64_t rows, int64_t D, int64_t n_head) {
  const int64_t R = rows_padded(rows);
  const Plan pl = plan(R, n_head);
  grow(q, s.p, s.p_halves, size_t(pl.G * R * pl.seg));
  grow(q, s.stat, s.stat_floats, size_t(pl.G) * Stats::floats(R, D));
}

void release(sycl::queue& q, Scratch& s) {
  q.wait();
  if (s.p) sycl::free(s.p, q);
  if (s.stat) sycl::free(s.stat, q);
  s = Scratch{};
}

void attend(sycl::queue& q, const sycl::half* qt, const sycl::half* kt, const sycl::half* vt, int64_t n, int64_t n_head, int64_t D,
            int64_t d_real, Scratch& s, sycl::half* out, int64_t out_k) {
  if (D == 96) run<96>(q, qt, kt, vt, n, n_head, d_real, s, out, out_k);
  else if (D == 64) run<64>(q, qt, kt, vt, n, n_head, d_real, s, out, out_k);
  else throw std::runtime_error("vision attention: no kernel for head size " + std::to_string(D));
}

}  // namespace eklow::vattn
