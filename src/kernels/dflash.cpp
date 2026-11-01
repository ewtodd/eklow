#include "kernels/dflash.hpp"

#include <limits>
#include <stdexcept>

namespace eklow::dflash {

namespace {
constexpr int kHead = 128, kHalf = 64, kChunk = 2048, kWg = 256, kPer = kChunk / kWg, kTop = 16;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
}  // namespace

void heads(sycl::queue& q, const float* src, int64_t ld, int64_t n_tok, int nq, int nk, int64_t q_off, int64_t k_off, int64_t v_off,
           const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K, float* V) {
  const int n_heads = nq + 2 * nk;
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(n_tok), size_t(n_heads) * kHalf), sycl::range<2>(1, kHalf)), [=](sycl::nd_item<2> it) {
    const int64_t t = int64_t(it.get_group(0));
    const int h = int(it.get_group(1)), i = int(it.get_local_id(1));
    const bool is_q = h < nq, is_k = !is_q && h < nq + nk;
    const int head = is_q ? h : is_k ? h - nq : h - nq - nk;
    const float* s = src + t * ld + (is_q ? q_off : is_k ? k_off : v_off) + int64_t(head) * kHead;
    const float a = s[i], b = s[i + kHalf];
    if (!is_q && !is_k) {
      V[(t * nk + head) * kHead + i] = a;
      V[(t * nk + head) * kHead + i + kHalf] = b;
      return;
    }
    const float ss = sycl::reduce_over_group(it.get_group(), a * a + b * b, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(kHead) + eps);
    const float* w = is_q ? q_norm : k_norm;
    const float y0 = a * inv * w[i], y1 = b * inv * w[i + kHalf];
    const float angle = float(pos[t]) * inv_freq[i];
    const float c = sycl::cos(angle), sn = sycl::sin(angle);
    float* d = (is_q ? Q + (t * nq + head) * kHead : K + (t * nk + head) * kHead);
    d[i] = y0 * c - y1 * sn;
    d[i + kHalf] = y0 * sn + y1 * c;
  });
}

void conv(sycl::queue& q, const float* x, const float* dyn, const float* base, int side, int64_t rows, int64_t block_len, int64_t hidden,
          int64_t group, float* out, bool accumulate) {
  const int64_t groups = hidden / group;
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(hidden)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), c = int64_t(id[1]);
    const bool has_prev = t % block_len != 0;
    const float* d = dyn + t * 4 * groups + int64_t(side) * 2 * groups + c / group;  // kernel size 2: [side][tap][group]
    float acc = (base[(int64_t(side) * 2) * hidden + c] + d[0]) * x[t * hidden + c];
    if (has_prev) acc += (base[(int64_t(side) * 2 + 1) * hidden + c] + d[groups]) * x[(t - 1) * hidden + c];
    out[t * hidden + c] = accumulate ? out[t * hidden + c] + acc : acc;
  });
}

int64_t topk_chunks(int64_t n) { return (n + kChunk - 1) / kChunk; }

namespace {
// (value, id) pairs: larger value first, smaller id first among equals.
inline bool better(float va, int32_t ia, float vb, int32_t ib) { return va > vb || (va == vb && ia < ib); }
}  // namespace

void topk16(sycl::queue& q, const float* x, int64_t rows, int64_t n, int32_t* scratch_ids, float* scratch_vals, int32_t* ids, float* vals) {
  const int64_t chunks = topk_chunks(n);
  if (chunks * kTop > kChunk) throw std::runtime_error("topk16: row too long for the merge");
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(rows), size_t(chunks) * kWg), sycl::range<2>(1, kWg)), [=](sycl::nd_item<2> it) {
    const int64_t r = int64_t(it.get_group(0)), ch = int64_t(it.get_group(1));
    const int i = int(it.get_local_id(1));
    float v[kPer];
    int32_t id[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int64_t e = ch * kChunk + i + int64_t(j) * kWg;
      id[j] = int32_t(e);
      v[j] = e < n ? x[r * n + e] : kNegInf;
    }
    auto g = it.get_group();
    for (int round = 0; round < kTop; ++round) {
      float bv = v[0];
      int32_t bi = id[0];
#pragma unroll
      for (int j = 1; j < kPer; ++j)
        if (better(v[j], id[j], bv, bi)) bv = v[j], bi = id[j];
      const float m = sycl::reduce_over_group(g, bv, sycl::maximum<float>());
      const int32_t mi = sycl::reduce_over_group(g, bv == m ? bi : std::numeric_limits<int32_t>::max(), sycl::minimum<int32_t>());
      if (i == 0) {
        scratch_vals[(r * chunks + ch) * kTop + round] = m;
        scratch_ids[(r * chunks + ch) * kTop + round] = mi;
      }
#pragma unroll
      for (int j = 0; j < kPer; ++j)
        if (id[j] == mi && v[j] == m) v[j] = kNegInf, id[j] = std::numeric_limits<int32_t>::max();
    }
  });
  // The merge: a row's chunks * 16 survivors, at most 256 * 8 of them.
  q.parallel_for(sycl::nd_range<2>(sycl::range<2>(size_t(rows), kWg), sycl::range<2>(1, kWg)), [=](sycl::nd_item<2> it) {
    const int64_t r = int64_t(it.get_group(0));
    const int i = int(it.get_local_id(1));
    const int64_t total = chunks * kTop;
    float v[kPer];
    int32_t id[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int64_t e = i + int64_t(j) * kWg;
      v[j] = e < total ? scratch_vals[r * total + e] : kNegInf;
      id[j] = e < total ? scratch_ids[r * total + e] : std::numeric_limits<int32_t>::max();
    }
    auto g = it.get_group();
    for (int round = 0; round < kTop; ++round) {
      float bv = v[0];
      int32_t bi = id[0];
#pragma unroll
      for (int j = 1; j < kPer; ++j)
        if (better(v[j], id[j], bv, bi)) bv = v[j], bi = id[j];
      const float m = sycl::reduce_over_group(g, bv, sycl::maximum<float>());
      const int32_t mi = sycl::reduce_over_group(g, bv == m ? bi : std::numeric_limits<int32_t>::max(), sycl::minimum<int32_t>());
      if (i == 0) {
        vals[r * kTop + round] = m;
        ids[r * kTop + round] = mi;
      }
#pragma unroll
      for (int j = 0; j < kPer; ++j)
        if (id[j] == mi && v[j] == m) v[j] = kNegInf, id[j] = std::numeric_limits<int32_t>::max();
    }
  });
}

void predecessor_ids(sycl::queue& q, const int32_t* cand, const int32_t* anchors, int64_t blocks, int64_t P, int32_t* ids) {
  q.parallel_for(sycl::range<3>(size_t(blocks), size_t(P), kTop), [=](sycl::id<3> id) {
    const int64_t b = int64_t(id[0]), i = int64_t(id[1]), a = int64_t(id[2]);
    ids[(b * P + i) * kTop + a] = i == 0 ? anchors[b] : cand[(b * P + i - 1) * kTop + a];
  });
}

}  // namespace eklow::dflash
