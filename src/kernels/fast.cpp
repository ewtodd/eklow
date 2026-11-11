#include "kernels/fast.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace eklow::fast {

namespace {

// The 6-bit value (0..63) of element i of a Q6_K block (ql[128], qh[64], int8 scales[16], fp16 d),
// as ggml lays it out.
inline int q6k_value(const uint8_t* blk, int i) {
  const int half = i / 128, r = i % 128, l = r % 32, quarter = r / 32;
  const uint8_t* ql = blk + half * 64;
  const uint8_t* qh = blk + 128 + half * 32;
  switch (quarter) {
    case 0: return (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
    case 1: return (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
    case 2: return (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
    default: return (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
  }
}

}  // namespace

Q6 repack_q6k(sycl::queue& q, const ref::Weight& raw) {
  if (raw.type != gguf::Type::Q6_K || raw.k % 256) throw std::runtime_error("fast::repack_q6k needs a Q6_K matrix");
  const int64_t k = raw.k, n = raw.n;
  const size_t lo = size_t(n * k / 2), hi = size_t(n * k / 4), sc = size_t(n * k / 16), d = size_t(n * k / 256) * 2;
  auto* p = sycl::malloc_device<uint8_t>(lo + hi + sc + d, q);
  if (!p) throw std::runtime_error("fast::repack_q6k: out of device memory");
  Q6 w{k, n, p, p + lo, reinterpret_cast<int8_t*>(p + lo + hi), reinterpret_cast<sycl::half*>(p + lo + hi + sc)};
  const ref::Weight r = raw;
  // One work-item per 4 elements: one byte of high bits, two of low nibbles.
  q.parallel_for(sycl::range<2>(size_t(n), size_t(k / 4)), [=](sycl::id<2> id) {
    const int64_t row = int64_t(id[0]), e = int64_t(id[1]) * 4;
    const uint8_t* blk = r.data + uint64_t(row) * r.row_bytes + uint64_t(e / 256) * 210;
    int v[4];
    for (int j = 0; j < 4; ++j) v[j] = q6k_value(blk, int((e + j) % 256));
    const int64_t at = row * k + e;
    w.lo[at / 2] = uint8_t((v[0] & 15) | (v[1] & 15) << 4);
    w.lo[at / 2 + 1] = uint8_t((v[2] & 15) | (v[3] & 15) << 4);
    w.hi[at / 4] = uint8_t(v[0] >> 4 | (v[1] >> 4) << 2 | (v[2] >> 4) << 4 | (v[3] >> 4) << 6);
    if (e % 16 == 0) w.sc[at / 16] = int8_t(blk[192 + (e % 256) / 16]);
    if (e % 256 == 0) w.d[at / 256] = sycl::bit_cast<sycl::half>(uint16_t(blk[208] | blk[209] << 8));
  }).wait();
  return w;
}

void free_q6(sycl::queue& q, Q6& w) {
  sycl::free(w.lo, q);
  w = Q6{};
}

void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out) {
  constexpr int kWg = 256;
  q.parallel_for(sycl::nd_range<1>(size_t(rows * kWg), kWg), [=](sycl::nd_item<1> it) {
    const int64_t r = int64_t(it.get_group(0));
    const int lid = int(it.get_local_id(0));
    const float* xr = x + r * cols;
    float ss = 0;
    for (int64_t i = lid; i < cols; i += kWg) ss += xr[i] * xr[i];
    ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
    const float inv = 1.0f / sycl::sqrt(ss / float(cols) + eps);
    for (int64_t i = lid; i < cols; i += kWg) out[r * cols + i] = xr[i] * inv * (w ? w[i] : 1.0f);
  });
}

void matmul_q6(sycl::queue& q, const Q6& w, const float* x, int64_t rows, float* out) {
  // One sub-group per output column, 16 sub-groups per work-group. Per 256-element super-block
  // each lane takes 16 consecutive elements: 8 bytes of low nibbles, 4 of high bits, one scale,
  // for up to 8 input rows per pass, then the sub-group sums its lanes.
  constexpr int kSg = 16, kRows = 8, kWg = 256;
  const Q6 ww = w;
  if (ww.n % (kWg / kSg)) throw std::runtime_error("fast::matmul_q6: rows not a multiple of 16");
  for (int64_t r0 = 0; r0 < rows; r0 += kRows) {
    const int64_t rc = std::min<int64_t>(kRows, rows - r0);
    q.parallel_for(sycl::nd_range<1>(size_t(ww.n * kSg), kWg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
      const int64_t v = int64_t(it.get_global_id(0)) / kSg;
      const int lane = int(it.get_sub_group().get_local_id()[0]);
      const int64_t k = ww.k;
      float acc[kRows] = {};
      for (int64_t b = 0; b < k / 256; ++b) {
        const int64_t e = v * k + b * 256 + lane * 16;
        const sycl::uint2 lo = *reinterpret_cast<const sycl::uint2*>(ww.lo + e / 2);
        const uint32_t hi = *reinterpret_cast<const uint32_t*>(ww.hi + e / 4);
        const float s = float(ww.d[e / 256]) * float(ww.sc[e / 16]);
        float wv[16];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          const uint32_t l = j < 8 ? lo.x() : lo.y();
          wv[j] = float(int(((l >> (4 * (j % 8))) & 15) | ((hi >> (2 * j)) & 3) << 4) - 32);
        }
#pragma unroll
        for (int r = 0; r < kRows; ++r) {
          if (r >= rc) break;
          const sycl::float4* xr = reinterpret_cast<const sycl::float4*>(x + (r0 + r) * k + b * 256 + lane * 16);
          float dot = 0;
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const sycl::float4 xv = xr[c];
            dot += xv.x() * wv[4 * c] + xv.y() * wv[4 * c + 1] + xv.z() * wv[4 * c + 2] + xv.w() * wv[4 * c + 3];
          }
          acc[r] += dot * s;
        }
      }
      auto sg = it.get_sub_group();
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        if (r >= rc) break;
        const float sum = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
        if (lane == 0) out[(r0 + r) * ww.n + v] = sum;
      }
    });
  }
}

void matmul_f32(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out) {
  constexpr int kTok = 4, kChunk = 256;  // tokens per work-group; floats of K per staged chunk
  if (k % 4 || n > 256 || n % 32) throw std::runtime_error("fast::matmul_f32: unsupported shape");
  const int32_t K = int32_t(k), N = int32_t(n), M = int32_t(n_tok);
  const int32_t groups = (M + kTok - 1) / kTok;
  // One work-group per 4 tokens, one work-item per output column: the tokens' chunk of K is staged in
  // shared memory and read by all the columns, each item reading its weight row in float4s.
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> xs(sycl::range<1>(kTok * kChunk), h);
    h.parallel_for(sycl::nd_range<1>(size_t(groups) * size_t(N), size_t(N)), [=](sycl::nd_item<1> it) {
      const int32_t g = int32_t(it.get_group(0)), c = int32_t(it.get_local_id(0));
      const sycl::float4* wr = reinterpret_cast<const sycl::float4*>(w + int64_t(c) * K);
      float acc[kTok] = {};
      for (int32_t k0 = 0; k0 < K; k0 += kChunk) {
        const int32_t len = sycl::min<int32_t>(kChunk, K - k0);
        for (int32_t i = c; i < kTok * kChunk; i += N) {
          const int32_t t = g * kTok + i / kChunk, kk = i % kChunk;
          xs[i] = t < M && kk < len ? x[int64_t(t) * K + k0 + kk] : 0.0f;
        }
        sycl::group_barrier(it.get_group());
        for (int32_t kk = 0; kk < len; kk += 4) {
          const sycl::float4 wv = wr[(k0 + kk) / 4];
          for (int t = 0; t < kTok; ++t) {
            const sycl::float4 xv = *reinterpret_cast<const sycl::float4*>(&xs[t * kChunk + kk]);
            acc[t] += wv.x() * xv.x() + wv.y() * xv.y() + wv.z() * xv.z() + wv.w() * xv.w();
          }
        }
        sycl::group_barrier(it.get_group());
      }
      for (int t = 0; t < kTok; ++t)
        if (g * kTok + t < M) out[int64_t(g * kTok + t) * N + c] = acc[t];
    });
  });
}

}  // namespace eklow::fast
