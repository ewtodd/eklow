#include "kernels/fast.hpp"

#include <stdexcept>
#include <string>

namespace eklow::fast {

namespace {

inline float half_at(const uint8_t* p) { return float(sycl::bit_cast<sycl::half>(uint16_t(p[0] | p[1] << 8))); }

// One Q6_K weight (block of 256: ql[128], qh[64], int8 scales[16], fp16 d), as ggml lays it out.
inline float q6k_weight(const uint8_t* blk, int i) {
  const int half = i / 128, r = i % 128, l = r % 32, quarter = r / 32;
  const uint8_t* ql = blk + half * 64;
  const uint8_t* qh = blk + 128 + half * 32;
  const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + half * 8;
  int q;
  switch (quarter) {
    case 0: q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); break;
    case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); break;
    case 2: q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); break;
    default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); break;
  }
  return half_at(blk + 208) * float(sc[l / 16 + 2 * quarter]) * float(q - 32);
}

}  // namespace

void embed(sycl::queue& q, const ref::Weight& table, const int32_t* tokens, int64_t n_tok, float scale, float* out) {
  const ref::Weight w = table;
  if (w.type != gguf::Type::Q6_K && w.type != gguf::Type::F16 && w.type != gguf::Type::F32)
    throw std::runtime_error(std::string("fast::embed: no kernel for ") + gguf::type_name(w.type));
  q.parallel_for(sycl::range<2>(size_t(n_tok), size_t(w.k)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
    const uint8_t* row = w.data + uint64_t(tokens[t]) * w.row_bytes;
    float v;
    if (w.type == gguf::Type::Q6_K) v = q6k_weight(row + (i / 256) * 210, int(i % 256));
    else if (w.type == gguf::Type::F16) v = half_at(row + 2 * i);
    else v = reinterpret_cast<const float*>(row)[i];
    out[t * w.k + i] = v * scale;
  });
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

void rope_neox(sycl::queue& q, float* x, const int32_t* pos, int64_t n_tok, int64_t n_head, int64_t head_dim,
               int64_t n_rot, const float* inv_freq) {
  const int64_t half = n_rot / 2;
  q.parallel_for(sycl::range<3>(size_t(n_tok), size_t(n_head), size_t(half)), [=](sycl::id<3> id) {
    const int64_t t = int64_t(id[0]), h = int64_t(id[1]), i = int64_t(id[2]);
    const float theta = float(pos[t]) * inv_freq[i];
    const float c = sycl::cos(theta), s = sycl::sin(theta);
    float* v = x + (t * n_head + h) * head_dim;
    const float a = v[i], b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  });
}

void gelu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
    const float g = gate[id[0]];
    out[id[0]] = 0.5f * g * (1.0f + sycl::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))) * up[id[0]];
  });
}

void add_scale(sycl::queue& q, const float* a, const float* b, float s, int64_t n, float* out) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { out[id[0]] = (a[id[0]] + b[id[0]]) * s; });
}

void softcap(sycl::queue& q, float* x, float cap, int64_t n) {
  q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) { x[id[0]] = cap * sycl::tanh(x[id[0]] / cap); });
}

void matmul_q6k(sycl::queue& q, const ref::Weight& w, const float* x, int64_t rows, float* out) {
  if (w.type != gguf::Type::Q6_K || w.k % 256) throw std::runtime_error("fast::matmul_q6k needs a Q6_K matrix");
  // One sub-group per output column; each lane takes Q6_K blocks of the row in turn and keeps a
  // partial sum per input row (up to 8 per pass), reduced across the sub-group at the end.
  constexpr int kSg = 16, kRows = 8;
  const ref::Weight ww = w;
  for (int64_t r0 = 0; r0 < rows; r0 += kRows) {
    const int64_t rc = std::min<int64_t>(kRows, rows - r0);
    q.parallel_for(sycl::nd_range<1>(size_t(ww.n * kSg), kSg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
      const int64_t v = int64_t(it.get_group(0));
      const int lane = int(it.get_local_id(0));
      const uint8_t* row = ww.data + uint64_t(v) * ww.row_bytes;
      float acc[kRows] = {};
      for (int64_t b = lane; b < ww.k / 256; b += kSg) {
        const uint8_t* blk = row + b * 210;
        for (int i = 0; i < 256; ++i) {
          const float wv = q6k_weight(blk, i);
#pragma unroll
          for (int r = 0; r < kRows; ++r)
            if (r < rc) acc[r] += wv * x[(r0 + r) * ww.k + b * 256 + i];
        }
      }
      auto sg = it.get_sub_group();
#pragma unroll
      for (int r = 0; r < kRows; ++r) {
        if (r >= rc) break;
        const float s = sycl::reduce_over_group(sg, acc[r], sycl::plus<float>());
        if (lane == 0) out[(r0 + r) * ww.n + v] = s;
      }
    });
  }
}

}  // namespace eklow::fast
