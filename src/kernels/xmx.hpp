// XMX matrix products for Q4_0 weights: activations quantized to int8 per 32-element block (as
// llama.cpp's MMQ does), weights fed to dpas as signed 4-bit (a Q4_0 nibble q is the s4 value q - 8
// with bit 3 flipped), each K=32 dpas one Q4_0 block, the activation and weight block scales applied
// to its int32 result (DECISIONS 2026-10-24, "XMX").
//
// Layouts, both written by eklow so that every load is contiguous:
//   weights: tiles of 8 output columns x 32 K, [N/8][K/32], 128 bytes each, in the dpas B layout
//     for s8 x s4: element (k, n) is nibble ((k/8)*8 + n)*8 + k%8, low nibble first; plus fp16
//     scales [N/8][K/32][8];
//   activations: tiles of 8 rows x 32 K, [M_pad/8][K/32], 256 bytes each, row-major (the dpas A
//     layout); plus fp32 scales [M_pad/8][K/32][8]. Rows past M are zero.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/ref.hpp"

namespace eklow::xmx {

struct Weight {
  int64_t k = 0, n = 0;
  uint8_t* tiles = nullptr;
  sycl::half* scales = nullptr;
};

// Repacks a Q4_0 matrix (ggml layout, on the device) into a new allocation; the caller frees both
// with free_weight / its own allocator. K must be a multiple of 32 and N of 8.
Weight repack_q4_0(sycl::queue& q, const ref::Weight& raw);
void free_weight(sycl::queue& q, Weight& w);

struct Activations {
  int64_t m = 0, m_pad = 0, k = 0;  // m_pad: m rounded up to 32
  int64_t capacity = 0;              // int8 elements allocated: any m_pad x k up to it fits
  int8_t* q = nullptr;
  float* s = nullptr;
};
// Buffers for up to `max_m` rows of `k`.
Activations alloc_activations(sycl::queue& q, int64_t max_m, int64_t k);
void free_activations(sycl::queue& q, Activations& a);

// Quantizes x[m][k] (fp32, row-major) into `a` (sized for at least m rows): per row and 32-block,
// scale = max|x| / 127, q = round(x / scale).
void quantize(sycl::queue& q, const float* x, int64_t m, Activations& a);

// out[m][n] = sum over blocks of (a block . w block) * a.scale * w.scale, fp32, row-major.
// Small m (decode) runs a K-split GEMV schedule, larger m a tiled GEMM. quantize and matmul only
// enqueue (the queue must be in order); the caller waits.
void matmul(sycl::queue& q, const Activations& a, const Weight& w, float* out);

}  // namespace eklow::xmx
