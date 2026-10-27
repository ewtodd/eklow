// XMX matrix products for Q4_0 weights, in two forms (DECISIONS 2026-10-24, "XMX", "fp16 GEMM"):
//   up to kFp16Rows rows (decode): activations quantized to int8 per 32-element block (as
//     llama.cpp's MMQ does), weights fed to dpas as signed 4-bit (a Q4_0 nibble q is the s4 value
//     q - 8 with bit 3 flipped), each K=32 dpas one Q4_0 block with both block scales applied to its
//     int32 result: least memory traffic, but the scaling costs the vector units about as much as
//     the dpas;
//   more rows (prefill): fp16 activations; each work-group dequantizes a K block's weight tiles to
//     fp16 once in shared local memory for all its rows, and the dpas accumulate across all of K.
//
// Layouts, all written by eklow so that every load is contiguous:
//   weights: tiles of 8 output columns x 32 K, [N/8][K/32] (column-group major), 128 bytes each, in
//     the dpas B layout for s8 x s4: element (k, n) is nibble ((k/8)*8 + n)*8 + k%8, low nibble first;
//     plus fp16 scales [N/8][K/32][8];
//   int8 activations: tiles of 8 rows x 32 K, [M_pad/8][K/32], 256 bytes each, row-major (the dpas
//     A layout); plus fp32 scales [M_pad/8][K/32][8];
//   fp16 activations: tiles of 8 rows x 16 K, [M_pad/8][K/16], 256 bytes each, row-major.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

#include "kernels/ref.hpp"

namespace eklow::xmx {

struct Weight {
  int64_t k = 0, n = 0;
  uint8_t* tiles = nullptr;
  sycl::half* scales = nullptr;
};

// Repacks a Q4_0 matrix (ggml layout, on the device) into a new allocation; the caller frees both
// with free_weight / its own allocator. K must be a multiple of 32 and N of 8. With interleave,
// the two halves of raw's rows (stacked gate and up projections) alternate in tiles of 8 columns:
// output columns 16i..16i+7 are rows 8i..8i+7 of the first half and 16i+8..16i+15 the same rows of
// the second, so that one thread of a matmul holds a gate tile and its up tile.
Weight repack_q4_0(sycl::queue& q, const ref::Weight& raw, bool interleave = false);
void free_weight(sycl::queue& q, Weight& w);

// Rows above which matmul takes fp16 activations.
constexpr int64_t kFp16Rows = 32;

struct Activations {
  int64_t m = 0, m_pad = 0, k = 0;  // m_pad: m rounded up to 32 (int8) or 256 (fp16)
  int64_t capacity = 0;              // elements allocated in each form: any m_pad x k up to it fits
  bool fp16 = false;                 // the form the last producer wrote
  int8_t* q = nullptr;
  float* s = nullptr;
  sycl::half* h = nullptr;
};
// Buffers for up to `max_m` rows of `k`.
Activations alloc_activations(sycl::queue& q, int64_t max_m, int64_t k);
void free_activations(sycl::queue& q, Activations& a);

// Writes x[m][k] (fp32, row-major) into `a` in the form matmul takes for m rows: int8 per row and
// 32-block, scale = max|x| / 127, q = round(x / scale); or fp16.
void quantize(sycl::queue& q, const float* x, int64_t m, Activations& a);

// For kernels that produce activations themselves: where element (row, k) of a matrix with kb
// 32-blocks per row goes, where its block's scale goes, and the quantization rule applied to a run
// of values sharing one block (amax: the block's max |x|); and where it goes in fp16 (kh: 16-element
// runs per row).
inline int64_t act_index(int64_t row, int64_t k, int64_t kb) { return ((row / 8) * kb + k / 32) * 256 + (row % 8) * 32 + k % 32; }
inline int64_t act16_index(int64_t row, int64_t k, int64_t kh) { return ((row / 8) * kh + k / 16) * 128 + (row % 8) * 16 + k % 16; }
inline int64_t act_scale_index(int64_t row, int64_t k, int64_t kb) { return ((row / 8) * kb + k / 32) * 8 + row % 8; }
template <int N>
inline float quantize_run(const float (&x)[N], float amax, int8_t* dst) {
  const float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
  for (int j = 0; j < N; ++j) dst[j] = int8_t(sycl::round(x[j] * id));
  return d;
}
// Marks `a` as holding m rows of width k, in the form for m rows (a.fp16 says which), for a kernel
// about to write them that way. Rows from m to the padded count keep whatever they held; matmul
// never stores their results.
void set_rows(Activations& a, int64_t m, int64_t k);

// out[m][n] = a . w^T, fp32, row-major. Up to 256 int8 rows run K-split GEMV schedules; fp16
// activations the shared-memory GEMM (n a multiple of 128). quantize and matmul only enqueue (the
// queue must be in order); the caller waits.
void matmul(sycl::queue& q, const Activations& a, const Weight& w, float* out);

// GeGLU: gelu_tanh(gate) * up for w a gate|up matrix repacked with interleave, into `out` (n/2
// wide) in the form the next matmul takes. On fp16 activations it is the GEMM's epilogue; the
// int8 path goes through `scratch` (fp32, m x n).
void matmul_geglu(sycl::queue& q, const Activations& a, const Weight& w, Activations& out, float* scratch);

// A layer's experts of one shape, as device tables of their tile and scale pointers (the experts
// stay separate xmx::Weights; the grouped kernels index the tables by expert id).
struct ExpertTable {
  int64_t k = 0, n = 0, count = 0;
  const uint8_t** tiles = nullptr;
  const sycl::half** scales = nullptr;
};
ExpertTable make_expert_table(sycl::queue& q, const std::vector<Weight>& experts);
void free_expert_table(sycl::queue& q, ExpertTable& t);

// Grouped one-row GEMVs on int8 activations: for pairs p < P, out[p][:] = W_{ids[p]} * a[p / row_div][:]
// (ids on the device). out is [P][n], fp32.
void matmul_grouped(sycl::queue& q, const Activations& a, const ExpertTable& w, const int32_t* ids, int64_t P,
                    int64_t row_div, float* out);
// The grouped gate_up (interleaved experts) then GeGLU into P-row int8 `out`, through `scratch` [P][n].
void matmul_geglu_grouped(sycl::queue& q, const Activations& a, const ExpertTable& w, const int32_t* ids, int64_t P,
                          int64_t row_div, Activations& out, float* scratch);

}  // namespace eklow::xmx
