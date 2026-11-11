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

// The weight formats. Q4_0 (every path takes it); Q8_0 and F16 (from F16, BF16 or F32 files)
// only the fp16 GEMM, for EmbeddingGemma 2 and the vision encoders. Tiles of 8 columns x 32 K in the same
// [N/8][K/32] order: Q8_0 256 bytes of int8 laid out as the fp16 B operand (each 16-K half, element (k, n) at
// (k/2)*16 + n*2 + k%2) with an fp16 scale per column; F16 512 bytes, the B operand itself, no scales.
//
// The Qwen formats (DECISIONS 2026-10-31, "the matmuls"), all decode GEMVs on int8 activations and fp16 GEMMs, each
// with the bytes per weight of ggml's own and the same values (src/kernels/xmx_fmt.hpp has the layouts): Q8_0I
// (int8 tiles for the s8 dpas, 256 bytes, fp16 scales per tile as Q4_0's: 8.5 bits), Q4_K (4.5), Q5_K (5.5), Q6_K
// (6.5625), IQ4_XS (4.25). The K-quants and IQ4_XS need K a multiple of 256; `scales` holds their per-superblock
// records (the sub-block scales and mins, the high bits, d) as bytes.
enum class WeightFormat : uint8_t { Q4_0, Q8_0, F16, Q8_0I, Q4_K, Q5_K, Q6_K, IQ4_XS };

struct Weight {
  int64_t k = 0, n = 0;
  uint8_t* tiles = nullptr;
  sycl::half* scales = nullptr;  // null for F16
  WeightFormat format = WeightFormat::Q4_0;
};

// Repacks a Q4_0 matrix (ggml layout, on the device) into a new allocation; the caller frees both
// with free_weight / its own allocator. K must be a multiple of 32 and N of 8. With interleave,
// the two halves of raw's rows (stacked gate and up projections) alternate in tiles of 8 columns:
// output columns 16i..16i+7 are rows 8i..8i+7 of the first half and 16i+8..16i+15 the same rows of
// the second, so that one thread of a matmul holds a gate tile and its up tile.
Weight repack_q4_0(sycl::queue& q, const ref::Weight& raw, bool interleave = false);
// The bytes of a repacked matrix: the tiles and the fp16 scales (each a multiple of 128).
uint64_t weight_tile_bytes(int64_t k, int64_t n);
uint64_t weight_scale_bytes(int64_t k, int64_t n);
// The same into tiles and scales the caller owns (w.k, w.n set; sizes as above); only enqueues, so `raw` must
// stay put until the queue has run it. Streamed experts are repacked this way, from a raw staging buffer.
void repack_q4_0_into(sycl::queue& q, const ref::Weight& raw, bool interleave, const Weight& w);
// Repacks a Q4_0, Q8_0, F16, BF16 or F32 matrix (on the device) into a new allocation, Q4_0 as repack_q4_0 does,
// Q8_0 as Q8_0, the rest as F16. K a multiple of 32, N of 8.
Weight repack(sycl::queue& q, const ref::Weight& raw);
// Rows [n0, n0 + n) of w (output columns), a view into its tiles; n0 and n a multiple of 8.
Weight rows_of(const Weight& w, int64_t n0, int64_t n);
void free_weight(sycl::queue& q, Weight& w);

// The Qwen formats' repack (Q8_0 as Q8_0I, Q4_K, Q5_K, Q6_K, IQ4_XS: ggml tensor types, on the device) into the
// layouts of src/kernels/xmx_fmt.hpp, the same bytes per weight as ggml's. K a multiple of 32 (Q8_0I) or 256
// (the rest), N of 8; interleave as repack_q4_0's. format_of is false for types without such a layout.
bool format_of(gguf::Type t, WeightFormat& f);
const char* format_name(WeightFormat f);
// The bytes of a repacked matrix's tiles and of its `scales` array (the fp16 scales of Q4_0 and Q8_0I, the
// superblock records of the rest).
uint64_t weight_tile_bytes(WeightFormat f, int64_t k, int64_t n);
uint64_t weight_meta_bytes(WeightFormat f, int64_t k, int64_t n);
Weight repack_format(sycl::queue& q, const ref::Weight& raw, WeightFormat f, bool interleave = false);
// Into allocations the caller owns (w.k, w.n, w.format set; sizes as above); only enqueues.
void repack_format_into(sycl::queue& q, const ref::Weight& raw, bool interleave, const Weight& w);
// `count` matrices at once (experts in slots): matrix x reads raw.data + x * raw_stride and writes w.tiles + x * tile_stride, w.scales (bytes) + x * meta_stride. K-quant formats.
void repack_format_batch_into(sycl::queue& q, const ref::Weight& raw, bool interleave, const Weight& w, int64_t count, uint64_t raw_stride,
                              uint64_t tile_stride, uint64_t meta_stride);

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
// fp16 forces the fp16 form whatever m is (the Q8_0 and F16 weights have only the fp16 GEMM).
void quantize(sycl::queue& q, const float* x, int64_t m, Activations& a, bool fp16 = false);

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
void set_rows(Activations& a, int64_t m, int64_t k, bool fp16 = false);

// out[m][n] = a . w^T, fp32, row-major. Up to 256 int8 rows run K-split GEMV schedules; fp16
// activations the shared-memory GEMM (n a multiple of 128). quantize and matmul only enqueue (the
// queue must be in order); the caller waits.
void matmul(sycl::queue& q, const Activations& a, const Weight& w, float* out);

// The activation of the fused GLU kernels: the GeGLU (gelu_tanh(gate) * up) or Qwen's SwiGLU (silu(gate) * up).
// The kernels named geglu serve both; the default is GeGLU, so existing callers are unchanged.
enum class Glu : uint8_t { Gelu, Silu };

// GeGLU: gelu_tanh(gate) * up for w a gate|up matrix repacked with interleave, into `out` (n/2
// wide) in the form the next matmul takes. On fp16 activations it is the GEMM's epilogue; the
// int8 path goes through `scratch` (fp32, m x n).
void matmul_geglu(sycl::queue& q, const Activations& a, const Weight& w, Activations& out, float* scratch, Glu act = Glu::Gelu);

}  // namespace eklow::xmx
