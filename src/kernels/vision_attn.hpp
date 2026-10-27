// Dense bidirectional attention for the vision encoders (heads of 72 and 64 dims, one K/V head per query head, up to
// ~10,000 rows), on the matrix units. The shape of attention.cpp's long-prefill path (DECISIONS 2026-10-27, "global
// prefill attention as two GEMMs"), cut for narrow heads and without the paged cache: per head, the keys in segments
// of 2,048; each segment Q.K^T (256 rows x 128 keys a work-group, the epilogue taking each row's maximum and
// writing P = exp(S - m) in fp16 with the 128-key sums), a combining step, then P.V into a running fp32 output.
//
// Operands are dpas tiles written by the caller (the encoder's prep kernel), per head h, with D the head size
// padded to a multiple of 32 (72 -> 96, zeros in the padding) and R = rows rounded up to 256 (zero rows):
//   Q: A operand tiles of 8 rows x 16 dims, [R/8][D/16][8 rows][16]                       (R * D halves a head)
//   K: B operand tiles of 8 keys x 16 dims, [R/8][D/16], element (dim d, key n) at (d/2)*16 + n*2 + d%2
//   V: B operand tiles of 16 keys x 8 dims, [R/16][D/8], element (key k, dim n) at (k/2)*16 + n*2 + k%2
// (q_index, k_index, v_index below). Keys at or past n are masked; the softmax scale is folded into Q.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/xmx.hpp"

namespace eklow::vattn {

constexpr int64_t kSeg = 2048;

inline int64_t rows_padded(int64_t n) { return (n + 255) / 256 * 256; }
inline int64_t q_index(int64_t row, int64_t d, int64_t D) { return ((row / 8) * (D / 16) + d / 16) * 128 + (row % 8) * 16 + d % 16; }
inline int64_t k_index(int64_t key, int64_t d, int64_t D) { return ((key / 8) * (D / 16) + d / 16) * 128 + ((d % 16) / 2) * 16 + (key % 8) * 2 + d % 2; }
inline int64_t v_index(int64_t key, int64_t d, int64_t D) { return ((key / 16) * (D / 8) + d / 8) * 128 + ((key % 16) / 2) * 16 + (d % 8) * 2 + key % 2; }

// Device buffers reused across calls.
struct Scratch {
  sycl::half* p = nullptr;  // probabilities of a segment, fp16 A tiles
  float* stat = nullptr;    // per-row statistics and the running output
  size_t p_halves = 0, stat_floats = 0;
};
// Bytes reserve() allocates for up to `rows` rows of head size D and n_head heads (the heads go through together in
// groups, as many as fit a 64 MB probability buffer).
uint64_t scratch_bytes(int64_t rows, int64_t D, int64_t n_head);
void reserve(sycl::queue& q, Scratch& s, int64_t rows, int64_t D, int64_t n_head);
void release(sycl::queue& q, Scratch& s);

// out (fp16 activation tiles, xmx::act16_index with row width `out_k`) element (row, h * d_real + d) = softmax(Q_h K_h^T)
// V_h for rows < n, d < d_real. D is 64 or 96. Only enqueues.
void attend(sycl::queue& q, const sycl::half* qt, const sycl::half* kt, const sycl::half* vt, int64_t n, int64_t n_head, int64_t D,
            int64_t d_real, Scratch& s, sycl::half* out, int64_t out_k);

}  // namespace eklow::vattn
