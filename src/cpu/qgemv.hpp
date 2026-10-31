// GEMVs of the Qwen weight formats on the host CPU, for the cold experts (DECISIONS 2026-10-31): Q8_0, Q4_K, Q5_K,
// Q6_K and IQ4_XS beside q4_gemv.hpp's Q4_0, on ggml's own bytes (a matrix is n rows of k, each row k / block blocks,
// read straight from the GGUF mapping or a pinned copy) and the same int8 activations (cpu::Q8Blocks: per 32-block
// scale and sum), so that the card's decode kernels (src/kernels/xmx_fmt.hpp) and these compute the same integer dots.
// AVX-VNNI's _mm256_dpbusd_avx_epi32 takes unsigned bytes against signed: the 4- and 5-bit values of the K-quants
// (unsigned) go in as they are and their minimum comes off with the activations' block sum; Q8_0, Q6_K (q - 32) and
// IQ4_XS (a pshufb table lookup) are signed and go through |w| x sign(a, w).
#pragma once

#include <cstdint>

#include "cpu/q4_gemv.hpp"

namespace eklow::cpu {

enum class QType { Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS };

const char* qtype_name(QType t);
int qtype_block(QType t);        // weights per block (32 or 256)
int qtype_block_bytes(QType t);  // bytes per block
inline uint64_t qrow_bytes(QType t, int64_t k) { return uint64_t(k / qtype_block(t)) * uint64_t(qtype_block_bytes(t)); }

// y[r] = dot(x, rows[r]) for r < n; k a multiple of the block. Rows split over the pool in grains of 16.
void gemv_q(QType t, const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, Pool& pool);
// The same on one thread for rows [lo, hi).
void gemv_q_range(QType t, const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks& x, float* y);

// Rows [lo, hi) against nx inputs at once (in groups of four): y[j][r] = dot(x[j], rows[r]). Each weight row is read
// and unpacked once per group of four, so the tokens of a verify block that pick one cold expert share its read.
void gemv_q_multi(QType t, const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks* const* x, int nx, float* const* y);

// The GLU of gate and up rows stacked [gate (f); up (f)] in gu[f * 2] into h[f]: silu(gate) * up or gelu_tanh(gate) * up.
enum class CpuGlu { Silu, Gelu };
void glu(const float* gu, int64_t f, CpuGlu act, float* h);

}  // namespace eklow::cpu
