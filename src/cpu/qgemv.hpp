// GEMVs of the GGUF formats Inkling 2 Mini's layers 20-23 keep their experts in, on the host CPU for the cold experts
// (DECISIONS 2026-10-31): Q8_0 and Q5_K, on ggml's own bytes (a matrix is n rows of k, each row k / block blocks, read
// straight from the GGUF mapping or a pinned copy) and int8 activations (cpu::Q8Blocks: per 32-block scale and sum).
// AVX-VNNI's _mm256_dpbusd_avx_epi32 takes unsigned bytes against signed: Q5_K's 5-bit values (unsigned) go in as they
// are and their minimum comes off with the activations' block sum; Q8_0 is signed and goes through |w| x sign(a, w).
// NVFP4 is in nvfp4.hpp, with its own activation layout.
#pragma once

#include <cstdint>

#include "cpu/pool.hpp"

namespace eklow::cpu {

enum class QType { Q8_0, Q5_K, NVFP4 };

const char* qtype_name(QType t);
int qtype_block(QType t);        // weights per block (32, 256 or 64)
int qtype_block_bytes(QType t);  // bytes per block (34, 176 or 36)
inline uint64_t qrow_bytes(QType t, int64_t k) { return uint64_t(k / qtype_block(t)) * uint64_t(qtype_block_bytes(t)); }

// Rows [lo, hi) of a Q8_0 or Q5_K matrix against nx inputs at once (in groups of four, then 3, 2 or 1): y[j][r] =
// dot(x[j], rows[r]). `rows` points at row 0; each weight row is read and unpacked once per group, so the tokens of a
// prefill chunk that pick one cold expert share its read. One thread; split rows over a Pool yourself (grains of 16).
void gemv_q_multi(QType t, const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const Q8Blocks* const* x, int nx, float* const* y);

// y[r] = dot(x, rows[r]) for r < n, the rows split over the pool in grains of 16.
void gemv_q(QType t, const uint8_t* rows, int64_t k, int64_t n, const Q8Blocks& x, float* y, Pool& pool);

}  // namespace eklow::cpu
