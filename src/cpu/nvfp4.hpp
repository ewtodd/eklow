// NVFP4 GEMVs on the host CPU (ggml type 40, as Inkling 2 Mini's layers 0-19 store their experts): 64 weights in 36
// bytes, 4 UE4M3 scale bytes (one a 16-weight sub-block) then 32 bytes of E2M1 codes; in sub-block s byte j's low
// nibble is weight j and its high nibble weight j + 8. The GEMV reads those bytes as they are stored.
//
// Weights: a pshufb lookup turns the 32 code bytes' nibbles into the doubled E2M1 values 0, 1, 2, 3, 4, 6, 8, 12 with
// sign, plus 12 (unsigned bytes 0..24, so that AVX-VNNI's unsigned x signed _mm256_dpbusd_avx_epi32 takes them as
// they are): the low nibbles of the 32 bytes are 4 sub-blocks' weights 0..7 and the high nibbles their weights 8..15,
// so two dpbusd (low half, high half) against an activation vector in that order give 8 int32 lanes, each of 4 weights
// of one sub-block (lanes 2s, 2s + 1 of sub-block s). The offset 12 comes off through the accumulator's start value
// (-12 x the lane's activation sum, precomputed), and each lane is scaled by (the sub-block's scale / 2) x the
// activation's scale in fp32.
//
// Activations: int8 with a scale per 16 (the weights' sub-block, not Q8Blocks' 32), stored in that lane order.
#pragma once

#include <cstdint>
#include <vector>

#include "cpu/pool.hpp"

namespace eklow::cpu {

struct NvBlocks {
  int64_t k = 0;
  std::vector<int8_t> q;       // k values; per 64-block: bytes 0..31 the sub-blocks' values 0..7, 32..63 their values 8..15
  std::vector<float> s;        // k / 8: per 64-block 8 lane scales (the sub-block's activation scale twice each)
  std::vector<int32_t> init;   // k / 8: per 64-block 8 lanes, -12 x the sum of the lane's 8 values (4 low, 4 high)
};

// Quantizes k floats (a multiple of 64) per 16-block: s = max|x| / 127, q = round(x / s), s = 0 gives q = 0.
void nv_quantize_into(const float* x, int64_t k, NvBlocks& r);
NvBlocks nv_quantize(const float* x, int64_t k);

constexpr int kNvBlock = 64, kNvBlockBytes = 36;
inline uint64_t nv_row_bytes(int64_t k) { return uint64_t(k / kNvBlock) * kNvBlockBytes; }

// Rows [lo, hi) of an NVFP4 matrix (row 0 at `rows`) against nx inputs at once (groups of four, then 3, 2, 1): y[j][r] =
// dot(x[j], row r) with the weights as ggml decodes them (the per-expert scale not applied). One thread.
void gemv_nvfp4_multi(const uint8_t* rows, int64_t k, int64_t lo, int64_t hi, const NvBlocks* const* x, int nx, float* const* y);

// y[r] = dot(x, rows[r]) for r < n, the rows split over the pool in grains of 16.
void gemv_nvfp4(const uint8_t* rows, int64_t k, int64_t n, const NvBlocks& x, float* y, Pool& pool);

}  // namespace eklow::cpu
