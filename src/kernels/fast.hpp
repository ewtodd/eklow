// The serving kernels other than the matmuls (kernels/xmx.hpp), shared by the engines: per-token work over a batch
// of tokens from several sequences. They only enqueue; the queue must be in order. They are checked
// end to end, through the engine's dumps against the oracle's (eklow-engine-run).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"

namespace eklow::fast {

// A Q6_K matrix (the tied embedding and output head) split into aligned arrays, each [n][...]:
// low 4 bits of element e in byte e/2 (low nibble for even e), its top 2 bits at bit 2*(e%4) of
// byte e/4, Q6_K's int8 scale per 16 elements and fp16 scale per 256. The same 6.5625 bits per
// weight and the same values as Q6_K, but every lane's loads are aligned and contiguous.
struct Q6 {
  int64_t k = 0, n = 0;
  uint8_t *lo = nullptr, *hi = nullptr;
  int8_t* sc = nullptr;
  sycl::half* d = nullptr;
};
// From a Q6_K matrix in device memory; one allocation, freed with free_q6.
Q6 repack_q6k(sycl::queue& q, const ref::Weight& raw);
void free_q6(sycl::queue& q, Q6& w);

// out[r][:] = x[r][:] / sqrt(mean(x[r]^2) + eps) * (w ? w : 1); out may be x.
void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out);

// out[r][v] = x[r] . W[v], r < rows.
void matmul_q6(sycl::queue& q, const Q6& w, const float* x, int64_t rows, float* out);

// The MoE router, per token: softmax of logits[t][n_expert], the top n_used, their weights
// renormalized by the sum (floored at 2^-14) and multiplied by each expert's down scale (on the
// device). ids[t][n_used] int32 and weights[t][n_used] fp32, by descending probability. n_expert a
// multiple of 32, at most 256; n_used at most 16.
void moe_route(sycl::queue& q, const float* logits, int64_t n_tok, int64_t n_expert, int64_t n_used, const float* down_scale,
               int32_t* ids, float* weights, int64_t ld = 0);  // ld: floats between tokens' logits (0: n_expert); down_scale may be null (1)
// out[t][:] = sum_i weights[t][i] * y[row(t, i)][:], summed in the order i = 0..n_used-1; row(t, i)
// is t * n_used + i, or rows[t * n_used + i] when given (the grouped GEMM's sorted order), where a
// negative row is a pair left out (a cold expert). `extra`, when given, holds the cold experts' share
// (already weighted, from the host) in rows [extra_row[t]][cols], added to token t's sum unless
// extra_row[t] is negative.
void moe_combine(sycl::queue& q, const float* y, const float* weights, int64_t n_tok, int64_t n_used, int64_t cols, float* out,
                 const int32_t* rows = nullptr, const float* extra = nullptr, const int32_t* extra_row = nullptr);
// Decode with cold experts: one kernel that copies the routing (ids [n_ids]) and the experts' input
// row (the first 8-row tile of int8 activations, a_bytes, and its scales, s_n floats) into pinned host
// memory, so the host needs one event instead of three copies' worth of launches and waits.
sycl::event host_pack(sycl::queue& q, const int32_t* ids, int64_t n_ids, const int8_t* a, int64_t a_bytes, const float* s, int64_t s_n,
                      int32_t* h_ids, int8_t* h_a, float* h_s);
// The cold pairs' output rows from pinned host memory: row k of host_y [n][cols] becomes row pair[k] of y.
struct PairList {
  int32_t pair[32];
};
void cold_scatter(sycl::queue& q, const float* host_y, const PairList& pairs, int64_t n, int64_t cols, float* y);
// Row-major int8 for the host: out[r][cols] and one fp32 scale per 32-block, s[r][cols / 32] = max|x| / 127,
// q = round(x / s), as xmx::quantize_run does it.
void quantize_rows(sycl::queue& q, const float* x, int64_t rows, int64_t cols, int8_t* out, float* scale);

// Prefill's grouping of the (token, slot) pairs p = t * n_used + i by expert (ids as moe_route wrote
// them), a stable counting sort so that results do not depend on scheduling: expert e's pairs, in
// token order, fill rows from its segment start, and segments start at multiples of `tile` rows.
//   tiles: [0] = number of row tiles, then from tiles[4] four ints a tile, expert by expert: expert,
//     first row, rows holding pairs (the rest padding);
//     at most ceil(P / tile) + n_expert tiles, so the buffer holds 4 + 4 * that.
//   row_of_pair[p]: the pair's row in the sorted order.
//   scratch: int32, (ceil(P / 256) + 1) * n_expert.
// n_expert at most 256. Rows of a segment past its expert's count are padding.
// cold (n_expert bytes, may be null): pairs of experts flagged 1 are left out of the tile list and get
// row_of_pair -1 (moe_gather and moe_combine skip them).
// group_of (n experts of the model, int32, may be null): groups are group_of[expert] instead of the expert, and pairs
// of experts with a negative group are left out; n_expert is then the number of groups, and row_of_pair keeps
// what it held for those pairs (a second grouping fills in the rest of an earlier one, as the streamed experts
// do). The tile list counts rows from 0, row_of_pair from row_base.
void moe_group(sycl::queue& q, const int32_t* ids, int64_t n_pairs, int64_t n_expert, int64_t tile, int32_t* scratch, int32_t* tiles,
               int32_t* row_of_pair, const uint8_t* cold = nullptr, const int32_t* group_of = nullptr, int32_t row_base = 0);
// Writes x[p / n_used][:] (fp32, cols a multiple of 16) for each pair p to its sorted row in the
// fp16 activation layout (xmx::act16_index) at `out`, as row - row_base; pairs with a row below row_base
// (and cold ones) are skipped, so that the pairs a second grouping added can be gathered alone.
void moe_gather(sycl::queue& q, const float* x, const int32_t* row_of_pair, int64_t n_pairs, int64_t n_used, int64_t cols,
                sycl::half* out, int32_t row_base = 0);

// The same from fp16 rows x [token][cols] (layer-major prefill keeps the MoE input of every sub-chunk of a group in fp16).
void moe_gather(sycl::queue& q, const sycl::half* x, const int32_t* row_of_pair, int64_t n_pairs, int64_t n_used, int64_t cols,
                sycl::half* out, int32_t row_base = 0);

// out[t][c] = w[c] . x[t] for fp32 w [n][k] and any number of tokens (n a multiple of 32, at most
// 256; k a multiple of 4): the router at prefill, kept in fp32 because routing is sensitive to
// its precision.
void matmul_f32(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out);

// out[t][c] = w[c] . x[t] for fp32 w [n][k] (k a multiple of 4) and up to 16 tokens
// (the router, ssm_alpha / ssm_beta and the shared expert's gate in decode and verify steps).
void matmul_f32_small(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out);

}  // namespace eklow::fast
