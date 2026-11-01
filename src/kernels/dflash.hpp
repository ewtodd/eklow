// The DFlash 2 draft's serving kernels (src/model/dflash2_fast.cpp), beside the matrix-unit matmuls (xmx.hpp) and
// attention (attention.hpp): the per-head norm and RoPE of the draft's q, k and v, the grouped dynamic convolution, and
// the top 16 of each row of a head's logits. They only enqueue, on an in-order queue, and are checked against
// ref_dflash.hpp's by eklow-dflash-test.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace eklow::dflash {

// The heads of rows of a fused projection: row t of src (stride ld) holds nq query heads at column q_off, nk key heads
// at k_off and nk value heads at v_off, head_dim 128 each. Q[t][h] = rope(rmsnorm(q) * q_norm), K = rope(rmsnorm(k) *
// k_norm), V = v (copied), written contiguously ([t][head][128]; Q and its norm are skipped when nq is 0). NeoX RoPE at
// angle pos[t] * inv_freq[i], i < 64, in fp32 as Hugging Face and llama.cpp compute it.
void heads(sycl::queue& q, const float* src, int64_t ld, int64_t n_tok, int nq, int nk, int64_t q_off, int64_t k_off, int64_t v_off,
           const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K, float* V);

// model.py's grouped dynamic causal convolution, side `side` (ref_dflash.hpp has the formula), kernel size 2, over rows
// that are blocks of block_len tokens. With accumulate, out[t][c] += the result (the residual add of the second side).
void conv(sycl::queue& q, const float* x, const float* dyn, const float* base, int side, int64_t rows, int64_t block_len, int64_t hidden,
          int64_t group, float* out, bool accumulate);

// The 16 largest of each of `rows` rows of n values, by descending value and ascending id for equal values, as
// ref::topk_rows. Two kernels: each work-group takes the top 16 of a 2,048-value chunk of a row, one more merges a
// row's chunks. scratch_ids and scratch_vals hold rows * chunks(n) * 16 entries.
int64_t topk_chunks(int64_t n);
void topk16(sycl::queue& q, const float* x, int64_t rows, int64_t n, int32_t* scratch_ids, float* scratch_vals, int32_t* ids, float* vals);

// ids[b][i][a] = the anchor of block b for i = 0, else cand[b][i - 1][a]: the selector's predecessor tokens (P positions of
// 16 candidates per block).
void predecessor_ids(sycl::queue& q, const int32_t* cand, const int32_t* anchors, int64_t blocks, int64_t P, int32_t* ids);

}  // namespace eklow::dflash
