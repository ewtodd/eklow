// Two kernels of the qwen35moe engine (Qwen3.6-35B-A3B, removed 2026-11-11) that outlived it: the gated attention's heads, which
// vega's reader (Qwen3.5-0.8B, the same attention) runs, and a row gather, which Bolt's engine uses for the hidden states its
// draft reads. They only enqueue; the queue must be in order.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/xmx.hpp"

namespace eklow::qwenk {

// The gated attention's heads from the fused projection, row t of qkv = [per query head h: q (D) | gate (D)] x n_head,
// then k (n_head_kv x D), then v (n_head_kv x D): Q = rope(rmsnorm_per_head(q) * q_norm), K = rope(rmsnorm(k) * k_norm), V =
// v as it is, G = the gates as they are (the sigmoid comes with the output's quantization). NeoX RoPE over the first n_rot
// dims (pairs i, i + n_rot / 2) at angle p * inv_freq[i], inv_freq [n_rot / 2] in fp32 (base^(-2i / n_rot)), where p is
// one of the token's three position components, as llama.cpp's IMRoPE interleaves them (ggml_rope_multi): pair j takes the
// time component when j % 3 == 0 and j < 3 * sections[0], the row's when j % 3 == 1 and j < 3 * sections[1], the column's
// when j % 3 == 2 and j < 3 * sections[2]. `pos` is [3][n_tok] (time, row, column); text has all three equal. Q and G
// [t][n_head][D], K and V [t][n_head_kv][D]. One work-group per (token, head), D / 2 work-items; D at most 512.
void attn_heads(sycl::queue& q, const float* qkv, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, int64_t n_rot,
                const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const int* sections, const float* inv_freq,
                float* Q, float* K, float* V, float* G);

// dst[r][col0 .. col0 + cols) = x[idx[r]][0 .. cols), r < n (dst rows are ld floats apart).
void gather_rows(sycl::queue& q, const float* x, const int32_t* idx, int64_t n, int64_t cols, float* dst, int64_t ld, int64_t col0);

}  // namespace eklow::qwenk
