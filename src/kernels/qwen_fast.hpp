// The serving kernels of the qwen35moe engine (Qwen3.6-35B-A3B) that kernels/fast.hpp does not have: the gated attention's
// heads, the sigmoid gate and the quantization for the output projection, the residual and the norm of the MoE's input, the
// MoE's tail (routed sum, shared expert under its sigmoid gate, residual, the next layer's norm), and a row gather for the
// captured hidden states and the logits rows. They only enqueue; the queue must be in order. Checked end to end through the
// engine's dumps against the reference forward (eklow-engine-run).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/xmx.hpp"

namespace eklow::qwenk {

// The gated attention's heads from the fused projection, row t of qkv = [per query head h: q (D) | gate (D)] x n_head,
// then k (n_head_kv x D), then v (n_head_kv x D): Q = rope(rmsnorm_per_head(q) * q_norm), K = rope(rmsnorm(k) * k_norm), V =
// v as it is, G = the gates as they are (the sigmoid comes with the output's quantization). NeoX RoPE over the first n_rot
// dims (pairs i, i + n_rot / 2) at angle pos * inv_freq[i], inv_freq [n_rot / 2] in fp32 (base^(-2i / n_rot)). Q and G
// [t][n_head][D], K and V [t][n_head_kv][D]. One work-group per (token, head), D / 2 work-items; D at most 512.
void attn_heads(sycl::queue& q, const float* qkv, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, int64_t n_rot,
                const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K,
                float* V, float* G);

// act = quantize(x * sigmoid(gate)) for rows of `cols` (a multiple of 1024), in the form matmul takes for `rows` rows.
void gate_quant(sycl::queue& q, const float* x, const float* gate, int64_t rows, int64_t cols, xmx::Activations& act);

// x_out = x + o (o null: x_out is x and is not written), then n = rmsnorm(x_out) * w, written to h_out (if given) and
// quantized into act (if given). cols a multiple of 128, at most 8192. x_out may be x.
void add_norm_quant(sycl::queue& q, const float* o, const float* x, const float* w, float eps, int64_t rows, int64_t cols,
                    float* x_out, xmx::Activations* act, float* h_out);

// The layer's tail, per row: s = sum_i weights[t][i] * y[t * n_used + i] (y given: decode's expert outputs, [pair][cols]) or
// moe[t] (the prefill's already summed rows); x_out = x2 + s + sigmoid(sgate[t * ld_sg]) * shared[t]; and with w_next the
// next layer's norm of x_out into h_out (if given) and act (if given). x_out may not be x2.
void moe_tail(sycl::queue& q, const float* x2, const float* y, const float* weights, int64_t n_used, const float* moe,
              const float* sgate, int64_t ld_sg, const float* shared, const float* w_next, float eps, int64_t rows, int64_t cols,
              float* x_out, xmx::Activations* act, float* h_out);

// dst[r][col0 .. col0 + cols) = x[idx[r]][0 .. cols), r < n (dst rows are ld floats apart).
void gather_rows(sycl::queue& q, const float* x, const int32_t* idx, int64_t n, int64_t cols, float* dst, int64_t ld, int64_t col0);

// dst[r][c] = src[r][c] for the columns [c0, c0 + n) of dst's rows, src rows `n` wide (assembles a matrix from parts).
void copy_cols(sycl::queue& q, const float* src, int64_t rows, int64_t n, float* dst, int64_t ld, int64_t c0);

}  // namespace eklow::qwenk
