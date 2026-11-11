// Small kernels the dense side models share (vega's reader, src/vega/reader.hpp; Qwen3-Reranker, src/model/rerank_qwen3.hpp):
// the residual add with RMSNorm into the fp16 GEMMs' activations, and the attention heads of a Qwen3 layer. They only
// enqueue; the queue must be in order.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/xmx.hpp"

namespace eklow::dense {

// x += add (when given), then rmsnorm(x) * w into `act` as fp16 activations (marked as act_rows rows: the rows past `rows`
// keep what they held) and, when h is given, into h in fp32. A work-group of 256 per row; E a multiple of 256, at most 2048.
void norm_quant(sycl::queue& q, float* x, const float* add, const float* w, int64_t rows, int64_t E, float eps, xmx::Activations& act,
                int64_t act_rows, float* h);

// Qwen3's attention heads from the fused projection, row t of qkv = q (n_head x D) | k (n_head_kv x D) | v (n_head_kv x D):
// Q = rope(rmsnorm(q_h) * q_norm), K = rope(rmsnorm(k_h) * k_norm), V as it is, NeoX RoPE over all D dims (pairs i,
// i + D/2) at angle pos[t] * inv_freq[i]. Q [t][n_head][D], K and V [t][n_head_kv][D]. D at most 512.
void qk_heads(sycl::queue& q, const float* qkv, int64_t rows, int64_t n_head, int64_t n_head_kv, int64_t D, const float* q_norm,
              const float* k_norm, float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K, float* V);

}  // namespace eklow::dense
