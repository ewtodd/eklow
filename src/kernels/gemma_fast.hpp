// Gemma 4's own serving kernels (the attention heads from their projections, the residual and norm chain, the per-layer
// inputs, the soft-cap), beside kernels/fast.hpp's shared ones and in the same namespace; only engine/engine_gemma4 calls
// them. Same contract as fast.hpp: they only enqueue, on an in-order queue.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/fast.hpp"
#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"

namespace eklow::fast {

// The heads of a layer from its projections, read at q + t * ld_q (n_head x D), k + t * ld_k and v + t * ld_v (n_head_kv
// x D each; v null: K's projection, as on a global layer without a V projection; k null: Q only, a layer that reads
// another's K/V): Q = rope(rmsnorm_per_head(q) * q_norm), K = rope(rmsnorm(k) * k_norm), V = rmsnorm(v), unweighted. One
// work-group per head, one work-item per RoPE pair. NeoX RoPE at angle pos * inv_freq[i] in fp32, as Hugging Face and
// llama.cpp compute it (inv_freq already divided by any frequency factors). Q [t][n_head][D], K and V [t][n_head_kv][D].
void qkv_heads(sycl::queue& q, const float* qs, int64_t ld_q, const float* ks, int64_t ld_k, const float* vs, int64_t ld_v,
               int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, const float* q_norm, const float* k_norm,
               float eps, const int32_t* pos, const float* inv_freq, float* Q, float* K, float* V);

// A residual and the norm before the next matmul, quantized for it, one row per work-group:
//   o given:  x_out = (x + rmsnorm(o) * w_o) * scale;  otherwise x_out is x and is not written
//   w_next given: rmsnorm(x_out) * w_next into h_out (if given) and into act, in the form matmul
//   takes for `rows` rows.
// cols must be a multiple of 128 and at most 8192.
void residual_norm_quant(sycl::queue& q, const float* o, const float* w_o, const float* x, float scale, float* x_out,
                         const float* w_next, float eps, int64_t rows, int64_t cols, xmx::Activations* act, float* h_out);
void softcap(sycl::queue& q, float* x, float cap, int64_t n);

// The per-layer inputs, in place: pl[t][l][:] (n_layer blocks of `width`, the projection of the scaled embeddings) becomes
// (rmsnorm(pl[t][l] * in_scale) * norm + table[tokens[t]][l] * table_scale) * out_scale. The table is Q8_0 on the card.
void ple_inputs(sycl::queue& q, float* pl, const ref::Weight& table, const int32_t* tokens, int64_t n_tok, int64_t n_layer,
                int64_t width, const float* norm, float eps, float in_scale, float table_scale, float out_scale);

// gelu_tanh(g[t][:]) * pl[t][layer][:] for `rows` rows of `width` (a multiple of 32, at most 1024), into act in the form
// matmul takes for that many rows.
void ple_gate(sycl::queue& q, const float* g, const float* pl, int64_t layer, int64_t n_layer, int64_t rows, int64_t width,
              xmx::Activations& act);

}  // namespace eklow::fast
