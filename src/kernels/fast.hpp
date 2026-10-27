// The serving kernels other than the Q4_0 matmuls (kernels/xmx.hpp): per-token work over a batch
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

// out[t][:] = table row tokens[t] * scale
void embed(sycl::queue& q, const Q6& table, const int32_t* tokens, int64_t n_tok, float scale, float* out);

// out[r][:] = x[r][:] / sqrt(mean(x[r]^2) + eps) * (w ? w : 1); out may be x.
void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out);

// The heads of a layer from its fused projection, row t of qkv = [q (n_head x D) | k (n_head_kv x
// D) | v (n_head_kv x D, when has_v)]: Q = rope(rmsnorm_per_head(q) * q_norm), K = rope(rmsnorm(k)
// * k_norm), V = rmsnorm(v, or k's projection without has_v), unweighted. One work-group per head,
// one work-item per RoPE pair. NeoX RoPE at angle pos * inv_freq[i] in fp32, as Hugging Face and
// llama.cpp compute it (inv_freq already divided by any frequency factors). Q [t][n_head][D], K and V
// [t][n_head_kv][D].
void qkv_heads(sycl::queue& q, const float* qkv, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim,
               bool has_v, const float* q_norm, const float* k_norm, float eps, const int32_t* pos, const float* inv_freq,
               float* Q, float* K, float* V);

// A residual and the norm before the next matmul, quantized for it, one row per work-group:
//   o given:  x_out = (x + rmsnorm(o) * w_o) * scale;  otherwise x_out is x and is not written
//   w_next given: rmsnorm(x_out) * w_next into h_out (if given) and into act, in the form matmul
//   takes for `rows` rows.
// cols must be a multiple of 128 and at most 8192.
void residual_norm_quant(sycl::queue& q, const float* o, const float* w_o, const float* x, float scale, float* x_out,
                         const float* w_next, float eps, int64_t rows, int64_t cols, xmx::Activations* act, float* h_out);
void softcap(sycl::queue& q, float* x, float cap, int64_t n);

// out[r][v] = x[r] . W[v], r < rows.
void matmul_q6(sycl::queue& q, const Q6& w, const float* x, int64_t rows, float* out);

// The 26B's router, per token: softmax of logits[t][n_expert], the top n_used, their weights
// renormalized by the sum (floored at 2^-14) and multiplied by each expert's down scale (on the
// device). ids[t][n_used] int32 and weights[t][n_used] fp32, by descending probability. n_expert a
// multiple of 32, at most 256; n_used at most 16.
void moe_route(sycl::queue& q, const float* logits, int64_t n_tok, int64_t n_expert, int64_t n_used, const float* down_scale,
               int32_t* ids, float* weights);
// out[t][:] = sum_i weights[t][i] * y[t * n_used + i][:], summed in the order i = 0..n_used-1.
void moe_combine(sycl::queue& q, const float* y, const float* weights, int64_t n_tok, int64_t n_used, int64_t cols, float* out);

// out[t][c] = w[c] . x[t] for fp32 w [n][k] (k a multiple of 4) and up to 4 tokens: the 26B's router
// at decode (ref::matmul's generic path ran one work-item per output).
void matmul_f32_small(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out);

// The 26B's decode step, fused (T * n_used <= 32 rows):
// after attention, x_out = x + rmsnorm(o) * w_o (the attention output; x_out may be x's other buffer), then
// n = rmsnorm(x_out) is written three ways: * w1 into int8 a1 (the dense FFN's input), * w2 into int8 a2
// (the experts' input) and * w3 into fp32 r_out (the router's input).
void residual_norm_quant3(sycl::queue& q, const float* o, const float* w_o, const float* x, float* x_out, float eps, int64_t rows,
                          int64_t cols, xmx::Activations& a1, const float* w1, xmx::Activations& a2, const float* w2,
                          float* r_out, const float* w3);
// The layer's tail in one kernel per row: d = rmsnorm(dense) * w_d; m = rmsnorm(sum_j weights[j] y[j]) * w_m;
// c = d + m; x_out = (x2 + rmsnorm(c) * w_post) * scale; and with w_next the next layer's attn_norm of x_out
// into int8 act. d_dump, m_dump, c_dump (each may be null) receive d, m, c for the dump sink.
void moe_tail(sycl::queue& q, const float* dense, const float* w_d, const float* y, const float* weights, int64_t n_used,
              const float* w_m, const float* w_post, const float* x2, float scale, float* x_out, const float* w_next, float eps,
              int64_t rows, int64_t cols, xmx::Activations* act, float* d_dump, float* m_dump, float* c_dump);

}  // namespace eklow::fast
