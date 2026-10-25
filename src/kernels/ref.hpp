// Reference kernels: plain SYCL, fp32 activations and accumulation, weights dequantized exactly as
// ggml stores them. They define what the fast kernels must compute and are checked against the
// oracle (llama.cpp's CPU backend) layer by layer; speed is not their job. Long launches are split
// so that none comes near xe's 5 s job timeout.
//
// Layouts follow ggml: a matrix's rows are contiguous, a weight of shape ne = {K, N} is N rows of K
// (quantized in blocks along K), activations are [tokens][features].
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

#include "model/gguf.hpp"

namespace eklow::ref {

struct Weight {
  gguf::Type type;
  int64_t k, n;          // ne[0] (row length), ne[1] (rows)
  const uint8_t* data;   // device memory
  uint64_t row_bytes;
};

// out[t][:] = dequant(table row tokens[t]) * scale
void embed(sycl::queue& q, const Weight& table, const int32_t* tokens, int64_t n_tok, float scale, float* out);

// out[r][:] = x[r][:] / sqrt(mean(x[r]^2) + eps) * (w ? w : 1), rows of `cols`.
void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out);

// out[m][n] = sum_k x[m][k] * W[n][k] for rows m in [0, m_rows).
void matmul(sycl::queue& q, const Weight& w, const float* x, int64_t m_rows, float* out);

// cos and sin of the NeoX RoPE angles pos * base^(-2i/n_rot) / freq_factors[i] for positions
// 0..n_tok-1 and i < n_rot/2, computed on the host in double (the A770M has no fp64), [pos][i].
struct RopeTables {
  std::vector<float> cos, sin;
};
RopeTables rope_tables(int64_t n_tok, int64_t n_rot, double base, const float* freq_factors);

// NeoX RoPE on x[t][h][0..n_rot) in place: pairs (i, i + n_rot/2) turn by the tables' angles
// (device copies of a RopeTables).
void rope_neox(sycl::queue& q, float* x, int64_t n_tok, int64_t n_head, int64_t head_dim, int64_t n_rot,
               const float* cos_tab, const float* sin_tab);

// Causal attention, one sequence at positions 0..n_tok-1: query head h reads K/V head
// h / (n_head / n_head_kv); keys more than window-1 positions back are masked when window > 0.
// out[t][h][d] = softmax_j(scale * q.k_j) . v_j
void attention(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t n_tok, int64_t n_head,
               int64_t n_head_kv, int64_t head_dim, int64_t window, float scale, float* out);

// Attention within one sequence in both directions: query t sees keys j with |t - j| <=
// half_window, or all of them when half_window is 0 (llama.cpp's symmetric sliding window).
void attention_bidir(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t n_tok, int64_t n_head,
                     int64_t n_head_kv, int64_t head_dim, int64_t half_window, float scale, float* out);

// out = gelu_tanh(gate) * up
void gelu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out);
// out = gelu_tanh(x)
void gelu(sycl::queue& q, const float* x, int64_t n, float* out);
// out = a * b
void mul(sycl::queue& q, const float* a, const float* b, int64_t n, float* out);
// out = a + b
void add(sycl::queue& q, const float* a, const float* b, int64_t n, float* out);
// x *= s
void scale(sycl::queue& q, float* x, float s, int64_t n);
// x = cap * tanh(x / cap)
void softcap(sycl::queue& q, float* x, float cap, int64_t n);

}  // namespace eklow::ref
