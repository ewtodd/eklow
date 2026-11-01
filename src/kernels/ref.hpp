// Reference kernels: plain SYCL, fp32 activations and accumulation, weights dequantized exactly as
// ggml stores them. They define what the fast kernels must compute and are checked against the
// oracle (llama.cpp's CPU backend) layer by layer; speed is not their job. Long launches are split
// so that none comes near xe's 5 s job timeout.
//
// The kernels are enqueued and not waited for: every queue here is in-order, so they run in the
// order called, and a caller waits before it reads results on the host or frees what they use.
// (A wait per kernel made the embedding model take a minute an input whenever a prefill held the
// card; DECISIONS 2026-10-26.)
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

// Attention over all n_tok keys for every query, no mask (a vision encoder's): query head h reads
// K/V head h / (n_head / n_head_kv); one work-item per (query, head), online softmax.
// out[t][h][d] = softmax_j(scale * q.k_j) . v_j
void attention_dense(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t n_tok, int64_t n_head,
                     int64_t n_head_kv, int64_t head_dim, float scale, float* out);

// x[r][:] += v[:], rows of `cols`.
void add_rows(sycl::queue& q, float* x, const float* v, int64_t rows, int64_t cols);

// out = gelu_quick(gate) * up, gelu_quick(x) = x / (1 + exp(-1.702 x)) (ggml_geglu_quick_split)
void gelu_quick_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out);
// out = x rounded to bf16 (nearest even) and back: what ggml's CPU backend does to the activations of a BF16 weight
void round_bf16(sycl::queue& q, const float* x, int64_t n, float* out);
// out = min(max(x, lo), hi)
void clamp(sycl::queue& q, const float* x, float lo, float hi, int64_t n, float* out);
// x[r][:] *= v[:], rows of `cols`.
void mul_rows(sycl::queue& q, float* x, const float* v, int64_t rows, int64_t cols);

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

// ---- Qwen3.5/3.6 (llama.cpp's qwen35moe): Gated DeltaNet layers, gated attention, MoE with a shared expert ----

// out = x / (1 + exp(-x)) (ggml_silu)
void silu(sycl::queue& q, const float* x, int64_t n, float* out);
// out = silu(gate) * up (ggml_swiglu_split)
void swiglu(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out);
// out = 1 / (1 + exp(-x))
void sigmoid(sycl::queue& q, const float* x, int64_t n, float* out);
// out = log(1 + exp(x)), x itself above 20 (ggml_softplus)
void softplus(sycl::queue& q, const float* x, int64_t n, float* out);
// out[r][c] = x[r][c] * s[r] (a gate with one value per row)
void mul_by_row_scalar(sycl::queue& q, const float* x, const float* s, int64_t rows, int64_t cols, float* out);

// llama.cpp's ggml_ssm_conv on a sequence that starts from a zero state: out[t][c] =
// sum_{j<kernel} w[c][j] * x[t - (kernel - 1) + j][c], x taken as 0 before the first token.
// x, out are [tokens][channels]; w is [channels][kernel] (the GGUF's ne = {kernel, channels}).
void causal_conv(sycl::queue& q, const float* x, const float* w, int64_t n_tok, int64_t channels, int64_t kernel,
                 float* out);

// build_gdn_l2_norm: each row of `cols` is divided by sqrt(sum of squares + eps), computed as ggml does,
// scale = 1/sqrt(mean(x^2) + eps/cols) / sqrt(cols). x is [tokens][heads][cols] inside a wider row: head h of
// token t starts at x + t * tok_stride + h * cols; out is contiguous [tokens][heads][cols].
void l2_norm_heads(sycl::queue& q, const float* x, int64_t tok_stride, int64_t n_tok, int64_t n_heads, int64_t cols,
                   float eps, float* out);

// The gated delta rule of one sequence from a zero state (delta-net-base.cpp's autoregressive form
// applied token by token). q, k: [tokens][n_k_heads][s], already l2-normalized; v: token t, head h at
// v + t * v_stride + h * s;
// g, beta: [tokens][n_v_heads] (g is the log decay, beta already through the sigmoid). Value head h reads
// key head h % n_k_heads (ggml_repeat tiles the key heads; the converter reorders the value heads to
// match). q is scaled by 1/sqrt(s). The state S[head][column j][i] (ggml's stored order: the key index
// i fastest) lives in `state` ([n_v_heads][s][s], zeroed here) and out[t][h][j] = sum_i S[j][i] q[i].
// One work-item per (head, column); long sequences are split into launches.
void gated_delta_rule(sycl::queue& q, const float* qv, const float* kv, const float* v, int64_t v_stride,
                      const float* g, const float* beta, int64_t n_tok, int64_t n_k_heads, int64_t n_v_heads, int64_t s,
                      float* state, float* out);

// out[r][:] = rmsnorm(x[r][:]) * w * silu(z[r][:]) (build_norm_gated), rows of `cols`.
void gated_rmsnorm(sycl::queue& q, const float* x, const float* z, const float* w, int64_t rows, int64_t cols,
                   float eps, float* out);

// full[t][h][0..2d) holds the query of head h in [0, d) and its gate in [d, 2d): split them.
void split_q_gate(sycl::queue& q, const float* full, int64_t n_tok, int64_t n_head, int64_t d, float* qout,
                  float* gate);

// dst[i][:] = src[ids[i]][:] (rows of `cols`)
void gather_rows(sycl::queue& q, const float* src, const int32_t* ids, int64_t n, int64_t cols, float* dst);
// dst[ids[i]][:] += w[i] * src[i][:]; the ids of one call must differ
void scatter_add_rows(sycl::queue& q, const float* src, const int32_t* ids, const float* w, int64_t n,
                      int64_t cols, float* dst);
// gu[i][0..2f) = gate then up of one row per token: gate[i][j] = gu[i][j], up[i][j] = gu[i][f + j]
void split_gate_up(sycl::queue& q, const float* gu, int64_t n, int64_t f, float* gate, float* up);

}  // namespace eklow::ref
