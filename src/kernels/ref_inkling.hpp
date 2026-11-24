// Reference kernels for Inkling 2 Mini (model/inkling2.hpp): plain SYCL, fp32, one sequence at positions 0..n_tok-1,
// each following the ggml graph of llama.cpp's src/models/inkling2.cpp at c74c71a6 (named there in each comment).
// Activations are [tokens][features]; like kernels/ref.hpp, they are enqueued on an in-order queue and not waited for.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "model/gguf.hpp"

namespace eklow::ref::inkling {

// graph::mix, attention over the depth: for each token, the states (earlier block outputs, then the current stream)
// are RMS-normed without a gain (eps), scored against each learned query (n_query rows of n_embd) with scale
// 1/sqrt(n_embd), softmaxed over the states, and the normed states summed with those weights.
// states: device array of n_states pointers to [n_tok][n_embd]; out: [n_query][n_tok][n_embd].
void depth_mix(sycl::queue& q, const float* const* states, int n_states, const float* query, int n_query, int64_t n_tok,
               int64_t n_embd, float eps, float* out);

// graph::convolve, a causal short convolution with a residual over `channels` channels of kernel 4:
//   out[t][c] = (x[t][c] + sum_k w[c][k] x[t-3+k][c] + sum_lag dyn[t][c / (channels / heads)][lag] x[t-lag][c]) * s
// with s = 1/sqrt(4 + 1), or 1/sqrt(2*4 + 1) when the input-dependent taps dyn ([n_tok][heads][4]) are given. hist is the
// recurrent state, [3][hist_stride] at this convolution's channels (x[-3], x[-2], x[-1]); it is read, then replaced by the
// last three inputs.
void short_conv(sycl::queue& q, const float* x, int64_t n_tok, int64_t channels, float* hist, int64_t hist_stride,
                const float* w, const float* dyn, int64_t heads, float* out);

// graph::clip: each row of `dim` (n_rows rows; row r is head r % heads) is multiplied by scale[head][:] when given, then
// scaled down to an L2 norm of at most threshold: x *= clamp(threshold / (|x| + 1e-6), 0, 1).
void l2_clip(sycl::queue& q, float* x, int64_t n_rows, int64_t heads, int64_t dim, const float* scale, float threshold);

// x[t][:] *= s[t], rows of `cols` (the full layers' log-position scale on Q and R).
void scale_tokens(sycl::queue& q, float* x, const float* s, int64_t n_tok, int64_t cols);

// The attention of a layer: query t, head h (K/V head h / (n_head / n_head_kv)) over keys j <= t (and t - j < window
// when window > 0):
//   s_j = scale * q.k_j + b(t - j),  b(d) = R[t][h] . table[d] for d < extent, else 0
//   out = sum_j exp(s_j) v_j / (exp(sink[t][h]) + sum_j exp(s_j))
// R: [n_tok][n_head][n_rel]; table: [extent][n_rel] (the layer's attn_rel_bias); sink: [n_tok][n_head].
void attention(sycl::queue& q, const float* Q, const float* K, const float* V, const float* R, const float* table,
               const float* sink, int64_t n_tok, int64_t n_head, int64_t n_head_kv, int64_t head_dim, int64_t n_rel,
               int64_t extent, int64_t window, float scale, float* out);

// x *= sigmoid(g), n elements (the attention's output gate).
void sigmoid_mul(sycl::queue& q, float* x, const float* g, int64_t n);

// build_moe_ffn's router with sigmoid gating: p = sigmoid(logits); the n_used experts of highest p + bias (ties to the
// lower index); weights = p of those / (sum + 1e-20) * w_scale. ids, weights: [n_tok][n_used].
void route(sycl::queue& q, const float* logits, const float* bias, int64_t n_tok, int64_t n_expert, int64_t n_used,
           float w_scale, int32_t* ids, float* weights);

// SITU on fused gate|up rows of 2 * n_ff: out = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb) (lb 0: u as is).
void situ(sycl::queue& q, const float* gate_up, int64_t rows, int64_t n_ff, float beta, float lb, float* out);

// out[t][:] = sum_s weights[t][s] * down[t][s][:], with n_embd columns.
void moe_combine(sycl::queue& q, const float* down, const float* weights, int64_t n_tok, int64_t n_used, int64_t n_embd,
                 float* out);

// graph::control's input: out[t] = [rmsnorm(cur[t]), rmsnorm(msg[t])] (no gain, eps), [n_tok][2 * n_embd].
void control_input(sycl::queue& q, const float* cur, const float* msg, int64_t n_tok, int64_t n_embd, float eps, float* out);

// graph::control's mix: out = cur - tanh(f) * cur + tanh(g) * msg, n elements.
void control_mix(sycl::queue& q, const float* cur, const float* msg, const float* g, const float* f, int64_t n, float* out);

// A matmul's activations rounded as ggml's x86 CPU backend quantizes them for a weight of type t (llama.cpp's CPU oracle
// computes with these; this makes eklow's reference compute the same products):
//   Q8_0, NVFP4 (vec_dot_type q8_0): per 32, d = fp16(amax / 127), x = rint(x * 127 / amax) * d
//   Q5_K (q8_K): per 256, m = the value of largest magnitude (the first), s = -127 / m, x = min(127, rint(s x)) * (1 / s)
//   BF16: x rounded to bf16 (nearest even); F32 and anything else: x unchanged.
// rows of k; out may be x.
void ggml_cpu_acts(sycl::queue& q, gguf::Type t, const float* x, int64_t rows, int64_t k, float* out);

}  // namespace eklow::ref::inkling
