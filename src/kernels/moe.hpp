// Inkling 2 Mini's mixture of experts on the card (DECISIONS 2026-11-24, M2): one layer's FFN over a step's tokens, as
// model/inkling2.hpp's reference computes it (ref::matmul_experts, ref::inkling::situ, ref::inkling::moe_combine):
//   out[t] = sum_s w[t][s] * down_scale[e] * down_e( situ( gate_up_scale[e] * gate_up_e(x[t]) ) ),  e = ids[t][s]
// with situ(g, u) = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), the gate the first n_ff outputs of gate_up and up the
// last n_ff (the GGUF's ffn_gate_up_exps rows). The routing (ids, w) comes from the caller.
//
// The experts keep the GGUF's bits: NVFP4 (layers 0-19) on the matrix units as fp16 (e2m1 x ue4m3 is exact in fp16, a K=16
// dpas is one NVFP4 scale group), Q5_K and Q8_0 (layers 20-23) in their xmx formats. A step of few tokens runs grouped
// GEMVs over the (token, expert) pairs; a prefill step sorts the pairs by expert and runs grouped GEMMs.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"

namespace eklow::moe {

// One layer's experts on the card.
struct Layer {
  xmx::ExpertTable gate_up, down;            // gate_up as the fused SITU kernels want it (interleaved if they do)
  const float* gate_up_scale = nullptr;     // per expert (NVFP4's second-level scale), or null for 1
  const float* down_scale = nullptr;
  int64_t n_embd = 0, n_ff = 0, n_expert = 0;
  uint64_t bytes = 0;                       // device bytes the layer's experts take
};

// The layer's experts from the GGUF's tensors as stored (in host memory; raw.data points at expert 0, expert e at
// raw.data + e * stride; scales host pointers or null), repacked onto the card.
Layer upload(sycl::queue& q, const ref::Weight& gate_up_raw, uint64_t gate_up_stride, const float* gate_up_scale,
             const ref::Weight& down_raw, uint64_t down_stride, const float* down_scale, int64_t n_expert);
void free_layer(sycl::queue& q, Layer& L);

// Device workspace for steps of up to max_tokens tokens with n_used experts each.
struct Scratch;
Scratch* reserve(sycl::queue& q, int64_t max_tokens, int64_t n_used, int64_t n_embd, int64_t n_ff);
void free_scratch(sycl::queue& q, Scratch* s);

// out[t][:] for t < n_tok, fp32 [n_tok][n_embd]; x fp32 [n_tok][n_embd]; ids int32 and w fp32 [n_tok][n_used] on the
// device. Enqueues only (in-order queue).
void ffn(sycl::queue& q, const Layer& L, const float* x, int64_t n_tok, const int32_t* ids, const float* w, int64_t n_used,
         float situ_beta, float situ_linear_beta, Scratch& s, float* out);

}  // namespace eklow::moe
