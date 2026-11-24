// Inkling 2 Mini's mixture of experts on the card (DECISIONS 2026-11-24, M2): one layer's FFN over a step's tokens, as
// model/inkling2.hpp's reference computes it (ref::matmul_experts, ref::inkling::situ, ref::inkling::moe_combine):
//   out[t] = sum_s w[t][s] * down_scale[e] * down_e( situ( gate_up_scale[e] * gate_up_e(x[t]) ) ),  e = ids[t][s]
// with situ(g, u) = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb), the gate the first n_ff outputs of gate_up and up the
// last n_ff (the GGUF's ffn_gate_up_exps rows). The routing (ids, w) comes from the caller.
//
// The experts keep the GGUF's bits (4.5 bits a weight for NVFP4, layers 0-19; Q5_K gate_up and Q8_0 down, layers 20-23), in
// layouts of their own (moe.cpp): every matrix is records of 8 output columns x 64 K, [column unit][K step][column], loaded
// contiguously, and decoded to fp16 on the way to the matrix units (e2m1 x ue4m3 is exact in fp16, a K=16 dpas is one NVFP4
// scale group). Activations are fp16 throughout. A step of few tokens runs grouped GEMVs over the experts' row tiles of 8; a
// prefill step sorts the (token, expert) pairs by expert and runs grouped GEMMs.
//
// One layer's data is one device allocation (Layer::base, Layer::bytes: both matrices' records and the per-expert scales), so
// that the engine can keep it in host memory and copy it back: `rebase` points a Layer at a copy of the bytes.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/ref.hpp"

namespace eklow::moe {

enum class Format : uint8_t { NVFP4, Q5_K, Q8_0 };

// One expert matrix: k inputs, n outputs; expert e's records at base + offset + e * expert_bytes. `cg` records of the
// same K step sit together (the gate and up halves of gate_up, which are interleaved by 8-column group; down's pairs).
struct Table {
  Format format = Format::NVFP4;
  int64_t k = 0, n = 0;
  int cg = 2;
  uint64_t offset = 0, expert_bytes = 0;
};

// One layer's experts on the card.
struct Layer {
  Table gate_up, down;
  uint64_t gate_up_scale_offset = 0, down_scale_offset = 0;  // in base; valid when the matching *_scale is non-null
  const float* gate_up_scale = nullptr;                      // per expert (NVFP4's second-level scale), or null for 1; inside base
  const float* down_scale = nullptr;
  int64_t n_embd = 0, n_ff = 0, n_expert = 0;
  uint8_t* base = nullptr;  // the one device allocation holding everything above
  uint64_t bytes = 0;       // its size
};

// The layer's experts from the GGUF's tensors as stored (in host memory; raw.data points at expert 0, expert e at
// raw.data + e * stride; scales host pointers or null), repacked onto the card. NVFP4 and Q5_K gate_up, NVFP4 and Q8_0 down.
Layer upload(sycl::queue& q, const ref::Weight& gate_up_raw, uint64_t gate_up_stride, const float* gate_up_scale,
             const ref::Weight& down_raw, uint64_t down_stride, const float* down_scale, int64_t n_expert);
// Frees L.base (not for a rebased layer: its bytes are the engine's).
void free_layer(sycl::queue& q, Layer& L);
// Points L at a copy of its bytes at new_base (the engine has copied base..base + bytes there): the tables' addresses and the
// scale pointers follow. Host-side only; L.base is not freed.
void rebase(sycl::queue& q, Layer& L, uint8_t* new_base);

// Device workspace for steps of up to max_tokens tokens with n_used experts each (ffn runs longer steps in slabs of
// scratch_slab() tokens; max_tokens above it costs nothing).
struct Scratch;
Scratch* reserve(sycl::queue& q, int64_t max_tokens, int64_t n_used, int64_t n_embd, int64_t n_ff);
void free_scratch(sycl::queue& q, Scratch* s);
uint64_t scratch_bytes(const Scratch& s);
int64_t scratch_slab();

// out[t][:] for t < n_tok, fp32 [n_tok][n_embd]; x fp32 [n_tok][n_embd]; ids int32 and w fp32 [n_tok][n_used] on the
// device. Enqueues only (in-order queue).
void ffn(sycl::queue& q, const Layer& L, const float* x, int64_t n_tok, const int32_t* ids, const float* w, int64_t n_used,
         float situ_beta, float situ_linear_beta, Scratch& s, float* out);

}  // namespace eklow::moe
