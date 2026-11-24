// One expert's FFN of Inkling 2 Mini on the host CPU (the cold experts of M3), with the reference's arithmetic
// (ref::matmul_experts, ref::inkling::situ) but int8 activations (the card's fp16): the gate_up and down GEMVs are
// AVX-VNNI (nvfp4.hpp for NVFP4, qgemv.hpp for Q5_K and Q8_0), on the GGUF's bytes as stored.
#pragma once

#include <cstdint>

#include "cpu/pool.hpp"
#include "cpu/qgemv.hpp"

namespace eklow::cpu {

// One expert as the GGUF stores it: gate_up rows [2 n_ff][n_embd] (gate rows 0..n_ff-1, up rows n_ff..2 n_ff-1), down
// rows [n_embd][n_ff]; `gu` and `dn` point at the expert's first row. The scales are the per-expert tensors'
// (ffn_gate_up_exps.scale, ffn_down_exps.scale; 1 when the file has none).
struct ExpertRef {
  QType gu_type, dn_type;
  const uint8_t* gu;
  const uint8_t* dn;
  float gu_scale, dn_scale;
};

// y[j][i] = dn_scale * down(situ(gu_scale * gate_up(x[j]))) for j < nx, i < n_embd (fp32 rows of n_embd), with
// situ(g, u) = beta tanh(g / beta) sigmoid(g) * (lb > 0 ? lb tanh(u / lb) : u): beta = situ_beta, lb = situ_linear_beta.
// n_embd must be a multiple of 256 (Q5_K) and n_ff of 64 (NVFP4) or 32 (Q8_0). The activations (x, and situ's output
// before the down matrix) are quantized to int8: per 16 for NVFP4, per 32 for Q8_0 and Q5_K.
// Rows split over `pool` in grains of 16; each weight row is read once for groups of four inputs, so a prefill chunk's
// nx inputs read the expert once (from RAM, then L2).
void expert_ffn(const ExpertRef& e, int64_t n_embd, int64_t n_ff, const float* const* x, int nx, float* const* y,
                float situ_beta, float situ_linear_beta, Pool& pool);

// Several experts of one layer on the same inputs, in one parallel pass over all of them (two pool barriers for the
// whole set, not two per expert): y[i * nx + j] is expert i's output for input j.
void experts_ffn(const ExpertRef* e, int ne, int64_t n_embd, int64_t n_ff, const float* const* x, int nx, float* const* y,
                 float situ_beta, float situ_linear_beta, Pool& pool);

}  // namespace eklow::cpu
