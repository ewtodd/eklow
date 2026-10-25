// The serving kernels other than the Q4_0 matmuls (kernels/xmx.hpp): per-token work over a batch
// of tokens from several sequences. They only enqueue; the queue must be in order. Each is checked
// against its reference in kernels/ref.hpp by eklow-engine-test.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/ref.hpp"

namespace eklow::fast {

// out[t][:] = dequant(table row tokens[t]) * scale; table Q6_K, F16 or F32.
void embed(sycl::queue& q, const ref::Weight& table, const int32_t* tokens, int64_t n_tok, float scale, float* out);

// out[r][:] = x[r][:] / sqrt(mean(x[r]^2) + eps) * (w ? w : 1); out may be x.
void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out);

// NeoX RoPE in place on x[t][h][0..n_rot) at positions pos[t]: angle pos * inv_freq[i], in fp32 as
// Hugging Face and llama.cpp compute it (inv_freq already divided by any frequency factors).
void rope_neox(sycl::queue& q, float* x, const int32_t* pos, int64_t n_tok, int64_t n_head, int64_t head_dim,
               int64_t n_rot, const float* inv_freq);

void gelu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out);
// out = (a + b) * s
void add_scale(sycl::queue& q, const float* a, const float* b, float s, int64_t n, float* out);
void softcap(sycl::queue& q, float* x, float cap, int64_t n);

// out[r][v] = x[r] . W[v] for a Q6_K matrix (the tied output head), r < rows.
void matmul_q6k(sycl::queue& q, const ref::Weight& w, const float* x, int64_t rows, float* out);

}  // namespace eklow::fast
