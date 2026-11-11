// Reference kernels of the DFlash draft (src/model/dflash2.*): plain SYCL, fp32, like ref.hpp's. They are the draft's own
// operations (attention over explicit key ranges, the top-k of a logits row); the rest of the reference forward uses
// ref.hpp's kernels. Like those they only enqueue, on an in-order queue.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/ref.hpp"

namespace eklow::ref {

// Attention of n_tok queries over keys K, V [n_keys][n_head_kv][head_dim] whose first row is position key0: query t
// (n_head heads, head h reads K/V head h / (n_head / n_head_kv)) sees the keys at positions lo[t]..hi[t] inclusive
// (device arrays), softmax(scale q.k) v. Out [n_tok][n_head][head_dim].
void attention_ranges(sycl::queue& q, const float* Q, const float* K, const float* V, int64_t key0, int64_t n_tok, int64_t n_head,
                      int64_t n_head_kv, int64_t head_dim, const int32_t* lo, const int32_t* hi, float scale, float* out);

// The k largest of each of `rows` rows of n values: ids[r][0..k) and vals[r][0..k) by descending value, equal values
// by ascending id.
void topk_rows(sycl::queue& q, const float* x, int64_t rows, int64_t n, int64_t k, int32_t* ids, float* vals);

// out = silu(gate) * up, silu(x) = x / (1 + exp(-x)) (SwiGLU)
void silu_mul(sycl::queue& q, const float* gate, const float* up, int64_t n, float* out);

// out = x as the serving kernels see it before a matmul: rows of `cols` quantized to int8 per 32-block (scale max|x| / 127,
// q = round(x / scale), xmx::quantize's rule) and back, or (fp16) rounded to fp16. For comparing the serving forward with
// the reference on the activations it really uses.
void round_acts(sycl::queue& q, const float* x, int64_t rows, int64_t cols, bool fp16, float* out);

// x[i] *= 1 + eps * u_i with u_i a fixed pseudo-random number in [-1, 1): a perturbation of the size of fp32 rounding,
// to measure how a computation amplifies it.
void jitter(sycl::queue& q, float* x, int64_t n, float eps);

}  // namespace eklow::ref
