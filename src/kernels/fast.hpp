// The serving kernels other than the matmuls (kernels/xmx.hpp), shared by the engines: per-token work over a batch
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

// out[r][:] = x[r][:] / sqrt(mean(x[r]^2) + eps) * (w ? w : 1); out may be x.
void rmsnorm(sycl::queue& q, const float* x, const float* w, int64_t rows, int64_t cols, float eps, float* out);

// out[r][v] = x[r] . W[v], r < rows.
void matmul_q6(sycl::queue& q, const Q6& w, const float* x, int64_t rows, float* out);

// out[t][c] = w[c] . x[t] for fp32 w [n][k] and any number of tokens (n a multiple of 32, at most
// 256; k a multiple of 4), in fp32 (vega's reader: its DeltaNet layers' alpha and beta).
void matmul_f32(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t n_tok, float* out);

}  // namespace eklow::fast
