// Inkling 2 Mini's own serving kernels (engine/engine_inkling), beside the shared ones (kernels/fast.hpp), the matmuls
// (kernels/xmx.hpp), the experts (kernels/moe.hpp) and the attention (kernels/attention.hpp). Each computes what the
// reference kernel of the same name in kernels/ref_inkling.hpp computes, over a step's batch of tokens from several
// sequences (rows in batch order, chunk after chunk). They only enqueue, on an in-order queue.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace eklow::inkling {

// Up to 8 states of [rows][n_embd], by value (no device pointer table).
struct States {
  const float* p[8] = {};
  int n = 0;
};

// ref::inkling::depth_mix: out [n_query][rows][n_embd]. With `scratch` (kMixScratch floats a row of `rows`), each row split
// across work-groups; without, up to 64 rows a work-group a row, above a sub-group a row.
constexpr int64_t kMixFewRows = 16;
constexpr int64_t kMixScratch = 40 * 16;  // a row's slices' sums: 40 values for each of up to 16 slices of 256 (n_embd <= 4096)
void depth_mix(sycl::queue& q, const States& s, const float* query, int n_query, int64_t rows, int64_t n_embd, float eps, float* out,
               float* scratch = nullptr);

// The chunks of a step, by value: where each starts in the batch, its length and its sequence slot.
struct Chunks {
  int32_t base[8] = {}, n[8] = {}, slot[8] = {};
  int count = 0;
};

// ref::inkling::short_conv for a step: x [T][channels] (row stride ld_x), the state of token t's sequence at
// conv + slot[t] * state_stride + (r * hist_stride) for rows r = 0..2 (x[-3], x[-2], x[-1] before the chunk), read for the
// chunk's first three tokens and then replaced by the chunk's last three inputs. dyn [T][heads][4] or null. out [T][channels].
void short_conv(sycl::queue& q, const float* x, int64_t ld_x, int64_t T, int64_t channels, const Chunks& ch, const int32_t* slot,
                const int32_t* base, float* conv, int64_t state_stride, int64_t hist_stride, const float* w, const float* dyn,
                int64_t heads, float* out);

// The heads of an attention from its projections and convolutions, per token t (position pos[t]):
//   Q [t][n_head][D]  = clip(q[t] * q_clip, qk_clip) * tau,   R [t][n_head][n_rel] = clip(r[t] * rel_clip, rel_clip_t) * tau
//   K [t][n_kv][D]    = clip(k[t], qk_clip),                  V [t][n_kv][D] = rmsnorm(v[t]) (no gain)
// q and r are read at qr + t * ld_qr (Q's n_head * D columns, then R's n_head * n_rel), k and v at row t of [T][n_kv * D].
// tau = 1 + alpha log(max(1, (pos + 1) / floor)) when `full`, else 1.
void heads(sycl::queue& q, const float* qr, int64_t ld_qr, const float* k, const float* v, const int32_t* pos, int64_t T,
           int64_t n_head, int64_t n_kv, int64_t D, int64_t n_rel, const float* q_clip, const float* rel_clip, float qk_clip,
           float rel_clip_t, bool full, float alpha, float floor, float eps, float* Q, float* R, float* K, float* V);

// att[t][i] *= sigmoid(g[t * ld_g + i]), i < n.
void gate(sycl::queue& q, float* att, const float* g, int64_t ld_g, int64_t T, int64_t n);

// out[t * ld_out + j] = w[j] . x[t * ld_x + :] for fp32 w [n][k]: one sub-group per output.
void matmul_f32(sycl::queue& q, const float* w, int64_t n, int64_t k, const float* x, int64_t ld_x, int64_t T, float* out,
                int64_t ld_out);

// ref::inkling::control_input: out [T][2 n_embd].
void control_input(sycl::queue& q, const float* cur, const float* msg, int64_t T, int64_t n_embd, float eps, float* out);

// out[r][:] = x[idx[r]][:] for r < rows, n columns.
void gather_rows(sycl::queue& q, const float* x, const int32_t* idx, int64_t rows, int64_t n, float* out);
// out[r][j] = x[r * ld + j] for j < n: the first n columns of rows of ld.
void gather_cols(sycl::queue& q, const float* x, int64_t ld, int64_t rows, int64_t n, float* out);

// The router's choice, as ref::inkling::route: per token the n_used experts of the highest sigmoid(logit) + bias (ties to the
// lower id), weighted by their sigmoid(logit), normalized and scaled. A sub-group a token (n_expert a multiple of 16, at
// most 256; n_used at most 16).
void route(sycl::queue& q, const float* logits, const float* bias, int64_t T, int64_t n_expert, int64_t n_used, float w_scale,
           int32_t* ids, float* weights);

// A loop controller at a few tokens, as ref control_input, the down projection, gain and forget and ref control_mix, in two
// kernels: cd [T][rank] (scratch) = down . [cur/rms(cur), msg/rms(msg)]; out = cur - tanh(forget . cd) cur + tanh(gain . cd)
// msg. down [rank][2 n_embd], gain and forget [n_embd][rank], fp32.
void control(sycl::queue& q, const float* cur, const float* msg, const float* down, const float* gain, const float* forget, int64_t T,
             int64_t n_embd, int64_t rank, float eps, float* cd, float* out);

// LoopCD's hidden-state contrast (arXiv 2610.02185): y <- y + w (y - first), over n values.
void contrast(sycl::queue& q, float* y, const float* first, float w, int64_t n);

// The step's routing and expert inputs into host memory (USM host allocations), for the cold experts: ids and w
// [T][n_used], x [T][n_embd]. Returns the kernel's event (the host waits on it alone).
sycl::event pack_host(sycl::queue& q, const int32_t* ids, const float* w, const float* x, int64_t T, int64_t n_used, int64_t n_embd,
                      int32_t* hids, float* hw, float* hx);

}  // namespace eklow::inkling
