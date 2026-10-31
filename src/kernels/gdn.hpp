// Gated DeltaNet layers' serving kernels (Qwen3.6-35B-A3B, llama.cpp's `qwen35moe`, build_layer_attn_linear): everything
// between the input matmuls and ssm_out, for a step that holds tokens of several slots (sequences), each slot's tokens
// contiguous and in order, as the engine's steps are: prefill chunks, single decode tokens, DFlash verify blocks.
//
// One token t of one slot, per layer, from x = attn_qkv . h [8192], z = attn_gate . h [4096], a = ssm_alpha . h [32],
// b = ssm_beta . h [32] (the engine's matmuls):
//   c = silu(conv4(x)) per channel over x at t-3..t (earlier rows from the slot's conv state);
//   q, k = l2_norm of c's first two 2048 slices as 16 heads of 128 (x / sqrt(sum x^2 + eps)); v = the rest, 32 heads of
//   128. The GGUF stores V heads tiled: value head h reads key head h % 16;
//   g = ssm_a * softplus(a + dt_bias), beta = sigmoid(b); per value head, state S (128 x 128, fp32), q scaled by 1/sqrt(128):
//     S *= exp(g); kv = S^T k; d = beta (v - kv); S += k d^T; o = S^T q;
//   y = rmsnorm(o_h) * norm_w * silu(z_h), into the activations ssm_out's matmul takes.
// Three kernels do it: prepare (conv, silu, l2 norm, g, beta), scan (the recurrence, one thread holding a few columns of a
// head's state in registers and walking the slot's tokens) and gated_norm (the norm, silu gate and quantization).
//
// Speculative rollback without a state copy: each slot's step carries `accept` and `commit` counts (SlotStep).
// A verify block of B tokens runs with commit 0: its outputs come from the committed state walked through B tokens in
// registers, nothing is written back to the state, and the tokens' post-conv inputs (q, k, v, exp(g), beta) and raw
// pre-conv rows are saved in the slot's *pending* buffer. The slot's next step says how many of them were accepted:
// the scan replays those onto the committed state first (the same code on the same saved values, so the result is
// bit-identical to having run only those tokens), the conv state advances through them the same way, and the rest are
// dropped. A step may also leave only its last tokens pending (commit < n), e.g. to commit a prefix and verify the rest.
// Plain prefill and decode commit everything.
//
// Layouts (this file's own; all fp32):
//   ssm state per (slot, layer): [32 value heads][128 value index][128 key index], so a column's 128 rows are contiguous;
//   conv state per (slot, layer): [3][8192]: the last three pre-conv rows of x, oldest first (zeros at a sequence's start);
//   pending per (slot, layer): inputs [max_pending][8192] (q | k | v, post-conv, q already scaled), [max_pending][64]
//   (exp(g)[32] | beta[32]) and raw x rows [max_pending][8192].
// A slot's states for all layers are contiguous (one copy each to and from the host for a snapshot).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

#include "kernels/xmx.hpp"

namespace eklow::gdn {

constexpr int kKeyHeads = 16, kValueHeads = 32, kHead = 128, kConvTaps = 4;
constexpr int kQk = kKeyHeads * kHead;       // 2048: q's (and k's) width
constexpr int kInner = kValueHeads * kHead;  // 4096: v's, o's, z's and y's width
constexpr int kChannels = 2 * kQk + kInner;  // 8192: x's width
constexpr int kMaxPendingLimit = 16;         // most tokens a slot may leave pending

// One slot's part of a step; device arrays of these (the layout is what the kernels read).
struct SlotStep {
  int32_t slot;    // which slot's state
  int32_t tok0;    // first row of its tokens in the step's arrays
  int32_t n;       // new tokens
  int32_t accept;  // of the tokens left pending by its previous step, how many were accepted (0..pending)
  int32_t commit;  // of the n new tokens, how many are written to the state (0..n); the rest stay pending
  int32_t pad[3];
};

// What the host must remember per slot: how many tokens its last step left pending. next() builds the SlotStep for its
// next step: `accept` of the pending ones (all of them when negative), `n` new tokens at row tok0, `commit` of them
// (all when negative), and updates `pending`. Throws when the counts do not fit.
struct SlotBook {
  int32_t pending = 0;
  SlotStep next(int32_t slot, int32_t tok0, int32_t n, int32_t accept = -1, int32_t commit = -1, int32_t max_pending = kMaxPendingLimit);
};

// A step's slots, on the card and the host. n_tokens is the sum of the slots' n.
struct Plan {
  const SlotStep* dev = nullptr;
  int32_t n_slots = 0, n_tokens = 0, max_n = 0;
};
// Uploads (to a device allocation that free_plan frees) / frees.
Plan make_plan(sycl::queue& q, const std::vector<SlotStep>& slots);
void free_plan(sycl::queue& q, Plan& p);

// The recurrent state of `n_slots` slots over `n_layers` layers.
struct Cache {
  int32_t n_layers = 0, n_slots = 0, max_pending = 0;
  float* ssm = nullptr;      // [slot][layer][32][128][128]
  float* conv = nullptr;     // [slot][layer][3][8192]
  float* pend_in = nullptr;  // [slot][layer][max_pending][8192]
  float* pend_gb = nullptr;  // [slot][layer][max_pending][64]
  float* pend_x = nullptr;   // [slot][layer][max_pending][8192]
  int64_t ssm_floats_per_layer() const { return int64_t(kValueHeads) * kHead * kHead; }
  int64_t conv_floats_per_layer() const { return int64_t(3) * kChannels; }
  // One slot's committed state, all layers: the snapshot's two pieces.
  int64_t ssm_bytes() const { return int64_t(n_layers) * ssm_floats_per_layer() * 4; }
  int64_t conv_bytes() const { return int64_t(n_layers) * conv_floats_per_layer() * 4; }
  int64_t snapshot_bytes() const { return ssm_bytes() + conv_bytes(); }
};
Cache alloc_cache(sycl::queue& q, int n_layers, int n_slots, int max_pending);
void free_cache(sycl::queue& q, Cache& c);
// Zeroes a slot's state (the start of a sequence); only enqueues. Pending contents need no reset.
void reset_slot(sycl::queue& q, const Cache& c, int slot);
// A slot's committed state to / from host memory (snapshot_bytes bytes: ssm then conv; pinned memory copies fastest).
// The slot must have nothing pending that is still to be accepted (step it with n = 0 and accept = pending first).
// Enqueued; returns the event, and the host buffer must stay put until it completes.
sycl::event snapshot_to_host(sycl::queue& q, const Cache& c, int slot, void* host);
sycl::event restore_from_host(sycl::queue& q, const Cache& c, int slot, const void* host);

struct Weights {
  const float* conv_w;   // [8192][4]: the 4 taps of each channel, tap 3 on the newest row (llama.cpp's ssm_conv1d)
  const float* ssm_a;    // [32], -exp(A_log)
  const float* dt_bias;  // [32]
  const float* norm_w;   // [128]: ssm_norm
  float eps;             // the norms' epsilon
};

// Per-step buffers shared by all layers (every kernel of a layer runs before the next layer's).
struct Scratch {
  float* in = nullptr;  // [n_tokens][8192]: q | k | v after conv, silu, l2 norm (q scaled)
  float* gb = nullptr;  // [n_tokens][64]: exp(g) | beta
  float* o = nullptr;   // [n_tokens][4096]: the scan's output
};
Scratch alloc_scratch(sycl::queue& q, int64_t max_tokens);
void free_scratch(sycl::queue& q, Scratch& s);

// A layer's inputs for the step: x [t][ldx], z [t][ldz], a and b [t][lda], [t][ldb] (row t = step row, tok0 + j).
struct Inputs {
  const float *x, *z, *a, *b;
  int64_t ldx = kChannels, ldz = kInner, lda = kValueHeads, ldb = kValueHeads;
};

// The kernels, one layer. They only enqueue (the queue must be in order) and return the kernel's event, for timing.
sycl::event prepare(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s);
sycl::event scan(sycl::queue& q, const Cache& c, int layer, const Plan& p, const Scratch& s);
// y = rmsnorm(o_h) * norm_w * silu(z_h) into `out` (set_rows(out, n_tokens, 4096): int8 up to xmx::kFp16Rows rows, fp16
// above), and the tokens left pending saved into the pending buffers.
sycl::event gated_norm(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s,
                xmx::Activations& out);
// All three.
sycl::event step(sycl::queue& q, const Weights& w, const Cache& c, int layer, const Plan& p, const Inputs& in, const Scratch& s,
          xmx::Activations& out);

}  // namespace eklow::gdn
