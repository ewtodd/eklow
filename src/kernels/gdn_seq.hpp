// Gated DeltaNet layers of a dense Qwen3.5 (llama.cpp's `qwen35`: Qwen3.5-0.8B, vega's reader) over a few sequences read in
// chunks, for any head count with as many value heads as key heads. kernels/gdn.hpp serves Qwen3.6-35B-A3B's layout (16
// key and 32 value heads, verify blocks, rollback) and is built around it; the reader needs none of that, only each
// sequence's state carried from one chunk to the next and, for prompts that share a long prefix, copied aside and back.
//
// One token t of one layer, from x = attn_qkv . h [3 * H * 128] (q | k | v), a = ssm_alpha . h [H], b = ssm_beta . h [H]:
//   c = silu(conv4(x)) per channel over x at t-3..t (rows before the step from the sequence's conv state, zeros at its
//   start);
//   q, k = c's first two slices as H heads of 128, each times 1 / sqrt(sum x^2 + 1e-6) (Hugging Face's l2norm, which the
//   reference runs), q also by 1 / sqrt(128); v = the rest;
//   g = ssm_a * softplus(a + dt_bias), beta = sigmoid(b); per head, state S (128 x 128, fp32):
//     S *= exp(g); kv = S^T k; d = beta (v - kv); S += k d^T; o = S^T q;
//   y = rmsnorm(o_h) * norm_w * silu(z_h), into fp16 activations for ssm_out's GEMM.
// A step's rows are runs, one per sequence, each a sequence's next tokens in order.
// Layouts (fp32): state [slot][layer][H][128 value index][128 key index] (a column's rows contiguous, as kernels/gdn.hpp
// keeps them); conv state [slot][layer][3][3 * H * 128], the last three raw rows of x, oldest first.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/xmx.hpp"

namespace eklow::gdn_seq {

constexpr int kHead = 128, kTaps = 4;

struct State {
  int32_t n_layers = 0, n_heads = 0, n_slots = 0;
  float* ssm = nullptr;   // [slot][layer][H][128][128]
  float* conv = nullptr;  // [slot][layer][3][3 H 128]
  int64_t channels() const { return int64_t(3) * n_heads * kHead; }
  int64_t ssm_floats() const { return int64_t(n_layers) * n_heads * kHead * kHead; }  // one slot's
  int64_t conv_floats() const { return int64_t(n_layers) * 3 * channels(); }
};
State alloc_state(sycl::queue& q, int n_layers, int n_heads, int n_slots);
void free_state(sycl::queue& q, State& s);
// Zeroes a slot (a sequence's start); only enqueues.
void reset(sycl::queue& q, const State& s, int slot);
// dst's slot = src's slot (states of the same layers and heads); only enqueues.
void copy(sycl::queue& q, const State& src, int src_slot, const State& dst, int dst_slot);

struct Weights {
  const float* conv_w;   // [3 H 128][4], the GGUF's ssm_conv1d (tap 3 on the newest row)
  const float* ssm_a;    // [H], -exp(A_log)
  const float* dt_bias;  // [H]
  const float* norm_w;   // [128], ssm_norm
  float eps;             // the gated norm's
};

// One sequence's rows of a step: its state's slot, its first row and its row count. A device array of these, and for
// every row of the step the index of its run.
struct Run {
  int32_t slot, row0, n, pad;
};

// Per-step buffers: in [rows][3 H 128] (q | k | v after the conv), gb [rows][2 H] (exp(g) | beta), o [rows][H 128].
struct Scratch {
  float *in = nullptr, *gb = nullptr, *o = nullptr;
};
Scratch alloc_scratch(sycl::queue& q, int64_t max_rows, int n_heads);
void free_scratch(sycl::queue& q, Scratch& s);

// One layer over a step of `rows` rows in `n_runs` runs: x [rows][3 H 128], z [rows][H 128], ab [rows][2 H] (a | b) as the
// GEMMs wrote them; the result y goes to `out` as fp16 activations of `rows` rows (xmx::quantize's fp16 form). Each run's
// state advances past its rows. Only enqueues (the queue must be in order).
void layer(sycl::queue& q, const Weights& w, const State& s, int layer, const Run* runs, int32_t n_runs, const int32_t* run_of_row,
           int64_t rows, const float* x, const float* z, const float* ab, const Scratch& sc, xmx::Activations& out);

}  // namespace eklow::gdn_seq
