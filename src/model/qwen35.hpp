// Qwen3.5 / 3.6 MoE (llama.cpp's qwen35moe: Qwen3.6-35B-A3B and its finetunes) from a GGUF: its
// configuration, its weights as stored, and the reference forward pass, which follows llama.cpp's
// graph (src/models/qwen35moe.cpp at the oracle's revision) in fp32:
//   x = embed(token)
//   per layer: h = rmsnorm(x) * attn_norm
//     Gated DeltaNet layers (all but every 4th: 3, 7, ..., 39):
//       qkv = h Wqkv (8192 = q 16 heads x 128, k 16 x 128, v 32 x 128); z = h Wgate (4096)
//       beta = sigmoid(h Wbeta) (32); g = (softplus(h Walpha + dt_bias)) * ssm_a (ssm_a = -exp(A_log))
//       qkv = silu(causal_conv4(qkv)) (a zero state before the first token); q, k l2-normalized per head
//       S[h] (128 x 128) = S[h] * exp(g); S += k (beta * (v - S^T k))^T; out = S^T (q / sqrt(128)); value
//       head h reads key head h % 16 (ggml_repeat tiles; the converter stores the value heads in tiled order)
//       a = (rmsnorm(out) * ssm_norm * silu(z)) Wout
//     gated attention layers: [q | gate] per head = h Wq (16 x 2 x 256), q and k (2 heads) RMS-normed per
//       head, NeoX rope over the first 64 of 256 dims at base 1e7 (IMRoPE for text: all position
//       components equal), softmax(q.k / 16) v causal, times sigmoid(gate), Wo
//     x = x + a
//   MoE on rmsnorm(x) * post_attention_norm = m: 256 experts, p = softmax(m Wrouter), the top 8 weights
//     p_i / sum(top 8) (floored at 2^-14), experts SwiGLU of 512 (gate = first half of a merged gate_up),
//     plus the shared expert (SwiGLU of 512) times sigmoid(m . ffn_gate_inp_shexp); x = x + that
//   logits = (rmsnorm(x) * output_norm) Wout
// Layer weights are 450 MB and the whole file is 20 GB, so they are copied to the card one layer at a time.
#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "model/gemma4.hpp"  // DumpSink
#include "model/gguf.hpp"

namespace eklow::model {

struct Qwen35Config {
  int n_layer, n_embd, n_head, n_head_kv, head_dim, n_vocab, n_rot;
  float rope_base, eps, expert_weights_scale = 0.0f;
  int n_expert, n_expert_used, n_ff_exp, n_ff_shexp;
  int ssm_conv, ssm_inner, ssm_head_dim, ssm_k_heads, ssm_v_heads;
  std::vector<bool> recurrent;  // per layer: Gated DeltaNet (true) or attention

  // Throws when the file is not a qwen35moe this code knows how to run. A NextN/MTP block after the
  // trunk is ignored.
  static Qwen35Config from_gguf(const gguf::File& f);
  int conv_dim() const { return 2 * ssm_k_heads * ssm_head_dim + ssm_inner; }
};

// One layer's tensors on the card, as stored; frees them when destroyed.
struct Qwen35Layer {
  ref::Weight wqkv, wz, w_beta, w_alpha, ssm_out;  // Gated DeltaNet
  const float *conv1d = nullptr, *dt_bias = nullptr, *ssm_a = nullptr, *ssm_norm = nullptr;
  ref::Weight wq, wk, wv, wo;  // attention
  const float *q_norm = nullptr, *k_norm = nullptr;
  const float *attn_norm = nullptr, *post_norm = nullptr;
  ref::Weight router, shexp_gate_inp, shexp_gate, shexp_up, shexp_down;
  // Routed experts: gate_up is either one merged tensor (gate = first n_ff_exp rows of each expert) or
  // separate gate and up tensors.
  // Weights of ONE expert (n = its rows); expert e starts e * n * row_bytes in (the unused ones have data null).
  ref::Weight exp_gate_up, exp_gate, exp_up, exp_down;

  Qwen35Layer() = default;
  Qwen35Layer(Qwen35Layer&& o) noexcept { *this = std::move(o); }
  Qwen35Layer& operator=(Qwen35Layer&& o) noexcept;
  ~Qwen35Layer();
  std::vector<void*> allocations;
  sycl::queue* q = nullptr;
};

class Qwen35Weights {
 public:
  // Copies the embedding, the output head and the output norm to the card; layers stay in the mapped file.
  static Qwen35Weights upload(const gguf::File& f, const Qwen35Config& c, sycl::queue& q);
  Qwen35Weights(Qwen35Weights&&) noexcept;
  Qwen35Weights& operator=(Qwen35Weights&&) = delete;
  Qwen35Weights(const Qwen35Weights&) = delete;
  ~Qwen35Weights();

  // Copies layer l's tensors to the card.
  Qwen35Layer load_layer(int l) const;

  ref::Weight tok_embd, output;
  const float* output_norm = nullptr;
  uint64_t bytes = 0;  // resident part

 private:
  Qwen35Weights(const gguf::File& f, const Qwen35Config& c, sycl::queue& q) : f_(&f), c_(&c), q_(&q) {}
  const gguf::File* f_;
  const Qwen35Config* c_;
  sycl::queue* q_;
  std::vector<void*> allocations_;
};

// The reference forward over one sequence (positions 0..tokens-1); returns logits for the last
// `logits_rows` positions, [row][vocab]. Dump names and shapes are llama.cpp's graph tensors (attn_norm-0,
// linear_attn_out-0, Qcur-3, ffn_moe_logits-0, l_out-39, result_norm). When `routing` is non-null, the selected
// expert ids are appended to it, layer by layer and, within a layer, token by token, n_expert_used ids per
// token (best first).
std::vector<float> reference_forward(sycl::queue& q, const Qwen35Config& c, const Qwen35Weights& w,
                                     const std::vector<int32_t>& tokens, int64_t logits_rows,
                                     const DumpSink* dump, std::vector<int32_t>* routing = nullptr);

}  // namespace eklow::model
