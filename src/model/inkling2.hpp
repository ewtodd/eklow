// Inkling 2 Mini (thinkingmachines/Inkling-2-Mini, GGUF architecture inkling2_loop) from a GGUF: its configuration,
// its weights as stored, and the reference forward pass in fp32, which follows llama.cpp c74c71a6's
// src/models/inkling2.cpp and inkling2-loop.cpp (DECISIONS 2026-11-24):
//   x = rmsnorm(embed(token)) * tok_norm
//   layers 0-5, then layers 6-17 three times (passes), then 18-23: 48 layer applications, each a cache slot with its own
//   K/V and convolution state (pass p > 0 of layer l is slot 24 + 12 (p - 1) + l - 6)
//   per layer, with `blocks` the outputs of the earlier blocks of 6 layers (attention residuals):
//     inputs = x, or depth_mix(blocks + [x]) by the layer's queries (one on sliding layers, Q/K/V on full ones)
//     on passes 2 and 3: input_j = control_j(input_j, message_j), messages mixed from the previous pass's last block
//     output and its output: (1 - tanh(forget)) * input + tanh(gain) * message, gates of rank 64 on both, normed
//     Q = input_q Wq, K = conv(input_k Wk), V = conv(input_v Wv), R = input_q Wrel (16 a head), sink = input_q Wsink
//     (full layers add input-dependent conv taps per K/V head); Q, K, R L2-clipped (Q and R after a per-channel scale),
//     V RMS-normed without a gain; full layers multiply Q and R by 1 + 0.1 log(max(1, (pos + 1) / 8192))
//     a = attention(scale 1/sqrt(128), bias R . table[distance] (0 past the table), the sink as a key of logit sink and
//     no value; window 1024 on sliding layers) * sigmoid(input_q Wgate), o = conv(a Wo)
//     a block's first layer pushes its input to `blocks` and restarts the stream at o; other layers add o
//     f = conv(MoE(rmsnorm(depth_mix(blocks + [x]) (controlled on passes 2-3)) * ffn_norm)); x += f
//   MoE: sigmoid router with a selection bias, top 8 of 128, weights normalized and scaled by 8, fused gate|up and down
//   (NVFP4 with a per-expert scale on layers 0-19), SITU activation (gate soft-cap 4, up soft-cap 25)
//   logits = softcap_30(rmsnorm(depth_mix(blocks + [x])) * output_norm . output^T), the first 200,058 of 201,024
// conv is a causal convolution of kernel 4 with a residual, scaled by 1/sqrt(5) (1/3 with the dynamic taps).
#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "model/dump.hpp"
#include "model/gguf.hpp"

namespace eklow::model {

struct Inkling2Config {
  int n_layer = 0;  // physical layers (24)
  int n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0;
  int n_vocab = 0;      // rows of the embedding and output tables (padded, 201,024)
  int n_vocab_out = 0;  // the tokenizer's vocabulary: logits beyond it are dropped (200,058)
  int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
  int n_rel = 0, rel_extent = 0, window = 0, n_conv = 0, block = 0, log_floor = 0;
  int core_start = 0, core_end = 0, n_passes = 0, gate_rank = 0;
  float eps = 0, qk_clip = 0, rel_clip = 0, log_alpha = 0, situ_beta = 0, situ_linear_beta = 0, w_scale = 0, softcap = 0;
  std::vector<bool> sliding;  // per physical layer

  // Throws when the file is not an Inkling 2 this code runs.
  static Inkling2Config from_gguf(const gguf::File& f);
  int n_core() const { return core_end - core_start; }
  int n_slot() const { return n_layer + (n_passes - 1) * n_core(); }
  // The physical layer of a cache slot.
  int phys(int slot) const { return slot < n_layer ? slot : core_start + (slot - n_layer) % n_core(); }
  // The cache slot of physical layer `layer` on pass `pass` (0-based; pass > 0 only for the looped core).
  int slot(int layer, int pass) const { return pass == 0 ? layer : n_layer + (pass - 1) * n_core() + (layer - core_start); }
  // Channels of one slot's convolution state: K, V, the attention output and the FFN output.
  int conv_channels() const { return 2 * n_head_kv * head_dim + 2 * n_embd; }
};

class Inkling2Weights {
 public:
  struct Experts {
    ref::Weight w;  // one expert's matrix: w.k inputs, w.n outputs; expert e at w.data + e * stride
    uint64_t stride = 0;
    const float* scale = nullptr;  // per expert (NVFP4's second level), or null
  };
  struct Layer {
    ref::Weight wq, wk, wv, wo, gate, rel, sink, k_conv_proj, v_conv_proj, router;  // conv_proj only on full layers
    Experts gate_up, down;
    const float *attn_norm = nullptr, *q_inp_norm = nullptr, *k_inp_norm = nullptr, *v_inp_norm = nullptr;
    const float *q_clip = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    const float *k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    const float *attn_res_query = nullptr, *ffn_res_query = nullptr, *ffn_norm = nullptr, *probs_bias = nullptr;
  };
  struct Loop {  // a looped layer's message queries (Q, K, V, FFN) and controllers
    const float* msg_query = nullptr;
    ref::Weight down[4], gain[4], forget[4];
  };

  // Every tensor the forward reads, on the queue's device as stored. On a CPU device the mapping is used in place.
  static Inkling2Weights upload(const gguf::File& f, const Inkling2Config& c, sycl::queue& q);
  Inkling2Weights(Inkling2Weights&&) noexcept;
  Inkling2Weights& operator=(Inkling2Weights&&) = delete;
  Inkling2Weights(const Inkling2Weights&) = delete;
  ~Inkling2Weights();

  ref::Weight tok_embd, output;
  const float *tok_norm = nullptr, *output_norm = nullptr, *output_res_query = nullptr;
  std::vector<Layer> layers;  // physical
  std::vector<Loop> loops;    // core_start..core_end-1
  uint64_t bytes = 0;

 private:
  explicit Inkling2Weights(sycl::queue& q) : q_(&q) {}
  sycl::queue* q_;
  std::vector<void*> allocations_;
};

// The reference forward over one sequence (positions 0..tokens-1, fresh caches); returns the logits of the last
// `logits_rows` positions, [row][n_vocab_out]. Tensors go to `dump` under llama.cpp's graph names with the slot as the
// layer number (l_out-47, Qcur-29, ffn_moe_topk-5, result_norm, ...), for positions dump_from.. only.
// teacher, when given, is asked after every layer application for "l_out-<slot>" (all positions, [T][n_embd]); when it
// fills the vector the stream continues from that instead of eklow's own (a check of each layer on the reference's
// inputs). emulate_ggml_cpu rounds each matmul's activations as ggml's CPU backend does (ref::inkling::ggml_cpu_acts), so
// that the reference computes the products llama.cpp's CPU oracle computes.
using Teacher = std::function<bool(const std::string& name, std::vector<float>& data)>;
std::vector<float> inkling2_reference_forward(sycl::queue& q, const Inkling2Config& c, const Inkling2Weights& w,
                                              const std::vector<int32_t>& tokens, int64_t logits_rows,
                                              const DumpSink* dump, int64_t dump_from = 0,
                                              const Teacher* teacher = nullptr, bool emulate_ggml_cpu = false);

}  // namespace eklow::model
