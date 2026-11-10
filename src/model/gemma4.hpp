// Gemma 4 E4B (dense, with per-layer embeddings and shared K/V: NVIDIA's Nemotron-3.5-Bolt-E4B) from a GGUF: its
// configuration, its weights on the card as stored, and the reference forward pass in fp32, which follows
// transformers' Gemma4ForConditionalGeneration text model (modeling_gemma4.py, 5.22.2):
//   x = embed(token) * sqrt(n_embd)
//   per-layer inputs, once from the embeddings:
//     P[l] = (rmsnorm_256(per_layer_model_proj . x / sqrt(n_embd)) * proj_norm + per_layer_embd(token)[l] * sqrt(256))
//            / sqrt(2)
//   per layer: h = rmsnorm(x) * attn_norm
//     q = rmsnorm_per_head(h Wq) * q_norm, roped
//     on a layer that owns its K/V: k = rmsnorm_per_head(h Wk) * k_norm, roped; v = rmsnorm_per_head(h Wv), unweighted
//     (a global layer without Wv takes K's projection before K's norm); the last n_kv_shared layers have no K/V of
//     their own and read the K/V of the last layer of their kind (sliding or global) before them
//     a = attention(q, k, v; scale 1, causal, window on sliding layers) Wo
//     x = rmsnorm(a) * attn_post_norm + x
//     x = rmsnorm((gelu_tanh(rmsnorm(x) * ffn_norm Wgate) * (... Wup)) Wdown) * ffn_post_norm + x
//     x = (x + rmsnorm(proj . (gelu_tanh(inp_gate . x) * P[l])) * post_norm) * layer_output_scale
//   logits = softcap_30(rmsnorm(x) * output_norm . token_embd^T)
// Sliding layers rope all their dims at base 1e4; global layers all theirs at base 1e6, each frequency divided by
// rope_freqs (proportional RoPE: Bolt's rotates the first quarter and leaves the rest, factors of 1e30).
#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "model/dump.hpp"
#include "model/gguf.hpp"

namespace eklow::model {

struct Gemma4Config {
  int n_layer, n_embd, n_ff, n_head, n_vocab, window;
  std::vector<bool> sliding;  // per layer
  std::vector<int> n_head_kv;
  int head_dim_sliding, head_dim_global;
  float rope_base_sliding, rope_base_global, eps, softcap;
  int n_embd_per_layer = 0;  // the per-layer inputs' width (0: none)
  int n_kv_shared = 0;       // the last layers that read an earlier layer's K/V

  // Throws when the file is not a dense Gemma 4 this code knows how to run.
  static Gemma4Config from_gguf(const gguf::File& f);
  int head_dim(int layer) const { return sliding[size_t(layer)] ? head_dim_sliding : head_dim_global; }
  // The layer whose K/V `layer` attends over: itself, or for a shared layer the last layer of its kind that owns one.
  int kv_source(int layer) const;
  bool owns_kv(int layer) const { return layer < n_layer - n_kv_shared; }
};

class Gemma4Weights {
 public:
  struct Layer {
    ref::Weight wq, wk, wv, wo, gate, up, down;  // wv.data null on a global layer without one, wk and wv on a shared layer
    ref::Weight inp_gate, proj;                   // the per-layer input's gate (n_embd -> 256) and projection back
    const float *attn_norm, *q_norm, *k_norm, *attn_post_norm, *ffn_norm, *ffn_post_norm, *post_norm;
    float out_scale;
  };

  // Copies every tensor the forward reads to device memory as stored.
  static Gemma4Weights upload(const gguf::File& f, const Gemma4Config& c, sycl::queue& q);
  Gemma4Weights(Gemma4Weights&&) noexcept;
  Gemma4Weights& operator=(Gemma4Weights&&) = delete;
  Gemma4Weights(const Gemma4Weights&) = delete;
  ~Gemma4Weights();

  ref::Weight tok_embd, pl_embd, pl_proj;
  const float *output_norm = nullptr, *pl_norm = nullptr;
  std::vector<float> rope_freqs;  // host copy, for the RoPE tables
  std::vector<Layer> layers;
  uint64_t bytes = 0;

 private:
  explicit Gemma4Weights(sycl::queue& q) : q_(&q) {}
  sycl::queue* q_;
  std::vector<void*> allocations_;
};

// The reference forward over one sequence (positions 0..tokens-1); returns logits for the last `logits_rows`
// positions, [row][vocab].
std::vector<float> reference_forward(sycl::queue& q, const Gemma4Config& c, const Gemma4Weights& w,
                                     const std::vector<int32_t>& tokens, int64_t logits_rows, const DumpSink* dump);

}  // namespace eklow::model
