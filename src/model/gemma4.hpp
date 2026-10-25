// Gemma 4 (dense: the 12B) from a GGUF: its configuration, its weights on the card as stored, and
// the reference forward pass, which follows llama.cpp's gemma4 graph (src/models/gemma4.cpp at the
// oracle's revision) in fp32:
//   x = embed(token) * sqrt(n_embd)
//   per layer: h = rmsnorm(x) * attn_norm
//     q = rmsnorm_per_head(h Wq) * q_norm, roped
//     k = rmsnorm_per_head(h Wk) * k_norm, roped; v = rmsnorm_per_head(h Wv or, on global layers
//     that have no Wv, h Wk), unweighted
//     a = attention(q, k, v; scale 1, causal, window 1024 on sliding layers) Wo
//     x = rmsnorm(a) * attn_post_norm + x
//     f = (gelu_tanh(rmsnorm(x) * ffn_norm Wgate) * (... Wup)) Wdown
//     x = (rmsnorm(f) * ffn_post_norm + x) * layer_output_scale
//   logits = softcap_30(rmsnorm(x) * output_norm . token_embd^T)
// Sliding layers rope all 256 dims at base 10000; global layers all 512 at base 1e6, each frequency
// divided by rope_freqs (Gemma 4's proportional RoPE).
#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "model/gguf.hpp"

namespace eklow::model {

struct Gemma4Config {
  int n_layer, n_embd, n_ff, n_head, n_vocab, window;
  std::vector<bool> sliding;  // per layer
  std::vector<int> n_head_kv;
  int head_dim_sliding, head_dim_global;
  float rope_base_sliding, rope_base_global, eps, softcap;

  // Throws when the file is not a dense Gemma 4 this code knows how to run.
  static Gemma4Config from_gguf(const gguf::File& f);
  int head_dim(int layer) const { return sliding[size_t(layer)] ? head_dim_sliding : head_dim_global; }
};

class Gemma4Weights {
 public:
  struct Layer {
    ref::Weight wq, wk, wv, wo, gate, up, down;  // wv.data is null on global layers
    const float *attn_norm, *q_norm, *k_norm, *attn_post_norm, *ffn_norm, *ffn_post_norm;
    float out_scale;
  };

  // Copies every tensor to device memory as stored.
  static Gemma4Weights upload(const gguf::File& f, const Gemma4Config& c, sycl::queue& q);
  Gemma4Weights(Gemma4Weights&&) noexcept;
  Gemma4Weights& operator=(Gemma4Weights&&) = delete;
  Gemma4Weights(const Gemma4Weights&) = delete;
  ~Gemma4Weights();

  ref::Weight tok_embd;
  const float* output_norm = nullptr;
  std::vector<float> rope_freqs;  // host copy, for the RoPE tables
  std::vector<Layer> layers;
  uint64_t bytes = 0;

 private:
  explicit Gemma4Weights(sycl::queue& q) : q_(&q) {}
  sycl::queue* q_;
  std::vector<void*> allocations_;
};

// Receives intermediate tensors under llama.cpp's graph names (attn_norm-0, Qcur_normed-0, ...,
// l_out-47, result_norm) with ggml shapes (ne[0] fastest). `wants` is asked before each tensor is
// copied off the card.
struct DumpSink {
  std::function<bool(const std::string&)> wants;
  std::function<void(const std::string& name, const std::vector<float>& data, const std::vector<int64_t>& ne)> put;
};

// The reference forward over one sequence (positions 0..tokens-1); returns logits for the last
// `logits_rows` positions, [row][vocab].
std::vector<float> reference_forward(sycl::queue& q, const Gemma4Config& c, const Gemma4Weights& w,
                                     const std::vector<int32_t>& tokens, int64_t logits_rows, const DumpSink* dump);

}  // namespace eklow::model
