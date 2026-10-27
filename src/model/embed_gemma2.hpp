// EmbeddingGemma 2 (gemma-embedding2, 271M) from a GGUF, following llama.cpp's graph
// (src/models/gemma-embedding2.cpp at the oracle's revision):
//   x = embed(token) * sqrt(n_embd)
//   per-layer inputs: P = rmsnorm_per_layer(per_layer_model_proj . x / sqrt(n_embd)) * proj_norm
//   per layer: Gemma's block with attention in both directions (sliding layers: keys within
//     window/2 positions either side), then
//     x = (x + rmsnorm(proj . (gelu(inp_gate . x) * P[layer])) * post_norm) * layer_output_scale
//   e = output . (rmsnorm(x) * output_norm), averaged over the sequence and L2-normalized.
// Each input is one sequence: BOS, its tokens, EOS.
//
// A request's inputs go through together, in passes of up to max_tokens tokens: the weights' matrices run as
// fp16 GEMMs on the matrix units (xmx, the Q8_0 and F16 forms) over every row of the pass, attention runs input by
// input, and each input is averaged on the card. One input at a time on the reference kernels gave ~900 tok/s
// (DECISIONS 2026-10-27, "EmbeddingGemma 2 in GEMMs").
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"

namespace eklow::model {

class EmbedGemma2 {
 public:
  // Copies the weights to the card.
  EmbedGemma2(sycl::queue& q, const gguf::File& f);
  ~EmbedGemma2();
  EmbedGemma2(const EmbedGemma2&) = delete;
  EmbedGemma2& operator=(const EmbedGemma2&) = delete;

  // The normalized embedding of one sequence of token ids.
  std::vector<float> embed(const std::vector<int32_t>& tokens);
  // Those of several, in passes of up to max_tokens() tokens, with a single wait for the card.
  std::vector<std::vector<float>> embed(const std::vector<std::vector<int32_t>>& seqs);

  int64_t dim() const { return n_out_; }
  int64_t max_tokens() const { return max_tok_; }
  // Allocates the activations for passes of up to n_tok tokens (the longest input) now, so that a request never
  // allocates.
  void reserve(int64_t n_tok) { ensure(n_tok); }
  uint64_t bytes() const { return bytes_; }

 private:
  struct Layer {
    xmx::Weight wq, wk, wv, wo, gate, up, down, inp_gate, proj;
    const float *attn_norm, *q_norm, *k_norm, *attn_post_norm, *ffn_norm, *ffn_post_norm, *post_norm;
    float out_scale;
    bool sliding;
    int64_t n_head_kv, head_dim;
  };
  float* buf(size_t floats);
  void ensure(int64_t n_tok);
  // Enqueues one pass over seqs[i0, i1) (at most max_tok_ tokens together) and the copy of their averaged last
  // layers, [i1 - i0][dim], to `means` on the host.
  // The pass's host arrays go into `keep`, which must live until the queue has run the copies from them.
  void forward(const std::vector<std::vector<int32_t>>& seqs, size_t i0, size_t i1, float* means, std::vector<std::vector<int32_t>>& keep);

  sycl::queue& q_;
  std::vector<void*> owned_;
  std::vector<xmx::Weight> repacked_;
  uint64_t bytes_ = 0;
  int64_t n_embd_, n_ff_, n_head_, n_layer_, n_pl_, n_out_, window_, max_tok_ = 0;
  float eps_, rope_base_s_, rope_base_g_;
  ref::Weight tok_embd_;
  xmx::Weight pl_proj_, output_;
  const float *pl_norm_, *output_norm_;
  std::vector<Layer> layers_;
  // activations and RoPE tables for passes of up to max_tok_ tokens
  std::vector<float*> act_;
  xmx::Activations a_, a0_;  // fp16 inputs of the GEMMs; a0_ holds the scaled embeddings for the per-layer inputs
};

}  // namespace eklow::model
