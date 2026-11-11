// Qwen3-Reranker (llama.cpp's `qwen3` with rank pooling: ggml-org's Qwen3-Reranker-0.6B GGUF, which carries the yes/no
// head as cls.output and the prompt as tokenizer.chat_template.rerank) as llama.cpp's server scores a query-document pair
// (tools/server, format_prompt_rerank and send_rerank, and llama-graph.cpp's rank pooling at 7d7f028):
//   prompt = the rerank template with {query} and {document} replaced (one scan, the inserted text not scanned again),
//   tokenized with special tokens parsed and no BOS;
//   Qwen3: 28 layers of hidden 1024; attention with 16 query heads over 8 K/V heads of 128, each q and k head RMS-normed
//   (q_norm, k_norm) and NeoX-rotated over all 128 dims at base 1e6; SwiGLU of 3,072; RMSNorm eps 1e-6;
//   score = softmax(cls.output . output_norm(h_last))[0], the probability of "yes".
// Pairs are read whole, several to a pass: their tokens fill a pass's rows one pair after another and each attends to its
// own (causal, keys from its first token), so a pass needs the K/V of one layer at a time. A pair is at most `max_tokens`
// tokens (llama.cpp scores a pair in one ubatch: 4,096 in fiona's deployment).
//
// Arithmetic: the matrices are xmx's tiles of the file's Q8_0 weights under fp16 activations with fp32 accumulation;
// norms, RoPE, the residual stream and the head are fp32; attention is kernels/attention.hpp's over an fp16 cache.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"
#include "text/tokenizer_qwen.hpp"

namespace eklow::model {

class RerankQwen3 {
 public:
  // Copies the weights to the card (the token table stays in the mapped file). max_tokens: the longest pair, and the rows of
  // a pass.
  RerankQwen3(sycl::queue& q, const gguf::File& f, int64_t max_tokens = 4096);
  ~RerankQwen3();
  RerankQwen3(const RerankQwen3&) = delete;
  RerankQwen3& operator=(const RerankQwen3&) = delete;

  // The prompt of one pair, as tokens.
  std::vector<int32_t> prompt(const std::string& query, const std::string& document) const;
  // The probability of "yes" for each tokenized pair, in order. Throws std::length_error for a pair longer than
  // max_tokens().
  std::vector<double> score(const std::vector<std::vector<int32_t>>& pairs);

  int64_t max_tokens() const { return max_tok_; }
  uint64_t bytes() const { return bytes_; }
  const text::QwenTokenizer& tokenizer() const { return *tok_; }

 private:
  struct Layer {
    xmx::Weight qkv, o, gate_up, down;
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm;
  };
  float* dev(size_t floats, const char* what);
  // One pass over pairs [i0, i1) (their tokens together at most max_tok_): their last rows' outputs into `last` (host).
  void pass(const std::vector<std::vector<int32_t>>& pairs, size_t i0, size_t i1, std::vector<float>& last);

  sycl::queue& q_;
  const gguf::File& f_;
  std::unique_ptr<text::QwenTokenizer> tok_;
  std::string template_;
  int64_t E_, F_, H_, Hkv_, D_, n_layer_, max_tok_;
  float eps_;
  uint64_t bytes_ = 0;
  const gguf::Tensor* embd_;
  std::vector<float> cls_, out_norm_;  // [2][E] (yes, no), [E]: the head, on the host
  std::vector<Layer> layers_;
  std::vector<void*> owned_;
  std::vector<xmx::Weight> repacked_;
  float *x_, *out_, *qkv_, *Q_, *K_, *V_, *att_, *inv_freq_;
  int32_t *pos_, *cpos_, *slot_, *lo_, *hi_, *tiles_, *table_;
  xmx::Activations a_, a2_;
  sycl::half *kc_, *vc_;
  int64_t pages_ = 0, max_tiles_ = 0;
  attn::Scratch scratch_;
};

}  // namespace eklow::model
