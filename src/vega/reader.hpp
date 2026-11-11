// vega's reader: Qwen3.5-0.8B (llama.cpp's `qwen35`, dense: 24 layers of hidden 1024, every 4th gated attention with 8
// query heads over 2 K/V heads of 256, the rest Gated DeltaNet with 16 key and 16 value heads of 128, SwiGLU of 3,584)
// from its GGUF, run on the card only as deep as vegaml reads it, and pooled as vegaml's Perception.pooled pools it
// (vegaml 0.16.1, _common.py):
//   per feature layer (vega-U: 13 and 19, the outputs of those decoder layers): the mean of the state span, the mean of
//   the question span, and the last token; per option the mean of its span at the option layers (19); and the state span
//   at the deepest feature layer cut into k = max(2, min(C, n / chunk_tokens)) pieces (token j of n in piece j k / n),
//   each averaged, when it holds at least chunk_min_tokens tokens: the engine's memory.
// Prompts are read in chunks of up to `chunk` tokens with the DeltaNet state and the attention cache carried, so a
// prompt's length is bounded by the cache (`max_tokens`), not by the activations. Prompts whose state reaches
// prefix_min tokens share their prefix (everything up to the state's end) when several of a read have the same one, as
// vegaml's _pooled_prefix does: it is read once and each continues from a copy of its DeltaNet state. The other prompts of
// a read go through together, up to kSlots at a time: their tokens fill each pass's rows one prompt after another, each
// prompt with its own DeltaNet state and its own pages of the cache (the cache holds max_tokens positions in all), so a
// request's prompts share the GEMMs' rows and their scans run side by side.
//
// Arithmetic: the matrices are fp16 tiles of the file's BF16 weights (exact) under fp16 activations with fp32
// accumulation; norms, the DeltaNet's recurrence, the residual stream and the pooling are fp32; attention is
// kernels/attention.hpp's over an fp16 cache.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/gdn_seq.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"

namespace eklow::vega {

// One prompt and the spans pooled from it, [a, b) token positions (build_prompt's).
struct Prompt {
  std::vector<int32_t> ids;
  int32_t s0 = 0, s1 = 0, q0 = 0, q1 = 0;
  std::vector<std::pair<int32_t, int32_t>> options;
};

struct Pooled {
  std::vector<float> item;    // [feature layers][state, question, last][d]
  std::vector<float> opts;    // [options][option layers][d]
  std::vector<float> chunks;  // [C][d], the first n_chunks valid
  int32_t n_chunks = 0;
};

struct PoolSpec {
  int32_t chunks = 16, chunk_min_tokens = 128, chunk_tokens = 128, prefix_min = 4096;
};

class Reader {
 public:
  static constexpr int kSlots = 4;  // prompts read side by side
  // feature_layers / option_layers: 0-based decoder layers (vega_config.json's). max_tokens: the longest prompt read()
  // takes; chunk: tokens a pass runs at once (activations scale with it).
  Reader(sycl::queue& q, const gguf::File& f, std::vector<int> feature_layers, std::vector<int> option_layers, int64_t max_tokens,
         int64_t chunk = 1024);
  ~Reader();
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;

  // Pooled vectors of each prompt, in order. Throws std::length_error when a prompt is longer than max_tokens().
  std::vector<Pooled> read(const std::vector<Prompt>& prompts, const PoolSpec& spec);

  int64_t d_model() const { return E_; }
  int64_t max_tokens() const { return max_tok_; }
  uint64_t bytes() const { return bytes_; }
  const std::vector<int>& feature_layers() const { return feat_; }
  const std::vector<int>& option_layers() const { return optl_; }

 private:
  struct Layer {
    bool recurrent;
    const float *attn_norm, *post_norm;
    xmx::Weight in, out, z, gate_up, down;  // in: attn_qkv (DeltaNet) or q|gate, k, v (attention); out: ssm_out or attn_output
    const float* ab = nullptr;              // DeltaNet: ssm_alpha | ssm_beta, fp32 [2 H][E]
    gdn_seq::Weights gdn{};
    int gdn_index = -1;
    const float *q_norm = nullptr, *k_norm = nullptr;
    int attn_index = -1;
  };
  // A pooled span: positions [a, b) of sequence `seq` at the output of capture slot `cap` add into sums row `dst`.
  struct Seg {
    int32_t cap, seq, a, b, dst;
  };
  // A sequence of a pass: positions [p0, p1) of `ids`, its DeltaNet state's slot, and its first page of the cache.
  struct Seq {
    const std::vector<int32_t>* ids;
    int64_t p0, p1;
    int32_t slot, page0;
  };
  float* dev(size_t floats, const char* what);
  void forward(const std::vector<Seq>& seqs, const std::vector<Seg>& segs, float* sums);

  sycl::queue& q_;
  std::vector<int> feat_, optl_, caps_;  // caps_: the layers whose outputs are pooled, ascending
  int64_t E_, F_, H_, Hkv_, D_, n_rot_, Hd_, n_run_, max_tok_, chunk_;
  int sections_[3] = {0, 0, 0};
  float eps_;
  uint64_t bytes_ = 0;
  const gguf::Tensor* embd_;
  std::vector<Layer> layers_;
  std::vector<void*> owned_;
  std::vector<xmx::Weight> repacked_;
  // per-chunk activations
  float *x_, *h_, *big_, *zb_, *ab_, *out_, *Q_, *K_, *V_, *G_, *att_, *inv_freq_, *sums_;
  int32_t *pos_, *slot_, *tiles_, *segs_, *runs_, *run_of_row_, *pool_runs_;
  xmx::Activations a_, a2_;
  gdn_seq::State state_, saved_;
  gdn_seq::Scratch gs_;
  std::vector<void*> kc_, vc_;
  int32_t* table_ = nullptr;
  int64_t pages_ = 0, max_tiles_ = 0, max_segs_ = 0;
  attn::Scratch attn_scratch_;
  std::map<std::string, double> profile_;
};

}  // namespace eklow::vega
