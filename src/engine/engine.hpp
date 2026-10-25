// The serving forward pass: Gemma 4 12B over a step of tokens from several sequences, with a KV
// cache per sequence slot. The computation is the reference forward's (model/gemma4.hpp) with the
// serving kernels: Q4_0 matmuls on the matrix units with int8 activations, fp16 KV.
//
// A step is a list of chunks, one per sequence in it: a prefill chunk (many tokens) or a decode
// token. Each slot's positions must continue where its previous step left off. Logits come back for
// the last `logits_rows` tokens of each chunk.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/xmx.hpp"
#include "model/gemma4.hpp"
#include "model/gguf.hpp"

namespace eklow::engine {

struct Config {
  int n_slots = 2;
  int64_t max_ctx = 131072;  // positions per slot
  int64_t max_batch = 2048;  // tokens per step
  int64_t max_logits = 64;   // logits rows per step
  int64_t page_size = 256;
};

struct Chunk {
  int slot;
  int64_t start_pos;  // position of tokens[0]
  std::vector<int32_t> tokens;
  int64_t logits_rows = 1;
};

class Engine {
 public:
  Engine(sycl::queue& q, const gguf::File& f, const Config& cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Logits for each chunk's last logits_rows tokens, chunk after chunk, [row][vocab]. `dump`
  // receives the same intermediate tensors as the reference forward.
  std::vector<float> step(const std::vector<Chunk>& chunks, const model::DumpSink* dump = nullptr);
  // Forgets a slot's sequence and returns its pages.
  void release(int slot);
  // Positions the slot holds (where its next chunk starts).
  int64_t length(int slot) const { return slots_[size_t(slot)].length; }

  const model::Gemma4Config& model() const { return cfg_; }
  uint64_t weight_bytes() const { return weight_bytes_; }
  uint64_t cache_bytes() const { return cache_bytes_; }

 private:
  struct Layer {
    xmx::Weight wq, wk, wv, wo, gate, up, down;  // wv.tiles is null on global layers
    const float *attn_norm, *q_norm, *k_norm, *attn_post_norm, *ffn_norm, *ffn_post_norm;
    float out_scale;
    int ring_index, page_index;  // index among the sliding / global layers
  };
  struct Slot {
    int64_t length = 0;
    std::vector<int32_t> pages;
  };

  float* dev(int64_t n);
  void ensure_pages(int slot, int64_t positions);

  sycl::queue& q_;
  Config c_;
  model::Gemma4Config cfg_;
  std::vector<void*> owned_;
  uint64_t weight_bytes_ = 0, cache_bytes_ = 0;

  ref::Weight tok_embd_;
  const float* output_norm_ = nullptr;
  const float *inv_freq_sliding_ = nullptr, *inv_freq_global_ = nullptr;
  std::vector<Layer> layers_;

  // KV cache
  sycl::half *ring_k_ = nullptr, *ring_v_ = nullptr, *page_k_ = nullptr, *page_v_ = nullptr;
  int64_t ring_layer_stride_ = 0, page_layer_stride_ = 0, n_pages_ = 0, max_pages_ = 0;
  int32_t* page_table_ = nullptr;
  std::vector<int32_t> page_table_host_, free_pages_;
  std::vector<Slot> slots_;

  // workspace, sized for max_batch tokens
  float *x_, *x2_, *h_, *qb_, *kb_, *vb_, *att_, *o_, *gate_, *up_, *rows_, *logits_;
  int32_t *tok_, *slot_, *pos_, *chunk_pos_, *chunk_base_, *chunk_last_;
  xmx::Activations act_;
  attn::Scratch attn_scratch_;
};

}  // namespace eklow::engine
