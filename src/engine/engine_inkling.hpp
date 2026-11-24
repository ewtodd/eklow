// The serving forward pass of Inkling 2 Mini (M2, DECISIONS 2026-11-24) over a step of tokens from several sequences: the
// reference forward's computation (model/inkling2.hpp) with the serving kernels. The dense matrices on the matrix units in
// their Q8_0 format (int8 activations up to 32 rows, fp16 above), the experts through kernels/moe.hpp, the attention
// through attn::attend with Inkling's bias and sink (kernels/attention.hpp), fp16 K/V.
//
// A step is a list of chunks, one per sequence in it (a prefill chunk or a decode token); each slot's positions continue
// where its previous step left off. Logits come back for the last `logits_rows` tokens of each chunk.
//
// Every layer application (48: 24 layers, layers 6-17 three times) is a cache slot with its own K/V and its own
// convolution state (the last three inputs of its four short convolutions), per sequence slot. The 8 full-attention
// applications keep pages from a pool; the 40 sliding ones a ring of window + max_chunk positions per sequence slot.
//
// Experts that do not fit stay in pinned host memory, whole layers of them (Config::staged_expert_layers), and are copied
// onto the card before their layer runs in each step: correct and simple, slow for decode (M3 places experts properly).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/moe.hpp"
#include "kernels/xmx.hpp"
#include "model/dump.hpp"
#include "model/gguf.hpp"
#include "model/inkling2.hpp"

namespace eklow::engine::inkling {

struct Config {
  int n_slots = 1;
  int64_t max_ctx = 65536;   // positions per slot, a multiple of attn::kPage
  int64_t max_batch = 1024;  // tokens per step
  int64_t max_chunk = 1024;  // tokens per chunk: the sliding rings hold the window and one chunk
  int64_t max_logits = 64;   // logits rows per step
  // Physical layers whose experts live in host memory and are copied onto the card each step they run.
  std::vector<int> staged_expert_layers = {};
};

struct Chunk {
  int slot;
  int64_t start_pos;  // position of tokens[0]
  std::vector<int32_t> tokens;
  int64_t logits_rows = 1;
};

class Engine {
 public:
  // The weights go to the card (staged layers' experts to pinned host memory); the token embedding's rows are read from
  // the GGUF's mapping, which must outlive the engine.
  Engine(sycl::queue& q, const gguf::File& f, const Config& cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Logits for each chunk's last logits_rows tokens, chunk after chunk, [row][n_vocab_out]. `dump` receives the reference
  // forward's tensor names (l_out-<slot> and so on) over the step's rows. `teacher`, when given, replaces each layer
  // application's output with its own (as the reference's --teacher: all of the step's rows, [T][n_embd]).
  std::vector<float> step(const std::vector<Chunk>& chunks, const model::DumpSink* dump = nullptr,
                          const model::Teacher* teacher = nullptr);

  // Forgets a slot's sequence (its pages and its convolution state).
  void release(int slot);
  int64_t length(int slot) const { return slots_[size_t(slot)].length; }

  void set_profile(std::map<std::string, double>* p) { profile_ = p; }
  const model::Inkling2Config& model() const { return cfg_; }
  const Config& config() const { return c_; }
  uint64_t weight_bytes() const { return weight_bytes_; }
  uint64_t host_expert_bytes() const { return host_expert_bytes_; }
  uint64_t cache_bytes() const { return cache_bytes_; }

 private:
  struct Layer {
    xmx::Weight qrg, k, v, o;  // Q|R|gate stacked (they read the same input), K, V, the output projection
    const float* sink = nullptr;       // [n_head][n_embd] fp32 (from BF16)
    const float* k_dyn = nullptr;      // [4 n_head_kv][n_embd] fp32, full layers
    const float* v_dyn = nullptr;
    const float* router = nullptr;     // [n_expert][n_embd] fp32
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;
    const float *q_clip = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    const sycl::half* rel_tiles = nullptr;  // rel_bias as attn::bias_table prepares it
    const float *k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    const float *attn_res_query = nullptr, *ffn_res_query = nullptr, *ffn_norm = nullptr, *probs_bias = nullptr;
    moe::Layer experts;
    uint8_t* host_experts = nullptr;  // staged layers: the experts' bytes in pinned host memory
  };
  struct Loop {
    const float* msg_query = nullptr;  // [4][n_embd]
    const float* down[4] = {};         // [gate_rank][2 n_embd] fp32
    const float* gain32[4] = {};       // [n_embd][gate_rank] fp32, for steps of few tokens
    const float* forget32[4] = {};
    xmx::Weight gain[4], forget[4];    // the same as fp16 tiles, for the fp16 GEMM of prefill steps
  };
  struct Slot {
    int64_t length = 0;
    std::vector<int32_t> pages;
  };

  float* dev(int64_t n);
  void ensure_pages(int slot, int64_t positions);

  sycl::queue& q_;
  Config c_;
  model::Inkling2Config cfg_;
  std::vector<void*> owned_;
  uint64_t weight_bytes_ = 0, host_expert_bytes_ = 0, cache_bytes_ = 0;

  // The head on the matrix units; the embedding rows gathered from the GGUF's mapping per step.
  xmx::Weight head_w_;
  const gguf::Tensor* embd_ = nullptr;
  uint64_t embd_row_bytes_ = 0;
  uint8_t *emb_host_ = nullptr, *emb_dev_ = nullptr;
  int32_t* iota_ = nullptr;
  const float *tok_norm_ = nullptr, *output_norm_ = nullptr, *output_res_query_ = nullptr;
  std::vector<Layer> layers_;  // physical
  std::vector<Loop> loops_;
  uint8_t* staging_ = nullptr;  // a staged layer's experts on the card while it runs (their tables point here)
  int staged_layer_ = -1;       // whose experts staging_ holds now

  // K/V: per slot application, a ring (sliding) or pages (full).
  std::vector<int> ring_of_, page_of_;  // per application slot: its ring or page-cache index, -1 otherwise
  sycl::half *ring_k_ = nullptr, *ring_v_ = nullptr;
  int64_t ring_layer_stride_ = 0, ring_pages_ = 0, n_pages_ = 0, max_pages_ = 0;
  std::vector<std::pair<sycl::half*, sycl::half*>> pages_kv_;
  int32_t* page_table_ = nullptr;
  std::vector<int32_t> page_table_host_, free_pages_;
  std::vector<Slot> slots_;
  // Convolution state: [seq slot][application][3][conv_channels] fp32.
  float* conv_ = nullptr;

  // Workspace, sized for max_batch tokens.
  std::vector<float*> blk_;  // block and stream buffers [B][E]
  float *mixq_, *h_[3], *qrg_, *kr_, *vr_, *kd_, *vd_, *qb_, *kb_, *vb_, *rb_, *sk_, *att_, *o_, *oc_, *f_, *ci_, *cd_, *cg_, *cf_;
  float *lg_, *wts_, *rows_, *logits_, *msg_;
  int32_t *ids_, *tok_, *slot_, *pos_, *base_, *lrows_, *tiles_sliding_, *tiles_full_;
  int64_t max_tiles_sliding_ = 0, max_tiles_full_ = 0;
  xmx::Activations act_, act_head_;
  attn::Scratch attn_scratch_;
  moe::Scratch* moe_scratch_ = nullptr;
  std::map<std::string, double>* profile_ = nullptr;
};

}  // namespace eklow::engine::inkling
