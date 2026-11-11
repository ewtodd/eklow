// The serving forward pass: Gemma 4 E4B (Nemotron-3.5-Bolt-E4B) over a step of tokens from several sequences, with a KV
// cache per sequence slot. The computation is the reference forward's (model/gemma4.hpp) with the serving kernels: the
// file's K-quant matrices on the matrix units (int8 activations up to 32 rows, fp16 above), fp16 or int8 K/V.
//
// A step is a list of chunks, one per sequence in it: a prefill chunk (many tokens) or a decode token. Each slot's
// positions must continue where its previous step left off. Logits come back for the last `logits_rows` tokens of each
// chunk.
//
// The K/V cache holds only the layers that own K/V (the first n_layer - n_kv_shared): the shared layers attend over their
// source layer's cache with their own queries. Sliding layers keep a ring of window + max_chunk positions per slot,
// global layers pages from a pool.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/xmx.hpp"
#include "model/gemma4.hpp"
#include "model/gguf.hpp"
#include "model/image_rows.hpp"

namespace eklow::engine::gemma4 {

struct Config {
  int n_slots = 2;
  int64_t max_ctx = 65536;   // positions per slot, a multiple of attn::kPage
  int64_t max_batch = 2048;  // tokens per step
  int64_t max_chunk = 2048;  // tokens per chunk: the sliding rings hold the window and one chunk
  int64_t max_logits = 64;   // logits rows per step
  // The global layers' K/V as int8 with a scale per 16 dimensions (attention.hpp): half the memory, at a cost to decode.
  bool int8_global_kv = false;
  // Speculative decoding (a DFlash draft). Hidden states the draft reads: the outputs (l_out) of these layers for the rows
  // a chunk asks for (Chunk::capture_rows), into a device buffer [row][layers x n_embd] fp32 in step order. Empty: no capture.
  std::vector<int> capture_layers = {};
  int64_t max_capture_rows = 2048;  // rows of capture a step may ask for
};

// A verify chunk's longest block.
constexpr int64_t kMaxVerify = 16;

struct Chunk {
  int slot;
  int64_t start_pos;  // position of tokens[0]
  std::vector<int32_t> tokens;
  int64_t logits_rows = 1;
  // Spans of tokens whose input rows are given (an image's, unscaled) instead of looked up in the embedding table; their
  // per-layer inputs take the pad token's table rows. Attention stays causal over them. A span lies within one chunk, in
  // order.
  std::vector<model::ImageRows> images = {};
  // A verify chunk (a speculative decoder's block of 2 to kMaxVerify tokens: the anchor and the drafted ones): logits for
  // every row (logits_rows is ignored). A rejection is a rewind: the caller calls truncate(slot, start_pos + 1 + accepted)
  // after the step (within ring_slack() positions of the block's end).
  bool verify = false;
  // The last capture_rows rows of the chunk are captured (Config::capture_layers' hidden states).
  int64_t capture_rows = 0;
};

class Engine {
 public:
  // The weights go to the card; the token embedding's rows are read from the GGUF's mapping, which must outlive the engine.
  Engine(sycl::queue& q, const gguf::File& f, const Config& cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Logits for each chunk's last logits_rows tokens, chunk after chunk, [row][vocab]. `dump` receives the same
  // intermediate tensors as the reference forward.
  std::vector<float> step(const std::vector<Chunk>& chunks, const model::DumpSink* dump = nullptr);
  // The hidden states of the last step's captured rows: a device buffer [row][capture_layers.size() x n_embd] fp32, in step
  // order (chunk after chunk), valid until the next step; ids[row] is the row's slot and position.
  struct RowId {
    int32_t slot, pos;
  };
  struct Capture {
    const float* data = nullptr;
    int64_t rows = 0, width = 0;
    std::vector<RowId> ids;
  };
  const Capture& capture() const { return capture_; }
  // The output head on the matrix units, for a draft to project its block with the same weights instead of a second copy,
  // and the token embedding table as the GGUF stores it (host memory; the model scales the looked-up rows by sqrt(n_embd)).
  const xmx::Weight& head_weight() const { return head_w_; }
  ref::Weight embedding_table() const;
  const Config& config() const { return c_; }

  // Forgets a slot's sequence and returns its pages.
  void release(int slot);
  // Keeps the first `length` positions of a slot's sequence (a prompt that shares only a prefix with the last one) and
  // returns the pages past them. The sliding rings still hold what was overwritten there: rewinding a slot by more than
  // ring_slack() positions leaves the last window before `length` stale, and the caller must keep window positions fewer
  // and prefill them again.
  void truncate(int slot, int64_t length);
  int64_t ring_slack() const { return ring_pages_ * attn::kPage - cfg_.window; }
  // Positions the slot holds (where its next chunk starts).
  int64_t length(int slot) const { return slots_[size_t(slot)].length; }
  // The global layers' page pool: pages in all (max_ctx's worth per slot) and free now (a page is attn::kPage positions). The
  // sliding layers' rings are each slot's for good and are not in it.
  int64_t pool_pages() const { return n_pages_; }
  int64_t free_pages() const { return int64_t(free_pages_.size()); }

  // With a profile set, step waits after each section of the forward pass and adds its wall time under the section's
  // name: slower, for finding where the time goes.
  void set_profile(std::map<std::string, double>* p) { profile_ = p; }

  const model::Gemma4Config& model() const { return cfg_; }
  int n_slots() const { return c_.n_slots; }
  uint64_t weight_bytes() const { return weight_bytes_; }
  uint64_t cache_bytes() const { return cache_bytes_; }
  // Host time the last step took to enqueue its work; when it nears the step's wall time, the card waits on the host.
  double enqueue_seconds() const { return enqueue_s_; }

 private:
  // A projection that may come in parts of different formats (Bolt's Q4_K_M stores some V and down matrices in Q6_K):
  // each part is a run of stacked tensors of one type, its output columns side by side in its own buffer.
  struct Layer {
    std::vector<xmx::Weight> qkv;  // Q, K and V, stacked while their types agree; Q alone on a shared layer
    xmx::Weight wo, gate_up, down, inp_gate, proj;  // gate_up interleaved
    bool has_v = false;
    const float *attn_norm, *q_norm, *k_norm = nullptr, *attn_post_norm, *ffn_norm, *ffn_post_norm, *post_norm;
    float out_scale;
    int ring_index = -1, page_index = -1;  // the cache this layer reads: its own, or its source layer's
    bool writes_kv = false;
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

  // The head on the matrix units; the embedding rows gathered from the GGUF's mapping per step.
  xmx::Weight head_w_;
  xmx::Activations act_head_;
  const gguf::Tensor* embd_ = nullptr;
  uint64_t embd_row_bytes_ = 0;
  uint8_t *emb_host_ = nullptr, *emb_dev_ = nullptr;
  int32_t* iota_ = nullptr;
  // The per-layer inputs: the projection of the embeddings (fp16 GEMM), the token table (Q8_0) on the card.
  xmx::Weight pl_proj_;
  ref::Weight pl_table_{};
  const float* pl_norm_ = nullptr;
  int32_t pad_id_ = 0;
  // Capture of hidden states.
  float* cap_ = nullptr;
  int32_t* cap_idx_ = nullptr;
  Capture capture_;
  std::vector<int> cap_col_;  // per layer: its column block in the capture, or -1
  const float* output_norm_ = nullptr;
  const float *inv_freq_sliding_ = nullptr, *inv_freq_global_ = nullptr;
  std::vector<Layer> layers_;

  // KV cache
  sycl::half *ring_k_ = nullptr, *ring_v_ = nullptr;
  struct GlobalCache {
    void *k = nullptr, *v = nullptr;
    sycl::half *ks = nullptr, *vs = nullptr;
  };
  std::vector<GlobalCache> global_;
  int64_t ring_layer_stride_ = 0, ring_pages_ = 0, n_pages_ = 0, max_pages_ = 0;
  int64_t group_sliding_ = 0, group_global_ = 0, hkv_sliding_ = 0, hkv_global_ = 0;
  int32_t* page_table_ = nullptr;
  std::vector<int32_t> page_table_host_, free_pages_;
  std::vector<Slot> slots_;

  // workspace, sized for max_batch tokens
  float *xa_, *xb_, *h_, *qkv_, *qkv2_, *qb_, *kb_, *vb_, *att_, *o_, *gu_, *pl_, *pg_, *rows_, *logits_;
  int32_t *tok_, *slot_, *pos_, *tiles_sliding_, *tiles_global_;
  int64_t max_tiles_sliding_ = 0, max_tiles_global_ = 0;
  xmx::Activations act_, act_ffn_, act_pl_;  // matmul inputs: of width E or H x D, of the FFN's width, of the per-layer width
  attn::Scratch attn_scratch_;
  std::map<std::string, double>* profile_ = nullptr;
  double enqueue_s_ = 0;
};

}  // namespace eklow::engine::gemma4
