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
#include <map>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/fast.hpp"
#include "kernels/xmx.hpp"
#include "model/gemma4.hpp"
#include "model/gguf.hpp"

namespace eklow::engine {

struct Config {
  int n_slots = 2;
  int64_t max_ctx = 131072;  // positions per slot, a multiple of attn::kPage
  int64_t max_batch = 2048;  // tokens per step
  int64_t max_chunk = 1024;  // tokens per chunk: the sliding rings hold the window and one chunk
  int64_t max_logits = 64;   // logits rows per step
  // The global layers' K/V as int8 with a scale per 16 dimensions (attention.hpp): half the
  // memory, which 2 x 262k needs, at a cost to decode (the conversion adds to what each thread
  // waits on) and within the oracle's noise.
  bool int8_global_kv = false;
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
  // Keeps the first `length` positions of a slot's sequence (a prompt that shares only a prefix
  // with the last one) and returns the pages past them. The sliding rings still hold what was
  // overwritten there: rewinding a slot by more than ring_slack() positions leaves the last window
  // before `length` stale, and the caller must keep window positions fewer and prefill them again.
  void truncate(int slot, int64_t length);
  int64_t ring_slack() const { return ring_pages_ * attn::kPage - cfg_.window; }
  // Positions the slot holds (where its next chunk starts).
  int64_t length(int slot) const { return slots_[size_t(slot)].length; }

  // With a profile set, step waits after each section of the forward pass and adds its wall time
  // under the section's name: slower, for finding where the time goes.
  void set_profile(std::map<std::string, double>* p) { profile_ = p; }

  const model::Gemma4Config& model() const { return cfg_; }
  int n_slots() const { return c_.n_slots; }
  uint64_t weight_bytes() const { return weight_bytes_; }
  uint64_t cache_bytes() const { return cache_bytes_; }
  // Host time the last step took to enqueue its work; when it nears the step's wall time, the
  // card waits on the host.
  double enqueue_seconds() const { return enqueue_s_; }

 private:
  struct Layer {
    xmx::Weight wqkv, wo, gate_up, down;  // wqkv: Q, K and (has_v) V stacked; gate_up interleaved
    bool has_v;
    const float *attn_norm, *q_norm, *k_norm, *attn_post_norm, *ffn_norm, *ffn_post_norm;
    float out_scale;
    int ring_index, page_index;  // index among the sliding / global layers
    // The 26B's routed experts: per-expert gate_up (interleaved) and down, the F32 router, its
    // input scale (ffn_gate_inp.scale / sqrt(E)), the expert-path norms, and each expert's down scale.
    std::vector<xmx::Weight> exp_gate_up, exp_down;
    ref::Weight w_router;
    const float *router_in_scale = nullptr, *post_ffw_norm_1 = nullptr, *pre_ffw_norm_2 = nullptr,
        *post_ffw_norm_2 = nullptr;
    std::vector<float> exp_down_scale;
    xmx::ExpertTable tab_gate_up, tab_down;  // device pointer tables over exp_gate_up / exp_down (decode path)
    const float* exp_down_scale_dev = nullptr;
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

  fast::Q6 head_;  // token embedding and tied output head
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
  float *x_, *x2_, *h_, *qkv_, *qb_, *kb_, *vb_, *att_, *o_, *gu_, *rows_, *logits_;
  // MoE (26B) workspace: the expert sum, the router logits, and a buffer that holds one expert's
  // gathered input rows and, after its down matmul, its output rows; plus the per-expert token
  // indices and weights and the expert activation buffers.
  float *moe_ = nullptr, *rlog_ = nullptr, *gx_ = nullptr, *dws_ = nullptr;
  int32_t* dids_ = nullptr;
  // Decode-path MoE (T * n_used <= kFp16Rows pairs): chosen expert ids and weights, the grouped
  // gate_up output before GeGLU.
  int32_t* moe_ids_ = nullptr;
  float *moe_w_ = nullptr, *moe_gu_ = nullptr;
  xmx::Activations act_exp_, act_exp_ffn_;
  int32_t *tok_, *slot_, *pos_, *tiles_sliding_, *tiles_global_;
  int64_t max_tiles_sliding_ = 0, max_tiles_global_ = 0;
  xmx::Activations act_, act_ffn_;  // matmul inputs: of width E or H x D, and of the FFN's width
  attn::Scratch attn_scratch_;
  std::map<std::string, double>* profile_ = nullptr;
  double enqueue_s_ = 0;
};

}  // namespace eklow::engine
