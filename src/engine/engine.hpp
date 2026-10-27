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
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpu/q4_gemv.hpp"
#include "kernels/attention.hpp"
#include "kernels/fast.hpp"
#include "kernels/xmx.hpp"
#include "model/gemma4.hpp"
#include "model/gguf.hpp"
#include "rt/copy_engine.hpp"

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
  // The 26B's experts: the `experts_on_card` busiest of each layer by the profile (a JSON of per-layer
  // selection counts, tools/ref/expert_profile.py) are repacked onto the card; the rest are computed on
  // the CPU from the GGUF's mapping (which must outlive the engine). -1: all of them; below the
  // model's count a profile is required.
  std::string expert_profile;
  int64_t experts_on_card = -1;  // -2: as many as fit in the free memory left after the rest, less card_margin_bytes
  uint64_t card_margin_bytes = 1ull << 30;
  int cpu_threads = 0;  // the cold experts' pool, caller included; 0: 16 (8, 12, 16, 20 tried: 16 and 20 equal)
  // Prefill with cold experts (DECISIONS 2026-10-28): the cold experts are held in pinned host memory (one copy, the
  // CPU computes from it too), and those with at least stream_min_rows pairs in a step are copied to the card by
  // the copy engine (at most stream_slots of them a layer), repacked there and computed by the grouped GEMMs, while
  // the CPU takes the other cold pairs. 0 slots: the CPU takes all of them, from the GGUF's mapping (stage 3a).
  // The costs pick how many candidates to stream: about the point where the copies take as long as the CPU would
  // on the rest.
  int stream_slots = 40;
  int stream_min_rows = 16;
  double stream_copy_us = 300;  // one expert from pinned memory to the card (copy-bench: 307 us)
  double stream_cpu_us = 22;    // one cold pair on the CPU, 16 threads
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
  // Experts of each layer on the card (the model's count when all are), and since the engine was
  // made, the (token, slot) pairs the MoE layers routed and how many of them went to the CPU.
  int64_t experts_on_card() const { return n_hot_; }
  // What the placement saw: the card's free memory before the experts went on (0 when it was not asked)
  // and the bytes of one expert from every layer.
  uint64_t free_at_placement() const { return free_at_placement_; }
  uint64_t expert_set_bytes() const { return expert_set_bytes_; }
  uint64_t pairs() const { return pairs_; }
  uint64_t cold_pairs() const { return cold_pairs_; }
  // Host seconds since the engine was made: waiting for the card's routing, computing cold experts on
  // the CPU; and of the layers that had cold experts, those whose hot experts were already finished on
  // the card when the CPU was (the card then waited for the host).
  struct ColdStats {
    double wait_s = 0, cpu_s = 0;
    uint64_t layers = 0, late_layers = 0;
  };
  const ColdStats& cold_stats() const { return cold_; }
  // Streaming (Config::stream_slots), since the engine was made: cold experts copied to the card for prefill and the
  // pairs they took off the CPU; layers in which anything was streamed.
  struct StreamStats {
    uint64_t experts = 0, pairs = 0, layers = 0;
  };
  const StreamStats& stream_stats() const { return stream_; }
  int stream_slots() const { return n_stream_; }
  uint64_t stream_slot_bytes() const { return 2 * uint64_t(n_stream_) * expert_bytes_; }
  uint64_t pinned_bytes() const { return pinned_bytes_; }
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
    // Cold experts: flags per expert (1: on the host), the same on the device for the grouping kernel,
    // and the GGUF's rows (gate_up [2 Fe][E], down [E][Fe], ggml Q4_0) of every expert.
    std::vector<uint8_t> cold;
    uint8_t* cold_dev = nullptr;
    int n_cold = 0;
    // Where the CPU reads each cold expert (gate_up then down, GGUF layout): the GGUF's mapping or, with streaming,
    // pinned host memory the copy engine reads too.
    std::vector<const uint8_t*> cold_gu, cold_dn;
    const uint8_t *gu_host = nullptr, *dn_host = nullptr;
  };
  struct Slot {
    int64_t length = 0;
    std::vector<int32_t> pages;
  };

  float* dev(int64_t n);
  void load_experts();
  // Layer L's cold experts for the pairs cp (ascending pair indices p = t * n_used + i, expert ids[p]):
  // gate_up, GeGLU and down on the CPU from the int8 input rows x[t]. Output row k (E floats) of y is
  // the k-th pair's, before its weight.
  void cold_compute(const Layer& L, const std::vector<int32_t>& cp, const int32_t* ids, const std::vector<cpu::Q8Blocks>& x, float* y);
  std::vector<int32_t> cold_list(const Layer& L, const int32_t* ids, int64_t P) const;
  void cold_decode(const Layer& L, int64_t T, int64_t P, sycl::event routed, sycl::event hot_done);
  void cold_prefill(const Layer& L, int64_t T, int64_t P, sycl::event ids_ready, sycl::event routed, sycl::event hot_done,
                    const std::function<void(const char*)>& mark);
  // The experts to stream for a layer of this step, busiest first, from the pairs per expert (cnt).
  std::vector<int32_t> choose_stream(const Layer& L, const std::vector<int32_t>& cnt) const;
  // Starts the copies of the listed experts into the slots and enqueues, behind the hot experts and the copies, the
  // grouping of their pairs and their GEMMs.
  void stream_experts(const Layer& L, const std::vector<int32_t>& list, int64_t P, int32_t hot_rows, int64_t s_tiles,
                      const std::function<void(const char*)>& mark);
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
  // MoE (26B) workspace: the expert sum, the router logits, the chosen expert ids and weights
  // (per pair), and the expert input rows (gx_: down outputs of up to kFp16Rows pairs at decode).
  float *moe_ = nullptr, *rlog_ = nullptr, *gx_ = nullptr;
  int32_t* moe_ids_ = nullptr;
  float* moe_w_ = nullptr;
  // Decode (T * n_used <= kFp16Rows pairs): the grouped gate_up output before GeGLU, and the int8
  // activation buffers.
  float* moe_gu_ = nullptr;
  xmx::Activations act_exp_, act_exp_ffn_;
  // Prefill: pairs sorted by expert in row tiles of moe_tile_ rows (fast::moe_group): the tile list,
  // its scratch, each pair's row; fp16 expert inputs (moe_a_, sharing moe_y_'s memory: they are dead
  // once the gate_up is done) and gate_up outputs, and the down outputs in fp32.
  // Row tile of the grouped GEMMs: prefill at 1k, 1,910 / 2,228 / 1,969 tok/s for 32 / 64 / 128 rows.
  int64_t moe_tile_ = 64, moe_max_tiles_ = 0;
  int32_t *moe_tiles_ = nullptr, *moe_scratch_ = nullptr, *moe_row_ = nullptr;
  sycl::half *moe_a_ = nullptr, *moe_h_ = nullptr;
  float* moe_y_ = nullptr;
  int32_t *tok_, *slot_, *pos_, *tiles_sliding_, *tiles_global_;
  int64_t max_tiles_sliding_ = 0, max_tiles_global_ = 0;
  xmx::Activations act_, act_ffn_;  // matmul inputs: of width E or H x D, and of the FFN's width
  attn::Scratch attn_scratch_;
  // Cold experts (see Config): the host pool, pinned buffers for the routing and inputs the host
  // reads (hp_ids_, hp_w_; prefill's int8 expert inputs hp_xq_ with scales hp_xs_; decode's int8 row tile
  // hp_aq_ hp_as_), for what it writes (hp_y_: decode's output rows; prefill's weighted sums of the tokens
  // with cold experts in hp_sum_ rows, token t's row in hp_map_[t] or -1, copied to the card's cold_sum_ and
  // cold_map_).
  int64_t n_hot_ = 0;
  // Streaming: S slots on the card (raw bytes as the copy engine lands them, then the repacked experts), their tables for the grouped
  // GEMMs, the experts-to-slots map for the grouping (device, and pinned for the host to fill), the second tile
  // list and grouping scratch; the pairs' inputs are gathered into att_ (dead during the MoE).
  int n_stream_ = 0;
  uint64_t expert_bytes_ = 0, expert_raw_gu_ = 0, pinned_bytes_ = 0;
  uint8_t *stream_raw_ = nullptr, *stream_slab_ = nullptr;
  std::vector<xmx::Weight> stream_w_gu_, stream_w_dn_;
  xmx::ExpertTable stream_gu_, stream_dn_;
  int32_t *stream_map_ = nullptr, *hp_smap_ = nullptr, *stream_tiles_ = nullptr, *stream_scratch_ = nullptr;
  int64_t stream_rows_cap_ = 0;
  std::unique_ptr<rt::CopyEngine> copy_;
  StreamStats stream_;
  uint64_t free_at_placement_ = 0, expert_set_bytes_ = 0;
  std::unique_ptr<cpu::Pool> pool_;
  int32_t* hp_ids_ = nullptr;
  float *hp_w_ = nullptr, *hp_xs_ = nullptr, *hp_y_ = nullptr, *hp_sum_ = nullptr, *hp_as_ = nullptr, *cold_sum_ = nullptr;
  int8_t *hp_aq_ = nullptr, *hp_xq_ = nullptr;
  int32_t *hp_map_ = nullptr, *cold_map_ = nullptr;
  std::vector<float> cold_y_, cold_gu_;
  std::vector<cpu::Q8Blocks> cold_g_;
  uint64_t pairs_ = 0, cold_pairs_ = 0;
  ColdStats cold_;
  std::map<std::string, double>* profile_ = nullptr;
  double enqueue_s_ = 0;
};

}  // namespace eklow::engine
