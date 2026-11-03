// The serving forward pass for Qwen3.6-35B-A3B (llama.cpp's qwen35moe) over a step of tokens from several sequences. The
// computation is the reference forward's (model/qwen35.hpp) on the serving kernels: the dense matrices in their file's formats
// on the matrix units (int8 activations up to 32 rows, fp16 GEMMs above), the Gated DeltaNet layers through gdn::step with a
// recurrent state per slot, the 10 attention layers on a paged int8 or fp16 KV cache, every layer's MoE with the hottest
// experts of each layer on the card and the rest computed by the CPU from pinned memory (or streamed to the card for prefill).
//
// A step is a list of chunks, at most one per sequence slot: a prefill chunk (many tokens), a decode token or a verify block
// (a speculative decoder's B tokens with logits for every row, leaving the DeltaNet's state for them pending until the
// slot's next chunk says how many were accepted). Each slot's positions must continue where its previous step left off.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu/qgemv.hpp"
#include "kernels/attention.hpp"
#include "kernels/fast.hpp"
#include "kernels/gdn.hpp"
#include "kernels/qwen_fast.hpp"
#include "kernels/xmx.hpp"
#include "model/dump.hpp"
#include "model/gguf.hpp"
#include "model/image_rows.hpp"
#include "model/qwen35.hpp"
#include "rt/copy_engine.hpp"

namespace eklow::engine {

struct Config {
  int n_slots = 2;
  int64_t max_ctx = 262144;  // positions per slot, a multiple of attn::kPage
  int64_t max_batch = 2048;  // tokens per step
  int64_t max_chunk = 2048;  // tokens per chunk
  // A lone prefill chunk (no other chunk in the step, not a verify chunk) of up to max_group tokens, a multiple of max_chunk, runs layer
  // by layer over sub-chunks of at most max_chunk, so that a layer's cold experts cross the link once per group, not once per chunk. 0: max_chunk.
  int64_t max_group = 0;
  // In a group, start a layer's copies once every sub-chunk is routed and run the hot experts after them (else each sub-chunk finishes its layer first).
  bool group_early_copy = true;
  int64_t max_logits = 64;   // logits rows per step
  // The attention layers' K/V as int8 with a scale per 16 dimensions (attention.hpp): half the memory.
  bool int8_kv = false;
  // The routed experts: the `experts_on_card` busiest of each layer by the profile (a JSON of per-layer selection counts,
  // tools/ref/expert_profile.py) are repacked onto the card; the rest are computed on the CPU from pinned host memory. -1: all
  // of them; below the model's count a profile is required.
  std::string expert_profile;
  int64_t experts_on_card = -1;  // -2: as many as fit in the free memory left after the rest, less the margin and the draft
  uint64_t card_margin_bytes = 1ull << 30;
  // Card memory kept free for the DFlash 2 draft (its weights and ring cache), subtracted from what the experts may take.
  uint64_t draft_reserve_bytes = uint64_t(1.2 * 1073741824.0);
  // Card memory kept free for the vision encoder (its weights and scratch for the largest image), subtracted like the draft's. An
  // encoder already resident when the engine is made is in the free memory the placement reads; this is for one loaded after it.
  uint64_t vision_reserve_bytes = 0;
  int cpu_threads = 0;  // the cold experts' pool, caller included; 0: 16
  // Prefill with cold experts: those with at least stream_min_rows (token, expert) pairs in a step are copied to the card by
  // the copy engine (at most stream_slots of them a layer), repacked there and computed by the grouped GEMMs, while the CPU takes
  // the other cold pairs. 0 slots: the CPU takes all of them. The costs pick how many candidates to stream.
  int stream_slots = 80;
  int stream_min_rows = 16;
  double stream_copy_us = 270;  // one expert (1.8 MB of Q4_K) from pinned memory to the card
  double stream_cpu_us = 10;    // one cold pair on the CPU, 16 threads, with its expert shared by many tokens (measured: 10 beat 7 and 14 at 2,048-token chunks)
  // Tokens a slot may leave pending after a verify chunk (at most gdn::kMaxPendingLimit).
  int max_pending = 16;
  // Hidden states the draft reads: the outputs (l_out) of these layers for the rows a chunk asks for (Chunk::capture_rows),
  // into a device buffer [row][layers x n_embd] fp32 in step order. Empty: no capture.
  std::vector<int> capture_layers = {1, 6, 11, 16, 22, 27, 32, 37};
  int64_t max_capture_rows = 2048;  // rows of capture a step may ask for
  // The head: the matrix-unit GEMV on int8 activations (xmx; any format the file has) or fp32 activations (fast::matmul_q6, Q6_K only).
  bool head_dpas = true;
  // The token embedding table on the card (0.5 GB less for experts) instead of rows gathered on the host per step.
  bool embed_on_card = false;
  // Count the (layer, expert) selections of every routed token (for a routing profile): slower, a host copy a layer.
  bool record_routing = false;
};

struct Chunk {
  int slot;
  int64_t start_pos;  // position of tokens[0]
  std::vector<int32_t> tokens;
  int64_t logits_rows = 1;  // logits for the last logits_rows tokens (a verify chunk: all of them)
  // A verify chunk (a speculative decoder's block of B <= max_pending tokens): its K/V is written and its DeltaNet inputs are
  // kept pending (commit = 0); the slot's next chunk says how many of the B tokens were accepted.
  bool verify = false;
  // For a slot whose previous chunk was a verify chunk: how many of that chunk's tokens were accepted (the DeltaNet replays them
  // before this chunk's tokens; the K/V past them is dropped, the slot's length moves back to the end of the accepted ones).
  // -1: all of them. start_pos must be the position after the accepted tokens.
  int accept = -1;
  // The last capture_rows rows of the chunk are captured (Config::capture_layers' hidden states).
  int64_t capture_rows = 0;
  // Images (Qwen3.6's vision rows). Spans of tokens whose input rows are given (host fp32, unscaled) in place of embedding lookups;
  // the tokens there are ignored. A span lies within the chunk, in order, never split across chunks, with its grid (nx, ny)
  // set; the DeltaNet layers see its rows like any others and attention over them is causal (llama.cpp's mtmd runs Qwen3-VL
  // images causally, its mask being the raster order when the rows share a time position).
  //
  // Positions. The K/V cache index of a token is its index in the slot's sequence (start_pos + j). The rotary position of the
  // 10 attention layers is another thing, as llama.cpp's M-RoPE has it: a text token's three components (time, row, column)
  // all equal its position, and an image takes max(nx, ny) positions for its n rows: row i of the span starting at position
  // p has (p, p + i / nx, p + i % nx), and the text after it continues from p + max(nx, ny). rope_delta is the sequence index
  // minus the rotary position at start_pos (0 until an image has passed; the spans of this chunk add to it as they go).
  std::vector<model::ImageRows> images = {};
  int64_t rope_delta = 0;
};

class Engine {
 public:
  Engine(sycl::queue& q, const gguf::File& f, const Config& cfg);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Logits for each chunk's last logits_rows tokens, chunk after chunk, [row][vocab]. `dump` receives intermediate tensors under
  // the reference forward's names (attn_norm-L, l_out-L, result_norm, ...).
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

  // Forgets a slot's sequence (K/V pages, DeltaNet state).
  void release(int slot);
  // Commits the pending tokens of a slot's last verify chunk (`accept` of them) without running a new token: the slot's length
  // moves back to the end of the accepted ones. snapshot() needs the slot settled.
  void settle(int slot, int accept);

  // Prefix reuse. A recurrent state cannot be truncated, so a sequence's state is saved where it may be needed again: the
  // snapshot of a slot (its DeltaNet and conv state, in pinned host memory) holds the position it was taken at (nothing may be
  // pending). restore() puts the state back into a slot whose K/V still holds the first `length` positions of the sequence the
  // snapshot belongs to (length = the snapshot's position) and cuts the K/V to it. truncate() is the K/V half alone: it is
  // allowed only to the slot's own length (a shorter K/V has no state to go with it); use restore().
  struct Snapshot {
    std::shared_ptr<uint8_t> host;  // pinned, snapshot_bytes
    uint64_t bytes = 0;
    int64_t position = -1;
  };
  Snapshot snapshot(int slot);
  void restore(int slot, const Snapshot& s, int64_t length);
  void truncate(int slot, int64_t length);
  uint64_t snapshot_bytes() const { return uint64_t(gcache_.snapshot_bytes()); }
  // Positions the slot holds (where its next chunk starts after any pending tokens are accepted in full).
  int64_t length(int slot) const { return slots_[size_t(slot)].length; }
  int pending(int slot) const { return slots_[size_t(slot)].pending; }

  // With a profile set, step waits after each section of the forward pass and adds its wall time under the section's name.
  void set_profile(std::map<std::string, double>* p) { profile_ = p; }

  // The output head as the matrix units run it (Config::head_dpas), for the DFlash 2 draft to project its block with the same
  // weights instead of a copy (0.4 GiB of the card).
  const xmx::Weight& head_weight() const {
    if (!c_.head_dpas) throw std::runtime_error("engine: the head is on the fp32-activation path");
    return head_w_;
  }
  const model::Qwen35Config& model() const { return cfg_; }
  int n_slots() const { return c_.n_slots; }
  const Config& config() const { return c_; }
  uint64_t weight_bytes() const { return weight_bytes_; }
  uint64_t cache_bytes() const { return cache_bytes_; }
  // What is where, for the startup log.
  std::string placement_line() const;
  int64_t experts_on_card() const { return n_hot_; }
  uint64_t free_at_placement() const { return free_at_placement_; }
  uint64_t expert_set_bytes() const { return expert_set_bytes_; }
  uint64_t pairs() const { return pairs_; }
  uint64_t cold_pairs() const { return cold_pairs_; }
  struct ColdStats {
    double wait_s = 0, cpu_s = 0;
    double gu_s = 0, glu_s = 0, dn_s = 0;  // cold_compute's phases
    uint64_t layers = 0, late_layers = 0, experts = 0;  // experts: distinct cold experts computed on the CPU
  };
  const ColdStats& cold_stats() const { return cold_; }
  struct StreamStats {
    uint64_t experts = 0, pairs = 0, layers = 0;
  };
  const StreamStats& stream_stats() const { return stream_; }
  int stream_slots() const { return n_stream_; }
  // Layer-major groups (max_group): per layer of each group.
  struct GroupStats {
    uint64_t groups = 0, layers = 0, experts = 0, waves = 0, streamed_pairs = 0, cpu_pairs = 0;
    double ids_wait_s = 0, copy_wait_s = 0, cpu_s = 0, join_s = 0;  // host waits: the group's routing; a wave's repack before the next copy; the CPU task (wall), and the wait for it after the waves
  };
  const GroupStats& group_stats() const { return grp_; }
  uint64_t pinned_bytes() const { return pinned_bytes_; }
  // Selections per [layer][expert] since the engine was made (Config::record_routing).
  const std::vector<std::vector<int64_t>>& routing_counts() const { return counts_; }
  double enqueue_seconds() const { return enqueue_s_; }

 private:
  // A dense matrix: one stacked weight, or (when the stacked tensors differ in type) the parts run apart and joined.
  struct Dense {
    std::vector<xmx::Weight> parts;
    int64_t n = 0;
  };
  struct Layer {
    bool recurrent = false;
    int gdn_idx = -1, attn_idx = -1;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    Dense in;   // DeltaNet: attn_qkv | attn_gate; attention: attn_q | attn_k | attn_v
    Dense out;  // ssm_out / attn_output
    xmx::Weight shexp_gu, shexp_dn;
    const float* w_rg = nullptr;  // fp32 [288][E]: the router's 256 rows, the shared expert's gate row, 31 zero rows
    const float* w_ab = nullptr;  // fp32 [64][E]: ssm_alpha | ssm_beta (dequantized), DeltaNet layers
    const float *q_norm = nullptr, *k_norm = nullptr;
    gdn::Weights gw{};
    // Routed experts.
    std::vector<xmx::Weight> exp_gu, exp_dn;
    xmx::ExpertTable tab_gu, tab_dn;
    std::vector<uint8_t> cold;
    uint8_t* cold_dev = nullptr;
    int n_cold = 0;
    // The GGUF's mapping of the three expert tensors, and where the CPU reads each cold expert (gate | up rows, then down).
    const uint8_t *gate_host = nullptr, *up_host = nullptr, *down_host = nullptr;
    std::vector<const uint8_t*> cold_gu, cold_dn;
  };
  struct Slot {
    int64_t length = 0;
    std::vector<int32_t> pages;
    int pending = 0;           // tokens of the last verify chunk not yet accepted
    int64_t verify_start = 0;  // position of the first of them
  };

  float* dev(int64_t n);
  void load_experts();
  void dense(const xmx::Activations& a, const Dense& d, float* out);
  void cold_compute(const Layer& L, const std::vector<int32_t>& cp, const int32_t* ids, const std::vector<cpu::Q8Blocks>& x, float* y);
  std::vector<int32_t> cold_list(const Layer& L, const int32_t* ids, int64_t P) const;
  void cold_decode(const Layer& L, int64_t T, int64_t P, sycl::event routed, sycl::event hot_done);
  void cold_prefill(const Layer& L, int64_t T, int64_t P, sycl::event ids_ready, sycl::event routed, sycl::event hot_done,
                    const std::function<void(const char*)>& mark);
  std::vector<int32_t> choose_stream(const Layer& L, const std::vector<int32_t>& cnt, bool group = false) const;
  void stream_experts(const Layer& L, const std::vector<int32_t>& list, int64_t P, int32_t hot_rows, int64_t s_tiles,
                      const std::function<void(const char*)>& mark);
  // Layer-major prefill of one lone chunk longer than max_chunk (step() hands it over after validating).
  struct GroupSub {
    int64_t off = 0, len = 0;
  };
  std::vector<float> step_group(const Chunk& c, int acc);
  void cold_group(const Layer& L, const std::vector<GroupSub>& subs, sycl::event ids_ready, sycl::event routed, const std::function<void(const char*)>& mark,
                  const std::function<void()>& after_first_copy = nullptr);
  void ensure_pages(int slot, int64_t positions);
  void free_pages_beyond(Slot& s, int64_t length);
  // A slot's accepted count for a chunk's `accept` (the pending tokens when negative); throws when it does not fit.
  int accepted(const Slot& s, int accept) const;
  void upload_plan(const std::vector<gdn::SlotStep>& steps, gdn::Plan& plan, size_t at = 0);

  sycl::queue& q_;
  Config c_;
  model::Qwen35Config cfg_;
  std::vector<void*> owned_;
  uint64_t weight_bytes_ = 0, cache_bytes_ = 0;

  // Head and embedding.
  xmx::Weight head_w_;
  fast::Q6 head_q6_;
  xmx::Activations act_head_;
  const float* output_norm_ = nullptr;
  const gguf::Tensor* embd_ = nullptr;  // the table, in the mapping
  uint64_t embd_row_bytes_ = 0;
  uint8_t *emb_host_ = nullptr, *emb_dev_ = nullptr;  // gathered rows (pinned, card)
  int32_t* iota_ = nullptr;
  ref::Weight embd_card_{};  // the whole table on the card, when asked
  std::vector<Layer> layers_;
  const float* inv_freq_ = nullptr;
  const float* ones_ = nullptr;  // 256 ones: moe_route's down scales

  // KV cache (attention layers) and the DeltaNet's state.
  struct KvLayer {
    void *k = nullptr, *v = nullptr;
    sycl::half *ks = nullptr, *vs = nullptr;
  };
  std::vector<KvLayer> kv_;
  int64_t n_pages_ = 0, max_pages_ = 0, group_ = 0;
  int32_t* page_table_ = nullptr;
  std::vector<int32_t> page_table_host_, free_pages_;
  std::vector<Slot> slots_;
  gdn::Cache gcache_{};
  gdn::Scratch gscratch_{};
  std::vector<gdn::SlotBook> books_;
  gdn::SlotStep* plan_dev_ = nullptr;
  gdn::SlotStep* plan_host_ = nullptr;

  // Workspace, sized for max_batch tokens.
  float *x_ = nullptr, *x2_ = nullptr, *h_ = nullptr, *o_ = nullptr, *qz_ = nullptr, *qb_ = nullptr, *gt_ = nullptr, *att_ = nullptr,
        *kb_ = nullptr, *vb_ = nullptr, *ab_ = nullptr, *rlog_ = nullptr, *sg_ = nullptr, *rg_ = nullptr, *moe_ = nullptr, *gx_ = nullptr,
        *moe_gu_ = nullptr, *sh_gu_ = nullptr, *rows_ = nullptr, *logits_ = nullptr, *tmp_ = nullptr;
  xmx::Activations act_, act_sh_, act_exp_ffn_;
  int32_t *tok_ = nullptr, *slot_ = nullptr, *pos_ = nullptr, *rpos_ = nullptr, *tiles_ = nullptr;  // pos_: K/V index; rpos_: [3][tokens] rotary
  int64_t max_tiles_ = 0;
  // Layer-major groups: tokens a step's arrays hold, sub-chunks a group has at most, the residual stream of all of them (x_), their MoE inputs in fp16, routing.
  int64_t cap_tokens_ = 0, gmax_ = 1;
  sycl::half* hg_ = nullptr;
  int32_t* gids_ = nullptr;
  float* gw_ = nullptr;
  GroupStats grp_;
  int32_t *moe_ids_ = nullptr, *moe_row_ = nullptr, *moe_tiles_ = nullptr, *moe_scratch_ = nullptr, *rows_idx_ = nullptr;
  float* moe_w_ = nullptr;
  int64_t moe_tile_ = 64, moe_max_tiles_ = 0;
  sycl::half *moe_a_ = nullptr, *moe_h_ = nullptr;
  float* moe_y_ = nullptr;
  int8_t* hq_ = nullptr;
  float* hqs_ = nullptr;
  attn::Scratch attn_scratch_;
  // Capture of hidden states.
  float* cap_ = nullptr;
  int32_t* cap_idx_ = nullptr;
  Capture capture_;
  std::vector<int> cap_col_;  // per layer: its column block in the capture, or -1

  // Experts.
  int64_t n_hot_ = 0;
  gguf::Type t_gu_ = gguf::Type::Q4_0, t_dn_ = gguf::Type::Q4_0;
  cpu::QType q_gu_ = cpu::QType::Q4_0, q_dn_ = cpu::QType::Q4_0;
  xmx::WeightFormat f_gu_ = xmx::WeightFormat::Q4_0, f_dn_ = xmx::WeightFormat::Q4_0;
  uint64_t gu_row_ = 0, dn_row_ = 0, raw_gu_ = 0, raw_dn_ = 0, raw_expert_ = 0;  // ggml bytes of a row, of an expert's gate|up, down, both
  uint64_t gu_tb_ = 0, gu_mb_ = 0, dn_tb_ = 0, dn_mb_ = 0, packed_expert_ = 0;     // the repacked pieces (aligned) and their sum
  int n_stream_ = 0;
  uint8_t *stream_raw_ = nullptr, *stream_slab_ = nullptr;
  std::vector<xmx::Weight> stream_w_gu_, stream_w_dn_;
  xmx::ExpertTable stream_gu_, stream_dn_;
  int32_t *stream_map_ = nullptr, *hp_smap_ = nullptr, *stream_tiles_ = nullptr, *stream_scratch_ = nullptr;
  int64_t stream_rows_cap_ = 0;
  std::unique_ptr<rt::CopyEngine> copy_;
  StreamStats stream_;
  uint64_t free_at_placement_ = 0, expert_set_bytes_ = 0, pinned_bytes_ = 0;
  std::unique_ptr<cpu::Pool> pool_;
  int32_t* hp_ids_ = nullptr;
  float *hp_w_ = nullptr, *hp_xs_ = nullptr, *hp_y_ = nullptr, *hp_sum_ = nullptr, *hp_as_ = nullptr, *cold_sum_ = nullptr;
  int8_t *hp_aq_ = nullptr, *hp_xq_ = nullptr;
  int32_t *hp_map_ = nullptr, *cold_map_ = nullptr;
  std::vector<float> cold_y_, cold_gu_;
  std::vector<cpu::Q8Blocks> cold_g_;
  uint64_t pairs_ = 0, cold_pairs_ = 0;
  ColdStats cold_;
  std::vector<std::vector<int64_t>> counts_;
  int32_t* rec_host_ = nullptr;
  std::map<std::string, double>* profile_ = nullptr;
  double enqueue_s_ = 0;
};

}  // namespace eklow::engine
