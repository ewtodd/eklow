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
// Experts (M3, DECISIONS 2026-11-24): with a routing profile, the busiest (layer, expert) pairs go on the card as long as its
// free memory less a margin and a reserve holds them, the rest stay in the GGUF's mapping and run on the CPU
// (cpu::expert_ffn): a step's cold pairs are computed by the host while the card runs the hot ones, and added.
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/moe.hpp"
#include "kernels/xmx.hpp"
#include "rt/copy_engine.hpp"
#include "cpu/expert.hpp"
#include <memory>
#include "model/dump.hpp"
#include "model/image_rows.hpp"
#include "model/gguf.hpp"
#include "model/inkling2.hpp"

namespace eklow::engine::inkling {

struct Config {
  int n_slots = 1;
  int64_t max_ctx = 65536;   // positions per slot, a multiple of attn::kPage
  int64_t max_batch = 1024;  // tokens per step
  int64_t max_chunk = 1024;  // tokens per chunk: the sliding rings hold the window and one chunk
  int64_t max_logits = 64;   // logits rows per step
  // The routing profile (JSON, "decode": counts [layer application][expert]); empty: every expert on the card.
  std::string expert_profile;
  uint64_t card_margin_bytes = uint64_t(1) << 30;  // free card memory left after the experts
  uint64_t card_reserve_bytes = 0;                 // and what is loaded after the engine (side models)
  int cpu_threads = 8;                             // the cold experts' pool, the calling thread included
  int64_t cpu_max_tokens = 128;                    // steps up to this many tokens run cold experts on the CPU; longer
                                                   // ones copy the layer's cold set onto the card and run it there
  int checkpoints = 2;                             // per slot: saved states a sequence can be rewound to (rewind)
  // Tokens a step may carry (0: max_batch). Past max_batch a step runs layer by layer over sub-batches of max_batch (each
  // layer application over every sub-batch before the next), so that a long prompt's cold sets are copied onto the card
  // once a step rather than once a chunk; the sequences' blocks are held for all of them (8 x max_step x n_embd floats).
  int64_t max_step = 0;
  // Positions of the full layers' K/V, one pool the slots share (0: n_slots x max_ctx, a slot's room each). At least
  // max_ctx, a multiple of attn::kPage. A smaller pool frees card memory for experts; when the slots' sequences need more
  // pages than are free, the scheduler releases or holds back (pages_needed).
  int64_t pool_positions = 0;
  // LoopCD's hidden-state form (arXiv 2610.02185; DECISIONS 2026-11-24): what the coda reads of the core, its two blocks'
  // outputs, becomes h + w (h - h_1), h_1 the same after the first pass. 0: the model as trained.
  float loopcd = 0.0f;
};

struct Chunk {
  int slot;
  int64_t start_pos;  // position of tokens[0]
  std::vector<int32_t> tokens;
  int64_t logits_rows = 1;
  // Images' rows (Inkling M5): each stands for n of the chunk's tokens from its offset, whose ids are not looked up (any
  // id; the server's are negative). The rows go in after the token norm, as llama.cpp's (an embeddings batch is not
  // normalized), and must outlive the step.
  std::vector<model::ImageRows> images = {};
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

  // Forgets a slot's sequence (its pages, its convolution state and its checkpoints).
  void release(int slot);
  // A slot's state beyond its K/V pages (the sliding rings and the convolution state) cannot be rewound by dropping pages:
  // checkpoint saves it to pinned host memory at the slot's current length (replacing its oldest checkpoint), and rewind
  // goes back to a checkpoint's position exactly (its rings and convolution state restored, later pages returned). The
  // server checkpoints where a prefill ends, so that a next turn that differs from the last answer resumes there.
  void checkpoint(int slot);
  bool rewind(int slot, int64_t pos);  // false when the slot has no checkpoint at pos (or pos is past its length)
  std::vector<int64_t> checkpoint_positions(int slot) const;
  int64_t length(int slot) const { return slots_[size_t(slot)].length; }
  // The full applications' page pool: pages in all and free now (a page is attn::kPage positions).
  int64_t pool_pages() const { return n_pages_; }
  int64_t free_pages() const { return int64_t(free_pages_.size()); }
  int64_t pages_held(int slot) const { return int64_t(slots_[size_t(slot)].pages.size()); }
  // Pages a slot's sequence would take beyond those it holds, to reach `positions`.
  int64_t pages_needed(int slot, int64_t positions) const {
    return std::max<int64_t>(0, (positions + attn::kPage - 1) / attn::kPage - int64_t(slots_[size_t(slot)].pages.size()));
  }

  void set_profile(std::map<std::string, double>* p) { profile_ = p; }
  const model::Inkling2Config& model() const { return cfg_; }
  const Config& config() const { return c_; }
  uint64_t weight_bytes() const { return weight_bytes_; }
  // The step's block buffers ([max_step][n_embd] fp32 each): nothing in them outlives a step, so between steps another
  // user may borrow them (Inkling's vision encoder, with the scheduler held), after drain() and done before the next
  // step.
  std::vector<std::pair<void*, size_t>> idle_workspace() const {
    std::vector<std::pair<void*, size_t>> out;
    for (int i = 0; i < 8; ++i) out.push_back({blk_[size_t(i)], size_t(max_step_ * cfg_.n_embd) * sizeof(float)});
    return out;
  }
  void drain() { q_.wait(); }
  uint64_t host_expert_bytes() const { return host_expert_bytes_; }
  int64_t cold_experts() const { return n_cold_; }
  uint64_t free_at_placement() const { return free_at_placement_; }
  uint64_t cache_bytes() const { return cache_bytes_; }

 private:
  struct Layer {
    xmx::Weight qrg, k, v, o;  // Q|R|gate stacked (they read the same input), K, V, the output projection
    const float* sink = nullptr;       // [n_head][n_embd] fp32 (from BF16)
    float* sink32 = nullptr;           // the same padded to 32 rows of zeros
    xmx::Weight sink16;                // the sinks padded to 128 rows on the fp16 GEMM (its N tile), for fp16 activations
    const float* k_dyn = nullptr;      // [4 n_head_kv][n_embd] fp32, full layers
    const float* v_dyn = nullptr;
    const float* router = nullptr;     // [n_expert][n_embd] fp32
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *v_norm = nullptr;
    const float *q_clip = nullptr, *rel_bias = nullptr, *rel_clip = nullptr;
    const sycl::half* rel_tiles = nullptr;  // rel_bias as attn::bias_table prepares it
    const float *k_conv = nullptr, *v_conv = nullptr, *attn_post_conv = nullptr, *ffn_post_conv = nullptr;
    const float *attn_res_query = nullptr, *ffn_res_query = nullptr, *ffn_norm = nullptr, *probs_bias = nullptr;
    moe::Layer experts;                    // the hot experts on the card
    std::vector<bool> cold;                // per expert
    std::vector<int32_t> cold_slot;        // per expert: its index in cold_ref, or -1
    std::vector<cpu::ExpertRef> cold_ref;  // the cold experts in the GGUF's mapping
    moe::Layer cold_set;                   // the same experts repacked, tables pointing into staging_
    uint8_t* cold_host = nullptr;          // their bytes, pinned
  };
  struct Loop {
    const float* msg_query = nullptr;  // [4][n_embd]
    const float* down[4] = {};         // [gate_rank][2 n_embd] fp32
    const float* gain32[4] = {};       // [n_embd][gate_rank] fp32, for steps of few tokens
    const float* forget32[4] = {};
    xmx::Weight gain[4], forget[4];    // the same as fp16 tiles, for the fp16 GEMM of prefill steps
    xmx::Weight down16[4];             // down ([gate_rank][2 n_embd]) padded to 128 rows, on the fp16 GEMM
  };
  struct Checkpoint {
    int64_t pos = -1;
    uint64_t age = 0;
    uint8_t* data = nullptr;  // pinned: every sliding application's ring K, then V, then the convolution state
  };
  struct Slot {
    int64_t length = 0;
    std::vector<int32_t> pages;
    std::vector<Checkpoint> ckpt;
  };

  float* dev(int64_t n);
  void ensure_pages(int slot, int64_t positions);
  void place_experts(const gguf::File& f);
  static cpu::QType cpu_type(gguf::Type t);
  // A layer application's cold pairs: on the host once `packed` (the routing and inputs in hids_, hw_, hx_) is done, then
  // their weighted rows added to f_.
  void cold(const Layer& L, int64_t T, sycl::event packed);

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
  int64_t n_cold_ = 0;
  // Two staging regions, each the largest cold set: while a long step runs one layer application's cold set from one, the
  // copy queue fills the other with the next application's (DECISIONS 2026-11-24, M6). staged_: whose set each holds;
  // copied_: its copy; used_: the step's last use of it, which the next copy into it waits for.
  uint8_t* staging_[2] = {nullptr, nullptr};
  int staged_[2] = {-1, -1};
  sycl::event copied_[2], used_[2];
  std::unique_ptr<rt::CopyEngine> copy_;  // the card's copy engine: a queue's memcpy would share the compute engine
  std::vector<int> exec_layers_;  // the physical layer of each layer application, in the order a step runs them
  int stage(int layer, int avoid);  // the region holding `layer`'s cold set, its copy issued if needed (never into `avoid`)
  int last_staged_use_ = 0;         // the region the latest cold set ran from
  int64_t max_step_ = 0;            // tokens a step may carry (Config::max_step, at least max_batch)
  int64_t tile_stride_s_ = 0, tile_stride_f_ = 0;  // a sub-batch's share of the tile buffers
  uint64_t free_at_placement_ = 0;
  // The cold path: the step's routing and expert inputs on the host, the cold rows, their device copy, the pool.
  int32_t* hids_ = nullptr;
  float *hw_ = nullptr, *hx_ = nullptr, *hrows_ = nullptr, *cr_ = nullptr;
  std::vector<float> cold_y_;
  std::unique_ptr<cpu::Pool> pool_;

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
  uint64_t ckpt_bytes_ = 0, ckpt_age_ = 0;
  // The copies of a checkpoint, in order: (device pointer, bytes) for a slot.
  std::vector<std::pair<uint8_t*, uint64_t>> state_parts(int slot) const;

  // Workspace, sized for max_batch tokens.
  std::vector<float*> blk_;  // block and stream buffers [B][E]
  float *mixq_, *h_[3], *qrg_, *kr_, *vr_, *kd_, *vd_, *qb_, *kb_, *vb_, *rb_, *sk_, *att_, *o_, *oc_, *f_, *ci_, *cd_, *cg_, *cf_;
  float* mixp_ = nullptr;
  float* skp_ = nullptr;  // a long step's sinks or controllers' down outputs on 128 columns (the fp16 GEMM's), [max_batch][128]
  float* hs_ = nullptr;   // a step of several sub-batches: the experts' inputs of all of them, [max_step][n_embd]  // depth_mix's per-slice sums at a few rows (ik::kMixScratch a row)
  float *lg_, *wts_, *rows_, *logits_, *msg_;
  int32_t *ids_, *tok_, *slot_, *pos_, *base_, *lrows_, *tiles_sliding_, *tiles_full_;
  int64_t max_tiles_sliding_ = 0, max_tiles_full_ = 0;
  xmx::Activations act_, act_head_;
  attn::Scratch attn_scratch_;
  moe::Scratch* moe_scratch_ = nullptr;
  std::map<std::string, double>* profile_ = nullptr;
};

}  // namespace eklow::engine::inkling
