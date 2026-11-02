// The DFlash 2 draft (Inco AI, August 2026; z-lab's dflash/model.py, DFlash2DraftModel) for Qwen3.6-35B-A3B and its
// finetunes, from the GGUF llama.cpp's converter writes (arch "dflash"), and what the engine needs of the target: its
// token embedding table and its output head. DECISIONS 2026-10-31 and TASK qwen/dflash have the whole story; in short:
//
//   context   per committed token the target's 8 layer outputs (layers 1, 6, ..., 37), 16,384 values, go through `fc`
//             and a norm (`enc.output_norm`), and per draft layer become K = rope(k_norm(k_proj(.))), V = v_proj(.) in the
//             draft's cache, a sliding window (2,048 or 4,096 positions);
//   block     [anchor, MASK x (B-1)] at positions p..p+B-1, embedded with the target's table, through six layers:
//               n = rmsnorm(x); dyn = conv_proj(n); n' = conv(n, dyn, side 0);
//               q, k, v = n' W; q, k normed per head, roped (NeoX, base 1e7, all 128 dims); attention over the window's
//               context keys and the block's own, non-causal in the block; a = conv(o_proj(.), dyn, side 1); x += a;
//               n = rmsnorm(x); dyn = conv_proj(n); m = conv(n, dyn, 0); x += conv(down(silu(gate m) * up m), dyn, 1);
//             then a final rmsnorm; the target's head on rows 1..B-1 gives each position's logits;
//   selector  the top 16 candidates of each position (their logits the unary scores) and the lattice of pair scores
//             unary[b] + <pred_codebook[prev] * hidden_projection(h), succ_codebook[cand b]>, walked from the anchor.
// The convolution is a 2-tap causal one over the block's positions only, per channel base[side][tap][c] plus a
// coefficient per token and group of 16 channels from conv_proj (model.py: GroupedDynamicCausalConv).
//
// Dflash2Ref is the reference forward (fp32, plain SYCL, ref.hpp's kernels and ref_dflash.hpp's); Dflash2 the serving
// forward (src/model/dflash2_fast.cpp): the matmuls on the matrix units in the draft file's own format, attention
// through attn::attend on a ring cache with a key range per token, the convolutions and the selector as kernels of
// src/kernels/dflash.*, the target's head on the card with the top 16 of each row taken there.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "kernels/attention.hpp"
#include "kernels/ref.hpp"
#include "kernels/xmx.hpp"
#include "model/gguf.hpp"
#include "serve/spec.hpp"

namespace eklow::model {

struct Dflash2Config {
  int n_layer, n_embd, n_ff, n_head, n_head_kv, head_dim, n_vocab;
  int window;      // the sliding window (keys a query sees: positions q - window + 1 .. q)
  int block_size;  // B the draft was trained for
  int conv_kernel, conv_group, rank, top_k;
  int mask_token;
  std::vector<int> target_layers;  // the target's layers whose outputs it reads (the GGUF stores each plus one)
  float eps, rope_base;
  int n_feat() const { return int(target_layers.size()) * n_embd; }
  // Throws when the file is not a DFlash 2 draft this code knows how to run.
  static Dflash2Config from_gguf(const gguf::File& f);
};

// The draft's tensors on the card as stored (the reference's), plus the target's embedding table and head.
class Dflash2Weights {
 public:
  struct Layer {
    ref::Weight wq, wk, wv, wo, gate, up, down, conv_attn, conv_ffn;  // conv_*: the kernel projections
    const float *attn_norm, *q_norm, *k_norm, *ffn_norm, *base_attn, *base_ffn;  // base_*: [side][tap][hidden]
  };
  static Dflash2Weights upload(const gguf::File& draft, const gguf::File& target, const Dflash2Config& c, sycl::queue& q);
  Dflash2Weights(Dflash2Weights&&) noexcept;
  Dflash2Weights& operator=(Dflash2Weights&&) = delete;
  Dflash2Weights(const Dflash2Weights&) = delete;
  ~Dflash2Weights();

  ref::Weight fc, sel_hidden, sel_pred, sel_succ;  // sel_*: rows of `rank` per token id
  ref::Weight tok_embd, head;                      // the target's, [n_vocab] rows of n_embd
  const float *hidden_norm = nullptr, *out_norm = nullptr;
  std::vector<Layer> layers;
  uint64_t bytes = 0;

 private:
  explicit Dflash2Weights(sycl::queue& q) : q_(&q) {}
  sycl::queue* q_;
  std::vector<void*> allocations_;
};

// What a block pass returns beside the lattice, for checks.
struct BlockDebug {
  int stage_layer = 0;                      // the layer whose intermediates `stages` records
  std::vector<float> hidden;                // [B-1][n_embd] after the final norm
  std::vector<std::vector<float>> layers;   // the residual stream [B][n_embd] after each layer
  std::vector<std::pair<std::string, std::vector<float>>> stages;  // stage_layer's intermediates by name
  std::vector<std::vector<float>> parts;    // per layer, [B][..]: the context-free pieces named in dflash2.cpp (attention output, ...)
};

class Dflash2Ref {
 public:
  // `window` 0: the draft's own. max_pos bounds the positions (RoPE tables are built to it).
  // With `emulate_acts` every matmul's input is rounded the way the serving forward quantizes it (int8 per 32-block up to 32
  // rows, fp16 above), so that the two can be compared to rounding error.
  Dflash2Ref(sycl::queue& q, const Dflash2Config& c, const Dflash2Weights& w, int64_t max_pos, int window = 0, bool emulate_acts = false, float jitter = 0);
  ~Dflash2Ref();
  // Adds the context rows of positions pos0 .. pos0 + n - 1 (feats: host fp32 [n][n_feat]); they replace whatever the
  // cache held there (the keys of an earlier block, which are dropped).
  void inject(std::span<const float> feats, int64_t pos0, int64_t n);
  // A block of B tokens at positions pos..pos+B-1 (the context rows below pos must have been injected): the lattice
  // for positions 1..B-1. Candidates by descending logit, equal logits by ascending id.
  serve::Lattice block(int32_t anchor, int64_t pos, int B, BlockDebug* debug = nullptr);

 private:
  sycl::queue& q_;
  const Dflash2Config& c_;
  const Dflash2Weights& w_;
  int window_;
  bool emulate_;
  float jitter_;
  int64_t max_pos_;
  std::vector<float*> k_, v_;  // per layer [max_pos][n_head_kv][head_dim]
  float *cos_ = nullptr, *sin_ = nullptr;
};

// What the draft reads of the target, on the card: the token embedding table as stored (any type ref::embed takes) and
// the output head in a format xmx::matmul takes (Q8_0I, Q6_K, ...). Views: the engine owns them.
struct Dflash2Target {
  ref::Weight tok_embd;
  xmx::Weight head;
  // If set, tok_embd.data is HOST memory (the target's file as mapped) and the draft copies the rows a pass embeds (each
  // block's anchor and the mask token) to a small table on the card, instead of keeping the whole table there: 515 MiB
  // of the card, 7 experts of every layer, for a few row copies a pass.
  bool host_table = false;
};

// A Dflash2Target loaded from a GGUF holding the target's `token_embd.weight` and `output.weight` (the target file, or
// tools/dflash_test/extract_head.py's extract of it), for tools and tests.
class OwnedTarget {
 public:
  OwnedTarget(sycl::queue& q, const gguf::File& target);
  OwnedTarget(const OwnedTarget&) = delete;
  ~OwnedTarget();
  Dflash2Target view() const { return {tok_embd, head}; }

 private:
  sycl::queue& q_;
  ref::Weight tok_embd{};
  xmx::Weight head{};
  void* raw_ = nullptr;
};

// Where a draft block's time goes (Dflash2::draft with a Profile waits between phases, so it is slower).
struct Profile {
  double embed_ms = 0, layers_ms = 0, head_ms = 0, select_ms = 0, host_ms = 0;
  double inject_fc_ms = 0, inject_kv_ms = 0, inject_write_ms = 0;
};

// The serving forward. Slots are independent sequences, each with a ring cache of the draft's window; a block (or
// several, one per slot) runs as one pass of up to 32 rows through the layers.
class Dflash2 {
 public:
  struct Options {
    int slots = 1;
    int max_block = 16;      // the largest B asked of draft()
    int max_inject = 1024;   // rows a context injection runs through the matmuls at once
    int window = 0;          // 0: the draft's own
  };
  // `draft` must stay open (its tensors are read at construction only). Formats: whatever xmx::format_of takes (Q8_0 is
  // the draft's shipped one); BF16 and F16 files only run in Dflash2Ref.
  Dflash2(sycl::queue& q, const gguf::File& draft, const Dflash2Target& target, const Options& opt);
  ~Dflash2();
  Dflash2(const Dflash2&) = delete;
  const Dflash2Config& config() const { return c_; }
  int window() const { return window_; }

  // A slot's context is the committed tokens' positions [first, end): reset empties it, truncate(slot, pos) drops the rows from
  // pos on (positions the next inject or block write over).
  void reset(int slot);
  void truncate(int slot, int64_t pos);
  int64_t context_end(int slot) const { return end_[size_t(slot)]; }

  // Adds the context rows of positions pos0 .. pos0 + n - 1, which continue the slot's context (pos0 <= context_end; a
  // later pos0 restarts the context there). feats is the target's layer outputs on the card, fp32 [n][n_feat] (layers 1, 6, ... in order, each
  // n_embd wide). Rows further back than a later block can see (the window) are not computed.
  void inject(int slot, const float* feats, int64_t pos0, int64_t n);

  // The engine's form: n rows of a step in its order, each with its slot and position (host arrays). Consecutive rows of
  // one slot at consecutive positions form a run, injected as inject(slot, ...) does. The caller passes the rows that
  // are committed: after a verify block with a accepted tokens that is the block's first a + 1 rows (the anchor and the
  // accepted drafts), so the draft's context never holds a rejected token; the rows of the rest are not read. A run
  // may start past the slot's context end (a prefill that captured only its last window rows): the context then starts
  // there.
  void inject_rows(const float* feats, const int32_t* slots, const int64_t* positions, int64_t n);

  struct Request {
    int slot;
    int32_t anchor;  // the last sampled token, at position pos
    int64_t pos;     // must equal the slot's context_end
  };
  // The lattices of one block of B tokens per request (anchor, then B - 1 masks): all requests in one pass (B the same,
  // at most max_block, requests * B at most 32 rows).
  std::vector<serve::Lattice> lattices(const std::vector<Request>& reqs, int B, Profile* profile = nullptr, BlockDebug* debug = nullptr);
  // A drafted block: the walk over the lattice (serve::walk) at `temperature` (0: greedy).
  std::vector<serve::DraftBlock> draft(const std::vector<Request>& reqs, int B, float temperature, std::mt19937_64& rng, Profile* profile = nullptr);
  serve::DraftBlock draft(int slot, int32_t anchor, int64_t pos, int B, float temperature, std::mt19937_64& rng, Profile* profile = nullptr);

  // The device memory the draft holds (weights, caches, scratch).
  uint64_t bytes() const { return bytes_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
  sycl::queue& q_;
  Dflash2Config c_;
  int window_;
  Options opt_;
  std::vector<int64_t> end_, first_;
  uint64_t bytes_ = 0;
};

}  // namespace eklow::model
