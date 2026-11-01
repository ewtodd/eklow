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
  std::vector<float> hidden;  // [B-1][n_embd] after the final norm
};

class Dflash2Ref {
 public:
  // `window` 0: the draft's own. max_pos bounds the positions (RoPE tables are built to it).
  Dflash2Ref(sycl::queue& q, const Dflash2Config& c, const Dflash2Weights& w, int64_t max_pos, int window = 0);
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
  int64_t max_pos_;
  std::vector<float*> k_, v_;  // per layer [max_pos][n_head_kv][head_dim]
  float *cos_ = nullptr, *sin_ = nullptr;
};

}  // namespace eklow::model
