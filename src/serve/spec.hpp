// Speculative decoding with DFlash 2 drafts (DECISIONS 2026-10-31): what a draft block is, the walk over the
// selector's lattice that picks it, and the acceptance of a block against the target's logits, which keeps the
// target's output distribution exactly what plain decoding would sample. Host code, no device types: the draft
// (src/model/dflash2.*) hands over a Lattice, the engine the target's logits rows.
//
// The target's distribution p of a row is Sampler::distribution (penalties, bans, the tool grammar's allowed set,
// temperature, top-k, min-p, top-p, in the sampler's order), so everything the sampler can do enters p the same way
// it enters plain sampling; what the draft proposes only decides how much work is saved, never what comes out:
//   accept token d_i with probability min(1, p_i(d_i) / q_i(d_i)); on the first rejection draw from
//   max(0, p_i - q_i) renormalized (p_i itself if that is all zero); all accepted: draw from the next row.
// (z-lab's `_rejection_sample`.) Row i is the target's output after the anchor and d_0..d_{i-1}, so p_i is conditioned
// on the same speculative prefix as q_i, and a row's penalties count d_0..d_{i-1} as accepted, its bans and grammar
// state are those after them (`Constraint` below). Greedy decoding (temperature 0): accept while d_i is the row's
// argmax (under the same penalties, bans and grammar), else take the argmax: the tokens plain greedy decoding gives.
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

#include "serve/sampler.hpp"

namespace eklow::serve {

// The draft's candidates for P positions and the scores of the selector's pairs (model.py CandidateSelector.select):
// cand [P][k] by descending unary score, unary [P][k] their logits, score [P][k][k] with score[i][a][b] the score of
// candidate b at position i after candidate a of position i - 1 (position 0: after the anchor, the same for every a).
struct Lattice {
  int top_k = 0;
  std::vector<int32_t> cand;
  std::vector<float> unary, score;
  int positions() const { return top_k ? int(cand.size()) / top_k : 0; }
};

// A drafted block: tokens d_0..d_{n-1} for the positions after the anchor; when sampled, cand [n][k] (the position's
// candidate ids) and q [n][k] (the distribution d_i was drawn from, over them, given d_{i-1}).
struct DraftBlock {
  int top_k = 0;
  std::vector<int32_t> tokens;
  std::vector<int32_t> cand;
  std::vector<float> q;
  int size() const { return int(tokens.size()); }
  // q_i(token), 0 for a token outside the candidates (or for greedy blocks, which carry no q).
  double q_of(int i, int32_t token) const;
};

// The walk over the lattice: temperature 0 takes the best-scoring candidate at each position given the previous one
// (llama.cpp's walk, ties to the first); otherwise it samples the softmax of the scores at the temperature, over the
// k candidates only (the draft's q), with `rng`.
DraftBlock walk(const Lattice& lat, float temperature, std::mt19937_64& rng);

// What constrains the target's row after a speculative prefix: the allowed set (null: none) and the banned ids
// (ascending), both valid until the next call. `prefix` is d_0..d_{row-1}. A caller with a grammar and an output
// parser clones their states, feeds the prefix and reads their sets; with none the default is no constraint.
struct RowConstraint {
  const std::vector<int32_t>* allowed = nullptr;
  const std::vector<int32_t>* banned = nullptr;
};
using ConstraintFn = std::function<RowConstraint(int row, std::span<const int32_t> prefix)>;

struct VerifyResult {
  int accepted = 0;  // how many of the block's tokens are kept
  int32_t next = 0;  // the token after them: the correction at the first rejection, or the bonus token
};

// Rows 0..n (n = block.size()) of the target's logits, `row(i)`. The kept tokens are block.tokens[0..accepted) then
// `next`; the caller accepts them into the sampler (Sampler::accept) and the grammar. `sampler`'s generator supplies
// the randomness, so a run is reproducible from its seed.
VerifyResult verify(Sampler& sampler, const DraftBlock& block, const std::function<std::span<const float>(int)>& row,
                    const ConstraintFn& constraint = {});

}  // namespace eklow::serve
