// The scheduler's decisions about cached prefixes, as functions of numbers only, so that they are tested on the host
// (tools/server/prefix_store_test.cpp).
#pragma once

#include <algorithm>
#include <cstdint>

namespace eklow::serve {

// How much of a slot's sequence a new prompt reuses, given the `keep` positions it could (they agree, and leave the
// prompt's last token and no image cut): the slot holds `held`, its sliding rings the last ring_slack + window positions.
// Going on from `keep` needs the window of K/V before it in the rings, which they still hold only when the slot gives back
// at most ring_slack positions. Otherwise the sequence starts over: the positions before that window are overwritten too,
// and prefilling the window again would read them (each position's sliding layers look a window back), so keeping fewer
// positions does not help.
inline int64_t ring_safe_keep(int64_t keep, int64_t held, int64_t ring_slack) { return held - keep <= ring_slack ? keep : 0; }

// The disk cache's restore point for a prompt of `prompt` tokens, in pages: at most the full pages before its last token,
// less the draft's reach (`reach_pages`: the positions after a restore are prefilled again, so that the draft's context,
// which a restore does not bring back, is whole by the end of the prompt).
inline int64_t restore_limit(int64_t prompt, int64_t page, int64_t reach_pages) {
  return std::max<int64_t>(0, (prompt - 1) / page - reach_pages);
}

// Where a sequence's sliding parts go when it reaches `held` positions: the pages a restore at its point needs, [from, to).
// The point is the full pages held less the draft's reach, so that a later prompt extending this sequence restores there.
struct SlidingPages {
  int64_t from = 0, to = 0;
};
inline SlidingPages point_pages(int64_t held, int64_t page, int64_t reach_pages, int64_t window_pages) {
  const int64_t t = held / page - reach_pages;
  if (t <= 0) return {};
  return {std::max<int64_t>(0, t - window_pages), t};
}

// Whether a restore of `disk` positions beats prefilling from `keep` (the slot's reuse): the prefill it saves against the
// time to read, open and copy the pages. The rates are estimates for fiona (DECISIONS 2026-11-22).
struct RestoreCost {
  double prefill_tok_s = 2500;   // Bolt's prefill at depth (3,549 at 8k, 2,273 at 60k)
  double restore_pos_s = 60000;  // positions a second a restore brings back (16 KiB of global K/V each)
  double fixed_s = 0.02;         // the sliding parts and the start of a restore
};
inline bool prefer_restore(int64_t keep, int64_t disk, const RestoreCost& c) {
  if (disk <= keep) return false;
  return double(disk - keep) / c.prefill_tok_s > c.fixed_s + double(disk) / c.restore_pos_s;
}

}  // namespace eklow::serve
