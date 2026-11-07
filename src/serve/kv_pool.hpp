// The scheduler's decisions about the shared K/V pool, as plain functions over numbers so that they test on the host (no card):
// which idle slots give pages back and how far each is cut, and whether a waiting request fits. Positions are counted in pages of
// `page` positions (attn::kPage); the engine owns the pages, the scheduler the policy (scheduler.cpp).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace eklow::serve::kvpool {

inline int64_t pages_for(int64_t page, int64_t positions) { return positions <= 0 ? 0 : (positions + page - 1) / page; }

// The most positions a step of one sequence adds at once: a verify block (up to 16 rows) with a draft, else one token. A request is
// admitted only if its prompt and this much more fit, so that its first step after the prompt does not need a page nobody has.
inline int64_t step_growth(bool draft) { return draft ? 16 : 1; }

// A slot as the reclaim sees it.
struct Holder {
  int64_t pages = 0;               // held by the engine
  bool idle = false;               // no request: its content is only a prefix cache
  uint64_t used = 0;               // tick of its last use; the least recent gives back first
  std::vector<int64_t> snaps;      // positions of its snapshots (the DeltaNet state exists only there and at the slot's end)
};

struct Trim {
  int slot = 0;
  int64_t keep = 0;   // positions the slot keeps: 0 (release it) or a snapshot's position
  int64_t freed = 0;  // pages this gives back
};

struct ReclaimPlan {
  std::vector<Trim> trims;
  int64_t freed = 0;
  bool enough = false;
};

// Pages to take back from idle slots so that `need` are free (`free` are now): the idle slot used least recently first, cut from its end
// by as little as will do. A slot's recurrent state can only be put back at a snapshot, so what it keeps is the longest snapshot position
// that still leaves enough (else nothing: release). Slots with a request are never touched. `enough` is false if every idle slot
// together cannot cover it (the plan then releases them all).
inline ReclaimPlan plan_reclaim(int64_t page, int64_t free, int64_t need, const std::vector<Holder>& slots) {
  ReclaimPlan plan;
  int64_t deficit = need - free;
  std::vector<int> order;
  for (size_t i = 0; i < slots.size(); ++i)
    if (slots[i].idle && slots[i].pages > 0) order.push_back(int(i));
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return slots[size_t(a)].used < slots[size_t(b)].used; });
  for (int i : order) {
    if (deficit <= 0) break;
    const Holder& h = slots[size_t(i)];
    int64_t keep = 0;
    if (deficit < h.pages) {
      const int64_t room = (h.pages - deficit) * page;  // positions that leave `deficit` pages free
      for (int64_t p : h.snaps)
        if (p > keep && p <= room) keep = p;
    }
    const int64_t freed = h.pages - pages_for(page, keep);
    plan.trims.push_back({i, keep, freed});
    plan.freed += freed;
    deficit -= freed;
  }
  plan.enough = deficit <= 0;
  return plan;
}

// A slot as the admission sees it.
struct AdmitHolder {
  int64_t pages = 0;  // held by the engine
  bool idle = false;
  int64_t want = 0;   // with a request: the pages it needs through the end of its prompt and one step more (what it holds counts)
};

// Whether the request for slot `target` fits: its prompt and the step's growth need `target_want` pages in all, of which the slot already
// keeps `target_keep` (the pages past them go back to the pool when the slot restores a snapshot or is released); the pool's `free` pages,
// the pages of the other idle slots (reclaimable) and those the target gives up pay for it, less what the running requests still need to
// finish their prompts. Growth in generation is not reserved: that is the pool running out, and the scheduler ends a sequence then.
inline bool admission_fits(int64_t free, const std::vector<AdmitHolder>& slots, int target, int64_t target_keep, int64_t target_want) {
  int64_t avail = free, reserved = 0;
  for (size_t i = 0; i < slots.size(); ++i) {
    const AdmitHolder& h = slots[i];
    if (int(i) == target) avail += std::max<int64_t>(0, h.pages - target_keep);
    else if (h.idle) avail += h.pages;
    else reserved += std::max<int64_t>(0, h.want - h.pages);
  }
  return std::max<int64_t>(0, target_want - target_keep) + reserved <= avail;
}

}  // namespace eklow::serve::kvpool
