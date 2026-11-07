// eklow-kvpool-test: the scheduler's decisions about the shared K/V pool (kv_pool.hpp), on the host, no card.
//
//   eklow-kvpool-test
//
// 1. plan_reclaim: nothing to do when enough is free; the idle slot used least recently first; a cut keeps the longest snapshot
//    position that still leaves enough (else the slot is emptied); the next slot only when the first cannot cover it; slots with a
//    request never touched; every idle slot emptied, and `enough` false, when even that is not enough.
// 2. admission_fits: free pages, other idle slots' pages and the target's pages past its kept prefix pay; running requests' unfinished
//    prompts are reserved.
// 3. A random sweep of reclaim plans against the invariants (what is kept is 0 or a snapshot position, the freed pages cover the deficit
//    when `enough`, a slot with a request is never cut, a larger snapshot would not have fitted).
#include <cstdio>
#include <random>

#include "serve/kv_pool.hpp"

using namespace eklow::serve::kvpool;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
  std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}

constexpr int64_t P = 256;

Holder idle(int64_t positions, uint64_t used, std::vector<int64_t> snaps = {}) {
  Holder h;
  h.pages = pages_for(P, positions), h.idle = true, h.used = used, h.snaps = std::move(snaps);
  return h;
}
Holder busy(int64_t positions) {
  Holder h;
  h.pages = pages_for(P, positions);
  return h;
}

void test_reclaim() {
  std::printf("plan_reclaim\n");
  check(pages_for(P, 0) == 0 && pages_for(P, 1) == 1 && pages_for(P, 256) == 1 && pages_for(P, 257) == 2, "pages_for");
  {
    const auto plan = plan_reclaim(P, 10, 10, {idle(4096, 1)});
    check(plan.enough && plan.trims.empty() && plan.freed == 0, "enough free: no trims");
  }
  {
    // 16 pages held, 4 needed beyond the 0 free; snapshots at 1000, 2000, 3000 (pages 4, 8, 12): room = 12 pages = 3072 positions.
    const auto plan = plan_reclaim(P, 0, 4, {idle(4096, 1, {1000, 2000, 3000})});
    check(plan.enough && plan.trims.size() == 1 && plan.trims[0].keep == 3000 && plan.trims[0].freed == 16 - 12 && plan.freed == 4,
          "one idle slot: cut to the longest snapshot that leaves enough");
  }
  {
    // 6 pages needed: room = 10 pages = 2560: the snapshot at 2000 (8 pages) is the longest that fits.
    const auto plan = plan_reclaim(P, 0, 6, {idle(4096, 1, {1000, 2000, 3000})});
    check(plan.enough && plan.trims[0].keep == 2000 && plan.trims[0].freed == 8, "a bigger need keeps a shorter snapshot");
  }
  {
    // No snapshot fits in the room: emptied.
    const auto plan = plan_reclaim(P, 0, 4, {idle(4096, 1, {3900})});
    check(plan.enough && plan.trims[0].keep == 0 && plan.trims[0].freed == 16, "no snapshot that fits: the slot is emptied");
  }
  {
    // A snapshot exactly at the room's edge.
    const auto plan = plan_reclaim(P, 0, 4, {idle(4096, 1, {3072})});
    check(plan.trims[0].keep == 3072 && plan.trims[0].freed == 4, "a snapshot at the edge of the room is kept");
  }
  {
    // The least recently used goes first, and alone when it covers the need.
    const auto plan = plan_reclaim(P, 0, 3, {idle(2048, 9, {1024}), idle(2048, 2, {1024})});
    check(plan.trims.size() == 1 && plan.trims[0].slot == 1, "least recently used slot first");
  }
  {
    // The first (older) slot cannot cover it with a cut; it is emptied (8 pages) and the next one gives the rest.
    const auto plan = plan_reclaim(P, 0, 10, {idle(2048, 9, {1024}), idle(2048, 2)});
    check(plan.enough && plan.trims.size() == 2 && plan.trims[0].slot == 1 && plan.trims[0].keep == 0 && plan.trims[1].slot == 0 && plan.trims[1].keep == 1024 &&
              plan.freed == 12,
          "the next slot only for what the first cannot give");
  }
  {
    const auto plan = plan_reclaim(P, 0, 5, {busy(4096), idle(1024, 1)});
    check(!plan.enough && plan.trims.size() == 1 && plan.trims[0].slot == 1 && plan.freed == 4, "a slot with a request is never cut; not enough is reported");
  }
  {
    const auto plan = plan_reclaim(P, 3, 5, {idle(0, 1), idle(1024, 2, {512})});
    check(plan.enough && plan.trims.size() == 1 && plan.trims[0].slot == 1 && plan.trims[0].keep == 512 && plan.trims[0].freed == 2, "an empty slot is skipped; free pages count");
  }
}

void test_admission() {
  std::printf("admission_fits\n");
  auto A = [](int64_t pages, bool idle_, int64_t want = 0) {
    AdmitHolder h;
    h.pages = pages, h.idle = idle_, h.want = want;
    return h;
  };
  // Pool of 1024 pages. Target slot 0 (idle, holds 100 of which it keeps 60), slot 1 idle holds 500, 300 are free.
  check(admission_fits(300, {A(100, true), A(500, true)}, 0, 60, 900), "idle slots' pages and the target's pages beyond its prefix pay: 840 needed of 940");
  check(!admission_fits(300, {A(100, true), A(500, true)}, 0, 60, 1000), "one page over: waits");
  // The other slot is running: its pages are not reclaimable.
  check(!admission_fits(300, {A(100, true), A(500, false, 500)}, 0, 60, 500), "a running slot's pages are not available: 440 needed of 340");
  check(admission_fits(300, {A(100, true), A(500, false, 500)}, 0, 60, 390), "a request that fits beside the running slot goes in");
  // A running slot that has not finished its prompt reserves the rest.
  check(!admission_fits(300, {A(100, true), A(200, false, 400)}, 0, 100, 400), "the running prompt's unfinished part (200) is reserved: 300 needed of 300+0 - 200");
  check(admission_fits(300, {A(100, true), A(200, false, 400)}, 0, 100, 190), "...and a request within what is left goes in");
  // A running slot already past its prompt reserves nothing.
  check(admission_fits(10, {A(0, true), A(900, false, 100)}, 0, 0, 10), "a running slot past its prompt reserves nothing");
  // The target keeps all it holds: only the rest is needed.
  check(admission_fits(5, {A(100, true), A(0, true)}, 0, 100, 105), "the kept prefix is not paid twice");
  check(step_growth(true) == 16 && step_growth(false) == 1, "step_growth");
}

void test_random() {
  std::printf("random plans\n");
  std::mt19937_64 rng(7);
  int bad = 0, plans = 0, enough = 0;
  for (int it = 0; it < 20000; ++it) {
    std::vector<Holder> hs(size_t(1 + rng() % 4));
    for (Holder& h : hs) {
      h.pages = int64_t(rng() % 200);
      h.idle = rng() % 4 != 0;
      h.used = rng() % 50;
      for (int k = int(rng() % 5); k > 0 && h.pages > 0; --k) h.snaps.push_back(1 + int64_t(rng() % uint64_t(h.pages * P)));
    }
    const int64_t free = int64_t(rng() % 100), need = free + int64_t(rng() % 300) - 20;
    const ReclaimPlan plan = plan_reclaim(P, free, need, hs);
    ++plans;
    enough += plan.enough;
    int64_t freed = 0;
    bool ok = true;
    uint64_t last_used = 0;
    for (const Trim& t : plan.trims) {
      const Holder& h = hs[size_t(t.slot)];
      ok = ok && h.idle && h.pages > 0 && t.keep >= 0 && pages_for(P, t.keep) <= h.pages && t.freed == h.pages - pages_for(P, t.keep) && h.used >= last_used;
      last_used = h.used;
      if (t.keep > 0) {
        bool is_snap = false;
        for (int64_t s : h.snaps) is_snap = is_snap || s == t.keep;
        ok = ok && is_snap;
      }
      freed += t.freed;
    }
    ok = ok && freed == plan.freed && plan.enough == (free + freed >= need);
    if (need - free <= 0) ok = ok && plan.trims.empty();
    // Only idle slots, so everything idle emptied is the most it can give; `enough` false means that is what it did.
    if (!plan.enough) {
      int64_t all = 0;
      for (const Holder& h : hs) all += h.idle ? h.pages : 0;
      ok = ok && plan.freed == all;
    } else if (!plan.trims.empty()) {
      // The last cut is the only one that keeps something, and a longer snapshot would not have left enough.
      const Trim& t = plan.trims.back();
      const Holder& h = hs[size_t(t.slot)];
      const int64_t before = free + freed - t.freed;  // pages free before this cut
      for (int64_t s : h.snaps)
        if (s > t.keep) ok = ok && free + (freed - t.freed) + (h.pages - pages_for(P, s)) < need;
      for (size_t i = 0; i + 1 < plan.trims.size(); ++i) ok = ok && plan.trims[i].keep == 0;
      (void)before;
    }
    if (!ok) ++bad;
  }
  std::printf("  %d plans (%d with enough)\n", plans, enough);
  check(bad == 0, "invariants hold in every plan");
}

}  // namespace

int main() {
  test_reclaim();
  test_admission();
  test_random();
  std::printf("%s\n", g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
