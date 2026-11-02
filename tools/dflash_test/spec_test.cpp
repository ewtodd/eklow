// eklow-spec-test: speculative decoding's acceptance (src/serve/spec.*) keeps the target's output distribution.
// Many trials on a small synthetic vocabulary: sequences of L tokens sampled by plain decoding (Sampler) and by draft +
// verify steps (a deliberately poor drafter, so every branch runs: acceptance, rejection with a residual, the bonus
// token), compared with the exact distribution plain sampling has (enumerated through Sampler::distribution) by a
// chi-square test, for temperature, top-k, top-p, min-p, penalties, bans, a grammar's allowed sets and all of them
// together; greedy runs must match plain greedy exactly. Two broken variants of verify must fail the same test (the
// check that the test can see), and the refactored Sampler::sample must draw what the previous implementation did.
//
//   eklow-spec-test [trials]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "serve/sampler.hpp"
#include "serve/spec.hpp"

namespace old_sampler {
// The Sampler as it was before Sampler::distribution (git 692bb33), verbatim, to check that sample() still draws the
// same tokens from the same seed.
class OldSampler {
 public:
  explicit OldSampler(const eklow::serve::SamplingParams& p);
  int32_t sample(std::span<const float> logits, const std::vector<int32_t>* allowed = nullptr, const std::vector<int32_t>* banned = nullptr);
  void accept(int32_t id) { ++counts_[id]; }

 private:
  eklow::serve::SamplingParams p_;
  std::mt19937_64 rng_;
  std::unordered_map<int32_t, int32_t> counts_;
  std::vector<std::pair<float, int32_t>> cand_;
};
}  // namespace old_sampler


namespace old_sampler {
using namespace eklow::serve;

OldSampler::OldSampler(const SamplingParams& p) : p_(p), rng_(p.seed) {}

int32_t OldSampler::sample(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned) {
  const int32_t V = int32_t(logits.size());
  auto adjusted = [&](int32_t id) {
    float l = logits[size_t(id)];
    if (auto it = counts_.find(id); it != counts_.end()) l -= p_.presence_penalty + p_.frequency_penalty * float(it->second);
    return l;
  };
  auto by_logit = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
  const bool bans = banned && !banned->empty();
  // The first banned id above `after` (V when none): the loops over the vocabulary below compare each id with it only.
  size_t b = 0;
  auto next_ban = [&](int32_t after) {
    while (bans && b < banned->size() && (*banned)[b] <= after) ++b;
    return bans && b < banned->size() ? std::min((*banned)[b], V) : V;
  };
  cand_.clear();
  if (allowed) {
    if (allowed->empty()) throw std::runtime_error("sampler: no token allowed");
    for (int32_t id : *allowed)
      if (!bans || !std::binary_search(banned->begin(), banned->end(), id)) cand_.push_back({adjusted(id), id});
    if (cand_.empty())
      for (int32_t id : *allowed) cand_.push_back({adjusted(id), id});
  } else if (p_.top_k > 0 || p_.temperature <= 0) {
    // The k best in one pass over the vocabulary, as a heap whose front is the worst kept (greedy keeps one): the
    // same set nth_element over all 262k candidates picked, at a fraction of its 1.2 ms a token.
    const size_t keep = p_.temperature <= 0 ? 1 : size_t(p_.top_k);
    for (int32_t id = 0, ban = next_ban(-1); id < V; ++id) {
      if (id == ban) {
        ban = next_ban(id);
        continue;
      }
      const std::pair<float, int32_t> c{counts_.empty() ? logits[size_t(id)] : adjusted(id), id};
      if (cand_.size() < keep) {
        cand_.push_back(c);
        std::push_heap(cand_.begin(), cand_.end(), by_logit);
      } else if (by_logit(c, cand_.front())) {
        std::pop_heap(cand_.begin(), cand_.end(), by_logit);
        cand_.back() = c;
        std::push_heap(cand_.begin(), cand_.end(), by_logit);
      }
    }
  } else {
    cand_.reserve(size_t(V));
    for (int32_t id = 0, ban = next_ban(-1); id < V; ++id) {
      if (id == ban) {
        ban = next_ban(id);
        continue;
      }
      cand_.push_back({counts_.empty() ? logits[size_t(id)] : adjusted(id), id});
    }
  }
  if (p_.temperature <= 0) return std::min_element(cand_.begin(), cand_.end(), by_logit)->second;

  // top-k first: it bounds the sort that top-p needs (selection, then sorting only the k).
  const size_t k = p_.top_k > 0 ? std::min(size_t(p_.top_k), cand_.size()) : cand_.size();
  if (k < cand_.size()) std::nth_element(cand_.begin(), cand_.begin() + std::ptrdiff_t(k - 1), cand_.end(), by_logit);
  cand_.resize(k);
  std::sort(cand_.begin(), cand_.end(), by_logit);
  // Softmax at the temperature, in double.
  std::vector<double> prob(k);
  const double top = cand_[0].first;
  double sum = 0;
  for (size_t i = 0; i < k; ++i) sum += prob[i] = std::exp((double(cand_[i].first) - top) / double(p_.temperature));
  for (auto& x : prob) x /= sum;
  size_t n = k;
  while (n > 1 && prob[n - 1] == 0) --n;  // underflowed: never picked, not even by the fallback below
  if (p_.min_p > 0)
    while (n > 1 && prob[n - 1] < double(p_.min_p) * prob[0]) --n;
  if (p_.top_p < 1) {
    double acc = 0;
    for (size_t i = 0; i < n; ++i) {
      acc += prob[i];
      if (acc >= double(p_.top_p)) {
        n = i + 1;
        break;
      }
    }
  }
  double total = 0;
  for (size_t i = 0; i < n; ++i) total += prob[i];
  double r = std::uniform_real_distribution<double>(0, total)(rng_);
  for (size_t i = 0; i < n; ++i) {
    r -= prob[i];
    if (r <= 0) return cand_[i].second;
  }
  return cand_[n - 1].second;
}

}  // namespace old_sampler

using namespace eklow::serve;

namespace {

constexpr int V = 6, L = 3, NDRAFT = 3, K = 4;

// The synthetic target: logits of a context, a fixed pseudo-random function of its last two tokens.
std::vector<float> target_logits(const std::vector<int32_t>& ctx) {
  uint64_t h = 0x9e3779b97f4a7c15ull;
  for (size_t i = ctx.size() >= 2 ? ctx.size() - 2 : 0; i < ctx.size(); ++i) h = (h ^ uint64_t(ctx[i] + 1)) * 0xff51afd7ed558ccdull;
  h ^= ctx.size() % 2;
  std::mt19937_64 g(h);
  std::vector<float> l(V);
  for (auto& x : l) x = std::uniform_real_distribution<float>(-2.0f, 2.0f)(g);
  return l;
}

struct Case {
  const char* name;
  SamplingParams sp;
  bool bans = false, grammar = false;
};

// Constraint sets after a context: bans are fixed ({2}, and 5 once a 4 has been seen); the "grammar" allows the tokens
// t with (t + last) % 3 != 0 (the first token: all but 0).
struct Constraints {
  const Case* c;
  std::vector<int32_t> allowed, banned;
  RowConstraint at(const std::vector<int32_t>& ctx) {
    allowed.clear();
    banned.clear();
    RowConstraint rc;
    if (c->bans) {
      banned.push_back(2);
      if (std::find(ctx.begin(), ctx.end(), 4) != ctx.end()) banned.push_back(5);
      rc.banned = &banned;
    }
    if (c->grammar) {
      const int last = ctx.empty() ? 1 : ctx.back();
      for (int t = 0; t < V; ++t)
        if ((t + last) % 3 != 0) allowed.push_back(t);
      rc.allowed = &allowed;
    }
    return rc;
  }
};

// Plain sampling's exact distribution over sequences of L tokens.
void enumerate(const Case& c, std::vector<int32_t>& ctx, double prob, std::map<std::vector<int32_t>, double>& out) {
  if (int(ctx.size()) == L) {
    out[ctx] = prob;
    return;
  }
  Sampler s(c.sp);
  for (int32_t t : ctx) s.accept(t);
  Constraints cons{&c, {}, {}};
  const RowConstraint rc = cons.at(ctx);
  std::vector<std::pair<int32_t, double>> d;
  s.distribution(target_logits(ctx), rc.allowed, rc.banned, {}, d);
  for (auto [t, p] : d) {
    ctx.push_back(t);
    enumerate(c, ctx, prob * p, out);
    ctx.pop_back();
  }
}

// The drafter: candidates, unary scores and pair scores that are a fixed pseudo-random function of the context, so
// the draft is wrong a good part of the time and sometimes proposes tokens the target bans or the grammar forbids.
Lattice drafter(const std::vector<int32_t>& ctx, int positions) {
  uint64_t h = 12345 + ctx.size() * 7919 + (ctx.empty() ? 0 : uint64_t(ctx.back()) * 104729);
  std::mt19937_64 g(h);
  Lattice lat;
  lat.top_k = K;
  for (int i = 0; i < positions; ++i) {
    std::vector<int32_t> ids(V);
    for (int t = 0; t < V; ++t) ids[size_t(t)] = t;
    std::shuffle(ids.begin(), ids.end(), g);
    std::vector<float> un(K);
    for (auto& x : un) x = std::uniform_real_distribution<float>(-1.5f, 1.5f)(g);
    std::sort(un.begin(), un.end(), std::greater<>());
    for (int c = 0; c < K; ++c) {
      lat.cand.push_back(ids[size_t(c)]);
      lat.unary.push_back(un[size_t(c)]);
    }
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) lat.score.push_back(un[size_t(b)] + std::uniform_real_distribution<float>(-1.0f, 1.0f)(g));
  }
  return lat;
}

// A drafter close to the target (its candidates are the target's top tokens along the greedy chain, its pair scores
// the target's logits after one candidate, with noise), so that long accepted runs and bonus tokens are common.
bool g_good_drafter = false;
Lattice good_drafter(const std::vector<int32_t>& ctx, int positions) {
  std::mt19937_64 g(777 + ctx.size() * 31 + (ctx.empty() ? 0 : uint64_t(ctx.back())));
  auto noise = [&] { return std::uniform_real_distribution<float>(-0.4f, 0.4f)(g); };
  Lattice lat;
  lat.top_k = K;
  std::vector<int32_t> chain = ctx;
  std::vector<int32_t> prev_cand;
  for (int i = 0; i < positions; ++i) {
    auto top = [&](const std::vector<int32_t>& h) {
      const auto l = target_logits(h);
      std::vector<int32_t> ids(V);
      for (int t = 0; t < V; ++t) ids[size_t(t)] = t;
      std::sort(ids.begin(), ids.end(), [&](int a, int b) { return l[size_t(a)] > l[size_t(b)]; });
      return std::make_pair(ids, l);
    };
    auto [ids, l] = top(chain);
    for (int c = 0; c < K; ++c) {
      lat.cand.push_back(ids[size_t(c)]);
      lat.unary.push_back(l[size_t(ids[size_t(c)])] + noise());
    }
    for (int a = 0; a < K; ++a) {
      std::vector<int32_t> h = chain;
      if (i > 0) h.back() = prev_cand[size_t(a)];
      const auto la = target_logits(h);
      for (int b = 0; b < K; ++b) lat.score.push_back(la[size_t(ids[size_t(b)])] + noise());
    }
    prev_cand.assign(lat.cand.end() - K, lat.cand.end());
    chain.push_back(ids[0]);
  }
  return lat;
}

enum class Mode { Correct, AlwaysAccept, NoResidual };

// verify with a defect, for the mutation check: AlwaysAccept takes a drafted token whenever p > 0 (ignoring q);
// NoResidual draws the correction from p instead of max(0, p - q).
VerifyResult broken_verify(Mode mode, Sampler& sampler, const DraftBlock& block, const std::function<std::span<const float>(int)>& row,
                           const ConstraintFn& constraint) {
  const int n = block.size();
  std::vector<std::pair<int32_t, double>> dist;
  auto dist_of = [&](int i) {
    const std::span<const int32_t> prefix(block.tokens.data(), size_t(i));
    const RowConstraint c = constraint(i, prefix);
    sampler.distribution(row(i), c.allowed, c.banned, prefix, dist);
  };
  auto draw = [&](const std::vector<std::pair<int32_t, double>>& d) {
    double r = sampler.uniform();
    for (auto& [id, p] : d) {
      r -= p;
      if (r < 0) return id;
    }
    return d.back().first;
  };
  for (int i = 0; i < n; ++i) {
    dist_of(i);
    const int32_t d = block.tokens[size_t(i)];
    double p = 0;
    for (auto& [id, pr] : dist)
      if (id == d) p = pr;
    if (mode == Mode::AlwaysAccept ? p > 0 : sampler.uniform() * block.q_of(i, d) < p) continue;
    return {i, draw(dist)};  // NoResidual (and, for AlwaysAccept, the p = 0 case)
  }
  dist_of(n);
  return {n, draw(dist)};
}

// One sequence of L tokens by draft + verify steps.
std::vector<int32_t> spec_sequence(const Case& c, Sampler& sampler, std::mt19937_64& draft_rng, Mode mode, long& steps, long& kept) {
  std::vector<int32_t> ctx;
  Constraints cons{&c, {}, {}};
  while (int(ctx.size()) < L) {
    const Lattice lat = g_good_drafter ? good_drafter(ctx, NDRAFT) : drafter(ctx, NDRAFT);
    const DraftBlock block = walk(lat, c.sp.temperature, draft_rng);
    auto row = [&](int i) {
      std::vector<int32_t> h = ctx;
      h.insert(h.end(), block.tokens.begin(), block.tokens.begin() + i);
      static thread_local std::vector<float> keep;
      keep = target_logits(h);
      return std::span<const float>(keep);
    };
    ConstraintFn cf = [&](int i, std::span<const int32_t> prefix) {
      std::vector<int32_t> h = ctx;
      h.insert(h.end(), prefix.begin(), prefix.end());
      return cons.at(h);
    };
    const VerifyResult r = mode == Mode::Correct ? verify(sampler, block, row, cf) : broken_verify(mode, sampler, block, row, cf);
    ++steps;
    for (int i = 0; i < r.accepted && int(ctx.size()) < L; ++i) {
      ctx.push_back(block.tokens[size_t(i)]);
      sampler.accept(ctx.back());
      ++kept;
    }
    if (int(ctx.size()) < L) {
      ctx.push_back(r.next);
      sampler.accept(r.next);
      ++kept;
    }
  }
  return ctx;
}

std::vector<int32_t> plain_sequence(const Case& c, Sampler& sampler) {
  std::vector<int32_t> ctx;
  Constraints cons{&c, {}, {}};
  while (int(ctx.size()) < L) {
    const RowConstraint rc = cons.at(ctx);
    ctx.push_back(sampler.sample(target_logits(ctx), rc.allowed, rc.banned));
    sampler.accept(ctx.back());
  }
  return ctx;
}

// Chi-square of observed counts against exact probabilities (cells expecting under 5 pooled); returns z, the
// Wilson-Hilferty normal score of the statistic (about N(0, 1) when the distributions agree; > 4 is a p of 3e-5).
double chi_z(const std::map<std::vector<int32_t>, double>& exact, const std::map<std::vector<int32_t>, long>& seen, long trials,
             double* stat_out, int* df_out) {
  double stat = 0, pool_e = 0, pool_o = 0;
  int cells = 0;
  for (auto& [seq, p] : exact) {
    const double e = p * double(trials);
    const long o = seen.count(seq) ? seen.at(seq) : 0;
    if (e < 5) {
      pool_e += e, pool_o += double(o);
    } else {
      stat += (double(o) - e) * (double(o) - e) / e;
      ++cells;
    }
  }
  if (pool_e >= 5) {
    stat += (pool_o - pool_e) * (pool_o - pool_e) / pool_e;
    ++cells;
  }
  // A sequence the exact distribution gives probability 0 must never occur.
  for (auto& [seq, o] : seen)
    if (!exact.count(seq)) stat += 1e9;
  const int df = std::max(1, cells - 1);
  *stat_out = stat;
  *df_out = df;
  const double k = double(df);
  return (std::cbrt(stat / k) - (1 - 2 / (9 * k))) / std::sqrt(2 / (9 * k));
}

}  // namespace

int main(int argc, char** argv) {
  const long trials = argc > 1 ? std::atol(argv[1]) : 200000;
  std::vector<Case> cases;
  auto mk = [&](const char* name, float temp, int top_k, float top_p, float min_p, float pres, float freq, bool bans, bool grammar) {
    Case c;
    c.name = name;
    c.sp.temperature = temp;
    c.sp.top_k = top_k;
    c.sp.top_p = top_p;
    c.sp.min_p = min_p;
    c.sp.presence_penalty = pres;
    c.sp.frequency_penalty = freq;
    c.bans = bans;
    c.grammar = grammar;
    cases.push_back(c);
  };
  mk("temperature 1", 1.0f, 0, 1.0f, 0, 0, 0, false, false);
  mk("temperature 0.7", 0.7f, 0, 1.0f, 0, 0, 0, false, false);
  mk("top-k 3", 1.0f, 3, 1.0f, 0, 0, 0, false, false);
  mk("top-p 0.8", 1.2f, 0, 0.8f, 0, 0, 0, false, false);
  mk("min-p 0.25", 1.0f, 0, 1.0f, 0.25f, 0, 0, false, false);
  mk("top-k 4, top-p 0.9, min-p 0.1", 0.9f, 4, 0.9f, 0.1f, 0, 0, false, false);
  mk("penalties", 1.0f, 0, 1.0f, 0, 0.8f, 0.5f, false, false);
  mk("bans", 1.0f, 0, 1.0f, 0, 0, 0, true, false);
  mk("grammar", 1.0f, 0, 1.0f, 0, 0, 0, false, true);
  mk("grammar + bans", 0.8f, 0, 1.0f, 0, 0, 0, true, true);
  mk("everything", 0.9f, 5, 0.9f, 0.05f, 0.6f, 0.4f, true, true);

  int failures = 0;
  // 1. The refactored sampler draws what the old one did, whatever the filters.
  {
    long diff = 0, n = 0;
    for (const Case& c : cases) {
      SamplingParams sp = c.sp;
      sp.seed = 99;
      Sampler a(sp);
      old_sampler::OldSampler b(sp);
      Constraints cons{&c, {}, {}};
      std::vector<int32_t> ctx;
      for (int i = 0; i < 5000; ++i) {
        const RowConstraint rc = cons.at(ctx);
        const auto l = target_logits(ctx);
        const int32_t x = a.sample(l, rc.allowed, rc.banned), y = b.sample(l, rc.allowed, rc.banned);
        diff += x != y, ++n;
        a.accept(x), b.accept(y);
        ctx.push_back(x);
        if (ctx.size() > 6) ctx.clear(), a = Sampler(sp), b = old_sampler::OldSampler(sp);
      }
    }
    std::printf("sampler: %ld of %ld draws differ from the previous implementation's: %s\n", diff, n, diff ? "FAIL" : "ok");
    failures += diff != 0;
  }
  // 2. Distribution tests.
  for (const Case& c : cases) {
    std::map<std::vector<int32_t>, double> exact;
    std::vector<int32_t> ctx;
    enumerate(c, ctx, 1.0, exact);
    for (int variant = 0; variant < 4; ++variant) {
      const Mode mode = variant <= 1 ? Mode::Correct : variant == 2 ? Mode::AlwaysAccept : Mode::NoResidual;
      g_good_drafter = variant == 1;
      std::map<std::vector<int32_t>, long> seen_plain, seen_spec;
      SamplingParams sp = c.sp;
      sp.seed = 1;
      long steps = 0, kept = 0;
      if (variant == 0) {
        for (long t = 0; t < trials; ++t) {
          Sampler s(sp);
          s = Sampler(SamplingParams{sp.temperature, sp.top_k, sp.top_p, sp.min_p, sp.presence_penalty, sp.frequency_penalty, uint64_t(t * 2 + 1)});
          ++seen_plain[plain_sequence(c, s)];
        }
        double st;
        int df;
        const double z = chi_z(exact, seen_plain, trials, &st, &df);
        std::printf("%-34s plain sampling : chi2 %7.1f df %3d z %+5.2f %s\n", c.name, st, df, z, z < 4 ? "ok" : "FAIL");
        failures += z >= 4;
      }
      for (long t = 0; t < trials; ++t) {
        Sampler s(SamplingParams{sp.temperature, sp.top_k, sp.top_p, sp.min_p, sp.presence_penalty, sp.frequency_penalty, uint64_t(t * 2 + 1000003)});
        std::mt19937_64 drng(uint64_t(t) * 7 + 5);
        ++seen_spec[spec_sequence(c, s, drng, mode, steps, kept)];
      }
      double st;
      int df;
      const double z = chi_z(exact, seen_spec, trials, &st, &df);
      const char* name = variant == 0 ? "draft + verify" : variant == 1 ? "good draft + verify" : mode == Mode::AlwaysAccept ? "BROKEN accept-all" : "BROKEN no-residual";
      const bool want_pass = mode == Mode::Correct;
      const bool pass = (z < 4) == want_pass;
      std::printf("%-34s %-18s: chi2 %7.1f df %3d z %+6.2f  (%.2f tokens a step) %s\n", c.name, name, st, df, z, double(kept) / double(steps),
                  pass ? (want_pass ? "ok" : "caught") : (want_pass ? "FAIL" : "NOT CAUGHT"));
      failures += !pass;
    }
  }
  // 3. Greedy: the same tokens as plain greedy decoding, for every constraint set, with the poor and the good drafter.
  for (int good = 0; good < 2; ++good)
  for (bool bans : {false, true})
    for (bool grammar : {false, true}) {
      g_good_drafter = good;
      Case c;
      c.name = "greedy";
      c.sp.temperature = 0;
      c.sp.presence_penalty = 0.5f;
      c.sp.frequency_penalty = 0.3f;
      c.bans = bans;
      c.grammar = grammar;
      long bad = 0, steps = 0, kept = 0;
      for (long t = 0; t < 20000; ++t) {
        Sampler a(c.sp), b(c.sp);
        std::mt19937_64 drng(uint64_t(t) * 7 + 5);
        bad += plain_sequence(c, a) != spec_sequence(c, b, drng, Mode::Correct, steps, kept);
      }
      std::printf("greedy, %s drafter (bans %d, grammar %d): %ld of 20000 sequences differ from plain greedy (%.2f tokens a step): %s\n", good ? "good" : "poor", bans, grammar,
                  bad, double(kept) / double(steps), bad ? "FAIL" : "ok");
      failures += bad != 0;
    }
  std::printf("%s\n", failures ? "FAILED" : "all ok");
  return failures != 0;
}
