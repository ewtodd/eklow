#include "serve/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace eklow::serve {

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed) {}

size_t Sampler::prepare(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned,
                        std::span<const int32_t> extra) {
  const int32_t V = int32_t(logits.size());
  // Speculative tokens (a draft's, assumed accepted) count for the penalties like accepted ones, without being kept.
  std::unordered_map<int32_t, int32_t> extra_counts;
  for (int32_t t : extra) ++extra_counts[t];
  const bool no_counts = counts_.empty() && extra_counts.empty();
  auto adjusted = [&](int32_t id) {
    float l = logits[size_t(id)];
    int32_t c = 0;
    if (auto it = counts_.find(id); it != counts_.end()) c += it->second;
    if (auto it = extra_counts.find(id); it != extra_counts.end()) c += it->second;
    if (c > 0) l -= p_.presence_penalty + p_.frequency_penalty * float(c);
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
    // Once k are kept, an id whose logit does not exceed the worst kept one cannot enter (ids ascend, so it would lose an
    // equal logit to the earlier id): that test alone is the loop for nearly every id, and a verify block runs it per row.
    float worst = std::numeric_limits<float>::infinity();
    for (int32_t id = 0, ban = next_ban(-1); id < V; ++id) {
      if (no_counts && !(logits[size_t(id)] > worst) && cand_.size() >= keep) {
        if (id == ban) ban = next_ban(id);
        continue;
      }
      if (id == ban) {
        ban = next_ban(id);
        continue;
      }
      const std::pair<float, int32_t> c{no_counts ? logits[size_t(id)] : adjusted(id), id};
      if (cand_.size() < keep) {
        cand_.push_back(c);
        std::push_heap(cand_.begin(), cand_.end(), by_logit);
      } else if (by_logit(c, cand_.front())) {
        std::pop_heap(cand_.begin(), cand_.end(), by_logit);
        cand_.back() = c;
        std::push_heap(cand_.begin(), cand_.end(), by_logit);
      }
      if (cand_.size() >= keep) worst = cand_.front().first;
    }
  } else {
    cand_.reserve(size_t(V));
    for (int32_t id = 0, ban = next_ban(-1); id < V; ++id) {
      if (id == ban) {
        ban = next_ban(id);
        continue;
      }
      cand_.push_back({no_counts ? logits[size_t(id)] : adjusted(id), id});
    }
  }
  if (p_.temperature <= 0) {
    std::swap(cand_[0], *std::min_element(cand_.begin(), cand_.end(), by_logit));
    cand_.resize(1);
    prob_.assign(1, 1.0);
    return 1;
  }

  // top-k first: it bounds the sort that top-p needs (selection, then sorting only the k).
  const size_t k = p_.top_k > 0 ? std::min(size_t(p_.top_k), cand_.size()) : cand_.size();
  if (k < cand_.size()) std::nth_element(cand_.begin(), cand_.begin() + std::ptrdiff_t(k - 1), cand_.end(), by_logit);
  cand_.resize(k);
  std::sort(cand_.begin(), cand_.end(), by_logit);
  // Softmax at the temperature, in double.
  std::vector<double>& prob = prob_;
  prob.resize(k);
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
  return n;
}

int32_t Sampler::sample(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned) {
  const size_t n = prepare(logits, allowed, banned, {});
  if (p_.temperature <= 0) return cand_[0].second;
  double total = 0;
  for (size_t i = 0; i < n; ++i) total += prob_[i];
  double r = std::uniform_real_distribution<double>(0, total)(rng_);
  for (size_t i = 0; i < n; ++i) {
    r -= prob_[i];
    if (r <= 0) return cand_[i].second;
  }
  return cand_[n - 1].second;
}

void Sampler::distribution(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned,
                           std::span<const int32_t> extra, std::vector<std::pair<int32_t, double>>& out) {
  const size_t n = prepare(logits, allowed, banned, extra);
  double total = 0;
  for (size_t i = 0; i < n; ++i) total += prob_[i];
  out.clear();
  for (size_t i = 0; i < n; ++i) out.push_back({cand_[i].second, prob_[i] / total});
}

}  // namespace eklow::serve
