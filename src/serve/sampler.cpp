#include "serve/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace eklow::serve {

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed) {}

int32_t Sampler::sample(std::span<const float> logits, const std::vector<int32_t>* allowed) {
  const int32_t V = int32_t(logits.size());
  auto adjusted = [&](int32_t id) {
    float l = logits[size_t(id)];
    if (auto it = counts_.find(id); it != counts_.end()) l -= p_.presence_penalty + p_.frequency_penalty * float(it->second);
    return l;
  };
  cand_.clear();
  if (allowed) {
    if (allowed->empty()) throw std::runtime_error("sampler: no token allowed");
    for (int32_t id : *allowed) cand_.push_back({adjusted(id), id});
  } else if (counts_.empty()) {
    cand_.reserve(size_t(V));
    for (int32_t id = 0; id < V; ++id) cand_.push_back({logits[size_t(id)], id});
  } else {
    cand_.reserve(size_t(V));
    for (int32_t id = 0; id < V; ++id) cand_.push_back({adjusted(id), id});
  }
  auto by_logit = [](const auto& a, const auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); };
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

}  // namespace eklow::serve
