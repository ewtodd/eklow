#include "serve/spec.hpp"

#include <algorithm>
#include <cmath>
#include <thread>

namespace eklow::serve {

double DraftBlock::q_of(int i, int32_t token) const {
  if (q.empty()) return 0;
  for (int c = 0; c < top_k; ++c)
    if (cand[size_t(i * top_k + c)] == token) return double(q[size_t(i * top_k + c)]);
  return 0;
}

DraftBlock walk(const Lattice& lat, float temperature, std::mt19937_64& rng) {
  const int K = lat.top_k, P = lat.positions();
  DraftBlock b;
  b.top_k = K;
  int prev = 0;
  std::vector<double> prob(static_cast<size_t>(K));
  for (int i = 0; i < P; ++i) {
    const float* s = &lat.score[(size_t(i) * K + size_t(prev)) * K];
    int pick = int(std::max_element(s, s + K) - s);
    if (temperature > 0) {
      // softmax at the temperature over the candidates, in double; the draw is an inverse-CDF one.
      double sum = 0;
      for (int c = 0; c < K; ++c) sum += prob[size_t(c)] = std::exp((double(s[c]) - double(s[pick])) / double(temperature));
      for (auto& x : prob) x /= sum;
      double r = std::uniform_real_distribution<double>(0, 1)(rng);
      int chosen = K - 1;
      for (int c = 0; c < K; ++c) {
        r -= prob[size_t(c)];
        if (r < 0) {
          chosen = c;
          break;
        }
      }
      pick = chosen;
      for (int c = 0; c < K; ++c) {
        b.cand.push_back(lat.cand[size_t(i) * K + size_t(c)]);
        b.q.push_back(float(prob[size_t(c)]));
      }
    }
    b.tokens.push_back(lat.cand[size_t(i) * K + size_t(pick)]);
    prev = pick;
  }
  return b;
}

VerifyResult verify(Sampler& sampler, const DraftBlock& block, const std::function<std::span<const float>(int)>& row,
                    const ConstraintFn& constraint) {
  const int n = block.size();
  // The rows' constraints in order, up to the first drafted token the row forbids (rejected there with probability 1, so the
  // states past it are never needed, and are not fed a token they refuse).
  std::vector<RowConstraint> cons;
  for (int i = 0; i <= n; ++i) {
    const std::span<const int32_t> prefix(block.tokens.data(), size_t(i));
    cons.push_back(constraint ? constraint(i, prefix) : RowConstraint{});
    if (i == n) break;
    const RowConstraint& c = cons.back();
    const int32_t d = block.tokens[size_t(i)];
    const bool refused = (c.allowed && std::find(c.allowed->begin(), c.allowed->end(), d) == c.allowed->end()) ||
                         (c.banned && std::binary_search(c.banned->begin(), c.banned->end(), d));
    if (refused) break;
  }
  const int m = int(cons.size());  // rows 0..m-1 have their constraints
  std::vector<std::vector<std::pair<int32_t, double>>> dists(static_cast<size_t>(m));
  auto compute = [&](Sampler& s, int i) {
    s.distribution(row(i), cons[size_t(i)].allowed, cons[size_t(i)].banned, std::span<const int32_t>(block.tokens.data(), size_t(i)),
                   dists[size_t(i)]);
  };
  if (m > 1 && !sampler.greedy()) {
    std::vector<Sampler> forks;
    forks.reserve(size_t(m));
    for (int i = 0; i < m; ++i) forks.push_back(sampler.fork());
    std::vector<std::thread> threads;
    for (int i = 1; i < m; ++i) threads.emplace_back([&, i] { compute(forks[size_t(i)], i); });
    compute(forks[0], 0);
    for (auto& t : threads) t.join();
  } else {
    for (int i = 0; i < m; ++i) compute(sampler, i);
  }
  std::vector<std::pair<int32_t, double>> dist;
  auto distribution_of = [&](int i) {
    if (i < m) {
      dist.swap(dists[size_t(i)]);
      return;
    }
    // past the precomputed rows (a refused token that was accepted anyway: every allowed token banned), in order
    const std::span<const int32_t> prefix(block.tokens.data(), size_t(i));
    const RowConstraint c = constraint ? constraint(i, prefix) : RowConstraint{};
    sampler.distribution(row(i), c.allowed, c.banned, prefix, dist);
  };
  auto draw = [&](const std::vector<std::pair<int32_t, double>>& d, double total) {
    double r = sampler.uniform() * total;
    for (const auto& [id, p] : d) {
      r -= p;
      if (r < 0) return id;
    }
    return d.back().first;
  };
  for (int i = 0; i < n; ++i) {
    distribution_of(i);
    const int32_t d = block.tokens[size_t(i)];
    if (sampler.greedy()) {
      if (dist[0].first == d) continue;
      return {i, dist[0].first};
    }
    double p = 0;
    for (const auto& [id, pr] : dist)
      if (id == d) p = pr;
    const double q = block.q_of(i, d);
    if (sampler.uniform() * q < p) continue;
    // First rejection: the residual max(0, p - q), over p's support (q outside it leaves nothing). q is non-zero only on
    // the position's k candidates: look them up sorted rather than scan them for each of p's (up to the vocabulary's) ids.
    std::vector<std::pair<int32_t, double>> qs;
    for (int c = 0; c < block.top_k && !block.q.empty(); ++c)
      qs.push_back({block.cand[size_t(i * block.top_k + c)], double(block.q[size_t(i * block.top_k + c)])});
    std::sort(qs.begin(), qs.end());
    auto q_at = [&](int32_t id) {
      const auto it = std::lower_bound(qs.begin(), qs.end(), std::pair<int32_t, double>{id, -1.0});
      return it != qs.end() && it->first == id ? it->second : 0.0;
    };
    std::vector<std::pair<int32_t, double>> res;
    double total = 0;
    for (const auto& [id, pr] : dist) {
      const double r = std::max(0.0, pr - q_at(id));
      if (r > 0) res.push_back({id, r}), total += r;
    }
    if (total > 0) return {i, draw(res, total)};
    return {i, draw(dist, 1.0)};
  }
  distribution_of(n);
  return {n, sampler.greedy() ? dist[0].first : draw(dist, 1.0)};
}

}  // namespace eklow::serve
