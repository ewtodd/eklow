// llama-sampler-ref: llama.cpp's own samplers (the libllama of the build eklow's sampler follows, DECISIONS 2026-11-11) on a
// row from common.hpp: the chain common_sampler_init builds for the settings, its candidates before the draw, and N draws.
// Built outside CMake against that libllama by build-llama-ref.sh.
//
//   llama-sampler-ref V SEED "settings" DRAWS
#include <llama.h>

#include <cstdio>
#include <map>

#include "common.hpp"

using namespace samplertest;

static llama_sampler* chain_for(const Settings& s, int V, uint32_t seed, bool with_draw) {
  llama_sampler* ch = llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (!s.bias.empty()) {
    std::vector<llama_logit_bias> b;
    for (auto [id, v] : s.bias) b.push_back({id, v});
    llama_sampler_chain_add(ch, llama_sampler_init_logit_bias(V, int32_t(b.size()), b.data()));
  }
  if (s.mirostat != 0) {
    llama_sampler_chain_add(ch, llama_sampler_init_temp(s.temp));
    if (with_draw)
      llama_sampler_chain_add(ch, s.mirostat == 1 ? llama_sampler_init_mirostat(V, seed, s.tau, s.eta, 100)
                                                  : llama_sampler_init_mirostat_v2(seed, s.tau, s.eta));
    return ch;
  }
  for (char c : s.order) {
    switch (c) {
      case 'e': llama_sampler_chain_add(ch, llama_sampler_init_penalties(V, s.last_n, s.repeat, s.freq, s.present)); break;
      case 's': llama_sampler_chain_add(ch, llama_sampler_init_top_n_sigma(s.sigma)); break;
      case 'k': llama_sampler_chain_add(ch, llama_sampler_init_top_k(s.top_k)); break;
      case 'y': llama_sampler_chain_add(ch, llama_sampler_init_typical(s.typ, size_t(s.min_keep))); break;
      case 'p': llama_sampler_chain_add(ch, llama_sampler_init_top_p(s.top_p, size_t(s.min_keep))); break;
      case 'm': llama_sampler_chain_add(ch, llama_sampler_init_min_p(s.min_p, size_t(s.min_keep))); break;
      case 'x': llama_sampler_chain_add(ch, llama_sampler_init_xtc(s.xtc_p, s.xtc_t, size_t(s.min_keep), seed)); break;
      case 't': llama_sampler_chain_add(ch, llama_sampler_init_temp_ext(s.temp, s.dyn, s.dyn_exp)); break;
      default: break;  // 'd': DRY with a multiplier of 0 is a no-op
    }
  }
  if (with_draw) {
    const char last = s.order.empty() ? 0 : s.order.back();
    const bool greedy = (last == 't' && s.temp == 0.0f && s.dyn == 0.0f) || (last == 'k' && s.top_k == 1);
    llama_sampler_chain_add(ch, greedy ? llama_sampler_init_greedy() : llama_sampler_init_dist(seed));
  }
  return ch;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: llama-sampler-ref V SEED SETTINGS DRAWS\n");
    return 2;
  }
  const int V = std::atoi(argv[1]), draws = std::atoi(argv[4]);
  const uint64_t seed = std::strtoull(argv[2], nullptr, 10);
  const Settings s = parse(argv[3]);
  const std::vector<float> l = logits(V, seed);
  const std::vector<int> hist = history(l, seed, 80);
  auto fresh = [&] {
    std::vector<llama_token_data> d(static_cast<size_t>(V));
    for (int i = 0; i < V; ++i) d[size_t(i)] = {i, l[size_t(i)], 0.0f};
    return d;
  };
  {
    llama_sampler* ch = chain_for(s, V, 1, false);
    for (int t : hist) llama_sampler_accept(ch, t);
    std::vector<llama_token_data> d = fresh();
    llama_token_data_array a{d.data(), d.size(), -1, false};
    llama_sampler_apply(ch, &a);
    float mx = -INFINITY;
    for (size_t i = 0; i < a.size; ++i) mx = std::max(mx, a.data[i].logit);
    double sum = 0;
    std::map<int, float> e;
    for (size_t i = 0; i < a.size; ++i)
      if (a.data[i].logit != -INFINITY) sum += e[a.data[i].id] = expf(a.data[i].logit - mx);
    for (auto [id, v] : e) std::printf("%d %.9g\n", id, double(v) / sum);
    llama_sampler_free(ch);
  }
  std::printf("draws\n");
  std::map<int, long> h;
  for (int k = 0; k < draws; ++k) {
    llama_sampler* ch = chain_for(s, V, uint32_t(k * 2654435761u + 17), true);
    for (int t : hist) llama_sampler_accept(ch, t);
    std::vector<llama_token_data> d = fresh();
    llama_token_data_array a{d.data(), d.size(), -1, false};
    llama_sampler_apply(ch, &a);
    ++h[a.data[a.selected].id];
    llama_sampler_free(ch);
  }
  for (auto [id, n] : h) std::printf("%d %ld\n", id, n);
  return 0;
}
