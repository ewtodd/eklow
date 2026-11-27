// eklow-sampler-test: eklow's sampler (src/serve/sampler.*) on the row llama-sampler-ref sees (common.hpp), in its output
// format, so tools/sampler_test/compare.py can set the two side by side: the candidates after the chain with their
// probabilities (deterministic chains: equal id for id), and DRAWS draws (any chain: a chi-square).
//
//   eklow-sampler-test V SEED "settings" DRAWS
//   eklow-sampler-test --bench LOGITS.f32 V "settings" [ROWS]
//
// --bench: the time of sample() and distribution() a row over real logits (fp32 rows of V, as eklow-engine-run-inkling's
// logits.f32), and the candidates the chain keeps, mean over ROWS rows (64).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>

#include "common.hpp"
#include "serve/sampler.hpp"

using namespace eklow::serve;

namespace {
SamplingParams params(const samplertest::Settings& s) {
  SamplingParams p;
  p.temperature = s.temp, p.top_k = s.top_k, p.top_p = s.top_p, p.min_p = s.min_p, p.typical_p = s.typ, p.top_n_sigma = s.sigma;
  p.xtc_probability = s.xtc_p, p.xtc_threshold = s.xtc_t, p.dynatemp_range = s.dyn, p.dynatemp_exponent = s.dyn_exp;
  p.repeat_penalty = s.repeat, p.frequency_penalty = s.freq, p.presence_penalty = s.present, p.penalty_last_n = s.last_n;
  p.mirostat = s.mirostat, p.mirostat_tau = s.tau, p.mirostat_eta = s.eta, p.min_keep = s.min_keep;
  p.samplers.clear();
  const std::map<char, Stage> chars = {{'e', Stage::Penalties}, {'d', Stage::Dry}, {'s', Stage::TopNSigma}, {'k', Stage::TopK}, {'y', Stage::TypicalP},
                                       {'p', Stage::TopP}, {'m', Stage::MinP}, {'x', Stage::Xtc}, {'t', Stage::Temperature}};
  for (char c : s.order)
    if (auto it = chars.find(c); it != chars.end()) p.samplers.push_back(it->second);
  for (auto [id, b] : s.bias) p.logit_bias.push_back({id, b});
  return p;
}

int bench(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr, "usage: eklow-sampler-test --bench LOGITS.f32 V SETTINGS [ROWS]\n");
    return 2;
  }
  const size_t V = size_t(std::atoll(argv[3]));
  const int rows = argc > 5 ? std::atoi(argv[5]) : 64;
  std::ifstream f(argv[2], std::ios::binary);
  std::vector<float> l(V * size_t(rows));
  f.read(reinterpret_cast<char*>(l.data()), std::streamsize(l.size() * 4));
  const int got = int(size_t(f.gcount()) / 4 / V);
  if (got < 1) {
    std::fprintf(stderr, "eklow-sampler-test: %s holds no row of %zu\n", argv[2], V);
    return 1;
  }
  const SamplingParams p = params(samplertest::parse(argv[4]));
  Sampler smp(p);
  std::vector<std::pair<int32_t, double>> d;
  double ts = 0, td = 0, kept = 0;
  for (int r = 0; r < got; ++r) {
    const std::span<const float> row(l.data() + size_t(r) * V, V);
    auto t0 = std::chrono::steady_clock::now();
    (void)smp.sample(row);
    auto t1 = std::chrono::steady_clock::now();
    smp.distribution(row, nullptr, nullptr, {}, d);
    auto t2 = std::chrono::steady_clock::now();
    ts += std::chrono::duration<double>(t1 - t0).count(), td += std::chrono::duration<double>(t2 - t1).count(), kept += double(d.size());
  }
  std::printf("%d rows of %zu, \"%s\": sample %.3f ms, distribution %.3f ms a row; %.1f candidates kept\n", got, V, argv[4], ts * 1e3 / got,
              td * 1e3 / got, kept / got);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--bench") return bench(argc, argv);
  if (argc < 5) {
    std::fprintf(stderr, "usage: eklow-sampler-test V SEED SETTINGS DRAWS\n");
    return 2;
  }
  const int V = std::atoi(argv[1]), draws = std::atoi(argv[4]);
  const uint64_t seed = std::strtoull(argv[2], nullptr, 10);
  const samplertest::Settings s = samplertest::parse(argv[3]);
  const std::vector<float> l = samplertest::logits(V, seed);
  const std::vector<int> hist = samplertest::history(l, seed, 80);
  SamplingParams p = params(s);
  std::vector<int32_t> h(hist.begin(), hist.end());
  if (s.mirostat == 0) {
    Sampler smp(p);
    smp.accept(std::span<const int32_t>(h));
    std::vector<std::pair<int32_t, double>> d;
    smp.distribution(l, nullptr, nullptr, {}, d);
    std::map<int32_t, double> m(d.begin(), d.end());
    for (auto [id, pr] : m) std::printf("%d %.9g\n", id, pr);
  }
  std::printf("draws\n");
  std::map<int32_t, long> counts;
  for (int k = 0; k < draws; ++k) {
    p.seed = uint64_t(k) * 2654435761u + 17;
    Sampler smp(p);
    smp.accept(std::span<const int32_t>(h));
    ++counts[smp.sample(l)];
  }
  for (auto [id, n] : counts) std::printf("%d %ld\n", id, n);
  return 0;
}
