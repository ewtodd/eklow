// Picking the next token from a row of logits, on the host, as llama.cpp's server does (common/sampling.cpp and
// src/llama-sampler.cpp of the build pinned in DECISIONS 2026-11-11): its default chain (penalties, dry, top_n_sigma, top_k,
// typical_p, top_p, min_p, xtc, temperature, then a draw from the softmax; or greedy when the chain ends in a temperature of
// 0 or a top_k of 1), its defaults, the GGUF's general.sampling.* over them and a request's fields over both. The filters
// see the untempered logits; the penalties count the last penalty_last_n tokens of the prompt and the completion together.
// eklow's own constraints (a tool grammar's mask, the output parser's banned special tokens) apply before the chain.
#pragma once

#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "model/gguf.hpp"

namespace eklow::serve {

// llama.cpp's sampler names (common_sampler_type).
enum class Stage : uint8_t { Penalties, Dry, TopNSigma, TopK, TypicalP, TopP, MinP, Xtc, Temperature, Infill, AdaptiveP };

struct SamplingParams {
  // llama.cpp's common_params_sampling defaults.
  float temperature = 0.8f;
  float dynatemp_range = 0.0f, dynatemp_exponent = 1.0f;
  int32_t top_k = 40;  // <= 0: no limit
  float top_p = 0.95f;
  float min_p = 0.05f;
  float typical_p = 1.0f;
  float top_n_sigma = -1.0f;
  float xtc_probability = 0.0f, xtc_threshold = 0.1f;
  int32_t penalty_last_n = 64;
  float repeat_penalty = 1.0f, frequency_penalty = 0.0f, presence_penalty = 0.0f;
  float dry_multiplier = 0.0f, dry_base = 1.75f;
  int32_t dry_allowed_length = 2, dry_penalty_last_n = 64;
  int32_t mirostat = 0;
  float mirostat_tau = 5.0f, mirostat_eta = 0.1f;
  float adaptive_target = -1.0f;
  int32_t min_keep = 0;
  std::vector<Stage> samplers = {Stage::Penalties, Stage::Dry,  Stage::TopNSigma, Stage::TopK,       Stage::TypicalP,
                                 Stage::TopP,      Stage::MinP, Stage::Xtc,       Stage::Temperature};
  std::vector<std::pair<int32_t, float>> logit_bias;  // added to a token's logit before the chain (-inf: never)
  uint64_t seed = 0;
};

// The GGUF's general.sampling.* over `p`, as llama.cpp's common_init_sampler_from_model reads them: sequence, top_k, top_p,
// min_p, xtc_probability, xtc_threshold, temp, penalty_last_n, penalty_repeat, mirostat, mirostat_tau, mirostat_eta.
void apply_gguf_sampling(const gguf::File& f, SamplingParams& p);
// A chat request's sampling fields over `p`, under llama-server's names: temperature, dynatemp_range, dynatemp_exponent, top_k,
// top_p, min_p, typical_p, top_n_sigma, xtc_probability, xtc_threshold, repeat_last_n, repeat_penalty, frequency_penalty,
// presence_penalty, dry_*, mirostat, mirostat_tau, mirostat_eta, adaptive_*, min_keep, samplers (names, or llama.cpp's
// one-letter string), logit_bias ([[id or text, bias or false], ...] or {"id": bias}), seed. `tokenize` turns a logit_bias
// text into ids. Throws std::invalid_argument for a malformed value or what eklow does not implement (DRY with a
// multiplier, adaptive_p, infill).
void apply_request_sampling(const json::Value& body, SamplingParams& p,
                            const std::function<std::vector<int32_t>(const std::string&)>& tokenize);

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p);
  // `allowed`, when given, holds the only token ids that may be picked (a constraint's mask); it must not be empty.
  // `banned`, ascending, holds ids that are never picked (unless they are all that `allowed` leaves).
  int32_t sample(std::span<const float> logits, const std::vector<int32_t>* allowed = nullptr,
                 const std::vector<int32_t>* banned = nullptr);
  // A token of the prompt or the completion, for the penalties' window (ids below zero, an image's rows, are skipped, as
  // llama-server skips LLAMA_TOKEN_NULL).
  void accept(int32_t id);
  void accept(std::span<const int32_t> ids) {
    for (int32_t id : ids) accept(id);
  }
  // The distribution sample() draws from, as (id, probability) pairs summing to 1 (greedy: the one token), with `extra`
  // in the penalties' window after the accepted tokens (speculative decoding's rows, src/serve/spec.hpp). XTC's random
  // removal enters as the mixture it is.
  void distribution(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned,
                    std::span<const int32_t> extra, std::vector<std::pair<int32_t, double>>& out);
  // A uniform draw in [0, 1) from the sampler's generator, for the acceptance tests of speculative decoding.
  double uniform() { return std::uniform_real_distribution<double>(0, 1)(rng_); }
  bool greedy() const { return greedy_; }
  // Whether distribution() describes the next pick: not under mirostat, whose state moves with each pick (such a request
  // decodes plain).
  bool speculative() const { return p_.mirostat == 0; }

  struct Cand {
    int32_t id;
    float logit, p;
  };

 private:
  // The candidates after the chain, the draw left to do. xtc: 0 skip its removal, 1 apply it, 2 draw whether to.
  void run(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned,
           std::span<const int32_t> extra, int xtc);
  void final_probs(std::vector<std::pair<int32_t, double>>& out) const;

  SamplingParams p_;
  bool greedy_ = false;
  std::mt19937_64 rng_;
  std::vector<int32_t> history_;  // the last tokens accepted, at most the penalties' window
  float mu_ = 0;                  // mirostat's
  std::vector<Cand> cur_;
  bool sorted_ = false;
};

}  // namespace eklow::serve
