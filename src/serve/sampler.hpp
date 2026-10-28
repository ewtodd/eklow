// Picking the next token from a row of logits, on the host: penalties over the tokens generated so
// far, then temperature, top-k, min-p and top-p over the survivors, as llama.cpp's default chain
// orders them. Temperature 0 is greedy.
#pragma once

#include <cstdint>
#include <random>
#include <span>
#include <unordered_map>
#include <vector>

namespace eklow::serve {

struct SamplingParams {
  // Gemma 4's model card.
  float temperature = 1.0f;
  int32_t top_k = 64;  // 0: no limit
  float top_p = 0.95f;
  float min_p = 0.0f;
  // OpenAI's: logit -= presence * (seen) + frequency * count, over the completion's tokens.
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  uint64_t seed = 0;
};

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p);

  // `allowed`, when given, holds the only token ids that may be picked (a constraint's mask);
  // it must not be empty. `banned`, ascending, holds ids that are never picked (unless they are all
  // that `allowed` leaves).
  int32_t sample(std::span<const float> logits, const std::vector<int32_t>* allowed = nullptr,
                 const std::vector<int32_t>* banned = nullptr);
  // Counts a token of the completion for the penalties.
  void accept(int32_t id) { ++counts_[id]; }

 private:
  SamplingParams p_;
  std::mt19937_64 rng_;
  std::unordered_map<int32_t, int32_t> counts_;
  std::vector<std::pair<float, int32_t>> cand_;
};

}  // namespace eklow::serve
