// laya's System One over vega, as laya.Agent(vegaml=True) with a policy loaded answers it (laya 3a38cbe vega.py
// VegaAgent._decide, vegaml 0.16.1 Vega.decide in mode "auto"), for yes/no ("noul") questions:
//   - each question is asked of the reader with its own prompt (vegaml reads one prompt per question);
//   - a question whose id has a policy head worth using (Policy::head_for) is answered by that head over the pooled
//     state, question and last-token vectors; any other by the engine (zero-shot, thermally calibrated);
//   - the policy's calibration rescales the answer of each question it fitted a temperature for (rounded to 6 places);
//   - noul is the probability of yes, rounded to 6 places (laya's _fold).
#pragma once

#include <sycl/sycl.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/tokenizer_qwen.hpp"
#include "vega/engine.hpp"
#include "vega/policy.hpp"
#include "vega/prompt.hpp"
#include "vega/reader.hpp"

namespace eklow::vega {

// A request the endpoint refuses, with the HTTP status laya gives it (400 malformed, 413 too long, 422 a question laya
// or eklow cannot answer).
struct Refusal : std::runtime_error {
  int status;
  Refusal(int s, const std::string& m) : std::runtime_error(m), status(s) {}
};

struct CheckerAnswer {
  std::string qid;
  double noul = 0;
  double probs[2] = {0, 0};  // false, true (after the calibration)
  std::string readout;       // "engine" or "ttt"
  bool calibrated = false;
};

struct CheckerResult {
  std::vector<CheckerAnswer> answers;  // in the request's order
  int64_t tokens = 0;                  // prompt tokens read
  double read_ms = 0, total_ms = 0;
};

class Checker {
 public:
  // reader: the reader's GGUF (Qwen3.5-0.8B, BF16); vega_dir: a vega-U model directory for that reader; policy_dir: a
  // laya-policy/1 folder, or empty for the engine alone. max_tokens bounds a prompt (the reader's cache).
  Checker(sycl::queue& q, const std::string& reader, const std::string& vega_dir, const std::string& policy_dir, int64_t max_tokens,
          int64_t chunk = 1024);

  // laya's request body {"state": ..., "questions": {id: {"type": "noul", "instructions": ..., "criteria"|"labels"?}}}.
  // Throws Refusal.
  CheckerResult predict(const json::Value& body);
  CheckerResult predict(const std::string& state, const std::vector<std::pair<std::string, Question>>& questions);

  // Two requests, a short state and one of a pass's length, so that every kernel has run once and the driver has made its
  // buffers for them before anyone measures what is left of the card.
  void warm();

  const Engine& engine() const { return *engine_; }
  const Policy* policy() const { return policy_.get(); }
  Reader& reader() { return *reader_; }
  const Reader& reader() const { return *reader_; }
  const text::QwenTokenizer& tokenizer() const { return *tok_; }
  PromptConfig prompt_config() const;
  PoolSpec pool_spec() const;

 private:
  std::unique_ptr<gguf::File> file_;
  std::unique_ptr<text::QwenTokenizer> tok_;
  std::unique_ptr<Engine> engine_;
  std::unique_ptr<Policy> policy_;
  std::unique_ptr<Reader> reader_;
};

// One noul question's answer from its prompt's pooled vectors: the policy's head for the question's id when it has one
// worth using, else the engine; then the policy's calibration for that id.
CheckerAnswer answer(const Engine& engine, const Policy* policy, const std::string& qid, const Question& q, const Pooled& pooled);

// laya's question (one entry of "questions") in vegaml's vocabulary; Refusal(422) when laya would refuse it or it is not
// a noul question.
Question noul_question(const std::string& qid, const json::Value& def);

}  // namespace eklow::vega
