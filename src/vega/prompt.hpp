// vega's questions and the prompt its reader reads, as vegaml 0.16.1 builds them (_common.py option_texts and
// build_prompt, template 0, the options in their own order), token for token:
//   [the question first: "Question to keep in mind while reading ({t}): {ins}\n\n", at most 128 tokens, only when the
//    state is shorter than query_first_max_state tokens]
//   "Observation:\n" STATE "\n\nMeasurement ({t}): {ins}\nPossible outcomes:" ("\n* {option}") x K "\nOutcome:"
// with the instructions cut to 256 tokens, each option to opt_max_tokens, the state to what max_len leaves, and {t} the
// question's type (choice, score, boolean). No BOS (Qwen3.5 has none).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "text/tokenizer_qwen.hpp"
#include "vega/reader.hpp"

namespace eklow::vega {

// vegaml's question types, numbered as its engine numbers them (QTYPES).
enum class QType : int32_t { Choice = 0, Score = 1, Boolean = 2 };
const char* qtype_name(QType t);

// A question in vegaml's vocabulary. criteria: for a choice, its options in order (an option's description may be empty:
// the option is its id alone); for a score, its levels' descriptions (the ids are ignored); for a boolean, the
// descriptions of "false" and "true" if given.
struct Question {
  QType type = QType::Boolean;
  std::string instructions;
  std::vector<std::pair<std::string, std::optional<std::string>>> criteria;
};

// The option texts in label order (a boolean's are [no, yes], so the second probability is yes).
std::vector<std::string> option_texts(const Question& q);
// The option ids the engine's probabilities are keyed by ("False"/"True" for a boolean, as vegaml's engine keys them).
std::vector<std::string> option_ids(const Question& q);

struct PromptConfig {
  int64_t max_len = 73728, opt_max_tokens = 32, ins_max_tokens = 256, query_first_max_state = 0;
};

Prompt build_prompt(const text::QwenTokenizer& tok, const std::vector<int32_t>& state_ids, const Question& q, const PromptConfig& c);

}  // namespace eklow::vega
