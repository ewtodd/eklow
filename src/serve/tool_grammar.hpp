// Constrained decoding of Gemma 4's tool calls: inside <|tool_call>...<tool_call|> only tokens that
// keep the call one the output parser accepts (text/output_parser.cpp: parse_tool_call) may be
// sampled, so no call comes out malformed. The call is "call:", then the name of one of the
// request's tools, then the arguments: an object of keys (bare, or between <|"|> tokens) and
// values (strings between <|"|> tokens, objects, arrays, true, false, null, numbers in Python's
// notation), whitespace allowed where the parser skips it; then <tool_call|> and nothing else.
// Outside a call nothing is constrained.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "text/tokenizer.hpp"

namespace eklow::serve {

// What the grammar needs from the vocabulary, built once per tokenizer: the non-special tokens'
// bytes, grouped by first byte.
struct ToolVocab {
  explicit ToolVocab(const text::Tokenizer& tok);
  int32_t call_open, call_close, quote;
  std::vector<std::string> bytes;                 // per token id; empty for special tokens
  std::vector<std::vector<int32_t>> by_first;     // [256] non-special ids by first byte
  std::vector<int32_t> string_ok;                 // every non-special id, and <|"|>
  // Allowed sets by state, shared by every request: a set depends only on the state's mode, its
  // open containers and, in a name or word, the text so far (and the tool names).
  mutable std::mutex mu;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache;
};

class ToolGrammar {
 public:
  ToolGrammar(std::shared_ptr<const ToolVocab> vocab, std::vector<std::string> tool_names);

  // Whether the next token is constrained: generation is inside a call.
  bool active() const { return st_.mode != Mode::Out; }
  // The token ids that may come next (when active).
  const std::vector<int32_t>& allowed();
  // Follows a generated token. Returns false if the token was not allowed (the state is then
  // unchanged).
  bool accept(int32_t id);

 private:
  enum class Mode { Out, Prefix, Name, KeyOrEnd, KeyStart, Key, QuotedKey, Colon, Value, ValueOrEnd, String, Word, After, Done };
  struct State {
    Mode mode = Mode::Out;
    int prefix = 0;           // characters of "call:" seen
    std::string buf;          // the name or the current word
    std::string stack;        // 'o' object, 'a' array
  };
  // One symbol: a byte, or -1 for the <|"|> token.
  static bool step(State& s, int c, const std::vector<std::string>& names);
  bool feed(State& s, int32_t id) const;

  std::shared_ptr<const ToolVocab> v_;
  std::vector<std::string> names_;
  State st_;
  std::string names_key_;
  std::vector<int32_t> allowed_;
  bool allowed_valid_ = false;
};

}  // namespace eklow::serve
