// Constrained decoding of Qwen3.6's tool calls: after a <tool_call> token only tokens that keep the call one the output
// parser accepts (text/output_parser_qwen.cpp: parse_qwen_tool_call) may be sampled, until </tool_call>. The call is
//   \n<function=NAME>\n  ( <parameter=KEY>\nVALUE\n</parameter>\n )*  </function>\n
// exactly as the chat template writes it, with
//   - NAME one of the request's tools, and each KEY one of that tool's parameters (from "properties", and "required"
//     names that have none), each at most once, with </function> only once every required one has been written;
//   - VALUE for a parameter that may be a string (a string type, a union holding it, or no type: the parser reads the
//     raw text) free text that never holds "</parameter>" except as the end of the value, which is "\n</parameter>";
//     for one that may not, a JSON value of the declared kinds (integer, number, boolean, null, object, array: nested
//     content any JSON, strings with escapes, white space between tokens inside containers) ending at the same
//     "\n</parameter>"; for one with an enum of scalars, exactly one of the members as the template writes it (raw
//     for strings, tojson for the others);
//   - then only the </tool_call> token.
// Outside a call nothing is constrained, and several calls in a row are allowed. Special tokens never appear inside a
// call except the one that closes it.
//
// Like the Gemma grammar's, this is a byte automaton: a token is allowed if its bytes run through it, the vocabulary is
// bucketed by first byte, and the allowed set of a state is cached across requests by the state (mode, what has been
// matched, the open containers) and the tools' fingerprint, so getting the set for a token costs a lookup.
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "text/tokenizer_qwen.hpp"
#include "text/tool_schema.hpp"

namespace eklow::serve {

// What the grammar needs from the vocabulary, built once per tokenizer: the non-special tokens' bytes, grouped by first byte.
struct QwenToolVocab {
  explicit QwenToolVocab(const text::QwenTokenizer& tok);
  int32_t call_open, call_close;
  std::vector<std::string> bytes;              // per token id; empty for special tokens
  std::vector<std::vector<int32_t>> by_first;  // [256] non-special ids by first byte
  // Allowed sets by state, shared by every request.
  mutable std::mutex mu;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache;
  mutable size_t cached_ints = 0;
};

class QwenToolGrammar {
 public:
  QwenToolGrammar(std::shared_ptr<const QwenToolVocab> vocab, std::shared_ptr<const text::ToolSpecs> tools);

  // Whether the next token is constrained: generation is inside a call.
  bool active() const { return st_.mode != Mode::Out; }
  // The token ids that may come next (when active); never empty inside a call.
  const std::vector<int32_t>& allowed();
  // Follows a generated token. Returns false if the token was not allowed (the state is then unchanged).
  bool accept(int32_t id);
  // The shortest text that, written from here, completes the call (not counting the </tool_call> token), for tests
  // and for cutting a call short.
  std::string completion() const;

 private:
  enum class Mode : uint8_t { Out, Prefix, Name, NameEnd, Item, Free, Enum, Json, Term, Closed };
  // JSON parser states; Start is the first byte of a top-level value, which is bare (no white space, no string).
  enum class J : uint8_t { Start, Value, ValueOrEnd, KeyOrEnd, Key, Colon, String, Escape, Hex, Neg, Zero, Int, Frac0, Frac, Exp0, ExpSign, Exp, Lit, After, Done };
  struct State {
    Mode mode = Mode::Out;
    J j = J::Start;
    uint8_t idx = 0;        // Prefix and Term: bytes of the literal matched; Hex: digits seen; Lit: letters matched
    bool key_string = false;
    uint8_t r = 0;          // Free: bytes of "</parameter>" matched
    bool prenl = false;     // Free: that match began after a newline
    bool lastnl = false;    // Free: the last byte was a newline
    int tool = -1, param = -1;
    unsigned kinds = 0;     // Json: the kinds the top-level value may be
    std::string buf;        // Name, Item, Enum: the text so far; Lit: the word
    std::string stack;      // Json: 'o' and 'a' for the open containers
    std::string used;       // Item: per parameter of the tool, '1' once written
  };
  struct Plan {
    enum class Kind { Free, Enum, Json } kind = Kind::Free;
    unsigned kinds = 0;
    std::vector<std::string> enums;  // the texts an enum allows
  };
  struct ToolPlan {
    std::string name;
    std::vector<std::string> headers;  // "<parameter=KEY>\n"
    std::vector<bool> required;
    std::vector<Plan> plans;
  };

  bool step(State& s, int c) const;
  bool step_json(State& s, int c) const;
  bool feed(State& s, int32_t id) const;
  void begin_value(State& s) const;
  bool all_required(const State& s) const;
  std::string state_key(const State& s) const;

  std::shared_ptr<const QwenToolVocab> v_;
  std::vector<ToolPlan> tools_;
  std::string fp_;  // fingerprint of the tools, for the cache
  State st_;
  std::vector<int32_t> allowed_;
  bool allowed_valid_ = false;
};

}  // namespace eklow::serve
