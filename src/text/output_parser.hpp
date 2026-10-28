// Gemma 4's generated output, parsed as it streams, token by token. A model turn is
//   [<|channel>thought\n REASONING <channel|>] [<|tool_call>call:NAME{ARGS}<tool_call|> ...] [CONTENT]
// ending at <turn|>, <eos>, or <|tool_response> (the model waiting for tool results), the inverse of
// what the chat template writes for an assistant message. ARGS is the template's format_argument
// with bare keys: key:value pairs, strings between <|"|> tokens, numbers in Python's notation,
// true/false/null, {...} and [...]. Because <|"|> is a single special token, string boundaries are
// unambiguous whatever the string holds; keys containing ':' or ',' cannot be told apart (the
// template does not escape them).
//
// Text is released only as whole UTF-8 characters, so a character split across byte-fallback
// tokens never reaches a client half-written.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "text/tokenizer.hpp"

namespace eklow::text {

struct ToolCall {
  std::string name;
  json::Value arguments;  // an object
};

struct OutputDelta {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;  // each call once, when its <tool_call|> arrives
  std::vector<std::string> errors;   // malformed tool calls, with what was wrong
  bool done = false;                 // an end token arrived
  int32_t end_token = -1;            // which one
};

class OutputParser {
 public:
  // `in_thinking`: the prompt already opened the thought channel (enable_thinking after a tool
  // response ends the prompt with "<|channel>thought\n").
  explicit OutputParser(const Tokenizer& tok, bool in_thinking = false);

  OutputDelta push(int32_t id);
  // Whatever text is still held back (an incomplete UTF-8 tail, a call never closed).
  OutputDelta finish();

  // Token ids the parser treats as the end of a turn.
  bool is_end(int32_t id) const { return id == eos_ || id == turn_end_ || id == tool_response_; }

  // The special tokens the parser cannot take next: all but its state's own markers and the turn's ends. The sampler
  // never picks them, so a structural token where it does not parse (the 26B once wrote <channel|> in the middle of
  // a word of its answer) cannot reach the output (DECISIONS 2026-10-27, "structural tokens only where they parse").
  const std::vector<int32_t>& banned() const { return banned_[size_t(state_)]; }

 private:
  enum class State { Content, ChannelName, Thinking, ToolCall, Done };
  void emit_text(std::string& pending, std::string& out, bool flush);
  void finish_call(OutputDelta& d);

  const Tokenizer& tok_;
  int32_t channel_open_, channel_close_, call_open_, call_close_, quote_, eos_, turn_end_, tool_response_, turn_open_;
  State state_ = State::Content;
  std::vector<int32_t> banned_[5];  // per State
  std::string pending_;  // bytes of the current text state not yet released
  std::string channel_name_;
  // The current tool call, as text runs and quote markers: each element is either a text run or
  // one <|"|> (marked by `quote`).
  struct Piece {
    std::string text;
    bool quote;
  };
  std::vector<Piece> call_;
};

// The tool-call body parser on its own (the part between <|tool_call> and <tool_call|>); throws
// std::runtime_error saying what is malformed.
ToolCall parse_tool_call(const std::vector<std::pair<std::string, bool>>& pieces);

}  // namespace eklow::text
