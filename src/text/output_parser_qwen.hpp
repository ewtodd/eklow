// Qwen3.6's generated output, parsed as it streams, token by token. With thinking on the prompt ends "<think>\n" and
// the model writes
//   REASONING </think> CONTENT [<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n ... </function>\n</tool_call> ...]
// ending at <|im_end|> or <|endoftext|>: the inverse of what chat_qwen36.cpp writes for an assistant message
// (reasoning, then content, then the calls). With thinking off the prompt already holds the empty think block and
// the output starts at the content. <think>, </think>, <tool_call> and </tool_call> are single tokens; <function=,
// <parameter= and the rest are text, so a call is read from the text between the two tool-call tokens.
//
// Whitespace the template adds is stripped: the newline before </think> (and the one the prompt's "<think>\n" leaves
// at the front), the "\n\n" after it, the "\n\n" between content and the first call and the "\n" between calls, so that
// a rendered assistant message parses back to itself (the template trims reasoning and content, which have no
// whitespace at their edges to lose). Newlines, and only newlines, are stripped at the ends of reasoning and of
// content; a parameter's value loses the one newline the template puts on each side of it and nothing else.
//
// Parameter values are converted with the request's tool schemas (text/tool_schema.hpp: convert_param): the template
// writes strings raw and everything else as JSON, so a string-typed parameter is its text and the others are read as
// JSON; keys and values the schema does not describe stay text, as in vLLM's qwen3_coder parser.
//
// Text is released only as whole UTF-8 characters, so a character split across byte-level tokens never reaches a
// client half-written.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "text/output.hpp"  // ToolCall, OutputDelta
#include "text/tokenizer_qwen.hpp"
#include "text/tool_schema.hpp"

namespace eklow::text {

class QwenOutputParser {
 public:
  // `in_thinking`: the prompt ends "<think>\n", so the output starts inside the reasoning. `tools` (not owned, may be
  // null, must outlive the parser) says what the parameters' types are.
  explicit QwenOutputParser(const QwenTokenizer& tok, bool in_thinking = false, const ToolSpecs* tools = nullptr);

  OutputDelta push(int32_t id);
  // Whatever text is still held back (an incomplete UTF-8 tail, a call never closed).
  OutputDelta finish();

  // Token ids the parser treats as the end of a turn.
  bool is_end(int32_t id) const { return id == im_end_ || id == endoftext_; }

  // The special tokens the parser cannot take next: all but its state's own markers and the turn's ends. The sampler
  // never picks them, so a structural token where it does not parse cannot reach the output (DECISIONS 2026-10-27,
  // "structural tokens only where they parse"). In reasoning only </think> and the ends are taken; in content
  // <tool_call> and the ends; in a call only </tool_call> (a turn that ends inside one is an error).
  const std::vector<int32_t>& banned() const { return banned_[size_t(state_)]; }

 private:
  enum class State { Thinking, Content, ToolCall, Done };
  void emit_text(std::string& out, bool flush);
  void begin_block();
  void finish_call(OutputDelta& d);

  const QwenTokenizer& tok_;
  const ToolSpecs* tools_;
  int32_t think_close_, call_open_, call_close_, im_end_, endoftext_;
  State state_ = State::Content;
  std::vector<int32_t> banned_[4];  // per State
  std::string pending_;             // bytes of the current text state not yet released
  size_t held_newlines_ = 0;        // newlines released text may end with, withheld until more text follows
  bool block_started_ = false;      // the current reasoning or content block has had a character that is not a newline
  bool content_emitted_ = false, after_call_ = false;
  std::string call_;  // the text of the current tool call
};

// The text between <tool_call> and </tool_call>; throws std::runtime_error saying what is malformed. `tools` may be null.
ToolCall parse_qwen_tool_call(const std::string& body, const ToolSpecs* tools);

}  // namespace eklow::text
