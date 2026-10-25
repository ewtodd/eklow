// Gemma 4's chat template, ported from Google's chat_template.jinja ("Google Gemma 4 Canonical Chat
// Template", published 2026-07-09; the GGUF's tokenizer.chat_template is the same file). The port
// renders what Hugging Face's apply_chat_template renders, byte for byte
// (tools/ref/chat_compare.py), and splits the output into parts so that text from the request is
// tokenized without special tokens: a user who writes "<|turn>" gets those characters, not a turn
// boundary.
//
// Inputs are the template's: `messages` (OpenAI-style role/content/tool_calls/tool_call_id/name,
// plus reasoning or reasoning_content, and Gemma's legacy tool_responses), `tools` (function
// declarations) and the options below. Tool-call arguments must be objects, as the template
// requires; the server parses OpenAI's argument strings before rendering.
#pragma once

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/json_value.hpp"

namespace eklow::text {

struct ChatPart {
  std::string text;
  bool input;  // from the request: tokenize without parsing special tokens
};

struct ChatOptions {
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  bool preserve_thinking = false;
  std::string bos_token = "<bos>";
};

// Where the template raises, or where Jinja2 would fail on the input (a missing key used in
// arithmetic, a string where an object is required).
struct ChatError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// `tools` may be null (no tools).
std::vector<ChatPart> render_gemma4(const json::Value& messages, const json::Value& tools, const ChatOptions& opt);
std::string join(const std::vector<ChatPart>& parts);

// Whether `tmpl` is the template this port reproduces (the engine refuses a GGUF that carries
// another one rather than render it wrong).
bool is_gemma4_template(std::string_view tmpl);

}  // namespace eklow::text
