// Gemma 4's chat templates, ported from two chat_template.jinja files: Google's ("Google Gemma 4 Canonical Chat
// Template", published 2026-07-09, 18,683 bytes) and NVIDIA's for Nemotron-3.5-Bolt-E4B ("Nemotron updated Gemma4 chat
// template", published 2026-08-20, 20,161 bytes), which is Google's with three changes: a `conversational_mode`
// switch that adds a spoken-conversation instruction to the system message, an assistant message's content written
// before its tool calls instead of after their responses, and no empty thought channel in the generation prompt when
// thinking is off. A GGUF's tokenizer.chat_template is the same file as the repository's. The port renders what Hugging
// Face's apply_chat_template renders, byte for byte (tools/ref/gemma4_chat_compare.py), and splits the output into parts
// so that text from the request is tokenized without special tokens: a user who writes "<|turn>" gets those characters,
// not a turn boundary.
//
// Inputs are the template's: `messages` (OpenAI-style role/content/tool_calls/tool_call_id/name,
// plus reasoning or reasoning_content, and Gemma's legacy tool_responses), `tools` (function
// declarations) and the options below. Tool-call arguments must be objects, as the template
// requires; the server parses OpenAI's argument strings before rendering.
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/json_value.hpp"
#include "text/chat.hpp"

namespace eklow::text {

enum class Gemma4Template { Google, Bolt };

struct ChatOptions {
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  bool preserve_thinking = false;
  // Bolt only (Google's template has no such variable): the system message gets "You are having a spoken conversation
  // with the user. ..." after its own text, or becomes that when there is none. Jinja takes any truthy value as on.
  bool conversational_mode = false;
  std::string bos_token = "<bos>";
};

// `tools` may be null (no tools). Throws ChatError where the template raises, or where Jinja2 would fail on the input.
std::vector<ChatPart> render_gemma4(Gemma4Template which, const json::Value& messages, const json::Value& tools,
                                    const ChatOptions& opt);

// Which template `tmpl` is, if it is one of the two files exactly (byte for byte); the caller refuses a GGUF that
// carries anything else rather than render it wrong.
std::optional<Gemma4Template> gemma4_template_of(std::string_view tmpl);
const char* gemma4_template_name(Gemma4Template which);

}  // namespace eklow::text
