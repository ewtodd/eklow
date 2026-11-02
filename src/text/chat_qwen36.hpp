// Qwen3.6's chat templates, ported from the two chat_template.jinja files of the Qwen3.6-35B-A3B family: stock
// Qwen/Qwen3.6-35B-A3B (revision 995ad96e) and ornith-ai/Ornith-1.5-35B-A3B (10fbf86f). They differ in one branch:
// stock drops an assistant turn's reasoning before the last user query (unless preserve_thinking is set), Ornith
// re-renders it always, exactly as generated. The port renders what Hugging Face's apply_chat_template renders, byte
// for byte (tools/ref/qwen_chat_compare.py), and splits the output into parts so that text from the request is
// tokenized without added tokens: a user who writes "<think>" or "<|im_end|>" gets those characters, not the tokens.
//
// Inputs are the template's: `messages` (OpenAI-style role/content/tool_calls plus reasoning_content), `tools`
// (OpenAI function declarations) and the options below. Tool-call arguments must be objects, as the template
// requires (`arguments|items`); the server parses OpenAI's argument strings before rendering. String arguments are
// written raw and everything else with transformers' tojson (Python's json.dumps), which is what
// output_parser_qwen.cpp reads back with the request's tool schemas.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/json_value.hpp"
#include "text/chat.hpp"  // ChatPart, ChatError, join

namespace eklow::text {

enum class QwenTemplate { Stock, Ornith };

struct QwenChatOptions {
  bool add_generation_prompt = false;
  // Off ends the prompt with "<think>\n\n</think>\n\n" instead of "<think>\n".
  bool enable_thinking = true;
  // Stock only: keep the reasoning of earlier assistant turns. Ornith's template always does.
  bool preserve_thinking = false;
  // "Picture N: " and "Video N: " before each image and video part.
  bool add_vision_id = false;
};

// `tools` may be null (no tools). Throws ChatError where the template raises, or where Jinja2 would fail on the input.
std::vector<ChatPart> render_qwen36(QwenTemplate which, const json::Value& messages, const json::Value& tools,
                                    const QwenChatOptions& opt);

// Which template a GGUF's tokenizer.chat_template is, if it is one of the two published files exactly (byte for
// byte); anything else is refused by the caller rather than rendered wrong.
std::optional<QwenTemplate> qwen36_template_of(std::string_view tmpl);
const char* qwen36_template_name(QwenTemplate which);

}  // namespace eklow::text
