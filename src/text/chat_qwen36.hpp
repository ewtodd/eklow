// Qwen3.6's chat templates, ported from the three chat_template.jinja files of the Qwen3.6-35B-A3B family: stock
// Qwen/Qwen3.6-35B-A3B (revision 995ad96e), ornith-ai/Ornith-1.5-35B-A3B (10fbf86f) and InternScience/Sage-A1.5
// (0e3205cf). Stock drops an assistant turn's reasoning before the last user query (unless preserve_thinking is set),
// Ornith re-renders it always, exactly as generated. Sage is stock plus a `verification` switch: when it is on, the
// system content (which must then exist and be non-blank) ends with "\n\nVerification protocol: enabled". The port renders what Hugging Face's apply_chat_template renders, byte
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

enum class QwenTemplate { Stock, Ornith, Sage };

struct QwenChatOptions {
  bool add_generation_prompt = false;
  // Off ends the prompt with "<think>\n\n</think>\n\n" instead of "<think>\n".
  bool enable_thinking = true;
  // Stock only: keep the reasoning of earlier assistant turns. Ornith's template always does.
  bool preserve_thinking = false;
  // "Picture N: " and "Video N: " before each image and video part.
  bool add_vision_id = false;
  // Sage only: the template variable `verification` as the request gave it; nullopt is Undefined (not sent). Jinja decides
  // what it means: truthy turns the protocol on, a string other than true/1/yes/on (any case) then raises, null is off.
  std::optional<json::Value> verification;
};

// `tools` may be null (no tools). Throws ChatError where the template raises, or where Jinja2 would fail on the input.
std::vector<ChatPart> render_qwen36(QwenTemplate which, const json::Value& messages, const json::Value& tools,
                                    const QwenChatOptions& opt);

// Which template a GGUF's tokenizer.chat_template is, if it is one of the three published files exactly (byte for
// byte); anything else is refused by the caller rather than rendered wrong.
std::optional<QwenTemplate> qwen36_template_of(std::string_view tmpl);
// Whether the first message is a system message whose content, rendered and trimmed, is not empty (false where the
// template would raise on the content). The server applies --verification on only to such conversations.
bool qwen36_has_system_text(const json::Value& messages);
const char* qwen36_template_name(QwenTemplate which);

}  // namespace eklow::text
