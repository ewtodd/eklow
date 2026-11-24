// Inkling 2 Mini's chat template, ported from its chat_template.jinja (Hugging Face's file, byte for byte what the GGUF's
// tokenizer.chat_template holds), and the request handling around it that llama-server c74c71a6's --jinja path does, which
// is what eklow follows (the M0 ground truth). The port renders what llama.cpp's Jinja engine renders, and so differs from
// Python's Jinja2 only where the two engines differ (DECISIONS 2026-11-24, "the text layer"); it splits the output into
// parts so that text from the request is tokenized without special tokens: a user who writes "<|end_message|>" gets those
// characters, not a block boundary (llama.cpp parses special tokens everywhere).
//
// The template writes, in order:
//   - the tools, one `<|message_system|>tool_declare<|content_xml|>[...]<|end_message|>` block (the JSON array of
//     {description, name, parameters, type} with keys sorted, compact separators, non-ASCII left as it is);
//   - each message as blocks `ROLE_TOKEN[name]<|content_KIND|>TEXT<|end_message|>`: an assistant's reasoning as a thinking
//     block, its content as a text block (one block per content part), each tool call as
//     `<|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...sorted keys...}}<|end_message|>`, and
//     `<|content_model_end_sampling|>` after an assistant message; a tool result's header names the tool, from `name` or by
//     matching tool_call_id against the calls of the assistant messages;
//   - before the first message that is not a system message, or at the end if there is none, a system block
//     `Thinking effort level: X` (0.9 by default; reasoning_effort names a level or gives a number in [0, 0.99]).
// It adds no generation prompt: the model opens its own `<|message_model|>` block, so a prompt ends after the last message.
#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/json_value.hpp"
#include "text/chat.hpp"
#include "text/inkling_mode.hpp"

namespace eklow::text {

// The template over the values it sees: `messages` as llama.cpp hands them to the engine (role, content as a string or
// a list of {type, text} parts, reasoning_content, name, tool_call_id, tool_calls with `arguments` already parsed), `tools`
// as OpenAI's {type, function: {name, description, parameters}} (null or an empty list: none), and the template variable
// `reasoning_effort` (nullptr: not defined). Throws ChatError where the template raises (the message is the template's),
// or where llama.cpp's engine fails on the input.
std::vector<ChatPart> render_inkling(const json::Value& messages, const json::Value& tools,
                                     const json::Value* reasoning_effort);

struct InklingChat {
  std::vector<ChatPart> parts;  // the prompt, without a trailing generation prompt
  // Continuing the final assistant message (the request ends with one, or says continue_final_message): the block that
  // message opened, which the prompt ends with and the output parser reads first (its token ids, tokenized as part of the
  // prompt, are the parser's prefill); empty otherwise.
  std::string generation_prompt;
  std::vector<ChatPart> generation_parts;
  InklingMode mode;  // for the output parser
};

// A chat request as llama-server receives it (an OpenAI chat completion body), rendered the way its --jinja path renders it
// for this model: messages and tools validated and normalized (null content as "", tool-call arguments strings parsed to
// objects, role "developer" as system), reasoning_effort and chat_template_kwargs.reasoning_effort, response_format,
// tool_choice, and a final assistant message continued rather than closed (prefill). Throws ChatError where llama-server
// answers with an error or the template raises. Request fields it does not know are ignored.
//
// eklow does not follow llama.cpp in two things: other chat_template_kwargs than reasoning_effort are ignored (llama.cpp
// lets a kwarg named "messages" or "tools" replace them), and request text never parses as special tokens (the parts).
InklingChat render_inkling_request(const json::Value& body);

// Whether `tmpl` is the template that was ported (byte for byte): the caller refuses a GGUF that carries anything else
// rather than render it wrong.
bool is_inkling_template(std::string_view tmpl);

// The number a JSON number becomes in llama.cpp (nlohmann json, int64 reads): an integer above INT64_MAX but within UINT64
// wraps around to a negative one, a larger integer is a double, and an infinite double is null. Applied to every request
// before use; exposed for tests.
json::Value llama_json(const json::Value& v);

// json::parse as llama.cpp's reader (nlohmann) takes text: it refuses what eklow's parser repairs (half a surrogate pair
// in a \u escape) or rounds (a float that overflows), and its numbers are llama_json's. Throws ChatError. A request body
// should be read with it too.
json::Value parse_llama_json(std::string_view text);

// `json.dumps(v, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))`, with llama.cpp's escapes: what
// the template's tojson writes.
std::string jinja_tojson(const json::Value& v, bool sort_keys);

}  // namespace eklow::text
