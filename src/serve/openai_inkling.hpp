// Inkling 2 Mini's /v1/chat/completions request (M4): the body rendered by the ported template with llama-server's request
// handling (text/chat_inkling: render_inkling_request), tokenized with request text never read as special tokens, and the
// generation settings as llama-server takes them. The response is built by the Gemma server's ChatResponse
// (serve/openai_gemma4.hpp), which writes a tool call's arguments as the model wrote them when the parser kept that text.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "serve/openai_gemma4.hpp"
#include "serve/sampler.hpp"
#include "text/inkling_mode.hpp"
#include "text/tokenizer_inkling.hpp"

namespace eklow::serve::inkling {

using gemma4::ApiError;

struct ChatDefaults {
  SamplingParams sampling;  // what a request leaves out (llama.cpp's, then the GGUF's)
};

struct ChatRequest {
  std::string model;
  std::vector<int32_t> prompt;
  std::vector<int32_t> prefill;  // the ids of the assistant block being continued, as the prompt ends with them (or none)
  text::InklingMode mode;
  SamplingParams sampling;
  int64_t max_tokens = 0;
  bool prefill_only = false;  // max_tokens (or max_completion_tokens) 0: the prompt into a slot, nothing generated
  std::vector<std::string> stop;
  bool stream = false, include_usage = false;
  // The prompt's images (M5): each image part's placeholder (<|unused_200054|>, as the template writes it) became the
  // image's rows, ids below zero naming them (gemma4::image_row_id), in the order of the parts.
  std::vector<gemma4::ImageInput> images;
  // "speculative": a block size (2 to 8), true (the server's), false or 0 (plain decode); -1 when not sent
  int draft_block = -1;
};

// `body` as text::parse_llama_json reads it. Throws ApiError (400) for what llama-server refuses or the template raises.
// With `images` (a server with Inkling's vision tower) each image part is encoded; without, an image part is a 400.
ChatRequest parse_chat(const json::Value& body, const text::InklingTokenizer& tok, const ChatDefaults& d,
                       const gemma4::ImageEncoder* images = nullptr);

}  // namespace eklow::serve::inkling
