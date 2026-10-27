// The OpenAI-compatible surface of the chat model: a /v1/chat/completions request turned into a
// prompt and generation settings, and the scheduler's events turned into the response, whole or as
// server-sent-event chunks. Field names follow what llama.cpp's server answers, which is what the
// house's clients were written against (reasoning in reasoning_content, cached prompt tokens in
// usage.prompt_tokens_details).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "serve/sampler.hpp"
#include "serve/scheduler.hpp"
#include "text/tokenizer.hpp"

namespace eklow::serve {

// A request the server refuses, with the HTTP status and OpenAI's error type and code, and any
// numbers to report beside them.
struct ApiError : std::runtime_error {
  ApiError(int status, std::string type, std::string code, const std::string& message)
      : std::runtime_error(message), status(status), type(std::move(type)), code(std::move(code)) {}
  int status;
  std::string type, code;
  std::vector<std::pair<std::string, int64_t>> numbers;
};
// llama.cpp's answer to a prompt longer than the context, which clients match by its wording and
// fields (exceed_context_size_error, n_prompt_tokens, n_ctx).
ApiError context_exceeded(int64_t prompt_tokens, int64_t n_ctx);
std::string error_json(const ApiError& e);

// An image through the vision encoder: `n` rows of fp32 [n][n_embd], and a hash of its content (and the token budget
// it was encoded at) that names the image in prefix reuse (image_row_id).
struct EncodedImage {
  uint64_t hash = 0;
  int64_t n = 0;
  std::shared_ptr<const std::vector<float>> rows;
};
// The bytes of an image (JPEG, PNG, GIF or BMP) to its rows. Throws ApiError for a picture it cannot read.
using ImageEncoder = std::function<EncodedImage(std::span<const uint8_t> bytes)>;

// The bytes behind an image_url part's url: a base64 data: URL (or bare base64, as llama.cpp's server takes too) of a
// JPEG, PNG, GIF or BMP. Throws ApiError (400) for an http(s) URL (the server fetches nothing), WebP, or anything else.
std::vector<uint8_t> image_bytes(const std::string& url);

struct ChatDefaults {
  SamplingParams sampling;  // what a request leaves out
  int64_t max_ctx;          // positions per slot
  ImageEncoder encode_image = {};  // null: the server has no vision encoder and refuses image parts
};

struct ChatRequest {
  std::string model;
  std::vector<int32_t> prompt;
  std::vector<ImageInput> images;  // the prompt's image spans (their positions hold ids below zero naming the rows)
  SamplingParams sampling;
  int64_t max_tokens = 0;
  std::vector<std::string> stop;
  bool stream = false, include_usage = false, thinking = false, tools = false;
  bool in_thinking = false;  // the prompt ends inside the thought channel
  std::vector<std::string> tool_names;
};

// Renders the conversation with Gemma 4's template and tokenizes it (request text without special
// tokens). An image part becomes what llama.cpp's server and mtmd make of it: the prompt is split at the image,
// <|image>, the image's rows and <image|> go in, with no newline of their own. Throws ApiError for what the server
// cannot do or the request gets wrong.
ChatRequest parse_chat(const json::Value& body, const text::Tokenizer& tok, const ChatDefaults& d);

std::string new_completion_id();

// Builds the response as events arrive. `id` and `created` are the completion's.
class ChatResponse {
 public:
  ChatResponse(std::string id, std::string model, int64_t created) : id_(std::move(id)), model_(std::move(model)), created_(created) {}
  // Streaming: the chunks this event becomes ("data: ...\n\n" each, the last followed by
  // "data: [DONE]" when the event is done).
  std::string chunks(const Event& e, bool include_usage);
  // Whole: folds an event in; body() once the done event is in.
  void add(const Event& e);
  std::string body() const;

 private:
  std::string chunk(const std::string& delta_json, const std::string& finish) const;
  std::string usage_json(const Event& e) const;
  std::string id_, model_;
  int64_t created_;
  bool role_sent_ = false;
  int n_calls_ = 0;
  std::string reasoning_, content_, finish_;
  std::vector<std::string> calls_;  // tool calls as JSON objects
  std::string usage_;
};

}  // namespace eklow::serve
