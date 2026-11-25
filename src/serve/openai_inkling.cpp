#include "serve/openai_inkling.hpp"

#include <random>
#include <stdexcept>

#include "text/chat.hpp"
#include "text/chat_inkling.hpp"

namespace eklow::serve::inkling {

namespace {

ApiError bad(const std::string& message) { return ApiError(400, "invalid_request_error", "", message); }

const json::Value* field(const json::Value& o, std::string_view k) { return o.is_object() ? o.find(k) : nullptr; }

double number(const json::Value& v, const char* name) {
  if (v.is_int()) return std::stod(v.int_digits());
  if (v.is_float()) return v.as_double();
  throw bad(std::string(name) + " must be a number");
}

std::vector<int32_t> encode(const text::InklingTokenizer& tok, const std::vector<text::ChatPart>& parts) {
  std::vector<text::InklingTokenizer::Piece> pieces;
  pieces.reserve(parts.size());
  for (const auto& p : parts) pieces.push_back({p.text, !p.input});
  return tok.encode(pieces);
}

}  // namespace

ChatRequest parse_chat(const json::Value& body, const text::InklingTokenizer& tok, const ChatDefaults& d) {
  if (!body.is_object()) throw bad("the body must be a JSON object");
  ChatRequest r;
  r.sampling = d.sampling;
  if (const json::Value* m = field(body, "model"); m && m->is_string()) r.model = m->as_string();
  if (const json::Value* n = field(body, "n"); n && !n->is_null() && number(*n, "n") != 1) throw bad("only n = 1 is supported");

  text::InklingChat chat;
  try {
    chat = text::render_inkling_request(body);
  } catch (const text::ChatError& e) {
    throw bad(e.what());
  }
  r.prompt = encode(tok, chat.parts);
  if (!chat.generation_parts.empty()) r.prefill = encode(tok, chat.generation_parts);
  r.mode = chat.mode;

  // llama-server's sampling fields over the server's defaults; a random seed unless one is sent.
  r.sampling.seed = std::random_device{}();
  try {
    apply_request_sampling(body, r.sampling, [&](const std::string& t) { return tok.encode(t, false); });
  } catch (const std::invalid_argument& e) {
    throw bad(e.what());
  }
  // 0 asks for no generation: the prompt goes into a slot (as llama.cpp's server takes n_predict 0). The first of the two
  // fields that is sent counts.
  for (const char* k : {"max_completion_tokens", "max_tokens"})
    if (const json::Value* v = field(body, k); v && !v->is_null()) {
      r.max_tokens = int64_t(number(*v, k));
      if (r.max_tokens < 0) throw bad(std::string(k) + " must not be negative");
      r.prefill_only = r.max_tokens == 0;
      break;
    }
  if (const json::Value* v = field(body, "stop"); v && !v->is_null()) {
    if (v->is_string()) {
      if (!v->as_string().empty()) r.stop.push_back(v->as_string());
    } else if (v->is_array()) {
      for (const auto& s : v->items()) {
        if (!s.is_string()) throw bad("stop must hold strings");
        if (!s.as_string().empty()) r.stop.push_back(s.as_string());
      }
    } else {
      throw bad("stop must be a string or an array of strings");
    }
  }
  if (const json::Value* v = field(body, "stream"); v) r.stream = v->truthy();
  if (const json::Value* so = field(body, "stream_options"); so && so->is_object())
    if (const json::Value* u = so->find("include_usage")) r.include_usage = u->truthy();
  return r;
}

}  // namespace eklow::serve::inkling
