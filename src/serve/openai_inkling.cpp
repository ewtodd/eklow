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

ChatRequest parse_chat(const json::Value& body, const text::InklingTokenizer& tok, const ChatDefaults& d, const gemma4::ImageEncoder* images) {
  if (!body.is_object()) throw bad("the body must be a JSON object");
  ChatRequest r;
  r.sampling = d.sampling;
  if (const json::Value* m = field(body, "model"); m && m->is_string()) r.model = m->as_string();
  if (const json::Value* n = field(body, "n"); n && !n->is_null() && number(*n, "n") != 1) throw bad("only n = 1 is supported");

  text::InklingChat chat;
  try {
    chat = text::render_inkling_request(body, images != nullptr);
  } catch (const text::ChatError& e) {
    throw bad(e.what());
  }
  r.prompt = encode(tok, chat.parts);
  if (!chat.generation_parts.empty()) r.prefill = encode(tok, chat.generation_parts);
  r.mode = chat.mode;

  // Images (only with an encoder: the renderer refuses image_url parts without): the parts' bytes in the order the
  // template wrote their placeholders (messages, then parts), each encoded and its placeholder expanded to its rows.
  std::vector<std::vector<uint8_t>> pictures;
  if (const json::Value* msgs = field(body, "messages"); images && msgs && msgs->is_array())
    for (const auto& m : msgs->items()) {
      const json::Value* content = m.is_object() ? field(m, "content") : nullptr;
      if (!content || !content->is_array()) continue;
      for (const auto& part : content->items()) {
        const json::Value* type = part.is_object() ? field(part, "type") : nullptr;
        if (!type || !type->is_string() || type->as_string() != "image_url") continue;
        const json::Value* iu = field(part, "image_url");
        const json::Value* url = iu && iu->is_object() ? iu->find("url") : nullptr;
        if (!url || !url->is_string()) throw bad("an image_url part needs image_url.url");
        pictures.push_back(gemma4::image_bytes(url->as_string()));
      }
    }
  if (!pictures.empty()) {
    const int32_t placeholder = tok.id(text::kInklingImagePlaceholder);
    std::vector<int32_t> out;
    size_t k = 0;
    for (const int32_t id : r.prompt) {
      if (id != placeholder) {
        out.push_back(id);
        continue;
      }
      if (k >= pictures.size()) throw bad("more image placeholders than image parts");
      const gemma4::EncodedImage e = (*images)(pictures[k++]);
      r.images.push_back({int64_t(out.size()), e.n, e.rows});
      for (int64_t row = 0; row < e.n; ++row) out.push_back(gemma4::image_row_id(e.hash, row));
    }
    if (k != pictures.size()) throw bad("image parts without their placeholders in the prompt");
    r.prompt = std::move(out);
  }

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
  if (const json::Value* v = field(body, "speculative"); v && !v->is_null()) {
    if (v->is_int()) r.draft_block = int(std::max<double>(0, std::min<double>(8, number(*v, "speculative"))));
    else r.draft_block = v->truthy() ? -1 : 0;
    if (r.draft_block == 1) r.draft_block = 0;
  }
  if (const json::Value* v = field(body, "stream"); v) r.stream = v->truthy();
  if (const json::Value* so = field(body, "stream_options"); so && so->is_object())
    if (const json::Value* u = so->find("include_usage")) r.include_usage = u->truthy();
  return r;
}

}  // namespace eklow::serve::inkling
