#include "serve/openai.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string_view>

#include "core/json.hpp"
#include "text/chat_gemma4.hpp"

namespace eklow::serve {

namespace {

ApiError bad(const std::string& message) { return ApiError(400, "invalid_request_error", "", message); }

const json::Value* field(const json::Value& o, std::string_view k) { return o.is_object() ? o.find(k) : nullptr; }

double number(const json::Value& v, const char* name) {
  if (v.is_int()) return std::stod(v.int_digits());
  if (v.is_float()) return v.as_double();
  throw bad(std::string(name) + " must be a number");
}

int base64_digit(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  return c == '+' ? 62 : c == '/' ? 63 : -1;
}

// The picture formats stb_image reads here (image/stb_image.cpp), by their signatures; WebP is told apart for a clearer
// refusal.
bool has_prefix(const std::vector<uint8_t>& b, std::string_view sig, size_t at = 0) {
  return b.size() >= at + sig.size() && std::equal(sig.begin(), sig.end(), b.begin() + std::ptrdiff_t(at), [](char s, uint8_t x) { return uint8_t(s) == x; });
}

// OpenAI's messages as the template takes them: tool-call arguments parsed from their JSON strings. A content array
// with images becomes one string with `marker` where each image goes, as llama.cpp's server hands it to the template
// (text parts joined by newlines, none around a marker); the images' bytes go to `images` in order.
json::Value template_messages(const json::Value& msgs, const std::string& marker, bool vision, std::vector<std::vector<uint8_t>>& images) {
  json::Value out = json::Value::array();
  for (const auto& m : msgs.items()) {
    if (!m.is_object()) throw bad("each message must be an object");
    json::Value c = m;
    if (const json::Value* content = field(m, "content"); content && content->is_array()) {
      bool has_image = false;
      for (const auto& part : content->items()) {
        const json::Value* type = field(part, "type");
        if (type && type->is_string() && type->as_string() == "image_url") has_image = true;
        else if (!type || !type->is_string() || type->as_string() != "text")
          // Worded as clients expect, so that they drop images rather than retry them.
          throw ApiError(400, "invalid_request_error", "unsupported_content",
                         "unsupported content[].type: only text and image_url parts are supported");
      }
      if (has_image && !vision)
        throw ApiError(400, "invalid_request_error", "unsupported_content",
                       "image_url is not supported: this server was started without a vision projector (--mmproj), only text parts");
      if (has_image) {
        std::string text;
        bool last_was_marker = false;
        for (const auto& part : content->items()) {
          const bool is_image = part.find("type")->as_string() == "image_url";
          if (is_image) {
            const json::Value* iu = field(part, "image_url");
            const json::Value* url = iu && iu->is_object() ? iu->find("url") : (iu && iu->is_string() ? iu : nullptr);
            if (!url || !url->is_string()) throw bad("an image_url part needs image_url.url");
            images.push_back(image_bytes(url->as_string()));
            text += marker;
            last_was_marker = true;
          } else {
            const json::Value* t = field(part, "text");
            if (!last_was_marker && !text.empty()) text += '\n';
            last_was_marker = false;
            if (t && t->is_string()) text += t->as_string();
          }
        }
        c.set("content", json::Value::string(text));
      }
    }
    if (const json::Value* calls = field(m, "tool_calls"); calls && calls->is_array()) {
      json::Value fixed = json::Value::array();
      for (json::Value call : calls->items()) {
        const json::Value* fn = field(call, "function");
        if (!fn || !fn->is_object()) throw bad("a tool call needs a function");
        json::Value f = *fn;
        if (const json::Value* args = field(*fn, "arguments"); args && args->is_string()) {
          json::Value parsed;
          try {
            parsed = json::parse(args->as_string().empty() ? "{}" : args->as_string());
          } catch (const std::exception& e) {
            throw bad(std::string("tool call arguments are not JSON: ") + e.what());
          }
          f.set("arguments", std::move(parsed));
        }
        call.set("function", std::move(f));
        fixed.items().push_back(std::move(call));
      }
      c.set("tool_calls", std::move(fixed));
    }
    out.items().push_back(std::move(c));
  }
  return out;
}

std::string random_hex(int n) {
  static thread_local std::mt19937_64 rng(std::random_device{}());
  static const char* hex = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < n; ++i) s += hex[rng() % 16];
  return s;
}

}  // namespace

std::vector<uint8_t> image_bytes(const std::string& url) {
  auto starts = [&](std::string_view p) { return url.compare(0, p.size(), p) == 0; };
  auto unsupported = [](const std::string& m) { return ApiError(400, "invalid_request_error", "unsupported_content", m); };
  if (starts("http://") || starts("https://"))
    throw unsupported("image_url: the server does not fetch http(s) URLs; send the image as a base64 data: URL");
  if (starts("file://")) throw unsupported("image_url: file:// URLs are not supported; send the image as a base64 data: URL");
  std::string_view b64 = url;
  if (starts("data:")) {
    const size_t comma = url.find(',');
    const std::string_view head = std::string_view(url).substr(0, comma == std::string::npos ? url.size() : comma);
    if (comma == std::string::npos || head.compare(0, 11, "data:image/") != 0) throw bad("image_url: not a data:image/...;base64 URL");
    if (head.size() < 7 || head.substr(head.size() - 7) != ";base64") throw bad("image_url: the data: URL must be base64 encoded");
    b64 = std::string_view(url).substr(comma + 1);
  }
  std::vector<uint8_t> bytes;
  bytes.reserve(b64.size() / 4 * 3);
  uint32_t acc = 0;
  int bits = 0;
  for (char ch : b64) {
    if (ch == '=' || ch == '\n' || ch == '\r' || ch == ' ') continue;
    const int d = base64_digit(ch);
    if (d < 0) throw bad("image_url: invalid base64");
    acc = (acc << 6) | uint32_t(d);
    bits += 6;
    if (bits >= 8) bits -= 8, bytes.push_back(uint8_t(acc >> bits));
  }
  if (bytes.empty()) throw bad("image_url: no image data");
  if (has_prefix(bytes, "RIFF") && has_prefix(bytes, "WEBP", 8))
    throw unsupported("image_url: WebP is not supported; send JPEG, PNG, GIF or BMP");
  if (!(has_prefix(bytes, "\xFF\xD8") || has_prefix(bytes, "\x89PNG") || has_prefix(bytes, "GIF8") || has_prefix(bytes, "BM")))
    throw unsupported("image_url: not a JPEG, PNG, GIF or BMP picture");
  return bytes;
}

std::string error_json(const ApiError& e) {
  json::Writer w(false);
  w.begin_object().key("error").begin_object();
  if (!e.code.empty()) w.kv("code", e.code);
  else w.kv("code", int64_t(e.status));
  w.kv("message", std::string(e.what())).kv("type", e.type);
  for (const auto& [k, v] : e.numbers) w.kv(k, v);
  w.end_object().end_object();
  return w.str();
}

ApiError context_exceeded(int64_t prompt_tokens, int64_t n_ctx) {
  ApiError e(400, "exceed_context_size_error", "", "the request exceeds the available context size, try increasing it");
  e.numbers = {{"n_prompt_tokens", prompt_tokens}, {"n_ctx", n_ctx}};
  return e;
}

ChatRequest parse_chat(const json::Value& body, const text::Tokenizer& tok, const ChatDefaults& d) {
  if (!body.is_object()) throw bad("the body must be a JSON object");
  ChatRequest r;
  r.sampling = d.sampling;
  if (const json::Value* m = field(body, "model"); m && m->is_string()) r.model = m->as_string();
  const json::Value* msgs = field(body, "messages");
  if (!msgs || !msgs->is_array() || msgs->items().empty()) throw bad("messages must be a non-empty array");
  if (const json::Value* n = field(body, "n"); n && !n->is_null() && number(*n, "n") != 1) throw bad("only n = 1 is supported");

  json::Value tools;
  bool tools_off = false;
  if (const json::Value* tc = field(body, "tool_choice"); tc && tc->is_string() && tc->as_string() == "none") tools_off = true;
  if (const json::Value* t = field(body, "tools"); t && t->is_array() && !t->items().empty() && !tools_off) tools = *t;
  r.tools = !tools.is_null();
  if (r.tools)
    for (const auto& t : tools.items())
      if (const json::Value* fn = field(t, "function"); fn && fn->is_object())
        if (const json::Value* n = fn->find("name"); n && n->is_string()) r.tool_names.push_back(n->as_string());

  // Gemma 4's template only turns thinking on or off. On unless the request says otherwise:
  // reasoning_effort "none", or think / enable_thinking false (the template variable wins).
  r.thinking = true;
  if (const json::Value* e = field(body, "reasoning_effort"); e && e->is_string()) r.thinking = e->as_string() != "none";
  if (const json::Value* t = field(body, "think"); t && t->is_bool()) r.thinking = t->as_bool();
  if (const json::Value* k = field(body, "chat_template_kwargs"); k && k->is_object()) {
    if (const json::Value* e = k->find("reasoning_effort"); e && e->is_string()) r.thinking = e->as_string() != "none";
    if (const json::Value* t = k->find("enable_thinking"); t && !t->is_null()) r.thinking = t->truthy();
  }

  text::ChatOptions opt;
  opt.add_generation_prompt = true;
  opt.enable_thinking = r.thinking;
  // Where the images go: a marker per request, so that no text of the conversation can be taken for one.
  const std::string marker = "<__media_" + random_hex(16) + "__>";
  std::vector<std::vector<uint8_t>> pictures;
  std::vector<text::ChatPart> parts;
  try {
    parts = text::render_gemma4(template_messages(*msgs, marker, bool(d.encode_image), pictures), tools, opt);
  } catch (const text::ChatError& e) {
    throw bad(std::string("the chat template rejects the conversation: ") + e.what());
  }
  // mtmd tokenizes the text before and after each marker on its own and puts <|image>, the rows and <image|> between.
  std::vector<text::Tokenizer::Piece> pieces;
  auto flush = [&] {
    if (pieces.empty()) return;
    for (int32_t id : tok.encode(pieces)) r.prompt.push_back(id);
    pieces.clear();
  };
  size_t placed = 0;
  for (const auto& p : parts) {
    std::string_view t = p.text;
    while (p.input && !pictures.empty()) {
      const size_t at = t.find(marker);
      if (at == std::string_view::npos) break;
      if (at > 0) pieces.push_back({t.substr(0, at), false});
      flush();
      if (placed >= pictures.size()) break;
      const int32_t open = tok.id("<|image>"), close = tok.id("<image|>");
      if (open < 0 || close < 0) throw bad("the model's vocabulary has no image tokens");
      EncodedImage img = d.encode_image(pictures[placed++]);
      r.prompt.push_back(open);
      r.images.push_back({int64_t(r.prompt.size()), img.n, img.rows});
      for (int64_t row = 0; row < img.n; ++row) r.prompt.push_back(image_row_id(img.hash, row));
      r.prompt.push_back(close);
      t.remove_prefix(at + marker.size());
    }
    if (!t.empty()) pieces.push_back({t, !p.input});
  }
  flush();
  if (placed != pictures.size()) throw bad("an image sits where the chat template cannot place it");
  {
    const std::string text = text::join(parts);
    const std::string open = "<|channel>thought\n";
    r.in_thinking = text.size() >= open.size() && text.compare(text.size() - open.size(), open.size(), open) == 0;
  }
  if (int64_t(r.prompt.size()) >= d.max_ctx) throw context_exceeded(int64_t(r.prompt.size()), d.max_ctx);

  if (const json::Value* v = field(body, "temperature"); v && !v->is_null()) r.sampling.temperature = float(number(*v, "temperature"));
  if (const json::Value* v = field(body, "top_p"); v && !v->is_null()) r.sampling.top_p = float(number(*v, "top_p"));
  if (const json::Value* v = field(body, "top_k"); v && !v->is_null()) r.sampling.top_k = int32_t(number(*v, "top_k"));
  if (const json::Value* v = field(body, "min_p"); v && !v->is_null()) r.sampling.min_p = float(number(*v, "min_p"));
  if (const json::Value* v = field(body, "presence_penalty"); v && !v->is_null()) r.sampling.presence_penalty = float(number(*v, "presence_penalty"));
  if (const json::Value* v = field(body, "frequency_penalty"); v && !v->is_null()) r.sampling.frequency_penalty = float(number(*v, "frequency_penalty"));
  if (const json::Value* v = field(body, "seed"); v && !v->is_null()) r.sampling.seed = uint64_t(number(*v, "seed"));
  else r.sampling.seed = std::random_device{}();
  for (const char* k : {"max_completion_tokens", "max_tokens"})
    if (const json::Value* v = field(body, k); v && !v->is_null() && r.max_tokens == 0) {
      r.max_tokens = int64_t(number(*v, k));
      if (r.max_tokens <= 0) throw bad(std::string(k) + " must be positive");
    }
  if (const json::Value* v = field(body, "stop"); v && !v->is_null()) {
    if (v->is_string()) r.stop.push_back(v->as_string());
    else if (v->is_array())
      for (const auto& s : v->items()) {
        if (!s.is_string()) throw bad("stop must hold strings");
        if (!s.as_string().empty()) r.stop.push_back(s.as_string());
      }
    else throw bad("stop must be a string or an array of strings");
  }
  if (const json::Value* v = field(body, "stream"); v) r.stream = v->truthy();
  if (const json::Value* so = field(body, "stream_options"); so && so->is_object())
    if (const json::Value* u = so->find("include_usage")) r.include_usage = u->truthy();
  return r;
}

std::string ChatResponse::chunk(const std::string& delta_json, const std::string& finish) const {
  json::Writer w(false);
  w.begin_object().kv("id", id_).kv("object", "chat.completion.chunk").kv("created", created_).kv("model", model_);
  w.key("choices").begin_array().begin_object().kv("index", 0).key("delta").raw(delta_json).key("finish_reason");
  if (finish.empty()) w.null();
  else w.value(finish);
  w.end_object().end_array().end_object();
  return "data: " + w.str() + "\n\n";
}

std::string ChatResponse::usage_json(const Event& e) const {
  json::Writer w(false);
  w.begin_object().kv("prompt_tokens", e.prompt_tokens).kv("completion_tokens", e.completion_tokens);
  w.kv("total_tokens", e.prompt_tokens + e.completion_tokens);
  w.key("prompt_tokens_details").begin_object().kv("cached_tokens", e.cached_tokens).end_object().end_object();
  return w.str();
}

namespace {

std::string call_json(const text::ToolCall& c, const std::string& id, int index, bool with_index) {
  json::Writer w(false);
  w.begin_object();
  if (with_index) w.kv("index", index);
  w.kv("id", id).kv("type", "function").key("function").begin_object().kv("name", c.name).kv("arguments", json::dump(c.arguments));
  w.end_object().end_object();
  return w.str();
}

// A malformed tool call reaches the client as text, so that it shows; the scheduler counts it.
std::string malformed(const std::vector<std::string>& errors) {
  std::string s;
  for (const auto& e : errors) s += "[malformed tool call: " + e + "]";
  return s;
}

}  // namespace

std::string ChatResponse::chunks(const Event& e, bool include_usage) {
  std::string out;
  auto delta = [&](const std::function<void(json::Writer&)>& fill) {
    json::Writer w(false);
    w.begin_object();
    if (!role_sent_) {
      w.kv("role", "assistant");
      role_sent_ = true;
    }
    fill(w);
    w.end_object();
    out += chunk(w.str(), "");
  };
  if (!e.delta.reasoning.empty()) delta([&](json::Writer& w) { w.kv("reasoning_content", e.delta.reasoning); });
  const std::string content = e.delta.content + malformed(e.delta.errors);
  if (!content.empty()) delta([&](json::Writer& w) { w.kv("content", content); });
  for (const auto& c : e.delta.tool_calls) {
    const int i = n_calls_++;
    delta([&](json::Writer& w) { w.key("tool_calls").begin_array().raw(call_json(c, id_ + "-call-" + std::to_string(i), i, true)).end_array(); });
  }
  if (!e.done) return out;
  if (!e.error.empty()) {
    out += "data: " + error_json(ApiError(500, "server_error", "", e.error)) + "\n\n";
  } else {
    out += chunk(role_sent_ ? "{}" : "{\"role\":\"assistant\"}", e.finish_reason);
    if (include_usage) {
      json::Writer w(false);
      w.begin_object().kv("id", id_).kv("object", "chat.completion.chunk").kv("created", created_).kv("model", model_);
      w.key("choices").begin_array().end_array().key("usage").raw(usage_json(e)).end_object();
      out += "data: " + w.str() + "\n\n";
    }
  }
  out += "data: [DONE]\n\n";
  return out;
}

void ChatResponse::add(const Event& e) {
  reasoning_ += e.delta.reasoning;
  content_ += e.delta.content + malformed(e.delta.errors);
  for (const auto& c : e.delta.tool_calls) {
    calls_.push_back(call_json(c, id_ + "-call-" + std::to_string(n_calls_), n_calls_, false));
    ++n_calls_;
  }
  if (e.done) {
    if (!e.error.empty()) throw ApiError(500, "server_error", "", e.error);
    finish_ = e.finish_reason;
    usage_ = usage_json(e);
  }
}

std::string ChatResponse::body() const {
  json::Writer w(false);
  w.begin_object().kv("id", id_).kv("object", "chat.completion").kv("created", created_).kv("model", model_);
  w.key("choices").begin_array().begin_object().kv("index", 0).key("message").begin_object().kv("role", "assistant");
  w.key("content");
  if (content_.empty() && !calls_.empty()) w.null();
  else w.value(content_);
  if (!reasoning_.empty()) w.kv("reasoning_content", reasoning_);
  if (!calls_.empty()) {
    w.key("tool_calls").begin_array();
    for (const auto& c : calls_) w.raw(c);
    w.end_array();
  }
  w.end_object().kv("finish_reason", finish_).end_object().end_array();
  w.key("usage").raw(usage_).end_object();
  return w.str();
}

std::string new_completion_id() { return "chatcmpl-" + random_hex(24); }

}  // namespace eklow::serve
