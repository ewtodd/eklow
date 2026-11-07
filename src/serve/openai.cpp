#include "serve/openai.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string_view>

#include "core/json.hpp"
#include "text/chat.hpp"
#include "text/chat_qwen36.hpp"

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

// OpenAI's messages as the template takes them: tool-call arguments parsed from their JSON strings, the `developer`
// role as `system` (the official Ornith GGUFs' template maps it so; Qwen's own would refuse it), and content arrays
// checked to hold text and image parts only: an image_url part is decoded into `images` (in conversation order; null: refused)
// and becomes the template's {"type": "image"}.
json::Value template_messages(const json::Value& msgs, std::vector<std::vector<uint8_t>>* images) {
  json::Value out = json::Value::array();
  for (const auto& m : msgs.items()) {
    if (!m.is_object()) throw bad("each message must be an object");
    json::Value c = m;
    if (const json::Value* role = field(m, "role"); role && role->is_string() && role->as_string() == "developer")
      c.set("role", json::Value::string("system"));
    if (const json::Value* content = field(m, "content"); content && content->is_array()) {
      json::Value parts = json::Value::array();
      for (const auto& part : content->items()) {
        const json::Value* type = field(part, "type");
        const std::string t = type && type->is_string() ? type->as_string() : "";
        if (t == "image_url" && images) {
          const json::Value* iu = field(part, "image_url");
          const json::Value* url = iu && iu->is_object() ? iu->find("url") : (iu && iu->is_string() ? iu : nullptr);
          if (!url || !url->is_string()) throw bad("an image_url part needs image_url.url");
          images->push_back(image_bytes(url->as_string()));
          json::Value p = json::Value::object();
          p.set("type", json::Value::string("image"));
          parts.items().push_back(std::move(p));
          continue;
        }
        if (t == "image_url" || t == "image" || t == "input_image" || t == "video" || t == "video_url" || t == "input_audio" || t == "audio_url")
          // Worded as clients expect, so that they drop images rather than retry them.
          throw ApiError(400, "invalid_request_error", "unsupported_content",
                         t + (t == "image_url" ? " is not supported: image input is off (this server was started without a vision projector, --mmproj)"
                                                : " is not supported: this model takes text and image_url parts only"));
        if (t != "text") throw ApiError(400, "invalid_request_error", "unsupported_content", "unsupported content[].type: only text and image_url parts are supported");
        parts.items().push_back(part);
      }
      c.set("content", std::move(parts));
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

ChatRequest parse_chat(const json::Value& body, const text::QwenTokenizer& tok, const ChatDefaults& d) {
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
  if (r.tools) r.tool_specs = std::make_shared<const text::ToolSpecs>(tools);

  // Thinking is on unless the request says otherwise: reasoning_effort "none", or think / enable_thinking false (the
  // template variable wins). preserve_thinking (stock's and Sage's templates) keeps the reasoning of earlier turns.
  r.thinking = true;
  bool preserve = false;
  // verification (Sage's template only; the others never look at it): the template variable as sent, else the server's default.
  std::optional<json::Value> verification;
  // The server's default (--verification on) applies only to a conversation that has system text to carry the trigger (read from
  // the messages as the template takes them, so a `developer` message counts as the system one); for any
  // other it is as if absent, never an error. A value the request sends keeps the template's own rules, raises included.
  std::vector<std::vector<uint8_t>> image_data;
  const json::Value tmsgs = template_messages(*msgs, d.images ? &image_data : nullptr);
  if (d.verification && text::qwen36_has_system_text(tmsgs)) verification = json::Value::boolean(true);
  if (const json::Value* e = field(body, "reasoning_effort"); e && e->is_string()) r.thinking = e->as_string() != "none";
  if (const json::Value* t = field(body, "think"); t && t->is_bool()) r.thinking = t->as_bool();
  if (const json::Value* t = field(body, "preserve_thinking"); t && !t->is_null()) preserve = t->truthy();
  if (const json::Value* t = field(body, "verification")) verification = *t;
  if (const json::Value* k = field(body, "chat_template_kwargs"); k && k->is_object()) {
    if (const json::Value* t = k->find("verification")) verification = *t;
    if (const json::Value* e = k->find("reasoning_effort"); e && e->is_string()) r.thinking = e->as_string() != "none";
    if (const json::Value* t = k->find("enable_thinking"); t && !t->is_null()) r.thinking = t->truthy();
    if (const json::Value* t = k->find("preserve_thinking"); t && !t->is_null()) preserve = t->truthy();
  }

  text::QwenChatOptions opt;
  opt.add_generation_prompt = true;
  opt.enable_thinking = r.thinking;
  opt.preserve_thinking = preserve;
  opt.verification = std::move(verification);
  std::vector<text::ChatPart> parts;
  try {
    parts = text::render_qwen36(d.chat_template, tmsgs, tools, opt);
  } catch (const text::ChatError& e) {
    throw bad(std::string("the chat template rejects the conversation: ") + e.what());
  }
  // Template text may hold added tokens; the request's own text never does.
  std::vector<text::QwenTokenizer::Piece> pieces;
  for (const auto& p : parts)
    if (!p.text.empty()) pieces.push_back({p.text, !p.input});
  r.prompt = tok.encode(pieces);
  if (d.images) {
    // Each <|image_pad|> the template wrote (request text cannot make the token) stands for the next image: the rows
    // replace it between <|vision_start|> and <|vision_end|>, which the template wrote around it.
    const int32_t pad = tok.id("<|image_pad|>");
    size_t pads = 0;
    for (int32_t id : r.prompt) pads += pad >= 0 && id == pad;
    if (pads != image_data.size())
      throw bad("the chat template placed " + std::to_string(pads) + " images for " + std::to_string(image_data.size()) + " image parts (images in system or tool messages are not supported)");
    std::vector<EncodedImage> enc;
    for (const auto& bytes : image_data) {
      enc.push_back((*d.images)(bytes));
      if (enc.back().n <= 0 || enc.back().nx * enc.back().ny != enc.back().n || !enc.back().rows) throw ApiError(500, "server_error", "", "the image encoder returned no rows");
      if (enc.back().n > d.max_image_rows)
        throw bad("an image takes " + std::to_string(enc.back().n) + " rows, more than a prefill chunk of " + std::to_string(d.max_image_rows));
    }
    std::vector<int32_t> expanded;
    size_t k = 0;
    for (int32_t id : r.prompt) {
      if (id != pad) {
        expanded.push_back(id);
        continue;
      }
      const EncodedImage& e = enc[k++];
      r.images.push_back({int64_t(expanded.size()), e.n, e.nx, e.ny, e.rows});
      for (int64_t row = 0; row < e.n; ++row) expanded.push_back(image_row_id(e.hash, row));
    }
    r.prompt = std::move(expanded);
  }
  {
    const std::string text = text::join(parts);
    const std::string open = "<think>\n";
    r.in_thinking = text.size() >= open.size() && text.compare(text.size() - open.size(), open.size(), open) == 0;
  }
  // Where the next request is likely to diverge, as the starts of messages (<|im_start|> is template text, so always the real
  // token): the generation prompt (the conversation before it is what the next turn shares even when it drops this turn's
  // reasoning, stock's template), the last message (an edited or retried one), and the first after the system message (the
  // system prompt and tools that a family of conversations shares).
  if (const int32_t im_start = tok.id("<|im_start|>"); im_start >= 0) {
    std::vector<int64_t> starts;
    for (size_t i = 0; i < r.prompt.size(); ++i)
      if (r.prompt[i] == im_start) starts.push_back(int64_t(i));
    const std::vector<int32_t> system = tok.encode("system", false);
    if (!starts.empty()) {
      r.snapshot_points.push_back(starts.back());
      if (starts.size() >= 2) r.snapshot_points.push_back(starts[starts.size() - 2]);
      for (int64_t s : starts)
        if (system.size() == 1 && s + 1 < int64_t(r.prompt.size()) && r.prompt[size_t(s) + 1] != system[0]) {
          r.snapshot_points.push_back(s);
          break;
        }
      // A document pasted into the last message and then a question: the next question shares the text up to the last paragraph
      // breaks (a token ending in "\n\n"; the pre-tokenizer cuts after a newline run whatever follows, so the prefix tokenizes the same).
      if (starts.size() >= 2) {
        int found = 0;
        for (int64_t i = starts.back() - 4; i > starts[starts.size() - 2] + 3 && found < 2; --i)
          if (r.prompt[size_t(i)] >= 0 && tok.bytes(r.prompt[size_t(i)]).find("\n\n") != std::string::npos) r.snapshot_points.push_back(i + 1), ++found;
      }
      std::sort(r.snapshot_points.begin(), r.snapshot_points.end());
      r.snapshot_points.erase(std::unique(r.snapshot_points.begin(), r.snapshot_points.end()), r.snapshot_points.end());
      if (r.snapshot_points.front() == 0) r.snapshot_points.erase(r.snapshot_points.begin());
      // Never inside an image (its rows are prefilled together).
      std::erase_if(r.snapshot_points, [&](int64_t p) {
        for (const ImageInput& im : r.images)
          if (p > im.start && p < im.start + im.n) return true;
        return false;
      });
    }
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
  if (const json::Value* v = field(body, "speculative"); v && !v->is_null()) {
    if (v->is_int()) r.draft_block = int(std::max<double>(0, std::min<double>(16, number(*v, "speculative"))));
    else r.draft_block = v->truthy() ? -1 : 0;
    if (r.draft_block == 1) r.draft_block = 0;
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
  w.key("prompt_tokens_details").begin_object().kv("cached_tokens", e.cached_tokens).end_object();
  // eklow's own: what speculative decoding did for the request (absent without a draft).
  if (e.verifies > 0) w.key("speculation").begin_object().kv("verifies", e.verifies).kv("drafted", e.drafted).kv("accepted", e.accepted).end_object();
  w.end_object();
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
