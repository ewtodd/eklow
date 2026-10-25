#include "serve/openai.hpp"

#include <functional>
#include <random>

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

// OpenAI's messages as the template takes them: tool-call arguments parsed from their JSON
// strings, text-only content.
json::Value template_messages(const json::Value& msgs) {
  json::Value out = json::Value::array();
  for (const auto& m : msgs.items()) {
    if (!m.is_object()) throw bad("each message must be an object");
    json::Value c = m;
    // Worded as clients expect, so that they drop images rather than retry them.
    if (const json::Value* content = field(m, "content"); content && content->is_array())
      for (const auto& part : content->items()) {
        const json::Value* type = field(part, "type");
        if (!type || !type->is_string() || type->as_string() != "text")
          throw ApiError(400, "invalid_request_error", "unsupported_content",
                         "image_url is not supported: this server does not support image input yet, only text parts");
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
  std::vector<text::ChatPart> parts;
  try {
    parts = text::render_gemma4(template_messages(*msgs), tools, opt);
  } catch (const text::ChatError& e) {
    throw bad(std::string("the chat template rejects the conversation: ") + e.what());
  }
  std::vector<text::Tokenizer::Piece> pieces;
  for (const auto& p : parts) pieces.push_back({p.text, !p.input});
  r.prompt = tok.encode(pieces);
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
