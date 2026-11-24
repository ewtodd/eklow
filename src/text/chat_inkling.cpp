#include "text/chat_inkling.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>

#include "text/pyfmt.hpp"

namespace eklow::text {

namespace {

using json::Value;
// A value as the template sees it: nullptr is Jinja's Undefined (a missing key), which differs from None (JSON null).
using V = const Value*;

V sub(V obj, std::string_view key) { return obj && obj->is_object() ? obj->find(key) : nullptr; }
bool truthy(V v) { return v && v->truthy(); }
bool is_string(V v) { return v && v->is_string(); }
bool eq(V v, std::string_view s) { return v && v->is_string() && v->as_string() == s; }
// {{ x }} of a string-like value: str(x), nothing for Undefined.
std::string print(V v) { return v ? py::str(*v) : std::string(); }

class Out {
 public:
  void lit(std::string_view s) { add(s, false); }
  void in(std::string_view s) { add(s, true); }
  void add(std::string_view s, bool input) {
    if (s.empty()) return;
    if (!parts.empty() && parts.back().input == input) parts.back().text += s;
    else parts.push_back({std::string(s), input});
  }
  void append(const Out& o) {
    for (const auto& p : o.parts) add(p.text, p.input);
  }
  std::vector<ChatPart> parts;
};

// ---- JSON as llama.cpp reads and writes it ------------------------------------------------------------------------

bool fits_int64(const std::string& d) {
  errno = 0;
  std::strtoll(d.c_str(), nullptr, 10);
  return errno == 0;
}

// \b \f \n \r \t, the other controls as \u00xx (lower case), nothing else escaped: json.dumps(ensure_ascii=False).
void quote(std::string& o, const std::string& s) {
  o += '"';
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned char>(c));
          o += buf;
        } else {
          o += c;
        }
    }
  }
  o += '"';
}

void dump(std::string& o, const Value& v, bool sort_keys) {
  switch (v.kind()) {
    case Value::Kind::Null: o += "null"; break;
    case Value::Kind::Bool: o += v.as_bool() ? "true" : "false"; break;
    case Value::Kind::Int: o += v.int_digits(); break;
    case Value::Kind::Float: {
      const double d = v.as_double();
      o += std::isnan(d) ? "NaN" : std::isinf(d) ? (d < 0 ? "-Infinity" : "Infinity") : py::float_repr(d);
      break;
    }
    case Value::Kind::String: quote(o, v.as_string()); break;
    case Value::Kind::Array: {
      o += '[';
      bool first = true;
      for (const auto& x : v.items()) {
        if (!first) o += ',';
        first = false;
        dump(o, x, sort_keys);
      }
      o += ']';
      break;
    }
    case Value::Kind::Object: {
      std::vector<const Value::Member*> ms;
      for (const auto& m : v.members()) ms.push_back(&m);
      // Python orders by code point, which is the byte order of UTF-8.
      if (sort_keys) std::stable_sort(ms.begin(), ms.end(), [](auto* a, auto* b) { return a->first < b->first; });
      o += '{';
      bool first = true;
      for (const auto* m : ms) {
        if (!first) o += ',';
        first = false;
        quote(o, m->first);
        o += ':';
        dump(o, m->second, sort_keys);
      }
      o += '}';
      break;
    }
  }
}

// ---- the template ---------------------------------------------------------------------------------------------------

// Jinja's `trim` in llama.cpp's engine: the bytes C's isspace takes in the "C" locale.
std::string trim(const std::string& s) {
  auto sp = [](unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r'); };
  size_t b = 0, e = s.size();
  while (b < e && sp(s[b])) ++b;
  while (e > b && sp(s[e - 1])) --e;
  return s.substr(b, e - b);
}

// std::to_string(double) with the trailing zeros taken off but one: llama.cpp's engine prints a float with it, where
// Python prints repr (0.123456789 is 0.123457, 1e-07 is 0.0).
std::string float_text(double d) {
  std::string out = std::to_string(d);
  out.erase(out.find_last_not_of('0') + 1);
  if (out.back() == '.') out.push_back('0');
  return out;
}

void emit_thinking_effort(Out& o, V reasoning_effort) {
  static const std::pair<const char*, double> kMap[] = {{"none", 0.0},  {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7},
                                                         {"high", 0.9}, {"xhigh", 0.99},   {"max", 0.99}};
  double num;
  if (!reasoning_effort || reasoning_effort->is_null()) {
    num = 0.9;
  } else if (reasoning_effort->is_string()) {
    const std::string key = trim(reasoning_effort->as_string());
    const auto* hit = std::find_if(std::begin(kMap), std::end(kMap), [&](const auto& e) { return key == e.first; });
    if (hit == std::end(kMap)) throw ChatError("Unknown reasoning_effort: " + reasoning_effort->as_string());
    num = hit->second;
  } else if (reasoning_effort->is_int() || reasoning_effort->is_float()) {
    num = reasoning_effort->as_double();
  } else if (reasoning_effort->is_bool()) {
    num = reasoning_effort->as_bool() ? 1.0 : 0.0;
  } else {
    // `| float` has no meaning for a list or an object in llama.cpp's engine (Python's Jinja2 gives 0.0).
    throw ChatError(std::string("Unknown (built-in) filter 'float' for type ") + (reasoning_effort->is_array() ? "Array" : "Object"));
  }
  if (num < 0.0 || num > 0.99) throw ChatError("reasoning_effort must be in [0.0, 0.99]");
  o.lit("<|message_system|><|content_text|>Thinking effort level: ");
  o.lit(num == 0.0 ? std::string("0") : float_text(num));
  o.lit("<|end_message|>");
}

constexpr std::pair<const char*, const char*> kRoleToken[] = {{"user", "<|message_user|>"},
                                                              {"assistant", "<|message_model|>"},
                                                              {"system", "<|message_system|>"},
                                                              {"tool", "<|message_tool|>"}};

const char* role_token(V role) {
  if (!is_string(role)) return nullptr;
  for (const auto& [name, tok] : kRoleToken)
    if (role->as_string() == name) return tok;
  return nullptr;
}

std::vector<const Value*> iterate(V v) {
  std::vector<const Value*> out;
  if (!v) return out;
  if (v->is_array()) {
    for (const auto& x : v->items()) out.push_back(&x);
  } else if (v->is_object()) {
    throw ChatError("chat_template: iterating an object");
  } else if (v->is_string()) {
    throw ChatError("chat_template: iterating a string");
  } else {
    throw ChatError("chat_template: cannot iterate over " + py::str(*v));
  }
  return out;
}

}  // namespace

bool is_inkling_template(std::string_view tmpl) {
  // FNV-1a over thinkingmachines/Inkling-2-Mini's chat_template.jinja (6,868 bytes; its GGUFs carry the same text).
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : tmpl) h = (h ^ c) * 0x100000001b3ull;
  return tmpl.size() == 6868 && h == 0x73916286448643c8ull;
}

std::string jinja_tojson(const Value& v, bool sort_keys) {
  std::string o;
  dump(o, v, sort_keys);
  return o;
}

Value parse_llama_json(std::string_view text) {
  // nlohmann refuses what eklow's parser repairs: a \u escape of half a surrogate pair.
  bool in_string = false;
  for (size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (!in_string) {
      in_string = c == '"';
      continue;
    }
    if (c == '"') {
      in_string = false;
    } else if (c == '\\' && i + 1 < text.size()) {
      if (text[i + 1] != 'u') {
        ++i;
        continue;
      }
      auto hex4 = [&](size_t at) -> int64_t {
        if (at + 6 > text.size() || text[at] != '\\' || text[at + 1] != 'u') return -1;
        int64_t v = 0;
        for (size_t k = 2; k < 6; ++k) {
          const char h = text[at + k];
          v = v * 16 + (h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -100000);
        }
        return v;
      };
      const int64_t cp = hex4(i);
      if (cp >= 0xD800 && cp <= 0xDBFF) {
        const int64_t lo = hex4(i + 6);
        if (lo < 0xDC00 || lo > 0xDFFF) throw ChatError("[json.exception.parse_error.101] surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF");
        i += 11;
      } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
        throw ChatError("[json.exception.parse_error.101] surrogate U+DC00..U+DFFF must be preceded by U+D800..U+DBFF");
      } else {
        i += 5;
      }
    }
  }
  try {
    return llama_json(json::parse(text));
  } catch (const std::runtime_error& e) {
    if (dynamic_cast<const ChatError*>(&e)) throw;
    throw ChatError(std::string("[json.exception.parse_error.101] ") + e.what());
  }
}

Value llama_json(const Value& v) {
  switch (v.kind()) {
    case Value::Kind::Int: {
      const std::string& d = v.int_digits();
      if (fits_int64(d)) return v;
      if (d[0] != '-') {
        errno = 0;
        const unsigned long long u = std::strtoull(d.c_str(), nullptr, 10);
        if (errno == 0) return Value::integer(int64_t(u));  // read back as int64: wraps
      }
      return Value::number(std::strtod(d.c_str(), nullptr));
    }
    case Value::Kind::Float:
      // A float that overflowed (1e999) is a parse error in nlohmann.
      if (!std::isfinite(v.as_double())) throw ChatError("[json.exception.out_of_range.406] number overflow parsing a float");
      return v;
    case Value::Kind::Array: {
      std::vector<Value> items;
      for (const auto& x : v.items()) items.push_back(llama_json(x));
      return Value::array(std::move(items));
    }
    case Value::Kind::Object: {
      Value o = Value::object();
      for (const auto& [k, x] : v.members()) o.set(k, llama_json(x));
      return o;
    }
    default: return v;
  }
}

std::vector<ChatPart> render_inkling(const Value& messages, const Value& tools, const Value* reasoning_effort) {
  Out o;
  if (tools.truthy()) {
    // fn = tool.function if tool.function is defined else tool; the spec's keys are written sorted.
    std::vector<Value> specs;
    for (V tool : iterate(&tools)) {
      V fn = sub(tool, "function") ? sub(tool, "function") : tool;
      Value spec = Value::object();
      V d = sub(fn, "description");
      spec.set("description", truthy(d) ? *d : Value::string(""));
      V n = sub(fn, "name");
      spec.set("name", n ? *n : Value::null());
      V p = sub(fn, "parameters");
      spec.set("parameters", truthy(p) ? *p : Value::object());
      V t = sub(tool, "type");
      spec.set("type", truthy(t) ? *t : Value::string("function"));
      specs.push_back(std::move(spec));
    }
    o.lit("<|message_system|>tool_declare<|content_xml|>");
    o.in(jinja_tojson(Value::array(std::move(specs)), true));
    o.lit("<|end_message|>");
  }

  bool effort_emitted = false;
  for (V message : iterate(&messages)) {
    V role = sub(message, "role");
    const char* rtok = role_token(role);
    if (!rtok) throw ChatError("Unknown message role: " + print(role));
    const bool assistant = eq(role, "assistant");
    if (!effort_emitted && !eq(role, "system")) {
      emit_thinking_effort(o, reasoning_effort);
      effort_emitted = true;
    }

    // The header: the role's token and, for a tool result, the tool; for an assistant, its name.
    Out hdr;
    hdr.lit(rtok);
    if (eq(role, "tool")) {
      if (truthy(sub(message, "name"))) {
        hdr.in(print(sub(message, "name")));
      } else if (truthy(sub(message, "tool_call_id"))) {
        V id = sub(message, "tool_call_id");
        // Every call of every assistant message with this id adds its name, in order (the loop does not stop at the first).
        for (V prev : iterate(&messages)) {
          if (!eq(sub(prev, "role"), "assistant") || !truthy(sub(prev, "tool_calls"))) continue;
          for (V tc : iterate(sub(prev, "tool_calls"))) {
            V tid = sub(tc, "id");
            V fname = sub(sub(tc, "function"), "name");
            if (tid && *tid == *id && fname) hdr.in(print(fname));
          }
        }
      }
    } else if (assistant && truthy(sub(message, "name"))) {
      hdr.in(print(sub(message, "name")));
    }

    Out r;  // `rendered`
    V reasoning = sub(message, "reasoning_content");
    V content = sub(message, "content");
    V calls = sub(message, "tool_calls");
    if (assistant && truthy(reasoning)) {
      r.append(hdr);
      r.lit("<|content_thinking|>");
      r.in(print(reasoning));
      r.lit("<|end_message|>");
    }
    if (is_string(content) && (truthy(content) || !(assistant && (truthy(calls) || truthy(reasoning))))) {
      r.append(hdr);
      r.lit("<|content_text|>");
      r.in(content->as_string());
      r.lit("<|end_message|>");
    } else if (truthy(content)) {
      for (V part : iterate(content)) {
        if (is_string(part)) {
          r.append(hdr);
          r.lit("<|content_text|>");
          r.in(part->as_string());
          r.lit("<|end_message|>");
          continue;
        }
        V type = sub(part, "type");
        if (!type || eq(type, "text") || eq(type, "input_text")) {
          V text = sub(part, "text");
          r.append(hdr);
          r.lit("<|content_text|>");
          r.in(is_string(text) ? text->as_string() : std::string());
          r.lit("<|end_message|>");
        } else if (eq(type, "image") || eq(type, "input_image") || eq(type, "image_url")) {
          if (!(eq(role, "user") || eq(role, "tool"))) throw ChatError("Image content is only supported for user and tool messages");
          r.append(hdr);
          r.lit("<|content_image|><|unused_200054|><|end_message|>");
        } else if (eq(type, "audio") || eq(type, "input_audio") || eq(type, "audio_url")) {
          throw ChatError("Inkling 2 does not accept audio input");
        } else {
          throw ChatError("Unsupported content part type: " + print(type));
        }
      }
    }

    if (assistant && truthy(calls)) {
      for (V tc : iterate(calls)) {
        V fn = sub(tc, "function");
        V name = sub(fn, "name");
        if (!is_string(name)) throw ChatError("tool call function name must be a string");
        V mname = sub(message, "name");
        if (truthy(mname) && !(*mname == *name))
          throw ChatError("assistant name " + print(mname) + " does not match its tool call " + name->as_string());
        V a = sub(fn, "arguments");
        const Value empty = Value::object();
        V args = truthy(a) ? a : &empty;
        if (args->is_string())
          throw ChatError("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream");
        if (!args->is_object()) throw ChatError("tool call arguments must be an object");
        r.lit("<|message_model|>");
        r.in(name->as_string());
        r.lit("<|content_invoke_tool_json|>");
        r.lit("{\"name\":");
        r.in(jinja_tojson(*name, true));
        r.lit(",\"args\":{");
        // dictsort(case_sensitive=true): by key, code point order.
        std::vector<const Value::Member*> ms;
        for (const auto& m : args->members()) ms.push_back(&m);
        std::sort(ms.begin(), ms.end(), [](auto* x, auto* y) { return x->first < y->first; });
        std::string body;
        for (size_t i = 0; i < ms.size(); ++i) {
          dump(body, Value::string(ms[i]->first), false);
          body += ':';
          dump(body, ms[i]->second, false);
          if (i + 1 < ms.size()) body += ',';
        }
        r.in(body);
        r.lit("}}<|end_message|>");
      }
    }
    if (assistant) r.lit("<|content_model_end_sampling|>");
    o.append(r);
  }
  if (!effort_emitted) emit_thinking_effort(o, reasoning_effort);
  return std::move(o.parts);
}

// ---- llama-server's request path --------------------------------------------------------------------------------------

namespace {

// A message as llama.cpp's common_chat_msg holds it after reading the request.
struct Msg {
  std::string role, content, reasoning_content, name, tool_call_id;
  std::vector<std::pair<std::string, std::string>> parts;  // (type, text) of a content array
  bool has_parts = false;
  struct Call {
    std::string name, arguments, id;
  };
  std::vector<Call> calls;

  // common_chat_msg::empty(): what a message continued from nothing would be.
  bool empty() const {
    return content.empty() && parts.empty() && calls.empty() && reasoning_content.empty() && name.empty() && tool_call_id.empty();
  }
  // render_content("\n\n"): the content, or the text parts joined.
  std::string render_content() const {
    if (!content.empty()) return content;
    std::string text;
    for (const auto& [type, t] : parts) {
      if (type != "text") continue;
      if (!text.empty()) text += "\n\n";
      text += t;
    }
    return text;
  }
};

// json_value<std::string>(obj, key, def): the string, or def when the key is missing, null or not a string.
std::string str_or(const Value& obj, std::string_view key, const std::string& def) {
  const Value* v = obj.find(key);
  return v && v->is_string() ? v->as_string() : def;
}

[[noreturn]] void fail(const std::string& what) { throw ChatError(what); }

const Value& need(const Value& obj, std::string_view key, const char* what) {
  const Value* v = obj.find(key);
  if (!v) fail(std::string("Failed to parse messages: ") + what);
  return *v;
}

std::string need_string(const Value& obj, std::string_view key, const char* what) {
  const Value& v = need(obj, key, what);
  if (!v.is_string()) fail(std::string("Failed to parse messages: ") + what + " is not a string");
  return v.as_string();
}

// The message as llama.cpp reads it (common_chat_msgs_parse_oaicompat), after the checks oaicompat_chat_params_parse makes.
Msg read_message(const Value& message) {
  if (!message.is_object()) fail("Failed to parse messages: Expected 'message' to be an object");
  Msg m;
  const std::string role0 = str_or(message, "role", "");
  const Value* content = message.find("content");
  if (role0 != "assistant" && !content) fail("All non-assistant messages must contain 'content'");
  if (role0 == "assistant" && !content && !message.find("tool_calls"))
    fail("Assistant message must contain either 'content' or 'tool_calls'!");
  m.role = need_string(message, "role", "Missing 'role'");
  if (content) {
    if (content->is_string()) {
      m.content = content->as_string();
    } else if (content->is_array()) {
      m.has_parts = true;
      for (const auto& part : content->items()) {
        const std::string type = str_or(part, "type", "");
        if (type != "text" && type != "tool_reference") fail("unsupported content[].type");
        if (type == "text") m.parts.emplace_back(type, need_string(part, "text", "content part text"));
        else m.parts.emplace_back(type, need_string(part, "name", "content part name"));
      }
    } else if (!content->is_null()) {
      fail("Expected 'content' to be a string or an array");
    }
  }
  if (const Value* calls = message.find("tool_calls"); calls && !calls->is_null()) {
    if (!calls->is_array()) fail("Failed to parse messages: tool_calls is not an array");
    for (const auto& tc : calls->items()) {
      const Value* type = tc.find("type");
      if (!type) fail("Failed to parse messages: Missing tool call type");
      if (!eq(type, "function")) fail("Failed to parse messages: Unsupported tool call type");
      const Value* fc = tc.find("function");
      if (!fc) fail("Failed to parse messages: Missing tool call function");
      if (!fc->find("name")) fail("Failed to parse messages: Missing tool call name");
      Msg::Call c;
      c.name = need_string(*fc, "name", "tool call name");
      const Value& args = need(*fc, "arguments", "tool call arguments");
      c.arguments = args.is_string() ? args.as_string() : jinja_tojson(args, false);
      if (const Value* id = tc.find("id")) {
        if (!id->is_string()) fail("Failed to parse messages: tool call id is not a string");
        c.id = id->as_string();
      }
      m.calls.push_back(std::move(c));
    }
  } else if (calls == nullptr && !content) {
    fail("Failed to parse messages: Expected 'content' or 'tool_calls'");
  }
  if (message.find("reasoning_content")) m.reasoning_content = need_string(message, "reasoning_content", "reasoning_content");
  if (message.find("name")) m.name = need_string(message, "name", "name");
  if (message.find("tool_call_id")) m.tool_call_id = need_string(message, "tool_call_id", "tool_call_id");
  return m;
}

// common_chat_msg::to_json_oaicompat, then the workarounds the template needs (role "developer" is system, tool-call
// arguments are objects).
Value message_json(const Msg& m) {
  Value j = Value::object();
  j.set("role", Value::string(m.role == "developer" ? "system" : m.role));
  if (!m.content.empty()) {
    j.set("content", Value::string(m.content));
  } else if (!m.parts.empty()) {
    std::vector<Value> parts;
    for (const auto& [type, text] : m.parts) {
      Value p = Value::object();
      p.set("type", Value::string(type));
      p.set(type == "tool_reference" ? "name" : "text", Value::string(text));
      parts.push_back(std::move(p));
    }
    j.set("content", Value::array(std::move(parts)));
  } else {
    j.set("content", Value::string(""));
  }
  if (!m.reasoning_content.empty()) j.set("reasoning_content", Value::string(m.reasoning_content));
  if (!m.name.empty()) j.set("name", Value::string(m.name));
  if (!m.tool_call_id.empty()) j.set("tool_call_id", Value::string(m.tool_call_id));
  if (!m.calls.empty()) {
    std::vector<Value> calls;
    for (const auto& c : m.calls) {
      Value fn = Value::object();
      fn.set("name", Value::string(c.name));
      try {
        fn.set("arguments", parse_llama_json(c.arguments));
      } catch (const ChatError& e) {
        fail(std::string("Failed to parse tool call arguments as JSON: ") + e.what());
      }
      Value tc = Value::object();
      tc.set("type", Value::string("function"));
      tc.set("function", std::move(fn));
      if (!c.id.empty()) tc.set("id", Value::string(c.id));
      calls.push_back(std::move(tc));
    }
    j.set("tool_calls", Value::array(std::move(calls)));
  }
  return j;
}

// common_chat_tools_parse_oaicompat, as the template's tools array (OpenAI's shape, description always a string).
Value read_tools(const Value* tools, std::vector<std::string>& names) {
  if (!tools || tools->is_null()) return Value::null();
  if (!tools->is_array()) fail("Failed to parse tools: Expected 'tools' to be an array");
  std::vector<Value> out;
  for (const auto& tool : tools->items()) {
    const Value* type = tool.find("type");
    if (!type) fail("Failed to parse tools: Missing tool type");
    if (!eq(type, "function")) fail("Failed to parse tools: Unsupported tool type");
    const Value* fn = tool.find("function");
    if (!fn) fail("Failed to parse tools: Missing tool function");
    const Value* name = fn->find("name");
    if (!name || !name->is_string()) fail("Failed to parse tools: the function's name is missing or not a string");
    const Value* desc = fn->find("description");
    if (desc && !desc->is_string()) fail("Failed to parse tools: the function's description is not a string");
    const Value* defer = fn->find("defer_loading") ? fn->find("defer_loading") : tool.find("defer_loading");
    if (defer && !defer->is_bool()) fail("Failed to parse tools: Expected 'defer_loading' to be a boolean");
    const Value* params = fn->find("parameters");
    Value f = Value::object();
    f.set("name", *name);
    f.set("description", desc ? *desc : Value::string(""));
    f.set("parameters", params ? *params : Value::object());
    Value t = Value::object();
    t.set("type", Value::string("function"));
    t.set("function", std::move(f));
    names.push_back(name->as_string());
    out.push_back(std::move(t));
  }
  return Value::array(std::move(out));
}

}  // namespace

InklingChat render_inkling_request(const Value& raw) {
  const Value body = llama_json(raw);
  if (!body.is_object()) fail("'messages' is required");
  InklingChat chat;
  InklingMode& mode = chat.mode;

  std::vector<std::string> tool_names;
  const Value tools = read_tools(body.find("tools"), tool_names);
  const bool has_tools = !tool_names.empty();

  const Value* tc = body.find("tool_choice");
  const std::string tool_choice = tc && tc->is_string() ? tc->as_string() : "auto";
  if (tool_choice == "auto") mode.tool_choice = InklingToolChoice::Auto;
  else if (tool_choice == "none") mode.tool_choice = InklingToolChoice::None;
  else if (tool_choice == "required") mode.tool_choice = InklingToolChoice::Required;
  else fail("Invalid tool_choice: " + tool_choice);

  // json_schema, grammar and response_format: only whether a schema constrains the answer, and the errors they raise.
  Value json_schema = body.find("json_schema") ? *body.find("json_schema") : Value::null();
  const std::string grammar = str_or(body, "grammar", "");
  if (!json_schema.is_null() && !grammar.empty()) fail("Cannot use both json_schema and grammar");
  auto json_empty = [](const Value& v) { return v.is_null() || (v.is_object() && v.members().empty()) || (v.is_array() && v.items().empty()); };
  auto json_or = [](const Value& obj, std::string_view key, Value def) {
    const Value* v = obj.find(key);
    return v && !v->is_null() ? *v : def;
  };
  if (body.find("response_format")) {
    const Value rf = json_or(body, "response_format", Value::object());
    const std::string type = str_or(rf, "type", "");
    if (type == "json_object") {
      if (rf.find("schema") || json_empty(json_schema)) json_schema = json_or(rf, "schema", Value::object());
    } else if (type == "json_schema") {
      if (rf.find("json_schema")) json_schema = json_or(json_or(rf, "json_schema", Value::object()), "schema", Value::object());
      else json_schema = json_or(rf, "schema", Value::object());
    } else if (!type.empty() && type != "text") {
      fail("response_format type must be one of \"text\", \"json_object\" or \"json_schema\", but got: " + type);
    }
  }
  mode.response_format = json_schema.is_object();

  const Value* messages_v = body.find("messages");
  if (!messages_v) fail("'messages' is required");
  if (!messages_v->is_array()) fail("Expected 'messages' to be an array");
  std::vector<Msg> msgs;
  for (const auto& m : messages_v->items()) msgs.push_back(read_message(m));

  const Value* agp = body.find("add_generation_prompt");
  bool add_generation_prompt = agp && agp->is_bool() ? agp->as_bool() : true;
  enum class Cont { None, Auto, Content, Reasoning } cont = Cont::None;
  if (const Value* c = body.find("continue_final_message")) {
    if (c->is_null()) cont = Cont::None;
    else if (c->is_bool()) cont = c->as_bool() ? Cont::Auto : Cont::None;
    else if (eq(c, "reasoning_content")) cont = Cont::Reasoning;
    else if (eq(c, "content")) cont = Cont::Content;
    else fail("Invalid continue_final_message: expected a boolean, \"content\" or \"reasoning_content\"");
  }
  if (cont == Cont::None && !msgs.empty() && msgs.back().role == "assistant") {
    if (msgs.size() >= 2 && msgs[msgs.size() - 2].role == "assistant")
      fail("Cannot have 2 or more assistant messages at the end of the list.");
    cont = Cont::Auto;
    add_generation_prompt = false;
  }
  if (cont != Cont::None && add_generation_prompt) fail("Cannot set both add_generation_prompt and continue_final_message to true.");
  if (cont != Cont::None && !msgs.empty() && msgs.back().role == "assistant" && !msgs.back().calls.empty())
    fail("Cannot continue an assistant message that contains tool calls.");

  if (const Value* rfmt = body.find("reasoning_format")) {
    if (!rfmt->is_string()) fail("reasoning_format is not a string");
    const std::string& n = rfmt->as_string();
    if (n != "none" && n != "auto" && n != "deepseek" && n != "deepseek-legacy") fail("Unknown reasoning format: " + n);
    mode.extract_reasoning = n != "none";
  }
  if (has_tools && mode.tool_choice != InklingToolChoice::None && body.find("grammar"))
    fail("Cannot use custom grammar constraints with tools.");

  // chat_template_kwargs: only reasoning_effort reaches the template (and enable_thinking must not be a string); the
  // OpenAI field reasoning_effort overrides the kwarg.
  Value effort;
  bool have_effort = false;
  if (const Value* kw = body.find("chat_template_kwargs"); kw && kw->is_object()) {
    if (const Value* e = kw->find("enable_thinking"); e && e->is_string())
      fail("invalid type for \"enable_thinking\" (expected boolean, got string)");
    if (const Value* e = kw->find("reasoning_effort")) effort = *e, have_effort = true;
  }
  if (const std::string re = str_or(body, "reasoning_effort", ""); !re.empty()) {
    effort = Value::string(re);
    have_effort = true;
  }
  // (the levels llama.cpp finds for this template are none, minimal, low, medium, high and max; none is rejected: none)

  const std::string parallel_key = "parallel_tool_calls";
  const Value* par = body.find(parallel_key);
  mode.parallel_tool_calls = par && par->is_bool() ? par->as_bool() : true;
  mode.tool_names = tool_names;

  // The final message of a continuation is not rendered by the template; it opens the block the model goes on with.
  std::optional<Msg> last;
  if (cont != Cont::None && !msgs.empty()) {
    last = std::move(msgs.back());
    msgs.pop_back();
    if (cont == Cont::Auto)
      cont = !last->reasoning_content.empty() && last->content.empty() && last->parts.empty() ? Cont::Reasoning : Cont::Content;
  }
  std::vector<Value> mj;
  for (const auto& m : msgs) mj.push_back(message_json(m));
  chat.parts = render_inkling(Value::array(std::move(mj)), tools, have_effort ? &effort : nullptr);

  if (last && !last->empty()) {
    Out g;
    if (!last->reasoning_content.empty()) {
      g.lit("<|message_model|><|content_thinking|>");
      g.in(last->reasoning_content);
    }
    if (cont == Cont::Content) {
      if (!g.parts.empty()) g.lit("<|end_message|>");
      g.lit("<|message_model|><|content_text|>");
      g.in(last->render_content());
    }
    for (const auto& p : g.parts) {
      chat.generation_prompt += p.text;
      if (!chat.parts.empty() && chat.parts.back().input == p.input) chat.parts.back().text += p.text;
      else chat.parts.push_back(p);
    }
    mode.prefill = chat.generation_prompt;
  }
  return chat;
}

}  // namespace eklow::text
