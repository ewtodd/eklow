#include "text/chat_qwen36.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "text/pyfmt.hpp"

namespace eklow::text {

namespace {

using json::Value;
// A value as the template sees it: nullptr is Jinja's Undefined (a missing key, or an attribute of something that has
// none), which differs from None (JSON null).
using V = const Value*;

V sub(V obj, std::string_view key) { return obj && obj->is_object() ? obj->find(key) : nullptr; }

bool truthy(V v) { return v && v->truthy(); }
bool is_string(V v) { return v && v->is_string(); }
bool is_mapping(V v) { return v && v->is_object(); }
bool eq(V v, std::string_view s) { return v && v->is_string() && v->as_string() == s; }

// {{ x }}: str(x), nothing for Undefined.
std::string print(V v) { return v ? py::str(*v) : std::string(); }

// A string operand of `+`: Jinja2 raises on Undefined, Python on anything but a string.
const std::string& need_string(V v, const char* what) {
  if (!is_string(v)) throw ChatError(std::string("chat_template: ") + what + " must be a string");
  return v->as_string();
}

// What `for x in v` walks: list items, a string's characters, a dict's keys; nothing for Undefined.
std::vector<Value> iter(V v) {
  std::vector<Value> out;
  if (!v) return out;
  if (v->is_array()) return v->items();
  if (v->is_object()) {
    for (const auto& m : v->members()) out.push_back(Value::string(m.first));
    return out;
  }
  if (v->is_string()) {
    const std::string& s = v->as_string();
    for (size_t i = 0; i < s.size();) {
      unsigned char c = s[i];
      size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
      out.push_back(Value::string(s.substr(i, n)));
      i += n;
    }
    return out;
  }
  throw ChatError("chat_template: cannot iterate over " + py::str(*v));
}

// `needle in container` (a string): a substring of a string, a key of a dict, an element of a list; false for
// Undefined, which iterates as empty; a TypeError for anything else.
bool contains(V c, std::string_view needle) {
  if (!c) return false;
  if (c->is_string()) return c->as_string().find(needle) != std::string::npos;
  if (c->is_object()) return c->find(needle) != nullptr;
  if (c->is_array()) {
    for (const auto& x : c->items())
      if (x.is_string() && x.as_string() == needle) return true;
    return false;
  }
  throw ChatError("chat_template: argument of type " + std::string(c->is_null() ? "NoneType" : "number or boolean") + " is not iterable");
}

// A string that remembers which of its parts came from the request.
class Text {
 public:
  void lit(std::string_view s) { add(s, false); }
  void in(std::string_view s) { add(s, true); }
  void add(std::string_view s, bool input) {
    if (s.empty()) return;
    if (!parts.empty() && parts.back().input == input) parts.back().text += s;
    else parts.push_back({std::string(s), input});
  }
  void append(const Text& o) {
    for (const auto& p : o.parts) add(p.text, p.input);
  }
  std::string str() const { return join(parts); }
  // The characters [b, e) of the string.
  Text slice(size_t b, size_t e) const {
    Text out;
    size_t pos = 0;
    for (const auto& p : parts) {
      size_t lo = std::max(pos, b), hi = std::min(pos + p.text.size(), e);
      if (lo < hi) out.add(std::string_view(p.text).substr(lo - pos, hi - lo), p.input);
      pos += p.text.size();
    }
    return out;
  }
  // | trim: Python's str.strip over the whole string, wherever its parts begin and end.
  void trim() {
    std::string all = str();
    std::string t = py::strip(all);
    if (t.size() == all.size()) return;
    size_t lead = t.empty() ? all.size() : all.find(t);
    *this = slice(lead, lead + t.size());
  }
  std::vector<ChatPart> parts;
};

// The template's own namespace counters for "Picture N: " and "Video N: ".
struct Counts {
  long image = 0, video = 0;
};

// The macro render_content(content, do_vision_count, is_system_content).
Text render_content(V content, bool count, bool is_system, Counts& n, bool add_vision_id) {
  Text out;
  if (is_string(content)) {
    out.in(content->as_string());
  } else if (!content || content->is_array()) {  // iterable and not a mapping (Undefined iterates as empty)
    for (const auto& item_value : iter(content)) {
      V item = &item_value;
      if (contains(item, "image") || contains(item, "image_url") || eq(sub(item, "type"), "image")) {
        if (is_system) throw ChatError("System message cannot contain images.");
        if (count) ++n.image;
        if (add_vision_id) out.lit("Picture " + std::to_string(n.image) + ": ");
        out.lit("<|vision_start|><|image_pad|><|vision_end|>");
      } else if (contains(item, "video") || eq(sub(item, "type"), "video")) {
        if (is_system) throw ChatError("System message cannot contain videos.");
        if (count) ++n.video;
        if (add_vision_id) out.lit("Video " + std::to_string(n.video) + ": ");
        out.lit("<|vision_start|><|video_pad|><|vision_end|>");
      } else if (contains(item, "text")) {
        out.in(print(sub(item, "text")));
      } else {
        throw ChatError("Unexpected item type in content.");
      }
    }
  } else if (content->is_null()) {
    // ''
  } else {
    throw ChatError("Unexpected content type.");
  }
  return out;
}

void json_string(std::string& o, std::string_view s) {
  o += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) {
          static const char* hex = "0123456789abcdef";
          o += "\\u00";
          o += hex[c >> 4];
          o += hex[c & 15];
        } else {
          o += char(c);
        }
    }
  }
  o += '"';
}

// transformers' tojson for chat templates: json.dumps(x, ensure_ascii=False), so ", " and ": " separators, keys in
// insertion order, floats as repr, non-ASCII text as it is.
void tojson(std::string& o, const Value& v) {
  switch (v.kind()) {
    case Value::Kind::Null: o += "null"; break;
    case Value::Kind::Bool: o += v.as_bool() ? "true" : "false"; break;
    case Value::Kind::Int: o += v.int_digits(); break;
    case Value::Kind::Float: {
      const double d = v.as_double();
      o += std::isnan(d) ? "NaN" : std::isinf(d) ? (d < 0 ? "-Infinity" : "Infinity") : py::float_repr(d);
      break;
    }
    case Value::Kind::String: json_string(o, v.as_string()); break;
    case Value::Kind::Array: {
      o += '[';
      bool first = true;
      for (const auto& x : v.items()) {
        if (!first) o += ", ";
        first = false;
        tojson(o, x);
      }
      o += ']';
      break;
    }
    case Value::Kind::Object: {
      o += '{';
      bool first = true;
      for (const auto& [k, x] : v.members()) {
        if (!first) o += ", ";
        first = false;
        json_string(o, k);
        o += ": ";
        tojson(o, x);
      }
      o += '}';
      break;
    }
  }
}

std::string tojson(const Value& v) {
  std::string o;
  tojson(o, v);
  return o;
}

uint64_t fnv(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) h = (h ^ c) * 0x100000001b3ull;
  return h;
}

}  // namespace

std::optional<QwenTemplate> qwen36_template_of(std::string_view tmpl) {
  // FNV-1a over the three chat_template.jinja files: stock Qwen/Qwen3.6-35B-A3B at 995ad96e (7,764 bytes, no final
  // newline), ornith-ai/Ornith-1.5-35B-A3B at 10fbf86f (7,536 bytes, with one) and InternScience/Sage-A1.5 at 0e3205cf
  // (9,025 bytes, no final newline; its tokenizer_config.json copy has one, 9,026 bytes, and is accepted too).
  const uint64_t h = fnv(tmpl);
  if (tmpl.size() == 7764 && h == 0x8ae3a8bc6e68f114ull) return QwenTemplate::Stock;
  if (tmpl.size() == 7536 && h == 0x94e2de6b0aac72c0ull) return QwenTemplate::Ornith;
  if (tmpl.size() == 9025 && h == 0xf15d4bcc1307dcbaull) return QwenTemplate::Sage;
  if (tmpl.size() == 9026 && h == 0x29647bc4565bff10ull) return QwenTemplate::Sage;
  return std::nullopt;
}

const char* qwen36_template_name(QwenTemplate which) {
  switch (which) {
    case QwenTemplate::Stock: return "stock";
    case QwenTemplate::Ornith: return "ornith";
    case QwenTemplate::Sage: return "sage";
  }
  return "stock";
}

std::vector<ChatPart> render_qwen36(QwenTemplate which, const Value& messages, const Value& tools_value, const QwenChatOptions& opt) {
  if (!messages.is_array()) throw ChatError("chat_template: messages must be a list");
  const auto& msgs = messages.items();
  if (msgs.empty()) throw ChatError("No messages provided.");
  // Hugging Face takes only dicts (or functions) as tools and a list as the list of them.
  if (!tools_value.is_null() && !tools_value.is_array()) throw ChatError("Tools should either be a JSON schema, or a callable function");
  for (const auto& t : tools_value.is_array() ? tools_value.items() : std::vector<Value>{})
    if (!t.is_object()) throw ChatError("Tools should either be a JSON schema, or a callable function");
  const bool have_tools = tools_value.is_array() && !tools_value.items().empty();

  Text o;
  Counts counts;
  const size_t n = msgs.size();
  V first_role = sub(&msgs[0], "role");
  // Sage's verification switch: `verify_on = verification is defined and verification`, and the two raises after it.
  static constexpr std::string_view kVerifyTail = "Verification protocol: enabled";
  bool verify_on = false;
  if (which == QwenTemplate::Sage && opt.verification) {
    const Value& v = *opt.verification;
    verify_on = v.truthy();
    if (verify_on && v.is_string()) {
      std::string low = v.as_string();
      for (char& c : low) c = char(std::tolower(static_cast<unsigned char>(c)));
      if (low != "true" && low != "1" && low != "yes" && low != "on")
        throw ChatError("verification must be a boolean, not the string " + tojson(v) + ". Send a JSON boolean: {\"chat_template_kwargs\": {\"verification\": true}}.");
    }
    if (verify_on) {
      bool blank = true;
      if (eq(first_role, "system")) {
        Counts scratch;
        Text c = render_content(sub(&msgs[0], "content"), false, true, scratch, opt.add_vision_id);
        c.trim();
        blank = c.parts.empty();
      }
      if (blank) throw ChatError("verification=true requires a non-empty system message: the verification protocol trigger is appended to the end of the system content.");
    }
  }
  // content = content + verify_trigger if (verify_on and not content.endswith(verify_tail)) else content
  const auto verify = [&](Text& content) {
    if (verify_on && !content.str().ends_with(kVerifyTail)) content.lit("\n\nVerification protocol: enabled");
  };
  if (have_tools) {
    o.lit("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (const auto& tool : tools_value.items()) {
      o.lit("\n");
      o.in(tojson(tool));
    }
    o.lit("\n</tools>");
    o.lit(
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
        "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
        "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
        "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
        "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
        "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
        "the question like normal with your current knowledge and do not tell the user about function calls\n"
        "</IMPORTANT>");
    if (eq(first_role, "system")) {
      Text content = render_content(sub(&msgs[0], "content"), false, true, counts, opt.add_vision_id);
      content.trim();
      verify(content);
      if (!content.parts.empty()) {
        o.lit("\n\n");
        o.append(content);
      }
    }
    o.lit("<|im_end|>\n");
  } else if (eq(first_role, "system")) {
    Text content = render_content(sub(&msgs[0], "content"), false, true, counts, opt.add_vision_id);
    content.trim();
    verify(content);
    o.lit("<|im_start|>system\n");
    o.append(content);
    o.lit("<|im_end|>\n");
  }

  // The last user query: the last user message, scanning back, that is not a tool response in a user's clothes.
  long last_query = long(n) - 1;
  bool multi_step_tool = true;
  for (long idx = long(n) - 1; idx >= 0; --idx) {
    V message = &msgs[size_t(idx)];
    if (multi_step_tool && eq(sub(message, "role"), "user")) {
      Text c = render_content(sub(message, "content"), false, false, counts, opt.add_vision_id);
      c.trim();
      const std::string s = c.str();
      const bool wrapped = s.starts_with("<tool_response>") && s.ends_with("</tool_response>");
      if (!wrapped) {
        multi_step_tool = false;
        last_query = idx;
      }
    }
  }
  if (multi_step_tool) throw ChatError("No user query found in messages.");

  for (size_t i = 0; i < n; ++i) {
    V message = &msgs[i];
    Text content = render_content(sub(message, "content"), true, false, counts, opt.add_vision_id);
    content.trim();
    V role = sub(message, "role");
    if (eq(role, "system")) {
      if (i != 0) throw ChatError("System message must be at the beginning.");
    } else if (eq(role, "user")) {
      o.lit("<|im_start|>user\n");
      o.append(content);
      o.lit("<|im_end|>\n");
    } else if (eq(role, "assistant")) {
      Text reasoning;
      V rc = sub(message, "reasoning_content");
      if (is_string(rc)) {
        reasoning.in(rc->as_string());
      } else {
        const std::string flat = content.str();
        const size_t close = flat.find("</think>");
        if (close != std::string::npos) {
          // content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
          size_t e = close;
          while (e > 0 && flat[e - 1] == '\n') --e;
          size_t b = 0;
          if (e >= 7) {
            const size_t open = flat.rfind("<think>", e - 7);
            if (open != std::string::npos) b = open + 7;
          }
          while (b < e && flat[b] == '\n') ++b;
          reasoning = content.slice(b, e);
          // content.split('</think>')[-1].lstrip('\n')
          size_t s = flat.rfind("</think>") + 8;
          while (s < flat.size() && flat[s] == '\n') ++s;
          content = content.slice(s, flat.size());
        }
      }
      reasoning.trim();
      if (which == QwenTemplate::Ornith || opt.preserve_thinking || long(i) > last_query) {
        o.lit("<|im_start|>assistant\n<think>\n");
        o.append(reasoning);
        o.lit("\n</think>\n\n");
        o.append(content);
      } else {
        o.lit("<|im_start|>assistant\n");
        o.append(content);
      }
      V tool_calls = sub(message, "tool_calls");
      if (truthy(tool_calls) && (tool_calls->is_array() || tool_calls->is_string())) {
        const bool content_nonblank = !py::strip(content.str()).empty();
        bool first = true;
        for (const auto& tc_value : iter(tool_calls)) {
          V tc = &tc_value;
          if (V fn = sub(tc, "function")) tc = fn;
          const std::string& name = need_string(sub(tc, "name"), "tool_calls[].function.name");
          if (first && content_nonblank) o.lit("\n\n<tool_call>\n<function=");
          else if (first) o.lit("<tool_call>\n<function=");
          else o.lit("\n<tool_call>\n<function=");
          first = false;
          o.in(name);
          o.lit(">\n");
          if (V args = sub(tc, "arguments")) {
            // tool_call.arguments|items
            if (!is_mapping(args)) throw ChatError("Can only get item pairs from a mapping.");
            for (const auto& [key, value] : args->members()) {
              o.lit("<parameter=");
              o.in(key);
              o.lit(">\n");
              o.in(value.is_string() ? value.as_string() : tojson(value));
              o.lit("\n</parameter>\n");
            }
          }
          o.lit("</function>\n</tool_call>");
        }
      }
      o.lit("<|im_end|>\n");
    } else if (eq(role, "tool")) {
      if (i > 0 && truthy(&msgs[i - 1]) && !eq(sub(&msgs[i - 1], "role"), "tool")) o.lit("<|im_start|>user");
      o.lit("\n<tool_response>\n");
      o.append(content);
      o.lit("\n</tool_response>");
      if (i + 1 < n) {
        if (!eq(sub(&msgs[i + 1], "role"), "tool")) o.lit("<|im_end|>\n");
      } else {
        o.lit("<|im_end|>\n");
      }
    } else {
      throw ChatError("Unexpected message role.");
    }
  }

  if (opt.add_generation_prompt) {
    o.lit("<|im_start|>assistant\n");
    o.lit(opt.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n");
  }
  return o.parts;
}

}  // namespace eklow::text
