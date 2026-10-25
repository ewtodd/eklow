#include "text/chat_gemma4.hpp"

#include <algorithm>

#include "text/pyfmt.hpp"

namespace eklow::text {

namespace {

using json::Value;
// A value as the template sees it: nullptr is Jinja's Undefined (a missing key read with x['k']),
// which differs from None (JSON null, or a missing key read with x.get('k')).
using V = const Value*;

const Value kNone;

V sub(V obj, std::string_view key) { return obj && obj->is_object() ? obj->find(key) : nullptr; }

// dict.get(key): the value, or None.
V get(V obj, std::string_view key) {
  if (!obj || !obj->is_object()) throw ChatError("chat_template: .get('" + std::string(key) + "') on a non-object");
  V v = obj->find(key);
  return v ? v : &kNone;
}

bool truthy(V v) { return v && v->truthy(); }
bool is_none(V v) { return v && v->is_null(); }
bool is_string(V v) { return v && v->is_string(); }
bool is_mapping(V v) { return v && v->is_object(); }
// Jinja's `is sequence`: anything with a length and an item lookup, so strings, lists, dicts and
// Undefined (which has length 0).
bool is_sequence(V v) { return !v || v->is_string() || v->is_array() || v->is_object(); }
bool eq(V v, std::string_view s) { return v && v->is_string() && v->as_string() == s; }
bool same(V a, V b) { return (a ? *a : kNone) == (b ? *b : kNone); }

// {{ x }}: str(x), nothing for Undefined.
std::string print(V v) { return v ? py::str(*v) : std::string(); }
// x | upper
std::string upper(V v) { return py::upper(print(v)); }

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

// v | dictsort: the members ordered by key.lower(), ties in insertion order.
std::vector<const Value::Member*> dictsort(V v) {
  if (!is_mapping(v)) throw ChatError("chat_template: dictsort of " + (v ? py::str(*v) : std::string("an undefined value")));
  std::vector<const Value::Member*> out;
  for (const auto& m : v->members()) out.push_back(&m);
  std::stable_sort(out.begin(), out.end(), [](auto* a, auto* b) { return py::lower(a->first) < py::lower(b->first); });
  return out;
}

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
  // | trim over the whole output, wherever its parts begin and end.
  void trim() {
    std::string all = join(parts);
    std::string t = py::strip(all);
    if (t.size() == all.size()) return;
    size_t lead = all.find(t), drop_front = t.empty() ? all.size() : lead, keep = t.size();
    std::vector<ChatPart> out;
    size_t pos = 0;
    for (auto& p : parts) {
      size_t b = std::max(pos, drop_front), e = std::min(pos + p.text.size(), drop_front + keep);
      if (b < e) out.push_back({p.text.substr(b - pos, e - b), p.input});
      pos += p.text.size();
    }
    parts = std::move(out);
  }
  std::vector<ChatPart> parts;
};

constexpr std::string_view Q = "<|\"|>";

void format_argument(Out& o, V arg, bool escape_keys);

void quoted(Out& o, std::string_view s) {
  o.lit(Q);
  o.in(s);
  o.lit(Q);
}

// {%- if add_comma %},{%- else -%} {%- set add_comma = true -%} {% endif -%}
void comma(Out& o, bool& add_comma) {
  if (add_comma) o.lit(",");
  else add_comma = true;
}

void required_list(Out& o, V items) {
  o.lit("required:[");
  auto list = iter(items);
  for (size_t i = 0; i < list.size(); ++i) {
    quoted(o, print(&list[i]));
    if (i + 1 < list.size()) o.lit(",");
  }
  o.lit("]");
}

void format_parameters(Out& o, V properties, bool filter_keys) {
  static constexpr std::string_view standard[] = {"description", "type", "properties", "required", "nullable"};
  bool found_first = false;
  for (const auto* m : dictsort(properties)) {
    const std::string& key = m->first;
    V value = &m->second;
    bool add_comma = false;
    if (filter_keys && std::find(std::begin(standard), std::end(standard), key) != std::end(standard)) continue;
    if (found_first) o.lit(",");
    found_first = true;
    o.in(key);
    o.lit(":{");
    if (V d = sub(value, "description"); truthy(d)) {
      o.lit("description:");
      quoted(o, print(d));
      add_comma = true;
    }
    const std::string type = upper(sub(value, "type"));
    if (type == "STRING") {
      if (V e = sub(value, "enum"); truthy(e)) {
        comma(o, add_comma);
        o.lit("enum:");
        format_argument(o, e, true);
      }
    } else if (type == "ARRAY") {
      V items = sub(value, "items");
      if (is_mapping(items) && truthy(items)) {
        comma(o, add_comma);
        o.lit("items:{");
        bool first_item = true;
        for (const auto* im : dictsort(items)) {
          V iv = &im->second;
          if (is_none(iv)) continue;
          if (!first_item) o.lit(",");
          first_item = false;
          if (im->first == "properties") {
            o.lit("properties:{");
            if (is_mapping(iv)) format_parameters(o, iv, false);
            o.lit("}");
          } else if (im->first == "required") {
            required_list(o, iv);
          } else if (im->first == "type") {
            o.lit("type:");
            if (is_string(iv)) {
              Value up = Value::string(py::upper(iv->as_string()));
              format_argument(o, &up, true);
            } else {
              // item_value | map('upper') | list
              std::vector<Value> ups;
              for (const auto& x : iter(iv)) ups.push_back(Value::string(py::upper(py::str(x))));
              Value up = Value::array(std::move(ups));
              format_argument(o, &up, true);
            }
          } else {
            o.in(im->first);
            o.lit(":");
            format_argument(o, iv, true);
          }
        }
        o.lit("}");
      }
    }
    if (truthy(sub(value, "nullable"))) {
      comma(o, add_comma);
      o.lit("nullable:true");
    }
    if (type == "OBJECT") {
      V props = sub(value, "properties");
      if (props && is_mapping(props)) {
        comma(o, add_comma);
        o.lit("properties:{");
        format_parameters(o, props, false);
        o.lit("}");
      } else if (is_mapping(value)) {
        comma(o, add_comma);
        o.lit("properties:{");
        format_parameters(o, value, true);
        o.lit("}");
      }
      if (V r = sub(value, "required"); truthy(r)) {
        comma(o, add_comma);
        required_list(o, r);
      }
    }
    comma(o, add_comma);
    o.lit("type:");
    o.lit(Q);
    o.in(type);
    o.lit(Q);
    o.lit("}");
  }
}

void format_function_declaration(Out& o, V tool) {
  V fn = sub(tool, "function");
  if (!fn) throw ChatError("chat_template: a tool without 'function'");
  o.lit("declaration:");
  o.in(print(sub(fn, "name")));
  o.lit("{description:");
  quoted(o, print(sub(fn, "description")));
  V params = sub(fn, "parameters");
  if (truthy(params)) {
    o.lit(",parameters:{");
    if (V p = sub(params, "properties"); truthy(p)) {
      o.lit("properties:{");
      format_parameters(o, p, false);
      o.lit("},");
    }
    if (V r = sub(params, "required"); truthy(r)) {
      required_list(o, r);
      o.lit(",");
    }
    if (V t = sub(params, "type"); truthy(t)) {
      o.lit("type:");
      quoted(o, upper(t));
      o.lit("}");
    }
  }
  if (is_mapping(fn) && fn->find("response")) {
    V rd = fn->find("response");
    o.lit(",response:{");
    if (V d = sub(rd, "description"); truthy(d)) {
      o.lit("description:");
      quoted(o, print(d));
      o.lit(",");
    }
    if (upper(sub(rd, "type")) == "OBJECT") {
      o.lit("type:");
      quoted(o, upper(sub(rd, "type")));
      o.lit("}");
    }
  }
  o.lit("}");
}

void format_argument(Out& o, V arg, bool escape_keys) {
  if (is_none(arg)) {
    o.lit("null");
  } else if (is_string(arg)) {
    quoted(o, arg->as_string());
  } else if (arg && arg->is_bool()) {
    o.lit(arg->as_bool() ? "true" : "false");
  } else if (is_mapping(arg)) {
    o.lit("{");
    bool first = true;
    for (const auto* m : dictsort(arg)) {
      if (!first) o.lit(",");
      first = false;
      if (escape_keys) quoted(o, m->first);
      else o.in(m->first);
      o.lit(":");
      format_argument(o, &m->second, escape_keys);
    }
    o.lit("}");
  } else if (is_sequence(arg)) {
    // Lists, and Undefined, which iterates as empty.
    o.lit("[");
    auto list = iter(arg);
    for (size_t i = 0; i < list.size(); ++i) {
      format_argument(o, &list[i], escape_keys);
      if (i + 1 < list.size()) o.lit(",");
    }
    o.lit("]");
  } else {
    o.in(print(arg));
  }
}

// strip_thinking: drops every <|channel>...<channel|> span, keeps what is outside, and trims.
std::string strip_thinking(V text) {
  const std::string& s = need_string(text, "content given to strip_thinking");
  std::string result;
  size_t pos = 0;
  for (;;) {
    size_t end = s.find("<channel|>", pos);
    std::string_view part = std::string_view(s).substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    size_t open = part.find("<|channel>");
    result += open == std::string_view::npos ? part : part.substr(0, open);
    if (end == std::string::npos) break;
    pos = end + std::string_view("<channel|>").size();
  }
  return py::strip(result);
}

void format_tool_response_block(Out& o, const std::string& tool_name, V response) {
  o.lit("<|tool_response>");
  if (is_mapping(response)) {
    o.lit("response:");
    o.in(tool_name);
    o.lit("{");
    auto items = dictsort(response);
    for (size_t i = 0; i < items.size(); ++i) {
      o.in(items[i]->first);
      o.lit(":");
      format_argument(o, &items[i]->second, false);
      if (i + 1 < items.size()) o.lit(",");
    }
    o.lit("}");
  } else {
    o.lit("response:");
    o.in(tool_name);
    o.lit("{value:");
    format_argument(o, response, false);
    o.lit("}");
  }
  o.lit("<tool_response|>");
}

void media(Out& o, V type) {
  if (eq(type, "image") || eq(type, "image_url")) o.lit("<|image|>");
  else if (eq(type, "audio") || eq(type, "input_audio")) o.lit("<|audio|>");
  else if (eq(type, "video")) o.lit("<|video|>");
}

enum class Prev { None, Think, Tool, ToolCall, ToolResponse };

}  // namespace

bool is_gemma4_template(std::string_view tmpl) {
  // FNV-1a over Google's chat_template.jinja (18,683 bytes, published 2026-07-09).
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : tmpl) h = (h ^ c) * 0x100000001b3ull;
  return tmpl.size() == 18683 && h == 0x645a5f8cd6ec0ad0ull;
}

std::string join(const std::vector<ChatPart>& parts) {
  std::string s;
  for (const auto& p : parts) s += p.text;
  return s;
}

std::vector<ChatPart> render_gemma4(const Value& messages, const Value& tools_value, const ChatOptions& opt) {
  if (!messages.is_array()) throw ChatError("chat_template: messages must be a list");
  V tools = tools_value.is_null() ? nullptr : &tools_value;
  const auto& msgs = messages.items();
  Out o;
  Prev prev = Prev::None;
  V prev_non_tool_role = nullptr;
  size_t first = 0;

  o.lit(opt.bos_token);
  V first_role = msgs.empty() ? nullptr : sub(&msgs[0], "role");
  bool system_first = !msgs.empty() && (eq(first_role, "system") || eq(first_role, "developer"));
  if (opt.enable_thinking || truthy(tools) || system_first) {
    o.lit("<|turn>system\n");
    if (opt.enable_thinking) {
      o.lit("<|think|>\n");
      prev = Prev::Think;
    }
    if (system_first) {
      V content = sub(&msgs[0], "content");
      if (is_string(content)) {
        o.in(py::strip(content->as_string()));
      } else if (is_sequence(content)) {
        for (const auto& item : iter(content)) {
          o.in(py::strip(print(sub(&item, "text"))));
          o.lit(" ");
        }
      }
      first = 1;
    }
    if (truthy(tools)) {
      for (const auto& tool : iter(tools)) {
        o.lit("<|tool>");
        Out decl;
        format_function_declaration(decl, &tool);
        decl.trim();
        o.append(decl);
        o.lit("<tool|>");
      }
      prev = Prev::Tool;
    }
    o.lit("<turn|>\n");
  }

  const size_t n = msgs.size() - first;
  auto at = [&](size_t i) -> V { return &msgs[first + i]; };
  long last_user = -1;
  for (size_t i = 0; i < n; ++i)
    if (eq(sub(at(i), "role"), "user")) last_user = long(i);

  for (size_t i = 0; i < n; ++i) {
    V message = at(i);
    V msg_role = sub(message, "role");
    if (eq(msg_role, "tool")) continue;
    prev = Prev::None;
    if (!is_string(msg_role)) throw ChatError("chat_template: a message without a string role");
    const std::string role = msg_role->as_string() == "assistant" ? "model" : msg_role->as_string();
    const bool continue_same_model_turn = role == "model" && eq(prev_non_tool_role, "assistant");
    if (!continue_same_model_turn) {
      o.lit("<|turn>");
      o.in(role);
      o.lit("\n");
    }

    V reasoning = get(message, "reasoning");
    V thinking = truthy(reasoning) ? reasoning : get(message, "reasoning_content");
    V tool_calls = get(message, "tool_calls");
    const bool gate = long(i) > last_user || (opt.preserve_thinking && truthy(tool_calls));
    if (truthy(thinking) && gate) {
      o.lit("<|channel>thought\n");
      o.in(need_string(thinking, "reasoning"));
      o.lit("\n<channel|>");
    }

    if (truthy(tool_calls)) {
      for (const auto& tc : iter(tool_calls)) {
        V function = sub(&tc, "function");
        o.lit("<|tool_call>call:");
        o.in(need_string(sub(function, "name"), "tool_calls[].function.name"));
        o.lit("{");
        V args = sub(function, "arguments");
        if (is_mapping(args)) {
          bool firstarg = true;
          for (const auto* m : dictsort(args)) {
            if (!firstarg) o.lit(",");
            firstarg = false;
            o.in(m->first);
            o.lit(":");
            format_argument(o, &m->second, false);
          }
        } else if (!is_none(args)) {
          throw ChatError(
              "chat_template: tool_calls[].function.arguments must be a JSON object (mapping), not a string. "
              "Deserialize arguments before passing to the template.");
        }
        o.lit("}<tool_call|>");
      }
      prev = Prev::ToolCall;
    }

    bool tr_out = false;
    if (V legacy = get(message, "tool_responses"); truthy(legacy)) {
      for (const auto& tr : iter(legacy)) {
        V name = sub(&tr, "name");
        std::string tool_name = truthy(name) ? need_string(name, "tool_responses[].name") : "unknown";
        format_tool_response_block(o, tool_name, sub(&tr, "response"));
        tr_out = true;
        prev = Prev::ToolResponse;
      }
    } else if (truthy(tool_calls)) {
      // The tool messages that follow this one answer its calls.
      for (size_t k = i + 1; k < n; ++k) {
        V follow = at(k);
        if (!eq(sub(follow, "role"), "tool")) break;
        // ns_tname.name = follow.get('name') or 'unknown', then the name of the call whose id
        // matches (the last one, if several do; None matches None).
        static const Value unknown = Value::string("unknown");
        V fname = get(follow, "name");
        V tname = truthy(fname) ? fname : &unknown;
        for (const auto& tc : iter(tool_calls))
          if (same(get(&tc, "id"), get(follow, "tool_call_id"))) tname = sub(sub(&tc, "function"), "name");
        const std::string tool_name = need_string(tname, "the tool response's name");
        V body = get(follow, "content");
        if (is_string(body)) {
          format_tool_response_block(o, tool_name, body);
        } else if (is_sequence(body)) {
          std::string text;
          auto parts = iter(body);
          for (const auto& part : parts)
            if (eq(get(&part, "type"), "text")) {
              V t = get(&part, "text");
              text += need_string(t, "tool content text");
            }
          Value tv = Value::string(text);
          format_tool_response_block(o, tool_name, &tv);
          for (const auto& part : parts) media(o, get(&part, "type"));
        } else {
          format_tool_response_block(o, tool_name, body);
        }
        tr_out = true;
        prev = Prev::ToolResponse;
      }
    }

    Out captured;
    V content = get(message, "content");
    if (is_string(content)) {
      captured.in(role == "model" ? strip_thinking(content) : py::strip(content->as_string()));
    } else if (is_sequence(content)) {
      for (const auto& item : iter(content)) {
        V type = get(&item, "type");
        if (eq(type, "text")) {
          V text = sub(&item, "text");
          captured.in(role == "model" ? strip_thinking(text) : py::strip(print(text)));
        } else {
          media(captured, type);
        }
      }
    }
    o.append(captured);
    const bool has_content = !py::strip(join(captured.parts)).empty();

    V next_role = nullptr;
    bool next_found = false;
    for (size_t j = i + 1; j < n; ++j) {
      V r = sub(at(j), "role");
      if (!eq(r, "tool")) {
        next_role = r;
        next_found = true;
        break;
      }
    }
    const bool continues_into_next = role == "model" && eq(next_role, "assistant") && (!truthy(tool_calls) || tr_out);
    if (prev == Prev::ToolCall && !tr_out) o.lit("<|tool_response>");
    else if (continues_into_next) {
    } else if (!(tr_out && !has_content && !next_found)) {
      o.lit("<turn|>\n");
    }
    prev_non_tool_role = msg_role;
  }

  if (opt.add_generation_prompt) {
    if (prev != Prev::ToolResponse && prev != Prev::ToolCall) {
      o.lit("<|turn>model\n");
      if (!opt.enable_thinking) o.lit("<|channel>thought\n<channel|>");
    } else if (prev == Prev::ToolResponse && opt.enable_thinking) {
      o.lit("<|channel>thought\n");
    }
  }
  return o.parts;
}

}  // namespace eklow::text
