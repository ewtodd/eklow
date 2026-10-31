#include "text/tool_schema.hpp"

#include <algorithm>
#include <cctype>

#include "text/unicode.hpp"

namespace eklow::text {

namespace {

std::string lower_trim(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  std::string o = s.substr(b, e - b);
  for (char& c : o) c = char(std::tolower(static_cast<unsigned char>(c)));
  return o;
}

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

unsigned kind_of_name(const std::string& raw) {
  const std::string t = lower_trim(raw);
  if (t == "string" || t == "str" || t == "text" || t == "varchar" || t == "char" || t == "enum") return kString;
  if (starts(t, "int") || starts(t, "uint") || starts(t, "long") || starts(t, "short") || starts(t, "unsigned")) return kInteger;
  if (starts(t, "num") || starts(t, "float") || starts(t, "double")) return kNumber;
  if (t == "boolean" || t == "bool" || t == "binary") return kBoolean;
  if (t == "object" || t == "dict" || t == "map" || starts(t, "dict")) return kObject;
  if (t == "array" || t == "arr" || starts(t, "list")) return kArray;
  if (t == "null" || t == "none") return kNull;
  return 0;
}

unsigned kinds_of_type(const json::Value* type) {
  unsigned m = 0;
  if (!type) return 0;
  if (type->is_string()) return kind_of_name(type->as_string());
  if (type->is_array())
    for (const auto& t : type->items())
      if (t.is_string()) m |= kind_of_name(t.as_string());
  return m;
}

unsigned kind_of_value(const json::Value& v) {
  switch (v.kind()) {
    case json::Value::Kind::Null: return kNull;
    case json::Value::Kind::Bool: return kBoolean;
    case json::Value::Kind::Int: return kInteger;
    case json::Value::Kind::Float: return kNumber;
    case json::Value::Kind::String: return kString;
    case json::Value::Kind::Array: return kArray;
    case json::Value::Kind::Object: return kObject;
  }
  return 0;
}

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

}  // namespace

unsigned declared_kinds(const json::Value& schema) {
  if (!schema.is_object()) return kString;
  unsigned m = kinds_of_type(schema.find("type"));
  if (!m)
    for (const char* key : {"anyOf", "oneOf"})
      if (const auto* branches = schema.find(key); branches && branches->is_array())
        for (const auto& b : branches->items())
          if (b.is_object()) m |= kinds_of_type(b.find("type"));
  if (!m)
    if (const auto* e = schema.find("enum"); e && e->is_array())
      for (const auto& x : e->items()) m |= kind_of_value(x);
  if (!m)
    if (const auto* c = schema.find("const")) m = kind_of_value(*c);
  return m ? m : unsigned(kString);
}

const ParamSpec* ToolSpec::param(const std::string& key) const {
  for (const auto& p : params)
    if (p.name == key) return &p;
  return nullptr;
}

ToolSpecs::ToolSpecs(const json::Value& tools) {
  if (!tools.is_array()) return;
  for (const auto& tool : tools.items()) {
    const json::Value* fn = tool.is_object() && tool.find("function") && tool.find("function")->is_object() ? tool.find("function") : &tool;
    if (!fn->is_object() || !fn->find("name") || !fn->find("name")->is_string()) continue;
    ToolSpec spec;
    spec.name = unicode::nfc(fn->find("name")->as_string());
    const json::Value* params = fn->find("parameters");
    const json::Value* props = params && params->is_object() ? params->find("properties") : nullptr;
    if (props && props->is_object())
      for (const auto& [key, schema] : props->members()) {
        ParamSpec p;
        p.name = unicode::nfc(key);
        p.schema = schema;
        p.kinds = declared_kinds(schema);
        if (const auto* e = schema.is_object() ? schema.find("enum") : nullptr; e && e->is_array()) {
          p.enum_values = e->items();
          for (auto& m : p.enum_values)
            if (m.is_string()) m = json::Value::string(unicode::nfc(m.as_string()));
        }
        spec.params.push_back(std::move(p));
      }
    if (const auto* req = params && params->is_object() ? params->find("required") : nullptr; req && req->is_array())
      for (const auto& r : req->items()) {
        if (!r.is_string()) continue;
        const std::string rname = unicode::nfc(r.as_string());
        if (ParamSpec* p = const_cast<ParamSpec*>(spec.param(rname))) {
          p->required = true;
        } else {
          ParamSpec p2;
          p2.name = rname;
          p2.required = true;
          spec.params.push_back(std::move(p2));
        }
      }
    tools_.push_back(std::move(spec));
  }
}

const ToolSpec* ToolSpecs::find(const std::string& name) const {
  for (const auto& t : tools_)
    if (t.name == name) return &t;
  return nullptr;
}

json::Value convert_param(const ParamSpec* spec, const std::string& raw) {
  if (!spec) return raw == "null" ? json::Value::null() : json::Value::string(raw);
  const unsigned kinds = spec->kinds;
  if (kinds == kString) return json::Value::string(raw);
  // The text as JSON (the template's tojson), if it is a value of an allowed kind.
  json::Value v;
  bool parsed = false;
  try {
    v = json::parse(raw);
    parsed = true;
  } catch (const std::exception&) {
  }
  if (parsed) {
    bool ok = false;
    switch (v.kind()) {
      case json::Value::Kind::Null: ok = (kinds & kNull) || !(kinds & kString); break;
      case json::Value::Kind::Bool: ok = kinds & kBoolean; break;
      case json::Value::Kind::Int: ok = kinds & (kInteger | kNumber); break;
      case json::Value::Kind::Float: ok = kinds & kNumber; break;
      case json::Value::Kind::String: ok = false; break;
      case json::Value::Kind::Array: ok = kinds & kArray; break;
      case json::Value::Kind::Object: ok = kinds & kObject; break;
    }
    if (ok) return v;
  }
  if (kinds & kBoolean) {
    // vLLM lower-cases first, so Python's True and False are understood.
    size_t b = 0, e = raw.size();
    while (b < e && is_ws(raw[b])) ++b;
    while (e > b && is_ws(raw[e - 1])) --e;
    std::string w = raw.substr(b, e - b);
    for (char& c : w) c = char(std::tolower(static_cast<unsigned char>(c)));
    if (w == "true" || w == "false") return json::Value::boolean(w == "true");
  }
  return json::Value::string(raw);
}

}  // namespace eklow::text
