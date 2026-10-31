// What the request's tool declarations say about a call's parameters, for reading a Qwen tool call back
// (output_parser_qwen.cpp) and for constraining one (serve/tool_grammar_qwen.cpp). The tools are OpenAI's
// {"type": "function", "function": {"name", "parameters": {"properties", "required"}}} (a bare function object
// works too); anything that does not look like a declaration is skipped, since the template tojson's whatever it is given.
#pragma once

#include <string>
#include <vector>

#include "core/json_value.hpp"

namespace eklow::text {

// The kinds of JSON value a parameter's schema allows, as bits. A schema that names none (no type, or one we do
// not know) is free text: its value is the raw string, as vLLM's qwen3_coder parser reads it.
enum ValueKind : unsigned { kString = 1, kInteger = 2, kNumber = 4, kBoolean = 8, kObject = 16, kArray = 32, kNull = 64 };

// From "type" (a name or a list of names, any case; vLLM's spellings such as int32, float, dict, list, bool are
// understood), else from the branches of anyOf / oneOf, else from the kinds of the members of "enum" or the "const".
// Never 0: no information means kString.
unsigned declared_kinds(const json::Value& schema);

struct ParamSpec {
  std::string name;
  json::Value schema;  // the property's schema; null for a required name with no property
  bool required = false;
  unsigned kinds = kString;
  std::vector<json::Value> enum_values;  // the schema's "enum", if it has one
};

struct ToolSpec {
  std::string name;
  std::vector<ParamSpec> params;  // properties in declaration order, then required names that have none
  const ParamSpec* param(const std::string& key) const;
};

class ToolSpecs {
 public:
  ToolSpecs() = default;
  // `tools` is the request's array (or null).
  explicit ToolSpecs(const json::Value& tools);
  const ToolSpec* find(const std::string& name) const;
  const std::vector<ToolSpec>& tools() const { return tools_; }

 private:
  std::vector<ToolSpec> tools_;
};

// A parameter's text, as the template wrote it, back to a JSON value (the template writes strings raw and every
// other value with tojson). `spec` is null when the tool or the key is not declared.
//   - no spec: the raw string, except that exactly `null` is null (JSON null is what the template writes for None);
//   - string only: the raw string, whatever it looks like (a string "null" or "5" stays a string);
//   - otherwise the text as JSON, if it is a value of an allowed kind (an integer for integer, an integer or a
//     number for number, true/false in any case for boolean, an object, an array, null unless the schema allows
//     strings and not null); a union that allows strings and fails to parse as the others is the raw string;
//   - a value that is not of its declared kind stays the raw string, as in vLLM (which also returns False for a
//     boolean that is neither word; eklow keeps the text rather than invent a value).
json::Value convert_param(const ParamSpec* spec, const std::string& raw);

}  // namespace eklow::text
