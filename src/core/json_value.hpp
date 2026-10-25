// JSON values as a request carries them, with the distinctions Python's json module makes, since
// the chat template is defined by what Jinja2 does with them in Python:
//   - integers and floats are different kinds; an integer keeps its digits exactly (Python ints
//     are unbounded), a float is a double;
//   - objects keep their keys in first-seen order, and a repeated key keeps its first position
//     with its last value (a Python dict built from the pairs).
// parse() is strict RFC 8259 and refuses nesting deeper than kMaxDepth, so a hostile request
// cannot exhaust the stack (llama.cpp #30834 crashes llama-server that way).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace eklow::json {

constexpr int kMaxDepth = 128;

class Value {
 public:
  enum class Kind { Null, Bool, Int, Float, String, Array, Object };
  using Member = std::pair<std::string, Value>;

  Value() = default;
  static Value null() { return Value(); }
  static Value boolean(bool b);
  static Value integer(std::string digits);  // canonical decimal digits, optional leading '-'
  static Value integer(int64_t i);
  static Value number(double d);
  static Value string(std::string s);
  static Value array(std::vector<Value> items = {});
  static Value object(std::vector<Member> members = {});

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool is_bool() const { return kind_ == Kind::Bool; }
  bool is_int() const { return kind_ == Kind::Int; }
  bool is_float() const { return kind_ == Kind::Float; }
  bool is_string() const { return kind_ == Kind::String; }
  bool is_array() const { return kind_ == Kind::Array; }
  bool is_object() const { return kind_ == Kind::Object; }

  bool as_bool() const { return b_; }
  const std::string& int_digits() const { return s_; }
  double as_double() const { return d_; }
  const std::string& as_string() const { return s_; }
  const std::vector<Value>& items() const;
  const std::vector<Member>& members() const;
  std::vector<Value>& items();
  std::vector<Member>& members();

  // Object member lookup; nullptr when absent or when this is not an object.
  const Value* find(std::string_view key) const;
  // Appends, or replaces the value of an existing key in place (Python dict semantics).
  void set(std::string key, Value v);

  // Python truthiness: None, False, 0, 0.0, "", [] and {} are false.
  bool truthy() const;
  // Python ==, as the template's comparisons see it (1 == 1.0, True == 1).
  friend bool operator==(const Value& a, const Value& b);

 private:
  Kind kind_ = Kind::Null;
  bool b_ = false;
  double d_ = 0;
  std::string s_;
  std::shared_ptr<std::vector<Value>> arr_;
  std::shared_ptr<std::vector<Member>> obj_;
};

// Throws std::runtime_error with the byte offset of the problem.
Value parse(std::string_view text);
// Compact JSON, keys in order; for tools and tests.
std::string dump(const Value& v);

}  // namespace eklow::json
