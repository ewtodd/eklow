// A small streaming JSON writer for tool output (probe records, GGUF dumps). Commas and nesting are
// tracked; keys and values are written in call order. Doubles are written with the shortest digits
// that round-trip; ones that are not finite become null.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace eklow::json {

class Writer {
 public:
  explicit Writer(bool pretty = true) : pretty_(pretty) {}

  Writer& begin_object();
  Writer& end_object();
  Writer& begin_array();
  Writer& end_array();
  Writer& key(std::string_view k);

  Writer& value(std::string_view s);
  Writer& value(const char* s) { return value(std::string_view(s)); }
  Writer& value(const std::string& s) { return value(std::string_view(s)); }
  Writer& value(bool b);
  Writer& value(int64_t i);
  Writer& value(uint64_t u);
  Writer& value(int i) { return value(int64_t(i)); }
  Writer& value(unsigned u) { return value(uint64_t(u)); }
  Writer& value(double d);
  Writer& null();
  // An already-formatted JSON value (an integer's exact digits).
  Writer& raw(std::string_view v);

  // key + value in one call.
  template <class T>
  Writer& kv(std::string_view k, const T& v) { key(k); return value(v); }

  const std::string& str() const { return out_; }

 private:
  void before_value();
  void newline();
  bool pretty_;
  std::string out_;
  // One entry per open container: whether it already holds an element.
  std::vector<bool> stack_;
  bool after_key_ = false;
};

std::string escape(std::string_view s);

}  // namespace eklow::json
