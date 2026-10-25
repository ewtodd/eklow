#include "core/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>

namespace eklow::json {

std::string escape(std::string_view s) {
  std::string o;
  o.reserve(s.size() + 2);
  o += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += char(c);
        }
    }
  }
  o += '"';
  return o;
}

void Writer::newline() {
  if (!pretty_) return;
  out_ += '\n';
  out_.append(2 * stack_.size(), ' ');
}

void Writer::before_value() {
  if (after_key_) {
    after_key_ = false;
    return;
  }
  if (!stack_.empty()) {
    if (stack_.back()) out_ += ',';
    stack_.back() = true;
    newline();
  }
}

Writer& Writer::begin_object() {
  before_value();
  out_ += '{';
  stack_.push_back(false);
  return *this;
}

Writer& Writer::end_object() {
  bool had = stack_.back();
  stack_.pop_back();
  if (had) newline();
  out_ += '}';
  return *this;
}

Writer& Writer::begin_array() {
  before_value();
  out_ += '[';
  stack_.push_back(false);
  return *this;
}

Writer& Writer::end_array() {
  bool had = stack_.back();
  stack_.pop_back();
  if (had) newline();
  out_ += ']';
  return *this;
}

Writer& Writer::key(std::string_view k) {
  before_value();
  out_ += escape(k);
  out_ += pretty_ ? ": " : ":";
  after_key_ = true;
  return *this;
}

Writer& Writer::value(std::string_view s) {
  before_value();
  out_ += escape(s);
  return *this;
}

Writer& Writer::value(bool b) {
  before_value();
  out_ += b ? "true" : "false";
  return *this;
}

Writer& Writer::value(int64_t i) {
  before_value();
  out_ += std::to_string(i);
  return *this;
}

Writer& Writer::value(uint64_t u) {
  before_value();
  out_ += std::to_string(u);
  return *this;
}

Writer& Writer::value(double d) {
  before_value();
  if (!std::isfinite(d)) {
    out_ += "null";
    return *this;
  }
  // Shortest digits that read back as the same double.
  char buf[32];
  auto r = std::to_chars(buf, buf + sizeof buf, d);
  out_.append(buf, r.ptr);
  return *this;
}

Writer& Writer::raw(std::string_view v) {
  before_value();
  out_ += v;
  return *this;
}

Writer& Writer::null() {
  before_value();
  out_ += "null";
  return *this;
}

}  // namespace eklow::json
