#include "core/json_value.hpp"

#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "core/json.hpp"

namespace eklow::json {

Value Value::boolean(bool b) {
  Value v;
  v.kind_ = Kind::Bool;
  v.b_ = b;
  return v;
}

Value Value::integer(std::string digits) {
  Value v;
  v.kind_ = Kind::Int;
  v.s_ = digits == "-0" ? "0" : std::move(digits);
  v.d_ = std::strtod(v.s_.c_str(), nullptr);
  return v;
}

Value Value::integer(int64_t i) { return integer(std::to_string(i)); }

Value Value::number(double d) {
  Value v;
  v.kind_ = Kind::Float;
  v.d_ = d;
  return v;
}

Value Value::string(std::string s) {
  Value v;
  v.kind_ = Kind::String;
  v.s_ = std::move(s);
  return v;
}

Value Value::array(std::vector<Value> items) {
  Value v;
  v.kind_ = Kind::Array;
  v.arr_ = std::make_shared<std::vector<Value>>(std::move(items));
  return v;
}

Value Value::object(std::vector<Member> members) {
  Value v;
  v.kind_ = Kind::Object;
  v.obj_ = std::make_shared<std::vector<Member>>();
  for (auto& m : members) v.set(std::move(m.first), std::move(m.second));
  return v;
}

const std::vector<Value>& Value::items() const {
  if (kind_ != Kind::Array) throw std::runtime_error("JSON value is not an array");
  return *arr_;
}

std::vector<Value>& Value::items() {
  if (kind_ != Kind::Array) throw std::runtime_error("JSON value is not an array");
  return *arr_;
}

const std::vector<Value::Member>& Value::members() const {
  if (kind_ != Kind::Object) throw std::runtime_error("JSON value is not an object");
  return *obj_;
}

std::vector<Value::Member>& Value::members() {
  if (kind_ != Kind::Object) throw std::runtime_error("JSON value is not an object");
  return *obj_;
}

const Value* Value::find(std::string_view key) const {
  if (kind_ != Kind::Object) return nullptr;
  for (const auto& [k, v] : *obj_)
    if (k == key) return &v;
  return nullptr;
}

void Value::set(std::string key, Value v) {
  for (auto& [k, old] : members())
    if (k == key) {
      old = std::move(v);
      return;
    }
  obj_->emplace_back(std::move(key), std::move(v));
}

bool Value::truthy() const {
  switch (kind_) {
    case Kind::Null: return false;
    case Kind::Bool: return b_;
    case Kind::Int: return s_ != "0";
    case Kind::Float: return d_ != 0;
    case Kind::String: return !s_.empty();
    case Kind::Array: return !arr_->empty();
    case Kind::Object: return !obj_->empty();
  }
  return false;
}

bool operator==(const Value& a, const Value& b) {
  using K = Value::Kind;
  auto numeric = [](const Value& v) { return v.kind_ == K::Bool || v.kind_ == K::Int || v.kind_ == K::Float; };
  if (numeric(a) && numeric(b)) {
    if (a.kind_ == K::Int && b.kind_ == K::Int) return a.s_ == b.s_;
    auto val = [](const Value& v) { return v.kind_ == K::Bool ? double(v.b_) : v.d_; };
    return val(a) == val(b);
  }
  if (a.kind_ != b.kind_) return false;
  switch (a.kind_) {
    case K::Null: return true;
    case K::String: return a.s_ == b.s_;
    case K::Array: return *a.arr_ == *b.arr_;
    case K::Object: {
      if (a.obj_->size() != b.obj_->size()) return false;
      for (const auto& [k, v] : *a.obj_) {
        const Value* o = b.find(k);
        if (!o || !(v == *o)) return false;
      }
      return true;
    }
    default: return false;
  }
}

namespace {

struct Parser {
  std::string_view s;
  size_t i = 0;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("JSON: ") + what + " at byte " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  bool eat(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c)) fail((std::string("expected '") + c + "'").c_str());
  }
  bool lit(std::string_view w) {
    if (s.substr(i, w.size()) == w) {
      i += w.size();
      return true;
    }
    return false;
  }

  static void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) o += char(0xc0 | cp >> 6), o += char(0x80 | (cp & 0x3f));
    else if (cp < 0x10000) o += char(0xe0 | cp >> 12), o += char(0x80 | (cp >> 6 & 0x3f)), o += char(0x80 | (cp & 0x3f));
    else
      o += char(0xf0 | cp >> 18), o += char(0x80 | (cp >> 12 & 0x3f)), o += char(0x80 | (cp >> 6 & 0x3f)),
          o += char(0x80 | (cp & 0x3f));
  }

  uint32_t hex4() {
    if (i + 4 > s.size()) fail("truncated \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= c - '0';
      else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
      else fail("bad \\u escape");
    }
    return v;
  }

  std::string str() {
    // The opening quote is consumed by the caller.
    std::string o;
    for (;;) {
      if (i >= s.size()) fail("unterminated string");
      unsigned char c = s[i++];
      if (c == '"') return o;
      if (c < 0x20) fail("control character in string");
      if (c != '\\') {
        o += char(c);
        continue;
      }
      if (i >= s.size()) fail("unterminated escape");
      switch (s[i++]) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xd800 && cp < 0xdc00 && s.substr(i, 2) == "\\u") {
            size_t save = i;
            i += 2;
            uint32_t lo = hex4();
            if (lo >= 0xdc00 && lo < 0xe000) cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            else i = save;
          }
          // A lone surrogate cannot be UTF-8; it becomes U+FFFD.
          if (cp >= 0xd800 && cp < 0xe000) cp = 0xfffd;
          put_utf8(o, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }

  Value number() {
    size_t start = i;
    bool is_float = false;
    if (i < s.size() && s[i] == '-') ++i;
    if (i < s.size() && s[i] == '0') ++i;
    else if (i < s.size() && s[i] >= '1' && s[i] <= '9')
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    else fail("bad number");
    if (i < s.size() && s[i] == '.') {
      is_float = true;
      ++i;
      if (i >= s.size() || s[i] < '0' || s[i] > '9') fail("bad fraction");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      is_float = true;
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
      if (i >= s.size() || s[i] < '0' || s[i] > '9') fail("bad exponent");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    std::string lex(s.substr(start, i - start));
    if (!is_float) return Value::integer(lex);
    return Value::number(std::strtod(lex.c_str(), nullptr));
  }

  Value value(int depth) {
    if (depth > kMaxDepth) fail("nested too deeply");
    ws();
    if (i >= s.size()) fail("unexpected end");
    char c = s[i];
    if (c == '{') {
      ++i;
      Value o = Value::object();
      if (eat('}')) return o;
      do {
        ws();
        if (i >= s.size() || s[i] != '"') fail("expected a key");
        ++i;
        std::string k = str();
        expect(':');
        o.set(std::move(k), value(depth + 1));
      } while (eat(','));
      expect('}');
      return o;
    }
    if (c == '[') {
      ++i;
      Value a = Value::array();
      if (eat(']')) return a;
      do a.items().push_back(value(depth + 1));
      while (eat(','));
      expect(']');
      return a;
    }
    if (c == '"') {
      ++i;
      return Value::string(str());
    }
    if (lit("true")) return Value::boolean(true);
    if (lit("false")) return Value::boolean(false);
    if (lit("null")) return Value::null();
    return number();
  }
};

void dump_to(Writer& w, const Value& v) {
  switch (v.kind()) {
    case Value::Kind::Null: w.null(); break;
    case Value::Kind::Bool: w.value(v.as_bool()); break;
    case Value::Kind::Int: w.raw(v.int_digits()); break;
    case Value::Kind::Float: w.value(v.as_double()); break;
    case Value::Kind::String: w.value(v.as_string()); break;
    case Value::Kind::Array:
      w.begin_array();
      for (const auto& x : v.items()) dump_to(w, x);
      w.end_array();
      break;
    case Value::Kind::Object:
      w.begin_object();
      for (const auto& [k, x] : v.members()) {
        w.key(k);
        dump_to(w, x);
      }
      w.end_object();
      break;
  }
}

}  // namespace

Value parse(std::string_view text) {
  Parser p{text};
  Value v = p.value(0);
  p.ws();
  if (p.i != text.size()) p.fail("trailing characters");
  return v;
}

std::string dump(const Value& v) {
  Writer w(false);
  dump_to(w, v);
  return w.str();
}

}  // namespace eklow::json
