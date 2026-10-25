#include "text/pyfmt.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace eklow::text::py {

namespace {

// Decodes one code point; a malformed byte comes back as itself with length 1.
uint32_t next_cp(std::string_view s, size_t& i) {
  unsigned char c = s[i];
  int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
  if (n == 0 || i + n > s.size()) {
    ++i;
    return c;
  }
  uint32_t cp = n == 1 ? c : n == 2 ? c & 0x1f : n == 3 ? c & 0x0f : c & 0x07;
  for (int k = 1; k < n; ++k) cp = cp << 6 | (static_cast<unsigned char>(s[i + k]) & 0x3f);
  i += n;
  return cp;
}

void put(std::string& o, uint32_t cp) {
  if (cp < 0x80) o += char(cp);
  else if (cp < 0x800) o += char(0xc0 | cp >> 6), o += char(0x80 | (cp & 0x3f));
  else if (cp < 0x10000) o += char(0xe0 | cp >> 12), o += char(0x80 | (cp >> 6 & 0x3f)), o += char(0x80 | (cp & 0x3f));
  else
    o += char(0xf0 | cp >> 18), o += char(0x80 | (cp >> 12 & 0x3f)), o += char(0x80 | (cp >> 6 & 0x3f)),
        o += char(0x80 | (cp & 0x3f));
}

bool is_space(uint32_t c) {
  return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

uint32_t to_upper(uint32_t c) {
  if (c >= 'a' && c <= 'z') return c - 32;
  if (c == 0xb5) return 0x39c;                                         // micro sign -> Greek capital mu
  if ((c >= 0xe0 && c <= 0xfe && c != 0xf7)) return c - 32;            // Latin-1
  if (c == 0xff) return 0x178;
  if (c >= 0x100 && c <= 0x17f && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17f) {
    // Latin Extended-A pairs capital and small letters; capitals sit on even code points except
    // in 0x139-0x148 and 0x179-0x17e, where they sit on odd ones.
    bool odd_upper = (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17e);
    if (odd_upper ? (c % 2 == 0) : (c % 2 == 1)) return c - 1;
    return c;
  }
  if (c == 0x131) return 'I';
  if (c == 0x17f) return 'S';
  if (c >= 0x3b1 && c <= 0x3c9 && c != 0x3c2) return c - 32;           // Greek
  if (c == 0x3c2) return 0x3a3;
  if (c >= 0x430 && c <= 0x44f) return c - 32;                         // Cyrillic
  if (c >= 0x450 && c <= 0x45f) return c - 80;
  return c;
}

uint32_t to_lower(uint32_t c) {
  if (c >= 'A' && c <= 'Z') return c + 32;
  if (c >= 0xc0 && c <= 0xde && c != 0xd7) return c + 32;
  if (c == 0x178) return 0xff;
  if (c >= 0x100 && c <= 0x17f && c != 0x130 && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17f) {
    bool odd_upper = (c >= 0x139 && c <= 0x148) || (c >= 0x179 && c <= 0x17e);
    if (odd_upper ? (c % 2 == 1) : (c % 2 == 0)) return c + 1;
    return c;
  }
  if (c >= 0x391 && c <= 0x3a9 && c != 0x3a2) return c + 32;
  if (c >= 0x410 && c <= 0x42f) return c + 32;
  if (c >= 0x400 && c <= 0x40f) return c + 80;
  return c;
}

// Python's str.isprintable() for the code points JSON strings commonly carry: control characters,
// the C1 block, separators other than space and the format characters are not printable.
bool printable(uint32_t c) {
  if (c < 0x20 || c == 0x7f || (c >= 0x80 && c <= 0xa0) || c == 0xad) return false;
  if (c == 0x1680 || (c >= 0x2000 && c <= 0x200f) || (c >= 0x2028 && c <= 0x202f) || (c >= 0x205f && c <= 0x2064) ||
      (c >= 0x2066 && c <= 0x206f) || c == 0x3000 || c == 0xfeff || (c >= 0xfff9 && c <= 0xfffb))
    return false;
  if ((c >= 0xd800 && c <= 0xdfff) || (c >= 0xe000 && c <= 0xf8ff) || c >= 0xf0000) return false;
  return true;
}

}  // namespace

std::string float_repr(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d < 0 ? "-inf" : "inf";
  if (d == 0) return std::signbit(d) ? "-0.0" : "0.0";
  // Shortest round-trip digits, as d.ddde+X.
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::scientific);
  std::string sci(buf, r.ptr);
  bool neg = sci[0] == '-';
  if (neg) sci.erase(0, 1);
  size_t e = sci.find('e');
  std::string digits = sci.substr(0, 1) + (e > 2 ? sci.substr(2, e - 2) : "");
  int exp = std::atoi(sci.c_str() + e + 1);
  std::string out = neg ? "-" : "";
  // Python's repr is positional for 1e-4 <= |x| < 1e16, in exponent form otherwise.
  if (exp >= -4 && exp < 16) {
    if (exp < 0) {
      out += "0." + std::string(size_t(-exp - 1), '0') + digits;
    } else if (size_t(exp) + 1 >= digits.size()) {
      out += digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
    } else {
      out += digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
    }
    return out;
  }
  out += digits.substr(0, 1);
  if (digits.size() > 1) out += "." + digits.substr(1);
  char eb[16];
  std::snprintf(eb, sizeof eb, "e%c%02d", exp < 0 ? '-' : '+', exp < 0 ? -exp : exp);
  return out + eb;
}

std::string repr_string(std::string_view s) {
  bool has_single = s.find('\'') != std::string_view::npos, has_double = s.find('"') != std::string_view::npos;
  char q = has_single && !has_double ? '"' : '\'';
  std::string o(1, q);
  for (size_t i = 0; i < s.size();) {
    uint32_t c = next_cp(s, i);
    if (c == uint32_t(q) || c == '\\') {
      o += '\\';
      o += char(c);
    } else if (c == '\t') {
      o += "\\t";
    } else if (c == '\n') {
      o += "\\n";
    } else if (c == '\r') {
      o += "\\r";
    } else if (printable(c)) {
      put(o, c);
    } else {
      char b[16];
      if (c < 0x100) std::snprintf(b, sizeof b, "\\x%02x", c);
      else if (c < 0x10000) std::snprintf(b, sizeof b, "\\u%04x", c);
      else std::snprintf(b, sizeof b, "\\U%08x", c);
      o += b;
    }
  }
  return o + q;
}

std::string repr(const json::Value& v) {
  if (v.is_string()) return repr_string(v.as_string());
  return str(v);
}

std::string str(const json::Value& v) {
  switch (v.kind()) {
    case json::Value::Kind::Null: return "None";
    case json::Value::Kind::Bool: return v.as_bool() ? "True" : "False";
    case json::Value::Kind::Int: return v.int_digits();
    case json::Value::Kind::Float: return float_repr(v.as_double());
    case json::Value::Kind::String: return v.as_string();
    case json::Value::Kind::Array: {
      std::string o = "[";
      bool first = true;
      for (const auto& x : v.items()) {
        if (!first) o += ", ";
        first = false;
        o += repr(x);
      }
      return o + "]";
    }
    case json::Value::Kind::Object: {
      std::string o = "{";
      bool first = true;
      for (const auto& [k, x] : v.members()) {
        if (!first) o += ", ";
        first = false;
        o += repr_string(k) + ": " + repr(x);
      }
      return o + "}";
    }
  }
  return "";
}

std::string strip(std::string_view s) {
  size_t b = 0, e = s.size();
  while (b < e) {
    size_t j = b;
    if (!is_space(next_cp(s, j))) break;
    b = j;
  }
  while (e > b) {
    size_t j = e - 1;
    while (j > b && (static_cast<unsigned char>(s[j]) & 0xc0) == 0x80) --j;
    size_t k = j;
    if (!is_space(next_cp(s, k))) break;
    e = j;
  }
  return std::string(s.substr(b, e - b));
}

std::string upper(std::string_view s) {
  std::string o;
  for (size_t i = 0; i < s.size();) {
    uint32_t c = next_cp(s, i);
    if (c == 0xdf) {
      o += "SS";  // German sharp s
      continue;
    }
    put(o, to_upper(c));
  }
  return o;
}

std::string lower(std::string_view s) {
  std::string o;
  for (size_t i = 0; i < s.size();) {
    uint32_t c = next_cp(s, i);
    if (c == 0x130) {
      o += "i\xcc\x87";  // dotted capital I lowers to i + combining dot above
      continue;
    }
    put(o, to_lower(c));
  }
  return o;
}

}  // namespace eklow::text::py
