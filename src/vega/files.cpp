#include "vega/files.hpp"

#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "core/json_value.hpp"

namespace eklow::vega {

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

// FIPS 180-4.
std::string sha256_hex(std::string_view bytes) {
  static constexpr uint32_t K[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
      0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
      0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
      0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
      0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
      0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  auto block = [&](const unsigned char* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) w[i] = uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | p[4 * i + 3];
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
      const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  };
  const auto* p = reinterpret_cast<const unsigned char*>(bytes.data());
  const size_t n = bytes.size();
  size_t i = 0;
  for (; i + 64 <= n; i += 64) block(p + i);
  unsigned char tail[128] = {};
  const size_t rest = n - i;
  std::memcpy(tail, p + i, rest);
  tail[rest] = 0x80;
  const size_t len = rest < 56 ? 64 : 128;
  const uint64_t bits = uint64_t(n) * 8;
  for (int j = 0; j < 8; ++j) tail[len - 1 - j] = uint8_t(bits >> (8 * j));
  block(tail);
  if (len == 128) block(tail + 64);
  static const char* hex = "0123456789abcdef";
  std::string out;
  for (uint32_t v : h)
    for (int j = 7; j >= 0; --j) out += hex[(v >> (4 * j)) & 15];
  return out;
}

int64_t Array::size() const {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

std::vector<float> Array::floats() const {
  const int64_t n = size();
  std::vector<float> out(static_cast<size_t>(n));
  auto rd = [&](auto tag) {
    using T = decltype(tag);
    if (data.size() < size_t(n) * sizeof(T)) throw std::runtime_error("npy: data shorter than its shape");
    for (int64_t i = 0; i < n; ++i) {
      T v;
      std::memcpy(&v, data.data() + size_t(i) * sizeof(T), sizeof(T));
      out[size_t(i)] = float(v);
    }
  };
  if (dtype == "<f4") rd(float{});
  else if (dtype == "<f8") rd(double{});
  else if (dtype == "<i4") rd(int32_t{});
  else if (dtype == "<i8") rd(int64_t{});
  else if (dtype == "|b1" || dtype == "|u1") rd(uint8_t{});
  else throw std::runtime_error("npy: no float view of dtype " + dtype);
  return out;
}

std::string Array::unicode() const {
  if (dtype.size() < 3 || dtype.substr(0, 2) != "<U" || size() != 1) throw std::runtime_error("npy: not a single unicode string");
  std::string out;
  for (size_t i = 0; i + 4 <= data.size(); i += 4) {
    uint32_t c;
    std::memcpy(&c, data.data() + i, 4);
    if (c == 0) break;
    if (c < 0x80) {
      out += char(c);
    } else if (c < 0x800) {
      out += char(0xc0 | c >> 6), out += char(0x80 | (c & 63));
    } else if (c < 0x10000) {
      out += char(0xe0 | c >> 12), out += char(0x80 | ((c >> 6) & 63)), out += char(0x80 | (c & 63));
    } else {
      out += char(0xf0 | c >> 18), out += char(0x80 | ((c >> 12) & 63)), out += char(0x80 | ((c >> 6) & 63)), out += char(0x80 | (c & 63));
    }
  }
  return out;
}

namespace {

uint64_t le(std::string_view b, size_t at, int bytes) {
  if (at + size_t(bytes) > b.size()) throw std::runtime_error("truncated");
  uint64_t v = 0;
  for (int i = bytes - 1; i >= 0; --i) v = v << 8 | uint8_t(b[at + size_t(i)]);
  return v;
}

// The value of `key` in NumPy's header dict, a Python literal: "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }".
std::string header_field(const std::string& h, const std::string& key) {
  const size_t k = h.find("'" + key + "'");
  if (k == std::string::npos) throw std::runtime_error("npy header without " + key);
  size_t v = h.find(':', k) + 1;
  while (v < h.size() && h[v] == ' ') ++v;
  size_t e = v;
  if (h[v] == '\'') e = h.find('\'', v + 1) + 1;
  else if (h[v] == '(') e = h.find(')', v) + 1;
  else
    while (e < h.size() && h[e] != ',' && h[e] != '}') ++e;
  return h.substr(v, e - v);
}

// Python's repr of a float: the shortest digits that round-trip, positional for exponents in [-4, 16), else scientific.
std::string py_float(double x) {
  if (std::isnan(x)) return "NaN";
  if (std::isinf(x)) return x > 0 ? "Infinity" : "-Infinity";
  if (x == 0) return std::signbit(x) ? "-0.0" : "0.0";
  char buf[64];
  const auto r = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::scientific);
  std::string s(buf, r.ptr);
  std::string sign;
  if (s[0] == '-') sign = "-", s.erase(0, 1);
  const size_t e = s.find('e');
  std::string digits = s.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  const int exp10 = std::stoi(s.substr(e + 1));
  if (exp10 < -4 || exp10 >= 16) {
    std::string m = digits.substr(0, 1) + (digits.size() > 1 ? "." + digits.substr(1) : "");
    char eb[16];
    std::snprintf(eb, sizeof eb, "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
    return sign + m + eb;
  }
  if (exp10 < 0) return sign + "0." + std::string(size_t(-exp10 - 1), '0') + digits;
  std::string ip = digits.substr(0, std::min<size_t>(digits.size(), size_t(exp10 + 1)));
  ip += std::string(size_t(exp10 + 1) - ip.size(), '0');
  const std::string fp = digits.size() > size_t(exp10 + 1) ? digits.substr(size_t(exp10 + 1)) : "0";
  return sign + ip + "." + fp;
}

void py_dump(const json::Value& v, std::string& out, bool sort_keys, bool compact, bool ascii) {
  switch (v.kind()) {
    case json::Value::Kind::Null: out += "null"; return;
    case json::Value::Kind::Bool: out += v.as_bool() ? "true" : "false"; return;
    case json::Value::Kind::Int: out += v.int_digits(); return;
    case json::Value::Kind::Float: out += py_float(v.as_double()); return;
    case json::Value::Kind::String: {
      out += '"';
      const std::string& s = v.as_string();
      for (size_t i = 0; i < s.size();) {
        const unsigned char c = uint8_t(s[i]);
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0xf0) cp = c & 7, len = 4;
        else if (c >= 0xe0) cp = c & 15, len = 3;
        else if (c >= 0xc0) cp = c & 31, len = 2;
        for (size_t k = 1; k < len && i + k < s.size(); ++k) cp = cp << 6 | (uint8_t(s[i + k]) & 63);
        i += len;
        char b[16];
        if (cp == '"') out += "\\\"";
        else if (cp == '\\') out += "\\\\";
        else if (cp == '\n') out += "\\n";
        else if (cp == '\r') out += "\\r";
        else if (cp == '\t') out += "\\t";
        else if (cp == '\b') out += "\\b";
        else if (cp == '\f') out += "\\f";
        else if (cp < 0x20 || (ascii && cp >= 0x7f && cp < 0x10000)) std::snprintf(b, sizeof b, "\\u%04x", cp), out += b;
        else if (ascii && cp >= 0x10000) {
          const uint32_t u = cp - 0x10000;
          std::snprintf(b, sizeof b, "\\u%04x\\u%04x", 0xd800 + (u >> 10), 0xdc00 + (u & 0x3ff));
          out += b;
        } else out.append(s, i - len, len);
      }
      out += '"';
      return;
    }
    case json::Value::Kind::Array: {
      out += '[';
      for (size_t i = 0; i < v.items().size(); ++i) {
        if (i) out += compact ? "," : ", ";
        py_dump(v.items()[i], out, sort_keys, compact, ascii);
      }
      out += ']';
      return;
    }
    case json::Value::Kind::Object: {
      std::vector<const json::Value::Member*> ms;
      for (const auto& m : v.members()) ms.push_back(&m);
      if (sort_keys) std::sort(ms.begin(), ms.end(), [](auto* a, auto* b) { return a->first < b->first; });
      out += '{';
      for (size_t i = 0; i < ms.size(); ++i) {
        if (i) out += compact ? "," : ", ";
        py_dump(json::Value::string(ms[i]->first), out, sort_keys, compact, ascii);
        out += compact ? ":" : ": ";
        py_dump(ms[i]->second, out, sort_keys, compact, ascii);
      }
      out += '}';
      return;
    }
  }
}

}  // namespace

Array parse_npy(std::string_view b, const std::string& what) {
  try {
    if (b.size() < 10 || b.substr(0, 6) != std::string_view("\x93NUMPY", 6)) throw std::runtime_error("not an .npy");
    const int major = uint8_t(b[6]);
    const size_t hl = major == 1 ? le(b, 8, 2) : le(b, 8, 4), h0 = major == 1 ? 10 : 12;
    const std::string h(b.substr(h0, hl));
    Array a;
    const std::string descr = header_field(h, "descr");
    a.dtype = descr.substr(1, descr.size() - 2);
    const bool fortran = header_field(h, "fortran_order") == "True";
    const std::string shape = header_field(h, "shape");
    for (size_t i = 1; i < shape.size();) {
      while (i < shape.size() && (shape[i] == ' ' || shape[i] == ',')) ++i;
      if (i >= shape.size() || shape[i] == ')') break;
      size_t j = i;
      while (j < shape.size() && std::isdigit(uint8_t(shape[j]))) ++j;
      a.shape.push_back(std::stoll(shape.substr(i, j - i)));
      i = j;
    }
    a.data = std::string(b.substr(h0 + hl));
    // A transposed array (a TTT head's projection, Vt[:k].T) is saved in Fortran order: put it in C order.
    if (fortran && a.shape.size() > 1) {
      const size_t width = a.dtype.size() > 2 ? size_t(std::stoi(a.dtype.substr(2))) * (a.dtype[1] == 'U' ? 4 : 1) : 1;
      const int64_t n = a.size();
      if (a.data.size() < size_t(n) * width) throw std::runtime_error("data shorter than its shape");
      std::string c(size_t(n) * width, '\0');
      const size_t nd = a.shape.size();
      std::vector<int64_t> idx(nd, 0);
      for (int64_t i = 0; i < n; ++i) {  // i walks C order; its Fortran offset has the first index fastest
        int64_t f = 0, stride = 1;
        for (size_t d = 0; d < nd; ++d) f += idx[d] * stride, stride *= a.shape[d];
        std::memcpy(c.data() + size_t(i) * width, a.data.data() + size_t(f) * width, width);
        for (size_t d = nd; d-- > 0;)
          if (++idx[d] < a.shape[d]) break;
          else idx[d] = 0;
      }
      a.data = std::move(c);
    }
    return a;
  } catch (const std::exception& e) {
    throw std::runtime_error(what + ": " + e.what());
  }
}

std::map<std::string, Array> parse_npz(std::string_view b, const std::string& what) {
  try {
    // The end of central directory record, then its zip64 form when the counts or offsets overflowed.
    size_t eocd = std::string_view::npos;
    for (size_t i = b.size() >= 22 ? b.size() - 22 : 0;; --i) {
      if (le(b, i, 4) == 0x06054b50) {
        eocd = i;
        break;
      }
      if (i == 0 || b.size() - i > 65557) break;
    }
    if (eocd == std::string_view::npos) throw std::runtime_error("not a zip");
    uint64_t entries = le(b, eocd + 10, 2), cd = le(b, eocd + 16, 4);
    if (entries == 0xffff || cd == 0xffffffff) {
      if (eocd < 20 || le(b, eocd - 20, 4) != 0x07064b50) throw std::runtime_error("zip64 locator missing");
      const size_t z = size_t(le(b, eocd - 20 + 8, 8));
      if (le(b, z, 4) != 0x06064b50) throw std::runtime_error("zip64 record missing");
      entries = le(b, z + 32, 8), cd = le(b, z + 48, 8);
    }
    std::map<std::string, Array> out;
    size_t p = size_t(cd);
    for (uint64_t e = 0; e < entries; ++e) {
      if (le(b, p, 4) != 0x02014b50) throw std::runtime_error("bad central directory");
      const uint64_t method = le(b, p + 10, 2), nl = le(b, p + 28, 2), el = le(b, p + 30, 2), cl = le(b, p + 32, 2);
      uint64_t csize = le(b, p + 20, 4), usize = le(b, p + 24, 4), off = le(b, p + 42, 4);
      const std::string name(b.substr(p + 46, nl));
      for (size_t x = p + 46 + nl; x + 4 <= p + 46 + nl + el;) {
        const uint64_t id = le(b, x, 2), sz = le(b, x + 2, 2);
        if (id == 1) {
          size_t y = x + 4;
          if (usize == 0xffffffff) usize = le(b, y, 8), y += 8;
          if (csize == 0xffffffff) csize = le(b, y, 8), y += 8;
          if (off == 0xffffffff) off = le(b, y, 8), y += 8;
        }
        x += 4 + sz;
      }
      if (method != 0 || csize != usize) throw std::runtime_error(name + " is compressed (np.savez_compressed); save with np.savez");
      if (le(b, size_t(off), 4) != 0x04034b50) throw std::runtime_error("bad local header for " + name);
      const size_t data = size_t(off) + 30 + le(b, size_t(off) + 26, 2) + le(b, size_t(off) + 28, 2);
      if (data + usize > b.size()) throw std::runtime_error(name + " runs past the end");
      std::string key = name;
      if (key.size() > 4 && key.substr(key.size() - 4) == ".npy") key.resize(key.size() - 4);
      out.emplace(key, parse_npy(b.substr(data, size_t(usize)), what + ":" + name));
      p += 46 + nl + el + cl;
    }
    return out;
  } catch (const std::exception& e) {
    throw std::runtime_error(what + ": " + e.what());
  }
}

void write_npy(const std::string& path, const std::vector<int64_t>& shape, const float* data) {
  std::string sh = "(";
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) sh += (i ? ", " : "") + std::to_string(shape[i]), n *= shape[i];
  sh += shape.size() == 1 ? ",)" : ")";
  std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': " + sh + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write("\x93NUMPY\x01\x00", 8);
  const uint16_t hl = uint16_t(h.size());
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(h.data(), std::streamsize(h.size()));
  f.write(reinterpret_cast<const char*>(data), std::streamsize(n * 4));
}

std::map<std::string, Tensor> read_safetensors(const std::string& path) {
  const std::string b = read_file(path);
  if (b.size() < 8) throw std::runtime_error(path + ": not safetensors");
  const uint64_t n = le(b, 0, 8);
  if (8 + n > b.size()) throw std::runtime_error(path + ": header runs past the end");
  const json::Value h = json::parse(std::string_view(b).substr(8, n));
  std::map<std::string, Tensor> out;
  for (const auto& [name, v] : h.members()) {
    if (name == "__metadata__") continue;
    const std::string& dt = v.find("dtype")->as_string();
    Tensor t;
    int64_t count = 1;
    for (const auto& d : v.find("shape")->items()) t.shape.push_back(std::stoll(d.int_digits())), count *= t.shape.back();
    const auto& offs = v.find("data_offsets")->items();
    const uint64_t a = std::stoull(offs.at(0).int_digits()), e = std::stoull(offs.at(1).int_digits());
    const size_t width = dt == "F32" ? 4 : dt == "BF16" ? 2 : 0;
    if (!width) throw std::runtime_error(path + ": " + name + " is " + dt + " (F32 or BF16 only)");
    if (e - a != uint64_t(count) * width || 8 + n + e > b.size()) throw std::runtime_error(path + ": bad offsets for " + name);
    t.data.resize(static_cast<size_t>(count));
    const char* src = b.data() + 8 + n + a;
    for (int64_t i = 0; i < count; ++i) {
      uint32_t u = 0;
      if (width == 4) std::memcpy(&u, src + 4 * i, 4);
      else u = uint32_t(uint8_t(src[2 * i]) | uint8_t(src[2 * i + 1]) << 8) << 16;
      std::memcpy(&t.data[size_t(i)], &u, 4);
    }
    out.emplace(name, std::move(t));
  }
  return out;
}

std::string python_json(const json::Value& v, bool sort_keys, bool compact, bool ensure_ascii) {
  std::string out;
  py_dump(v, out, sort_keys, compact, ensure_ascii);
  return out;
}

}  // namespace eklow::vega
