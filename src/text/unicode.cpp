#include "text/unicode.hpp"

#include <algorithm>
#include <vector>

#include "text/unicode_tables.hpp"

namespace eklow::text::unicode {

namespace {

constexpr uint8_t kUnsafeBit = 16;

std::vector<uint8_t> build_class_table() {
  std::vector<uint8_t> t(0x110000, 0);
  for (const auto& r : kClassRanges) std::fill(t.begin() + r.lo, t.begin() + r.hi + 1, r.v);
  for (const auto& r : kUnsafeRanges)
    for (uint32_t cp = r.lo; cp <= r.hi; ++cp) t[cp] |= kUnsafeBit;
  return t;
}

const std::vector<uint8_t> g_table = build_class_table();

uint8_t ccc(uint32_t cp) {
  if (cp < 0x300) return 0;
  auto it = std::upper_bound(std::begin(kCccRanges), std::end(kCccRanges), cp, [](uint32_t c, const CccRange& r) { return c < r.lo; });
  if (it == std::begin(kCccRanges)) return 0;
  --it;
  return cp <= it->hi ? it->v : 0;
}

constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

void decompose(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= kSBase && cp < kSBase + kSCount) {
    const uint32_t s = cp - kSBase;
    out.push_back(kLBase + s / kNCount);
    out.push_back(kVBase + (s % kNCount) / kTCount);
    if (s % kTCount) out.push_back(kTBase + s % kTCount);
    return;
  }
  if (cp >= 0xA0) {
    auto it = std::lower_bound(std::begin(kDecompCp), std::end(kDecompCp), cp);
    if (it != std::end(kDecompCp) && *it == cp) {
      const size_t i = size_t(it - std::begin(kDecompCp));
      out.insert(out.end(), kDecompPool + kDecompOff[i], kDecompPool + kDecompOff[i + 1]);
      return;
    }
  }
  out.push_back(cp);
}

// The primary composite of a starter and a following character, or 0.
uint32_t compose_pair(uint32_t a, uint32_t b) {
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount)
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount) return a + (b - kTBase);
  auto it = std::lower_bound(std::begin(kPairs), std::end(kPairs), Pair{a, b, 0},
                             [](const Pair& x, const Pair& y) { return x.first != y.first ? x.first < y.first : x.second < y.second; });
  return it != std::end(kPairs) && it->first == a && it->second == b ? it->composite : 0;
}

}  // namespace

const uint8_t* const kClassTable = g_table.data();

void append_utf8(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out += char(cp);
  } else if (cp < 0x800) {
    out += char(0xC0 | cp >> 6);
    out += char(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += char(0xE0 | cp >> 12);
    out += char(0x80 | (cp >> 6 & 0x3F));
    out += char(0x80 | (cp & 0x3F));
  } else {
    out += char(0xF0 | cp >> 18);
    out += char(0x80 | (cp >> 12 & 0x3F));
    out += char(0x80 | (cp >> 6 & 0x3F));
    out += char(0x80 | (cp & 0x3F));
  }
}

namespace {

// Length of the well-formed sequence at s[i], or 0; `bad` gets the length of the maximal ill-formed subpart then.
size_t seq_len(std::string_view s, size_t i, size_t& bad) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  bad = 1;
  if (c < 0x80) return 1;
  size_t need;
  unsigned char lo = 0x80, hi = 0xBF;
  if (c >= 0xC2 && c <= 0xDF) need = 2;
  else if (c >= 0xE0 && c <= 0xEF) need = 3, lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
  else if (c >= 0xF0 && c <= 0xF4) need = 4, lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
  else return 0;
  for (size_t k = 1; k < need; ++k) {
    if (i + k >= s.size()) return bad = k, 0;
    const unsigned char d = static_cast<unsigned char>(s[i + k]);
    if (d < lo || d > hi) return bad = k, 0;
    lo = 0x80, hi = 0xBF;
  }
  return need;
}

}  // namespace

bool valid_utf8(std::string_view s) {
  size_t bad;
  for (size_t i = 0; i < s.size();) {
    if (static_cast<unsigned char>(s[i]) < 0x80) {
      ++i;
      continue;
    }
    const size_t n = seq_len(s, i, bad);
    if (!n) return false;
    i += n;
  }
  return true;
}

std::string lossy_utf8(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t bad;
  for (size_t i = 0; i < s.size();) {
    const size_t n = seq_len(s, i, bad);
    if (n) {
      out.append(s.substr(i, n));
      i += n;
    } else {
      out += "\xEF\xBF\xBD";
      i += bad;
    }
  }
  return out;
}

std::string nfc(std::string_view s) {
  // Strings in which nothing is flagged are in NFC already; below U+0300 nothing is.
  bool plain = true;
  for (size_t i = 0; i < s.size() && plain;) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0xCC) {  // ASCII, and Latin-1 and the rest of U+0080-U+02FF (lead bytes C2-CB)
      ++i;
      continue;
    }
    uint32_t cp;
    i += decode(s.data() + i, s.size() - i, cp);
    plain = !(g_table[cp] & kUnsafeBit);
  }
  if (plain) return std::string(s);

  std::vector<uint32_t> cps;
  cps.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    uint32_t cp;
    i += decode(s.data() + i, s.size() - i, cp);
    decompose(cp, cps);
  }
  // Canonical ordering: runs of non-starters sorted by combining class, stably.
  for (size_t i = 0; i < cps.size();) {
    if (!ccc(cps[i])) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < cps.size() && ccc(cps[j])) ++j;
    std::stable_sort(cps.begin() + long(i), cps.begin() + long(j), [](uint32_t a, uint32_t b) { return ccc(a) < ccc(b); });
    i = j;
  }
  // Composition: each character combines with the last starter unless something of the same or a higher class
  // (or another starter) sits between them.
  std::vector<uint32_t> out;
  out.reserve(cps.size());
  long starter = -1;
  uint8_t last = 0;
  for (uint32_t c : cps) {
    const uint8_t cc = ccc(c);
    if (starter >= 0) {
      const bool between = long(out.size()) - 1 > starter;
      if (!between || (last != 0 && last < cc)) {
        if (uint32_t comp = compose_pair(out[size_t(starter)], c)) {
          out[size_t(starter)] = comp;
          continue;
        }
      }
    }
    if (cc == 0) starter = long(out.size()), last = 0;
    else last = cc;
    out.push_back(c);
  }
  std::string r;
  r.reserve(s.size());
  for (uint32_t c : out) append_utf8(r, c);
  return r;
}

}  // namespace eklow::text::unicode
