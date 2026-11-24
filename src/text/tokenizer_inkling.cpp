#include "text/tokenizer_inkling.hpp"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

#include "model/gguf.hpp"
#include "text/unicode.hpp"
#include "text/unicode_case_tables.hpp"

namespace eklow::text {

namespace {

// GPT-2's bytes_to_unicode: printable bytes stand for themselves, the rest for U+0100 and up in order.
std::array<uint32_t, 256> byte_to_char() {
  std::array<uint32_t, 256> m{};
  uint32_t next = 256;
  for (uint32_t b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    m[b] = printable ? b : next++;
  }
  return m;
}

uint64_t pair_key(int32_t a, int32_t b) { return uint64_t(uint32_t(a)) << 32 | uint32_t(b); }
constexpr uint64_t kEmpty = ~0ull;
constexpr size_t npos = std::string_view::npos;

// Bits of a code point: unicode.hpp's classes (\p{L} 1, \p{M} 2, \p{N} 4, \s 8) and, for letters, the case classes the
// regex tells apart: kUpper for Lu Lt Lm Lo (what [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] takes of the letters), kLower for Ll Lm Lo.
constexpr uint8_t kUpper = 16, kLower = 32;
// Built on first use: unicode.cpp's class table is built before main in another translation unit, in no fixed order.
std::vector<uint8_t> build_bits() {
  std::vector<uint8_t> t(0x110000, 0);
  for (const auto& r : unicode::kCaseRanges) {
    const uint8_t v = r.v == 1 ? kUpper : r.v == 2 ? kLower : (kUpper | kLower);
    std::fill(t.begin() + r.lo, t.begin() + r.hi + 1, v);
  }
  for (uint32_t cp = 0; cp < 0x110000; ++cp) t[cp] |= unicode::classes(cp);
  return t;
}
// The table, built on the first call.
const uint8_t* bit_table() {
  static const std::vector<uint8_t> t = build_bits();
  return t.data();
}

// [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and [\p{Ll}\p{Lm}\p{Lo}\p{M}]
inline bool in_upper(uint8_t k) { return (k & (kUpper | unicode::kMark)) != 0; }
inline bool in_lower(uint8_t k) { return (k & (kLower | unicode::kMark)) != 0; }

// (?i:'s|'t|'re|'ve|'m|'ll|'d) after the apostrophe at s[i-1]: the end of the match, or 0. Case-insensitive the way
// Oniguruma folds: ASCII letters, and U+017F (long s) is an s.
size_t contraction(const char* s, size_t n, size_t i) {
  auto at = [&](size_t& p, char letter) {
    if (p >= n) return false;
    uint32_t c;
    const size_t l = unicode::decode(s + p, n - p, c);
    if (c == uint32_t(letter) || c == uint32_t(letter - 32) || (letter == 's' && c == 0x17F)) return p += l, true;
    return false;
  };
  size_t p;
  p = i;
  if (at(p, 's')) return p;
  p = i;
  if (at(p, 't')) return p;
  p = i;
  if (at(p, 'r') && at(p, 'e')) return p;
  p = i;
  if (at(p, 'v') && at(p, 'e')) return p;
  p = i;
  if (at(p, 'm')) return p;
  p = i;
  if (at(p, 'l') && at(p, 'l')) return p;
  p = i;
  if (at(p, 'd')) return p;
  return 0;
}

// The optional contraction ending a word that ends at e.
inline size_t with_contraction(const char* s, size_t n, size_t e) {
  if (e < n && s[e] == '\'')
    if (size_t c = contraction(s, n, e + 1)) return c;
  return e;
}

// X*Y+(?i:contraction)? at p, with X = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] and Y = [\p{Ll}\p{Lm}\p{Lo}\p{M}]; npos if it does not match.
// X* takes its whole run; Y+ needs a character where it stops, else X* gives characters back one at a time until one is
// in Y (a Lm, Lo or mark), which Y+ takes alone: what backtracking finds.
size_t word_lower(const uint8_t* B, const char* s, size_t n, size_t p) {
  size_t q = p, last_y = npos, last_y_end = 0;
  while (q < n) {
    uint32_t c;
    const size_t l = unicode::decode(s + q, n - q, c);
    const uint8_t k = B[c];
    if (!in_upper(k)) break;
    if (in_lower(k)) last_y = q, last_y_end = q + l;
    q += l;
  }
  size_t e;
  uint32_t c;
  if (q < n && (unicode::decode(s + q, n - q, c), in_lower(B[c]))) {
    e = q;
    while (e < n) {
      const size_t l = unicode::decode(s + e, n - e, c);
      if (!in_lower(B[c])) break;
      e += l;
    }
  } else if (last_y != npos) {
    e = last_y_end;
  } else {
    return npos;
  }
  return with_contraction(s, n, e);
}

// X+Y*(?i:contraction)? at p.
size_t word_upper(const uint8_t* B, const char* s, size_t n, size_t p) {
  size_t q = p;
  uint32_t c;
  while (q < n) {
    const size_t l = unicode::decode(s + q, n - q, c);
    if (!in_upper(B[c])) break;
    q += l;
  }
  if (q == p) return npos;
  while (q < n) {
    const size_t l = unicode::decode(s + q, n - q, c);
    if (!in_lower(B[c])) break;
    q += l;
  }
  return with_contraction(s, n, q);
}

// The pre-token starting at s[i] (valid UTF-8, i < n): its end.
size_t next_piece(const char* s, size_t n, size_t i) {
  const uint8_t* B = bit_table();
  uint32_t c0;
  const size_t l0 = unicode::decode(s + i, n - i, c0);
  const uint8_t k0 = B[c0];
  // The two word alternatives, each with its optional first character [^\r\n\p{L}\p{N}] taken first, then not.
  const bool can_prefix = c0 != '\r' && c0 != '\n' && !(k0 & (unicode::kLetter | unicode::kNumber));
  for (auto word : {word_lower, word_upper}) {
    if (can_prefix)
      if (size_t e = word(B, s, n, i + l0); e != npos) return e;
    if (size_t e = word(B, s, n, i); e != npos) return e;
  }
  // \p{N}{1,3}
  if (k0 & unicode::kNumber) {
    size_t p = i + l0;
    for (int k = 1; k < 3 && p < n; ++k) {
      uint32_t c;
      const size_t l = unicode::decode(s + p, n - p, c);
      if (!(B[c] & unicode::kNumber)) break;
      p += l;
    }
    return p;
  }
  //  ?[^\s\p{L}\p{N}]+[\r\n/]*
  {
    size_t p = c0 == ' ' ? i + 1 : i, q = p;
    while (q < n) {
      uint32_t c;
      const size_t l = unicode::decode(s + q, n - q, c);
      if (B[c] & (unicode::kLetter | unicode::kNumber | unicode::kSpace)) break;
      q += l;
    }
    if (q > p) {
      while (q < n && (s[q] == '\r' || s[q] == '\n' || s[q] == '/')) ++q;
      return q;
    }
  }
  // \s*[\r\n]+ | \s+(?!\S) | \s+ : over the run of white space that starts here (the first character is one, since
  // everything else matched above). The first alternative takes the run up to its last line break; the second all of
  // it at the end of the text, else all but its last character; the third a lone character.
  size_t j = i, last_break = 0, last_len = 0, count = 0;
  while (j < n) {
    uint32_t c;
    const size_t l = unicode::decode(s + j, n - j, c);
    if (!(B[c] & unicode::kSpace)) break;
    j += l;
    last_len = l;
    ++count;
    if (c == '\r' || c == '\n') last_break = j;
  }
  if (last_break) return last_break;
  if (j == n || count < 2) return j;
  return j - last_len;
}

}  // namespace

InklingTokenizer InklingTokenizer::from_gguf(const gguf::File& f) {
  const auto* m = f.find("tokenizer.ggml.model");
  const auto* pre = f.find("tokenizer.ggml.pre");
  if (!m || m->as_string() != "gpt2" || !pre || pre->as_string() != "inkling2")
    throw std::runtime_error(f.path() + ": not an Inkling 2 tokenizer (gpt2 model with the inkling2 pre-tokenizer)");
  InklingTokenizer t;
  const auto& toks = f.at("tokenizer.ggml.tokens").as_array();
  const auto& types = f.at("tokenizer.ggml.token_type").as_array();
  if (types.size() != toks.size()) throw std::runtime_error(f.path() + ": token and token type counts differ");
  const size_t n = toks.size();
  if (n != size_t(inkling::kLastAdded) + 1) throw std::runtime_error(f.path() + ": expected 200,058 tokens");
  const auto b2c = byte_to_char();
  std::unordered_map<uint32_t, uint8_t> c2b;
  for (uint32_t b = 0; b < 256; ++b) c2b.emplace(b2c[b], uint8_t(b));

  t.pieces_.reserve(n);
  t.bytes_.resize(n);
  t.byte_ids_.fill(-1);
  t.ids_.reserve(n);
  t.by_bytes_.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const std::string& p = toks[i].as_string();
    t.pieces_.push_back(p);
    if (!t.ids_.emplace(p, int32_t(i)).second) throw std::runtime_error(f.path() + ": duplicate token '" + p + "'");
    const int64_t ty = types[i].as_int();
    if (int32_t(i) >= inkling::kFirstAdded) {
      // The 60 added tokens: control (3), or unused (5) for the placeholders.
      if (ty != 3 && ty != 4 && ty != 5)
        throw std::runtime_error(f.path() + ": token " + std::to_string(i) + " is not a control token");
      if (p.size() < 5 || p.compare(0, 2, "<|") != 0 || p.compare(p.size() - 2, 2, "|>") != 0)
        throw std::runtime_error(f.path() + ": added token " + std::to_string(i) + " is not <|...|>");
      t.bytes_[i] = p;
      t.added_.emplace(p, int32_t(i));
      continue;
    }
    if (ty != 1) throw std::runtime_error(f.path() + ": unexpected token type " + std::to_string(ty) + " at " + std::to_string(i));
    std::string& raw = t.bytes_[i];
    for (size_t k = 0; k < p.size();) {
      uint32_t cp;
      k += unicode::decode(p.data() + k, p.size() - k, cp);
      auto it = c2b.find(cp);
      if (it == c2b.end()) throw std::runtime_error(f.path() + ": token " + std::to_string(i) + " is not a byte-level piece");
      raw += char(it->second);
    }
    t.by_bytes_.emplace(raw, int32_t(i));
    if (raw.size() == 1) t.byte_ids_[uint8_t(raw[0])] = int32_t(i);
  }
  for (int32_t b : t.byte_ids_)
    if (b < 0) throw std::runtime_error(f.path() + ": the 256 single-byte tokens are not all present");
  for (const char* name : {"<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>", "<|content_text|>",
                           "<|content_thinking|>", "<|end_message|>", "<|content_invoke_tool_json|>", "<|content_model_end_sampling|>"})
    if (t.id(name) < 0) throw std::runtime_error(f.path() + ": no " + name);
  if (t.id("<|message_user|>") != inkling::kMessageUser || t.id("<|content_invoke_tool_json|>") != inkling::kContentInvokeToolJson)
    throw std::runtime_error(f.path() + ": the added tokens are not where Inkling 2's are");

  // Merges are "left right" in the GGUF's spelling, which holds no ASCII space (a space is U+0120).
  const auto& merges = f.at("tokenizer.ggml.merges").as_array();
  unsigned nbits = 4;
  while ((size_t(1) << nbits) < merges.size() * 4) ++nbits;
  t.merge_shift_ = 64 - nbits;
  t.merge_keys_.assign(size_t(1) << nbits, kEmpty);
  t.merge_vals_.assign(size_t(1) << nbits, 0);
  for (size_t r = 0; r < merges.size(); ++r) {
    const std::string& s = merges[r].as_string();
    const size_t sp = s.find(' ');
    if (sp == std::string::npos || sp == 0) throw std::runtime_error(f.path() + ": malformed merge '" + s + "'");
    const int32_t a = t.id(std::string_view(s).substr(0, sp)), b = t.id(std::string_view(s).substr(sp + 1));
    const int32_t c = t.id(s.substr(0, sp) + s.substr(sp + 1));
    if (a < 0 || b < 0 || c < 0) throw std::runtime_error(f.path() + ": merge '" + s + "' names an unknown token");
    const uint64_t key = pair_key(a, b);
    size_t h = size_t((key * 0x9E3779B97F4A7C15ull) >> t.merge_shift_);
    while (t.merge_keys_[h] != kEmpty && t.merge_keys_[h] != key) h = (h + 1) & (t.merge_keys_.size() - 1);
    if (t.merge_keys_[h] == key) continue;  // a repeated pair keeps its first (lowest) rank
    t.merge_keys_[h] = key;
    t.merge_vals_[h] = uint64_t(r) << 32 | uint32_t(c);
  }
  t.eos_ = int32_t(f.at("tokenizer.ggml.eos_token_id").as_int());
  return t;
}

int32_t InklingTokenizer::id(std::string_view piece) const {
  auto it = ids_.find(std::string(piece));
  return it == ids_.end() ? -1 : it->second;
}

uint64_t InklingTokenizer::merge(int32_t a, int32_t b) const {
  const uint64_t key = pair_key(a, b);
  size_t h = size_t((key * 0x9E3779B97F4A7C15ull) >> merge_shift_);
  const size_t mask = merge_keys_.size() - 1;
  for (;; h = (h + 1) & mask) {
    if (merge_keys_[h] == key) return merge_vals_[h];
    if (merge_keys_[h] == kEmpty) return kEmpty;
  }
}

void InklingTokenizer::bpe(const unsigned char* w, size_t n, std::vector<int32_t>& out) const {
  if (n == 1) {
    out.push_back(byte_ids_[w[0]]);
    return;
  }
  // ignore_merges: a piece that is a token is that token.
  if (auto it = by_bytes_.find(std::string(reinterpret_cast<const char*>(w), n)); it != by_bytes_.end()) {
    out.push_back(it->second);
    return;
  }
  if (n <= 48) {
    // Short words, which nearly all are: the ranks of the adjacent pairs in arrays, the best one merged each round.
    int32_t ids[48];
    uint32_t rank[48];  // of the pair (i, i+1); ~0 when it does not merge
    uint32_t into[48];
    for (size_t i = 0; i < n; ++i) ids[i] = byte_ids_[w[i]];
    auto score = [&](size_t i) {
      const uint64_t m = merge(ids[i], ids[i + 1]);
      rank[i] = m == kEmpty ? ~0u : uint32_t(m >> 32);
      into[i] = m == kEmpty ? 0 : uint32_t(m);
    };
    for (size_t i = 0; i + 1 < n; ++i) score(i);
    while (n > 1) {
      size_t best = 0;
      for (size_t i = 1; i + 1 < n; ++i)
        if (rank[i] < rank[best]) best = i;
      if (rank[best] == ~0u) break;
      ids[best] = int32_t(into[best]);
      // Drop the right half; the pairs after it keep their ranks, the two around the merge are scored again.
      for (size_t i = best + 1; i + 1 < n; ++i) ids[i] = ids[i + 1];
      for (size_t i = best + 1; i + 2 < n; ++i) rank[i] = rank[i + 1], into[i] = into[i + 1];
      --n;
      if (best + 1 < n) score(best);
      if (best > 0) score(best - 1);
    }
    out.insert(out.end(), ids, ids + n);
    return;
  }
  struct Sym {
    int32_t id;
    int32_t prev, next;  // -1 at the ends
  };
  std::vector<Sym> syms(n);
  for (size_t i = 0; i < n; ++i) syms[i] = {byte_ids_[w[i]], int32_t(i) - 1, i + 1 < n ? int32_t(i) + 1 : -1};
  struct Cand {
    uint32_t rank;
    int32_t pos;
    int32_t left, right, merged;
    bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
  };
  std::priority_queue<Cand, std::vector<Cand>, std::greater<>> heap;
  auto push = [&](int32_t pos) {
    if (pos < 0 || syms[size_t(pos)].next < 0) return;
    const int32_t l = syms[size_t(pos)].id, r = syms[size_t(syms[size_t(pos)].next)].id;
    const uint64_t m = merge(l, r);
    if (m != kEmpty) heap.push({uint32_t(m >> 32), pos, l, r, int32_t(uint32_t(m))});
  };
  for (int32_t i = 0; i < int32_t(n); ++i) push(i);
  while (!heap.empty()) {
    const Cand c = heap.top();
    heap.pop();
    Sym& s = syms[size_t(c.pos)];
    // Stale: the left symbol was merged away or changed, or its right neighbour changed.
    if (s.id != c.left || s.next < 0 || syms[size_t(s.next)].id != c.right) continue;
    const int32_t gone = s.next;
    s.id = c.merged;
    s.next = syms[size_t(gone)].next;
    if (s.next >= 0) syms[size_t(s.next)].prev = c.pos;
    syms[size_t(gone)].id = -1;
    push(s.prev);
    push(c.pos);
  }
  for (int32_t i = 0; i >= 0; i = syms[size_t(i)].next) out.push_back(syms[size_t(i)].id);
}

std::vector<std::pair<size_t, size_t>> InklingTokenizer::pretokenize(std::string_view text) {
  std::vector<std::pair<size_t, size_t>> out;
  for (size_t i = 0; i < text.size();) {
    const size_t e = next_piece(text.data(), text.size(), i);
    out.emplace_back(i, e);
    i = e;
  }
  return out;
}

// One stretch between added tokens: the pre-tokenizer, then byte-level BPE.
void InklingTokenizer::encode_plain(std::string_view s, std::vector<int32_t>& out) const {
  if (s.empty()) return;
  std::string repaired;
  // Requests are valid UTF-8; for anything else, what Hugging Face would see after its own lossy decoding.
  if (!unicode::valid_utf8(s)) {
    repaired = unicode::lossy_utf8(s);
    s = repaired;
  }
  const char* p = s.data();
  const size_t n = s.size();
  for (size_t i = 0; i < n;) {
    const size_t e = next_piece(p, n, i);
    bpe(reinterpret_cast<const unsigned char*>(p) + i, e - i, out);
    i = e;
  }
}

std::vector<int32_t> InklingTokenizer::encode(std::string_view text, bool parse_special) const {
  Piece p{text, parse_special};
  return encode(std::span<const Piece>(&p, 1));
}

std::vector<int32_t> InklingTokenizer::encode(std::span<const Piece> pieces) const {
  bool any_special = false, all_special = true;
  for (const auto& p : pieces) any_special |= p.special, all_special &= p.special;
  std::string joined;
  std::string_view full;
  if (pieces.size() == 1) {
    full = pieces[0].text;
  } else {
    for (const auto& p : pieces) joined += p.text;
    full = joined;
  }
  std::vector<bool> allowed;  // per byte, when pieces differ: may an added token match here
  const bool mixed = any_special && !all_special;
  if (mixed)
    for (const auto& p : pieces) allowed.insert(allowed.end(), p.text.size(), p.special);
  auto may = [&](size_t i, size_t len) {
    return !mixed || std::all_of(allowed.begin() + long(i), allowed.begin() + long(i + len), [](bool b) { return b; });
  };
  std::vector<int32_t> out;
  out.reserve(full.size() / 3 + 8);
  size_t start = 0;
  // Every added token is "<|" ... "|>" with no "|>" inside; a match must lie wholly where added tokens are allowed.
  for (size_t i = any_special ? full.find("<|") : npos; i != npos; i = full.find("<|", i)) {
    const size_t close = full.find("|>", i + 2);
    const size_t len = close == npos ? 0 : close + 2 - i;
    auto it = len >= 5 && len <= 40 ? added_.find(std::string(full.substr(i, len))) : added_.end();
    if (it == added_.end() || !may(i, len)) {
      ++i;
      continue;
    }
    encode_plain(full.substr(start, i - start), out);
    out.push_back(it->second);
    i += len;
    start = i;
  }
  encode_plain(full.substr(start), out);
  return out;
}

std::vector<int32_t> InklingTokenizer::special_ids() const {
  std::vector<int32_t> ids;
  for (int32_t i = inkling::kFirstAdded; size_t(i) < pieces_.size(); ++i) ids.push_back(i);
  return ids;
}

std::string InklingTokenizer::bytes(int32_t id) const {
  if (id < 0 || size_t(id) >= pieces_.size()) throw std::out_of_range("token id " + std::to_string(id));
  return bytes_[size_t(id)];
}

std::string InklingTokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
  std::string raw;
  for (int32_t id : ids) {
    if (id < 0 || size_t(id) >= pieces_.size()) throw std::out_of_range("token id " + std::to_string(id));
    if (skip_special && is_special(id)) continue;
    raw += bytes_[size_t(id)];
  }
  return unicode::lossy_utf8(raw);
}

}  // namespace eklow::text
