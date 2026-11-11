#include "text/tokenizer_qwen.hpp"

#include <algorithm>
#include <cstring>
#include <queue>
#include <stdexcept>

#include "model/gguf.hpp"
#include "text/unicode.hpp"

namespace eklow::text {

namespace {

// tokenizer.json's added_tokens: id, content, "special". The GGUF holds 33 control and user-defined tokens; these are
// the 26 Hugging Face knows (the other seven, <|audio_start|> .. <|audio_pad|>, are in tokenizer_config.json only). The ids
// are Qwen3.5's and 3.6's; Qwen3's vocabulary (Qwen3-Reranker's) has the same 26 from 151643 on, so from_gguf finds each by
// its text.
struct Added {
  int32_t id;
  const char* text;
  bool special;
};
constexpr Added kAdded[] = {
    {248044, "<|endoftext|>", true},        {248045, "<|im_start|>", true},        {248046, "<|im_end|>", true},
    {248047, "<|object_ref_start|>", true}, {248048, "<|object_ref_end|>", true},  {248049, "<|box_start|>", true},
    {248050, "<|box_end|>", true},          {248051, "<|quad_start|>", true},      {248052, "<|quad_end|>", true},
    {248053, "<|vision_start|>", true},     {248054, "<|vision_end|>", true},      {248055, "<|vision_pad|>", true},
    {248056, "<|image_pad|>", true},        {248057, "<|video_pad|>", true},       {248058, "<tool_call>", false},
    {248059, "</tool_call>", false},        {248060, "<|fim_prefix|>", false},     {248061, "<|fim_middle|>", false},
    {248062, "<|fim_suffix|>", false},      {248063, "<|fim_pad|>", false},        {248064, "<|repo_name|>", false},
    {248065, "<|file_sep|>", false},        {248066, "<tool_response>", false},    {248067, "</tool_response>", false},
    {248068, "<think>", false},             {248069, "</think>", false},
};

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

// The classes a word's letters may have: Qwen3.5's and 3.6's pre-tokenizer counts marks with letters ([\p{L}\p{M}]+),
// Qwen2's and Qwen3's do not (\p{L}+, and a mark is punctuation to it).
constexpr uint8_t kLettersMarks = unicode::kLetter | unicode::kMark;

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

// The pre-token starting at s[i] (valid UTF-8, i < n): its end. lm: the classes of a word's letters (kLettersMarks, or
// kLetter for the Qwen2 regex, whose alternatives below read \p{L}+ and  ?[^\s\p{L}\p{N}]+[\r\n]*).
size_t next_piece(const char* s, size_t n, size_t i, uint8_t lm = kLettersMarks) {
  uint32_t c0;
  const size_t l0 = unicode::decode(s + i, n - i, c0);
  const uint8_t k0 = unicode::classes(c0);
  auto is_lm = [lm](uint8_t k) { return (k & lm) != 0; };
  // (?i:'s|'t|'re|'ve|'m|'ll|'d)
  if (c0 == '\'')
    if (size_t e = contraction(s, n, i + 1)) return e;
  // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
  {
    auto run = [&](size_t p) {
      while (p < n) {
        uint32_t c;
        const size_t l = unicode::decode(s + p, n - p, c);
        if (!is_lm(unicode::classes(c))) break;
        p += l;
      }
      return p;
    };
    if (c0 != '\r' && c0 != '\n' && !(k0 & (unicode::kLetter | unicode::kNumber)) && i + l0 < n) {
      uint32_t c1;
      unicode::decode(s + i + l0, n - i - l0, c1);
      if (is_lm(unicode::classes(c1))) return run(i + l0);
    }
    if (is_lm(k0)) return run(i);
  }
  // \p{N}
  if (k0 & unicode::kNumber) return i + l0;
  //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
  {
    size_t p = c0 == ' ' ? i + 1 : i, q = p;
    while (q < n) {
      uint32_t c;
      const size_t l = unicode::decode(s + q, n - q, c);
      if (unicode::classes(c) & (lm | unicode::kNumber | unicode::kSpace)) break;
      q += l;
    }
    if (q > p) {
      while (q < n && (s[q] == '\r' || s[q] == '\n')) ++q;
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
    if (!(unicode::classes(c) & unicode::kSpace)) break;
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

QwenTokenizer QwenTokenizer::from_gguf(const gguf::File& f) {
  const auto* m = f.find("tokenizer.ggml.model");
  const auto* pre = f.find("tokenizer.ggml.pre");
  if (!m || m->as_string() != "gpt2" || !pre || (pre->as_string() != "qwen35" && pre->as_string() != "qwen2"))
    throw std::runtime_error(f.path() + ": not a Qwen tokenizer (gpt2 model with the qwen35 or qwen2 pre-tokenizer)");
  QwenTokenizer t;
  t.lm_ = pre->as_string() == "qwen35" ? kLettersMarks : unicode::kLetter;
  // Qwen3's tokenizer.json normalizes to NFC as Qwen3.6's does; converters before the key existed did not write it.
  const auto* nfc = f.find("tokenizer.ggml.normalizer.nfc");
  if (t.lm_ == kLettersMarks && (!nfc || !std::get<bool>(nfc->v)))
    throw std::runtime_error(f.path() + ": the tokenizer does not say it normalizes to NFC");
  const auto& toks = f.at("tokenizer.ggml.tokens").as_array();
  const auto& types = f.at("tokenizer.ggml.token_type").as_array();
  if (types.size() != toks.size()) throw std::runtime_error(f.path() + ": token and token type counts differ");
  const auto b2c = byte_to_char();
  std::unordered_map<uint32_t, uint8_t> c2b;
  for (uint32_t b = 0; b < 256; ++b) c2b.emplace(b2c[b], uint8_t(b));

  const size_t n = toks.size();
  t.pieces_.reserve(n);
  t.bytes_.resize(n);
  t.special_.assign(n, false);
  t.hf_special_.assign(n, 0);
  t.known_.assign(n, 0);
  t.byte_ids_.fill(-1);
  // The added tokens follow the base vocabulary: the first, <|endoftext|>, is where it ends.
  int32_t base = -1;
  for (size_t i = 0; i < n && base < 0; ++i)
    if (toks[i].as_string() == kAdded[0].text) base = int32_t(i);
  if (base < 0) throw std::runtime_error(f.path() + ": no <|endoftext|>");
  for (size_t i = 0; i < n; ++i) {
    const std::string& p = toks[i].as_string();
    t.pieces_.push_back(p);
    if (!t.ids_.emplace(p, int32_t(i)).second) throw std::runtime_error(f.path() + ": duplicate token '" + p + "'");
    const int64_t ty = types[i].as_int();
    // llama.cpp's token types: 1 normal, 3 control, 4 user-defined, 5 unused.
    if (ty == 3 || ty == 4 || ty == 5) {
      t.special_[i] = true;
      continue;
    }
    if (ty != 1) throw std::runtime_error(f.path() + ": unexpected token type " + std::to_string(ty) + " at " + std::to_string(i));
    if (int32_t(i) >= base) throw std::runtime_error(f.path() + ": a normal token past the base vocabulary");
    std::string& raw = t.bytes_[i];
    for (size_t k = 0; k < p.size();) {
      uint32_t cp;
      k += unicode::decode(p.data() + k, p.size() - k, cp);
      auto it = c2b.find(cp);
      if (it == c2b.end()) throw std::runtime_error(f.path() + ": token " + std::to_string(i) + " is not a byte-level piece");
      raw += char(it->second);
    }
    t.known_[i] = 1;
    if (raw.size() == 1) t.byte_ids_[uint8_t(raw[0])] = int32_t(i);
  }
  for (int32_t b : t.byte_ids_)
    if (b < 0) throw std::runtime_error(f.path() + ": the 256 single-byte tokens are not all present");
  for (const Added& a : kAdded) {
    const int32_t id = base + (a.id - kAdded[0].id);
    if (size_t(id) >= n || t.pieces_[size_t(id)] != a.text || !t.special_[size_t(id)])
      throw std::runtime_error(f.path() + ": token " + std::to_string(id) + " is not " + a.text);
    t.bytes_[size_t(id)] = a.text;
    t.known_[size_t(id)] = 1;
    t.hf_special_[size_t(id)] = a.special;
    t.added_.emplace_back(a.text, id);
  }
  std::stable_sort(t.added_.begin(), t.added_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });

  // Merges are "left right" in the GGUF's spelling, which holds no ASCII space (a space is U+0120).
  const auto& merges = f.at("tokenizer.ggml.merges").as_array();
  unsigned bits = 4;
  while ((size_t(1) << bits) < merges.size() * 4) ++bits;
  t.merge_shift_ = 64 - bits;
  t.merge_keys_.assign(size_t(1) << bits, kEmpty);
  t.merge_vals_.assign(size_t(1) << bits, 0);
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
  // Qwen3.5's GGUFs (vega's reader) carry no BOS; Qwen3.6's do, and none is ever added.
  const auto* bos = f.find("tokenizer.ggml.bos_token_id");
  t.bos_ = bos ? int32_t(bos->as_int()) : -1;
  t.eos_ = int32_t(f.at("tokenizer.ggml.eos_token_id").as_int());
  return t;
}

int32_t QwenTokenizer::id(std::string_view piece) const {
  auto it = ids_.find(std::string(piece));
  return it == ids_.end() ? -1 : it->second;
}

uint64_t QwenTokenizer::merge(int32_t a, int32_t b) const {
  const uint64_t key = pair_key(a, b);
  size_t h = size_t((key * 0x9E3779B97F4A7C15ull) >> merge_shift_);
  const size_t mask = merge_keys_.size() - 1;
  for (;; h = (h + 1) & mask) {
    if (merge_keys_[h] == key) return merge_vals_[h];
    if (merge_keys_[h] == kEmpty) return kEmpty;
  }
}

void QwenTokenizer::bpe(const unsigned char* w, size_t n, std::vector<int32_t>& out) const {
  if (n == 1) {
    out.push_back(byte_ids_[w[0]]);
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

std::vector<std::pair<size_t, size_t>> QwenTokenizer::pretokenize(std::string_view text) {
  std::vector<std::pair<size_t, size_t>> out;
  for (size_t i = 0; i < text.size();) {
    const size_t e = next_piece(text.data(), text.size(), i);
    out.emplace_back(i, e);
    i = e;
  }
  return out;
}

// One stretch between added tokens: NFC, the pre-tokenizer, byte-level BPE.
void QwenTokenizer::encode_plain(std::string_view s, std::vector<int32_t>& out) const {
  if (s.empty()) return;
  std::string repaired, normalized;
  // Requests are valid UTF-8; for anything else, what Hugging Face would see after its own lossy decoding.
  if (!unicode::valid_utf8(s)) {
    repaired = unicode::lossy_utf8(s);
    s = repaired;
  }
  normalized = unicode::nfc(s);
  const char* p = normalized.data();
  const size_t n = normalized.size();
  for (size_t i = 0; i < n;) {
    const size_t e = next_piece(p, n, i, lm_);
    bpe(reinterpret_cast<const unsigned char*>(p) + i, e - i, out);
    i = e;
  }
}

std::vector<int32_t> QwenTokenizer::encode(std::string_view text, bool parse_special) const {
  Piece p{text, parse_special};
  return encode(std::span<const Piece>(&p, 1));
}

std::vector<int32_t> QwenTokenizer::encode(std::span<const Piece> pieces) const {
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
  // Every added token starts with '<'; a match must lie wholly where added tokens are allowed.
  for (size_t i = any_special ? full.find('<') : std::string::npos; i != std::string::npos; i = full.find('<', i)) {
    const std::pair<std::string, int32_t>* hit = nullptr;
    for (auto& a : added_)
      if (full.compare(i, a.first.size(), a.first) == 0 && may(i, a.first.size())) {
        hit = &a;
        break;
      }
    if (!hit) {
      ++i;
      continue;
    }
    encode_plain(full.substr(start, i - start), out);
    out.push_back(hit->second);
    i += hit->first.size();
    start = i;
  }
  encode_plain(full.substr(start), out);
  return out;
}

std::vector<int32_t> QwenTokenizer::special_ids() const {
  std::vector<int32_t> ids;
  for (size_t i = 0; i < special_.size(); ++i)
    if (special_[i]) ids.push_back(int32_t(i));
  return ids;
}

std::string QwenTokenizer::bytes(int32_t id) const {
  if (id < 0 || size_t(id) >= pieces_.size()) throw std::out_of_range("token id " + std::to_string(id));
  return known_[size_t(id)] ? bytes_[size_t(id)] : std::string();
}

std::string QwenTokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
  std::string raw;
  for (int32_t id : ids) {
    if (id < 0 || size_t(id) >= pieces_.size()) throw std::out_of_range("token id " + std::to_string(id));
    if (!known_[size_t(id)] || (skip_special && hf_special_[size_t(id)])) continue;
    raw += bytes_[size_t(id)];
  }
  return unicode::lossy_utf8(raw);
}

}  // namespace eklow::text
