#include "text/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <stdexcept>

#include "model/gguf.hpp"

namespace eklow::text {

namespace {

constexpr std::string_view kSpace = "\xe2\x96\x81";  // U+2581, "▁"

size_t utf8_len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c >> 5) == 0x6) return 2;
  if ((c >> 4) == 0xe) return 3;
  if ((c >> 3) == 0x1e) return 4;
  return 0;  // a continuation byte or invalid lead
}

// Whether s is well-formed UTF-8 (no overlongs, surrogates or code points past U+10FFFF).
bool valid_utf8(std::string_view s) {
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    size_t n = utf8_len(c);
    if (n == 0 || i + n > s.size()) return false;
    uint32_t cp = n == 1 ? c : n == 2 ? c & 0x1f : n == 3 ? c & 0x0f : c & 0x07;
    for (size_t k = 1; k < n; ++k) {
      unsigned char d = s[i + k];
      if ((d & 0xc0) != 0x80) return false;
      cp = cp << 6 | (d & 0x3f);
    }
    if ((n == 2 && cp < 0x80) || (n == 3 && cp < 0x800) || (n == 4 && cp < 0x10000) || cp > 0x10ffff ||
        (cp >= 0xd800 && cp <= 0xdfff))
      return false;
    i += n;
  }
  return true;
}

uint64_t pair_key(int32_t a, int32_t b) { return uint64_t(uint32_t(a)) << 32 | uint32_t(b); }

}  // namespace

Tokenizer Tokenizer::from_gguf(const gguf::File& f) {
  if (auto* m = f.find("tokenizer.ggml.model"); !m || m->as_string() != "gemma4")
    throw std::runtime_error(f.path() + ": not a Gemma 4 tokenizer");
  Tokenizer t;
  const auto& toks = f.at("tokenizer.ggml.tokens").as_array();
  const auto& types = f.at("tokenizer.ggml.token_type").as_array();
  if (types.size() != toks.size()) throw std::runtime_error(f.path() + ": token and token type counts differ");
  t.pieces_.reserve(toks.size());
  t.special_.resize(toks.size());
  t.byte_of_.assign(toks.size(), -1);
  t.byte_ids_.fill(-1);
  for (size_t i = 0; i < toks.size(); ++i) {
    const std::string& p = toks[i].as_string();
    t.pieces_.push_back(p);
    if (!t.ids_.emplace(p, int32_t(i)).second) throw std::runtime_error(f.path() + ": duplicate token '" + p + "'");
    const int64_t ty = types[i].as_int();
    // llama.cpp's token types: 3 control, 4 user-defined (both Hugging Face added tokens), 6 byte.
    if (ty == 3 || ty == 4) {
      t.special_[i] = true;
      t.added_.emplace_back(p, int32_t(i));
    } else if (ty == 6) {
      unsigned b = 0;
      if (p.size() != 6 || p.compare(0, 3, "<0x") != 0 || std::sscanf(p.c_str() + 3, "%2X", &b) != 1)
        throw std::runtime_error(f.path() + ": malformed byte token '" + p + "'");
      t.byte_of_[i] = int16_t(b);
      t.byte_ids_[b] = int32_t(i);
    }
  }
  for (int32_t b : t.byte_ids_)
    if (b < 0) throw std::runtime_error(f.path() + ": the 256 byte-fallback tokens are not all present");
  std::sort(t.added_.begin(), t.added_.end(), [](auto& a, auto& b) { return a.first.size() > b.first.size(); });

  // Merges are "left right"; pieces hold no ASCII spaces (they became U+2581), so the first space
  // after the first character separates the two.
  const auto& merges = f.at("tokenizer.ggml.merges").as_array();
  t.merges_.reserve(merges.size() * 2);
  for (size_t r = 0; r < merges.size(); ++r) {
    const std::string& m = merges[r].as_string();
    size_t sp = m.find(' ', 1);
    if (sp == std::string::npos) throw std::runtime_error(f.path() + ": malformed merge '" + m + "'");
    int32_t a = t.id(std::string_view(m).substr(0, sp)), b = t.id(std::string_view(m).substr(sp + 1));
    int32_t c = t.id(m.substr(0, sp) + m.substr(sp + 1));
    if (a < 0 || b < 0 || c < 0) throw std::runtime_error(f.path() + ": merge '" + m + "' names an unknown token");
    t.merges_.emplace(pair_key(a, b), uint64_t(r) << 32 | uint32_t(c));
  }
  t.bos_ = int32_t(f.at("tokenizer.ggml.bos_token_id").as_int());
  t.eos_ = int32_t(f.at("tokenizer.ggml.eos_token_id").as_int());
  return t;
}

int32_t Tokenizer::id(std::string_view piece) const {
  auto it = ids_.find(std::string(piece));
  return it == ids_.end() ? -1 : it->second;
}

void Tokenizer::bpe(std::string_view word, std::vector<int32_t>& out) const {
  struct Sym {
    int32_t id;
    int32_t prev, next;  // -1 at the ends
  };
  std::vector<Sym> syms;
  syms.reserve(word.size());
  std::string ch;
  for (size_t i = 0; i < word.size();) {
    size_t n = utf8_len(static_cast<unsigned char>(word[i]));
    if (n == 0 || i + n > word.size()) n = 1;  // a stray byte: falls back to its byte token below
    ch.assign(word.substr(i, n));
    auto it = ids_.find(ch);
    if (it != ids_.end() && n == utf8_len(static_cast<unsigned char>(word[i]))) {
      syms.push_back({it->second, int32_t(syms.size()) - 1, int32_t(syms.size()) + 1});
    } else {
      for (unsigned char b : ch) syms.push_back({byte_ids_[b], int32_t(syms.size()) - 1, int32_t(syms.size()) + 1});
    }
    i += n;
  }
  if (syms.empty()) return;
  syms.back().next = -1;

  struct Cand {
    uint32_t rank;
    int32_t pos;
    int32_t left, right, merged;
    bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
  };
  std::priority_queue<Cand, std::vector<Cand>, std::greater<>> heap;
  auto push = [&](int32_t pos) {
    if (pos < 0 || syms[pos].next < 0) return;
    int32_t l = syms[pos].id, r = syms[syms[pos].next].id;
    auto it = merges_.find(pair_key(l, r));
    if (it != merges_.end()) heap.push({uint32_t(it->second >> 32), pos, l, r, int32_t(uint32_t(it->second))});
  };
  for (int32_t i = 0; i < int32_t(syms.size()); ++i) push(i);
  while (!heap.empty()) {
    Cand c = heap.top();
    heap.pop();
    Sym& s = syms[c.pos];
    // Stale: the left symbol was merged away or changed, or its right neighbour changed.
    if (s.id != c.left || s.next < 0 || syms[s.next].id != c.right) continue;
    int32_t gone = s.next;
    s.id = c.merged;
    s.next = syms[gone].next;
    if (s.next >= 0) syms[s.next].prev = c.pos;
    syms[gone].id = -1;
    push(s.prev);
    push(c.pos);
  }
  for (int32_t i = 0; i >= 0; i = syms[i].next) out.push_back(syms[i].id);
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
  Piece p{text, parse_special};
  return encode(std::span<const Piece>(&p, 1));
}

std::vector<int32_t> Tokenizer::encode(std::span<const Piece> pieces) const {
  std::string text;
  std::vector<bool> allowed;  // per byte: may a special token match here
  for (const auto& p : pieces) {
    text += p.text;
    allowed.insert(allowed.end(), p.text.size(), p.special);
  }
  std::vector<int32_t> out;
  std::string norm;
  auto plain = [&](std::string_view s) {
    if (s.empty()) return;
    norm.clear();
    for (char c : s) {
      if (c == ' ') norm += kSpace;
      else norm += c;
    }
    bpe(norm, out);
  };
  size_t start = 0;
  // Every added token starts with '<'; a match must lie wholly where special tokens are allowed.
  for (size_t i = text.find('<'); i != std::string::npos; i = text.find('<', i)) {
    const std::pair<std::string, int32_t>* hit = nullptr;
    if (allowed[i])
      for (auto& a : added_)
        if (text.compare(i, a.first.size(), a.first) == 0 &&
            std::all_of(allowed.begin() + long(i), allowed.begin() + long(i + a.first.size()), [](bool b) { return b; })) {
          hit = &a;
          break;
        }
    if (!hit) {
      ++i;
      continue;
    }
    plain(std::string_view(text).substr(start, i - start));
    out.push_back(hit->second);
    i += hit->first.size();
    start = i;
  }
  plain(std::string_view(text).substr(start));
  return out;
}

std::string Tokenizer::bytes(int32_t id) const {
  if (id < 0 || size_t(id) >= pieces_.size()) throw std::out_of_range("token id " + std::to_string(id));
  if (byte_of_[id] >= 0) return std::string(1, char(byte_of_[id]));
  const std::string& p = pieces_[id];
  if (special_[id]) return p;
  std::string out;
  for (size_t i = 0; i < p.size();) {
    if (p.compare(i, kSpace.size(), kSpace) == 0) {
      out += ' ';
      i += kSpace.size();
    } else {
      out += p[i++];
    }
  }
  return out;
}

std::string Tokenizer::decode(std::span<const int32_t> ids, bool skip_special) const {
  std::string out, bytes;
  size_t nbytes = 0;
  auto flush = [&] {
    if (bytes.empty()) return;
    // A run of byte tokens that is not valid UTF-8 becomes one U+FFFD per byte, as Hugging Face's
    // ByteFallback decoder does.
    if (valid_utf8(bytes)) out += bytes;
    else
      for (size_t k = 0; k < nbytes; ++k) out += "\xef\xbf\xbd";
    bytes.clear();
    nbytes = 0;
  };
  for (int32_t id : ids) {
    if (id < 0 || size_t(id) >= pieces_.size()) throw std::out_of_range("token id " + std::to_string(id));
    if (byte_of_[id] >= 0) {
      bytes += char(byte_of_[id]);
      ++nbytes;
      continue;
    }
    flush();
    if (special_[id]) {
      if (!skip_special) out += pieces_[id];
      continue;
    }
    const std::string& p = pieces_[id];
    for (size_t i = 0; i < p.size();) {
      if (p.compare(i, kSpace.size(), kSpace) == 0) {
        out += ' ';
        i += kSpace.size();
      } else {
        out += p[i++];
      }
    }
  }
  flush();
  return out;
}

}  // namespace eklow::text
