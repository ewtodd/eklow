// Gemma 4's tokenizer, built from the GGUF and following Hugging Face's tokenizer.json for it:
//   1. the 24 added tokens (GGUF types control and user-defined: <bos>, <|turn>, <|"|>, ...) are
//      split out first, longest match first, when special tokens are parsed;
//   2. the rest is normalized by replacing every space with U+2581 ("▁"); the pre-tokenizer splits
//      on spaces, of which none are left, so each stretch between added tokens is one BPE word;
//   3. BPE over that word: characters, unknown ones as their UTF-8 bytes (<0xXX>, byte fallback),
//      merged pair by pair, lowest merge rank first and leftmost first among equal ranks.
// No BOS is added; the chat template writes it. llama.cpp also splits on newlines, which Hugging
// Face does not; this follows Hugging Face.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace eklow::gguf {
class File;
}

namespace eklow::text {

class Tokenizer {
 public:
  // Throws std::runtime_error when the file has no Gemma 4 tokenizer.
  static Tokenizer from_gguf(const gguf::File& f);

  std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;

  // A rendered prompt as pieces, special tokens recognized only inside pieces that allow them
  // (template text; request text does not). BPE still runs over everything between two special
  // tokens, across piece boundaries, as it does on the joined text, so a prompt without injected
  // special-token text tokenizes exactly as encode(join) would.
  struct Piece {
    std::string_view text;
    bool special;
  };
  std::vector<int32_t> encode(std::span<const Piece> pieces) const;
  // Byte tokens are joined into UTF-8 (invalid sequences become U+FFFD); "▁" becomes a space;
  // added tokens are written as their text unless skip_special.
  std::string decode(std::span<const int32_t> ids, bool skip_special = false) const;

  size_t size() const { return pieces_.size(); }
  const std::string& piece(int32_t id) const { return pieces_.at(size_t(id)); }
  // -1 when the piece is not in the vocabulary.
  int32_t id(std::string_view piece) const;
  bool is_special(int32_t id) const { return special_.at(size_t(id)); }
  int32_t bos() const { return bos_; }
  int32_t eos() const { return eos_; }

 private:
  void bpe(std::string_view word, std::vector<int32_t>& out) const;

  std::vector<std::string> pieces_;
  std::unordered_map<std::string, int32_t> ids_;
  std::vector<bool> special_;
  std::vector<int16_t> byte_of_;  // the byte a <0xXX> token stands for, or -1
  std::array<int32_t, 256> byte_ids_{};
  // (left id << 32 | right id) -> rank in the high half, merged id in the low half.
  std::unordered_map<uint64_t, uint64_t> merges_;
  // Added tokens, longest first.
  std::vector<std::pair<std::string, int32_t>> added_;
  int32_t bos_ = -1, eos_ = -1;
};

}  // namespace eklow::text
