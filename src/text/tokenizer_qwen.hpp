// Qwen3.6's tokenizer (the Qwen3.6-35B-A3B family: stock and finetunes, one tokenizer.json), built from the GGUF and
// following Hugging Face's tokenizers on that tokenizer.json:
//   1. the 26 added tokens (<|im_start|>, <think>, <tool_call>, ...) are split out of the text first, longest match
//      first, where special tokens are parsed;
//   2. each stretch between them is normalized to NFC;
//   3. the pre-tokenizer splits it with
//        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//      (matched here by hand over code points, text/unicode.hpp's classes);
//   4. byte-level BPE over each piece: its UTF-8 bytes as GPT-2's printable characters, merged pair by pair, lowest
//      merge rank first and leftmost first among equal ranks.
// No BOS is added. Decoding joins the bytes of the tokens and replaces ill-formed UTF-8 as Hugging Face does.
//
// The GGUF's other control tokens (the audio and TTS ones, the [PAD...] filler up to the head's width) are not in
// tokenizer.json, so they are never produced from text, never decoded, and count as special for the sampler's bans.
#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace eklow::gguf {
class File;
}

namespace eklow::text {

class QwenTokenizer {
 public:
  // Throws std::runtime_error when the file has no Qwen3.6 tokenizer.
  static QwenTokenizer from_gguf(const gguf::File& f);

  std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;

  // A rendered prompt as pieces, added tokens recognized only inside pieces that allow them (template text; request
  // text does not). Everything between two added tokens is normalized and split as one string, across piece
  // boundaries, as on the joined text, so a prompt without injected added-token text tokenizes exactly as
  // encode(join) would.
  struct Piece {
    std::string_view text;
    bool special;
  };
  std::vector<int32_t> encode(std::span<const Piece> pieces) const;

  // The bytes of the ids, joined, with ill-formed UTF-8 replaced by U+FFFD; added tokens as their text, except
  // that skip_special leaves out the ones tokenizer.json marks special (<|im_start|>, <|endoftext|>, ...; not
  // <think>, <tool_call>, ...). Ids the tokenizer does not know (the GGUF's extra tokens) write nothing.
  std::string decode(std::span<const int32_t> ids, bool skip_special = false) const;

  // One token's bytes as decode() would write them, without repairing UTF-8: a byte-level token's bytes (possibly
  // part of a character), an added token its text, nothing for ids the tokenizer does not know.
  std::string bytes(int32_t id) const;

  // The split of NFC-normalized text into pre-tokens, as byte ranges; for checking.
  static std::vector<std::pair<size_t, size_t>> pretokenize(std::string_view text);

  size_t size() const { return pieces_.size(); }
  // The GGUF's spelling of the token: the bytes as GPT-2's printable characters, the added tokens' text.
  const std::string& piece(int32_t id) const { return pieces_.at(size_t(id)); }
  // -1 when the piece (in the GGUF's spelling) is not in the vocabulary.
  int32_t id(std::string_view piece) const;
  // Added tokens, the GGUF's other control tokens and its unused ones.
  bool is_special(int32_t id) const { return special_.at(size_t(id)); }
  // Every special token's id, ascending.
  std::vector<int32_t> special_ids() const;
  int32_t bos() const { return bos_; }
  int32_t eos() const { return eos_; }

 private:
  void encode_plain(std::string_view s, std::vector<int32_t>& out) const;
  void bpe(const unsigned char* w, size_t n, std::vector<int32_t>& out) const;
  uint64_t merge(int32_t a, int32_t b) const;  // rank << 32 | merged id, or ~0

  std::vector<std::string> pieces_;
  std::unordered_map<std::string, int32_t> ids_;
  std::vector<bool> special_;
  std::vector<uint8_t> hf_special_;  // tokenizer.json's "special" flag, for skip_special
  std::vector<std::string> bytes_;   // per id: the raw bytes of a byte-level token, the text of an added token
  std::vector<uint8_t> known_;       // ids tokenizer.json has (base vocabulary and the 26 added tokens)
  std::array<int32_t, 256> byte_ids_{};
  // Open-addressing table over (left id, right id): key, then rank << 32 | merged id.
  std::vector<uint64_t> merge_keys_, merge_vals_;
  unsigned merge_shift_ = 0;
  // The 26 added tokens, longest first.
  std::vector<std::pair<std::string, int32_t>> added_;
  int32_t bos_ = -1, eos_ = -1;
};

}  // namespace eklow::text
