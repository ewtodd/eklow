// Inkling 2 Mini's tokenizer (GGUF tokenizer.ggml.model gpt2, pre-tokenizer inkling2), built from the GGUF and following
// Hugging Face's tokenizers on the model's tokenizer.json, which llama.cpp's inkling2 pre-type reproduces (M0: identical
// ids on every string and prompt tried):
//   1. the 60 added tokens (<|message_user|>, <|content_thinking|>, <|unused_200007|>, ... 199998-200057, all special
//      to Hugging Face) are split out of the text first where special tokens are parsed; no normalizer, so no NFC;
//   2. each stretch between them is split by GPT-4o's regex
//        [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//        |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//        |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//      (matched here by hand over code points, with text/unicode.hpp's classes and the case classes of
//      unicode_case_tables.hpp; the alternatives are tried in order and each one backtracks as the regex would);
//   3. byte-level BPE over each piece with ignore_merges: a piece that is itself in the vocabulary is one token, else
//      its UTF-8 bytes as GPT-2's printable characters are merged pair by pair, lowest merge rank first and leftmost
//      first among equal ranks.
// No BOS is added (the model opens its own turns). Decoding joins the bytes of the tokens and replaces ill-formed UTF-8
// as Hugging Face does.
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

// The tokens the chat template and the output parser name.
namespace inkling {
inline constexpr int32_t kUnused = 199998;
inline constexpr int32_t kEndOfText = 199999;  // the GGUF's padding token; the model also ends a turn with it
inline constexpr int32_t kMessageUser = 200000;
inline constexpr int32_t kMessageModel = 200001;
inline constexpr int32_t kMessageSystem = 200002;
inline constexpr int32_t kMessageTool = 200003;
inline constexpr int32_t kContentText = 200004;
inline constexpr int32_t kContentImage = 200005;
inline constexpr int32_t kModelEndSampling = 200006;  // the GGUF's eos and eot token
inline constexpr int32_t kContentThinking = 200008;
inline constexpr int32_t kEndMessage = 200010;
inline constexpr int32_t kContentToolError = 200022;
inline constexpr int32_t kContentXml = 200024;
inline constexpr int32_t kBeginOfText = 200028;
inline constexpr int32_t kContentInvokeToolJson = 200049;
inline constexpr int32_t kContentInvokeToolText = 200057;
inline constexpr int32_t kFirstAdded = 199998, kLastAdded = 200057;
}  // namespace inkling

class InklingTokenizer {
 public:
  // Throws std::runtime_error when the file has no Inkling 2 tokenizer.
  static InklingTokenizer from_gguf(const gguf::File& f);

  std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;

  // A rendered prompt as pieces, added tokens recognized only inside pieces that allow them (template text; request
  // text does not: eklow's rule, where llama.cpp parses special tokens everywhere). Everything between two added tokens
  // is split as one string, across piece boundaries, as on the joined text, so a prompt without injected added-token
  // text tokenizes exactly as encode(join) would.
  struct Piece {
    std::string_view text;
    bool special;
  };
  std::vector<int32_t> encode(std::span<const Piece> pieces) const;

  // The bytes of the ids, joined, with ill-formed UTF-8 replaced by U+FFFD; added tokens as their text, except that
  // skip_special leaves them out (Hugging Face marks all 60 special).
  std::string decode(std::span<const int32_t> ids, bool skip_special = false) const;

  // One token's bytes as decode() would write them, without repairing UTF-8: a byte-level token's bytes (possibly part
  // of a character), an added token its text.
  std::string bytes(int32_t id) const;

  // The split of text into pre-tokens, as byte ranges; for checking. The text must be well-formed UTF-8.
  static std::vector<std::pair<size_t, size_t>> pretokenize(std::string_view text);

  size_t size() const { return pieces_.size(); }
  // The GGUF's spelling of the token: the bytes as GPT-2's printable characters, the added tokens' text.
  const std::string& piece(int32_t id) const { return pieces_.at(size_t(id)); }
  // -1 when the piece (in the GGUF's spelling) is not in the vocabulary.
  int32_t id(std::string_view piece) const;
  // The 60 added tokens.
  bool is_special(int32_t id) const { return id >= inkling::kFirstAdded && size_t(id) < pieces_.size(); }
  // Every special token's id, ascending.
  std::vector<int32_t> special_ids() const;
  int32_t bos() const { return -1; }
  int32_t eos() const { return eos_; }

 private:
  void encode_plain(std::string_view s, std::vector<int32_t>& out) const;
  void bpe(const unsigned char* w, size_t n, std::vector<int32_t>& out) const;
  uint64_t merge(int32_t a, int32_t b) const;  // rank << 32 | merged id, or ~0

  std::vector<std::string> pieces_;
  std::unordered_map<std::string, int32_t> ids_;
  std::vector<std::string> bytes_;  // per id: the raw bytes of a byte-level token, the text of an added token
  // The raw bytes of every base token, to its id (ignore_merges).
  std::unordered_map<std::string, int32_t> by_bytes_;
  std::array<int32_t, 256> byte_ids_{};
  // Open-addressing table over (left id, right id): key, then rank << 32 | merged id.
  std::vector<uint64_t> merge_keys_, merge_vals_;
  unsigned merge_shift_ = 0;
  std::unordered_map<std::string, int32_t> added_;  // "<|...|>" -> id
  int32_t eos_ = -1;
};

}  // namespace eklow::text
