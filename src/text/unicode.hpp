// What Qwen's tokenizer needs of Unicode, from the tables tools/gen/gen_unicode_tables.py wrote
// (unicode_tables.hpp): the classes its pre-tokenizer's regex names, NFC normalization, and UTF-8 helpers.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace eklow::text::unicode {

enum : uint8_t { kLetter = 1, kMark = 2, kNumber = 4, kSpace = 8 };

// Class bits of a code point (kLetter | kMark | kNumber | kSpace; 0 for everything else, including unassigned code
// points). A flat table over all 0x110000 code points, built before main.
extern const uint8_t* const kClassTable;
inline uint8_t classes(uint32_t cp) { return kClassTable[cp] & 15; }

// The code point starting at p[0] (at most `avail` bytes remain) and its length in bytes. The input must be
// well-formed UTF-8 (see valid_utf8); a stray byte comes back as itself with length 1.
inline size_t decode(const char* p, size_t avail, uint32_t& cp) {
  const unsigned char c = static_cast<unsigned char>(p[0]);
  if (c < 0x80) return cp = c, 1;
  if (c < 0xE0 && avail >= 2) return cp = (c & 0x1Fu) << 6 | (p[1] & 0x3Fu), 2;
  if (c < 0xF0 && avail >= 3) return cp = (c & 0x0Fu) << 12 | (p[1] & 0x3Fu) << 6 | (p[2] & 0x3Fu), 3;
  if (avail >= 4) return cp = (c & 0x07u) << 18 | (p[1] & 0x3Fu) << 12 | (p[2] & 0x3Fu) << 6 | (p[3] & 0x3Fu), 4;
  return cp = c, 1;
}
void append_utf8(std::string& out, uint32_t cp);

// Whether s is well-formed UTF-8 (no overlongs, surrogates or code points past U+10FFFF).
bool valid_utf8(std::string_view s);
// s with every ill-formed sequence replaced by U+FFFD, one per maximal ill-formed subpart: Rust's
// String::from_utf8_lossy, which Hugging Face's byte-level decoder applies.
std::string lossy_utf8(std::string_view s);

// Unicode normalization form C of well-formed UTF-8 text.
std::string nfc(std::string_view s);

}  // namespace eklow::text::unicode
