// What the chat template ports share: a rendered prompt is a list of parts, so that text from the request is
// tokenized without special tokens (a user who writes "<|im_start|>" gets those characters, not a turn boundary).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace eklow::text {

struct ChatPart {
  std::string text;
  bool input;  // from the request: tokenize without parsing special tokens
};

// Where the template raises, or where Jinja2 would fail on the input (a missing key used in arithmetic, a string
// where an object is required).
struct ChatError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string join(const std::vector<ChatPart>& parts) {
  std::string s;
  for (const auto& p : parts) s += p.text;
  return s;
}

}  // namespace eklow::text
