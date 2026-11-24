// What the output parser needs of a request, as llama-server's inkling parser (common/parsers/inkling.cpp, c74c71a6) builds
// itself from the request: whether tool calls are read, which tools there are, and how the reasoning is delivered.
#pragma once

#include <string>
#include <vector>

namespace eklow::text {

enum class InklingToolChoice { Auto, None, Required };

struct InklingMode {
  std::vector<std::string> tool_names;                 // the request's tools, in order
  InklingToolChoice tool_choice = InklingToolChoice::Auto;
  bool parallel_tool_calls = true;                     // more than one call in a turn (llama.cpp's default for this template)
  bool response_format = false;                        // a JSON schema constrains the answer: its text is read as content
  bool extract_reasoning = true;                       // reasoning_format is not "none"

  // Tool calls are read: tools were given and tool_choice is not "none".
  bool use_tools() const { return !tool_names.empty() && tool_choice != InklingToolChoice::None; }
};

}  // namespace eklow::text
