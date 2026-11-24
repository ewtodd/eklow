// What the output parser delivers as a turn streams: reasoning, content and finished tool calls.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/json_value.hpp"

namespace eklow::text {

struct ToolCall {
  std::string name;
  json::Value arguments;  // an object
  // Inkling 2's parser keeps what llama.cpp's does: the arguments exactly as the model wrote them (`arguments` is
  // that text read as JSON, an empty object when it is not a whole object), and whether the call was cut off (the
  // output ended, or went wrong, before its `<|end_message|>`).
  std::string arguments_text;
  bool truncated = false;
};

struct OutputDelta {
  std::string reasoning, content;
  std::vector<ToolCall> tool_calls;  // each call once, when its </tool_call> arrives
  std::vector<std::string> errors;   // malformed tool calls, with what was wrong
  bool done = false;                 // an end token arrived
  int32_t end_token = -1;            // which one
};

}  // namespace eklow::text
