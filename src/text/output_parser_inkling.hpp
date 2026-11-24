// Inkling 2 Mini's generated output, parsed as it streams, token by token, as llama.cpp's inkling parser (common/parsers/
// inkling.cpp at c74c71a6: a PEG over the output's tokens, run again on the whole output after each token) reads it. A turn is
// a list of blocks, each opened by <|message_model|> and closed by <|end_message|>:
//   <|message_model|><|content_thinking|> REASONING <|end_message|>
//   <|message_model|><|content_text|> CONTENT <|end_message|>
//   <|message_model|> [NAME] <|content_invoke_tool_json|> {"name":"NAME","args":{...}} <|end_message|>
// and the model ends the turn with <|content_model_end_sampling|> (or <|endoftext|>), which is not part of the text. The
// block kinds are told apart by tokens, never by their text. What the message holds, as llama.cpp decides it (each is
// matched, and listed in DECISIONS 2026-11-24 "the text layer"):
//   - several thinking blocks join into one reasoning text; reasoning that is only blanks (space, newline, return, tab) is
//     no reasoning; with reasoning_format "none" a thinking block is content, markers included;
//   - output that does not open with a header, a header followed by neither a thinking nor a text token (when no call can
//     follow), and everything after a closed call that cannot be a further call, is ignored;
//   - a call is read only when tools were given and tool_choice is not "none" (then a call is nothing, not even content);
//     its name is the one inside the JSON, taken as written between the quotes (escapes are not undone) and trimmed of
//     trailing blanks; the text between the header and <|content_invoke_tool_json|> is free (with tool_choice "required" it
//     must be absent or the tool's own name, and the JSON name must be one of the tools'); the arguments are the JSON text
//     as written, `{}` when it is not an object; the JSON is read with a strict grammar (no leading zeros, no raw escapes
//     but the JSON ones; nesting deeper than 64 stops the parser);
//   - a call whose JSON is wrong is not a call, unless it was the whole message so far (no reasoning, content or finished
//     call), in which case llama.cpp keeps what it had seen of it: the message of the last token that gave one;
//   - a call cut off by the end of the output keeps its name and the arguments as far as they went (`truncated`);
//   - with a response format, thinking blocks and then one text block whose content is a JSON value, closed at once.
// Text is released only as whole UTF-8 characters; a malformed sequence is one U+FFFD (what the PEG's UTF-8 reader sees).
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "text/inkling_mode.hpp"
#include "text/output.hpp"
#include "text/tokenizer_inkling.hpp"

namespace eklow::text {

class InklingOutputParser {
 public:
  // `prefill`: the tokens the prompt ended with when it continued an assistant message (InklingChat::generation_parts,
  // tokenized); they are read first and delivered nowhere, but are in message().
  InklingOutputParser(const InklingTokenizer& tok, const InklingMode& mode, std::span<const int32_t> prefill = {});

  OutputDelta push(int32_t id);
  // The output ended (the token limit, or a stop): what was cut off is delivered, and a format the output did not follow
  // is reported in `errors`.
  OutputDelta finish();

  // Token ids that end the turn.
  bool is_end(int32_t id) const { return id == inkling::kModelEndSampling || id == inkling::kEndOfText; }

  // Everything delivered so far, the prefill included.
  struct Message {
    std::string reasoning, content;
    std::vector<ToolCall> tool_calls;
    std::vector<std::string> errors;
  };
  const Message& message() const { return msg_; }

  // The parse of a call's text so far: how it stands (a call needs its header, a name, and its JSON, then the end token).
  enum class Parse { Fail, Ok, More };
  struct Call {
    Parse status = Parse::Fail;
    size_t end = 0;       // Ok: where the call's end token ended
    bool named = false;   // the name was read: the call is part of the message (More) or is one (Ok)
    std::string name, args;
  };

 private:
  enum class Phase { Idle, AfterHeader, Think, Text, Json, JsonEnd, Calls, Ignore, Frozen };

  void feed(int32_t id, OutputDelta& d);
  void start_block(Phase kind, OutputDelta& d);
  void end_block(OutputDelta& d);
  void release_text(const std::string& bytes, bool at_end, OutputDelta& d);
  void emit_text(const std::string& text, OutputDelta& d);
  void run_calls(OutputDelta& d);
  void add_call(const Call& c, bool truncated, OutputDelta& d);
  void freeze(OutputDelta& d);
  bool committed() const;
  Call parse_call(size_t pos) const;

  const InklingTokenizer& tok_;
  InklingMode mode_;
  bool rf_, tools_, required_;
  Phase phase_ = Phase::Idle;
  bool done_ = false, failed_ = false;
  std::string tail_;   // the open block's bytes not yet resolved (an incomplete UTF-8 character)
  std::string held_;   // reasoning that is blanks so far
  // The call phase: its bytes, the token id at the first byte of each token (-1 elsewhere), where the current call began
  std::string cb_;
  std::vector<int32_t> ct_;
  size_t call_from_ = 0;
  size_t n_calls_ = 0;
  Call open_;          // the current call as of the last token (More), for the end of the output
  size_t json_sent_ = 0;
  Message msg_;
};

}  // namespace eklow::text
