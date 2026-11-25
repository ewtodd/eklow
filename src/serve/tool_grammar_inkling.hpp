// Constrained decoding of Inkling 2 Mini's tool calls, token by token, with the semantics of llama.cpp's grammar sampler
// (c74c71a6) on the grammar its Inkling parser builds for a request with tools (common/parsers/inkling.cpp,
// grammar_lazy false, so the grammar holds from the first generated token). DECISIONS 2026-11-24, M4.
//
// tool_choice auto, grammar `root ::= trigger`:
//     trigger ::= | <|content_invoke_tool_json|> call-rest | !<|content_invoke_tool_json|> trigger
//   so anything may be generated until the model writes <|content_invoke_tool_json|> (200049, the one token the grammar
//   names there), and then the call is the JSON object {"name": STRING, "args": VALUE} (the `space` rule between the
//   tokens, VALUE any JSON value, no space before the "{"), then <|end_message|> (200010). With parallel calls
//   another block may follow: <|message_model|> (200001), any tokens up to the next 200049, and the next call.
// tool_choice required: blocks (<|message_model|> then thinking or text, each closed by 200010, in which any token goes), then
//   <|message_model|> and a call of one of the request's tools: the tool's name as text (optional), 200049 and the call,
//   its "name" the tool's name; with parallel calls more such blocks after it.
// The sampler may end the turn (the end-of-generation tokens 200006 and 199999) only where the grammar may end: in auto
// mode before a call and after a complete one, in required mode after a complete call.
//
// A token is judged as llama.cpp judges it: a token the grammar names (200001, 200004, 200008, 200010, 200049) matches only a
// rule that names it by id; every other token is matched by its text (special tokens as their "<|...|>" text; a token's bytes up
// to a NUL) against the grammar's characters, code point by code point, with a token's trailing partial UTF-8 character carried
// to the next token. Tokens llama.cpp can never accept are never allowed: those with an empty text or one that starts with NUL,
// and, because the sampler carries the partial character across every token, those that begin with a byte that cannot start or
// continue a character (the 332 stray continuation and NUL byte tokens, anywhere; and after a token that ends inside a character,
// any token that does not continue it).
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "text/tokenizer_inkling.hpp"

namespace eklow::serve {

// What the grammar needs from the vocabulary, built once per tokenizer and shared by every request.
struct InklingToolVocab {
  explicit InklingToolVocab(const text::InklingTokenizer& tok);

  struct Pu {  // llama.cpp's llama_partial_utf8: a character begun but not finished
    uint32_t value = 0;
    int n_remain = 0;  // bytes still to come; negative: the bytes were not UTF-8
  };

  size_t n = 0;                      // vocabulary size
  std::vector<std::string> piece;    // each token's text as llama.cpp's grammar reads it (cut at the first NUL)
  std::vector<uint8_t> dead;         // empty text, or one that starts with NUL: never allowed
  std::vector<uint32_t> cp_off;      // each token's code points when no character is pending, in cp_pool (n + 1 offsets)
  std::vector<uint32_t> cp_pool;
  std::vector<Pu> end_pu;            // the pending character after it (n_remain < 0: the token cannot start here)
  std::vector<int32_t> eog = {text::inkling::kEndOfText, text::inkling::kModelEndSampling};
  // Ascending ids that llama.cpp's grammar can never accept from a state with no character pending: the dead tokens and the stray
  // continuation and NUL byte tokens. InklingToolGrammar::unconstrained() allows every token but these, so a caller that samples
  // without the mask must ban exactly these (serve::Sampler's `banned`). The end-of-generation tokens are not among them.
  std::vector<int32_t> never;

  // Allowed sets by state, shared by every request: a set depends only on the state and on the grammar (choice, parallel,
  // the tool names), so requests with the same tools share. Cleared when it holds too many ids.
  using Set = std::shared_ptr<const std::vector<int32_t>>;
  mutable std::mutex mu;
  mutable std::unordered_map<std::string, Set> cache;
  mutable size_t cached_ids = 0;
};

class InklingToolGrammar {
 public:
  enum class Choice { Auto, Required };  // tool_choice none, or no tools: no grammar at all

  // Required needs at least one tool name.
  InklingToolGrammar(std::shared_ptr<const InklingToolVocab> v, Choice c, std::vector<std::string> tool_names, bool parallel);

  // The token ids the grammar allows next, ascending (the end-of-generation ids included where the turn may end). The
  // reference stays valid until the next accept() or allowed().
  const std::vector<int32_t>& allowed();
  // Every token is allowed next but the ones in InklingToolVocab::never (llama.cpp's grammar never accepts them): auto mode before
  // a call, with no character pending. The sampler can skip the mask if it bans exactly `never`. After a token that ends inside a
  // character this is false: use allowed().
  bool unconstrained() const;
  // Follows a sampled token; false (state unchanged) if it was not allowed.
  bool accept(int32_t id);

  // The turn may end here (an end-of-generation token is allowed).
  bool can_end() const;
  // An end-of-generation token was accepted.
  bool ended() const;

 private:
  struct Tools;  // names as code points, per tool
  struct State {
    uint8_t mode = 0, item = 0, sp = 0, sub = 0, jm = 0;
    bool blocks_ok = false;  // required: thinking and text blocks may still come
    bool all = true;         // every tool is still possible (alive is empty)
    uint32_t off = 0;        // characters of the name, or of a literal, matched
    InklingToolVocab::Pu pu; // a character begun by the last token
    std::string stack;       // the call arguments' open containers: 'a' array, 'o' object
    std::vector<uint32_t> alive;  // the tools whose name matches so far, when not all
  };
  bool accepting(const State& s) const;
  bool opaque(int32_t id) const;
  bool partial_ok(const State& s, InklingToolVocab::Pu pu) const;
  bool step_cp(State& s, uint32_t c) const;
  std::string key(const State& s) const;
  InklingToolVocab::Set compute(const State& s) const;
  bool feed(State& s, int32_t id) const;

  std::shared_ptr<const InklingToolVocab> v_;
  std::shared_ptr<const Tools> tools_;
  Choice choice_;
  bool parallel_;
  std::string prefix_;  // the cache key's part that names the grammar
  State st_;
  InklingToolVocab::Set allowed_;
  bool allowed_valid_ = false;
};

}  // namespace eklow::serve
