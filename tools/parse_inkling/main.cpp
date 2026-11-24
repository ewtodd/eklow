// eklow-parse-inkling: Inkling 2 Mini's output parser from the command line.
//
//   eklow-parse-inkling GGUF < TURNS > RESULTS
//     TURNS: one JSON object per line: {"ids": [token ids, fed one at a time], "tools": [names], "tool_choice":
//            "auto"|"none"|"required", "parallel_tool_calls": bool, "response_format": bool, "extract_reasoning": bool,
//            "prefill": [token ids of a continued assistant message]} (all but ids optional: the defaults are
//            llama-server's: no tools, parallel calls, reasoning extracted);
//     RESULTS: one JSON object per line: "message" (reasoning, content, tool_calls, errors: everything, the prefill
//            included), "prefilled" (the message after the prefill alone), "streamed" (what the deltas carried, the prefill
//            not), "end" (the token that ended the turn, or -1) and "deltas" (per token: the reasoning and content it
//            released, the number of calls and errors).
#include <cstdio>
#include <iostream>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/output_parser_inkling.hpp"
#include "text/tokenizer_inkling.hpp"

using namespace eklow;

namespace {

void write_message(json::Writer& w, const std::string& reasoning, const std::string& content, const std::vector<text::ToolCall>& calls,
                   const std::vector<std::string>& errors) {
  w.begin_object().kv("reasoning", reasoning).kv("content", content).key("tool_calls").begin_array();
  for (const auto& c : calls)
    w.begin_object().kv("name", c.name).kv("arguments_text", c.arguments_text).key("arguments").raw(json::dump(c.arguments)).kv("truncated", c.truncated).end_object();
  w.end_array().key("errors").begin_array();
  for (const auto& e : errors) w.value(e);
  w.end_array().end_object();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: eklow-parse-inkling GGUF < TURNS > RESULTS\n");
    return 2;
  }
  try {
    auto f = gguf::File::open(argv[1]);
    auto tok = text::InklingTokenizer::from_gguf(f);
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
      const json::Value turn = json::parse(line);
      text::InklingMode mode;
      if (const auto* t = turn.find("tools"))
        for (const auto& n : t->items()) mode.tool_names.push_back(n.as_string());
      if (const auto* c = turn.find("tool_choice"))
        mode.tool_choice = c->as_string() == "none" ? text::InklingToolChoice::None
                           : c->as_string() == "required" ? text::InklingToolChoice::Required
                                                          : text::InklingToolChoice::Auto;
      auto flag = [&](const char* key, bool def) {
        const auto* v = turn.find(key);
        return v ? v->as_bool() : def;
      };
      mode.parallel_tool_calls = flag("parallel_tool_calls", true);
      mode.response_format = flag("response_format", false);
      mode.extract_reasoning = flag("extract_reasoning", true);
      std::vector<int32_t> prefill;
      if (const auto* p = turn.find("prefill"))
        for (const auto& v : p->items()) prefill.push_back(int32_t(std::stoll(v.int_digits())));
      text::InklingOutputParser parser(tok, mode, prefill);
      const text::InklingOutputParser::Message prefilled = parser.message();
      std::string reasoning, content;
      std::vector<text::ToolCall> calls;
      std::vector<std::string> errors;
      int64_t end = -1;
      json::Writer deltas(false);
      deltas.begin_array();
      auto take = [&](const text::OutputDelta& d) {
        reasoning += d.reasoning;
        content += d.content;
        for (const auto& c : d.tool_calls) calls.push_back(c);
        for (const auto& e : d.errors) errors.push_back(e);
        if (d.done) end = d.end_token;
        deltas.begin_object().kv("reasoning", d.reasoning).kv("content", d.content).kv("calls", int64_t(d.tool_calls.size()))
            .kv("errors", int64_t(d.errors.size())).end_object();
      };
      for (const auto& v : turn.find("ids")->items()) take(parser.push(int32_t(std::stoll(v.int_digits()))));
      if (end < 0) take(parser.finish());
      deltas.end_array();
      json::Writer w(false);
      w.begin_object().key("message");
      const auto& m = parser.message();
      write_message(w, m.reasoning, m.content, m.tool_calls, m.errors);
      w.key("prefilled");
      write_message(w, prefilled.reasoning, prefilled.content, prefilled.tool_calls, prefilled.errors);
      w.key("streamed");
      write_message(w, reasoning, content, calls, errors);
      w.kv("end", end).key("deltas").raw(deltas.str()).end_object();
      std::cout << w.str() << '\n';
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-parse-inkling: %s\n", e.what());
    return 1;
  }
  return 0;
}
