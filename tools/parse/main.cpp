// eklow-parse: Gemma 4's output parser from the command line.
//
//   eklow-parse GGUF [--thinking] < IDS > RESULTS   IDS: one generated turn per line, token ids
//                                                   separated by spaces, fed one at a time;
//                                                   RESULTS: one JSON object per line: reasoning,
//                                                   content, tool_calls, errors, end (token id)
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/output_parser.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: eklow-parse GGUF [--thinking] < IDS > RESULTS\n");
    return 2;
  }
  const bool thinking = argc > 2 && !std::strcmp(argv[2], "--thinking");
  try {
    auto f = gguf::File::open(argv[1]);
    auto tok = text::Tokenizer::from_gguf(f);
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
      text::OutputParser p(tok, thinking);
      text::OutputDelta all;
      auto take = [&](text::OutputDelta d) {
        all.reasoning += d.reasoning;
        all.content += d.content;
        for (auto& c : d.tool_calls) all.tool_calls.push_back(std::move(c));
        for (auto& e : d.errors) all.errors.push_back(std::move(e));
        if (d.done) all.end_token = d.end_token;
      };
      std::istringstream ss(line);
      for (int32_t id; ss >> id;) take(p.push(id));
      take(p.finish());
      json::Writer w(false);
      w.begin_object().kv("reasoning", all.reasoning).kv("content", all.content).key("tool_calls").begin_array();
      for (const auto& c : all.tool_calls) {
        w.begin_object().kv("name", c.name).key("arguments").raw(json::dump(c.arguments)).end_object();
      }
      w.end_array().key("errors").begin_array();
      for (const auto& e : all.errors) w.value(e);
      w.end_array().kv("end", int64_t(all.end_token)).end_object();
      std::cout << w.str() << '\n';
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-parse: %s\n", e.what());
    return 1;
  }
  return 0;
}
