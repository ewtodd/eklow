// eklow-parse: Qwen3.6's output parser from the command line.
//
//   eklow-parse GGUF [--thinking] < IDS > RESULTS   IDS: one generated turn per line, token ids
//                                                   separated by spaces, fed one at a time;
//                                                   RESULTS: one JSON object per line: reasoning,
//                                                   content, tool_calls, errors, end (token id)
//
// A line may start with the request's tools as a JSON array
// and a TAB, which say what the parameters' types are; --thinking starts the turn inside the reasoning (the
// prompt ended "<think>\n"). Tokens are fed one at a time, as the server does.
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/output.hpp"
#include "text/output_parser_qwen.hpp"
#include "text/pyfmt.hpp"
#include "text/tokenizer_qwen.hpp"

using namespace eklow;

namespace {

void take(text::OutputDelta& all, text::OutputDelta d) {
  all.reasoning += d.reasoning;
  all.content += d.content;
  for (auto& c : d.tool_calls) all.tool_calls.push_back(std::move(c));
  for (auto& e : d.errors) all.errors.push_back(std::move(e));
  if (d.done) all.end_token = d.end_token;
}

// A value as JSON that keeps floats floats: json::dump writes 1.0 as 1, and the checks tell the two apart.
void write_value(json::Writer& w, const json::Value& v) {
  switch (v.kind()) {
    case json::Value::Kind::Float: w.raw(text::py::float_repr(v.as_double())); break;
    case json::Value::Kind::Array:
      w.begin_array();
      for (const auto& x : v.items()) write_value(w, x);
      w.end_array();
      break;
    case json::Value::Kind::Object:
      w.begin_object();
      for (const auto& [k, x] : v.members()) {
        w.key(k);
        write_value(w, x);
      }
      w.end_object();
      break;
    case json::Value::Kind::Null: w.null(); break;
    case json::Value::Kind::Bool: w.value(v.as_bool()); break;
    case json::Value::Kind::Int: w.raw(v.int_digits()); break;
    case json::Value::Kind::String: w.value(v.as_string()); break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: eklow-parse GGUF [--thinking] < IDS > RESULTS\n");
    return 2;
  }
  const bool thinking = argc > 2 && !std::strcmp(argv[2], "--thinking");
  try {
    auto f = gguf::File::open(argv[1]);
    const auto q = text::QwenTokenizer::from_gguf(f);
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
      text::ToolSpecs specs;
      if (size_t tab = line.find('\t'); tab != std::string::npos) {
        if (tab) specs = text::ToolSpecs(json::parse(line.substr(0, tab)));
        line.erase(0, tab + 1);
      }
      text::OutputDelta all;
      std::istringstream ss(line);
      text::QwenOutputParser p(q, thinking, &specs);
      for (int32_t id; ss >> id;) take(all, p.push(id));
      take(all, p.finish());
      json::Writer w(false);
      w.begin_object().kv("reasoning", all.reasoning).kv("content", all.content).key("tool_calls").begin_array();
      for (const auto& c : all.tool_calls) {
        w.begin_object().kv("name", c.name).key("arguments");
        write_value(w, c.arguments);
        w.end_object();
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
