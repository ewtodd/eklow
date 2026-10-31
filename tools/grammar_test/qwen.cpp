// eklow-grammar-test-qwen: the Qwen3.6 tool-call grammar (serve/tool_grammar_qwen.hpp) against the output parser and
// the tokenizer.
//
//   eklow-grammar-test-qwen --model GGUF [--walks N] [--seed S]
//       Random walks: random tool declarations (every kind of parameter, enums, required names); each walk is a
//       few random tokens drawn from the allowed sets, then the grammar's shortest completion tokenized as the
//       tokenizer would, then </tool_call>. Every token must be allowed when drawn, and the call must parse into
//       a declared tool with every required parameter, each parameter once, and values of the declared kinds.
//       Bans: in each state of the output parser, the special tokens it bans are exactly those it refuses.
//
//   eklow-grammar-test-qwen --model GGUF --lines < LINES > RESULTS
//       LINES: the request's tools as a JSON array, a TAB, token ids separated by spaces (a turn, or the part of
//       one that holds calls). RESULTS: one line each, "ok" if every token inside a call was allowed, else
//       "refused N" with the index of the first one that was not (and "inconsistent N" if the allowed set and
//       accept() ever disagree). tools/ref/qwen_grammar_compare.py drives it.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "serve/tool_grammar_qwen.hpp"
#include "text/output_parser_qwen.hpp"
#include "text/tokenizer_qwen.hpp"
#include "text/tool_schema.hpp"
#include "text/unicode.hpp"

using namespace eklow;

namespace {

// A parameter schema, as the JSON a request would carry.
const std::vector<std::string> kSchemas = {
    R"({"type": "string"})",
    R"({"type": "string", "description": "free text"})",
    R"({})",
    R"({"description": "no type"})",
    R"({"type": ["string", "null"]})",
    R"({"type": "integer"})",
    R"({"type": "int"})",
    R"({"type": "number"})",
    R"({"type": ["integer", "null"]})",
    R"({"type": "boolean"})",
    R"({"type": "null"})",
    R"({"type": "object", "properties": {"a": {"type": "string"}}})",
    R"({"type": "array", "items": {"type": "string"}})",
    R"({"type": ["object", "null"]})",
    R"({"anyOf": [{"type": "integer"}, {"type": "null"}]})",
    R"({"anyOf": [{"type": "array"}, {"type": "boolean"}]})",
    R"({"type": "string", "enum": ["c", "f", "kelvin"]})",
    R"({"type": "string", "enum": ["a", "ab", "abc", "", "é日本", "x y"]})",
    R"({"type": "integer", "enum": [1, 2, 30]})",
    R"({"type": "number", "enum": [0.5, 1.0, -2]})",
    R"({"enum": [true, false, null]})",
    R"({"type": "string", "enum": ["multi\nline", "ok"]})",
};

const std::vector<std::string> kKeys = {"city", "query", "path", "content", "unit", "n", "items", "a_b", "deep-key", "日本", "with space", "k1", "k2", "k3"};
const std::vector<std::string> kNames = {"get_weather", "web_search", "write_file", "todo", "terminal", "f", "名前"};

json::Value random_tools(std::mt19937_64& rng) {
  std::vector<json::Value> tools;
  std::vector<std::string> names = kNames;
  std::shuffle(names.begin(), names.end(), rng);
  for (size_t t = 0, n = 1 + rng() % 3; t < n; ++t) {
    std::vector<std::string> keys = kKeys;
    std::shuffle(keys.begin(), keys.end(), rng);
    std::vector<json::Value::Member> props;
    std::vector<json::Value> required;
    for (size_t k = 0, m = rng() % 5; k < m; ++k) {
      props.emplace_back(keys[k], json::parse(kSchemas[rng() % kSchemas.size()]));
      if (rng() % 2) required.push_back(json::Value::string(keys[k]));
    }
    if (rng() % 8 == 0) required.push_back(json::Value::string("undeclared_but_required"));
    json::Value params = json::Value::object();
    params.set("type", json::Value::string("object"));
    params.set("properties", json::Value::object(std::move(props)));
    params.set("required", json::Value::array(std::move(required)));
    json::Value fn = json::Value::object();
    fn.set("name", json::Value::string(names[t]));
    fn.set("description", json::Value::string("d"));
    fn.set("parameters", std::move(params));
    json::Value tool = json::Value::object();
    tool.set("type", json::Value::string("function"));
    tool.set("function", std::move(fn));
    tools.push_back(std::move(tool));
  }
  return json::Value::array(std::move(tools));
}

// Whether the parsed value is one the parameter's declared kinds allow (the grammar wrote it as JSON of those kinds,
// as free text, or as one member of an enum).
bool value_ok(const text::ParamSpec& p, const json::Value& v, const std::string& raw_body_hint) {
  (void)raw_body_hint;
  if (!p.enum_values.empty()) {
    // The grammar fixes the text for an enum of scalars none of which holds a newline or the closing tag.
    bool scalar = true;
    for (const auto& e : p.enum_values)
      scalar &= !e.is_array() && !e.is_object() && !(e.is_string() && (e.as_string().find('\n') != std::string::npos || e.as_string().find("</parameter>") != std::string::npos));
    if (scalar) {
      for (const auto& e : p.enum_values) {
        std::string raw = e.is_string() ? e.as_string() : e.is_null() ? "null" : e.is_bool() ? (e.as_bool() ? "true" : "false") : e.is_int() ? e.int_digits() : "";
        if (e.is_float()) continue;  // compared below by value
        if (json::dump(text::convert_param(&p, raw)) == json::dump(v)) return true;
      }
      for (const auto& e : p.enum_values)
        if (e.is_float() && v.is_float() && e.as_double() == v.as_double()) return true;
      return false;
    }
  }
  if (p.kinds == text::kString) return v.is_string();
  if (p.kinds & text::kString) return true;  // a union with strings: whatever the text turned out to be
  switch (v.kind()) {
    case json::Value::Kind::Null: return p.kinds & text::kNull;
    case json::Value::Kind::Bool: return p.kinds & text::kBoolean;
    case json::Value::Kind::Int: return p.kinds & (text::kInteger | text::kNumber);
    case json::Value::Kind::Float: return p.kinds & text::kNumber;
    case json::Value::Kind::Array: return p.kinds & text::kArray;
    case json::Value::Kind::Object: return p.kinds & text::kObject;
    case json::Value::Kind::String: return false;
  }
  return false;
}

size_t count_of(const std::string& s, const std::string& needle) {
  size_t n = 0;
  for (size_t i = s.find(needle); i != std::string::npos; i = s.find(needle, i + 1)) ++n;
  return n;
}

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  int walks = 2000;
  uint64_t seed = 1;
  bool lines = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") path = next();
    else if (a == "--walks") walks = std::atoi(next().c_str());
    else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--lines") lines = true;
    else {
      std::fprintf(stderr, "usage: eklow-grammar-test-qwen --model GGUF [--walks N] [--seed S] | --lines\n");
      return 2;
    }
  }
  try {
    auto f = gguf::File::open(path);
    const text::QwenTokenizer tok = text::QwenTokenizer::from_gguf(f);
    auto vocab = std::make_shared<const serve::QwenToolVocab>(tok);
    std::vector<int32_t> byte_id(256, -1);
    for (int32_t id = 0; id < int32_t(tok.size()); ++id)
      if (vocab->bytes[size_t(id)].size() == 1) byte_id[uint8_t(vocab->bytes[size_t(id)][0])] = id;

    if (lines) {
      std::ios::sync_with_stdio(false);
      std::string line;
      while (std::getline(std::cin, line)) {
        const size_t tab = line.find('\t');
        auto specs = std::make_shared<const text::ToolSpecs>(tab != std::string::npos && tab > 0 ? json::parse(line.substr(0, tab)) : json::Value());
        serve::QwenToolGrammar gr(vocab, specs);
        std::istringstream ss(tab == std::string::npos ? line : line.substr(tab + 1));
        std::string verdict = "ok";
        size_t index = 0;
        for (int32_t id; ss >> id; ++index) {
          const bool constrained = gr.active();
          bool listed = !constrained;
          if (constrained) {
            const auto& a = gr.allowed();
            listed = std::find(a.begin(), a.end(), id) != a.end();
          }
          const bool accepted = gr.accept(id);
          if (listed != accepted) {
            verdict = "inconsistent " + std::to_string(index);
            break;
          }
          if (!accepted) {
            verdict = "refused " + std::to_string(index);
            break;
          }
        }
        std::cout << verdict << '\n';
      }
      return 0;
    }

    std::mt19937_64 rng(seed);
    int closed = 0, bad_walks = 0;
    size_t checked_params = 0;
    for (int w = 0; w < walks; ++w) {
      const json::Value tools = random_tools(rng);
      auto specs = std::make_shared<const text::ToolSpecs>(tools);
      serve::QwenToolGrammar gr(vocab, specs);
      std::vector<int32_t> seq{vocab->call_open};
      auto fail = [&](const std::string& what) {
        if (++bad_walks <= 8) {
          std::string body;
          for (size_t i = 1; i < seq.size(); ++i) body += seq[i] == vocab->call_close ? "<|/call|>" : vocab->bytes[size_t(seq[i])];
          std::fprintf(stderr, "walk %d: %s\n  tools %s\n  call %s\n", w, what.c_str(), json::dump(tools).c_str(), body.c_str());
        }
      };
      gr.accept(vocab->call_open);
      bool ok = true;
      for (size_t steps = rng() % 40, k = 0; k < steps && ok; ++k) {
        const auto& allowed = gr.allowed();
        if (allowed.empty()) {
          fail("nothing allowed");
          ok = false;
          break;
        }
        const int32_t pick = allowed[rng() % allowed.size()];
        if (pick == vocab->call_close) break;
        if (!gr.accept(pick)) {
          fail("an allowed token was refused");
          ok = false;
          break;
        }
        seq.push_back(pick);
      }
      if (!ok) continue;
      // The rest: the grammar's shortest completion, as the tokenizer would split it, then the closing token.
      std::string rest;
      try {
        rest = gr.completion();
      } catch (const std::exception& e) {
        fail(e.what());
        continue;
      }
      // The tokenizer splits text; a rest that stops inside a character (the walk drew half of one) goes byte by byte.
      std::vector<int32_t> rest_ids;
      if (text::unicode::valid_utf8(rest)) {
        rest_ids = tok.encode(rest, false);
      } else {
        for (unsigned char c : rest) rest_ids.push_back(byte_id[c]);
      }
      for (int32_t id : rest_ids) {
        const auto& allowed = gr.allowed();
        if (!std::binary_search(allowed.begin(), allowed.end(), id) && std::find(allowed.begin(), allowed.end(), id) == allowed.end()) {
          fail("a token of the completion '" + rest + "' is not allowed");
          ok = false;
          break;
        }
        gr.accept(id);
        seq.push_back(id);
      }
      if (!ok) continue;
      {
        const auto& allowed = gr.allowed();
        if (std::find(allowed.begin(), allowed.end(), vocab->call_close) == allowed.end() || !gr.accept(vocab->call_close)) {
          fail("the call cannot close after its completion");
          continue;
        }
        seq.push_back(vocab->call_close);
      }
      if (gr.active()) {
        fail("still active after </tool_call>");
        continue;
      }
      ++closed;
      text::QwenOutputParser parser(tok, false, specs.get());
      text::OutputDelta d;
      for (int32_t id : seq) {
        auto x = parser.push(id);
        for (auto& c : x.tool_calls) d.tool_calls.push_back(std::move(c));
        for (auto& e : x.errors) d.errors.push_back(std::move(e));
      }
      if (!d.errors.empty() || d.tool_calls.size() != 1) {
        fail(d.errors.empty() ? "no call parsed" : "does not parse: " + d.errors[0]);
        continue;
      }
      const auto& call = d.tool_calls[0];
      const text::ToolSpec* spec = specs->find(call.name);
      if (!spec) {
        fail("call of an undeclared tool " + call.name);
        continue;
      }
      std::string body;
      for (size_t i = 1; i + 1 < seq.size(); ++i) body += vocab->bytes[size_t(seq[i])];
      if (count_of(body, "\n<parameter=") != call.arguments.members().size()) {
        fail("a parameter was written twice");
        continue;
      }
      bool good = true;
      for (const auto& p : spec->params)
        if (p.required && !call.arguments.find(p.name)) {
          fail("required parameter " + p.name + " missing");
          good = false;
        }
      for (const auto& [key, value] : call.arguments.members()) {
        const text::ParamSpec* p = spec->param(key);
        if (!p) {
          fail("undeclared parameter " + key);
          good = false;
        } else if (!value_ok(*p, value, body)) {
          fail("value " + json::dump(value) + " of " + key + " is not of its declared kind");
          good = false;
        }
        ++checked_params;
      }
      (void)good;
    }

    // Bans: in each state of the output parser, the special tokens it bans (never sampled) are exactly those it refuses.
    int bad_bans = 0, checked = 0;
    {
      // A call with its body written, so that closing it is well formed.
      std::vector<int32_t> call{vocab->call_open};
      for (int32_t id : tok.encode("\n<function=todo>\n</function>\n", false)) call.push_back(id);
      const std::vector<std::pair<const char*, std::pair<bool, std::vector<int32_t>>>> states = {
          {"content", {false, {}}}, {"reasoning", {true, {}}}, {"tool call", {false, call}}};
      for (const auto& [name, st] : states) {
        text::QwenOutputParser p(tok, st.first);
        for (int32_t id : st.second) p.push(id);
        const std::vector<int32_t>& banned = p.banned();
        for (int32_t s : tok.special_ids()) {
          text::QwenOutputParser q = p;
          const bool refused = !q.push(s).errors.empty(), is_banned = std::binary_search(banned.begin(), banned.end(), s);
          ++checked;
          if (refused != is_banned && ++bad_bans <= 5)
            std::fprintf(stderr, "%s: %s is %s but %s\n", name, tok.piece(s).c_str(), refused ? "refused" : "taken", is_banned ? "banned" : "not banned");
        }
      }
    }
    std::printf("walks: %d of %d closed into calls the parser accepts (%zu parameter values checked), %d failed\n", closed, walks, checked_params, bad_walks);
    std::printf("bans: %d of %d (state, special token) pairs banned exactly when refused\n", checked - bad_bans, checked);
    return bad_walks || bad_bans || closed != walks ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-grammar-test-qwen: %s\n", e.what());
    return 1;
  }
}
