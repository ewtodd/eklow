// eklow-chat: the Gemma 4 chat template port from the command line.
//
//   eklow-chat render [--tokens GGUF] < CASES > RESULTS
//                                         CASES: one JSON object per line, {"messages": [...],
//                                         "tools": [...] or null, "add_generation_prompt": bool,
//                                         "enable_thinking": bool, "preserve_thinking": bool};
//                                         RESULTS: one line per case, {"text": ..., "parts":
//                                         [[text, is_input], ...]} or {"error": ...}; with --tokens,
//                                         also "ids", the prompt tokenized with special tokens
//                                         recognized only in template text
//   eklow-chat check GGUF                 whether the GGUF's chat template is the ported one
#include <cstdio>
#include <iostream>
#include <optional>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/chat_gemma4.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

namespace {

bool flag(const json::Value& c, const char* key) {
  const json::Value* v = c.find(key);
  return v && v->truthy();
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "check" && argc > 2) {
    try {
      auto f = gguf::File::open(argv[2]);
      bool ok = text::is_gemma4_template(f.at("tokenizer.chat_template").as_string());
      std::printf("%s: chat template %s\n", argv[2], ok ? "is the ported Gemma 4 template" : "DIFFERS from the port");
      return ok ? 0 : 1;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "eklow-chat: %s\n", e.what());
      return 1;
    }
  }
  if (mode != "render" || (argc > 2 && (argc != 4 || std::string(argv[2]) != "--tokens"))) {
    std::fprintf(stderr, "usage: eklow-chat render [--tokens GGUF] < CASES > RESULTS | eklow-chat check GGUF\n");
    return 2;
  }
  std::optional<gguf::File> file;
  std::optional<text::Tokenizer> tok;
  if (argc == 4) {
    file.emplace(gguf::File::open(argv[3]));
    tok.emplace(text::Tokenizer::from_gguf(*file));
  }
  std::ios::sync_with_stdio(false);
  std::string line;
  while (std::getline(std::cin, line)) {
    json::Writer w(false);
    try {
      json::Value c = json::parse(line);
      const json::Value* messages = c.find("messages");
      const json::Value* tools = c.find("tools");
      text::ChatOptions opt;
      opt.add_generation_prompt = flag(c, "add_generation_prompt");
      opt.enable_thinking = flag(c, "enable_thinking");
      opt.preserve_thinking = flag(c, "preserve_thinking");
      auto parts = text::render_gemma4(messages ? *messages : json::Value::array(), tools ? *tools : json::Value(), opt);
      w.begin_object().kv("text", text::join(parts)).key("parts").begin_array();
      for (const auto& p : parts) w.begin_array().value(p.text).value(p.input).end_array();
      w.end_array();
      if (tok) {
        std::vector<text::Tokenizer::Piece> pieces;
        for (const auto& p : parts) pieces.push_back({p.text, !p.input});
        w.key("ids").begin_array();
        for (int32_t id : tok->encode(pieces)) w.value(int64_t(id));
        w.end_array();
      }
      w.end_object();
    } catch (const std::exception& e) {
      w = json::Writer(false);
      w.begin_object().kv("error", std::string(e.what())).end_object();
    }
    std::cout << w.str() << '\n';
  }
  return 0;
}
