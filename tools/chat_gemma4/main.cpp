// eklow-chat-gemma4: the two Gemma 4 chat template ports (Google's, Bolt's) from the command line.
//
//   eklow-chat-gemma4 render [--tokens GGUF] [--variant google|bolt] < CASES > RESULTS
//                                         CASES: one JSON object per line, {"messages": [...],
//                                         "tools": [...] or null, "add_generation_prompt": bool,
//                                         "enable_thinking": bool, "preserve_thinking": bool,
//                                         "conversational_mode": bool (Bolt)};
//                                         RESULTS: one line per case, {"text": ..., "parts":
//                                         [[text, is_input], ...]} or {"error": ...}; with --tokens,
//                                         also "ids", the prompt tokenized with special tokens
//                                         recognized only in template text.
//                                         The template is --variant, or with --tokens the variant the GGUF's chat
//                                         template is.
//   eklow-chat-gemma4 check FILE          which of the ported templates FILE carries: a GGUF's tokenizer.chat_template,
//                                         or any other file read whole (a chat_template.jinja)
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
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

std::string template_text(const std::string& path) {
  if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".gguf") == 0)
    return gguf::File::open(path).at("tokenizer.chat_template").as_string();
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(path + ": cannot open");
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "check" && argc > 2) {
    try {
      auto which = text::gemma4_template_of(template_text(argv[2]));
      std::printf("%s: chat template %s\n", argv[2],
                  which ? (std::string("is the ported Gemma 4 ") + text::gemma4_template_name(*which) + " template").c_str()
                        : "DIFFERS from the ported Gemma 4 templates (google, bolt)");
      return which ? 0 : 1;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "eklow-chat-gemma4: %s\n", e.what());
      return 1;
    }
  }
  std::string tokens_path, variant;
  bool usage = mode != "render";
  for (int i = 2; i < argc && !usage; ++i) {
    if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_path = argv[++i];
    else if (!std::strcmp(argv[i], "--variant") && i + 1 < argc) variant = argv[++i];
    else usage = true;
  }
  if (usage || (!variant.empty() && variant != "google" && variant != "bolt")) {
    std::fprintf(stderr, "usage: eklow-chat-gemma4 render [--tokens GGUF] [--variant google|bolt] < CASES > RESULTS | eklow-chat-gemma4 check FILE\n");
    return 2;
  }
  try {
    std::optional<gguf::File> file;
    std::optional<text::Tokenizer> tok;
    std::optional<text::Gemma4Template> which;
    if (variant == "google") which = text::Gemma4Template::Google;
    if (variant == "bolt") which = text::Gemma4Template::Bolt;
    if (!tokens_path.empty()) {
      file.emplace(gguf::File::open(tokens_path));
      tok.emplace(text::Tokenizer::from_gguf(*file));
      if (!which) {
        which = text::gemma4_template_of(file->at("tokenizer.chat_template").as_string());
        if (!which) throw std::runtime_error(tokens_path + ": the chat template is none of the ported Gemma 4 templates");
      }
    }
    if (!which) throw std::runtime_error("no template: give --variant or --tokens");
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
        opt.conversational_mode = flag(c, "conversational_mode");
        auto parts = text::render_gemma4(*which, messages ? *messages : json::Value::array(), tools ? *tools : json::Value(), opt);
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
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-chat-gemma4: %s\n", e.what());
    return 1;
  }
  return 0;
}
