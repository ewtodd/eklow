// eklow-chat: Qwen3.6's three chat template ports from the command line.
//
//   eklow-chat render [--tokens GGUF] [--variant stock|ornith|sage] < CASES > RESULTS
//                                         CASES: one JSON object per line, {"messages": [...],
//                                         "tools": [...] or null, "add_generation_prompt": bool,
//                                         "enable_thinking": bool, "preserve_thinking": bool,
//                                         "add_vision_id": bool, "verification": any (Sage)};
//                                         RESULTS: one line per case, {"text": ..., "parts":
//                                         [[text, is_input], ...]} or {"error": ...}; with --tokens,
//                                         also "ids", the prompt tokenized with special tokens
//                                         recognized only in template text.
//                                         The template is --variant, or with --tokens the variant the GGUF's chat
//                                         template is. enable_thinking is on unless a case says false.
//   eklow-chat check GGUF                 which of the ported templates the GGUF's chat template is
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/chat.hpp"
#include "text/chat_qwen36.hpp"
#include "text/tokenizer_qwen.hpp"

using namespace eklow;

namespace {

bool flag(const json::Value& c, const char* key, bool otherwise = false) {
  const json::Value* v = c.find(key);
  return v ? v->truthy() : otherwise;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "check" && argc > 2) {
    try {
      auto f = gguf::File::open(argv[2]);
      const std::string& tmpl = f.at("tokenizer.chat_template").as_string();
      auto which = text::qwen36_template_of(tmpl);
      std::printf("%s: chat template %s\n", argv[2],
                  which ? (std::string("is the ported Qwen3.6 ") + text::qwen36_template_name(*which) + " template").c_str()
                        : "DIFFERS from the ported Qwen3.6 templates (stock, ornith, sage)");
      return which ? 0 : 1;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "eklow-chat: %s\n", e.what());
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
  if (usage || (!variant.empty() && variant != "stock" && variant != "ornith" && variant != "sage")) {
    std::fprintf(stderr, "usage: eklow-chat render [--tokens GGUF] [--variant stock|ornith|sage] < CASES > RESULTS | eklow-chat check GGUF\n");
    return 2;
  }
  try {
    std::optional<gguf::File> file;
    std::optional<text::QwenTokenizer> qwen_tok;
    std::optional<text::QwenTemplate> qwen;
    if (variant == "stock") qwen = text::QwenTemplate::Stock;
    if (variant == "ornith") qwen = text::QwenTemplate::Ornith;
    if (variant == "sage") qwen = text::QwenTemplate::Sage;
    if (!tokens_path.empty()) {
      file.emplace(gguf::File::open(tokens_path));
      qwen_tok.emplace(text::QwenTokenizer::from_gguf(*file));
      if (!qwen) {
        qwen = text::qwen36_template_of(file->at("tokenizer.chat_template").as_string());
        if (!qwen) throw std::runtime_error(tokens_path + ": the chat template is none of the ported Qwen3.6 templates");
      }
    }
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
      json::Writer w(false);
      try {
        json::Value c = json::parse(line);
        const json::Value* messages = c.find("messages");
        const json::Value* tools = c.find("tools");
        std::vector<text::ChatPart> parts;
        text::QwenChatOptions opt;
        opt.add_generation_prompt = flag(c, "add_generation_prompt");
        opt.enable_thinking = flag(c, "enable_thinking", true);
        opt.preserve_thinking = flag(c, "preserve_thinking");
        opt.add_vision_id = flag(c, "add_vision_id");
        if (const json::Value* v = c.find("verification")) opt.verification = *v;
        if (!qwen) throw std::runtime_error("no template: give --variant or --tokens");
        parts = text::render_qwen36(*qwen, messages ? *messages : json::Value::array(), tools ? *tools : json::Value(), opt);
        w.begin_object().kv("text", text::join(parts)).key("parts").begin_array();
        for (const auto& p : parts) w.begin_array().value(p.text).value(p.input).end_array();
        w.end_array();
        if (qwen_tok) {
          w.key("ids").begin_array();
          std::vector<text::QwenTokenizer::Piece> pieces;
          for (const auto& p : parts) pieces.push_back({p.text, !p.input});
          for (int32_t id : qwen_tok->encode(pieces)) w.value(int64_t(id));
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
    std::fprintf(stderr, "eklow-chat: %s\n", e.what());
    return 1;
  }
  return 0;
}
