// eklow-chat: the chat template ports (Gemma 4's, and Qwen3.6's two) from the command line.
//
//   eklow-chat render [--tokens GGUF] [--variant stock|ornith] < CASES > RESULTS
//                                         CASES: one JSON object per line, {"messages": [...],
//                                         "tools": [...] or null, "add_generation_prompt": bool,
//                                         "enable_thinking": bool, "preserve_thinking": bool,
//                                         "add_vision_id": bool (Qwen)};
//                                         RESULTS: one line per case, {"text": ..., "parts":
//                                         [[text, is_input], ...]} or {"error": ...}; with --tokens,
//                                         also "ids", the prompt tokenized with special tokens
//                                         recognized only in template text.
//                                         Qwen3.6 is rendered with --variant, or when --tokens names a Qwen GGUF
//                                         with the variant its chat template is; otherwise Gemma 4. For Qwen,
//                                         enable_thinking is on unless a case says false.
//   eklow-chat check GGUF                 which of the ported templates the GGUF's chat template is
#include <cstdio>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "text/chat_gemma4.hpp"
#include "text/chat_qwen36.hpp"
#include "text/tokenizer.hpp"
#include "text/tokenizer_qwen.hpp"

using namespace eklow;

namespace {

bool flag(const json::Value& c, const char* key, bool otherwise = false) {
  const json::Value* v = c.find(key);
  return v ? v->truthy() : otherwise;
}

bool is_qwen(const gguf::File& f) {
  const auto* m = f.find("tokenizer.ggml.model");
  return m && m->as_string() == "gpt2";
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "check" && argc > 2) {
    try {
      auto f = gguf::File::open(argv[2]);
      const std::string& tmpl = f.at("tokenizer.chat_template").as_string();
      if (is_qwen(f)) {
        auto which = text::qwen36_template_of(tmpl);
        std::printf("%s: chat template %s\n", argv[2],
                    which ? (std::string("is the ported Qwen3.6 ") + text::qwen36_template_name(*which) + " template").c_str()
                          : "DIFFERS from both ported Qwen3.6 templates");
        return which ? 0 : 1;
      }
      bool ok = text::is_gemma4_template(tmpl);
      std::printf("%s: chat template %s\n", argv[2], ok ? "is the ported Gemma 4 template" : "DIFFERS from the port");
      return ok ? 0 : 1;
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
  if (usage || (!variant.empty() && variant != "stock" && variant != "ornith")) {
    std::fprintf(stderr, "usage: eklow-chat render [--tokens GGUF] [--variant stock|ornith] < CASES > RESULTS | eklow-chat check GGUF\n");
    return 2;
  }
  try {
    std::optional<gguf::File> file;
    std::optional<text::Tokenizer> gemma_tok;
    std::optional<text::QwenTokenizer> qwen_tok;
    std::optional<text::QwenTemplate> qwen;
    if (variant == "stock") qwen = text::QwenTemplate::Stock;
    if (variant == "ornith") qwen = text::QwenTemplate::Ornith;
    if (!tokens_path.empty()) {
      file.emplace(gguf::File::open(tokens_path));
      if (is_qwen(*file)) {
        qwen_tok.emplace(text::QwenTokenizer::from_gguf(*file));
        if (!qwen) {
          qwen = text::qwen36_template_of(file->at("tokenizer.chat_template").as_string());
          if (!qwen) throw std::runtime_error(tokens_path + ": the chat template is neither ported Qwen3.6 template");
        }
      } else {
        gemma_tok.emplace(text::Tokenizer::from_gguf(*file));
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
        if (qwen) {
          text::QwenChatOptions opt;
          opt.add_generation_prompt = flag(c, "add_generation_prompt");
          opt.enable_thinking = flag(c, "enable_thinking", true);
          opt.preserve_thinking = flag(c, "preserve_thinking");
          opt.add_vision_id = flag(c, "add_vision_id");
          parts = text::render_qwen36(*qwen, messages ? *messages : json::Value::array(), tools ? *tools : json::Value(), opt);
        } else {
          text::ChatOptions opt;
          opt.add_generation_prompt = flag(c, "add_generation_prompt");
          opt.enable_thinking = flag(c, "enable_thinking");
          opt.preserve_thinking = flag(c, "preserve_thinking");
          parts = text::render_gemma4(messages ? *messages : json::Value::array(), tools ? *tools : json::Value(), opt);
        }
        w.begin_object().kv("text", text::join(parts)).key("parts").begin_array();
        for (const auto& p : parts) w.begin_array().value(p.text).value(p.input).end_array();
        w.end_array();
        if (gemma_tok || qwen_tok) {
          w.key("ids").begin_array();
          if (qwen_tok) {
            std::vector<text::QwenTokenizer::Piece> pieces;
            for (const auto& p : parts) pieces.push_back({p.text, !p.input});
            for (int32_t id : qwen_tok->encode(pieces)) w.value(int64_t(id));
          } else {
            std::vector<text::Tokenizer::Piece> pieces;
            for (const auto& p : parts) pieces.push_back({p.text, !p.input});
            for (int32_t id : gemma_tok->encode(pieces)) w.value(int64_t(id));
          }
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
