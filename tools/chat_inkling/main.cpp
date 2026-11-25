// eklow-chat-inkling: Inkling 2 Mini's chat template port (and llama-server's request handling around it) from the command line.
//
//   eklow-chat-inkling render [--tokens GGUF] [--allow-image] < REQUESTS > RESULTS
//                                         REQUESTS: one OpenAI chat completion body per line (messages, tools, tool_choice,
//                                         reasoning_effort, chat_template_kwargs, ...), as llama-server receives it;
//                                         RESULTS: one line per request, {"prompt": ..., "parts": [[text, is_input], ...],
//                                         "generation_prompt": ..., "mode": {...the output parser's view of the request}}
//                                         or {"error": ...}; with --tokens, also "ids", the prompt tokenized with special
//                                         tokens recognized only in template text, and "roundtrip", whether they decode to it.
//                                         --allow-image: image_url parts as a server with Inkling's mmproj takes them.
//   eklow-chat-inkling check FILE         whether FILE (a GGUF's tokenizer.chat_template, or a chat_template.jinja read whole)
//                                         is the template that was ported, byte for byte
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
#include "text/chat_inkling.hpp"
#include "text/tokenizer_inkling.hpp"

using namespace eklow;

namespace {

std::string template_text(const std::string& path) {
  if (path.size() >= 5 && path.compare(path.size() - 5, 5, ".gguf") == 0)
    return gguf::File::open(path).at("tokenizer.chat_template").as_string();
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error(path + ": cannot open");
  std::ostringstream s;
  s << in.rdbuf();
  return s.str();
}

const char* tool_choice_name(text::InklingToolChoice c) {
  return c == text::InklingToolChoice::None ? "none" : c == text::InklingToolChoice::Required ? "required" : "auto";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "";
  if (mode == "check" && argc > 2) {
    try {
      const bool same = text::is_inkling_template(template_text(argv[2]));
      std::printf("%s: chat template %s\n", argv[2], same ? "is the ported Inkling 2 template" : "DIFFERS from the ported Inkling 2 template");
      return same ? 0 : 1;
    } catch (const std::exception& e) {
      std::fprintf(stderr, "eklow-chat-inkling: %s\n", e.what());
      return 1;
    }
  }
  std::string tokens_path;
  bool allow_image = false;
  bool usage = mode != "render";
  for (int i = 2; i < argc && !usage; ++i) {
    if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_path = argv[++i];
    else if (!std::strcmp(argv[i], "--allow-image")) allow_image = true;
    else usage = true;
  }
  if (usage) {
    std::fprintf(stderr, "usage: eklow-chat-inkling render [--tokens GGUF] [--allow-image] < REQUESTS > RESULTS | eklow-chat-inkling check FILE\n");
    return 2;
  }
  try {
    std::optional<gguf::File> file;
    std::optional<text::InklingTokenizer> tok;
    if (!tokens_path.empty()) {
      file.emplace(gguf::File::open(tokens_path));
      tok.emplace(text::InklingTokenizer::from_gguf(*file));
      if (!text::is_inkling_template(file->at("tokenizer.chat_template").as_string()))
        throw std::runtime_error(tokens_path + ": the chat template is not the ported Inkling 2 template");
    }
    std::ios::sync_with_stdio(false);
    std::string line;
    while (std::getline(std::cin, line)) {
      json::Writer w(false);
      try {
        const text::InklingChat chat = text::render_inkling_request(text::parse_llama_json(line), allow_image);
        w.begin_object().kv("prompt", text::join(chat.parts)).key("parts").begin_array();
        for (const auto& p : chat.parts) w.begin_array().value(p.text).value(p.input).end_array();
        w.end_array().kv("generation_prompt", chat.generation_prompt);
        w.key("mode").begin_object().key("tools").begin_array();
        for (const auto& n : chat.mode.tool_names) w.value(n);
        w.end_array()
            .kv("tool_choice", tool_choice_name(chat.mode.tool_choice))
            .kv("parallel_tool_calls", chat.mode.parallel_tool_calls)
            .kv("response_format", chat.mode.response_format)
            .kv("extract_reasoning", chat.mode.extract_reasoning)
            .end_object();
        if (tok) {
          std::vector<text::InklingTokenizer::Piece> pieces;
          for (const auto& p : chat.parts) pieces.push_back({p.text, !p.input});
          w.key("ids").begin_array();
          const std::vector<int32_t> ids = tok->encode(pieces);
          for (int32_t id : ids) w.value(int64_t(id));
          w.end_array().kv("roundtrip", tok->decode(ids) == text::join(chat.parts));
          // the ids of the continued block, as the prompt tokenizes it (the output parser's prefill)
          std::vector<text::InklingTokenizer::Piece> gp;
          for (const auto& p : chat.generation_parts) gp.push_back({p.text, !p.input});
          w.key("generation_ids").begin_array();
          for (int32_t id : tok->encode(gp)) w.value(int64_t(id));
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
    std::fprintf(stderr, "eklow-chat-inkling: %s\n", e.what());
    return 1;
  }
  return 0;
}
