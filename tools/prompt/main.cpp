// eklow-prompt: the prompt eklow-server would build for a chat request, without the card: the token ids (the oracle's
// tokens.json form, so that tools/ref/compare tooling and eklow-engine-run read it) and where the generation prompt
// starts (the position eklow-server takes a snapshot at).
//
//   eklow-prompt --model GGUF --request JSON [--chat-template stock|ornith] --out DIR
//
// Writes DIR/tokens.json ({"ids": [...], "snapshot_at": N, "in_thinking": bool}).
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "serve/openai.hpp"
#include "text/chat_qwen36.hpp"
#include "text/tokenizer_qwen.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string model, request, out, tmpl_name;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--request") request = next();
    else if (a == "--out") out = next();
    else if (a == "--chat-template") tmpl_name = next();
    else {
      std::fprintf(stderr, "usage: eklow-prompt --model GGUF --request JSON [--chat-template stock|ornith] --out DIR\n");
      return 2;
    }
  }
  try {
    if (model.empty() || request.empty() || out.empty()) throw std::runtime_error("need --model, --request and --out");
    const auto file = gguf::File::open(model);
    const text::QwenTokenizer tok = text::QwenTokenizer::from_gguf(file);
    text::QwenTemplate tmpl = text::QwenTemplate::Stock;
    if (!tmpl_name.empty()) {
      tmpl = tmpl_name == "ornith" ? text::QwenTemplate::Ornith : text::QwenTemplate::Stock;
    } else {
      const gguf::Value* t = file.find("tokenizer.chat_template");
      const auto known = t ? text::qwen36_template_of(t->as_string()) : std::nullopt;
      if (!known) throw std::runtime_error("the model's chat template is not one eklow knows: say --chat-template stock|ornith");
      tmpl = *known;
    }
    std::ifstream in(request);
    const json::Value body = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
    serve::ChatDefaults d{serve::SamplingParams{}, int64_t(1) << 30, tmpl};
    const serve::ChatRequest r = serve::parse_chat(body, tok, d);
    json::Writer w(false);
    w.begin_object().key("ids").begin_array();
    for (int32_t id : r.prompt) w.value(int64_t(id));
    w.end_array().kv("snapshot_at", r.snapshot_at).kv("in_thinking", r.in_thinking).end_object();
    std::ofstream(out + "/tokens.json") << w.str() << "\n";
    std::printf("%zu tokens, generation prompt at %lld, template %s\n", r.prompt.size(), (long long)r.snapshot_at, text::qwen36_template_name(tmpl));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-prompt: %s\n", e.what());
    return 1;
  }
}
