// eklow-prompt: the prompt eklow-server would build for a chat request, without the card: the token ids with image
// rows as -1 (the oracle's tokens.json form, so that tools/ref/compare tooling and eklow-engine-run read it), and
// where each image sits and how many rows it gets.
//
//   eklow-prompt --model GGUF --request JSON [--image-tokens N] --out DIR
//
// Writes DIR/tokens.json ({"ids": [...], "images": [{"pos", "n"}, ...]}). The rows per image are
// what the encoder would give (the patch grid pooled 3 x 3), counted from image::prepare as eklow-server does.
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "image/gemma4_image.hpp"
#include "model/gguf.hpp"
#include "serve/openai.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string model, request, out;
  int tokens = 280;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--request") request = next();
    else if (a == "--out") out = next();
    else if (a == "--image-tokens") tokens = std::atoi(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-prompt --model GGUF --request JSON [--image-tokens N] --out DIR\n");
      return 2;
    }
  }
  try {
    if (model.empty() || request.empty() || out.empty()) throw std::runtime_error("need --model, --request and --out");
    const auto file = gguf::File::open(model);
    const text::Tokenizer tok = text::Tokenizer::from_gguf(file);
    std::ifstream in(request);
    const json::Value body = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
    image::Spec spec;
    spec.patch = 48, spec.cut = 16, spec.min_patches = std::min(70, tokens), spec.max_patches = tokens;
    serve::ChatDefaults d{serve::SamplingParams{}, int64_t(1) << 30};
    d.encode_image = [&](std::span<const uint8_t> bytes) {
      const image::Patches p = image::prepare(image::decode(bytes), spec);
      serve::EncodedImage e;
      e.n = int64_t(p.cols / 3) * (p.rows / 3);
      e.rows = std::make_shared<std::vector<float>>();
      uint64_t h = 0xcbf29ce484222325ull ^ uint64_t(tokens);
      for (uint8_t b : bytes) h = (h ^ b) * 0x100000001b3ull;
      e.hash = h;
      return e;
    };
    const serve::ChatRequest r = serve::parse_chat(body, tok, d);
    json::Writer w(false);
    w.begin_object().key("ids").begin_array();
    for (int32_t id : r.prompt) w.value(int64_t(id < 0 ? -1 : id));
    w.end_array().key("images").begin_array();
    for (const auto& im : r.images) w.begin_object().kv("pos", im.start).kv("n", im.n).end_object();
    w.end_array().end_object();
    std::ofstream(out + "/tokens.json") << w.str() << "\n";
    std::printf("%zu tokens, %zu images (%lld rows)\n", r.prompt.size(), r.images.size(), [&] {
      long long n = 0;
      for (const auto& im : r.images) n += im.n;
      return n;
    }());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-prompt: %s\n", e.what());
    return 1;
  }
}
