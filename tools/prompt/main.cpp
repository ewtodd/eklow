// eklow-prompt: the prompt eklow-server would build for a chat request, without the card: the token ids (the oracle's
// tokens.json form, so that tools/ref/compare tooling and eklow-engine-run read it) and where the generation prompt
// starts (the position eklow-server takes a snapshot at).
//
//   eklow-prompt --model GGUF --request JSON [--chat-template stock|ornith|sage] [--image-grids NXxNY,...] --out DIR
//
// Writes DIR/tokens.json ({"ids": [...], "snapshot_points": [...], "in_thinking": bool, "images": [...]}). Image parts need
// --image-grids, the grid of rows of each image in order (no encoder runs: the rows are empty); their rows are id -1 in "ids" and
// each image is listed as eklow-oracle --mmproj lists it ("pos", "n", "nx", "ny").
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <memory>
#include <span>
#include <vector>
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
  std::string model, request, out, tmpl_name, grids;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--request") request = next();
    else if (a == "--out") out = next();
    else if (a == "--chat-template") tmpl_name = next();
    else if (a == "--image-grids") grids = next();
    else {
      std::fprintf(stderr, "usage: eklow-prompt --model GGUF --request JSON [--chat-template stock|ornith|sage] [--image-grids NXxNY,...] --out DIR\n");
      return 2;
    }
  }
  try {
    if (model.empty() || request.empty() || out.empty()) throw std::runtime_error("need --model, --request and --out");
    const auto file = gguf::File::open(model);
    const text::QwenTokenizer tok = text::QwenTokenizer::from_gguf(file);
    text::QwenTemplate tmpl = text::QwenTemplate::Stock;
    if (!tmpl_name.empty()) {
      tmpl = tmpl_name == "ornith" ? text::QwenTemplate::Ornith : tmpl_name == "sage" ? text::QwenTemplate::Sage : text::QwenTemplate::Stock;
    } else {
      const gguf::Value* t = file.find("tokenizer.chat_template");
      const auto known = t ? text::qwen36_template_of(t->as_string()) : std::nullopt;
      if (!known) throw std::runtime_error("the model's chat template is not one eklow knows: say --chat-template stock|ornith|sage");
      tmpl = *known;
    }
    std::ifstream in(request);
    const json::Value body = json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
    serve::ChatDefaults d{serve::SamplingParams{}, int64_t(1) << 30, tmpl};
    std::vector<std::pair<int64_t, int64_t>> grid;
    for (size_t at = 0; at < grids.size();) {
      const size_t end = std::min(grids.find(',', at), grids.size());
      int64_t nx = 0, ny = 0;
      if (std::sscanf(grids.substr(at, end - at).c_str(), "%lldx%lld", (long long*)&nx, (long long*)&ny) != 2) throw std::runtime_error("--image-grids: NXxNY,NXxNY...");
      grid.push_back({nx, ny});
      at = end + 1;
    }
    size_t next_image = 0;
    const serve::ImageEncoder stub = [&](std::span<const uint8_t> bytes) {
      if (next_image >= grid.size()) throw std::runtime_error("more images in the request than --image-grids");
      uint64_t h = 0xcbf29ce484222325ull;
      for (uint8_t b : bytes) h = (h ^ b) * 0x100000001b3ull;
      const auto [nx, ny] = grid[next_image++];
      return serve::EncodedImage{h, nx * ny, std::make_shared<const std::vector<float>>(), nx, ny};
    };
    if (!grid.empty()) d.images = &stub;
    const serve::ChatRequest r = serve::parse_chat(body, tok, d);
    json::Writer w(false);
    w.begin_object().key("ids").begin_array();
    for (int32_t id : r.prompt) w.value(int64_t(id < 0 ? -1 : id));
    w.end_array().key("snapshot_points").begin_array();
    for (int64_t p : r.snapshot_points) w.value(p);
    w.end_array().kv("in_thinking", r.in_thinking).key("images").begin_array();
    for (const serve::ImageInput& im : r.images) w.begin_object().kv("pos", im.start).kv("n", im.n).kv("nx", im.nx).kv("ny", im.ny).end_object();
    w.end_array().end_object();
    std::ofstream(out + "/tokens.json") << w.str() << "\n";
    std::printf("%zu tokens, snapshot points %zu (last at %lld), template %s\n", r.prompt.size(), r.snapshot_points.size(), (long long)(r.snapshot_points.empty() ? -1 : r.snapshot_points.back()), text::qwen36_template_name(tmpl));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-prompt: %s\n", e.what());
    return 1;
  }
}
