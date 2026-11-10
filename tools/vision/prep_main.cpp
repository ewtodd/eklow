// eklow-image-prep: the picture each encoder's preparation hands on, before it is cut into patches, as raw RGB bytes
// for tools/vision/prep_compare.py to hold against Hugging Face's processors. Host code only: no card needed.
//
//   eklow-image-prep decode IMAGE OUT.rgb               (stb_image's decode alone)
//   eklow-image-prep gemma TOKENS IMAGE OUT.rgb         (EmbeddingGemma 2's tower, the budget in 48-pixel tokens)
//   eklow-image-prep qwen MIN MAX IMAGE OUT.rgb         (Qwen3-VL's, the bounds in 32-pixel tokens)
//
// Prints "WIDTH HEIGHT" of what it wrote.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "image/gemma4_image.hpp"
#include "image/qwen3vl_image.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  try {
    const std::string kind = argc > 1 ? argv[1] : "";
    if (!((kind == "decode" && argc == 4) || (kind == "gemma" && argc == 5) || (kind == "qwen" && argc == 6))) {
      std::fprintf(stderr, "usage: eklow-image-prep decode IMAGE OUT.rgb | gemma TOKENS IMAGE OUT.rgb | qwen MIN MAX IMAGE OUT.rgb\n");
      return 2;
    }
    const std::string path = argv[argc - 2], out = argv[argc - 1];
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read " + path);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), {}};
    const image::Rgb rgb = image::decode(bytes);
    image::Rgb p = rgb;
    if (kind == "gemma") {
      image::Spec s;
      s.max_tokens = std::atoi(argv[2]);
      p = image::fitted(rgb, s);
    } else if (kind == "qwen") {
      image::QwenSpec s;
      s.min_tokens = std::atoi(argv[2]);
      s.max_tokens = std::atoi(argv[3]);
      p = image::prepare_qwen3vl(rgb, s);
    }
    std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(p.pixels.data()), std::streamsize(p.pixels.size()));
    std::printf("%d %d\n", p.width, p.height);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-image-prep: %s\n", e.what());
    return 1;
  }
}
