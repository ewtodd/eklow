// eklow-image: an image prepared for Gemma 4's vision embedder (image/gemma4_image.hpp), for
// checking against the oracle's first vision tensor (the im2col'd patches, clip.node_0.f32).
//
//   eklow-image --image FILE [--out PATCHES.f32]
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "image/gemma4_image.hpp"

int main(int argc, char** argv) {
  std::string path, out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--image") path = next();
    else if (a == "--out") out = next();
    else {
      std::fprintf(stderr, "usage: eklow-image --image FILE [--out PATCHES.f32]\n");
      return 2;
    }
  }
  try {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::invalid_argument("cannot read " + path);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(f), {}};
    const auto rgb = eklow::image::decode(bytes);
    const eklow::image::Spec spec;
    const auto size = eklow::image::fit(rgb.width, rgb.height, spec.patch, spec.min_patches, spec.max_patches);
    const auto p = eklow::image::prepare(rgb, spec);
    std::printf("%s: %d x %d, fitted to %d x %d, %d x %d patches of %lld\n", path.c_str(), rgb.width, rgb.height, size.width,
                size.height, p.cols, p.rows, (long long)p.dim);
    if (!out.empty())
      std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(p.data.data()), std::streamsize(p.data.size() * sizeof(float)));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-image: %s\n", e.what());
    return 1;
  }
}
