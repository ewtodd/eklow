// eklow-vision: Gemma 4's vision embedder (model/vision_gemma4.hpp) on one image, for checking
// against the oracle's image_0.f32 (the embeddings llama.cpp feeds the text model).
//
//   eklow-vision --mmproj GGUF --image FILE [--out ROWS.f32]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "image/gemma4_image.hpp"
#include "model/gguf.hpp"
#include "model/vision_gemma4.hpp"
#include "rt/device.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string mmproj, path, out;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--mmproj") mmproj = next();
    else if (a == "--image") path = next();
    else if (a == "--out") out = next();
    else {
      std::fprintf(stderr, "usage: eklow-vision --mmproj GGUF --image FILE [--out ROWS.f32]\n");
      return 2;
    }
  }
  try {
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    auto f = gguf::File::open(mmproj);
    model::VisionGemma4 v(q, f);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read " + path);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), {}};
    for (int pass = 0; pass < 2; ++pass) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto patches = image::prepare(image::decode(bytes), v.spec());
      const auto t1 = std::chrono::steady_clock::now();
      const auto rows = v.encode(patches);
      const auto t2 = std::chrono::steady_clock::now();
      std::printf("%d x %d patches: prepared in %.1f ms, embedded in %.1f ms\n", patches.cols, patches.rows,
                  std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
      if (pass == 1 && !out.empty()) {
        std::vector<float> host(size_t(rows.n * rows.dim));
        q.memcpy(host.data(), rows.data.get(), host.size() * sizeof(float)).wait();
        std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(host.data()), std::streamsize(host.size() * sizeof(float)));
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-vision: %s\n", e.what());
    return 1;
  }
}
