// eklow-vision: Gemma 4's vision encoder (model/vision_gemma4v.hpp) on one image, for checking
// against the oracle's image_0.f32 (the rows llama.cpp feeds the text model).
//
//   eklow-vision --mmproj GGUF --image FILE [--tokens N] [--out ROWS.f32] [--taps DIR [--tap-prefix P,...]]
//
// --tokens is the image token budget (llama.cpp's image_max_tokens; default 280, 70..1120).
// --taps writes the encoder's intermediate tensors that match a prefix (default: inp, pos_embd, layer_out-,
// pooled, std_scaled, projected) as DIR/clip.NAME.f32, as the oracle's --clip-tensors does.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "image/gemma4_image.hpp"
#include "model/gguf.hpp"
#include "model/vision_gemma4v.hpp"
#include "rt/device.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string mmproj, path, out, taps;
  std::vector<std::string> prefixes{"inp", "pos_embd", "layer_out-", "pooled", "std_scaled", "projected"};
  int tokens = 280;
  bool dump_patches = false, bf16_act = false, ref = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--mmproj") mmproj = next();
    else if (a == "--image") path = next();
    else if (a == "--out") out = next();
    else if (a == "--tokens") tokens = std::atoi(next().c_str());
    else if (a == "--taps") taps = next();
    else if (a == "--tap-prefix") {
      prefixes.clear();
      std::string s = next();
      for (size_t b = 0, e; b <= s.size(); b = e + 1) {
        e = s.find(',', b);
        if (e == std::string::npos) e = s.size();
        prefixes.push_back(s.substr(b, e - b));
      }
    } else if (a == "--dump-patches") dump_patches = true;
    else if (a == "--bf16-act") bf16_act = true;
    else if (a == "--ref") ref = true;
    else {
      std::fprintf(stderr, "usage: eklow-vision --mmproj GGUF --image FILE [--tokens N] [--out ROWS.f32] [--ref] [--bf16-act] [--taps DIR [--tap-prefix P,...]]\n");
      return 2;
    }
  }
  try {
    if (tokens < 1 || tokens > 1120) throw std::invalid_argument("--tokens must be 1..1120");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::invalid_argument("cannot read " + path);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), {}};
    // Prepare on the host first: a wrong file fails before the card is touched.
    const auto rgb = image::decode(bytes);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    auto f = gguf::File::open(mmproj);
    model::VisionGemma4V v(q, f, ref);
    v.set_bf16_activations(bf16_act);
    std::printf("%s path, card bytes at rest: %.1f MiB; reserve(%d) would hold %.1f MiB\n", v.reference() ? "reference" : "fast", double(v.bytes()) / 1048576.0, tokens, double(v.card_bytes(tokens)) / 1048576.0);
    const image::Spec spec = v.spec(tokens);
    if (!taps.empty())
      v.set_tap([&](std::string_view name, const float* dev, int64_t rows, int64_t dim) {
        bool want = false;
        for (const auto& p : prefixes) want |= name.substr(0, p.size()) == p;
        if (!want) return;
        std::vector<float> host(size_t(rows * dim));
        q.memcpy(host.data(), dev, host.size() * sizeof(float)).wait();
        std::ofstream(taps + "/clip." + std::string(name) + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(host.data()), std::streamsize(host.size() * sizeof(float)));
      });
    for (int pass = 0; pass < (taps.empty() ? 2 : 1); ++pass) {
      const auto t0 = std::chrono::steady_clock::now();
      const auto patches = image::prepare(rgb, spec);
      const auto t1 = std::chrono::steady_clock::now();
      if (dump_patches && !taps.empty())
        std::ofstream(taps + "/patches.f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(patches.data.data()), std::streamsize(patches.data.size() * sizeof(float)));
      const auto rows = v.encode(patches);
      if (pass == 0) std::printf("held after the first encode: %.1f MiB\n", double(v.bytes()) / 1048576.0);
      const auto t2 = std::chrono::steady_clock::now();
      std::printf("%d x %d image, %d x %d patches, %lld tokens x %lld: prepared in %.1f ms, encoded in %.1f ms\n", rgb.width, rgb.height,
                  patches.cols, patches.rows, (long long)rows.n, (long long)rows.dim,
                  std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
      if (pass == (taps.empty() ? 1 : 0) && !out.empty()) {
        std::vector<float> host(size_t(rows.n * rows.dim));
        q.memcpy(host.data(), rows.data.get(), host.size() * sizeof(float)).wait();
        std::ofstream(out, std::ios::binary).write(reinterpret_cast<const char*>(host.data()), std::streamsize(host.size() * sizeof(float)));
      }
    }
    v.release();
    std::printf("held after release(): %.1f MiB\n", double(v.bytes()) / 1048576.0);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-vision: %s\n", e.what());
    return 1;
  }
}
