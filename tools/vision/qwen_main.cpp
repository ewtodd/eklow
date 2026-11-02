// eklow-vision-qwen: Qwen3.5/3.6's vision encoder (model/vision_qwen3vl.hpp) on images, for checking against the
// oracle's image_K.f32 (the rows llama.cpp feeds the text model) and for timing.
//
//   eklow-vision-qwen --mmproj GGUF --image FILE [--image FILE ...] [--max-tokens N] [--min-tokens N] [--out DIR]
//                     [--ref] [--resident] [--runs N] [--taps DIR [--tap-prefix P,...]] [--prepared-only]
//
// --out DIR writes image_K.f32 ([tokens][2048] fp32) and images.json (tokens and merged grid per image) as the oracle's
// --encode-only does. --max-tokens / --min-tokens are llama.cpp's --image-max-tokens / --image-min-tokens (default 4096 / 8
// there; 1024 / 8 here). --taps writes the graph tensors that match a prefix as DIR/clip.NAME.f32 (one image only).
// --runs N encodes each image N times and prints the times of the last (the first one includes the allocations).
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "image/qwen3vl_image.hpp"
#include "model/gguf.hpp"
#include "model/vision_qwen3vl.hpp"
#include "rt/device.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string mmproj, out, taps;
  std::vector<std::string> images, prefixes{"patch_bias", "inp_pos_emb", "layer_out-", "embeddings"};
  int max_tokens = 1024, min_tokens = 8, runs = 1;
  bool ref = false, resident = false, prepared_only = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--mmproj") mmproj = next();
    else if (a == "--image") images.push_back(next());
    else if (a == "--out") out = next();
    else if (a == "--max-tokens" || a == "--tokens") max_tokens = std::atoi(next().c_str());
    else if (a == "--min-tokens") min_tokens = std::atoi(next().c_str());
    else if (a == "--runs") runs = std::atoi(next().c_str());
    else if (a == "--taps") taps = next();
    else if (a == "--tap-prefix") {
      prefixes.clear();
      std::string s = next();
      for (size_t b = 0, e; b <= s.size(); b = e + 1) {
        e = s.find(',', b);
        if (e == std::string::npos) e = s.size();
        prefixes.push_back(s.substr(b, e - b));
      }
    } else if (a == "--ref") ref = true;
    else if (a == "--resident") resident = true;
    else if (a == "--prepared-only") prepared_only = true;
    else {
      std::fprintf(stderr, "usage: eklow-vision-qwen --mmproj GGUF --image FILE... [--max-tokens N] [--min-tokens N] [--out DIR] [--ref] [--resident] [--runs N] [--taps DIR]\n");
      return 2;
    }
  }
  try {
    if (max_tokens < 1 || min_tokens < 1) throw std::invalid_argument("token limits must be positive");
    std::vector<std::vector<uint8_t>> bytes;
    for (const auto& p : images) {
      std::ifstream in(p, std::ios::binary);
      if (!in) throw std::invalid_argument("cannot read " + p);
      bytes.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
      image::decode(bytes.back());  // a wrong file fails before the card is touched
    }
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    auto f = gguf::File::open(mmproj);
    model::VisionQwen3VL v(q, f, model::QwenVisionOptions{ref, resident});
    std::printf("%s path%s, card bytes at rest: %.1f MiB; reserve(%d) would hold %.1f MiB\n", v.reference() ? "reference" : "fast",
                v.resident() ? " (resident weights)" : "", double(v.bytes()) / 1048576.0, max_tokens, double(v.card_bytes(max_tokens)) / 1048576.0);
    const image::QwenSpec spec = v.spec(max_tokens, min_tokens);
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
    std::ostringstream json;
    for (size_t k = 0; k < images.size(); ++k) {
      const auto rgb = image::decode(bytes[k]);
      const auto t0 = std::chrono::steady_clock::now();
      const auto pic = image::prepare_qwen3vl(rgb, spec);
      const auto t1 = std::chrono::steady_clock::now();
      std::printf("%s: %d x %d -> %d x %d, prepared in %.1f ms\n", images[k].c_str(), rgb.width, rgb.height, pic.width, pic.height,
                  std::chrono::duration<double, std::milli>(t1 - t0).count());
      if (prepared_only) continue;
      model::QwenImageRows r;
      for (int run = 0; run < (taps.empty() ? runs : 1); ++run) {
        const auto t2 = std::chrono::steady_clock::now();
        r = v.encode(pic);
        const auto t3 = std::chrono::steady_clock::now();
        if (run == 0) std::printf("  held after the first encode: %.1f MiB\n", double(v.bytes()) / 1048576.0);
        std::printf("  %lld tokens (%d x %d) x %lld, encoded in %.1f ms\n", (long long)r.rows.n, r.grid_w, r.grid_h, (long long)r.rows.dim,
                    std::chrono::duration<double, std::milli>(t3 - t2).count());
      }
      json << (k ? ", " : "") << "{\"n\": " << r.rows.n << ", \"nx\": " << r.grid_w << ", \"ny\": " << r.grid_h << ", \"n_embd\": " << r.rows.dim << "}";
      if (!out.empty()) {
        std::vector<float> host(size_t(r.rows.n * r.rows.dim));
        q.memcpy(host.data(), r.rows.data.get(), host.size() * sizeof(float)).wait();
        std::ofstream(out + "/image_" + std::to_string(k) + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(host.data()), std::streamsize(host.size() * sizeof(float)));
      }
    }
    if (!out.empty()) std::ofstream(out + "/images.json") << "{\"images\": [" << json.str() << "]}\n";
    v.release();
    std::printf("held after release(): %.1f MiB\n", double(v.bytes()) / 1048576.0);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-vision-qwen: %s\n", e.what());
    return 1;
  }
}
