// eklow-inkling-vision-test: Inkling 2 Mini's vision tower (model/vision_inkling2v) against llama.cpp's mtmd at c74c71a6
// (the M5 ground truth, DECISIONS 2026-11-25 "Inkling M5"): for each case directory the oracle wrote (eklow-oracle --mmproj
// --image ... --clip-tensors embeddings,layer_out-,pixel_shuffle,projected), the same image through
// image::prepare_inkling and the encoder, each stage compared by relative L2 per row (median and max) and the image's
// embeddings (image_0.f32) overall.
//
//   eklow-inkling-vision-test --mmproj GGUF --case DIR IMAGE [--case DIR IMAGE ...] [--max-tokens N] [--reps N]
//
// --max-tokens is the image budget the oracle ran with (4096 unless --image-tokens). --reps times the encoder.
// Exit status 1 when the projected rows differ by more than 2e-2 (median) from the oracle's.
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "image/inkling_image.hpp"
#include "model/gguf.hpp"
#include "model/vision_inkling2v.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

std::vector<float> read_f32(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  std::vector<char> b((std::istreambuf_iterator<char>(in)), {});
  std::vector<float> v(b.size() / 4);
  std::memcpy(v.data(), b.data(), v.size() * 4);
  return v;
}

// relative L2 of each row: median and max
std::pair<double, double> compare(const std::vector<float>& got, const std::vector<float>& want, int64_t cols) {
  std::vector<double> rel;
  for (size_t r = 0; r * size_t(cols) < std::min(got.size(), want.size()); ++r) {
    double num = 0, den = 0;
    for (int64_t c = 0; c < cols; ++c) {
      const double a = got[r * cols + c], b = want[r * cols + c];
      num += (a - b) * (a - b), den += b * b;
    }
    rel.push_back(den > 0 ? std::sqrt(num / den) : std::sqrt(num));
  }
  if (rel.empty()) return {NAN, NAN};
  std::vector<double> s = rel;
  std::sort(s.begin(), s.end());
  return {s[s.size() / 2], s.back()};
}

}  // namespace

int main(int argc, char** argv) {
  std::string mmproj;
  std::vector<std::pair<std::string, std::string>> cases;
  int64_t max_tokens = 4096;
  int reps = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--mmproj" && i + 1 < argc) mmproj = argv[++i];
    else if (a == "--case" && i + 2 < argc) cases.push_back({argv[i + 1], argv[i + 2]}), i += 2;
    else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoll(argv[++i]);
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: eklow-inkling-vision-test --mmproj GGUF --case DIR IMAGE [...] [--max-tokens N] [--reps N]\n");
      return 2;
    }
  }
  if (mmproj.empty() || cases.empty()) return 2;
  bool ok = true;
  try {
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    gguf::File f = gguf::File::open(mmproj);
    const auto t0 = std::chrono::steady_clock::now();
    model::VisionInkling2V v(q, f);
    std::printf("loaded %.1f MB in %.2f s\n", double(v.bytes()) / 1e6, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (const auto& [dir, path] : cases) {
      std::ifstream in(path, std::ios::binary);
      const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), {});
      const image::Rgb rgb = image::decode(bytes);
      const image::Patches p = image::prepare_inkling(rgb, max_tokens);
      std::printf("%s: %dx%d -> %d x %d patches, %lld tokens\n", path.c_str(), rgb.width, rgb.height, p.cols, p.rows,
                  (long long)(p.cols * p.rows / 4));
      std::vector<std::string> lines;
      auto dump = [&](const std::string& name, const float* rows, int64_t n, int64_t cols) {
        const std::vector<float> want = read_f32(dir + "/clip." + name + ".f32");
        if (want.empty()) return;
        std::vector<float> got(size_t(n * cols));
        q.memcpy(got.data(), rows, got.size() * 4).wait();
        if (want.size() != got.size()) {
          lines.push_back(name + ": " + std::to_string(got.size()) + " values against the oracle's " + std::to_string(want.size()));
          return;
        }
        const auto [med, mx] = compare(got, want, cols);
        char b[200];
        std::snprintf(b, sizeof b, "%-16s rel L2 median %.2e max %.2e", name.c_str(), med, mx);
        lines.push_back(b);
      };
      const std::vector<float> rows = v.encode(p, dump);
      for (const auto& l : lines) std::printf("  %s\n", l.c_str());
      const std::vector<float> want = read_f32(dir + "/image_0.f32");
      const auto [med, mx] = compare(rows, want, v.n_out());
      std::printf("  %-16s rel L2 median %.2e max %.2e%s\n", "image_0", med, mx, want.size() == rows.size() ? "" : " (sizes differ)");
      if (!(med < 2e-2) || want.size() != rows.size()) ok = false;
      if (reps > 0) {
        q.wait();
        const auto s0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) (void)v.encode(p);
        std::printf("  encode %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s0).count() / reps);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-inkling-vision-test: %s\n", e.what());
    return 1;
  }
  std::printf("%s\n", ok ? "all ok" : "FAILED");
  return ok ? 0 : 1;
}
