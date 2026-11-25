#include "image/inkling_image.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eklow::image {

namespace {
int align_up(int x) { return (x + kInklingAlign - 1) / kInklingAlign * kInklingAlign; }
}  // namespace

Size inkling_size(int width, int height, int64_t max_tokens) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("the image is empty");
  const int64_t max_pixels = max_tokens * int64_t(kInklingAlign) * kInklingAlign;
  if (int64_t(align_up(width)) * align_up(height) <= max_pixels) return {align_up(width), align_up(height)};
  // mtmd's calc_size_preserved_ratio with align_size 40 and max_pixels: each side rounded to the alignment, then, past
  // the budget, scaled by sqrt(pixels / budget) and floored to it (in float, as there)
  const int f = kInklingAlign;
  auto round_by = [&](float x) { return int(std::round(x / float(f))) * f; };
  auto floor_by = [&](float x) { return int(std::floor(x / float(f))) * f; };
  int w = std::max(f, round_by(float(width))), h = std::max(f, round_by(float(height)));
  if (int64_t(w) * h > max_pixels) {
    const float beta = std::sqrt(float(height) * float(width) / float(max_pixels));
    h = std::max(f, floor_by(float(height) / beta));
    w = std::max(f, floor_by(float(width) / beta));
  }
  return {w, h};
}

Patches prepare_inkling(const Rgb& picture, int64_t max_tokens) {
  const Size s = inkling_size(picture.width, picture.height, max_tokens);
  const bool resized = int64_t(align_up(picture.width)) * align_up(picture.height) > max_tokens * int64_t(kInklingAlign) * kInklingAlign;
  const Rgb scaled = resized ? resize_bicubic(picture, s.width, s.height) : Rgb{};
  const Rgb& img = resized ? scaled : picture;
  Patches p;
  p.cols = s.width / kInklingPatch;
  p.rows = s.height / kInklingPatch;
  p.dim = 3 * kInklingPatch * kInklingPatch;
  p.data.assign(size_t(p.rows) * p.cols * p.dim, 0.0f);
  const float pad = (-1.0f - 127.5f) / 127.5f;  // the padding is -1 before the normalization
  for (int py = 0; py < p.rows; ++py)
    for (int px = 0; px < p.cols; ++px) {
      float* dst = p.data.data() + (size_t(py) * p.cols + px) * size_t(p.dim);
      for (int c = 0; c < 3; ++c)
        for (int y = 0; y < kInklingPatch; ++y)
          for (int x = 0; x < kInklingPatch; ++x) {
            const int iy = py * kInklingPatch + y, ix = px * kInklingPatch + x;
            const bool in = iy < img.height && ix < img.width;
            dst[(c * kInklingPatch + y) * kInklingPatch + x] =
                in ? (float(img.pixels[(size_t(iy) * img.width + ix) * 3 + c]) - 127.5f) / 127.5f : pad;
          }
    }
  return p;
}

}  // namespace eklow::image
