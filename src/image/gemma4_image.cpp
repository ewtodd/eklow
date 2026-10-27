#include "image/gemma4_image.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "stb/stb_image.h"

namespace eklow::image {

Rgb decode(std::span<const uint8_t> bytes, int64_t max_pixels) {
  int w = 0, h = 0, c = 0;
  if (bytes.empty() || !stbi_info_from_memory(bytes.data(), int(bytes.size()), &w, &h, &c))
    throw std::invalid_argument("the image is not in a format this server reads (JPEG, PNG, GIF, BMP, ...)");
  if (int64_t(w) * h > max_pixels) throw std::invalid_argument("the image has too many pixels");
  unsigned char* p = stbi_load_from_memory(bytes.data(), int(bytes.size()), &w, &h, &c, 3);
  if (!p) throw std::invalid_argument(std::string("the image cannot be decoded: ") + stbi_failure_reason());
  Rgb out{w, h, std::vector<uint8_t>(p, p + size_t(w) * size_t(h) * 3)};
  stbi_image_free(p);
  return out;
}

// The arithmetic is mtmd's, float where it is float.
Size fit(int width, int height, int patch, int min_patches, int max_patches) {
  const int area = patch * patch, min_pixels = min_patches * area, max_pixels = max_patches * area;
  auto round_by = [patch](float x) { return int(std::round(x / float(patch))) * patch; };
  auto ceil_by = [patch](float x) { return int(std::ceil(x / float(patch))) * patch; };
  auto floor_by = [patch](float x) { return int(std::floor(x / float(patch))) * patch; };
  int w = std::max(patch, round_by(float(width))), h = std::max(patch, round_by(float(height)));
  if (h * w > max_pixels) {
    const float beta = std::sqrt(float(height) * float(width) / float(max_pixels));
    h = std::max(patch, floor_by(float(height) / beta));
    w = std::max(patch, floor_by(float(width) / beta));
  } else if (h * w < min_pixels) {
    const float beta = std::sqrt(float(min_pixels) / (float(height) * float(width)));
    h = ceil_by(float(height) * beta);
    w = ceil_by(float(width) * beta);
  }
  return {w, h};
}

namespace {

constexpr int kPrecision = 22;  // fixed-point fraction bits: 32 - 8 (pixel) - 2 (headroom)

double bicubic(double x) {
  constexpr double a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

uint8_t clip8(int32_t v) { return uint8_t(std::clamp(v, 0, 255)); }

// Per output pixel along one axis: its first input pixel and count, and its weights in fixed
// point, ksize apart.
struct Taps {
  int ksize;
  std::vector<int> first, count;
  std::vector<int32_t> w;
};

Taps taps(int in, int out) {
  const double scale = double(in) / out, filterscale = std::max(scale, 1.0), support = 2.0 * filterscale;
  Taps t;
  t.ksize = int(std::ceil(support)) * 2 + 1;
  t.first.resize(size_t(out));
  t.count.resize(size_t(out));
  t.w.assign(size_t(out) * t.ksize, 0);
  std::vector<double> pre(size_t(t.ksize));
  for (int xx = 0; xx < out; ++xx) {
    const double center = (xx + 0.5) * scale, ss = 1.0 / filterscale;
    // int() truncates toward zero, as the C casts in Pillow and llama.cpp do.
    const int xmin = std::max(0, int(center - support + 0.5)), xmax = std::min(in, int(center + support + 0.5)) - xmin;
    double ww = 0;
    for (int x = 0; x < xmax; ++x) ww += pre[size_t(x)] = bicubic((x + xmin - center + 0.5) * ss);
    for (int x = 0; x < xmax; ++x) {
      const double v = ww != 0 ? pre[size_t(x)] / ww : pre[size_t(x)];
      t.w[size_t(xx) * t.ksize + x] = int32_t(v * double(1 << kPrecision) + (v < 0 ? -0.5 : 0.5));
    }
    t.first[size_t(xx)] = xmin;
    t.count[size_t(xx)] = xmax;
  }
  return t;
}

Rgb horizontal(const Rgb& src, int out_w) {
  const Taps t = taps(src.width, out_w);
  Rgb dst{out_w, src.height, std::vector<uint8_t>(size_t(out_w) * src.height * 3)};
  for (int y = 0; y < src.height; ++y)
    for (int xx = 0; xx < out_w; ++xx) {
      const int32_t* k = &t.w[size_t(xx) * t.ksize];
      const uint8_t* p = &src.pixels[(size_t(y) * src.width + t.first[size_t(xx)]) * 3];
      int32_t s0 = 1 << (kPrecision - 1), s1 = s0, s2 = s0;
      for (int x = 0; x < t.count[size_t(xx)]; ++x, p += 3) s0 += p[0] * k[x], s1 += p[1] * k[x], s2 += p[2] * k[x];
      uint8_t* d = &dst.pixels[(size_t(y) * out_w + xx) * 3];
      d[0] = clip8(s0 >> kPrecision), d[1] = clip8(s1 >> kPrecision), d[2] = clip8(s2 >> kPrecision);
    }
  return dst;
}

Rgb vertical(const Rgb& src, int out_h) {
  const Taps t = taps(src.height, out_h);
  const size_t row = size_t(src.width) * 3;
  Rgb dst{src.width, out_h, std::vector<uint8_t>(row * out_h)};
  std::vector<int32_t> acc(row);
  for (int yy = 0; yy < out_h; ++yy) {
    const int32_t* k = &t.w[size_t(yy) * t.ksize];
    std::fill(acc.begin(), acc.end(), 1 << (kPrecision - 1));
    for (int y = 0; y < t.count[size_t(yy)]; ++y) {
      const uint8_t* s = &src.pixels[size_t(t.first[size_t(yy)] + y) * row];
      for (size_t i = 0; i < row; ++i) acc[i] += s[i] * k[y];
    }
    for (size_t i = 0; i < row; ++i) dst.pixels[size_t(yy) * row + i] = clip8(acc[i] >> kPrecision);
  }
  return dst;
}

}  // namespace

Rgb resize_bicubic(const Rgb& src, int width, int height) {
  if (width <= 0 || height <= 0 || width > 65536 || height > 65536) throw std::invalid_argument("resize: bad size");
  const bool h = width != src.width, v = height != src.height;
  if (h && v) return vertical(horizontal(src, width), height);
  if (h) return horizontal(src, width);
  if (v) return vertical(src, height);
  return src;
}

Rgb resize_padded(const Rgb& src, int width, int height) {
  if (src.width == width && src.height == height) return src;
  const float scale = std::min(float(width) / float(src.width), float(height) / float(src.height));
  const int nw = std::min(int(std::ceil(float(src.width) * scale)), width), nh = std::min(int(std::ceil(float(src.height) * scale)), height);
  const Rgb r = resize_bicubic(src, nw, nh);
  Rgb dst{width, height, std::vector<uint8_t>(size_t(width) * height * 3, 0)};
  const int ox = (width - nw) / 2, oy = (height - nh) / 2;
  for (int y = 0; y < nh; ++y)
    std::copy_n(&r.pixels[size_t(y) * nw * 3], size_t(nw) * 3, &dst.pixels[(size_t(y + oy) * width + ox) * 3]);
  return dst;
}

Patches patchify(const Rgb& p, int patch) {
  if (p.width % patch || p.height % patch) throw std::invalid_argument("patchify: sides must be whole patches");
  Patches out;
  out.cols = p.width / patch;
  out.rows = p.height / patch;
  out.dim = int64_t(3) * patch * patch;
  out.data.resize(size_t(out.cols) * out.rows * out.dim);
  for (int r = 0; r < out.rows; ++r)
    for (int c = 0; c < out.cols; ++c) {
      float* d = &out.data[(size_t(r) * out.cols + c) * out.dim];
      for (int ch = 0; ch < 3; ++ch)
        for (int y = 0; y < patch; ++y)
          for (int x = 0; x < patch; ++x)
            d[(ch * patch + y) * patch + x] = float(p.pixels[((size_t(r) * patch + y) * p.width + size_t(c) * patch + x) * 3 + ch]) / 255.0f;
    }
  return out;
}

Patches prepare(const Rgb& picture, const Spec& s) {
  const Size t = fit(picture.width, picture.height, s.patch, s.min_patches, s.max_patches);
  return patchify(resize_padded(picture, t.width, t.height), s.cut ? s.cut : s.patch);
}

}  // namespace eklow::image
