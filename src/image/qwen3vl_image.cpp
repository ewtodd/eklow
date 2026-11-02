#include "image/qwen3vl_image.hpp"

#include <stdexcept>

namespace eklow::image {

Size qwen3vl_size(int width, int height, const QwenSpec& s) {
  if (width <= 0 || height <= 0) throw std::invalid_argument("qwen3vl_size: empty picture");
  return fit(width, height, s.patch * s.merge, s.min_tokens, s.max_tokens);
}

Rgb prepare_qwen3vl(const Rgb& picture, const QwenSpec& s) {
  const Size t = qwen3vl_size(picture.width, picture.height, s);
  return resize_padded(picture, t.width, t.height);
}

Patches patchify_merged(const Rgb& p, const QwenSpec& s) {
  const int patch = s.patch, m = s.merge, unit = patch * m;
  if (p.width % unit || p.height % unit) throw std::invalid_argument("patchify_merged: sides must be whole 32-pixel tiles");
  Patches out;
  out.cols = p.width / patch;
  out.rows = p.height / patch;
  out.dim = int64_t(3) * patch * patch;
  out.data.resize(size_t(out.cols) * out.rows * out.dim);
  size_t at = 0;
  for (int by = 0; by < out.rows; by += m)
    for (int bx = 0; bx < out.cols; bx += m)
      for (int dy = 0; dy < m; ++dy)
        for (int dx = 0; dx < m; ++dx) {
          float* d = &out.data[at++ * size_t(out.dim)];
          const int r = by + dy, c = bx + dx;
          for (int ch = 0; ch < 3; ++ch)
            for (int y = 0; y < patch; ++y)
              for (int x = 0; x < patch; ++x)
                d[(ch * patch + y) * patch + x] =
                    qwen_normalize(p.pixels[((size_t(r) * patch + y) * p.width + size_t(c) * patch + x) * 3 + ch], s.mean[ch], s.stddev[ch]);
        }
  return out;
}

}  // namespace eklow::image
