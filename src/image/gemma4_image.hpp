// Images as Gemma 4's vision embedder takes them, prepared exactly as llama.cpp's mtmd does at the
// oracle's revision for the encoder-free projector the 12B ships ("gemma4uv", tools/mtmd/clip.cpp
// and mtmd-image.cpp): decoded to RGB by stb_image (the same v2.30 llama.cpp vendors), resized to a
// whole number of 48-pixel patches, aspect ratio kept, between 70 and 1120 patches (Pillow's
// bicubic resampling in fixed point; the resized picture is centred on black when rounding leaves
// a strip), scaled to [0, 1] and cut into patches. Host code only.
#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace eklow::image {

struct Rgb {
  int width = 0, height = 0;
  std::vector<uint8_t> pixels;  // row-major, 3 bytes a pixel
};

// The bytes as stb_image reads them: JPEG, PNG, BMP or the first frame of a GIF (src/image/
// stb_image.cpp; not WebP). Throws std::invalid_argument for anything else and for pictures of more
// than max_pixels.
Rgb decode(std::span<const uint8_t> bytes, int64_t max_pixels = int64_t(1) << 26);

struct Size {
  int width, height;
};

// mtmd's calc_size_preserved_ratio with align = patch and the pixel bounds of min_patches and
// max_patches patches.
Size fit(int width, int height, int patch, int min_patches, int max_patches);

// Pillow's bicubic resampling (a = -0.5), as llama.cpp's resize_pillow.
Rgb resize_bicubic(const Rgb& src, int width, int height);

// mtmd's resize with PAD_CEIL: scaled to fit inside width x height, rounding each side up, and
// placed in the middle of a black picture of that size.
Rgb resize_padded(const Rgb& src, int width, int height);

struct Patches {
  int cols = 0, rows = 0;  // the patch grid
  int64_t dim = 0;         // 3 * patch * patch
  std::vector<float> data; // [rows * cols][dim], patches row by row, each [channel][y][x] in [0, 1]
};

// A picture whose sides are whole patches, cut as ggml's im2col cuts it.
Patches patchify(const Rgb& picture, int patch);

struct Spec {
  int patch = 48, min_patches = 70, max_patches = 1120;
};

// decode's output, fitted, resized and cut.
Patches prepare(const Rgb& picture, const Spec& spec = {});

}  // namespace eklow::image
