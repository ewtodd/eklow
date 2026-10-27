// Images as Gemma 4's vision encoders take them, prepared exactly as llama.cpp's mtmd does at the
// oracle's revision (tools/mtmd/clip.cpp and mtmd-image.cpp, mtmd_image_preprocessor_dyn_size): decoded
// to RGB by stb_image (the same v2.30 llama.cpp vendors), resized to a whole number of pooled patches
// (48 pixels for the encoder with 16-pixel patches and 3x3 pooling, the 12B's encoder-free embedder
// cuts 48 directly), aspect ratio kept, between a minimum and a maximum number of tokens (Pillow's
// bicubic resampling in fixed point; the resized picture is centred on black when rounding leaves a
// strip), scaled to [0, 1] and cut into patches. Host code only.
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
  // patch: the unit tokens are counted in (the pooled patch: 16 px times the pooling kernel for GEMMA4V);
  // cut: the side of the patches the encoder takes, 0 for patch.
  int patch = 48, min_patches = 70, max_patches = 1120, cut = 0;
};

// decode's output, fitted, resized and cut.
Patches prepare(const Rgb& picture, const Spec& spec = {});

}  // namespace eklow::image
