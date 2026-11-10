// Images as Gemma 4's vision encoders take them, prepared as Hugging Face's Gemma4ImageProcessor does (the
// model's reference; llama.cpp's mtmd at the oracle's revision differs, see DECISIONS 2026-11-10): decoded to
// RGB by stb_image (the same v2.30 llama.cpp vendors), resized straight to the largest whole number of pooled
// patches (48 pixels for the encoder with 16-pixel patches and 3x3 pooling) that keeps the aspect ratio and
// fits the token budget, smaller pictures scaled up to it (Pillow's bicubic resampling in fixed point, which
// torchvision's antialiased bicubic follows), scaled to [0, 1] and cut into patches. Host code only.
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

// Qwen-VL's smart_resize (mtmd's calc_size_preserved_ratio with align = patch): each side rounded to
// a multiple of patch, halves to even as Python's round() does, then scaled down or up into the pixel
// bounds of min_patches and max_patches patches.
Size fit(int width, int height, int patch, int min_patches, int max_patches);

// Hugging Face's get_aspect_ratio_preserving_size for Gemma 4: the largest sides, multiples of patch
// with the aspect ratio kept, of at most max_patches patches; a picture below that is scaled up.
// A side that rounds to nothing becomes one patch and the other takes the aspect ratio, at most
// max_patches long. Throws std::invalid_argument when both do.
Size fill(int width, int height, int patch, int max_patches);

// Pillow's bicubic resampling (a = -0.5), as llama.cpp's resize_pillow.
Rgb resize_bicubic(const Rgb& src, int width, int height);

struct Patches {
  int cols = 0, rows = 0;  // the patch grid
  int64_t dim = 0;         // 3 * patch * patch
  std::vector<float> data; // [rows * cols][dim], patches row by row, each [channel][y][x] in [0, 1]
};

// A picture whose sides are whole patches, cut as ggml's im2col cuts it.
Patches patchify(const Rgb& picture, int patch);

struct Spec {
  // patch: the unit tokens are counted in (the pooled patch: 16 px times the pooling kernel for GEMMA4V);
  // max_tokens: the budget in those units (Hugging Face's max_soft_tokens);
  // cut: the side of the patches the encoder takes, 0 for patch.
  int patch = 48, max_tokens = 280, cut = 0;
};

// decode's output resized to fill's size for the budget, before it is cut.
Rgb fitted(const Rgb& picture, const Spec& spec = {});

// fitted's picture cut into patches.
Patches prepare(const Rgb& picture, const Spec& spec = {});

}  // namespace eklow::image
