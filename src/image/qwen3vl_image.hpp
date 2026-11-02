// Images as Qwen3.5/3.6's vision encoder takes them (llama.cpp's PROJECTOR_TYPE_QWEN3VL, "qwen3vl_merger"), prepared as
// mtmd does at the oracle's revision (mtmd_image_preprocessor_dyn_size, tools/mtmd/mtmd-image.cpp and clip.cpp): decoded
// to RGB by stb_image (src/image/gemma4_image.hpp's decode), fitted to a multiple of 32 pixels (patch 16 x merge 2) by
// smart_resize's rule between a minimum and a maximum number of output tokens (each token is 32 x 32 pixels), resized
// with Pillow's bicubic and centred on black when rounding leaves a strip (the same resize as the Gemma encoder's), then
// scaled by (x / 255 - mean) / std (Ornith: 0.5 / 0.5). Host code only.
#pragma once

#include <cstdint>
#include <vector>

#include "image/gemma4_image.hpp"

namespace eklow::image {

struct QwenSpec {
  // llama.cpp's image_min_tokens / image_max_tokens (`--image-min-tokens`, `--image-max-tokens`); its own defaults for
  // Qwen-VL are 8 and 4096 (set_limit_image_tokens(8, 4096)), a GGUF may set pixel limits that replace them.
  int min_tokens = 8, max_tokens = 4096;
  int patch = 16, merge = 2;
  float mean[3] = {0.5f, 0.5f, 0.5f}, stddev[3] = {0.5f, 0.5f, 0.5f};
};

// The size prepare_qwen3vl will make of a width x height picture, in pixels (multiples of 32): the image's tokens are
// (width / 32) x (height / 32), known before the picture is decoded or resized.
Size qwen3vl_size(int width, int height, const QwenSpec& spec = {});

// The picture the encoder takes: a whole number of 32-pixel tiles on both sides.
Rgb prepare_qwen3vl(const Rgb& picture, const QwenSpec& spec = {});

// The encoder's input as ggml builds it, normalized and cut into patches of [channel][y][x] (768 floats), in the order
// of the model's spatial merge: the 2 x 2 patches of one output token follow each other (row-major inside the
// block, blocks row by row), so output token t is patches 4t .. 4t + 3. cols, rows count patches (16 pixels).
Patches patchify_merged(const Rgb& picture, const QwenSpec& spec = {});

// The pixel value the encoder sees for byte `v` of channel c: (v / 255 - mean) / std in float, as clip_image_f32::normalize.
inline float qwen_normalize(uint8_t v, float mean, float stddev) { return (float(v) / 255.0f - mean) / stddev; }

}  // namespace eklow::image
