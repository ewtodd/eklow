// Inkling 2's image preprocessing, as llama.cpp's mtmd_image_preprocessor_inkling2v (c74c71a6, the M0 oracle's revision;
// DECISIONS 2026-11-25, "Inkling M5"): the picture at its own size, each side padded up to a multiple of 40 (patch 20,
// merged 2 x 2) with -1 before the normalization, unless the padded picture holds more than max_tokens x 1,600 pixels;
// then it is resized (bicubic, mtmd's default) to calc_size_preserved_ratio's size (each side a multiple of 40) first.
// The values are normalized as (x - 127.5) / 127.5 (mean and std 0.5 a channel, on 0..255) and cut into 20 x 20
// patches, each [channel][y][x]. Host code only.
#pragma once

#include <cstdint>

#include "image/gemma4_image.hpp"

namespace eklow::image {

constexpr int kInklingPatch = 20, kInklingMerge = 2, kInklingAlign = kInklingPatch * kInklingMerge;

// The size the picture is encoded at (each side a multiple of 40).
Size inkling_size(int width, int height, int64_t max_tokens);
// The image tokens of a picture of that size: (width / 40) x (height / 40), in raster order.
inline int64_t inkling_tokens(const Size& s) { return int64_t(s.width / kInklingAlign) * (s.height / kInklingAlign); }
Patches prepare_inkling(const Rgb& picture, int64_t max_tokens);

}  // namespace eklow::image
