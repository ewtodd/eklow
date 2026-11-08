// The fast path's state of VisionQwen3VL (vision_qwen3vl_fast.cpp), and the arithmetic both paths share; included by
// vision_qwen3vl.cpp and vision_qwen3vl_fast.cpp only.
#pragma once

#include <cmath>
#include <memory>

#include "kernels/vision_attn.hpp"
#include "kernels/xmx.hpp"
#include "model/vision_qwen3vl.hpp"
#include "rt/copy_engine.hpp"

namespace eklow::model {

// ggml's GGML_SCALE_MODE_BILINEAR | ALIGN_CORNERS (ggml_compute_forward_upscale_f32) for one destination pixel of the
// learned position table (n_src x n_src, row-major) resized to rows x cols: the four source rows and the weights, in ggml's
// float arithmetic.
struct PosLerp {
  int i00, i01, i10, i11;  // table rows: (y0,x0) (y0,x1) (y1,x0) (y1,x1)
  float w00, w01, w10, w11;
};
inline PosLerp pos_lerp(int iy, int ix, int rows, int cols, int n_src) {
  float sf0 = float(cols) / float(n_src), sf1 = float(rows) / float(n_src);
  sf0 = cols > 1 && n_src > 1 ? float(cols - 1) / float(n_src - 1) : sf0;
  sf1 = rows > 1 && n_src > 1 ? float(rows - 1) / float(n_src - 1) : sf1;
  auto axis = [&](int i, float sf, int& a, int& b, float& d) {
    const float v = float(i) / sf;
    a = int(std::floor(v));
    b = a + 1;
    a = std::max(0, std::min(a, n_src - 1));
    b = std::max(0, std::min(b, n_src - 1));
    d = std::max(0.0f, std::min(v - float(a), 1.0f));
  };
  int y0, y1, x0, x1;
  float dy, dx;
  axis(iy, sf1, y0, y1, dy);
  axis(ix, sf0, x0, x1, dx);
  return {y0 * n_src + x0, y0 * n_src + x1, y1 * n_src + x0, y1 * n_src + x1,
          (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy};
}

struct VisionQwen3VL::Fast {
  // One streamed unit: a layer (qkv, o, up, down) or a piece of the merger (mm0's two halves, mm2) in part 0.
  struct Unit {
    uint8_t* blob = nullptr;  // pinned host memory (streamed) or device memory (resident)
  };
  struct LayerVecs {
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b, *qkv_b, *o_b, *up_b, *down_b;  // on the card
  };
  xmx::WeightFormat fmt = xmx::WeightFormat::F16;
  int64_t Dp = 0, Fp = 0, Kp = 0;  // head size padded to 32, ffn width padded to 128 (up's N), to 32 (down's K)
  uint64_t t_off[4] = {}, blob_bytes = 0;  // qkv, o, up, down in a layer's blob
  std::vector<Unit> units;                 // layers then mm0 low, mm0 high, mm2
  std::vector<LayerVecs> vecs;
  xmx::Weight patch_w;
  const float* pos = nullptr;  // host: the position table in the mapped file
  const float *post_w = nullptr, *post_b = nullptr, *mm0_b = nullptr, *mm2_b = nullptr, *patch_b = nullptr;  // on the card
  static constexpr int kStages = 2;
  uint8_t* stage[kStages] = {};
  std::unique_ptr<rt::CopyEngine> copy;
  std::unique_ptr<sycl::queue> copy_q;
  // Activations, for up to n_cap patches.
  int64_t n_cap = 0;
  float* x = nullptr;
  float* wide = nullptr;
  sycl::half *act_main = nullptr, *act_ffn = nullptr, *qt = nullptr, *kt = nullptr, *vt = nullptr;
  vattn::Scratch att;
  float* small = nullptr;            // per-image: position table and RoPE angles
  uint8_t* rgb = nullptr;            // per-image: the pixels
  std::shared_ptr<float> out_buf;    // per-image: the rows, handed out while no caller holds the last ones
  size_t out_floats = 0;
  int64_t wide_floats = 0;
  uint64_t held() const { return stage[0] ? uint64_t(kStages) * blob_bytes : 0; }
};

}  // namespace eklow::model
