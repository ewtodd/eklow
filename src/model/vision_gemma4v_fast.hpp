// The fast path's state (vision_gemma4v_fast.cpp); included by vision_gemma4v.cpp only so that its constructor can
// destroy it.
#pragma once

#include <memory>

#include "kernels/vision_attn.hpp"
#include "kernels/xmx.hpp"
#include "model/vision_gemma4v.hpp"
#include "rt/copy_engine.hpp"

namespace eklow::model {

struct VisionGemma4V::Fast {
  struct Layer {
    uint8_t* blob = nullptr;  // pinned host memory
    const float *ln1, *ln2, *q_norm, *k_norm, *attn_post, *ffn_post;  // on the card
  };
  xmx::WeightFormat fmt = xmx::WeightFormat::F16;
  int64_t Dp = 0, Fp = 0, Kp = 0;  // head size padded to 32, ffn width padded to 128 (gate|up), to 32 (down's K)
  uint64_t t_off[4] = {}, s_off[4] = {}, blob_bytes = 0;  // qkv, o, gate|up, down in a layer's blob
  std::vector<Layer> layers;
  xmx::Weight patch_w, proj_w;
  const float* pos = nullptr;  // host: the position tables in the mapped file
  static constexpr int kStages = 2;
  static constexpr int n_stage = kStages;  // staging buffers: layer l goes to l % 2; a third bought nothing
  uint8_t* stage[kStages] = {};
  std::unique_ptr<rt::CopyEngine> copy;
  std::unique_ptr<sycl::queue> copy_q;  // when the card has no copy-only engine
  // Activations, for up to n_cap patches.
  int64_t n_cap = 0;
  float *x = nullptr, *wide = nullptr;
  sycl::half *act_main = nullptr, *act_ffn = nullptr, *qt = nullptr, *kt = nullptr, *vt = nullptr;
  vattn::Scratch att;
  int64_t wide_floats = 0;
  uint64_t held() const { return stage[0] ? uint64_t(n_stage) * blob_bytes : 0; }
};

}  // namespace eklow::model
