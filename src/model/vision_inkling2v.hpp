// Inkling 2 Mini's vision tower (llama.cpp's inkling2v projector, mtmd/models/inkling2v.cpp at c74c71a6; DECISIONS
// 2026-11-25, "Inkling M5") on the card: a ViT of 23 layers (hidden 1,024, 16 heads of 64, FFN 4,096 with GELU-erf,
// LayerNorms with biases, layer scales), 20 x 20 patches with a CLS token, learned position embeddings (a 32 x 32 grid,
// resized bilinearly to the image's grid) and 2D RoPE (a head's first 32 dims rotated by the patch's column, the
// last 32 by its row, theta 10,000; the CLS token at 0, the patches from 1), the CLS dropped, 2 x 2 patches merged into
// one 4,096-wide token, then two linears (4,096 -> 4,096 -> 2,304, GELU-erf between, no biases). The linears run on
// the fp16 GEMM, attention on vision_attn; the weights stay on the card (~600 MB).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "model/gguf.hpp"
#include "image/gemma4_image.hpp"
#include "kernels/vision_attn.hpp"
#include "kernels/xmx.hpp"

namespace eklow::model {

class VisionInkling2V {
 public:
  // The mmproj file (general.architecture clip, clip.vision.projector_type inkling2v).
  VisionInkling2V(sycl::queue& q, const gguf::File& f);
  ~VisionInkling2V();
  VisionInkling2V(const VisionInkling2V&) = delete;
  VisionInkling2V& operator=(const VisionInkling2V&) = delete;

  // A stage's rows on the card ([rows][cols] fp32), for comparisons: "embeddings" (the patches' and the CLS token's,
  // before the first norm), "layer_out-L", "pixel_shuffle" (the merged tokens), "projected".
  using Dump = std::function<void(const std::string& name, const float* rows, int64_t n_rows, int64_t n_cols)>;
  // The image's tokens ([tokens][2304] fp32, raster order over the merged grid) from image::prepare_inkling's patches.
  std::vector<float> encode(const image::Patches& p, const Dump& dump = {});
  int64_t n_out() const { return n_out_; }
  uint64_t bytes() const { return bytes_; }

 private:
  struct Layer {
    xmx::Weight qkv, o, up, down;
    float *b_qkv = nullptr, *b_o = nullptr, *b_up = nullptr, *b_down = nullptr;
    float *ln1_w = nullptr, *ln1_b = nullptr, *ln2_w = nullptr, *ln2_b = nullptr, *ls1 = nullptr, *ls2 = nullptr;
  };
  void ensure(int64_t n_pos);
  float* upload_f32(const gguf::Tensor& t);
  xmx::Weight upload_f16(const std::vector<const gguf::Tensor*>& parts, int64_t k, int64_t k_pad);

  sycl::queue& q_;
  int64_t n_embd_ = 0, n_head_ = 0, d_head_ = 0, n_ff_ = 0, n_out_ = 0, n_proj_ = 0, grid_ = 0;
  float eps_ = 1e-5f, theta_ = 10000.0f;
  std::vector<Layer> layers_;
  xmx::Weight patch_w_, mm1_, mm2_;
  float *pre_ln_w_ = nullptr, *pre_ln_b_ = nullptr, *cls_ = nullptr;
  std::vector<float> pos_;  // host: the position grid [grid * grid][n_embd]
  std::vector<void*> owned_;
  uint64_t bytes_ = 0;
  // activations, for up to cap_ rows (positions)
  int64_t cap_ = 0;
  float *x_ = nullptr, *h_ = nullptr, *qkv_ = nullptr, *wide_ = nullptr, *pin_ = nullptr;
  int32_t* rp_ = nullptr;  // per row: the RoPE column and row
  sycl::half *qt_ = nullptr, *kt_ = nullptr, *vt_ = nullptr;
  xmx::Activations act_;
  vattn::Scratch att_;
};

}  // namespace eklow::model
