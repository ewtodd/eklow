// Gemma 4's vision encoder (llama.cpp's projector "gemma4v": the 26B A4B's mmproj and EmbeddingGemma 2's),
// following tools/mtmd/models/gemma4v.cpp and clip_graph::build_vit at the oracle's revision on the
// reference kernels. For an image prepared by image/gemma4_image.hpp (16-pixel patches):
//   x = patch_embd . (2 * pixels - 1) + pos_x[column] + pos_y[row]       (no patch bias)
//   27 (26B) or 16 (EmbeddingGemma 2) layers, each
//     x += post_norm(attn(rms(x))), x += post_norm(ffn(rms(x)))
//     attention: q, k per-head rmsnorm with weight, 2D RoPE (NeoX; the first half of a head turns by the
//     column, the second by the row, base 100), v per-head rmsnorm without weight, scale 1, no mask
//     ffn: down(gelu_quick(gate) * up)   (gelu_quick is what the file's missing use_gelu key selects)
//   3x3 average pooling over the patch grid, * sqrt(n_embd), (x - std_bias) * std_scale if present,
//   e = input_projection . rmsnorm(x)                                          (no weight)
// The rows stand in for the text model's token embeddings, unscaled, between <|image> and <image|>.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <vector>

#include "image/gemma4_image.hpp"
#include "kernels/ref.hpp"
#include "model/gguf.hpp"

namespace eklow::model {

// Embedding rows in device memory, freed with the last copy.
struct DeviceRows {
  std::shared_ptr<float> data;
  int64_t n = 0, dim = 0;
};

class VisionGemma4V {
 public:
  // Copies the vision tensors to the card. Throws for an mmproj without the gemma4v vision projector.
  VisionGemma4V(sycl::queue& q, const gguf::File& f);
  ~VisionGemma4V();
  VisionGemma4V(const VisionGemma4V&) = delete;
  VisionGemma4V& operator=(const VisionGemma4V&) = delete;

  // `p` is a prepare()d image with this encoder's spec(): its grid in 16-pixel patches must be a
  // multiple of the pooling kernel on both sides. One row per pooled patch, row by row. Safe to call
  // from several threads (one at a time runs). Throws std::invalid_argument for a grid the position
  // tables or the pooling do not take.
  DeviceRows encode(const image::Patches& p);

  // The spec for a budget of `tokens` output rows (llama.cpp's image_max_tokens; the minimum stays
  // 70). 0 is the model's own maximum, 1120.
  image::Spec spec(int tokens = 280) const;

  int64_t n_embd() const { return n_out_; }
  uint64_t bytes() const { return bytes_; }

  // For checking against the oracle: called with a name (llama.cpp's graph tensor name, "-N" for
  // layer N), the rows on the card and their shape [rows][dim]; the queue is idle when it runs.
  using Tap = std::function<void(std::string_view name, const float* dev, int64_t rows, int64_t dim)>;
  void set_tap(Tap t) { tap_ = std::move(t); }

 private:
  struct Linear {
    ref::Weight w;
    bool clamp = false;
    float in_min = 0, in_max = 0, out_min = 0, out_max = 0;  // Gemma4ClippableLinear
    const float* bias = nullptr;
  };
  struct Layer {
    Linear q, k, v, o, up, gate, down;
    const float *ln1, *ln2, *q_norm, *k_norm, *attn_post, *ffn_post, *ls1, *ls2, *ls_out;
  };
  void mm(const Linear& l, const float* x, int64_t rows, float* out, float* scratch);

  sycl::queue& q_;
  std::mutex mu_;
  std::vector<void*> owned_;
  uint64_t bytes_ = 0;
  int64_t n_in_, n_embd_, n_out_, n_pos_, n_ff_, n_head_, n_head_kv_, d_head_;
  int merge_, patch_;
  float eps_, rope_theta_;
  ref::Weight patch_embd_, proj_;
  Linear proj_lin_;
  const float *pos_, *std_bias_ = nullptr, *std_scale_ = nullptr;
  std::vector<Layer> layers_;
  Tap tap_;
};

}  // namespace eklow::model
