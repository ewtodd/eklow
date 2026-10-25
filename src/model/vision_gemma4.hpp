// Gemma 4's encoder-free vision embedder, the 12B's mmproj (projector "gemma4uv"), following
// llama.cpp's graph (tools/mtmd/models/gemma4uv.cpp at the oracle's revision) on the reference
// kernels. Per patch of a prepared image (image/gemma4_image.hpp):
//   x = layernorm(patch), x = patch_embd . x + bias, x = layernorm(x)   (PyTorch's eps, 1e-5)
//   x = layernorm(x + pos_x[column] + pos_y[row])                       (learned tables)
//   e = input_projection . rmsnorm(x)                                    (no weight)
// The rows stand in for the text model's token embeddings, unscaled, between <|image> and <image|>.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
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

class VisionGemma4 {
 public:
  // Copies the weights to the card. Throws for an mmproj without the gemma4uv vision projector.
  VisionGemma4(sycl::queue& q, const gguf::File& f);
  ~VisionGemma4();
  VisionGemma4(const VisionGemma4&) = delete;
  VisionGemma4& operator=(const VisionGemma4&) = delete;

  // One embedding row per patch, row by row. Safe to call from several threads (one at a time
  // runs). Throws std::invalid_argument for a grid wider or taller than the position tables.
  DeviceRows encode(const image::Patches& p);

  int64_t n_embd() const { return n_out_; }
  const image::Spec& spec() const { return spec_; }
  uint64_t bytes() const { return bytes_; }

 private:
  sycl::queue& q_;
  std::mutex mu_;
  std::vector<void*> owned_;
  uint64_t bytes_ = 0;
  image::Spec spec_;
  int64_t n_in_, n_embd_, n_out_, n_pos_;
  float eps_;
  ref::Weight patch_embd_, proj_;
  const float *patch_bias_, *norm1_w_, *norm1_b_, *norm2_w_, *norm2_b_, *norm3_w_, *norm3_b_, *pos_;
};

}  // namespace eklow::model
