// Gemma 4's vision encoder (llama.cpp's projector "gemma4v": the 26B A4B's mmproj and EmbeddingGemma 2's),
// following tools/mtmd/models/gemma4v.cpp and clip_graph::build_vit at the oracle's revision on the
// reference kernels, or (the default) on the matrix units. For an image prepared by image/gemma4_image.hpp (16-pixel patches):
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

// The fast path (DECISIONS 2026-10-28 "the vision encoder on the matrix units"): every matrix on xmx's fp16 GEMM
// (BF16/F32 weights as fp16, Q8_0 as Q8_0), attention as two GEMM-shaped kernels per segment of keys
// (kernels/vision_attn.hpp), the elementwise work in a few fused kernels, and the weights streamed one layer at a
// time from pinned host memory into two device staging buffers (copy of layer i + 1 beside layer i's compute), so
// the encoder holds only its activations and the staging on the card, and only while the caller lets it
// (release()). The reference path (constructor flag) keeps every weight on the card and is what the oracle checks
// were made on; files with biases, clamps or layer scales also take it.
class VisionGemma4V {
 public:
  // Fast path by default; reference = true runs the reference kernels (all weights uploaded as stored).
  // Throws for an mmproj without the gemma4v vision projector.
  VisionGemma4V(sycl::queue& q, const gguf::File& f, bool reference = false);
  ~VisionGemma4V();
  VisionGemma4V(const VisionGemma4V&) = delete;
  VisionGemma4V& operator=(const VisionGemma4V&) = delete;

  // `p` is a prepare()d image with this encoder's spec(): its grid in 16-pixel patches must be a
  // multiple of the pooling kernel on both sides. One row per pooled patch, row by row. Safe to call
  // from several threads (one at a time runs). Throws std::invalid_argument for a grid the position
  // tables or the pooling do not take. The fast path allocates its activations and staging on first use (or
  // reserve()) and keeps them until release(); an image larger than reserved grows them.
  DeviceRows encode(const image::Patches& p);

  // The spec for a budget of `tokens` output rows (llama.cpp's image_max_tokens; the minimum stays
  // 70). 0 is the model's own maximum, 1120.
  image::Spec spec(int tokens = 280) const;

  // Card memory. reserve(tokens) allocates what encode() needs for images of up to `tokens` output rows (the fast
  // path; the output rows themselves, which the caller owns, are not included). card_bytes(tokens) is what the
  // class holds while encoding such an image (resident weights, staging, activations): count it before placing
  // anything else on the card. bytes() is what it holds now. release() frees the staging and the activations,
  // leaving the small resident part (the patch embedding and the projection); the next encode() allocates again.
  void reserve(int tokens);
  uint64_t card_bytes(int tokens = 1120) const;
  uint64_t bytes() const { return bytes_ + held_; }
  void release();
  bool reference() const { return reference_; }

  int64_t n_embd() const { return n_out_; }

  // For checking against the oracle: called with a name (llama.cpp's graph tensor name, "-N" for
  // layer N), the rows on the card and their shape [rows][dim]; the queue is idle when it runs. The fast path
  // gives inp, pos_embd, layer_out-N and projected.
  using Tap = std::function<void(std::string_view name, const float* dev, int64_t rows, int64_t dim)>;
  void set_tap(Tap t) { tap_ = std::move(t); }
  // Round the activations of BF16 weights to bf16 before each matmul, as ggml's CPU backend does (the oracle).
  // Reference path only.
  void set_bf16_activations(bool on) { bf16_act_ = on; }

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

  bool load_fast(const gguf::File& f);
  DeviceRows encode_fast(const image::Patches& p);
  void ensure(int64_t n_patches);
  void free_activations();
  uint64_t activation_bytes(int64_t n_patches) const;

  sycl::queue& q_;
  std::mutex mu_;
  std::vector<void*> owned_;
  uint64_t bytes_ = 0;  // resident weights
  uint64_t held_ = 0;   // staging and activations now
  bool reference_ = false;
  int64_t n_in_, n_embd_, n_out_, n_pos_, n_ff_, n_head_, n_head_kv_, d_head_;
  int merge_, patch_;
  float eps_, rope_theta_;
  ref::Weight patch_embd_, proj_;
  Linear proj_lin_;
  const float *pos_, *std_bias_ = nullptr, *std_scale_ = nullptr;
  std::vector<Layer> layers_;
  Tap tap_;
  bool bf16_act_ = false;

  // Fast path state.
  struct Fast;
  std::unique_ptr<Fast> fast_;
};

}  // namespace eklow::model
