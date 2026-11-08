// Qwen3.5/3.6's vision encoder (llama.cpp's projector "qwen3vl_merger": Ornith-1.5's and stock Qwen3.6-35B-A3B's
// mmproj), following tools/mtmd/models/qwen3vl.cpp and clip_graph::build_attn at the oracle's revision, on the
// reference kernels (fp32) or (the default) on the matrix units. For an image prepared by image/qwen3vl_image.hpp:
//   x = patch_embd . pixels + patch_bias + pos[interpolated to the image's grid]   (two conv kernels for two frames,
//       both fed the same picture: one weight, their sum)
//   27 layers of 1152 (16 heads of 72): x += o(attn(ln1(x))), x += down(gelu_tanh(up(ln2(x)))) with LayerNorm
//     (weight and bias, eps 1e-6), a fused qkv with bias, "M-RoPE" of the vision kind (the first 36 dims of a head are
//     the pair's first member, the last 36 its second: pairs 0..17 turn by the patch's row, 18..35 by its column,
//     base 10000), scale 1/sqrt(72), no mask, biases on o, up and down
//   post LayerNorm, then four neighbours (a 2 x 2 block) side by side: [n / 4][4608] -> gelu(mm0) -> mm2 -> [n / 4][2048]
// The patches are in the model's merge order (the 4 patches of an output token are consecutive), so the output
// rows are the merged grid row by row. No deepstack layers (Ornith, stock Qwen3.6): a file with some is refused.
// The rows stand in for the text model's token embeddings between <|vision_start|> and <|vision_end|>.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string_view>
#include <vector>

#include "image/qwen3vl_image.hpp"
#include "kernels/ref.hpp"
#include "model/gguf.hpp"
#include "model/vision_gemma4v.hpp"  // DeviceRows

namespace eklow::model {

// The rows of one image and its merged grid (llama.cpp's nx, ny of the image tokens): rows.n == grid_w * grid_h,
// row r is the merged patch (r / grid_w, r % grid_w).
struct QwenImageRows {
  DeviceRows rows;
  int grid_w = 0, grid_h = 0;
};

struct QwenVisionOptions {
  bool reference = false;  // fp32 on the reference kernels, all weights on the card as stored
  bool resident = false;   // fast path: every layer's weights on the card (no streaming from host memory)
};

class VisionQwen3VL {
 public:
  using Options = QwenVisionOptions;
  VisionQwen3VL(sycl::queue& q, const gguf::File& f, Options o = Options());
  ~VisionQwen3VL();
  VisionQwen3VL(const VisionQwen3VL&) = delete;
  VisionQwen3VL& operator=(const VisionQwen3VL&) = delete;

  // The preprocessing spec for this file (mean and std from it) with llama.cpp's token limits `min_tokens` and
  // `max_tokens` (its --image-min-tokens / --image-max-tokens; its own defaults are 8 and 4096).
  image::QwenSpec spec(int max_tokens = 4096, int min_tokens = 8) const;

  // `picture` is image::prepare_qwen3vl's output (sides multiples of 32, at most n_pos x 16 pixels... any grid the
  // position table can interpolate to). Safe to call from several threads (one at a time runs). The fast path
  // allocates its activations and staging on first use (or reserve()) and keeps them until release().
  QwenImageRows encode(const image::Rgb& picture);
  // Decode, prepare and encode image bytes (JPEG, PNG, BMP, GIF as stb_image reads them). Throws std::invalid_argument
  // for what cannot be decoded.
  QwenImageRows encode(std::span<const uint8_t> bytes, const image::QwenSpec& spec);

  // Card memory, as VisionGemma4V's: reserve(tokens) allocates what encode() needs for images of up to `tokens` output
  // rows (the rows themselves, which the caller owns, excluded); card_bytes(tokens) is what the class holds while
  // encoding such an image; bytes() what it holds now; release() frees the staging and the activations.
  void reserve(int tokens);
  // Runs one synthetic image of `tokens` output rows through the encoder and drops the rows, so that the driver's scratch space for
  // its kernels exists before anything reads the card's free memory (reserve() first; the rows are not kept).
  void warm(int tokens);
  uint64_t card_bytes(int tokens = 1024) const;
  uint64_t bytes() const { return bytes_ + held_; }
  void release();
  bool reference() const { return reference_; }
  bool resident() const { return resident_; }

  int64_t n_embd() const { return n_out_; }  // the language model's width: rows are this wide

  // For checking against the oracle: called with llama.cpp's graph tensor name ("-N" for layer N) and the tensor on the
  // card, shape [rows][dim]; the queue is idle when it runs. The reference path gives patch_bias, inp_pos_emb, ln1-N,
  // Qcur_rope-N, Kcur_rope-N, kqv_out-N, attn_out-N, ffn_inp-N, ffn_out-N, layer_out-N and embeddings; the fast path
  // inp_pos_emb, layer_out-N and embeddings.
  using Tap = std::function<void(std::string_view name, const float* dev, int64_t rows, int64_t dim)>;
  void set_tap(Tap t) { tap_ = std::move(t); }

 private:
  struct Linear {
    ref::Weight w;
    const float* bias = nullptr;
  };
  struct Layer {
    Linear qkv, o, up, down;
    const float *ln1_w, *ln1_b, *ln2_w, *ln2_b;
  };
  QwenImageRows encode_ref(const image::Rgb& picture);
  QwenImageRows encode_fast(const image::Rgb& picture);
  bool load_fast(const gguf::File& f);
  void ensure(int64_t n_patches);
  void free_activations();
  uint64_t activation_bytes(int64_t n_patches) const;

  sycl::queue& q_;
  std::mutex mu_;
  std::vector<void*> owned_;
  uint64_t bytes_ = 0;  // resident weights
  uint64_t held_ = 0;   // staging and activations now
  bool reference_ = false, resident_ = false;
  int64_t n_in_ = 0, n_embd_ = 0, n_out_ = 0, n_pos_ = 0, n_ff_ = 0, n_head_ = 0, d_head_ = 0, n_hidden_ = 0;
  int merge_ = 2, patch_ = 16;
  float eps_ = 1e-6f, rope_theta_ = 10000.0f;
  float mean_[3] = {0.5f, 0.5f, 0.5f}, std_[3] = {0.5f, 0.5f, 0.5f};
  ref::Weight patch0_, patch1_, mm0_, mm2_;
  const float *patch_bias_ = nullptr, *pos_ = nullptr, *post_w_ = nullptr, *post_b_ = nullptr, *mm0_b_ = nullptr, *mm2_b_ = nullptr;
  std::vector<Layer> layers_;
  Tap tap_;

  struct Fast;
  std::unique_ptr<Fast> fast_;
};

}  // namespace eklow::model
