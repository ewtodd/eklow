// Input rows given as embeddings instead of token ids: an image's rows from the vision encoder, which stand in for
// the text model's embedding of a token (llama.cpp feeds them unscaled, where a Gemma-family token embedding is multiplied by
// sqrt(n_embd); Qwen's is not scaled). A span of a chunk or an embedding input.
#pragma once

#include <cstdint>

namespace eklow::model {

struct ImageRows {
  int64_t offset = 0;  // the first token of the span within its chunk or input
  int64_t n = 0;       // tokens (rows)
  const float* rows = nullptr;  // host memory, fp32 [n][n_embd]; must outlive the step
  // The grid of rows for an M-RoPE text model (Qwen3.6): nx columns by ny rows, n = nx * ny, in raster order. The span takes
  // max(nx, ny) positions of the text, not n (mtmd_image_tokens_get_n_pos), and its rows' positions are (t, row, column) from
  // the span's first position. 0: no grid (the span is n consecutive positions).
  int64_t nx = 0, ny = 0;
  int64_t n_pos() const { return nx > 0 ? (nx > ny ? nx : ny) : n; }
};

}  // namespace eklow::model
