// Input rows given as embeddings instead of token ids: an image's rows from the vision encoder, which stand in for
// the text model's embedding of a token (llama.cpp feeds them unscaled, where a token's embedding is multiplied by
// sqrt(n_embd)). A span of a chunk or an embedding input.
#pragma once

#include <cstdint>

namespace eklow::model {

struct ImageRows {
  int64_t offset = 0;  // the first token of the span within its chunk or input
  int64_t n = 0;       // tokens (rows)
  const float* rows = nullptr;  // host memory, fp32 [n][n_embd]; must outlive the step
};

}  // namespace eklow::model
