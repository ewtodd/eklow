// What a forward pass hands a caller that wants its intermediate tensors (the reference forwards and the engine).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace eklow::model {

// Receives intermediate tensors under llama.cpp's graph names (attn_norm-0, Qcur_normed-0, ...,
// l_out-47, result_norm) with ggml shapes (ne[0] fastest). `wants` is asked before each tensor is
// copied off the card.
struct DumpSink {
  std::function<bool(const std::string&)> wants;
  std::function<void(const std::string& name, const std::vector<float>& data, const std::vector<int64_t>& ne)> put;
};

}  // namespace eklow::model
