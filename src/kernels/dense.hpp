// Small kernels for dense models beside the chat engine (vega's reader, src/vega/reader.hpp): the residual add with
// RMSNorm into the fp16 GEMMs' activations. They only enqueue; the queue must be in order.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "kernels/xmx.hpp"

namespace eklow::dense {

// x += add (when given), then rmsnorm(x) * w into `act` as fp16 activations (marked as act_rows rows: the rows past `rows`
// keep what they held) and, when h is given, into h in fp32. A work-group of 256 per row; E a multiple of 256, at most 2048.
void norm_quant(sycl::queue& q, float* x, const float* add, const float* w, int64_t rows, int64_t E, float eps, xmx::Activations& act,
                int64_t act_rows, float* h);

}  // namespace eklow::dense
