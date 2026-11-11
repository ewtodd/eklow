// The decode GEMVs of Q4_K (xmx_fmt_gemv.hpp), a file of its own so that the formats compile in parallel.
#include "kernels/xmx_fmt_gemv.hpp"

namespace eklow::xmx::fmt {

template void gemv<WeightFormat::Q4_K>(sycl::queue&, const Activations&, const Weight&, float*);

}  // namespace eklow::xmx::fmt
