// The decode GEMVs of Q6_K (xmx_fmt_gemv.hpp), a file of its own so that the formats compile in parallel.
#include "kernels/xmx_fmt_gemv.hpp"

namespace eklow::xmx::fmt {

template void gemv<WeightFormat::Q6_K>(sycl::queue&, const Activations&, const Weight&, float*);

}  // namespace eklow::xmx::fmt
