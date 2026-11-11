// The decode GEMVs of Q8_0I (xmx_fmt_gemv.hpp), a file of its own so that the formats compile in parallel.
#include "kernels/xmx_fmt_gemv.hpp"

namespace eklow::xmx::fmt {

template void gemv<WeightFormat::Q8_0I>(sycl::queue&, const Activations&, const Weight&, float*);

}  // namespace eklow::xmx::fmt
