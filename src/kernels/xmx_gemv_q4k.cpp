// The decode GEMVs of Q4_K (xmx_fmt_gemv.hpp), a file of its own so that the formats compile in parallel.
#include "kernels/xmx_fmt_gemv.hpp"

namespace eklow::xmx::fmt {

template void gemv<WeightFormat::Q4_K>(sycl::queue&, const Activations&, const Weight&, float*);
template void gemv_grouped<WeightFormat::Q4_K>(sycl::queue&, const Activations&, const Activations*, const ExpertTable&, const int32_t*,
                                             int64_t, int64_t, int64_t, float*);

}  // namespace eklow::xmx::fmt
