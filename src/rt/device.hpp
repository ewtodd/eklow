// The card eklow runs on: the discrete Level Zero GPU with the most memory. The same cards also
// show up through OpenCL, and an integrated GPU reports the host's RAM as its own, so it would win
// on size alone (fiona's Iris Xe reports 28.6 GiB against the A770M's 16).
#pragma once

#include <sycl/sycl.hpp>

namespace eklow::rt {

// Level Zero's own flag; a device whose properties cannot be read counts as integrated.
bool integrated(const sycl::device& d);

// Throws std::runtime_error when there is no Level Zero GPU.
sycl::device pick_device();

}  // namespace eklow::rt
