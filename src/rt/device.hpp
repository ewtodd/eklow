// The card eklow runs on: the discrete Level Zero GPU with the most memory. The same cards also
// show up through OpenCL, and an integrated GPU reports the host's RAM as its own, so it would win
// on size alone (fiona's Iris Xe reports 28.6 GiB against the A770M's 16).
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <optional>

namespace eklow::rt {

// Level Zero's own flag; a device whose properties cannot be read counts as integrated.
bool integrated(const sycl::device& d);

// Device memory not in use by anyone (Level Zero Sysman; the nix package and dev shell set the variables
// it needs, DECISIONS 2026-10-24), or nullopt when the driver does not say. Allocations succeed past what
// the card holds and the first kernels then fail, so this, not malloc, says what fits.
std::optional<uint64_t> free_memory(const sycl::device& d);

// Throws std::runtime_error when there is no Level Zero GPU.
sycl::device pick_device();

}  // namespace eklow::rt
