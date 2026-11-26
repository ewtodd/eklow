// Copies by kernel instead of the runtime's memcpy/memset, into memory the card reads afterwards.
//
// Under Level Zero's driver in-order command lists (UR_L0_USE_DRIVER_INORDER_LISTS=1, and the v2 adapter) a memcpy into
// device memory the card had read before can leave its kernels reading the old contents, even after a q.wait(): Inkling's
// decode ran each step on the previous token's embedding (DECISIONS 2026-11-26). A kernel's writes go through the card's
// own caches, so what the next kernel reads is what was written, whatever the runtime. Sources may be device or host
// (USM) memory: a kernel reads pinned host memory over the bus.
#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace eklow::kcopy {

// dst = src, `bytes` of them (any count; 16-byte steps where both are aligned). The kernel's event (none for 0 bytes).
sycl::event copy(sycl::queue& q, void* dst, const void* src, size_t bytes);
// dst = 0
void zero(sycl::queue& q, void* dst, size_t bytes);

// Up to kMaxSegs copies of 4-byte words in one launch.
constexpr int kMaxSegs = 16;
struct Seg {
  void* dst;
  const void* src;
  uint32_t words;
};
void copy_segs(sycl::queue& q, const Seg* segs, int n);

}  // namespace eklow::kcopy
