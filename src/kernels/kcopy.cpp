#include "kernels/kcopy.hpp"

#include <stdexcept>

namespace eklow::kcopy {

sycl::event copy(sycl::queue& q, void* dst, const void* src, size_t bytes) {
  if (bytes == 0) return {};
  const auto d = reinterpret_cast<uintptr_t>(dst), s = reinterpret_cast<uintptr_t>(src);
  if (d % 16 == 0 && s % 16 == 0 && bytes % 16 == 0) {
    const size_t n = bytes / 16;
    auto* o = static_cast<sycl::uint4*>(dst);
    const auto* i = static_cast<const sycl::uint4*>(src);
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> k) { o[k] = i[k]; });
  } else if (d % 4 == 0 && s % 4 == 0 && bytes % 4 == 0) {
    const size_t n = bytes / 4;
    auto* o = static_cast<uint32_t*>(dst);
    const auto* i = static_cast<const uint32_t*>(src);
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> k) { o[k] = i[k]; });
  } else {
    auto* o = static_cast<uint8_t*>(dst);
    const auto* i = static_cast<const uint8_t*>(src);
    return q.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> k) { o[k] = i[k]; });
  }
}

void zero(sycl::queue& q, void* dst, size_t bytes) {
  if (bytes == 0) return;
  if (reinterpret_cast<uintptr_t>(dst) % 4 == 0 && bytes % 4 == 0) {
    auto* o = static_cast<uint32_t*>(dst);
    q.parallel_for(sycl::range<1>(bytes / 4), [=](sycl::id<1> k) { o[k] = 0; });
  } else {
    auto* o = static_cast<uint8_t*>(dst);
    q.parallel_for(sycl::range<1>(bytes), [=](sycl::id<1> k) { o[k] = 0; });
  }
}

void copy_segs(sycl::queue& q, const Seg* segs, int n) {
  if (n < 1) return;
  if (n > kMaxSegs) throw std::runtime_error("kcopy: more than 16 segments");
  struct Table {
    uint32_t* dst[kMaxSegs];
    const uint32_t* src[kMaxSegs];
    uint32_t begin[kMaxSegs + 1];
    int n;
  } t{};
  uint32_t total = 0;
  for (int k = 0; k < n; ++k) {
    t.dst[k] = static_cast<uint32_t*>(segs[k].dst);
    t.src[k] = static_cast<const uint32_t*>(segs[k].src);
    t.begin[k] = total;
    total += segs[k].words;
  }
  t.begin[n] = total;
  t.n = n;
  if (total == 0) return;
  q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
    const uint32_t w = uint32_t(id[0]);
    int k = 0;
    while (k + 1 < t.n && w >= t.begin[k + 1]) ++k;
    t.dst[k][w - t.begin[k]] = t.src[k][w - t.begin[k]];
  });
}

}  // namespace eklow::kcopy
