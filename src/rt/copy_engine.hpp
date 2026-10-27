// The card's copy (blitter) engine through Level Zero, for host-to-card copies that run beside
// kernels. A SYCL queue's memcpy goes to the compute engine and shares it with the kernels in flight
// (copy and grouped GEMM each took 2x as long when run together, DECISIONS 2026-10-28); a command list
// on the copy-only queue group runs on its own hardware.
#pragma once

#include <level_zero/ze_api.h>

#include <sycl/sycl.hpp>

#include <cstddef>

namespace eklow::rt {

class CopyEngine {
 public:
  // Throws std::runtime_error when the device has no copy-only queue group.
  explicit CopyEngine(const sycl::queue& q);
  ~CopyEngine();
  CopyEngine(const CopyEngine&) = delete;
  CopyEngine& operator=(const CopyEngine&) = delete;

  // Appends a copy; the engine runs them in order, in the background.
  void copy(void* dst, const void* src, size_t bytes);
  // Appends a marker behind everything copied so far and returns its index for wait().
  int fence();
  // True once the fence has passed.
  bool done(int fence_id) const;
  // Blocks until the fence has passed.
  void wait(int fence_id);

 private:
  static constexpr int kEvents = 64;
  ze_context_handle_t ctx_ = nullptr;
  ze_command_list_handle_t list_ = nullptr;
  ze_event_pool_handle_t pool_ = nullptr;
  ze_event_handle_t ev_[kEvents] = {};
  int next_ = 0;
};

}  // namespace eklow::rt
