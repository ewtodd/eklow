#include "rt/copy_engine.hpp"

#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <stdexcept>
#include <vector>

namespace eklow::rt {

namespace {
void check(ze_result_t r, const char* what) {
  if (r != ZE_RESULT_SUCCESS) throw std::runtime_error(std::string("copy engine: ") + what + " failed (" + std::to_string(int(r)) + ")");
}
}  // namespace

CopyEngine::CopyEngine(const sycl::queue& q) {
  ctx_ = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  ze_device_handle_t dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  uint32_t n = 0;
  check(zeDeviceGetCommandQueueGroupProperties(dev, &n, nullptr), "queue group query");
  std::vector<ze_command_queue_group_properties_t> props(n, ze_command_queue_group_properties_t{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES});
  check(zeDeviceGetCommandQueueGroupProperties(dev, &n, props.data()), "queue group query");
  int ordinal = -1;
  for (uint32_t i = 0; i < n; ++i)
    if ((props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && !(props[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)) {
      ordinal = int(i);
      break;
    }
  if (ordinal < 0) throw std::runtime_error("copy engine: no copy-only queue group");
  ze_command_queue_desc_t d{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC};
  d.ordinal = uint32_t(ordinal);
  d.index = 0;
  d.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  check(zeCommandListCreateImmediate(ctx_, dev, &d, &list_), "immediate command list");
  ze_event_pool_desc_t pd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC};
  pd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
  pd.count = kEvents;
  check(zeEventPoolCreate(ctx_, &pd, 1, &dev, &pool_), "event pool");
  for (int i = 0; i < kEvents; ++i) {
    ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC};
    ed.index = uint32_t(i);
    ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
    ed.wait = ZE_EVENT_SCOPE_FLAG_HOST;
    check(zeEventCreate(pool_, &ed, &ev_[i]), "event");
  }
}

CopyEngine::~CopyEngine() {
  // Fences may still be in flight; the callers wait for what they use.
  for (auto e : ev_)
    if (e) zeEventDestroy(e);
  if (pool_) zeEventPoolDestroy(pool_);
  if (list_) zeCommandListDestroy(list_);
}

void CopyEngine::copy(void* dst, const void* src, size_t bytes) { check(zeCommandListAppendMemoryCopy(list_, dst, src, bytes, nullptr, 0, nullptr), "copy"); }

int CopyEngine::fence() {
  const int id = next_;
  next_ = (next_ + 1) % kEvents;
  check(zeEventHostReset(ev_[id]), "event reset");
  check(zeCommandListAppendBarrier(list_, ev_[id], 0, nullptr), "barrier");
  return id;
}

bool CopyEngine::done(int id) const { return zeEventQueryStatus(ev_[id]) == ZE_RESULT_SUCCESS; }

void CopyEngine::wait(int id) { check(zeEventHostSynchronize(ev_[id], UINT64_MAX), "wait"); }

}  // namespace eklow::rt
