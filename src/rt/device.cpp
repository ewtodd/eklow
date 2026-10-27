#include "rt/device.hpp"

#include <level_zero/ze_api.h>

#include <algorithm>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <vector>

namespace eklow::rt {

bool integrated(const sycl::device& d) {
  ze_device_properties_t props{};
  props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  if (d.get_backend() != sycl::backend::ext_oneapi_level_zero ||
      zeDeviceGetProperties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d), &props) != ZE_RESULT_SUCCESS)
    return true;
  return props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED;
}

std::optional<uint64_t> free_memory(const sycl::device& d) {
  if (!d.has(sycl::aspect::ext_intel_free_memory)) return std::nullopt;
  return uint64_t(d.get_info<sycl::ext::intel::info::device::free_memory>());
}

sycl::device pick_device() {
  std::vector<sycl::device> gpus;
  for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
    if (d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
  if (gpus.empty()) throw std::runtime_error("no Level Zero GPUs (is the Level Zero driver on the loader path?)");
  auto rank = [](const sycl::device& d) {
    return std::pair{!integrated(d), d.get_info<sycl::info::device::global_mem_size>()};
  };
  return *std::max_element(gpus.begin(), gpus.end(), [&](const sycl::device& a, const sycl::device& b) { return rank(a) < rank(b); });
}

}  // namespace eklow::rt
