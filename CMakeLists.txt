cmake_minimum_required(VERSION 3.28)

project(eklow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# No C++20 modules; without this CMake 4 wants clang-scan-deps for every try_compile.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Device code is compiled ahead of time (ocloc) for these targets; "spir64" alone means JIT.
set(EKLOW_SYCL_TARGETS "intel_gpu_dg2_g10" CACHE STRING "comma-separated -fsycl-targets")

find_package(PkgConfig REQUIRED)
pkg_check_modules(LEVEL_ZERO REQUIRED IMPORTED_TARGET level-zero)

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Host-only pieces: no device code, no -fsycl.
add_library(eklow_host STATIC
  src/core/json.cpp
  src/model/gguf.cpp
)
target_include_directories(eklow_host PUBLIC src)

# SYCL targets compile and link with the same -fsycl flags; the link step runs the device
# compiler for the AOT targets.
function(eklow_sycl target)
  target_compile_options(${target} PRIVATE -fsycl -fsycl-targets=${EKLOW_SYCL_TARGETS})
  target_link_options(${target} PRIVATE -fsycl -fsycl-targets=${EKLOW_SYCL_TARGETS})
endfunction()

# Serving text layer: the tokenizer (and, later, the chat template and output parser). No device code.
add_library(eklow_text STATIC
  src/text/tokenizer.cpp
)
target_link_libraries(eklow_text PUBLIC eklow_host)

add_executable(eklow-tok tools/tok/main.cpp)
target_link_libraries(eklow-tok PRIVATE eklow_text)

add_executable(eklow-gguf tools/gguf/main.cpp)
target_link_libraries(eklow-gguf PRIVATE eklow_host)

# eklow-probe: what the card offers (memory, transfers, launch/sync and wake latency).
add_executable(eklow-probe tools/probe/probe.cpp)
target_link_libraries(eklow-probe PRIVATE eklow_host PkgConfig::LEVEL_ZERO)
eklow_sycl(eklow-probe)

# eklow-xmx-bench: dpas layouts checked against the CPU, XMX throughput, the Q4_0 inner loops and
# VRAM read bandwidth.
add_executable(eklow-xmx-bench tools/xmx_bench/main.cpp)
target_link_libraries(eklow-xmx-bench PRIVATE eklow_host)
eklow_sycl(eklow-xmx-bench)

install(TARGETS eklow-gguf eklow-tok eklow-probe eklow-xmx-bench RUNTIME DESTINATION bin)
