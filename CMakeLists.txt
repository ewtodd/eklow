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
# cpp-httplib is header-only; its CMake package would also want OpenSSL, which eklow does not use.
find_path(HTTPLIB_INCLUDE_DIR httplib.h REQUIRED)
find_package(Threads REQUIRED)

add_compile_options(-Wall -Wextra -Wno-unused-parameter)

# Host-only pieces: no device code, no -fsycl.
add_library(eklow_host STATIC
  src/core/json.cpp
  src/core/json_value.cpp
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
  src/text/chat_qwen36.cpp
  src/text/pyfmt.cpp
  src/text/tokenizer.cpp
  src/text/output_parser_qwen.cpp
  src/text/tokenizer_qwen.cpp
  src/text/tool_schema.cpp
  src/text/unicode.cpp
)
target_link_libraries(eklow_text PUBLIC eklow_host)

# Images for the vision encoders (EmbeddingGemma 2's tower, Qwen3-VL's): decoding (stb_image, vendored) and preparation. No device code.
add_library(eklow_image STATIC
  src/image/gemma4_image.cpp
  src/image/qwen3vl_image.cpp
  src/image/stb_image.cpp
)
target_include_directories(eklow_image PUBLIC src third_party)
set_source_files_properties(src/image/stb_image.cpp PROPERTIES COMPILE_OPTIONS "-w")

add_executable(eklow-tok tools/tok/main.cpp)
target_link_libraries(eklow-tok PRIVATE eklow_text)

add_executable(eklow-chat tools/chat/main.cpp)
target_link_libraries(eklow-chat PRIVATE eklow_text)

add_executable(eklow-parse tools/parse/main.cpp)
target_link_libraries(eklow-parse PRIVATE eklow_text)

# Device runtime: picking the card. SYCL plus Level Zero.
add_library(eklow_rt STATIC src/rt/device.cpp src/rt/copy_engine.cpp)
target_include_directories(eklow_rt PUBLIC src)
target_link_libraries(eklow_rt PUBLIC PkgConfig::LEVEL_ZERO)
eklow_sycl(eklow_rt)

# The model: configuration, weights on the card, the kernels and the reference forward.
add_library(eklow_model STATIC
  src/kernels/attention.cpp
  src/kernels/fast.cpp
  src/kernels/gdn.cpp
  src/kernels/qwen_fast.cpp
  src/kernels/ref.cpp
  src/kernels/xmx.cpp
  src/kernels/xmx_fmt.cpp
  src/kernels/xmx_gemv_iq4xs.cpp
  src/kernels/xmx_gemv_q4k.cpp
  src/kernels/xmx_gemv_q5k.cpp
  src/kernels/xmx_gemv_q6k.cpp
  src/kernels/xmx_gemv_q8.cpp
  src/model/embed_gemma2.cpp
  src/model/vision_gemma4v.cpp
  src/model/vision_gemma4v_fast.cpp
  src/model/vision_qwen3vl.cpp
  src/model/vision_qwen3vl_fast.cpp
  src/kernels/vision_attn.cpp
  src/model/qwen35.cpp
)
target_link_libraries(eklow_model PUBLIC eklow_host eklow_image eklow_rt)
eklow_sycl(eklow_model)

# The Qwen weight formats' kernels, checked against ggml's dequantization and timed (TASK qwen/mm).
add_executable(eklow-qfmt-test tools/qfmt_test/main.cpp)
target_link_libraries(eklow-qfmt-test PRIVATE eklow_model)
eklow_sycl(eklow-qfmt-test)

add_executable(eklow-embed tools/embed/main.cpp)
target_link_libraries(eklow-embed PRIVATE eklow_model eklow_text)
eklow_sycl(eklow-embed)

add_executable(eklow-vision tools/vision/main.cpp)
target_link_libraries(eklow-vision PRIVATE eklow_model)
eklow_sycl(eklow-vision)

add_executable(eklow-vision-qwen tools/vision/qwen_main.cpp)
target_link_libraries(eklow-vision-qwen PRIVATE eklow_model)
eklow_sycl(eklow-vision-qwen)

add_executable(eklow-attn-test tools/attn_test/main.cpp)
target_link_libraries(eklow-attn-test PRIVATE eklow_model)
eklow_sycl(eklow-attn-test)

add_executable(eklow-gdn-test tools/gdn_test/main.cpp)
target_link_libraries(eklow-gdn-test PRIVATE eklow_model)
eklow_sycl(eklow-gdn-test)

add_executable(eklow-dequant-test tools/dequant_test/main.cpp)
target_link_libraries(eklow-dequant-test PRIVATE eklow_model)
eklow_sycl(eklow-dequant-test)

add_executable(eklow-ref-run tools/ref_run/main.cpp)
target_link_libraries(eklow-ref-run PRIVATE eklow_model)
eklow_sycl(eklow-ref-run)

# The serving forward pass: KV cache, batched steps over several sequences.
add_library(eklow_engine STATIC src/engine/engine.cpp)
target_link_libraries(eklow_engine PUBLIC eklow_model eklow_cpu)
eklow_sycl(eklow_engine)

add_executable(eklow-engine-run tools/engine_run/main.cpp)
target_link_libraries(eklow-engine-run PRIVATE eklow_engine)
eklow_sycl(eklow-engine-run)

add_executable(eklow-engine-bench tools/engine_bench/main.cpp)
target_link_libraries(eklow-engine-bench PRIVATE eklow_engine)
eklow_sycl(eklow-engine-bench)

# The tool-call grammar: host code over the vocabulary, tested against the output parser.
add_library(eklow_grammar STATIC src/serve/tool_grammar_qwen.cpp)
target_link_libraries(eklow_grammar PUBLIC eklow_text)

add_executable(eklow-grammar-test-qwen tools/grammar_test/main.cpp)
target_link_libraries(eklow-grammar-test-qwen PRIVATE eklow_grammar)

# Serving: the request layer (OpenAI surface, Qwen3.6's chat template and tokenizer), the scheduler (continuous batching,
# speculative decoding with a DFlash 2 draft, snapshots of the recurrent state) and the server. The sampler and the
# acceptance are eklow_spec's.
add_library(eklow_serve STATIC src/serve/openai.cpp src/serve/scheduler.cpp)
target_link_libraries(eklow_serve PUBLIC eklow_spec eklow_grammar eklow_text eklow_engine)
eklow_sycl(eklow_serve)

add_executable(eklow-server tools/server/main.cpp)
target_include_directories(eklow-server PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(eklow-server PRIVATE eklow_serve Threads::Threads)
eklow_sycl(eklow-server)
# The streams' keep-alive (tools/server/keepalive.hpp), host only.
add_executable(eklow-keepalive-test tools/server/keepalive_test.cpp)
target_include_directories(eklow-keepalive-test PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(eklow-keepalive-test PRIVATE Threads::Threads)

add_executable(eklow-prompt tools/prompt/main.cpp)
target_link_libraries(eklow-prompt PRIVATE eklow_serve)
eklow_sycl(eklow-prompt)
# parse_chat with --verification on (Sage's template), host only: eklow-verification-test VOCAB_GGUF.
add_executable(eklow-verification-test tools/prompt/verification_test.cpp)
target_link_libraries(eklow-verification-test PRIVATE eklow_serve)
eklow_sycl(eklow-verification-test)
add_executable(eklow-gguf tools/gguf/main.cpp)
target_link_libraries(eklow-gguf PRIVATE eklow_host)

# CPU GEMVs for the cold experts (TASK: m6/cpu-experts): Q4_0 in q4_gemv.cpp, the Qwen formats in qgemv.cpp. Plain host
# code, AVX-VNNI, so the flags are per source file and there is no -fsycl.
add_library(eklow_cpu STATIC src/cpu/q4_gemv.cpp src/cpu/qgemv.cpp)
target_include_directories(eklow_cpu PUBLIC src)
target_compile_options(eklow_cpu PRIVATE -O3 -mavx2 -mfma -mf16c -mavxvnni)
target_link_libraries(eklow_cpu PUBLIC Threads::Threads)

# eklow-probe: what the card offers (memory, transfers, launch/sync and wake latency).
add_executable(eklow-probe tools/probe/probe.cpp)
target_link_libraries(eklow-probe PRIVATE eklow_host eklow_rt)
eklow_sycl(eklow-probe)

# eklow-xmx-bench: dpas layouts checked against the CPU, XMX throughput, the Q4_0 inner loops and
add_executable(eklow-copy-bench tools/copy_bench/main.cpp)
add_executable(eklow-tax tools/copy_bench/tax.cpp)
target_link_libraries(eklow-tax PRIVATE eklow_model)
eklow_sycl(eklow-tax)
target_link_libraries(eklow-copy-bench PRIVATE eklow_model)
eklow_sycl(eklow-copy-bench)

# VRAM read bandwidth.
add_executable(eklow-xmx-bench tools/xmx_bench/main.cpp)
target_link_libraries(eklow-xmx-bench PRIVATE eklow_host eklow_rt)
eklow_sycl(eklow-xmx-bench)

install(TARGETS eklow-qfmt-test eklow-gguf eklow-tok eklow-chat eklow-parse eklow-probe eklow-xmx-bench eklow-ref-run eklow-attn-test eklow-gdn-test eklow-embed eklow-vision eklow-vision-qwen eklow-engine-run eklow-engine-bench eklow-server eklow-prompt eklow-grammar-test-qwen RUNTIME DESTINATION bin)
# The Qwen finetunes' routing profiles, for eklow-server --expert-profile (which experts stay on the card).
install(FILES data/qwen36-experts.json data/ornith-1.5-experts.json DESTINATION share/eklow)

# The CPU GEMVs of the Qwen formats for the cold experts, checked and timed (TASK qwen/mm).
add_executable(eklow-cpu-qgemv-test tools/cpu_qgemv_test/main.cpp)
target_compile_options(eklow-cpu-qgemv-test PRIVATE -O2 -mavx2 -mf16c)
target_link_libraries(eklow-cpu-qgemv-test PRIVATE eklow_cpu eklow_host)
install(TARGETS eklow-cpu-qgemv-test RUNTIME DESTINATION bin)

# DFlash 2 drafting and speculative decoding (TASK qwen/dflash): the draft's kernels and model, the acceptance of a
# drafted block, and their checks. Appended here so as not to touch the lists above.
target_sources(eklow_model PRIVATE src/kernels/ref_dflash.cpp src/kernels/dflash.cpp src/model/dflash2.cpp src/model/dflash2_fast.cpp)
# The sampler and the acceptance of a drafted block (eklow_serve links them).
add_library(eklow_spec STATIC src/serve/sampler.cpp src/serve/spec.cpp)
target_link_libraries(eklow_spec PUBLIC eklow_host)

# The acceptance's statistical test: host code over the sampler only.
add_executable(eklow-spec-test tools/dflash_test/spec_test.cpp)
target_link_libraries(eklow-spec-test PRIVATE eklow_spec)
target_include_directories(eklow-spec-test PRIVATE src)
install(TARGETS eklow-spec-test RUNTIME DESTINATION bin)

add_executable(eklow-dflash-test tools/dflash_test/main.cpp)
target_link_libraries(eklow-dflash-test PRIVATE eklow_model eklow_spec)
eklow_sycl(eklow-dflash-test)
install(TARGETS eklow-dflash-test RUNTIME DESTINATION bin)

# The scheduler's K/V pool decisions (src/serve/kv_pool.hpp), host only.
add_executable(eklow-kvpool-test tools/server/kv_pool_test.cpp)
target_include_directories(eklow-kvpool-test PRIVATE src)

# The /metrics rendering (tools/server/metrics.hpp), host only (the scheduler's header pulls in the engine's, so it builds like the server).
add_executable(eklow-metrics-test tools/server/metrics_test.cpp)
target_link_libraries(eklow-metrics-test PRIVATE eklow_serve)
eklow_sycl(eklow-metrics-test)
