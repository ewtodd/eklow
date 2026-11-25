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
# cpp-httplib is header-only; its CMake package would also want OpenSSL for TLS, which eklow does not serve.
find_path(HTTPLIB_INCLUDE_DIR httplib.h REQUIRED)
find_package(Threads REQUIRED)
# OpenSSL's libcrypto: SHA-256 and AES-256-GCM for the disk prefix cache (src/serve/prefix_store.*).
find_package(OpenSSL REQUIRED COMPONENTS Crypto)

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

# Serving text layer: Gemma 4's tokenizer, chat template and output parser; Qwen's tokenizer (vega's reader and the reranker). No device code.
add_library(eklow_text STATIC
  src/text/chat_gemma4.cpp
  src/text/chat_inkling.cpp
  src/text/pyfmt.cpp
  src/text/tokenizer.cpp
  src/text/output_parser_gemma4.cpp
  src/text/output_parser_inkling.cpp
  src/text/tokenizer_inkling.cpp
  src/text/tokenizer_qwen.cpp
  src/text/unicode.cpp
)
target_link_libraries(eklow_text PUBLIC eklow_host)

# Images for the Gemma vision towers (Bolt's and EmbeddingGemma 2's): decoding (stb_image, vendored) and preparation. No device code.
add_library(eklow_image STATIC
  src/image/gemma4_image.cpp
  src/image/stb_image.cpp
)
target_include_directories(eklow_image PUBLIC src third_party)
set_source_files_properties(src/image/stb_image.cpp PROPERTIES COMPILE_OPTIONS "-w")

add_executable(eklow-image-prep tools/vision/prep_main.cpp)
target_link_libraries(eklow-image-prep PRIVATE eklow_image)

add_executable(eklow-tok tools/tok/main.cpp)
target_link_libraries(eklow-tok PRIVATE eklow_text)

# Device runtime: picking the card. SYCL plus Level Zero.
add_library(eklow_rt STATIC src/rt/device.cpp src/rt/copy_engine.cpp)
target_include_directories(eklow_rt PUBLIC src)
target_link_libraries(eklow_rt PUBLIC PkgConfig::LEVEL_ZERO)
eklow_sycl(eklow_rt)

# The model: configuration, weights on the card, the kernels and the reference forward.
add_library(eklow_model STATIC
  src/kernels/attention.cpp
  src/kernels/fast.cpp
  src/kernels/dense.cpp
  src/kernels/gdn_seq.cpp
  src/kernels/gemma_fast.cpp
  src/kernels/qwen_fast.cpp
  src/kernels/ref.cpp
  src/kernels/moe.cpp
  src/kernels/ref_inkling.cpp
  src/kernels/inkling_fast.cpp
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
  src/kernels/vision_attn.cpp
  src/model/gemma4.cpp
  src/model/inkling2.cpp
)
target_link_libraries(eklow_model PUBLIC eklow_host eklow_image eklow_rt)
eklow_sycl(eklow_model)

# The weight formats' kernels on the matrix units, checked against ggml's dequantization and timed.
add_executable(eklow-qfmt-test tools/qfmt_test/main.cpp)
target_link_libraries(eklow-qfmt-test PRIVATE eklow_model)
eklow_sycl(eklow-qfmt-test)

add_executable(eklow-embed tools/embed/main.cpp)
target_link_libraries(eklow-embed PRIVATE eklow_model eklow_text)
eklow_sycl(eklow-embed)

add_executable(eklow-vision tools/vision/main.cpp)
target_link_libraries(eklow-vision PRIVATE eklow_model)
eklow_sycl(eklow-vision)

add_executable(eklow-attn-test tools/attn_test/main.cpp tools/attn_test/inkling.cpp)
target_link_libraries(eklow-attn-test PRIVATE eklow_model)
eklow_sycl(eklow-attn-test)

add_executable(eklow-dequant-test tools/dequant_test/main.cpp)
target_link_libraries(eklow-dequant-test PRIVATE eklow_model)
eklow_sycl(eklow-dequant-test)

# Inkling 2 Mini's serving engine (M2, DECISIONS 2026-11-24) and its driver over M0's prompts.
add_library(eklow_inkling STATIC src/engine/engine_inkling.cpp)
target_link_libraries(eklow_inkling PUBLIC eklow_model eklow_cpu)
eklow_sycl(eklow_inkling)
add_executable(eklow-engine-run-inkling tools/engine_run_inkling/main.cpp)
target_link_libraries(eklow-engine-run-inkling PRIVATE eklow_inkling)
eklow_sycl(eklow-engine-run-inkling)

add_executable(eklow-ref-run tools/ref_run/main.cpp)
target_link_libraries(eklow-ref-run PRIVATE eklow_model)
eklow_sycl(eklow-ref-run)

# vega's checker (laya's System One): its reader, Qwen3.5-0.8B, on the card; its engine and laya's policies on the CPU.
add_library(eklow_vega STATIC
  src/vega/checker.cpp
  src/vega/engine.cpp
  src/vega/files.cpp
  src/vega/policy.cpp
  src/vega/prompt.cpp
  src/vega/reader.cpp
)
target_link_libraries(eklow_vega PUBLIC eklow_model eklow_text)
eklow_sycl(eklow_vega)

# Qwen3-Reranker: llama.cpp's rerank endpoint over eklow's kernels.
add_library(eklow_rerank STATIC src/model/rerank_qwen3.cpp)
target_link_libraries(eklow_rerank PUBLIC eklow_model eklow_text)
eklow_sycl(eklow_rerank)

add_executable(eklow-rerank tools/rerank/main.cpp)
target_include_directories(eklow-rerank PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(eklow-rerank PRIVATE eklow_rerank Threads::Threads)
eklow_sycl(eklow-rerank)

add_executable(eklow-vega tools/vega/main.cpp)
target_include_directories(eklow-vega PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(eklow-vega PRIVATE eklow_vega Threads::Threads)
eklow_sycl(eklow-vega)

# Gemma 4 E4B (Nemotron-3.5-Bolt-E4B)'s serving forward pass: paged K/V, batched steps over several sequences, verify blocks.
# Its engine and tools keep the *_gemma4 names and the gemma4 inner namespace of when Qwen3.6's stood beside them.
add_library(eklow_engine_gemma4 STATIC src/engine/engine_gemma4.cpp)
target_link_libraries(eklow_engine_gemma4 PUBLIC eklow_model)
eklow_sycl(eklow_engine_gemma4)

add_executable(eklow-engine-run-gemma4 tools/engine_run_gemma4/main.cpp)
target_link_libraries(eklow-engine-run-gemma4 PRIVATE eklow_engine_gemma4)
eklow_sycl(eklow-engine-run-gemma4)

add_executable(eklow-engine-bench-gemma4 tools/engine_bench_gemma4/main.cpp)
target_link_libraries(eklow-engine-bench-gemma4 PRIVATE eklow_engine_gemma4)
eklow_sycl(eklow-engine-bench-gemma4)

# The tool-call grammar: host code over the vocabulary, tested against the output parser.
add_library(eklow_grammar_gemma4 STATIC src/serve/tool_grammar_gemma4.cpp)
target_link_libraries(eklow_grammar_gemma4 PUBLIC eklow_text)

add_executable(eklow-grammar-test-gemma4 tools/grammar_test_gemma4/main.cpp)
target_link_libraries(eklow-grammar-test-gemma4 PRIVATE eklow_grammar_gemma4)

# Inkling 2 Mini's tool-call grammar: llama.cpp's grammar sampler on the grammar its Inkling parser builds, token by token.
add_library(eklow_grammar_inkling STATIC src/serve/tool_grammar_inkling.cpp)
target_link_libraries(eklow_grammar_inkling PUBLIC eklow_text)

add_executable(eklow-grammar-test-inkling tools/grammar_test_inkling/main.cpp)
target_link_libraries(eklow-grammar-test-inkling PRIVATE eklow_grammar_inkling)

add_executable(eklow-chat-inkling tools/chat_inkling/main.cpp)
target_link_libraries(eklow-chat-inkling PRIVATE eklow_text)

add_executable(eklow-chat-gemma4 tools/chat_gemma4/main.cpp)
target_link_libraries(eklow-chat-gemma4 PRIVATE eklow_text)

add_executable(eklow-parse-inkling tools/parse_inkling/main.cpp)
target_link_libraries(eklow-parse-inkling PRIVATE eklow_text)

add_executable(eklow-parse-gemma4 tools/parse_gemma4/main.cpp)
target_link_libraries(eklow-parse-gemma4 PRIVATE eklow_text)

# Serving: the request layer (OpenAI surface, Gemma 4's chat template), the scheduler (continuous batching, speculative
# decoding with a DFlash draft) and the server. The sampler and the acceptance are eklow_spec's.
# The disk prefix cache (keys, sealed page files, eviction) is host only, so its test builds without the card's toolchain.
add_library(eklow_prefix STATIC src/serve/prefix_store.cpp)
target_include_directories(eklow_prefix PUBLIC src)
target_link_libraries(eklow_prefix PUBLIC OpenSSL::Crypto Threads::Threads)
add_executable(eklow-prefix-store-test tools/server/prefix_store_test.cpp)
target_link_libraries(eklow-prefix-store-test PRIVATE eklow_prefix)
install(TARGETS eklow-prefix-store-test RUNTIME DESTINATION bin)

add_library(eklow_serve_gemma4 STATIC src/serve/openai_gemma4.cpp src/serve/scheduler_gemma4.cpp)
target_link_libraries(eklow_serve_gemma4 PUBLIC eklow_spec eklow_grammar_gemma4 eklow_text eklow_engine_gemma4 eklow_prefix)
eklow_sycl(eklow_serve_gemma4)

# Inkling 2 Mini's request layer and scheduler (M4); its responses are written by Bolt's ChatResponse.
add_library(eklow_serve_inkling STATIC src/serve/openai_inkling.cpp src/serve/scheduler_inkling.cpp)
target_link_libraries(eklow_serve_inkling PUBLIC eklow_serve_gemma4 eklow_grammar_inkling eklow_inkling)
eklow_sycl(eklow_serve_inkling)

add_executable(eklow-prompt-gemma4 tools/prompt_gemma4/main.cpp)
target_link_libraries(eklow-prompt-gemma4 PRIVATE eklow_serve_gemma4)
eklow_sycl(eklow-prompt-gemma4)

add_executable(eklow-server tools/server/main.cpp)
target_include_directories(eklow-server PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(eklow-server PRIVATE eklow_serve_gemma4 eklow_serve_inkling eklow_vega eklow_rerank Threads::Threads)
eklow_sycl(eklow-server)
# The streams' keep-alive (tools/server/keepalive.hpp), host only.
add_executable(eklow-keepalive-test tools/server/keepalive_test.cpp)
target_include_directories(eklow-keepalive-test PRIVATE ${HTTPLIB_INCLUDE_DIR})
target_link_libraries(eklow-keepalive-test PRIVATE Threads::Threads)

add_executable(eklow-gguf tools/gguf/main.cpp)
target_link_libraries(eklow-gguf PRIVATE eklow_host)

# eklow-probe: what the card offers (memory, transfers, launch/sync and wake latency).
add_executable(eklow-probe tools/probe/probe.cpp)
target_link_libraries(eklow-probe PRIVATE eklow_host eklow_rt)
eklow_sycl(eklow-probe)

# eklow-xmx-bench: dpas layouts checked against the CPU, XMX throughput, the Q4_0 inner loops and VRAM read bandwidth.
add_executable(eklow-xmx-bench tools/xmx_bench/main.cpp)
target_link_libraries(eklow-xmx-bench PRIVATE eklow_host eklow_rt)
eklow_sycl(eklow-xmx-bench)

install(TARGETS eklow-qfmt-test eklow-gguf eklow-tok eklow-probe eklow-xmx-bench eklow-ref-run eklow-attn-test eklow-embed eklow-vision eklow-image-prep eklow-engine-run-gemma4 eklow-engine-bench-gemma4 eklow-prompt-gemma4 eklow-server eklow-vega eklow-rerank RUNTIME DESTINATION bin)
install(TARGETS eklow-grammar-test-gemma4 eklow-chat-gemma4 eklow-parse-gemma4 RUNTIME DESTINATION bin)

# Inkling 2 Mini's experts on the card against the reference ops.
add_executable(eklow-moe-test tools/moe_test/main.cpp)
target_link_libraries(eklow-moe-test PRIVATE eklow_model)
eklow_sycl(eklow-moe-test)

# DFlash drafting and speculative decoding: the draft's kernels and model, the acceptance of a drafted block, and their
# checks.
target_sources(eklow_model PRIVATE src/kernels/ref_dflash.cpp src/kernels/dflash.cpp src/model/dflash2.cpp src/model/dflash2_fast.cpp)
# The sampler and the acceptance of a drafted block (eklow_serve_gemma4 links them).
add_library(eklow_spec STATIC src/serve/sampler.cpp src/serve/spec.cpp)
target_link_libraries(eklow_spec PUBLIC eklow_host)

# The acceptance's statistical test: host code over the sampler only.
# The sampler against llama.cpp's own (tools/sampler_test: llama_ref.cpp is built against a libllama by build-llama-ref.sh).
add_executable(eklow-sampler-test tools/sampler_test/main.cpp)
target_link_libraries(eklow-sampler-test PRIVATE eklow_spec)
target_include_directories(eklow-sampler-test PRIVATE src)

add_executable(eklow-spec-test tools/dflash_test/spec_test.cpp)
target_link_libraries(eklow-spec-test PRIVATE eklow_spec)
target_include_directories(eklow-spec-test PRIVATE src)
install(TARGETS eklow-spec-test RUNTIME DESTINATION bin)

add_executable(eklow-dflash-test tools/dflash_test/main.cpp)
target_link_libraries(eklow-dflash-test PRIVATE eklow_model eklow_spec)
eklow_sycl(eklow-dflash-test)
install(TARGETS eklow-dflash-test RUNTIME DESTINATION bin)

# The /metrics rendering (tools/server/metrics.hpp), host only (the scheduler's header pulls in the engine's, so it builds like the server).
add_executable(eklow-metrics-test tools/server/metrics_test.cpp)
target_link_libraries(eklow-metrics-test PRIVATE eklow_serve_gemma4)
eklow_sycl(eklow-metrics-test)

# The disk prefix cache and the rings' rewind on the card: logits against fresh prefills, a restore against a rewind in place,
# and the costs (tools/server/prefix_engine_test.cpp).
add_executable(eklow-prefix-engine-test tools/server/prefix_engine_test.cpp)
target_link_libraries(eklow-prefix-engine-test PRIVATE eklow_engine_gemma4 eklow_prefix eklow_text)
eklow_sycl(eklow-prefix-engine-test)

# Inkling 2 Mini's cold experts on the host CPU (M3): the pool, the Q8_0 / Q5_K / NVFP4 AVX-VNNI GEMVs and one expert's
# FFN (src/cpu). Plain host code, so the flags are per source file and there is no -fsycl.
add_library(eklow_cpu STATIC src/cpu/pool.cpp src/cpu/qgemv.cpp src/cpu/nvfp4.cpp src/cpu/expert.cpp)
target_include_directories(eklow_cpu PUBLIC src)
target_compile_options(eklow_cpu PRIVATE -O3 -mavx2 -mfma -mf16c -mavxvnni)
target_link_libraries(eklow_cpu PUBLIC Threads::Threads)

add_executable(eklow-cpu-expert-test tools/cpu_expert_test/main.cpp)
target_compile_options(eklow-cpu-expert-test PRIVATE -O2 -mavx2 -mfma -mf16c)
target_link_libraries(eklow-cpu-expert-test PRIVATE eklow_cpu eklow_host)
