# eklow

A from-scratch serving engine for **Gemma 4 26B A4B** (and the dense **Gemma 4 12B**) on a single Intel Arc
A770M, a 16 GB laptop GPU, in a machine with 32 GB of RAM. Both run as Google's QAT Q4_0 checkpoints, with
**EmbeddingGemma 2** beside them in the same process. It serves an OpenAI-compatible API with continuous
batching, prefix reuse, thinking, grammar-constrained tool calls, and two slots of Gemma 4's full 262,144-token
context.

C++20 and SYCL/ESIMD on Level Zero; no Python at runtime; every kernel is our own (no oneDNN, no oneMKL, no
llama.cpp). It builds with Nix against the fleet's nixpkgs (intel-llvm, compute-runtime).

It is *vertical*: two models, one card. It was built in four days by Claude Opus 5.5, with an engineering log
(`DECISIONS.md`) that records every measurement and dead end along the way.

## The machine

```
                 i7-12700H, 32 GB DDR5 (two channels)
          ┌─────────────────────────────────────────────┐
          │  CPU: tokenizer, scheduler, HTTP server;      │
          │    the 26B's cold experts at decode           │
          │    (AVX-VNNI, 16 threads; 7.2 GiB of them     │
          │    in pinned memory)                          │
          │  Iris Xe iGPU: not eklow's (llama.cpp titles) │
          └──────────────┬──────────────────────────────┘
                         │ PCIe 4.0 x8, ~11 GB/s
          ┌──────────────┴──────────────┐
          │ Arc A770M (DG2), 16 GB      │  32 Xe-cores of XMX: fp16 98.8 TFLOPS, int8 198 TOPS
          │ dense weights, KV cache,    │
          │ 51 hot experts a layer,     │
          │ 40 streaming slots,         │
          │ EmbeddingGemma 2            │
          └─────────────────────────────┘
```

Gemma 4 26B A4B (30 layers: 25 sliding-window layers with a 1,024-token window and 5 global ones with 2 K/V heads
of 512 dimensions; per layer a dense GeGLU of 2,112 beside 128 routed experts of 704, 8 active per token) is 14.4 GB
on disk, 12 GiB of it experts. With the global layers' KV at 6.4 GiB for 2 x 262k, the card holds the dense weights,
the caches and each layer's 51 most-used ("hot") experts, chosen by a routing profile
(`data/gemma-4-26b-a4b-experts.json`) and sized to the memory the driver reports free. The other 77 experts of each
layer are computed by the CPU at decode while the card computes the hot ones; for prefill, the busiest of them in each
chunk are copied to the card on its copy engine and run in the same grouped GEMMs, and the CPU takes the rest.
Gemma 4 12B (48 layers, dense) is 7.0 GB and fits the card whole.

## Performance

Measured on that machine (October 2026), through `eklow-server` as fiona runs it in production: the 26B with two
slots of 262,144 tokens, int8 global KV, 51 of 128 experts a layer on the card and EmbeddingGemma 2 loaded. The
method is unholy's (`tools/bench/serve_perf.py`, on unholy's unique Pile documents cut to Gemma token depths;
`docs/bench/serve-perf-26b-fiona-2026-10-27.txt`):

| | Gemma 4 26B A4B |
|---|---|
| prefill, one request, unique text (time to first token) | 1578 tok/s at 8k, 1257 at 32k, 493 at 128k, 352 at 252k (734 s) |
| decode, one stream | 37.6 tok/s at 8k, 34.9 at 32k, 30.9 at 128k, 26.7 at 252k |
| GSM8K chat (48 requests, thinking on), 1 / 2 at a time | 33.4 / 41.9 tok/s aggregate |

A question on a document already in the prefix cache starts in 0.2-0.7 s. At depth the global layers' attention is
what costs: it runs at ~11 TFLOPS of the card's 98.8 and grows with every token of context (DECISIONS 2026-10-27,
"global prefill attention: where its time goes"). Two streams at once add 25% rather than doubling; most likely the CPU,
which then computes both streams' cold experts, is what holds them.

With every expert on the card (contexts up to about 2 x 32k), `eklow-engine-bench` (one sequence, 1,024-token
chunks) gives prefill 1,234 tok/s at 32k depth (global attention 61% of it) and decode 68.1 tok/s at 0, 61.4 at 4k,
43.4 at 32k.

Embeddings and titles do not wait for it: an embedding takes 60 ms idle and 78 ms during a 32k-token prefill, and
llama.cpp's title model on the Iris Xe 368 ms idle and 457-471 ms while the 26B prefills or decodes.

Gemma 4 12B, the production model until 2026-10-27, on the same server at 2 x 262k (measured 2026-10-25/26, before
the last decode work): a 10.6k-token son-of-anton prompt reached its first token in 8.7 s (0.2 s with the prefix
cached), then decoded 27.4 tok/s; two 259k-token prompts at once prefilled in 36 min, then decoded ~11 tok/s each.
Before eklow, llama.cpp's Vulkan backend on this card gave the 12B ~110-150 tok/s of prefill at 16k depth and ~22
tok/s of decode, with 2 x 32k contexts.

Production settings: two slots of 262,144 tokens, 1,024-token prefill chunks, the profile from four recordings, a 1 GiB
card margin; the server prints the placement at startup (6.24 GiB of weights and 6.41 GiB of KV on the card, 51
experts a layer, 40 streaming slots, 7.2 GiB of cold experts in pinned host memory).

## Accuracy

Correctness is checked logit by logit, on a 2,616-token prompt and a chat turn, against two
references: llama.cpp's CPU backend (`eklow-oracle`, built from llama.cpp at the revision the reference follows) and
eklow's own reference forward pass (`eklow-ref-run`: plain SYCL kernels, fp32 activations, weights dequantized as
ggml stores them).

- **12B against llama.cpp.** Top-1 agreement 31/32 on the long prompt's last 32 positions at a mean KL of 1.3e-2,
  and 24/25 at 5.2e-3 on the chat turn, within a band set from llama.cpp's own CUDA-against-CPU noise.
- **26B against the fp32 reference.** The 26B's router turns tiny differences into different experts wherever
  two scores nearly tie, so any two implementations drift apart; llama.cpp's CPU and CUDA backends agree with each
  other only because both round activations to int8 the same way. Over the long prompt's last 256 positions the
  production engine (51 experts a layer on the card) is at 243/256 top-1 and a mean KL of 2.9e-2, and 242/256 at
  2.8e-2 with every expert on the card. llama.cpp's CPU backend scores 220/256 at 7.5e-2 against the same
  reference (DECISIONS 2026-10-26, "the 26B in the engine").

Every fast kernel has a reference kernel next to it (`src/kernels/ref.*`), and its test compares against that and
an exact CPU emulation of the quantized arithmetic (`eklow-kernel-test`, `eklow-attn-test`, `eklow-cpu-gemv-test`).

## Features

- **OpenAI-compatible server** (`eklow-server`):
  - endpoints: `/v1/chat/completions` (streamed or whole), `/v1/models`, `/health`; with an embedding model,
    `/v1/embeddings`, `/v1/models` and `/health` on a port of its own;
  - request features: SSE streaming, `reasoning_content` (thinking on unless `reasoning_effort` is `none`), tool
    calls and `tool_choice`, stop strings, seeds, temperature / top-k / top-p / min-p and penalties, usage with
    `cached_tokens`.
- **Tool calls that cannot be malformed**: a grammar over Gemma 4's tool-call syntax constrains sampling while a
  call is open (`src/serve/tool_grammar.cpp`, checked against the output parser by `eklow-grammar-test`).
- **The chat format and tokenizer, matching the references exactly**: Google's chat template ported to C++ and
  byte-identical to Hugging Face's rendering (a GGUF carrying any other template is refused); a tokenizer identical
  to Hugging Face's; a streaming parser for the thinking channel and tool calls.
- **Scheduling**: continuous batching of two slots with chunked prefill (1,024 tokens a sequence a step), decodes
  first, the rest of the step filled with prefill; a paged cache for the global layers and a ring per sequence for
  the sliding ones; the global layers' K/V in int8 with a scale per 16 dimensions (what lets 2 x 262k fit); prefix
  reuse, a new request taking the slot whose cached sequence shares the longest prefix with its prompt.
- **Embeddings beside chat**: EmbeddingGemma 2 on its own queue on the card, so an embedding does not wait behind a
  long prefill.
- **Mixture of experts on 16 GB** (the 26B): routing, grouping and the experts' GEMVs and GEMMs on the card with no
  host round trip when every expert fits; a placement by routing profile when they do not, with cold experts on the
  CPU (AVX-VNNI) at decode and streamed over PCIe for prefill.

## Building

You need an x86-64 Linux machine with Nix (flakes enabled) and an Intel Arc DG2 card.

```sh
nix build github:ewtodd/eklow          # result/bin/eklow-server and the tools, AOT for DG2 (intel_gpu_dg2_g10)
```

For development:

```sh
nix develop -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'
```

Installed binaries use the system's Level Zero driver from `/run/opengl-driver` (on NixOS,
`hardware.graphics.extraPackages = [ pkgs.intel-compute-runtime ]`). The dev shell carries its own copy of the
drivers, so it runs on any host.

## Running

```sh
eklow-server --model gemma-4-26b-a4b-it-qat-q4_0.gguf --alias gemma-4-26b-a4b \
  --slots 2 --ctx 262144 --kv-int8 --expert-profile data/gemma-4-26b-a4b-experts.json \
  --embed-model embeddinggemma-2-Q8_0.gguf --embed-port 8082 --host 0.0.0.0 --port 8081
```

```sh
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-26b-a4b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": true
}'
```

Loading takes 4-7 s once the files are in the page cache. The server prints what it placed where: the weights
and KV on the card, how many experts a layer stayed there, the streaming slots and the pinned memory. The package
installs the profile as `share/eklow/gemma-4-26b-a4b-experts.json`, and the flake's NixOS module runs the server as
`services.eklow` (`model`, `alias`, `contextSize`, `slots`, `kvInt8`, the embedding model, `extraArgs`). The comment
at the top of `tools/server/main.cpp` explains every flag.

`eklow-probe` measures what the card actually does (allocatable memory, copy rates, launch and host-signal latency),
and `eklow-copy-bench` what PCIe copies cost the kernels beside them: the first things to run on new hardware.

## Hardware assumptions

This is not a general-purpose engine:
- One Intel Arc DG2 card (the A770M; the kernels are tuned for its 32 Xe-cores, 128 registers a thread and 64 KB of
  shared memory a work-group), through Level Zero.
- Its 16 GB set the shapes: the 26B at 2 x 262k needs its cold experts in about 7.2 GiB of pinned host memory and a
  CPU with AVX-VNNI to compute them.
- Card memory is sized from the free memory the driver reports (Level Zero Sysman), because allocations succeed past
  what the card holds and the first kernels then fail. Nothing else may use the card beside the server: a second
  program on it starves the running server into `UR_RESULT_ERROR_OUT_OF_RESOURCES` until it restarts.
- PCIe traffic slows every kernel on the card that streams memory (up to 5.8x at full copy rate), so streaming is
  rationed by a cost model (DECISIONS 2026-10-27).

## Repository

| | |
|---|---|
| `src/engine` | the serving forward pass: steps of prefill chunks and decodes, the paged cache, expert placement, cold experts |
| `src/kernels` | XMX matmuls for Q4_0 (int8 GEMVs for decode, fp16 GEMMs for prefill, grouped expert forms), flash attention over the paged cache, fused small kernels and routing (`fast`); `ref` holds the reference kernels |
| `src/cpu` | the AVX-VNNI Q4_0 expert GEMVs for the 26B's cold experts |
| `src/model` | GGUF loading, Gemma 4's configuration and reference forward pass, EmbeddingGemma 2 |
| `src/serve`, `src/text` | scheduler, OpenAI layer, sampler and tool-call grammar; tokenizer, chat template and output parser |
| `src/rt` | device selection and free memory, the copy engine |
| `tools/` | `eklow-server`, the engine and kernel benchmarks and tests, the reference runs (`tools/ref`), the server benchmark (`tools/bench`) |
| `data/` | the 26B's routing profile |
| `nix/` | the package, the NixOS module, and llama.cpp's CPU backend as the oracle |
| `DESIGN.md` | the target and the shape of the system |
| `DECISIONS.md` | dated decisions with their measurements: why each piece is the way it is |
| `STATUS.md` | what's done and what's open |

## Third-party code and credits

eklow links [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) for its HTTP server; it carries no other
code from other projects.

The model's math follows llama.cpp's gemma4 graph (MIT), which is also the oracle (`nix/oracle.nix` builds it); the
Q4_0 / Q6_K layouts are ggml's. The chat template is a port of Google's `chat_template.jinja` for Gemma 4; the
tokenizer follows Hugging Face's `tokenizer.json`. Gemma 4 and EmbeddingGemma 2's weights are Google's, under their
own license.

## License

Copyright 2026 Ethan Todd.
Licensed under the Apache License, Version 2.0 (`LICENSE`).
