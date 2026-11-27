# eklow

eklow is there to be useful: a small assistant you can rely on, with the tools it needs beside it on the same card. One
`eklow-server` process runs its models on a single Intel Arc A770M (a 16 GB laptop GPU):

| | what it is for | endpoint |
|---|---|---|
| **Inkling 2 Mini** | the assistant: Thinking Machines' looped mixture of experts, with thinking, tool calls and image input, and Inco's DFlash 2 draft for speculative decoding | `POST /v1/chat/completions` |
| **EmbeddingGemma 2** | embeddings of text and images, for retrieval | `POST /v1/embeddings` |
| **Qwen3-Reranker-0.6B** | reranking retrieved documents, as llama.cpp's server does it | `POST /v1/rerank` |
| **vega's checker** | laya's System One: yes/no checks of a draft reply before it is sent (off on fiona since 2026-11-24, its card memory went to Inkling's experts) | `POST /v1/systemone` |

Text-to-speech and speech-to-text are planned for the same card.

C++20 and SYCL/ESIMD on Level Zero. There is no Python at runtime, and every kernel is eklow's own (no oneDNN, no oneMKL,
no llama.cpp). It builds with Nix. It was built by Claude (Opus 5.5 and Sonnet 5.5) with an engineering log, `DECISIONS.md`,
that records every measurement and dead end.

Inkling 2 Mini has been fiona's chat model since 2026-11-25: on a sandboxed replay of real assistant turns it scored 110
of 126 where Nemotron-3.5-Bolt-E4B, the model before it, scored 16 of 48 (DECISIONS 2026-11-24). eklow still serves Bolt
(with NVIDIA's DFlash draft and the disk prefix cache); until 2026-11-11 it also served Qwen3.6-35B-A3B and its finetunes,
whose path is gone (DECISIONS 2026-11-11 names the commit that still has it).

## The machine

```
                 i7-12700H, 32 GB DDR4
          ┌─────────────────────────────────────────────┐
          │  CPU: Inkling's cold experts (8 threads,       │
          │    8.6 GiB of the file's mapping),             │
          │    tokenizers, scheduler, HTTP server          │
          │  Iris Xe iGPU: not eklow's                     │
          └──────────────┬──────────────────────────────┘
                         │ PCIe 4.0 x8
          ┌──────────────┴──────────────────────────────┐
          │ Arc A770M (DG2), 16 GB: 32 Xe-cores of XMX    │
          │   Inkling's dense weights and hot experts      │
          │     (6.43 GiB), its caches (2.63 GiB)          │
          │   the DFlash 2 draft (714 MiB)                 │
          │   the engine's step buffers and scratch        │
          │   EmbeddingGemma 2 (0.27 GiB, projector 269)   │
          │   the reranker (794 MiB)                       │
          │   Inkling's vision tower (1 MiB: its weights   │
          │     stay in host memory)                       │
          │   ~1 GiB free                                  │
          └──────────────────────────────────────────────┘
```

Inkling 2 Mini is 24 physical layers of hidden 2,304, with 128 routed experts (8 a token) in every layer. Layers 6-17 run
three times a token, each pass with its own K/V and convolution state, so a token visits 48 layers and reads the looped
core's experts three times. There is no RoPE: attention takes a learned relative-position bias and a sink, on a sliding
window of 1,024 in 40 of the 48 visits and over the whole context in 8. Short causal convolutions on K, V and both
residual branches are its only per-token recurrence. Unsloth's UD-Q4_K_M is 15.2 GiB, so about half of its experts stay
on the CPU: the busiest go on the card by a routing profile (`data/inkling-2-mini-experts.json`), the CPU computes the
rest in decode, and long prefill steps copy the busiest cold ones onto the card.

## How well

Measured on the A770M through the server, in production's shape: two slots sharing one K/V pool of 65,536 positions,
fp16 K/V, the side models loaded, the draft loaded in blocks of 6 (2026-11-26, `docs/bench/inkling-readme-2026-11-26`).
Prefill is the time to the first token on unique text; decode is one stream of 256 tokens.

| Inkling 2 Mini | |
|---|---|
| prefill at 8k / 32k / 60k | 1,320 / 1,316 / 1,197 tok/s (6.2 s to the first token for 8k from empty, 50 s for 60k) |
| decode, greedy, plain / with the DFlash 2 draft | ~24 tok/s at 12k plain; on short prompts ~22 plain, ~34 with the draft (3.7-4.3 tokens a verify) |
| decode at 8k / 32k / 60k, temperature 1, top-p 0.95, plain | 16.4 / 15.0 / 14.3 tok/s |
| the same with the draft | 16.7 / 14.6 / 13.7 tok/s |
| GSM8K chat requests, temperature 1, top-p 1, with / without the draft | 23.8-35.0 / 17.4-19.0 tok/s (3.4-4.3 tokens a verify) |
| an image (~250 tokens) | 0.1 s to encode (972 tokens: 0.27 s) |

Decode with top-p and no top-k is held by the host sampler today: its top-p stage sorts all 200,058 candidates, ~18 ms a
token, and once a row for each drafted token, which is why the draft gains nothing there. A sampler that does not sort the
whole vocabulary is the next change.

| the side models (measured beside Bolt, 2026-11-11; the same kernels) | idle |
|---|---|
| rerank, 8 documents (2,214 tokens) | 97 ms |
| rerank, 24 documents (6,640 tokens) | 263 ms |
| embedding, one sentence / 1,648 tokens | 62 / 122 ms |

A side request holds the chat scheduler while it runs, so a stream pauses for about as long and then continues.

**Accuracy.**
- **Inkling 2 Mini.** eklow's fp32 reference forward matches llama.cpp to 1e-7 a tensor when given llama.cpp's own streams
  and rounding. Free-running, the model is unusually sensitive to rounding (llama.cpp's own CPU and CUDA runs differ by a
  mean KL of 5e-2 to 0.46), and the serving engine sits inside that spread: on prompts of 12k and 33k tokens its last
  logits agree with llama.cpp CUDA's top-1 on 237 of 256 and 114 of 128 positions (median KL 5.6e-3 and 2.6e-3).
  - On the private replay through eklow-server, 40 of 41 episodes pass where llama.cpp passed 41 of 42 on the same
    moments.
  - The vision tower matches llama.cpp's mtmd to ~1e-3 a row, and the logits after an image to a KL of at most 4e-10.
- **EmbeddingGemma 2.** Cosine 0.99987-0.99995 against `google/embeddinggemma-2` (fp32) for text up to 6,989 tokens, and
  0.99958-0.99994 for images.
- **The reranker.** Within 0.0021 of llama.cpp with an f16 cache, and within 0.00036 of an fp32 run of the same weights.
- **The checker.** Within 0.0014 of laya with an fp32 reader on the replay set's 29 turns, and no flag differs at
  son-of-anton's cut-offs.

Speculative decoding is lossless in distribution. The acceptance is checked by chi-square against plain sampling on fixed
logits for 17 sampler configurations (`eklow-spec-test`). The engine's rollback is checked with oracle drafts (the plain
run's own tokens, cut at random places, garbage after the cut): every token comes out as plain decoding's. The draft's
acceptance matches llama.cpp's on the same requests (GSM8K, greedy: 4.9 tokens a verify against 5.2).

## Features

- **OpenAI-compatible chat** (`eklow-server`): `/v1/chat/completions` (SSE streaming or whole), `/v1/models`, `/health`,
  and `/metrics` (Prometheus text, for llmtop).
  - Requests take `reasoning_content`, tool calls and `tool_choice`, stop strings, `seed`, usage with `cached_tokens`, and
    image_url parts.
  - With the draft, the usage reports `speculation`, `"speculative": false` decodes a request plain, and a number sets
    its block size.
- **The text layer, ported rather than interpreted.** Inkling's chat template renders byte for byte as llama-server's on
  the 41 ground-truth cases and 10,602 random and hand-written requests; its tokenizer gives Hugging Face's ids on 9,246
  strings; the streaming parser separates reasoning, content and tool calls. Bolt's and Gemma 4's templates are ported
  the same way.
- **Tool calls that cannot be malformed.** A grammar over Inkling's tool-call syntax constrains sampling while a call is
  open. Inside a drafted block, each row is checked against the grammar state after the tokens before it.
- **Sampling as llama-server does it**:
  - its chain: penalties, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature, mirostat 1 and 2;
  - its defaults, under the GGUF's `general.sampling.*`, under the request's fields;
  - checked candidate for candidate against llama.cpp's own samplers (`tools/sampler_test`).
- **Speculative decoding** (`--draft`, `--draft-block`): Inco's DFlash 2 draft reads Inkling's stream at three layer
  visits and its final norm, writes a block in one pass, and the target checks it as one chunk; a rejected token's
  convolution state is rolled back from per-row snapshots. It stays lossless for every sampler feature and the tool
  grammar. Two generating slots verify in one step.
- **Experts between the card and the CPU.** The busiest experts by a routing profile live on the card; the CPU computes
  a step's cold pairs (AVX2 and AVX-VNNI, from the file's mapping, at fiona's memory bandwidth) while the card runs the
  hot ones, and long prefill steps copy the busiest cold sets onto the card.
- **Scheduling.** Continuous batching of two slots with chunked prefill (1,024 tokens a chunk, up to 8,192 a step),
  generating slots first. The slots share one K/V pool for the full-attention visits, and the sliding ones keep rings.
  Prefix reuse keeps a slot's K/V for the next turn; checkpoints of the rings and convolution states in host memory let a
  slot go back to an earlier point of its conversation.
- **Image input** (`--mmproj`). Pictures are prepared as llama.cpp's mtmd does and encoded by Inkling's 23-layer tower on
  the matrix units. Its weights stay in host memory and stream in a layer at a time, and it runs in the engine's idle step
  buffers, so it costs the card 1 MiB.
- **Embeddings** (`--embed-model`, `--embed-mmproj`). EmbeddingGemma 2 runs on its own queue and port. It takes text and
  image inputs.
- **The reranker** (`--rerank-*`). llama.cpp's `POST /v1/rerank` (also `/rerank`, `/v1/reranking`) over ggml-org's
  Qwen3-Reranker GGUF, scoring llama.cpp's P(yes). The pairs of a request are packed into passes.
- **The end-of-turn checker** (`--checker-*`). laya's System One over vega: Qwen3.5-0.8B (vega's reader) on the card with
  prompts built token for token as vegaml builds them, vega-U's engine and laya's trained policy on the CPU.
- **For Bolt:** a prefix cache on disk (`--prefix-cache`), the K/V of prompts and replies stored page by page under
  AES-256-GCM, so that a conversation whose slot went to another one is read back instead of prefilled.
- **A keep-alive on streams** (`--keepalive`, 15 s): a long tool call does not look like a dead connection.

## Building

You need an x86-64 Linux machine with Nix (flakes enabled) and an Intel Arc DG2 card.

```sh
nix build github:ewtodd/eklow          # result/bin/eklow-server and the tools, compiled ahead of time for DG2
```

For development:

```sh
nix develop -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'
```

Installed binaries use the system's Level Zero driver from `/run/opengl-driver` (on NixOS,
`hardware.graphics.extraPackages = [ pkgs.intel-compute-runtime ]`). The dev shell carries its own copy of the drivers.

The model files are flake packages too (`nix/models.nix`), fetched at pinned revisions: `inkling-2-mini` and
`inkling-2-mini-mmproj` (Unsloth's), `qwen3-reranker`, and Bolt's files. Inco's draft is not packaged yet: llama.cpp's
converter from 55d48281 on writes its GGUF from `incoai/Inkling-2-Mini-DFlash2`.

## Running

```sh
eklow-server --model Inkling-2-Mini-UD-Q4_K_M.gguf --alias inkling-2-mini --host 0.0.0.0 --port 8081 \
  --expert-profile data/inkling-2-mini-experts.json --mmproj mmproj-F16.gguf \
  --draft Inkling-2-Mini-DFlash2-Q8_0.gguf \
  --embed-model embeddinggemma-2-Q8_0.gguf --embed-mmproj mmproj-embeddinggemma-2-Q8_0.gguf --embed-port 8082 \
  --rerank-model qwen3-reranker-0.6b-q8_0.gguf --rerank-port 8085
```

```sh
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": true
}'
curl http://localhost:8085/v1/rerank -H 'Content-Type: application/json' -d '{
  "query": "When was the boiler serviced?", "documents": ["The boiler was serviced on 14 March.", "A cake recipe."]
}'
```

The defaults are the model's own: two slots of 65,536 positions sharing one pool, fp16 K/V, the GGUF's sampling, blocks
of 6 for the draft. The server prints what it put on the card, part by part, and the free memory left. It refuses a GGUF
of any other architecture before it touches the device. The comment at the top of `tools/server/main.cpp` explains every
flag.

## The NixOS module

The flake's module, `services.eklow`, serves named chat instances, each as the unit `eklow-<name>.service`.
- An instance's options are `model`, `alias`, `port`, `slots`, `contextSize`, `mmproj`, `imageMaxTokens`, `expertProfile`,
  `draft`, `draftBlock`, `preserveThinking` and `extraArgs`.
- The side models are shared by whichever instance runs:
  - `embedModel`, `embedMmproj`, `embedPort` and `imageTokens`;
  - `reranker.{model, port, alias, maxTokens}`;
  - `checker.{reader, vega, policy, policyFiles, port, alias, maxTokens}`.
- Instances exclude each other (one card), and `autoStart` names the one started at boot.
- `checker.policyFiles` takes a private laya policy as three root-only files (agenix's, say), which the unit copies into its
  runtime directory for its dynamic user.
- `prefixCache.{enable, keyFile, maxGiB}` keeps a Bolt instance's disk prefix cache in `/var/lib/eklow-<name>/prefix-cache`.

```nix
services.eklow = {
  enable = true;
  package = inputs.eklow.packages.${system}.default;
  instances.inkling = {
    model = inputs.eklow.packages.${system}.inkling-2-mini;
    mmproj = inputs.eklow.packages.${system}.inkling-2-mini-mmproj;
    expertProfile = "${inputs.eklow.packages.${system}.default}/share/eklow/inkling-2-mini-experts.json";
    alias = "inkling-2-mini";
    port = 8081;
  };
  reranker.model = inputs.eklow.packages.${system}.qwen3-reranker;
};
```

The log, on stdout (`journalctl -u eklow-<name>`), has:
- a line per chat request as it ends (prompt, cached and completion tokens, time to first token, prefill and decode rates,
  the draft's acceptance);
- a line per embedding, check and rerank request;
- every 10 s while there is traffic, the engine's rates, the slots and where a step's time goes.

`eklow-probe` measures what the card actually does: allocatable memory, copy rates, launch and host-signal latency.

## Hardware assumptions

This is not a general-purpose engine:
- One Intel Arc DG2 card, through Level Zero (for Inkling, the runtime's v2 adapter, which eklow-server selects: a
  dependent small kernel costs ~2 us there against ~4.5). The kernels are tuned for the A770M's 32 Xe-cores, 128 registers
  a thread and 64 KB of shared memory a work-group.
- For Inkling, host memory for its cold experts (~8.6 GiB of the file's mapping) and the slots' checkpoints (pinned, 324
  MiB each), and CPU cores to compute the cold experts.
- Card memory is sized from the free memory the driver reports (Level Zero Sysman), because allocations succeed past what
  the card holds and the first kernels then fail.
- Nothing else may use the card beside the server. A second program on it starves the running server into
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`, or slows it to a crawl, until it goes away.

## Repository

| | |
|---|---|
| `src/engine` | the serving forward passes: Inkling's (`engine_inkling`: the loop, hot and cold experts, the shared K/V pool and sliding rings, checkpoints, the draft's captures and verify rollback) and Bolt's (`engine_gemma4`) |
| `src/kernels` | XMX matmuls for the files' formats (NVFP4 and Q4_K experts, int8 GEMVs for decode, fp16 GEMMs for prefill), flash attention over the paged cache with Inkling's bias and sink, long prefill attention in key segments, the short convolutions and depth mixes, the drafts' kernels, fused small kernels; `ref` holds the reference kernels |
| `src/cpu` | the cold experts' FFN on the host CPU |
| `src/model` | GGUF loading, Inkling 2 Mini's and Gemma 4 E4B's configurations and reference forward passes, the DFlash drafts, the vision towers, EmbeddingGemma 2, Qwen3-Reranker |
| `src/vega` | vega's checker: the Qwen3.5-0.8B reader, vegaml's prompt and engine, laya's policies and System One |
| `src/serve`, `src/text` | the schedulers, OpenAI layers and tool-call grammars, the sampler and lossless acceptance, the disk prefix cache; the tokenizers, the chat templates and output parsers |
| `src/rt`, `src/image` | device selection and free memory, the copy engine; image decoding and preparation |
| `tools/` | `eklow-server` (`tools/server`), the engine, kernel and draft tests and benchmarks, the reference runs and comparisons (`tools/ref`), the checker's and the reranker's tools, the server tests and benchmarks (`tools/server`, `tools/bench`) |
| `nix/` | the package, the NixOS module, the model files (`models.nix`), and llama.cpp builds as the oracle |
| `data/` | Inkling's routing profile |
| `DESIGN.md` | the original target and the shape of the system |
| `DECISIONS.md` | dated decisions with their measurements: why each piece is the way it is |
| `STATUS.md` | what's done and what's open |

## Third-party code and credits

eklow links [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) for its HTTP server and OpenSSL's libcrypto
(Apache-2.0) for the disk prefix cache's SHA-256 and AES-256-GCM, and vendors stb_image (public domain / MIT) to decode
pictures; it carries no other code from other projects.

The models' math follows llama.cpp's graphs (MIT), and llama.cpp is also the oracle (`nix/oracle*.nix` build it); the
quantized layouts are ggml's. The chat templates are ports of Inkling 2 Mini's, Gemma 4's and Nemotron-3.5-Bolt-E4B's
`chat_template.jinja`. The tokenizers follow Hugging Face's `tokenizer.json`. The checker follows vegaml and laya; the
reranker follows llama.cpp's rank pooling.

The models' licenses are their own:
- Inkling 2 Mini is Thinking Machines' (Apache-2.0); its GGUFs are Unsloth's.
- Inkling-2-Mini-DFlash2 is Inco's (Apache-2.0).
- EmbeddingGemma 2 and Gemma 4 E4B's projector are Google's (Apache-2.0).
- Qwen3.5-0.8B and Qwen3-Reranker-0.6B are Alibaba's (Apache-2.0).
- Nemotron-3.5-Bolt-E4B and its DFlash draft are NVIDIA's (OpenMDW-1.1).
- vega-U is Apache-2.0.

## License

Copyright 2026 Ethan Todd.
Licensed under the Apache License, Version 2.0 (`LICENSE`).
