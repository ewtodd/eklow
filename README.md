# eklow

eklow is there to be useful: a small assistant you can rely on, with the tools it needs beside it on the same card. One
`eklow-server` process runs four models on a single Intel Arc A770M (a 16 GB laptop GPU):

| | what it is for | endpoint |
|---|---|---|
| **Nemotron-3.5-Bolt-E4B** | the assistant: NVIDIA's post-trained Gemma 4 E4B, with thinking, tool calls and image input, and NVIDIA's DFlash draft for speculative decoding | `POST /v1/chat/completions` |
| **EmbeddingGemma 2** | embeddings of text and images, for retrieval | `POST /v1/embeddings` |
| **vega's checker** | laya's System One: yes/no checks of a draft reply before it is sent (son-of-anton's end-of-turn checks) | `POST /v1/systemone` |
| **Qwen3-Reranker-0.6B** | reranking retrieved documents, as llama.cpp's server does it | `POST /v1/rerank` |

Text-to-speech and speech-to-text are planned for the same card.

C++20 and SYCL/ESIMD on Level Zero. There is no Python at runtime, and every kernel is eklow's own (no oneDNN, no oneMKL,
no llama.cpp). It builds with Nix. It was built by Claude (Opus 5.5 and Sonnet 5.5) with an engineering log, `DECISIONS.md`,
that records every measurement and dead end. Until 2026-11-11 it also served Qwen3.6-35B-A3B and its finetunes (Ornith-1.5,
Sage-A1.5), a mixture of experts streamed between the card and the CPU. Bolt does the job better on a quarter of the card,
so that path is gone (DECISIONS 2026-11-11, which names the commit that still has it).

## The machine

```
                 i7-12700H, 32 GB DDR4
          ┌─────────────────────────────────────────────┐
          │  CPU: tokenizers, scheduler, HTTP server,     │
          │    vega's engine and laya's policies          │
          │  Iris Xe iGPU: not eklow's                     │
          └──────────────┬──────────────────────────────┘
                         │ PCIe 4.0 x8
          ┌──────────────┴──────────────────────────────┐
          │ Arc A770M (DG2), 16 GB: 32 Xe-cores of XMX    │
          │   Bolt (5.71 GiB) and its K/V (2.20 GiB)      │
          │   the DFlash draft (547 MiB)                   │
          │   Gemma 4's image projector (203 MiB)          │
          │   EmbeddingGemma 2 (0.27 GiB, projector 269)   │
          │   the checker's reader (1,193 MiB)             │
          │   the reranker (796 MiB)                       │
          │   ~3.2 GiB free                                │
          └──────────────────────────────────────────────┘
```

Bolt is 42 dense layers of hidden 2,560. Sliding-window layers (window 512) alternate with a global layer every sixth.
Each layer has its own slice of a per-layer input table, and the last 18 layers reuse the K/V of earlier ones. NVIDIA's
Q4_K_M file fits on the card whole, with two slots of its full 65,536-token context in fp16.

## How well

Measured on the A770M through the server, in production's shape: two slots of 65,536, fp16 K/V, and the other three
models loaded (DECISIONS 2026-11-10 and 2026-11-11).

| Bolt | |
|---|---|
| prefill at 8k / 32k / 60k | 3,549 / 2,920 / 2,273 tok/s (2.3 s to the first token for 8k from empty) |
| decode at 8k / 32k / 60k, plain | 44.8 / 39.3 / 35.6 tok/s |
| the same with the DFlash draft in blocks of 8 | 90.8 / 87.9 / 64.6 tok/s (2.8-3.4 tokens a verify) |
| GSM8K (48 requests, thinking on), 1 / 2 at a time, with the draft | 92.2 / 158.0 tok/s aggregate |
| an image (280 tokens) | 0.10 s to encode |

| the side models | idle | while Bolt streams |
|---|---|---|
| checker, three questions over a 4,366-token turn | 270 ms | 279-292 ms |
| checker, the longest turn (14,648 tokens) | 405 ms | 413-433 ms |
| rerank, 8 documents (2,214 tokens) | 97 ms | 98-104 ms |
| rerank, 24 documents (6,640 tokens) | 263 ms | 255-275 ms |
| embedding, one sentence / 1,648 tokens | 62 / 122 ms | 72-90 / 130-156 ms |

A side request holds the chat scheduler while it runs, so Bolt's stream pauses for about as long and then continues.

**Accuracy.**
- **Bolt.** eklow's fp32 reference forward matches transformers on the GGUF's own weights: top-1 256/256 at mean KL 5e-8
  over a 16k prompt's last 256 positions. The serving engine against the same gives 255/256 at 2.8-6.1e-5 for prefill, and
  1.7e-3 over decoded positions.
  - Prefill is inside transformers' own bf16-against-fp32 distance on the same prompts (8-10e-4).
  - The small-batch matmuls use int8 activations, so a step's results depend on its shape. Two runs of one request agree
    token for token only until a near-tie goes the other way.
  - On the private replay set of real assistant turns Bolt scored 30 and 31 of 54 against llama.cpp CUDA's 32 on the same
    file.
- **EmbeddingGemma 2.** Cosine 0.99987-0.99995 against `google/embeddinggemma-2` (fp32) for text up to 6,989 tokens, and
  0.99958-0.99994 for images.
- **The checker.** Within 0.0014 of laya with an fp32 reader on the replay set's 29 turns, and no flag differs at
  son-of-anton's cut-offs. With the vega-cv32 policy the largest difference is 0.0011.
- **The reranker.** Within 0.0021 of llama.cpp with an f16 cache, and within 0.00036 of an fp32 run of the same weights.

Speculative decoding is lossless in distribution. The acceptance is checked by chi-square against plain sampling on
fixed logits for 17 sampler configurations (`eklow-spec-test`), and through the server with 3,000 completions with the
draft against 3,000 without.

## Features

- **OpenAI-compatible chat** (`eklow-server`): `/v1/chat/completions` (SSE streaming or whole), `/v1/models`, `/health`,
  and `/metrics` (Prometheus text, for llmtop).
  - Requests take `reasoning_content` (thinking is on unless `reasoning_effort` is `none`), tool calls and `tool_choice`
    (`"none"` keeps the tools in the prompt and never samples a call), stop strings, `seed`, usage with `cached_tokens`,
    and image_url parts.
  - Bolt's template variables `enable_thinking`, `conversational_mode` and `preserve_thinking` can be set
    (`--preserve-thinking on` makes the last the default).
  - With the draft, the usage reports `speculation`, and `"speculative": false` decodes a request plain.
- **The text layer, ported rather than interpreted.** Bolt's chat template and Google's Gemma 4 template are
  byte-identical to `apply_chat_template` (0 differences on 2 x 21,014 cases). The streaming parser separates reasoning,
  content and tool calls. The tokenizers match Hugging Face's `tokenizer.json`.
- **Tool calls that cannot be malformed.** A grammar over Gemma 4's tool-call syntax constrains sampling while a call is
  open. Inside a drafted block, each row is checked against the grammar state after the tokens before it.
- **Sampling as llama-server does it**:
  - its chain: penalties, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature, mirostat 1 and 2;
  - its defaults, under the GGUF's `general.sampling.*`, under the request's fields;
  - checked candidate for candidate against llama.cpp's own samplers (`tools/sampler_test`).
- **Speculative decoding** (`--draft`, `--draft-block`): NVIDIA's DFlash draft writes a block of 8 tokens in one pass, and
  the target checks it as one chunk. It stays lossless for every sampler feature and the tool grammar. Two generating slots
  verify in one step.
- **Scheduling.** Continuous batching of two slots with chunked prefill (2,048 tokens a sequence a step), generating slots
  first. Prefix reuse keeps a slot's K/V for the next turn. The K/V cache is paged for the global layers, with rings for
  the sliding ones.
- **Image input** (`--mmproj`). Pictures are prepared as Hugging Face's processor does and encoded on the matrix units by
  Google's Gemma 4 E4B tower, which Bolt carries unchanged.
- **Embeddings** (`--embed-model`, `--embed-mmproj`). EmbeddingGemma 2 runs on its own queue and port. It takes text and
  image inputs.
- **The end-of-turn checker** (`--checker-*`). laya's System One over vega:
  - Qwen3.5-0.8B (vega's reader) runs on the card, with prompts built token for token as vegaml builds them and pooled as
    it pools them;
  - vega-U's engine and laya's trained policy run on the CPU, and the policy's hashes are checked at startup.
- **The reranker** (`--rerank-*`). llama.cpp's `POST /v1/rerank` (also `/rerank`, `/v1/reranking`) over ggml-org's
  Qwen3-Reranker GGUF, scoring llama.cpp's P(yes). The pairs of a request are packed into passes.
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

The model files are flake packages too (`nix/models.nix`):
- `bolt-e4b`, `bolt-e4b-mmproj` and `qwen3-reranker` are fetched at pinned revisions;
- `bolt-e4b-dflash` is quantized to Q8_0 at build time from NVIDIA's BF16 draft;
- `vega-reader` is Qwen3.5-0.8B converted at build time by the pinned llama.cpp.

## Running

```sh
eklow-server --model Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf --alias nemotron-3.5-bolt-e4b --host 0.0.0.0 --port 8081 \
  --draft Nemotron-3.5-Bolt-E4B-DFlash-Q8_0.gguf --mmproj mmproj-gemma-4-E4B-it-BF16.gguf \
  --embed-model embeddinggemma-2-Q8_0.gguf --embed-mmproj mmproj-embeddinggemma-2-Q8_0.gguf --embed-port 8082 \
  --checker-reader Qwen3.5-0.8B-BF16.gguf --checker-vega vega-U/ --checker-policy POLICY_DIR/ --checker-port 8084 \
  --rerank-model qwen3-reranker-0.6b-q8_0.gguf --rerank-port 8085
```

```sh
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "nemotron-3.5-bolt-e4b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": true
}'
curl http://localhost:8084/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "...",
  "questions": {"claim": {"type": "noul", "instructions": "Does the draft say it did something no tool call did?"}}
}'
curl http://localhost:8085/v1/rerank -H 'Content-Type: application/json' -d '{
  "query": "When was the boiler serviced?", "documents": ["The boiler was serviced on 14 March.", "A cake recipe."]
}'
```

The defaults are Bolt's own: two slots of its 65,536-token window, fp16 K/V, blocks of the draft's trained 8. The server
prints what it put on the card, part by part, and the free memory left. It refuses a GGUF of any other architecture before
it touches the device. The comment at the top of `tools/server/main.cpp` explains every flag.

## The NixOS module

The flake's module, `services.eklow`, serves named chat instances, each as the unit `eklow-<name>.service`.
- An instance's options are `model`, `alias`, `port`, `slots`, `contextSize`, `kvInt8`, `draft`, `draftBlock`, `mmproj`,
  `preserveThinking` and `extraArgs`.
- The side models are shared by whichever instance runs:
  - `embedModel`, `embedMmproj`, `embedPort` and `imageTokens`;
  - `checker.{reader, vega, policy, policyFiles, port, alias, maxTokens}`;
  - `reranker.{model, port, alias, maxTokens}`.
- Instances exclude each other (one card), and `autoStart` names the one started at boot.
- `checker.policyFiles` takes a private laya policy as three root-only files (agenix's, say), which the unit copies into its
  runtime directory for its dynamic user.

```nix
services.eklow = {
  enable = true;
  package = inputs.eklow.packages.${system}.default;
  instances.bolt = {
    model = inputs.eklow.packages.${system}.bolt-e4b;
    alias = "nemotron-3.5-bolt-e4b";
    port = 8081;
    draft = inputs.eklow.packages.${system}.bolt-e4b-dflash;
    mmproj = inputs.eklow.packages.${system}.bolt-e4b-mmproj;
    kvInt8 = false;
    preserveThinking = true;
  };
  checker = {
    reader = inputs.eklow.packages.${system}.vega-reader;
    policyFiles = { heads = "/run/agenix/eklow-checker-heads"; calibration = "/run/agenix/eklow-checker-calibration";
                    manifest = "/run/agenix/eklow-checker-manifest"; };
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
- One Intel Arc DG2 card, through Level Zero. The kernels are tuned for the A770M's 32 Xe-cores, 128 registers a thread and
  64 KB of shared memory a work-group.
- Card memory is sized from the free memory the driver reports (Level Zero Sysman), because allocations succeed past what
  the card holds and the first kernels then fail.
- Nothing else may use the card beside the server. A second program on it starves the running server into
  `UR_RESULT_ERROR_OUT_OF_RESOURCES`, or slows it to a crawl, until it goes away.

## Repository

| | |
|---|---|
| `src/engine` | Bolt's serving forward pass (`engine_gemma4`): steps of prefill chunks, decode tokens and verify blocks over the paged cache and the sliding rings |
| `src/kernels` | XMX matmuls for the files' formats (int8 GEMVs for decode, fp16 GEMMs for prefill), flash attention over the paged cache and long prefill attention in key segments, the draft's kernels, the DeltaNet prefill of vega's reader (`gdn_seq`), fused small kernels; `ref` holds the reference kernels |
| `src/model` | GGUF loading, Gemma 4 E4B's configuration and reference forward pass, the DFlash draft, the Gemma vision tower, EmbeddingGemma 2, Qwen3-Reranker |
| `src/vega` | vega's checker: the Qwen3.5-0.8B reader, vegaml's prompt and engine, laya's policies and System One |
| `src/serve`, `src/text` | the scheduler, OpenAI layer and tool-call grammar, the sampler and lossless acceptance; the tokenizers (Gemma's, and Qwen's for the reader and the reranker), Gemma 4's chat templates and output parser |
| `src/rt`, `src/image` | device selection and free memory, the copy engine; image decoding and preparation |
| `tools/` | `eklow-server` (`tools/server`), the engine, kernel and draft tests and benchmarks, the checker's and the reranker's tools and references (`tools/vega`, `tools/rerank`), the reference runs (`tools/ref`), the server tests and benchmarks (`tools/server`, `tools/bench`) |
| `nix/` | the package, the NixOS module, the model files (`models.nix`), and llama.cpp's CPU backend as the oracle |
| `DESIGN.md` | the original target and the shape of the system |
| `DECISIONS.md` | dated decisions with their measurements: why each piece is the way it is |
| `STATUS.md` | what's done and what's open |

## Third-party code and credits

eklow links [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) for its HTTP server and vendors stb_image (public
domain / MIT) to decode pictures; it carries no other code from other projects.

The models' math follows llama.cpp's graphs (MIT), and llama.cpp is also the oracle (`nix/oracle.nix` builds it); the
quantized layouts are ggml's. The chat templates are ports of the Gemma 4 and Nemotron-3.5-Bolt-E4B `chat_template.jinja`.
The tokenizers follow Hugging Face's `tokenizer.json`. The checker follows vegaml and laya; the reranker follows
llama.cpp's rank pooling.

The models' licenses are their own:
- Nemotron-3.5-Bolt-E4B and its DFlash draft are NVIDIA's (OpenMDW-1.1).
- Gemma 4 E4B's projector and EmbeddingGemma 2 are Google's (Apache-2.0).
- Qwen3.5-0.8B and Qwen3-Reranker-0.6B are Alibaba's (Apache-2.0).
- vega-U is Apache-2.0.

## License

Copyright 2026 Ethan Todd.
Licensed under the Apache License, Version 2.0 (`LICENSE`).
