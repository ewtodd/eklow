# eklow

A from-scratch serving engine for **Qwen3.6-35B-A3B** and its finetunes (stock Qwen3.6, **Ornith-1.5**,
**Sage-A1.5**), in our own Q4_K / Q8_0 mix, on a single Intel Arc A770M, a 16 GB laptop GPU, in a machine with 32 GB of
RAM, with **DFlash 2** speculative decoding and **EmbeddingGemma 2** beside it in the same process. It serves an
OpenAI-compatible API with continuous batching, prefix reuse over the model's recurrent state, thinking,
grammar-constrained tool calls, image input, and two slots of the model's full 262,144-token context.

C++20 and SYCL/ESIMD on Level Zero; no Python at runtime; every kernel is our own (no oneDNN, no oneMKL, no
llama.cpp). It builds with Nix against the fleet's nixpkgs (intel-llvm, compute-runtime).

It is *vertical*: one model family, one card. It was built in a week by Claude (Opus 5.5 and Sonnet 5.5), with an
engineering log (`DECISIONS.md`) that records every measurement and dead end along the way. It served Gemma 4 (26B A4B
and 12B) first, then Qwen3.6 from 2026-11-02 in its place; the 26B ran again beside Qwen's from 2026-11-03 to
2026-11-07 and was then removed (DECISIONS 2026-11-07; `9775e98` is its last state).

## The machine

```
                 i7-12700H, 32 GB DDR4 (two channels)
          ┌─────────────────────────────────────────────┐
          │  CPU: tokenizer, scheduler, HTTP server;      │
          │    the cold experts at decode and prefill     │
          │    (AVX-VNNI, 16 threads, from pinned         │
          │    memory: ~13 GiB); snapshots of the         │
          │    DeltaNet state (64 MiB each, pinned)       │
          │  Iris Xe iGPU: not eklow's (llama.cpp titles) │
          └──────────────┬──────────────────────────────┘
                         │ PCIe 4.0 x8, ~11 GB/s
          ┌──────────────┴──────────────┐
          │ Arc A770M (DG2), 16 GB      │  32 Xe-cores of XMX: fp16 98.8 TFLOPS, int8 198 TOPS
          │ dense weights, KV cache,    │
          │ DeltaNet states, the hottest│
          │ experts of every layer,     │
          │ streaming slots, the        │
          │ DFlash 2 draft,             │
          │ EmbeddingGemma 2            │
          └─────────────────────────────┘
```

Qwen3.6-35B-A3B has 40 layers of hidden 2,048: every fourth one gated attention (16 query heads, 2 K/V heads of 256
dimensions), the other 30 Gated DeltaNet (a 128 x 128 recurrent state per value head, 32 of them, ~60 MiB a sequence);
every layer 256 routed experts of 512 beside a shared one, 8 active a token. 35B parameters, 3B active; the routed
experts are 32 of the 35 and 18 GB at 4.5 bits a weight. Only the 10 attention layers keep a K/V cache (5.6 GiB for 2 x
262k in int8), so the long context is cheap on the card; the experts are the problem. The card holds the dense weights,
the caches, the draft and, by a routing profile (`data/qwen36-experts.json`, `data/ornith-1.5-experts.json`), as many of
each layer's busiest experts as the driver reports free memory for. The rest are computed by the CPU at decode while the
card computes the hot ones; for prefill the busiest cold experts of each chunk are copied to the card on its copy engine
and run in the same grouped GEMMs.

**DFlash 2** (Inco AI's draft, which serves stock Qwen3.6 and Ornith alike): a six-layer draft reads the
target's hidden states from eight layers and writes a block of 8 (or 16) candidate tokens in one pass. The target checks
the block as one chunk, and rejection sampling accepts a prefix so that the output distribution is exactly what plain
decoding samples, with every sampler feature and the tool-call grammar inside it (DECISIONS 2026-10-31).

**The recurrent state and prefix reuse.** A transformer's K/V can be cut anywhere; a DeltaNet state cannot be
un-run. For Qwen3.6 the server keeps snapshots of it (64 MiB, in pinned host memory, under a budget) at the starts of
the conversation's messages, every 16k tokens of a long prefill, and at the end of a generation, and a new request takes
the slot and snapshot that leave the least to prefill.

**One K/V pool for the slots** (`--kv-pool`, DECISIONS 2026-11-07): the slots draw the attention layers' K/V pages from one
pool instead of each owning a window's worth, so at 2 slots x 262,144 a pool of 262,144 positions halves the K/V and state
(3.03 GiB instead of 5.84) and the card keeps 91 experts a layer instead of 48 (decode with the draft +22% at 8k, +11% at
32k). An idle slot gives pages back and re-prefills later, a request that does not fit waits, and a sequence starved with
both slots busy ends with `finish_reason` "length".

## Performance

Measured on that machine, through `eklow-server` with the production shape: two slots of 262,144 tokens, int8 K/V,
2,048-token prefill chunks, EmbeddingGemma 2 loaded. The method is unholy's (`tools/bench/serve_perf.py`, unique Pile
text cut to the model's token depths).
- **Qwen3.6** (2026-11-01): each model in EKK8, Inco's DFlash 2 draft (Q8_0) in blocks of 4 with its token table on the
  host, 4 GiB of snapshots: 59 of 256 experts a layer on the card, 12.99 GiB of cold experts pinned (raw output in
  `docs/bench/release/{stock,ornith}/`). The stock column's prefill and decode rows are from 2026-11-03, with layer-major
  prefill groups (`--group 8`, which leaves 57 experts a layer on the card; DECISIONS 2026-11-03; raw output in
  `docs/bench/layermajor/R-g8-*`); its GSM8K rows and the Ornith column are from 2026-11-01, before the groups (GSM8K's
  prompts are shorter than a group).

| | stock Qwen3.6-35B-A3B | Ornith-1.5-35B-A3B |
|---|---|---|
| prefill, one request, unique text, at 8k / 32k / 128k / 252k | 1,605 / 1,855 / 997 / 708 tok/s | 1,039 / 1,043 / 726 / 546 tok/s |
| time to first token at 252k | 356 s | 461 s |
| decode, one stream, with the draft, at 8k / 32k / 128k / 252k | 44.2 / 42.6 / 37.3 / 33.7 tok/s | 44.3 / 40.4 / 33.0 / 30.4 tok/s |
| decode without the draft | 33.9 / 33.0 / 28.3 / 25.1 tok/s | 32.8 / 31.1 / 27.1 / 24.1 tok/s |
| GSM8K chat (48 requests, thinking on, up to 1,024 tokens), 1 / 2 at a time, with the draft | 57.0 / 67.2 tok/s aggregate | 54.6 / 61.5 tok/s |
| the same without the draft | 34.8 / 48.9 tok/s | 33.5 / 46.4 tok/s |

Decode is the median of five requests on a cached document, with 2.7-3.1 tokens a verify on the Pile's prose and 3.5 on
GSM8K. The two Qwen models run at the same speed (they share the architecture and the files' mix); Ornith answers the
GSM8K questions in ~280 tokens where stock's thinking runs to the 1,024-token cap, so its 48 requests take a third of
the time.

The draft pays where the text is predictable: on reasoning and code a block of 4 gives 3.5 tokens a verify and 1.5x plain
decode; on the Pile's prose 2.7 and 1.3x at 8k, nothing at 252k, where the verify's attention costs what the draft saves.
The block is a trade the card sets: a verify beyond its first row costs ~8.5 ms a row (the cold experts' CPU time grows
with the distinct experts the rows touch), so on GSM8K at 8k one stream decodes 35.4 tok/s plain, 56.3 with blocks of 4,
54.5 of 6, 53.1 of 8, 50.1 of 12 and 42.7 of 16, and on prose 4 beats 8 by a third; hence the default of 4
(`--draft-block`; a request may ask for its own with `"speculative": N`). Acceptance, to compare with llama.cpp's on the
same pair (blocks of 8, temperature 1, top-p 0.95, top-k 20, thinking on, 16 requests of z-lab's prompts each,
`docs/bench/serve-accept-*`): GSM8K 5.75 tokens a verify (llama.cpp 5.24), HumanEval 5.43 (4.98), MT-Bench 5.02 (4.53),
and with blocks of 4: 3.54 / 3.48 / 3.34.

Prefix reuse (`docs/bench/serve-perf-qwen-reuse-*`): a 10k-token conversation (a system prompt of reference text,
thinking on) takes 9.2 s cold. Its next turn, with stock's template (which drops the earlier turn's reasoning, so the
shared prefix ends before the generation prompt), prefills 105 of 10,057 tokens in 0.61 s; with Ornith's (which keeps
it, so the whole first answer is shared) 20 of 10,403 in 0.26 s. A question on a 100k-token document already read takes
0.28-0.31 s instead of 146-159 s. When a sampled answer is not what the template renders back (the
model sampled " D" + "amas", the tokenizer says " Damas"), the end-of-generation snapshot no longer matches and the next
turn falls back to the one before the generation prompt (Ornith's third turn: 677 tokens, 1.3 s); the server logs where
such a sequence and a prompt part. Needle in a haystack through the server, two requests at once and embeddings beside
them: the passphrase found at 88k and at 167k tokens (`docs/bench/serve-needle-qwen-*`).

Prefill is the cost of the experts: Qwen3.6's are 18 GB, only 57-59 of each layer's 256 fit on the card beside 2 x 262k
of K/V (more with the shared pool), and most of a chunk's expert work is cold. A long prompt runs in groups of up to 8
chunks layer by layer, so each cold expert crosses PCIe once per layer per group instead of once per chunk, and the
streamed experts are repacked in one launch (DECISIONS 2026-11-03); the long prefill attention runs in key segments as
two GEMMs.

Embeddings do not wait for a prefill: 91-238 ms during the needle test's 88k and 167k prefills.

## Accuracy

Correctness is checked logit by logit against eklow's own fp32 reference forward pass (`eklow-ref-run`: plain SYCL
kernels, fp32 activations, weights dequantized as ggml stores them).

- **Qwen3.6** (`bench.sh checkq`), on a 2,573-token prompt's last 256 positions and a chat turn, prefilling in chunks of
  1,024 and 2,048, decoding, and verifying blocks of 8 and 16 that accept 3 to 5 tokens. Stock Qwen3.6 in the production
  file (EKK8: Q4_K experts, Q8_0 dense matrices, Q6_K head): top-1 237-246 of 256 at a median KL of 1.7-3.5e-3; Ornith in
  EK8: 201-221 of 256 (DECISIONS 2026-11-01).

The int8 activations of the small-batch matmuls make a step's results depend on its shape, so two runs of one request
agree token for token only until a near-tie goes the other way (a request decoded alone and beside another do too).

The speculative decoding is checked as a distribution: a chi-square test of the acceptance over eleven sampler
configurations (`eklow-spec-test`), and through the server, 3,000 completions at temperature 1 with the draft against
3,000 without it (`tools/server/serve_test.py`).

Every fast kernel has a reference kernel next to it (`src/kernels/ref.*`), and its test compares against that and
an exact CPU emulation of the quantized arithmetic (`eklow-attn-test`, `eklow-gdn-test`, `eklow-qfmt-test`,
`eklow-cpu-qgemv-test`).

## Features

- **OpenAI-compatible server** (`eklow-server`):
  - endpoints: `/v1/chat/completions` (streamed or whole), `/v1/models`, `/health`, `/metrics` (Prometheus text, for llmtop); with an embedding model,
    `/v1/embeddings`, `/v1/models` and `/health` on a port of its own;
  - request features: SSE streaming, `reasoning_content` (thinking on unless `reasoning_effort` is `none`), tool calls
    and `tool_choice` (`"none"` keeps the tools in the prompt, so it stays cached, and never samples a call), stop strings, seeds, temperature / top-k / top-p / min-p and penalties, usage with
    `cached_tokens`; also `preserve_thinking` for stock's and Sage's templates (`--preserve-thinking on` makes it the
    default, so the turn after an agentic one reuses the cache instead of prefilling the turn again), sampling defaults from the model's
    `general.sampling.*`, and with a draft `speculation` in the usage (verify blocks, tokens drafted and accepted) and
    `"speculative": false` to decode one request plain;
  - a keep-alive on the streams (`--keepalive`, default 15 s): a stream silent that long gets an SSE comment, so a long
    tool call does not look like a dead connection (DECISIONS 2026-11-07).
- **The text layer, matching the references exactly**, its tokenizer identical to Hugging Face's:
  - Qwen3.6's three chat templates (stock's, which drops earlier turns' reasoning; Ornith's, which keeps it; Sage-A1.5's,
    stock plus a `verification` switch: a request's `verification: true` or `--verification on|off` appends the
    verification protocol to the system text) byte-identical to `apply_chat_template` and recognized from the GGUF
    (`--chat-template stock|ornith|sage` for a file that carries another, such as Unsloth's edit in the official Ornith
    GGUFs), and a streaming parser for the reasoning, content and tool calls (parameter values back to JSON by the
    request's schemas).
- **Tool calls that cannot be malformed**: a grammar over Qwen3.6's tool-call syntax constrains sampling while a call
  is open (`src/serve/tool_grammar_qwen.cpp`, checked against the output parser by `eklow-grammar-test-qwen`); inside a
  drafted block each row is checked against the grammar state after the tokens before it.
- **Speculative decoding** (`--draft`, `--draft-block`): lossless for the sampler's every feature; two
  generating slots verify in one step; a stop string, an end token or `max_tokens` inside an accepted block ends the
  request at that token.
  Qwen3.6 runs Inco's DFlash 2 draft.
- **Prefix reuse over a recurrent state** (`--snapshot-mib`, `--snapshot-every`): snapshots where the next
  request is likely to diverge, kept under a budget with the least recently used dropped first.
- **Scheduling**: continuous batching of two slots with chunked prefill (2,048 tokens a sequence a step), generating
  slots first, the rest of the step filled with prefill; a paged K/V cache for the attention layers, optionally one pool for all
  the slots (`--kv-pool`), in int8 with a scale per 16 dimensions (what lets 2 x 262k fit); layer-major prefill groups
  (`--group`).
- **Embeddings beside chat**: EmbeddingGemma 2 on its own queue on the card, so an embedding does not wait behind a
  long prefill; image inputs to `/v1/embeddings` with its projector (`--embed-mmproj`, `--image-tokens`).
- **A yes/no checker beside chat** (`--checker-*`): laya's System One (`POST /v1/systemone`, laya's request and answer)
  over vega (nandakishorm/vega-U): its reader, Qwen3.5-0.8B, on the card with prompts built token for token as vegaml
  builds them and pooled as it pools them; its engine and laya's trained policies (heads, temperatures, their hashes
  checked) on the CPU. For son-of-anton's end-of-turn checks (DECISIONS 2026-11-10, "vega's checker").
- **A reranker beside chat** (`--rerank-*`): llama.cpp's `POST /v1/rerank` over ggml-org's Qwen3-Reranker GGUF, its score
  llama.cpp's P(yes), the pairs of a request packed into passes (DECISIONS 2026-11-10, "Qwen3-Reranker on eklow").
- **Mixture of experts on 16 GB**: routing, grouping and the experts' GEMVs and GEMMs on the card; a placement by routing
  profile, with cold experts on the CPU (AVX-VNNI, Q4_K and Q4_0) at decode and streamed over PCIe for prefill.
- **Image input to chat** (`--mmproj`, llama.cpp's projector GGUF), OpenAI `image_url` parts (base64 `data:` URLs; JPEG,
  PNG, GIF, BMP), preprocessed as Hugging Face's processor does and encoded on the matrix units, the encoder's weights streamed from
  host memory, with prefix reuse that tells images apart: Qwen3.6's vision tower (which Ornith-1.5 shares unchanged):
  0.1-0.4 s an image at the default budget of 1,024 tokens (`--image-max-tokens`); the image's rows enter the prompt
  with Qwen's 3D rotary positions. Against llama.cpp the logits after an image stay within its own CPU-CUDA band
  (DECISIONS 2026-11-02).

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

Ornith-1.5 with its draft and projector (Sage-A1.5 and stock Qwen3.6 the same, with their own file and `--chat-template`):

```sh
eklow-server --model Ornith-1.5-35B-A3B-Q4_K-Q8_0.gguf --alias ornith-1.5-35b-a3b \
  --slots 2 --ctx 262144 --kv-int8 --expert-profile share/eklow/ornith-1.5-experts.json \
  --draft Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf --mmproj mmproj-Ornith-1.5-35B-A3B-bf16.gguf --snapshot-mib 4096 \
  --embed-model embeddinggemma-2-Q8_0.gguf --embed-port 8082 --host 0.0.0.0 --port 8081
```

```sh
curl http://localhost:8081/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ornith-1.5-35b-a3b",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": true
}'
```

The model files are ours (`tools/weights/` makes them: the published Qwen3.6, Ornith and Sage checkpoints requantized
to EKK8 or EK8). The server prints what it placed where: the weights and KV on the card, how many experts a layer stayed
there, the streaming slots, the pinned memory, the draft and the snapshots' budget. A GGUF whose architecture is not
`qwen35moe` is refused before any device is touched. The package installs the profiles as
`share/eklow/qwen36-experts.json` and `ornith-1.5-experts.json`. The comment at the top of `tools/server/main.cpp`
explains every flag.

The flake's NixOS module serves named instances, as unholy's does: `services.eklow.instances.<name>` (`model`, `alias`,
`port`, `expertProfile`, `mmproj`, `contextSize`, `slots`, `kvInt8`, `extraArgs`, `draft`, `draftBlock`, `prefillGroup`,
`prefillGroupBusy`, `chatTemplate`, `snapshotMiB`, `kvPool`, `verification`, `preserveThinking`) is the unit `eklow-<name>.service`, with
the embedding model and its projector shared (`embedModel`, `embedMmproj`, `embedPort`, `imageTokens`). The instances
hold the one card, so each conflicts with the others: starting one stops the one running. `autoStart` names the one
started at boot (by default the only one); the house switches them with `llm-mode` instead.

```nix
services.eklow = {
  enable = true;
  instances.ornith = { model = ...; alias = "ornith-1.5-35b-a3b"; port = 8081; draft = ...; mmproj = ...;
    expertProfile = "${config.services.eklow.package}/share/eklow/ornith-1.5-experts.json"; };
};
```

The log, on stdout (`journalctl -u eklow-<name>` under the module), is unholy's, with the speculation added:
a line per chat request as it ends, a line per embedding request and per rejected request, and every 10 s while there is
traffic (`--log-interval`) the engine's rates, slots, prefix hits, the draft's acceptance and where a
step's time goes:

```
13:55:21 request chatcmpl-e081499e185e54f2768b5c68: stop | prompt 20, completion 567 | queue 0.00 s, ttft 0.21 s, prefill 94.8 tok/s, decode 45.7 tok/s | draft 3.46 tokens/verify, 404 of 492 accepted
13:55:23 engine: prompt 2.5 tok/s, generation 46.0 tok/s | running 1, waiting 0 | kv 0.1% (693 of 524288 positions, 586 cached) | prefix hit 0.0% | draft 3.47 tokens/verify, 82.2% of 399 drafted accepted | per step: engine 63.3 ms, draft 4.9, sample 6.0, inject 0.6, snapshot 0.1 | snapshots 1 (64 MiB), 1 taken, 0 restored

```

Prefill in a request's line is over the uncached part of its prompt, from the slot to the first token; cached
positions are those kept in idle slots for prefix reuse.

With `--checker-reader` (Qwen3.5-0.8B as `tools/vega/make-reader.sh` converts it), `--checker-vega` (a vega-U model directory)
and `--checker-policy` (a laya policy folder), the server also answers laya's System One on `--checker-port`:

```sh
curl http://localhost:8084/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": "...",
  "questions": {"claim": {"type": "noul", "instructions": "Does the draft say it did something no tool call did?"}}
}'
# {"model":"vega-u","answers":{"claim":{"type":"noul","noul":0.31,"probabilities":{...},"readout":"engine",...}},"usage":{...}}
```

The module's `services.eklow.checker` (`reader`, `vega`, by default the published vega-U, `policy`, `port`, `maxTokens`)
adds it to whichever instance runs. `eklow-vega serve` serves the checker alone, for development; `eklow-vega check`
holds the reader and engine to `tools/vega/golden.json`, and `features`, `engine` and `predict` to
`tools/vega/ref_vega.py` (vegaml and laya themselves). Likewise `--rerank-model` (ggml-org's Qwen3-Reranker GGUF) and
`--rerank-port` serve llama.cpp's `POST /v1/rerank` (module `services.eklow.reranker`; `eklow-rerank serve` alone,
`tools/rerank/` its reference and gate).

`eklow-probe` measures what the card actually does (allocatable memory, copy rates, launch and host-signal latency),
and `eklow-copy-bench` what PCIe copies cost the kernels beside them: the first things to run on new hardware.

## Hardware assumptions

This is not a general-purpose engine:
- One Intel Arc DG2 card (the A770M; the kernels are tuned for its 32 Xe-cores, 128 registers a thread and 64 KB of
  shared memory a work-group), through Level Zero.
- Its 16 GB set the shapes: at 2 x 262k, Qwen3.6 needs its cold experts in about 13 GiB of pinned host memory, a CPU
  with AVX-VNNI to compute them, and room for the snapshots (4 GiB by default).
- Card memory is sized from the free memory the driver reports (Level Zero Sysman), because allocations succeed past
  what the card holds and the first kernels then fail. Nothing else may use the card beside the server: a second
  program on it starves the running server into `UR_RESULT_ERROR_OUT_OF_RESOURCES` until it restarts.
- PCIe traffic slows every kernel on the card that streams memory (up to 5.8x at full copy rate), so streaming is
  rationed by a cost model (DECISIONS 2026-10-27).

## Repository

| | |
|---|---|
| `src/engine` | the serving forward passes: steps of prefill chunks, decode tokens and verify blocks, the paged cache, the DeltaNet states and their snapshots, expert placement, cold experts |
| `src/kernels` | XMX matmuls for the files' formats (int8 GEMVs for decode, fp16 GEMMs for prefill, grouped expert forms), the Gated DeltaNet, flash attention over the paged cache and long prefill attention in key segments, the DFlash 2 draft's kernels, fused small kernels and routing (`fast`, `qwen_fast`); `ref` holds the reference kernels |
| `src/cpu` | the AVX-VNNI GEMVs for the cold experts |
| `src/model` | GGUF loading, Qwen3.6's configuration and reference forward pass, the DFlash 2 draft, the vision encoders, EmbeddingGemma 2 |
| `src/vega` | vega's checker: the Qwen3.5-0.8B reader, vegaml's prompt and engine, laya's policies and System One; `tools/vega` holds its tool (`eklow-vega`) and its Python reference |
| `src/model/rerank_qwen3.*` | Qwen3-Reranker; `tools/rerank` holds its tool (`eklow-rerank`) and its reference |
| `src/serve`, `src/text` | the scheduler, OpenAI layer and tool-call grammar, the sampler and lossless acceptance; tokenizers, chat templates and output parsers |
| `src/rt` | device selection and free memory, the copy engine |
| `tools/` | `eklow-server`, the engine and kernel benchmarks and tests, the reference runs (`tools/ref`), the weights' tooling (`tools/weights`), the server tests and benchmark (`tools/server`, `tools/bench`) |
| `data/` | the routing profiles |
| `nix/` | the package, the NixOS module, and llama.cpp's CPU backend as the oracle |
| `DESIGN.md` | the target and the shape of the system |
| `DECISIONS.md` | dated decisions with their measurements: why each piece is the way it is |
| `STATUS.md` | what's done and what's open |

## Third-party code and credits

eklow links [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) for its HTTP server; it carries no other
code from other projects.

The models' math follows llama.cpp's qwen35moe and gemma-embedding2 graphs (MIT), which is also the oracle (`nix/oracle.nix`
builds it); the quantized layouts are ggml's. The chat templates are ports of the `chat_template.jinja` of
Qwen/Qwen3.6-35B-A3B, ornith-ai/Ornith-1.5-35B-A3B and InternScience/Sage-A1.5; the tokenizers follow Hugging Face's
`tokenizer.json`. The DFlash 2 draft is Inco AI's (z-lab's `dflash/model.py` architecture). Qwen3.6's weights are
Alibaba's (Apache-2.0), Ornith-1.5's ornith-ai's (MIT), and EmbeddingGemma 2's Google's, under their own
licenses.

## License

Copyright 2026 Ethan Todd.
Licensed under the Apache License, Version 2.0 (`LICENSE`).
