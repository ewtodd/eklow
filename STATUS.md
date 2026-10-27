# STATUS

Live state, written as a handoff (2026-10-27). Design and targets: `DESIGN.md`; history and evidence:
`DECISIONS.md`; what it is and the headline numbers: `README.md`.

## Production now (2026-10-27)

- **fiona's `eklow.service` serves Gemma 4 26B A4B** (since 2026-10-27 09:03, eklow 71bfb75, deployed by the
  owner): two slots of 262,144 tokens, int8 global KV, `--expert-profile` from the package's
  `share/eklow/gemma-4-26b-a4b-experts.json`. Its startup line: 6.24 GiB of weights and 6.41 GiB of KV on the card,
  N = 51 of 128 experts a layer on the card (free card memory 6.08 GiB before them, 1.06 after, margin 1 GiB), 40
  streaming slots (255 MiB), 7.20 GiB of cold experts in pinned host memory. EmbeddingGemma 2 (0.27 GiB) in the same
  process on :8082; chat on :8081.
- **Not yet deployed, in main** (2026-10-27): the global layers' long prefill attention as two GEMMs per 2,048-key
  segment (merged from `perf/attn-wide`), and 2,048-token prefill chunks by default (`perf/chunk2048`); together
  2.2-2.3x prefill at 128k-252k through the server (below). Deploying them is the owner's (update the eklow input in
  `/etc/nixos`'s flake.lock, rebuild; the module sets no `--chunk`). The server then places N = 46 of 128 experts a
  layer: the rings hold one more chunk (~0.4 GB) and the attention's ~100 MB is reserved before the experts.
- **`/etc/nixos`** (the owner deploys; agents only commit, as son-of-anton-bot, no co-author trailers):
  `ac7f4993` (the eklow module serves the 26B: model by URL and sha256, alias `gemma-4-26b-a4b`, the profile in
  `extraArgs`; maitred's `gemma` upstream entry renamed to `eklow/gemma-4-26b-a4b`) and `f7391db6` (opencode gets an
  `eklow/gemma-4-26b-a4b` model; son-of-anton on e-desktop gets its effort metadata; son-of-anton discovers maitred's
  models itself). The eklow input update in `flake.lock` (to 71bfb75) is staged there but not committed: the
  owner's. Open WebUI still had the old model id/URL when this was written (the owner is fixing it).
- **Clients**: pick `eklow/gemma-4-26b-a4b` explicitly (maitred's `auto` does not route to it). Text only: image
  parts get a clear "not supported yet" error.
- **The card is the service's.** Anything else on the A770M beside it (a second eklow-server, a bench) can starve it
  into `UR_RESULT_ERROR_OUT_OF_RESOURCES` on every request until it restarts: that happened 2026-10-27 09:05-09:19
  when a test server was started beside it, and `systemctl restart eklow` fixed it. `bench.sh` and
  `tools/bench/serve-perf.sh` now refuse to run while the service is active. To use the card: `systemctl stop eklow`
  (the fiona user may; polkit allows it), work, `systemctl start eklow`, and check `journalctl -u eklow`.
- **Titles** stay on llama.cpp (`little-titles`, Vulkan on the Iris Xe, :8080), unaffected by eklow.

## Production performance (2026-10-27, `docs/bench/serve-perf-26b-fiona-2026-10-27.txt`)

Through the live service, by unholy's method (`tools/bench/serve_perf.py`): unholy's unique Pile documents
(`son-of-anton:/work/data/compare/doc-*.txt`) cut to Gemma token depths with `eklow-tok` (copies in
`fiona:~/eklow-bench/data/gemma-{8192,32768,131072,258048}.txt`, with unholy's `gsm48.jsonl`). Per depth, one request
with nothing cached (its time to first token gives prefill), then four more questions on the cached document;
max_tokens 256, temperature 1.0, top_p 0.95, thinking on; decode from the stream, median of 5. Then the 48 GSM8K
5-shot chat requests (thinking on, up to 1,024 tokens) one and two at a time.

| depth | prefill tok/s (time to first token) | decode tok/s |
|---|---|---|
| 8k | 1,578 (5.2 s) | 37.6 |
| 32k | 1,257 (26.1 s) | 34.9 |
| 128k | 493 (266 s) | 30.9 |
| 252k | 352 (734 s) | 26.7 |

- Main, not yet deployed (the same method from the build tree, production flags, 2,048-token chunks, N = 46;
  `docs/bench/serve-perf-26b-chunk2048-fiona-2026-10-27.txt`): prefill 1,865 (4.4 s) / 1,794 (18.3 s) / 1,177 (111 s) /
  765 (338 s) tok/s at 8k / 32k / 128k / 252k; decode 36.6 / 34.0 / 30.5 / 26.5; GSM8K two at a time 40.2 tok/s
  aggregate (-4%: the CPU's larger share of cold experts). With the segments alone at 1,024-token chunks (N = 50):
  1,631 / 1,529 / 1,040 / 703 (`docs/bench/serve-perf-26b-segments-fiona-2026-10-27.txt`).
- Cached document: the next question starts in 0.2-0.7 s.
- GSM8K: 33.4 tok/s aggregate one at a time (34.0 per stream), 41.9 two at a time (21.4 per stream). Two streams add
  only 25%: at N = 51 the CPU computes both streams' cold experts (not separated further).
- Beside the 26B (`coexist` probe, live service): an embedding 60 ms idle, 78 ms during a 32k prefill, 60 ms during
  decode; a title 368 ms idle, 457-471 ms during prefill or decode. Neither waits on the 26B.
- What limits it: at depth, the global layers' prefill attention (deployed: ~11 TFLOPS of 98.8, 61% of prefill at 32k
  depth with every expert on the card; main: ~31 TFLOPS, 34% there and 55% at 128k with N = 51); at short depth with N = 51, the cold experts (PCIe streaming's tax on every kernel,
  ~3x on the hot GEMMs, and the CPU). Decode: grouped expert GEMVs at ~180 GB/s of 399, and the CPU's cold pairs.
- Engine numbers with every expert on the card (contexts up to ~2 x 32k; `bench.sh engine26`): prefill 1,914 tok/s
  at 32k depth in main (1,234 with the flash kernel), decode 68.1 / 61.4 / 43.4 tok/s at 0 / 4k / 32k; two slots 53.9 per sequence at 0.
- Accuracy (`bench.sh check26`, against eklow's fp32 reference over the 2,616-token prompt's last 256 positions):
  243/256 top-1 at mean KL 2.9e-2 with N = 51 and streaming; 242/256 at 2.8e-2 with every expert on the card;
  llama.cpp's CPU backend is 220/256 at 7.5e-2 on the same positions (the band).

## Where things are

- Source of truth: `github.com/ewtodd/eklow` (working copy `~/Software/eklow` on e-desktop, main pushed). fiona's
  build/run tree is `~/eklow` (rsync of main, no `.git`, built): the A/B baseline the harness calls `eklow`.
  `~/eklow-base-383` and `~/eklow-base-75a` are old baselines, safe to delete.
- Weights on fiona (`~/models`): gemma-4-26b-a4b-it-qat-q4_0.gguf (sha256 3eca3b8f..., Hugging Face
  google/gemma-4-26B-A4B-it-qat-q4_0-gguf at d1c082be, the file `gemma-4-26B_q4_0-it.gguf`), gemma-4-12b-it-qat-q4_0.gguf
  (93567e57...), embeddinggemma-2-Q8_0.gguf, and the 12B's mmproj; e-desktop has the 26B too. The 26B's mmproj
  (gemma4v, 1.19 GB) is not downloaded. The service's copies come from the Nix store (fetchurl in the module).
- Toolchain: the fleet's nixpkgs (c3f15ff4): intel-llvm 7.1.1, level-zero 1.32.0, intel-compute-runtime 26.31.
- fiona: A770M on PCIe 4.0 x8 (~11 GB/s; `level_zero:0`), Iris Xe (`level_zero:1`, titles), i7-12700H (AVX-VNNI),
  32 GB DDR4.
- Image input: the branch `image-input` (cf6e8a2, pushed): image preparation, the 12B's gemma4uv embedder on
  reference kernels, the oracle through mtmd; not checked against the oracle, not in the engine.

## How to work on it

- Build / run on fiona:
  ```
  rsync -a --delete --exclude .git --exclude build --exclude 'result*' ~/Software/eklow/ fiona:eklow/   # e-desktop
  ssh fiona 'cd ~/eklow && nix develop path:. -c bash -c "cmake -S . -B build -G Ninja && ninja -C build"'
  ```
  `nix build path:.` on e-desktop builds the package; the oracle is `nix build .#oracle` (CPU-only; fiona does not
  accept unsigned store paths from e-desktop, so it runs where it is built; fiona has one at `~/eklow-perf/oracle`).
- The harness, `fiona:~/eklow-perf` (copies in `~/Software/eklow-perf` on e-desktop; edit there, scp to fiona):
  `bench.sh engine|profile|kernels|attn|check|engine26|check26` (each under `~/eklow-perf/gpu.lock`, refusing to run
  while `eklow.service` is up), `check.sh` (the 12B against llama.cpp CPU dumps in `~/eklow-oracle`), `check26.sh`
  (the 26B against the fp32 reference dumps in `~/eklow-ref-26b`; extra arguments go to eklow-engine-run, e.g.
  `--expert-profile data/gemma-4-26b-a4b-experts.json --experts-on-card 51`), `ab.sh TREE_A TREE_B ROUNDS args`
  (alternating engine benches, medians), and `AGENTS.md` (the workers' rules and the correctness bands).
- Reference data on fiona: `~/eklow-runs/{long,chat}/tokens.json` (the 2,616- and 25-token prompts);
  `~/eklow-oracle` (12B, llama.cpp CPU), `~/eklow-oracle-26b` (26B, llama.cpp CPU; `long256` with 256 logits rows),
  `~/eklow-oracle-26b-cuda` (llama.cpp CUDA, made on e-desktop's 4090), `~/eklow-ref-26b` (eklow-ref-run's fp32
  dumps: `long` 256 rows and every layer, `chat`); `~/eklow-runs/routing-2026-10-26` (the four routing recordings
  behind the profile); `~/eklow-bench` (the server benchmark's inputs and results).
- Workers: Sonnet subagents (the Agent tool, model sonnet), one per git worktree, `~/Software/eklow-wt/{attn,gemm,moe}`
  on e-desktop mirrored to `fiona:~/eklow-wt-{attn,gemm,moe}`. Each gets `TASK.md` (and a copy of the harness's
  `AGENTS.md`) in its worktree, commits on its branch as son-of-anton-bot, never pushes, and writes `REPORT.md`; old
  tasks and reports go to `old-reports/` (all excluded from git by `.git/info/exclude`). The orchestrator reviews the
  diff, re-runs the gates, merges into main and keeps DECISIONS / STATUS. Local Qwen workers on son-of-anton were
  dropped so the rack stays quiet.
- Worktrees now: `gemm` (m6/prefill), `moe` (m6/stream), `attn-wide` (perf/attn-wide) and `chunk` (perf/chunk2048)
  are merged and clean; `attn` (the old flash-kernel knock-outs) was removed 2026-10-27.

## Milestone 1 (foundation)

- [x] flake + CMake (intel-llvm stdenv, AOT for DG2)
- [x] GGUF loader, identical to the gguf Python package on the 12B (DECISIONS 2026-10-24)
- [x] `eklow-probe`: memory, transfers, launch/sync, host-signal latency (DECISIONS 2026-10-24)
- [x] `eklow-xmx-bench`: dpas layouts verified, fp16 98.8 TFLOPS, s8 x s4 with Q4_0 scales 222.8 TOPS,
      VRAM read 399 GB/s (DECISIONS 2026-10-24)
- [x] Level Zero runtime in fiona's system config (`/etc/nixos` ee3be49b, deployed); the package's
      binaries find it through /run/opengl-driver

## Milestone 2 (correct forward on the 12B)

- [x] oracle: `eklow-oracle` (llama.cpp CPU at the fleet's ea0c4c2c, AVX2), tensor dumps and
      `tools/ref/compare_dumps.py`; CPU vs CUDA noise floor on a chat prompt: mean KL 4.7e-3,
      top-1 96% (DECISIONS 2026-10-24)
- [x] tokenizer (`src/text/tokenizer.cpp`): identical to Hugging Face's `tokenizers` on 27,607
      strings / 1.34M tokens, encode and decode; 1.5M tokens/s (DECISIONS 2026-10-24)
- [x] chat template: Google's Gemma 4 template ported to C++ (`src/text/chat_gemma4.cpp`),
      identical to transformers' `apply_chat_template` on 20,416 conversations and, tokenized with
      special tokens only in template text, on 3.82M tokens; request text cannot inject special
      tokens (DECISIONS 2026-10-24)
- [x] reference kernels (`src/kernels/ref.cpp`) and the reference forward (`src/model/gemma4.cpp`,
      `eklow-ref-run`): on a 25-token chat prompt top-1 25/25, mean KL 3.9e-3; on 2,616 tokens top-1
      31/32, mean KL 1.9e-2, the same band as llama.cpp CUDA against CPU (DECISIONS 2026-10-24)
- [x] streaming output parser (`src/text/output_parser.cpp`): reasoning, content and tool calls
      recovered exactly from 40,000 random turns rendered by the template (DECISIONS 2026-10-24)
- [ ] image input (bidirectional attention among an image's tokens on sliding layers): with M3

## Milestone 3 (serving)

- [x] the serving forward pass (`src/engine/engine.cpp`, `eklow-engine-run`): steps of prefill
      chunks and decode tokens from several slots, paged fp16 KV (sliding rings, global page
      tables); within the oracle's noise floor for one chunk, chunked prefill, decode and two slots
      at once (DECISIONS 2026-10-24)
- [x] `eklow-server`: OpenAI-compatible /v1/chat/completions (streamed or whole; reasoning_content,
      tool calls, thinking on unless turned off), /v1/models, /health; host sampler with the model
      card's defaults; scheduler with continuous batching and per-slot prefix reuse; matches what
      son-of-anton sends and reads (a survey of its client, DECISIONS 2026-10-25)
- [x] EmbeddingGemma 2 in the same process, /v1/embeddings on its own port: cosine 0.9996-0.99995
      against the llama.cpp server it replaces (reference kernels, 50 ms an input); beside a long
      prefill the endpoint holds the scheduler, 0.06-3.8 s an input at 135k tokens of depth
      (DECISIONS 2026-10-26, "embeddings beside a long prefill")
- [x] NixOS module (`nix/module.nix`, models pinned by revision and hash) and fiona's config,
      deployed 2026-10-26: 2 x 262,144 with int8 global KV by default, which maitred listed as
      `eklow/gemma-4-12b` (DECISIONS 2026-10-26); the 26B in its place from 2026-10-27
- [x] a son-of-anton-sized prompt (10.6k tokens) through `eklow-server` at 2 x 262k: first token
      8.7 s cold, 0.2 s with the prefix cached, then 27.4 tok/s (`tools/server/ttft.py`; targets
      5 s and 1 s)
- [ ] son-of-anton's Signal probe scenarios on eklow against llama.cpp
- [x] grammar-constrained tool calls (`src/serve/tool_grammar.cpp`, `eklow-grammar-test`)
- [ ] image input (llama.cpp's server on fiona had no projector either)

## Milestone 6 (Gemma 4 26B A4B), started 2026-10-26

- [x] the reference forward with the MoE layers, inside the 26B's noise band (8x the 12B's)
- [x] a routing profile: each layer's busiest 96 of 128 experts take ~98% of its selections
- [x] Q4_0 expert GEMVs on the CPU (AVX-VNNI): ~100 us an expert from DRAM
- [x] the engine runs it (all experts on the card, short contexts), on a slow path (routing on the
      host, one expert at a time): against the fp32 reference 249/256 top-1, mean KL 1.7e-2 on the
      long prompt's last 256 positions, where llama.cpp CPU is 220/256, 7.5e-2 (DECISIONS
      2026-10-26, "the 26B in the engine"); prefill 1,540 tok/s at 1k, decode 18.9 tok/s
- [x] decode MoE on the card (routing kernel, grouped expert GEMVs): 50.5 / 46.4 tok/s at 0 / 4k,
      41.4 / 38.1 per sequence with two (DECISIONS 2026-10-26, "the 26B's decode MoE on the card")
- [x] decode in fewer kernels (fused norms and layer tail, the dense gate|up in the grouped GEMV):
      68.2 / 61.9 tok/s at 0 / 4k, 53.9 / 48.3 per sequence with two (DECISIONS 2026-10-26, "the
      26B's decode step in fewer kernels")
- [x] grouped expert GEMMs for prefill: 3,130 / 2,821 tok/s at 1k / 4k, 2,525 at 4k depth (from
      1,014 / 1,003 / 970; DECISIONS 2026-10-26, "the 26B's prefill MoE on the card")
- [x] through the server: chat and tool calls at 2 x 8k; 2 x 32k does not fit with every expert on
      the card
- [x] experts placed by the routing profile with the rest on the CPU: N is picked from the card's free
      memory (51 of 128 per layer at 2 x 262k, 84 at 2 x 128k), chat, tool calls and embeddings through
      the server at 2 x 262k, two needle requests of 88k and 167k tokens found; real-text engine26 at
      N = 112 / 96 / 64: decode 63 / 61 / 53 tok/s, prefill 2,420 / 1,445 / 612 tok/s (DECISIONS
      2026-10-27, "the 26B with cold experts on the CPU")
- [x] the busiest cold experts streamed to the card for prefill (copy engine, 40 slots): real-text
      prefill at N = 51 / 64 / 96 from 516 / 609 / 1,441 to 1,734 / 1,931 / 2,526 tok/s; a 107k-token
      prompt's first token at 2 x 262k in 140 s (main 298 s); decode unchanged (DECISIONS
      2026-10-27, "streaming the busy cold experts")
- [x] in production (2026-10-27): fiona serves the 26B at 2 x 262k (above)
- [ ] the tool-call eval against the 12B (the owner's harness, in `/home/e-work`)

## Milestone 4 (speed), started early

- [x] XMX matmul for Q4_0 (`src/kernels/xmx.cpp`): weights repacked to dpas tiles at load,
      activations quantized to int8 per block; exact against a CPU emulation of the quantized
      arithmetic. GEMV 110-240 GB/s (decode), column groups ~11 TOPS (64-256 rows), tile GEMM 28-53
      TOPS (2048 rows) (`eklow-kernel-test`, docs/bench/kernels-fiona-2026-10-24.json)
- [x] output head in an aligned Q6 layout: 3.4 ms of a decode step (was 69)
- [x] flash attention on the matrix units with paged dpas-tiled KV (`src/kernels/attention.cpp`,
      `eklow-attn-test`): decode 84-314 us per layer, prefill 6-9 TFLOPS (DECISIONS 2026-10-24)
- [x] `eklow-engine-bench`: prefill 1,509 / 1,413 / 1,194 tok/s over 1k / 4k / 16k prompts, 1,287
      / 962 tok/s at 4k / 16k depth; decode 32.1 / 29.0 / 27.6 tok/s at 0 / 4k / 16k
      (docs/bench/engine-fiona-2026-10-24.json); `--profile` times each section on the card
- [x] decode: stacked QKV and gate/up matmuls, fused layer tails (about 16 launches a layer, from
      34): decode 32.2 / 29.1 / 27.6 tok/s at 0 / 4k / 16k
- [x] attention: prefill work-groups sharing each K/V block through shared local memory: 10.4 /
      10.9 TFLOPS (sliding / global)
- [x] global prefill attention in key segments, each two GEMMs in the fp16 GEMM's shape (Q.K^T with the softmax in its
      epilogue, P.V into a running output): ~31 TFLOPS from ~10 at 16k-131k keys; the 26B's engine prefill at 128k
      depth 396 -> 787 tok/s, through the server 493 -> 1,040 at 128k (DECISIONS 2026-10-27)
- [ ] prefill at 16k depth: 962 tok/s of the 1,500 target (`eklow-engine-bench --depth 16384`);
      needs global attention near 20 TFLOPS and the GEMMs near 60 (DECISIONS 2026-10-24); the
      GEMM waits on its shared-memory loads, not their bandwidth (knockouts, DECISIONS 2026-10-26)
- [x] fp16 GEMM through shared local memory for prefill: 48-52 TFLOPS in isolation; prefill 1,364 /
      1,243 / 993 tok/s at 1k / 4k / 16k (DECISIONS 2026-10-24, "fp16 GEMM")
- [ ] decode GEMV toward the 399 GB/s the card reads (now 160-290 GB/s by shape at one row)
- [x] int8 global KV (`--kv-int8`): 2 x 262k in ~4.8 GB, within the noise floor; the server's
      default (`--kv-fp16` for fp16)
- [x] 2 x 262k end to end: 12.5 GiB on the card with the embedding model; two 259k-token prompts
      at once prefilled in 36 min (237 tok/s together, from empty), both needles recalled; decode
      ~11 tok/s per sequence at 259k (DECISIONS 2026-10-26)
- [x] re-measured at 3e37ce5 (docs/bench/engine-*-fiona-2026-10-25.json): prefill 1,507 / 1,410 /
      1,193 tok/s, 1,281 / 965 at 4k / 16k depth; decode 31.9 / 29.5 / 28.0 tok/s, two slots 29.5
      / 26.8 / 25.4 per sequence; with int8 KV 31.6 / 26.8 at 0 / 16k
- [x] decode attention beyond 4k keys: splits of at most 8 blocks and a two-level merge; global8 at
      16k 470 -> 302 us a layer, int8 decode at 16k 27.8 tok/s, 24.8 at 64k (DECISIONS 2026-10-26)
- [x] decode GEMVs in one kernel (K split inside a work-group, threads from the shape): decode
      +2-3% with one slot, +3-4% with two; GEMM: B tiles loaded early and a 128 x 128 tile for the
      narrow matrices at 1,024 rows, prefill +1-1.5% (DECISIONS 2026-10-26)
- [x] weight tiles column-group major, four tiles a load in the one-row GEMV: decode +9%; sliding
      decode attention in 32-dimension slices: +0.6-2.1% (DECISIONS 2026-10-26)

## Open, in order

1. **Global prefill attention at depth** (head 512): now in segments as two GEMMs, ~31 TFLOPS from ~10 (DECISIONS
   2026-10-27, "global prefill attention as two GEMMs"; in main, not yet deployed). Still 55% of prefill at 128k. P.V
   (~29 TFLOPS) is the slower half and no single piece of it holds it (knock-outs give back at most 5% each); 4,096-key
   segments gave +5% for 64 MB more. The sliding layers' prefill attention (FlashGroup, ~10 TFLOPS) is 4-6% of prefill.
2. **The tool-call eval, 26B against the 12B** (the owner's). The owner's prior is that the 26B is better; nothing
   here measures model quality, only fidelity to each model.
3. **Prefill chunks of 2,048 tokens**: done (main, the server's default; DECISIONS 2026-10-27): +14% prefill at 32k
   through the server for N = 46 against 50, decode -1%, GSM8K two at a time -4%.
4. **A routing profile from real traffic**: today's comes from four recordings (eklow's notes, two Python sources,
   a chat prompt). An export of live routing counts would let it follow what son-of-anton actually sends.
5. **Decode**: the grouped expert GEMVs read at ~180 GB/s of 399; with two streams at N = 51 the CPU's cold pairs
   look like the limit.
6. **Image input** (the `image-input` branch for the 12B; the 26B's vision encoder is not downloaded).
7. son-of-anton's Signal probe scenarios on eklow (M3).
8. Host-to-GPU signals must use `zeMemAllocHost` with the uncached bias; `sycl::malloc_host` is never seen by a
   running kernel under xe.
