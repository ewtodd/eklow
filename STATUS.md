# STATUS

Live state. Design and targets: `DESIGN.md`; history and evidence: `DECISIONS.md`.

## Where things are

- Source of truth: `github.com/ewtodd/eklow` (working copy `~/Software/eklow` on e-desktop); the
  build/run tree on fiona is `~/eklow` (rsync, no `.git`).
- Image input in progress on the branch `image-input` (picked up from fiona's `~/eklow-dev`, where
  it was written): image preparation, the gemma4uv embedder on reference kernels, the oracle
  through mtmd; not yet checked against the oracle or wired into the engine.
- Weights: `fiona:~/models/gemma-4-12b-it-qat-q4_0.gguf` (sha256 93567e57..., the file fiona's
  llama.cpp serves from `/var/cache/llama-cache`, which the fiona user cannot read).
- Toolchain: the fleet's nixpkgs (c3f15ff4): intel-llvm 7.1.1, level-zero 1.32.0,
  intel-compute-runtime 26.31.39395.13.
- fiona's GPUs: `level_zero:0` A770M (eklow), `level_zero:1` Iris Xe (little-titles on llama.cpp
  Vulkan). The `eklow` service holds the A770M; development sessions stop it (`systemctl stop
  eklow` as the fiona user, which polkit allows) and start it again after.

## Build / run (on fiona)

```
rsync -a --delete --exclude .git --exclude build --exclude 'result*' ~/Software/eklow/ fiona:eklow/   # from e-desktop
cd ~/eklow
nix develop path:. -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'
nix develop path:. -c ./build/eklow-probe --json probe.json
nix develop path:. -c ./build/eklow-gguf ~/models/gemma-4-12b-it-qat-q4_0.gguf --tensors
nix develop path:. -c ./build/eklow-xmx-bench --json xmx.json
```

The oracle is CPU-only and runs anywhere (fiona does not accept unsigned store paths from
e-desktop, so it runs on e-desktop):

```
nix build .#oracle
./result/bin/eklow-oracle --model gemma-4-12b-it-qat-q4_0.gguf --out DIR --prompt-file chat.txt
scp DIR/tokens.json fiona:run/                                   # the same tokens for eklow
nix develop path:. -c ./build/eklow-ref-run --model ~/models/gemma-4-12b-it-qat-q4_0.gguf \
    --out ~/run --tokens-json ~/run/tokens.json                  # on fiona, Gemma's server stopped
nix develop -c python3 tools/ref/compare_dumps.py DIR RUN_DIR
```

Speed work (2026-10-26): `~/eklow-perf` on fiona (copies in `~/Software/eklow-perf` on
e-desktop) holds `bench.sh` (a tree's engine bench, profile, kernel test or attention test, under
`~/eklow-perf/gpu.lock`), `check.sh` (logits against llama.cpp CPU dumps of the 2,616- and
25-token prompts in `~/eklow-oracle`; baseline 31/32 at mean KL 2.4e-2, 25/25 at 4.2e-3) and the
`AGENTS.md` the delegated runs read. Each run works in a worktree (`~/Software/eklow-wt/*` on
e-desktop, `~/eklow-wt-*` on fiona).

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
      deployed 2026-10-26: 2 x 262,144 with int8 global KV by default, which maitred lists as
      `eklow/gemma-4-12b` (DECISIONS 2026-10-26)
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
- [ ] the engine runs it (all experts on the card, short contexts), then MoE kernels, then experts
      placed by the profile with the rest on the CPU; the tool-call eval against the 12B

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

## Open

1. Host-to-GPU signals must use `zeMemAllocHost` with the uncached bias; `sycl::malloc_host` is
   never seen by a running kernel under xe.
2. Kernels that run long while llama.cpp shares the card have not been tried against xe's job
   timeout.
