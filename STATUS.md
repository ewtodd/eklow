# STATUS

Live state. Design and targets: `DESIGN.md`; history and evidence: `DECISIONS.md`.

## Where things are

- Source of truth: `github.com/ewtodd/eklow` (working copy `~/Software/eklow` on e-desktop); the
  build/run tree on fiona is `~/eklow` (rsync, no `.git`).
- Weights: `fiona:~/models/gemma-4-12b-it-qat-q4_0.gguf` (sha256 93567e57..., the file fiona's
  llama.cpp serves from `/var/cache/llama-cache`, which the fiona user cannot read).
- Toolchain: the fleet's nixpkgs (c3f15ff4): intel-llvm 7.1.1, level-zero 1.32.0,
  intel-compute-runtime 26.31.39395.13.
- fiona's GPUs: `level_zero:0` A770M (eklow), `level_zero:1` Iris Xe (little-titles on llama.cpp
  Vulkan). Development sessions stop `llama-server-gemma-4-12b`; embeddings stay up.

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
- [ ] sampler, OpenAI-compatible server (streaming, tools, thinking), scheduler with continuous
      batching, prefix cache, grammar-constrained tool calls
- [ ] EmbeddingGemma 2 in the same process
- [ ] NixOS module and fiona's config

## Milestone 4 (speed), started early

- [x] XMX matmul for Q4_0 (`src/kernels/xmx.cpp`): weights repacked to dpas tiles at load,
      activations quantized to int8 per block; exact against a CPU emulation of the quantized
      arithmetic. GEMV 110-240 GB/s (decode), column groups ~11 TOPS (64-256 rows), tile GEMM 28-53
      TOPS (2048 rows) (`eklow-kernel-test`, docs/bench/kernels-fiona-2026-10-24.json)
- [x] output head in an aligned Q6 layout: 3.4 ms of a decode step (was 69)
- [x] flash attention on the matrix units with paged dpas-tiled KV (`src/kernels/attention.cpp`,
      `eklow-attn-test`): decode 84-314 us per layer, prefill 6-9 TFLOPS (DECISIONS 2026-10-24)
- [x] `eklow-engine-bench`: prefill 1,141 / 1,057 / 867 tok/s at 1k / 4k / 16k, decode 22.2 /
      20.8 / 19.4 tok/s at 0 / 4k / 16k (docs/bench/engine-fiona-2026-10-24.json); `--profile`
      times each section of the forward pass on the card
- [x] decode: stacked QKV and gate/up matmuls, fused layer tails (about 16 launches a layer, from
      34): decode 32.2 / 29.1 / 27.6 tok/s at 0 / 4k / 16k
- [ ] attention: K/V blocks staged through shared local memory, shared by several query tiles
- [x] fp16 GEMM through shared local memory for prefill: 48-52 TFLOPS in isolation; prefill 1,364 /
      1,243 / 993 tok/s at 1k / 4k / 16k (DECISIONS 2026-10-24, "fp16 GEMM")
- [ ] decode GEMV toward the 399 GB/s the card reads
- [ ] global KV for 2 x 262k: 8.6 GB in fp16 does not fit beside the weights

## Open

1. Host-to-GPU signals must use `zeMemAllocHost` with the uncached bias; `sycl::malloc_host` is
   never seen by a running kernel under xe.
2. Kernels that run long while llama.cpp shares the card have not been tried against xe's job
   timeout.
