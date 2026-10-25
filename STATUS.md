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
nix develop -c python3 tools/ref/compare_dumps.py REF_DIR TEST_DIR
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
- [ ] tokenizer, checked against Hugging Face's
- [ ] chat template (the official Jinja), golden tests
- [ ] reference kernels and the reference forward, compared per layer against the oracle

## Open

1. Host-to-GPU signals must use `zeMemAllocHost` with the uncached bias; `sycl::malloc_host` is
   never seen by a running kernel under xe.
2. Kernels that run long while llama.cpp shares the card have not been tried against xe's job
   timeout.
