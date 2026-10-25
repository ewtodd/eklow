# STATUS

Live state. Design and targets: `DESIGN.md`; history and evidence: `DECISIONS.md`.

## Where things are

- Source of truth: this repo (`~/Software/eklow` on e-desktop); the build/run tree on fiona is
  `~/eklow` (rsync, no `.git`).
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
```

## Milestone 1 (foundation)

- [x] flake + CMake (intel-llvm stdenv, AOT for DG2)
- [x] GGUF loader, identical to the gguf Python package on the 12B (DECISIONS 2026-10-24)
- [x] `eklow-probe`: memory, transfers, launch/sync, host-signal latency (DECISIONS 2026-10-24)
- [ ] Level Zero runtime in fiona's system config (intel-compute-runtime)
- [ ] XMX microbenchmarks: fp16 and Q4_0 x int8 `dpas` peak and memory-fed GEMM

## Open

1. Host-to-GPU signals must use `zeMemAllocHost` with the uncached bias; `sycl::malloc_host` is
   never seen by a running kernel under xe.
2. Kernels that run long while llama.cpp shares the card have not been tried against xe's job
   timeout.
