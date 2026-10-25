# eklow

A serving engine for Gemma 4 on one Intel Arc A770M, in C++20 and SYCL on Level Zero.
Early: milestone 1 (foundation). See `DESIGN.md` for the target, `STATUS.md` for where things
stand and `DECISIONS.md` for the evidence behind each choice.

## Tools

- `eklow-probe [--json FILE] [--skip-alloc]`: device facts, allocatable memory, host<->device copy
  rates, kernel launch latency, and host-signal latency, on the discrete Level Zero GPU with the
  most memory (or `ONEAPI_DEVICE_SELECTOR=level_zero:N`).
- `eklow-xmx-bench [--json FILE]`: `dpas` layouts checked against the CPU, XMX throughput per
  operand type, the Q4_0 inner loops, and VRAM read bandwidth.
- `eklow-tok GGUF encode|decode|bench`: the tokenizer; `tools/ref/tok_compare.py` checks it against
  Hugging Face's `tokenizers` on Google's `tokenizer.json`.
- `eklow-chat render [--tokens GGUF]` / `eklow-chat check GGUF`: the Gemma 4 chat template port;
  `tools/ref/chat_compare.py` checks it against transformers' `apply_chat_template`.
- `eklow-ref-run --model GGUF --out DIR --tokens-json FILE`: the reference forward pass, dumped in
  the oracle's format; `eklow-oracle` (`nix build .#oracle`) is llama.cpp's CPU backend, and
  `tools/ref/compare_dumps.py` compares the two layer by layer.
- `eklow-gguf FILE [--tensors] [--json]`: what the GGUF loader reads from a file;
  `tools/ref/gguf_compare.py FILE JSON` checks it against the gguf Python package.

## Build

```
nix build            # AOT for DG2 (intel_gpu_dg2_g10)
nix develop -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'
```

Installed binaries use the system's Level Zero driver from `/run/opengl-driver` (on NixOS,
`hardware.graphics.extraPackages = [ pkgs.intel-compute-runtime ]`). The dev shell carries its own
copy of the drivers, so it runs on any host.

## License

Apache-2.0.
