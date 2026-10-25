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
- `eklow-gguf FILE [--tensors] [--json]`: what the GGUF loader reads from a file;
  `tools/ref/gguf_compare.py FILE JSON` checks it against the gguf Python package.

## Build

```
nix build            # AOT for DG2 (intel_gpu_dg2_g10)
nix develop -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'
```

The dev shell puts the Level Zero and OpenCL drivers on the loader path, so it runs on a host
whose system config does not carry intel-compute-runtime.

## License

Apache-2.0.
