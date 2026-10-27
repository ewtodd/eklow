# eklow

A serving engine for Gemma 4 and EmbeddingGemma 2 on one Intel Arc A770M (16 GB), in C++20 and
SYCL/ESIMD on Level Zero. It serves OpenAI-compatible chat (streaming, thinking, and tool calls
constrained by a grammar so they cannot come out malformed) and embeddings. It runs fiona's chat
and embedding service in place of llama.cpp; from 2026-10-27 that is Gemma 4 26B A4B at 2 x 262k,
the 12B before. Image input is not in yet (branch `image-input`).

`DESIGN.md` has the targets, `STATUS.md` where things stand, and `DECISIONS.md` the measurements
behind each choice.

## Numbers

fiona: an A770M on PCIe 4.0 x8, an i7-12700H and 32 GB RAM. Main at 71bfb75 (2026-10-27),
`eklow-engine-bench` with 1,024-token prefill chunks; one sequence, tok/s. The bench keeps the
global layers' KV in fp16; the server's int8 (what lets 2 x 262k fit) decodes a few percent slower
at depth.

| | prefill, 1k prompt | prefill at 4k depth | decode at 0 / 4k context |
|---|---|---|---|
| 12B | 1,516 | 1,290 | 34.8 / 32.1 |
| 26B A4B, every expert on the card (up to ~2 x 32k) | 3,130 | 2,525 | 68.1 / 61.4 |
| 26B A4B, 51 of 128 experts a layer on the card (what 2 x 262k leaves room for) | 1,734 | 1,644 | ~47 / ~47 |

Deeper in:
- **12B:** prefill ~970 tok/s at 16k depth; decode 30.0 tok/s at 16k.
- **26B, every expert on the card:** prefill 1,234 tok/s at 32k depth, where global attention is
  61% of the time; decode 43.4 tok/s at 32k.
- **26B through the server at 2 x 262k:** a 107k-token prompt reaches its first token in 140 s.
- **Two sequences at once:** the 26B, every expert on the card, decodes 53.9 tok/s each at 0
  context.

llama.cpp's Vulkan backend on the same card had given the 12B ~110-150 tok/s of prefill at 16k
depth and ~22 tok/s of decode, with contexts of 2 x 32k.

Correctness is checked logit for logit against references:
- **12B:** against llama.cpp's CPU backend, the long test prompt's top-1 matches on 31 of 32
  positions, mean KL 1.3e-2.
- **26B:** its routing makes any two implementations drift apart, so it is checked against eklow's
  own fp32 reference forward: 242/256 top-1 at KL 2.8e-2, where llama.cpp's CPU backend scores
  220/256 at 7.5e-2.

## How the 26B fits

Its weights are 13.4 GiB, 12 of them experts (128 a layer, 8 used per token). At 2 x 262k the
global layers' int8 KV takes another 6.4 GiB, so not every expert can stay on the card:
- **On the card:** each layer's busiest experts, by a routing profile
  (`data/gemma-4-26b-a4b-experts.json`, installed to `share/eklow`). The server places as many as
  the card's free memory allows.
- **Decode:** the rest run on the CPU (AVX-VNNI) while the card computes the hot ones.
- **Prefill:** the busiest cold experts of each chunk stream over PCIe on the copy engine and run
  in the card's grouped GEMMs; the CPU takes the remainder.

## Running

```
eklow-server --model gemma-4-26b-a4b-it-qat-q4_0.gguf --alias gemma-4-26b-a4b \
  --slots 2 --ctx 262144 --kv-int8 \
  --expert-profile data/gemma-4-26b-a4b-experts.json \
  --embed-model embeddinggemma-2-Q8_0.gguf --embed-port 8082
```

Chat, `/v1/models` and `/health` are on `--port` (8081); embeddings on `--embed-port`. The package
installs the profile as `share/eklow/gemma-4-26b-a4b-experts.json`. On NixOS,
the flake's module (`nixosModules.default`) runs it as `services.eklow`: `model`, `alias`,
`contextSize`, `slots`, `kvInt8`, the embedding model, and `extraArgs` for the expert profile.

## Tools

The server and the engine:
- `eklow-server`: the OpenAI-compatible server (above).
- `eklow-engine-run --model GGUF --out DIR --tokens-json FILE`: the serving engine over one prompt,
  dumped in the oracle's format for `tools/ref/compare_dumps.py`.
- `eklow-engine-bench --model GGUF [--prefill ...] [--depth ...] [--decode-ctx ...] [--slots N]
  [--profile]`: the engine's speed, and with `--profile` the card's time per section.
- `eklow-embed --model GGUF`: EmbeddingGemma 2, one input per line of stdin.

Correctness and kernels:
- `eklow-ref-run --model GGUF --out DIR --tokens-json FILE`: the reference forward pass (fp32
  activations). `eklow-oracle` (`nix build .#oracle`) is llama.cpp's CPU backend in the same
  format, and `tools/ref/compare_dumps.py` compares any two layer by layer.
- `eklow-kernel-test --model GGUF [--moe-model GGUF]`: the XMX matmuls, including the 26B's
  grouped expert kernels, against the reference kernels and an exact CPU emulation.
- `eklow-attn-test`: the paged cache attention against `ref::attention`, and its speed.
- `eklow-cpu-gemv-test --model GGUF`: the CPU's Q4_0 expert GEMVs, checked and timed.
- `eklow-copy-bench --model GGUF`: host-to-card copy rates for an expert, and their cost to kernels
  running beside them.

The card and the model file:
- `eklow-probe [--json FILE] [--skip-alloc]`: device facts, allocatable memory, copy rates, launch
  and host-signal latency.
- `eklow-xmx-bench [--json FILE]`: `dpas` layouts checked against the CPU, XMX throughput, VRAM
  bandwidth.
- `eklow-gguf FILE [--tensors] [--json]`: what the GGUF loader reads; `tools/ref/gguf_compare.py`
  checks it against the gguf Python package.

Text:
- `eklow-tok GGUF encode|decode|bench`: the tokenizer; `tools/ref/tok_compare.py` checks it against
  Hugging Face's `tokenizers`.
- `eklow-chat render|check`: the Gemma 4 chat template port; `tools/ref/chat_compare.py` checks it
  against transformers' `apply_chat_template`.
- `eklow-parse GGUF`: the output parser; `tools/ref/parse_compare.py` round-trips random assistant
  turns through the template and back.
- `eklow-grammar-test --model GGUF`: the tool-call grammar against the output parser.

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
