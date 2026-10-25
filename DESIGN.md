# eklow — design

A serving engine for **Gemma 4** on one Intel Arc A770M (DG2, 16 GB) in `fiona`, an Intel NUC with
an i7-12700H. C++20 + SYCL on Level Zero (nixpkgs `intel-llvm` 7.1.1), kernels compiled ahead of
time for DG2, packaged with nix. Named for Eklow Labs (Silicon Valley, season 5).

`STATUS.md` holds the live state and measured numbers; `DECISIONS.md` holds dated decisions with
their evidence. This file is the target and the shape of the system.

## Why

llama.cpp's Vulkan backend gives the 12B about 100-250 tok/s of prefill and ~22 tok/s of decode on
this card, against ceilings of roughly 2000+ and ~60. On fiona it also cannot finish a prompt past
~12k tokens: xe kills any GPU job over 5 s, and under i915 the same jobs silently produced garbage
(DECISIONS 2026-10-24). son-of-anton's first reply on a cold cache took about two minutes.

## Requirements (from the owner, 2026-10-24)

| | requirement |
|---|---|
| model | Gemma 4 12B first (Google's QAT Q4_0 GGUF), then Gemma 4 26B A4B; keep whichever wins the son-of-anton tool-call eval |
| priority | Gemma using son-of-anton's tools correctly, especially over Signal |
| prefill | matters more than decode |
| decode | >= 20 tok/s per stream |
| context | Gemma's full 262k, ideally for 2 sequences (the global layers' shared K=V vector stored once) |
| escalation | a flagged turn goes to a stronger **local** model through maitred, never a hosted API |
| toolchain | nixpkgs `intel-llvm` (the fleet's nixpkgs), Level Zero, no oneMKL/oneDNN at runtime |
| packaging | nix flake (`lib.mkEklow`) + NixOS module in `/etc/nixos` |

## Shape

- **One process owns the A770M** and serves Gemma 4 chat, EmbeddingGemma 2 and, later, a small
  tool-call decision model, under one scheduler. Embedding batches and checks fit into the same
  steps as chat prefill and decode, so separate processes no longer fight over the GPU.
- **Little-Titles stays on llama.cpp** on the Iris Xe iGPU (done, `/etc/nixos` 21b7c9c8). Its
  Qwen3.5 Gated-DeltaNet hybrid is not worth porting, and the 26B needs the VRAM.
- **Kernels are our own:** ESIMD `dpas` (XMX) GEMMs and attention, each with a reference kernel and
  a test. Compiled ahead of time for DG2 (`-fsycl-targets=intel_gpu_dg2_g10`), so nothing is
  compiled at startup.
- **Weights load straight from Google's QAT GGUFs.** Q4_0 is the format the model was trained
  for; the engine only rearranges it for XMX at upload.
- **Oracle:** llama.cpp's CPU backend on the same GGUF, with per-layer comparisons and KL/top-1
  checks on full logits.

## Model: Gemma 4 12B (from the GGUF, `eklow-gguf`)

48 layers, hidden 3840, FFN 15360 (GeGLU), 16 query heads, vocab 262144 (token embedding Q6_K,
tied output), final logit softcap 30, RMSNorm eps 1e-6, context 262144.

- **Sliding-window layers** (5 of every 6, window 1024): 16 query heads and 8 K/V heads of 256,
  RoPE base 10000 over all 256 dims.
- **Global layers** (every 6th: 5, 11, ..., 47): 16 query heads of 512 and one 512-wide K head
  with no `attn_v` — K and V are the same tensor. RoPE base 1e6 over the 512-dim head, scaled per
  frequency by the 256-entry `rope_freqs` tensor (Gemma 4's proportional RoPE).
- q/k norms per head, pre/post norms around attention and FFN, a scalar `layer_output_scale` per
  layer.

KV per token: sliding layers 40 x 2 x 8 x 256 x 2 B = 320 KiB but only for the last 1024 tokens
(a ring); global layers 8 x 512 x 2 B = 8 KiB (K=V once). A 262k sequence's global cache is about
2 GiB in fp16, which is what makes 2 x 262k plausible next to ~6.5 GiB of weights.

## Targets

| | now (llama.cpp Vulkan) | target |
|---|---|---|
| 12B prefill at 16k depth | ~110-150 tok/s | >= 1500 tok/s (about half of the fp16 XMX peak) |
| 26B prefill | ~200 tok/s at 4k | above the 12B |
| decode, one stream | ~22 tok/s | floor 20; expect 40-50 on the 12B |
| son-of-anton first reply, cold | ~2 min | <= 5 s; <= 1 s with the prefix cached |
| context | 2 x 32k | 2 x 262k (12B) |
| malformed tool calls | unknown | zero, by construction |

The 26B can plausibly prefill faster than the 12B: each token costs about 6 GFLOP of weights (3B
active) against the 12B's 22, or about 9 against 25 counting attention at 16k. Experts that do not
fit on the card stream over PCIe during prefill (1.5-2 GiB per 4096-token chunk is ~150 ms at the
measured 11.7 GB/s, hidden behind ~1 s of compute) and run on the CPU (AVX-VNNI) during decode.
M1 measures the card; these stay estimates until then.

## Milestones

- **M0, eval and prompt diet (son-of-anton, no engine).** Done: the Signal probe harness, the
  prompt diet (prefix 15.3k -> 10.4k tokens), and a decision-model bake-off and A/B (DECISIONS
  2026-10-24).
- **M1, foundation.** Level Zero runtime in fiona's config; `eklow-probe` (memory, transfers,
  launch/sync, host-signal latency); XMX GEMM microbenchmarks (fp16 and Q4_0 x int8 `dpas`), whose
  results set the real targets; the GGUF loader.
- **M2, correct forward pass on the 12B.** Reference kernels and per-layer comparison against the
  oracle; every Gemma 4 detail above; for image input, bidirectional attention among an image's
  tokens on the sliding layers; the tokenizer, checked against Hugging Face's; Google's chat
  template ported to C++, byte-identical to Hugging Face's rendering and refusing any GGUF that
  carries another template (DECISIONS 2026-10-24); a streaming parser for the thinking channel and
  tool calls.
- **M3, serving.** Continuous batching with chunked prefill; a paged global KV cache plus
  per-sequence sliding-window rings; a prefix cache whose window snapshots spill to host RAM;
  OpenAI chat with tools and streaming, `/v1/embeddings`, image and audio input (the 12B's vision
  and audio inputs are linear projectors); grammar-constrained tool calls. Replaces llama.cpp for
  chat and embeddings on fiona; maitred's upstreams stay on :8081 and :8082.
- **M4, speed.** XMX prefill GEMMs, flash attention for prefill, decode GEMVs, SYCL graphs to cut
  launch overhead.
- **M5, the decision model**, driven by the eval.
- **M6, the 26B A4B.** Grouped Q4_0 MoE kernels and the router; a placement planner that keeps hot
  experts on the card from a routing profile; cold experts streamed during prefill and run on the
  CPU during decode, with host-signaled events so the GPU never waits on a host round trip it
  could avoid; the 1.2 GB vision encoder in host RAM, paged in when an image arrives. Head-to-head
  with the 12B on the eval; keep the winner.
- **M7, optional.** Speculative decoding with the MTP head done right, plus n-gram drafts.

## Tool calls and the decision model

1. **Grammar, free.** Inside `<|tool_call>...<tool_call|>`, decoding is constrained by a grammar
   built from that request's tool schemas: only declared tool names and keys, required keys
   present, types and enums honored, strings closed. Broken JSON becomes impossible, and the
   parser never lets call syntax leak into a Signal reply.
2. **The learned check**, at two points: a completed call, and a turn that ends without a call. It
   sees a compact view of the conversation and cheap signals from Gemma itself (agreement among N
   candidate calls sampled in one batched decode, which costs about one decode; confidence at the
   decision tokens). The M0 bake-off found off-the-shelf decision models too weak to pay for
   themselves (DECISIONS 2026-10-24), so this is a model fine-tuned on son-of-anton's labelled
   steps, chosen by the eval. Budget: <= ~1 GB VRAM and <= ~100 ms per check.
3. **Loop guard:** a call identical to the previous one is flagged deterministically.

Policy on a flagged step: pick a better candidate or redraw (with a call forced or forbidden), up to
two retries; still flagged, mark the response so maitred can rerun the turn on a stronger local
model.

## Operations

- Development happens on fiona (`~/eklow`, an rsync of this repo without `.git`); the owner
  deploys. Development sessions may stop `llama-server-gemma-4-12b`; titles and embeddings stay up.
- Every fleet config change is build-checked and handed over as a `colmena apply --on fiona`
  command.
