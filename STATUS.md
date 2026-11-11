# STATUS

Live state, written as a handoff. Design and history: `DESIGN.md` (the original target) and `DECISIONS.md` (every decision
with its measurements); what eklow is and its numbers: `README.md`.

## Now (2026-11-11)

- **fiona runs eklow's milestone in production** (`eklow-bolt.service` since 2026-11-10 22:48): Nemotron-3.5-Bolt-E4B with
  NVIDIA's DFlash draft and Gemma 4's projector, EmbeddingGemma 2 with its projector, vega's end-of-turn checker (the
  vega-cv32 policy) and Qwen3-Reranker, in one `eklow-server` on the A770M, every cache fp16 (DECISIONS 2026-11-11, "fiona's
  milestone").
  - Card memory: 12.7 GiB, with 3.2 GiB free.
  - Bolt decodes at 90.8 / 87.9 / 64.6 tok/s at 8k / 32k / 60k with the draft (44.8 / 39.3 / 35.6 plain), and prefills
    at 3,549 / 2,920 / 2,273 tok/s.
  - Side requests take 70-430 ms while Bolt streams.
- **The Qwen3.6 chat path is removed** (branch `cleanup/qwen-chat`, DECISIONS 2026-11-11, "the Qwen3.6 chat path leaves
  eklow"; the owner's decision). Stock Qwen3.6, Ornith-1.5 and Sage-A1.5 served from 2026-11-02 to 2026-11-10; `main` at
  32152a6 is the last commit with them. Their GGUFs stay on Hugging Face (`ethanwtodd/Qwen3.6-35B-A3B-Q4_K-Q8_0-GGUF`,
  `Ornith-1.5-...`, `Sage-A1.5-...`, `Qwen3.6-35B-A3B-DFlash2-Q8_0-GGUF`).
  - Its host checks pass. The draft's reference forward is byte-identical before and after.
  - The GPU regression waits for a pause of the service.
- **Sampling is llama.cpp's** (DECISIONS 2026-11-11): its chain, defaults, GGUF keys and request fields, bit for bit
  against the pinned libllama on fixed logits. The owner accepts that a token at min-p's edge can be kept in one batch shape
  and dropped in another (int8 activations).
- **`preserve_thinking` is on for Bolt** (the owner's decision). The turn after a tool loop prefills 0.1-2.5k tokens instead
  of the whole loop, and the 64K window fills 1.2-1.7x faster. Its effect on answers is the owner's test, once the
  replay set carries the stored reasoning.
- **`GET /metrics` on Bolt's server** (DECISIONS 2026-11-11), for llmtop. It is checked against llmtop's parser, not yet on
  the card.

## Open, in order

1. **The GPU regression of `cleanup/qwen-chat`** in a pause of `eklow-bolt` (fiona `~/bolt-ref/regress.sh`, ~15 minutes):
   - the kernel tests;
   - the draft's lattices and the embeddings, against main's build;
   - the checker's golden states and its 29 turns;
   - the reranker against llama.cpp f16;
   - `serve_test.py basic` and decode at 8k.
2. **Text-to-speech and speech-to-text** on the same card, beside Bolt (planned).
3. **The checker's cut-offs** sit where its fitted features put them. Offtopic's cut-off lies inside the mass of its
   answers, so any implementation flips some of its flags (DECISIONS "vega's checker").
4. **Dead attention cases** of the Gemma 4 26B and 12B (head 512 groups 8 and 16, head 256 group 2), still in `attend` and
   `eklow-attn-test`.
5. **fp16 activations** for the small-batch matmuls, only if the min-p edge turns out to matter.

## Where things are

- Source of truth: `github.com/ewtodd/eklow`. Work happens on topic branches, merged into `main`.
- Toolchain: the fleet's nixpkgs (c3f15ff4): intel-llvm 7.1.1, level-zero 1.32.0, intel-compute-runtime 26.31. The oracle is
  llama.cpp at `ea0c4c2c` (`flake.nix`); the model files are flake packages (`nix/models.nix`).
- fiona: the A770M on PCIe 4.0 x8, Iris Xe (titles, llama.cpp), i7-12700H, 30 GB of RAM. Its eklow service is production:
  no server or GPU test may run on the card while any `eklow*` unit is active. `gpu.lock` does not cover the service. A
  test server beside it on 2026-11-11 slowed a real chat turn to 0.9 tok/s for six minutes.
- The private replay set (real chats, graded points, the checker's 29 turns) stays on e-workstation and fiona, never in
  the repository.

## How to work on it

- Build anywhere with Nix (`nix develop path:. -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'`). The host-only
  tests run there: `eklow-spec-test`, `eklow-sampler-test`, `eklow-metrics-test`, `eklow-keepalive-test`,
  `eklow-grammar-test-gemma4 --model BOLT.gguf`, and `tools/dflash_test/host-build.sh` (the draft's reference forward on
  the CPU).
- Run on fiona: rsync the tree without `.git` and `build`, build there, and use the card only in a pause of the service,
  under `flock ~/eklow-perf/gpu.lock`.
- Gates:
  - Bolt's: the engine against transformers (`eklow-engine-run-gemma4`, `tools/ref/bolt_hf_ref.py`, `kl_positions.py`:
    top-1 >= 250/256 and mean KL <= 2e-3 over a long prompt's last 256 positions, <= 5e-3 over decoded ones);
  - `eklow-attn-test`, `eklow-qfmt-test --check`, `eklow-dflash-test dump` against the previous build;
  - `tools/server/serve_test.py basic`;
  - the template and parser against Hugging Face (`tools/ref/gemma4_chat_compare.py`, `gemma4_parse_compare.py`);
  - the side models' own gates: `eklow-vega check`, `tools/rerank/rerank_compare.py gate`, `tools/bench/embed_bench.py`.
