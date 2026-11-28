# STATUS

Live state, written as a handoff. Design and history: `DESIGN.md` (the original target) and `DECISIONS.md` (every decision
with its measurements); what eklow is and its numbers: `README.md`.

## Now (2026-11-26)

- **Inkling 2 Mini replaces Bolt** (the owner's decision, DECISIONS 2026-11-24): the port went milestone by milestone
  (M0 ground truth, M1 host reference and text layer, M2 the forward on the card, M3 memory, M4 the server, speed, M5
  vision; the MTP draft to come). The owner stopped Bolt on 2026-11-24 20:06 and deployed Inkling on 2026-11-25
  (`eklow-inkling.service`, eklow 27d6cea, /etc/nixos 7f045dff); M5's deployment is committed in /etc/nixos for the owner
  to deploy.
  - **M0 is done**: llama.cpp c74c71a6's activations, logits, greedy continuations, rendered turns, parsed outputs and
    tokens on fixed synthetic prompts (67 to 32,950 tokens), from e-workstation's 4090 (DECISIONS "Inkling M0").
  - **M1 is done**: eklow's fp32 reference forward (`src/model/inkling2.*`, `src/kernels/ref_inkling.*`), the tokenizer,
    the chat template and the output parser (DECISIONS "Inkling M1"). The forward matches llama.cpp to 1e-7 per tensor
    when given llama.cpp's own streams and activation rounding, and free-running sits inside llama.cpp's own CPU-against-CUDA
    spread, which is wide for this model (mean KL 5e-2 to 0.46). The text layer has 0 differences on every gate.
  - **M2 is done** (DECISIONS "Inkling M2: the forward on the A770M" and its two kernel entries): `engine_inkling` with
    NVFP4 experts on the matrix units and the biased attention, inside llama.cpp's spread at 67 to 32,950 tokens, decode
    steps matching llama.cpp's greedy tokens. 637 tok/s prefill at 12k; decode 12.0 tok/s with every expert on the card
    (which fits only beside a short context and without Bolt's side models).
  - **M3 is done** (DECISIONS "Inkling M3: memory, ..." and its two kernel entries): placement by a routing profile, cold
    experts on the CPU in decode and staged onto the card in prefill, two slots of 64k, checkpoints and rewind. In
    production's shape (2.53 GiB kept for the side models): decode 11.4 to 10.7 tok/s at 67 to 33k positions, prefill ~365
    tok/s, inside llama.cpp's spread.
  - **M4 is done** (DECISIONS "Inkling M4: eklow-server for Inkling, ..." and "the tool-call grammar"): the request layer,
    the scheduler (two slots, prefix reuse through checkpoints), llama.cpp's tool grammar (equal allowed sets on 39,619
    sequences), the side models beside it. The private replay through it: 40 of 41 episodes on the 7 moments it covered,
    where the 4090 passed 41 of 42; the owner stopped it there. Decode 10.5 tok/s, prefill ~365 tok/s, a next turn's first
    token after ~2.4 s. LoopCD's hidden form did not help (108 against 111 of 132 on the 4090); `--loopcd` stays off.
  - **Speed** (DECISIONS "Inkling speed, first round", "... one K/V pool ...", "... second round", "... third round", "the
    expert GEMM at half occupancy (EU fusion), and the cold copies"): decode 10.2 -> ~22.2 tok/s at 12k (~21.3 at 33k),
    prefill 363 -> ~1,050 tok/s at 12k (~1,020 at 33k). One K/V pool of 64k for the two slots and no checker (the owner's
    choices) put ~780 more experts on the card. The expert GEMM had run at half occupancy (EU fusion off) since M2; fixed,
    it does ~45 TFLOPS. Prefill now goes to the expert GEMMs (~23%), the cold sets' copies (~20%: ~13 GB a 4,096-token
    step, which slow the kernels beside them by ~61 ms a GB), attention (~18%, its kernels still with EU fusion off).
  - **Level Zero's v2 adapter** (DECISIONS 2026-11-26): a dependent small kernel costs ~2 us there against ~4.5, so
    eklow-server runs Inkling on it: at 12k decode 22.7 -> 24.1 tok/s, prefill ~1,310 -> ~1,370 through the server.
    Uploads the card reads after go by kernel (`kernels/kcopy`); driver in-order lists stay off (they broke decode).
  - **M5 is done** (DECISIONS "Inkling M5: vision, ..."): the tower matches llama.cpp's mtmd (~1e-3 per row), image
    requests render as llama-server's, the logits after an image match (KL <= 4e-10). The tower's weights live in host
    memory and it runs in the engine's idle step buffers, so `--mmproj` costs the card 1 MiB and decode nothing; an image
    of ~250 tokens encodes in 0.1 s, 972 in 0.27 s. `--image-max-tokens` 1,024 by default.
  - **M6, speculative decoding with incoai's DFlash 2 draft, works end to end** (DECISIONS "Inkling M6 through the engine
    and the server"): rollback exact, acceptance as llama.cpp's, 1.4-1.8x decode through the server at temperature 1 and
    ~1.56x greedy; plain decode ~5% slower with the draft loaded (fewer experts on the card). Not yet in nix or /etc/nixos.
  - **Full layers' prefill attention in segments** (DECISIONS 2026-11-26): 1.9x at 33k, 2.26x at 65k in the kernel, prefill
    at 33k 1,184 -> 1,261 tok/s in the engine.

- **fiona ran Bolt in production until 2026-11-24 20:06**, when the owner stopped it (`eklow-bolt.service`, from
  2026-11-10 22:48; maitred's "fiona" route falls back to son-of-anton's models until Inkling is deployed):
  Nemotron-3.5-Bolt-E4B with
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
- **Branch `serve/disk-prefix-cache`** (DECISIONS 2026-11-22): the disk prefix cache, prefill-only requests (`max_tokens` 0),
  and the fix of the rings' rewind. Checked on the card on 2026-11-22.
  - **The rewind bug is in production (main) and the fix removes it.** A slot that changed conversation past a shared stretch
    kept stale sliding K/V: 54 of 64 top-1 and mean KL 0.59 against a fresh prefill. Through the server, main answered 2 of 3
    such turns wrongly; the fix matches a fresh slot exactly.
  - **A restore is bit for bit** a rewind in place. It brings back 149k positions a second from the disk cold (250k warm).
    A 20k conversation's turn after its slot went elsewhere gives its first token in 0.55 s instead of 4.5-6 s, with the
    same greedy text. That holds after a restart too.
  - **Storing costs nothing measurable** (prefill at 32k 3,043 against 3,025 tok/s; decode unchanged).
  - **Prefill-only requests** do what they should: the next turn's first token comes in 0.047 s; a request arriving during one
    waits 0.041 s; with both slots busy one gives its slot up.
  - The draft's rings are not stored (a restore prefills ~1k positions again, ~0.45 s of the 0.55); not worth it yet.
  - `serve_test.py basic` fails three stream and seed checks on main's build as on the branch (pre-existing, the draft's
    shapes).

## Open, in order

0. **A crash to deploy the fix for** (2026-11-27 21:39, 21:41): with the draft, a verify's row past a drafted tool-call token
   the target rejects could find the grammar allowing nothing; computed on a thread (since bd4b61e), its "sampler: no token
   allowed" ended eklow-server (house chat saw EOF). Fixed in 09ddb12 (eklow-spec-test reproduces it); fiona builds with it
   (`--override-input`, the /etc/nixos lock is not writable for the agent). Open: why that grammar state allows nothing (in
   plain decoding the same state, if ever reached, fails the step's requests instead).
   **Inkling's speed**: the owner deploys the v2 adapter (/etc/nixos 3241a67f, `colmena apply --on fiona`). Fusing
   decode's chains (llama.cpp PR 33060), batching independent kernels and int8 K/V were measured and are not on
   (DECISIONS 2026-11-26). M6 (the DFlash 2 draft) is left to finish: the block size at temperature 1, depth, the
   chi-square through the server and the draft's card reservation; its nix package is in, and /etc/nixos 20e23c5e
   (eklow 2ba14da, the draft on fiona's instance) is committed and build-checked for the owner to deploy
   (`colmena apply --on fiona`). Decode (DECISIONS 2026-11-27): the sampler's full sort for top-p is gone, a verify's rows
   are sampled in parallel, the cold path's round trips shrank (speculative greedy 34.0 -> 37.3 tok/s at 12k), blocks
   follow each request's acceptance, and a kept checkpoint at the system prompt's end lets a new conversation skip it
   (DECISIONS 2026-11-27). Decode is still ~60% of the Gemma 4 26B's plain (22 against 37 tok/s at 8k; ~35 with the draft
   on GSM8K-like text): next, the hot experts' decode call (~220 us a visit, ~5 kernels), the projections, and a single
   token's cold round trip. Attention: the full layers' band
   form without spills, P.V's occupancy; every Flash and FlashGroup kernel runs with EU fusion off (decode's attention,
   the sliding layers' prefill), and decode's sliding attention needs a new ESIMD kernel (33 us a call at ~127 GB/s).
1. **The GPU regression of `cleanup/qwen-chat`** in a pause of `eklow-bolt` (fiona `~/bolt-ref/regress.sh`, ~15 minutes):
   - the kernel tests;
   - the draft's lattices and the embeddings, against main's build;
   - the checker's golden states and its 29 turns;
   - the reranker against llama.cpp f16;
   - `serve_test.py basic` and decode at 8k.
2. **`serve/disk-prefix-cache` to main and fiona.**
   - The rewind fix (`cc69ee4`) can go first, alone.
   - For the cache, fiona needs a key (an agenix secret for `services.eklow.prefixCache.keyFile`).
   - son-of-anton's prefill-only requests should name `eklow/nemotron-3.5-bolt-e4b`.
   - Open: the draft's acceptance after a restore on a longer answer (the card run's was 10 tokens), and the queue in the
     periodic line under real traffic (the writer fell behind once during a cold 30k prefill; nothing was lost).
3. **`serve_test.py basic`'s stream and seed checks** fail on main's build with the draft (pre-existing): the checks expect
   token-identical output across step shapes, which int8 rounding does not give.
4. **Text-to-speech and speech-to-text** on the same card, beside Bolt (planned).
5. **The checker's cut-offs** sit where its fitted features put them. Offtopic's cut-off lies inside the mass of its
   answers, so any implementation flips some of its flags (DECISIONS "vega's checker").
6. **Dead attention cases** of the Gemma 4 26B and 12B (head 512 groups 8 and 16, head 256 group 2), still in `attend` and
   `eklow-attn-test`.
7. **fp16 activations** for the small-batch matmuls, only if the min-p edge turns out to matter.

## Where things are

- Source of truth: `github.com/ewtodd/eklow`. Work happens on topic branches, merged into `main`.
- Toolchain: the fleet's nixpkgs (c3f15ff4): intel-llvm 7.1.1, level-zero 1.32.0, intel-compute-runtime 26.31. The oracle is
  llama.cpp at `ea0c4c2c` (`flake.nix`); the model files are flake packages (`nix/models.nix`).
- fiona: the A770M on PCIe 4.0 x8, Iris Xe (titles, llama.cpp), i7-12700H, 30 GB of RAM. Its eklow service is production:
  no server or GPU test may run on the card while any `eklow*` unit is active. `gpu.lock` does not cover the service. A
  test server beside it on 2026-11-11 slowed a real chat turn to 0.9 tok/s for six minutes.
- The private replay set (real chats, graded points, the checker's 29 turns) stays on e-workstation and fiona, never in
  the repository. Inkling's replay runs from it too.
- Inkling's ground truth and tools live on e-workstation in `~/eklow-qwen-data/inkling-port`: `m0/` (the dumps, cases and
  prompts), `m1/` (eklow's reference dumps), `oracle-inkling` (`nix-build --impure nix/oracle-inkling.nix`), the llama.cpp
  c74c71a6 sources, Hugging Face's tokenizer and config (c9d02830), and `py` (a Python with numpy, jinja2, tokenizers).

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
  - the side models' own gates: `eklow-vega check`, `tools/rerank/rerank_compare.py gate`, `tools/bench/embed_bench.py`;
  - Inkling's: `tools/ref/inkling/m2_card.sh` on fiona (the engine against M0 and M1, with compare.py on e-workstation),
    `eklow-moe-test` and `eklow-attn-test --inkling` (the kernels against the reference ops), `tools/ref/inkling/text_gates.sh build` (tokenizer, template, parser against llama.cpp and Hugging Face), and
    the reference forward (`tools/ref/inkling/host-build.sh`, then `eklow-ref-run` with `--teacher` and
    `--emulate-ggml-cpu` against llama.cpp's CPU dumps, and free-running against M0, `tools/ref/inkling/compare.py`).
