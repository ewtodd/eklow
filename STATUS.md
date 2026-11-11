# STATUS

Live state, written as a handoff (2026-11-02). Design and targets: `DESIGN.md`; history and evidence:
`DECISIONS.md` (every entry from 2026-10-31 on is the Qwen3.6 port); what it is and the headline numbers: `README.md`.

## Now (2026-11-03)

- **Nemotron-3.5-Bolt-E4B on eklow, not deployed** (2026-11-10, branch `gemma/bolt-e4b`; DECISIONS 2026-11-10, six entries):
  NVIDIA's post-trained Gemma 4 E4B, from the Q4_K_M file NVIDIA ships, beside Qwen3.6 in the same `eklow-server` (it dispatches
  on `general.architecture`; the Gemma 4 engine, server and text layer back from `cc713a0`, dense, with per-layer inputs and
  shared K/V). The fp32 reference matches transformers to KL 5e-8 at 16k; the engine sits inside transformers' own bf16 distance.
  On fiona at production's shape (2 x 65,536 fp16, EmbeddingGemma 2 beside, 5.71 + 2.20 GiB): prefill 3,549 / 2,920 / 2,273 tok/s
  at 8k / 32k / 60k, decode 44.8 / 39.3 / 35.6 tok/s plain and 90.8 / 87.9 / 64.6 with NVIDIA's DFlash draft (blocks of 8,
  lossless by chi-square). Chat images through Google's E4B projector (the same tower) in 0.10 s. fiona's replay with llama.cpp's
  sampler: 30 and 31 of 54 in two runs against llama.cpp CUDA's 32 of 54 (27 and 27 under eklow's old sampler; DECISIONS
  2026-11-11 "Bolt under llama.cpp's sampler"). `preserve_thinking` on: the user turn after a tool loop prefills 0.1-2.5k tokens
  instead of the whole loop (up to 36k, 15 s, off), and the 64K window fills 1.2-1.7x faster; its effect on answers is not
  measured (the replay's chats lack the stored reasoning). The module runs it as an instance (`model`, `alias`, `port`; `draft`,
  `mmproj` as for Qwen). The owner deploys.
- **Sampling is llama.cpp's** (2026-11-11, branch `gemma/bolt-e4b`; DECISIONS "llama.cpp's sampler, exactly"), for Qwen and
  Gemma alike: llama-server's chain, defaults, GGUF keys and request fields, bit for bit against the pinned build's libllama on
  fixed logits; DRY, adaptive_p and infill are refused. Bolt and Sage both gain min-p 0.05. A token at min-p's edge can be kept in
  one batch shape and dropped in another (the int8 activations), so a two-at-a-time chi-square under the default chain is not a
  test of the draft; `serve_test.py sample` sends min_p 0.
- **vega's checker and the reranker are ready to serve, not merged or deployed** (`vega/checker`, 2026-11-10, DECISIONS "vega's
  checker" and "Qwen3-Reranker on eklow"), beside whichever chat model runs (`eklow-server --checker-*`, `--rerank-*`; module
  `services.eklow.checker`, `services.eklow.reranker`; `eklow-vega serve` and `eklow-rerank serve` alone):
  - the checker is laya's System One (`POST /v1/systemone`) for son-of-anton's end-of-turn checks: Qwen3.5-0.8B (vega's reader)
    on the card, vega-U's engine and laya's policy on the CPU. On the 29 replay turns it is within 0.0014 of laya with an fp32
    reader (no flag differs) and within 0.012 / 0.036 / 0.0023 (claim, unchecked, offtopic) of laya as fitted in bf16, as far as
    laya's own fp32, eager-attention and CPU runs land, flipping the one flag (0.00027 from its cut-off) they all flip. p50 271
    ms, p90 502 ms a turn; 1,197 MiB of the card.
  - the reranker is llama.cpp's `POST /v1/rerank` over ggml-org's Qwen3-Reranker-0.6B Q8_0: within 0.0021 of llama.cpp (f16
    K/V) and 0.00036 of an fp32 run of the same weights; 101 ms for 8 documents of ~1,200 characters, 264 ms for 24; 796 MiB.
  - All four on the card with fp16 caches (the owner's call): Bolt 5.71 GiB and EmbeddingGemma 2 2.20 GiB at 2 x 65,536 (the
    Bolt branch's measurement), the checker 1.17 GiB, the reranker 0.78 GiB: 9.86 GiB of 16.
  - To deploy: the reader GGUF (`tools/vega/make-reader.sh`) and the policy folder where the unit can read them, the reranker's
    GGUF (fetchurl, the module's example), the ports in son-of-anton's `[checker] base_url` and the rerank route. The Bolt-E4B
    server (`gemma/bolt-e4b`) takes the same flags once it calls `serve_systemone` and `serve_rerank` (tools/server/systemone.hpp,
    rerank.hpp) as qwen_server_main does.
- **Sage-A1.5 is ready to serve, not deployed** (2026-11-07, DECISIONS "Sage-A1.5 on eklow" and "Sage-A1.5's chat template"):
  `--chat-template sage` (recognised from the GGUF), `--verification on|off` (on only for conversations with system text);
  0 differences against Hugging Face on 28,017 cases. Its EKK8 is on Hugging Face (below), with KL 0.0127 at 4k against its Q8_0.
  On fiona's replay it never thought itself into the cap (Ornith 6/18) and writes about a third of the tokens. Ornith's first
  move is better. Through the server at production's shape: 43.2 / 41.3 tok/s at 8k / 32k with Inco's draft, which accepts
  11-30% more than Sage's own MTP head. With the shared pool (`serve/kv-pool`): 52.9 / 45.9.
- **eklow serves Qwen3.6-35B-A3B finetunes** (llama.cpp's `qwen35moe`: 30 Gated DeltaNet and 10 gated-attention
  layers, 256 routed experts of 512 + one shared, 8 used) with **DFlash 2** speculative decoding and **image input**
  (Qwen3-VL's encoder behind `--mmproj`). Two targets so far, same architecture: stock
  `Qwen/Qwen3.6-35B-A3B` and `ornith-ai/Ornith-1.5-35B-A3B`. `qwen/main` went into `main` on 2026-11-02.
- **Gemma 4 was removed from eklow on 2026-11-07** (the owner: Qwen3.6 and its finetunes only, for now; DECISIONS 2026-11-07),
  and came back as Bolt on 2026-11-10 (above); until then `eklow-server` refused any GGUF whose architecture was not `qwen35moe`. EmbeddingGemma 2 (and its vision tower) stays. The
  last state of the 26B's engine, server, draft path and tools is `d9ffaf0` (the restore beside Qwen) and `9775e98` (its
  speculative path; z-lab's draft measured and dropped). The plain-DFlash draft code (per-layer sliding/full attention, tied
  head, softcap) stays, with its tests in `tools/dflash_test`.
- **`--preserve-thinking on`** (module `preserveThinking`; DECISIONS 2026-11-08 "keep earlier turns' reasoning"): with Sage's
  (stock's) template a new user message re-rendered the whole agentic turn before it without reasoning, and the server prefilled
  that turn again (~1,050 tok/s at 190k on fiona). On, earlier turns keep their reasoning, as Ornith's template always did. Host
  test only; /etc/nixos sets it for sage; the owner deploys. The night's two three-minute waits were something else:
  son-of-anton's last step of a 500-step turn sends `tool_choice: "none"`, and eklow then drops the tools from the system
  message, so the whole prompt is prefilled again. Fixed on `serve/tool-choice-none`: the tools stay in the prompt and
  `<tool_call>` is never sampled (DECISIONS 2026-11-08 "tool_choice none").
- **The card margin is 256 MiB, not 1 GiB** (`serve/margin`, DECISIONS 2026-11-08; deployed 2026-11-07 22:11): everything the server
  allocated lazily (attention partials at depth, the vision encoders' scratch and per-image buffers) is now sized, zero-filled and
  warmed before the experts are placed; what is left is the driver's own buffers on the engine's first launches, 108 MiB measured
  over every stage (252k decode, two long prompts, images and embeddings, an 8,000-token embedding, verify blocks of 16, the pool
  reclaiming). At production's shape: 96 experts a layer against 91 (net +0.35 GiB: 0.75 from the margin, 0.40 now held from the
  start), plain decode +2% at 8k / 32k; with the draft -1.8% / +4.2% (acceptance noise). `EKLOW_ALLOC_TRACE=1` logs what the server
  allocates after startup; `tools/margin/` has the trace shim, the stage measurement and `pool.sh`. The next lever is EmbeddingGemma 2's
  projector, 264 MiB + 162 MiB of driver buffers for 1,120 rows when the server sends 280 (`reserve(280)` gave NaN rows once).
- **One K/V pool for the slots** (`serve/kv-pool`, merged into `main`; DECISIONS 2026-11-07): `--kv-pool N`
  positions for all slots together (default 0 = slots x ctx, unchanged). At `--slots 2 --ctx 262144 --kv-pool 262144` the K/V
  and state take 3.03 GiB instead of 5.84 and the card keeps 91 experts a layer instead of 48: decode with the draft +22% at
  8k, +11% at 32k, GSM8K +19% (measured, Sage-A1.5 EKK8). An idle slot gives pages back and re-prefills later; a request
  that does not fit waits; a sequence starved with both slots busy ends with "length". checkq, kv_pool_test.py and
  serve_test.py basic (stock) pass.
- **In production since 2026-11-02 ~20:22**: fiona serves **Ornith-1.5** with Inco's draft and the projector at
  2 x 262k, int8 KV (51 of 256 experts a layer on the card; eklow built on its own nixpkgs, intel-llvm 7.1.1: the
  fleet's 7.2.0 left 0.8 GiB less free, 39 experts). No further evals (`check-source` stopped after stock's GSM8K:
  293/300, 97.7% +- 1.7%, `docs/bench/check-source/stock/`).
- **Layer-major prefill** (`main` ecabbb1, DECISIONS 2026-11-03): a lone long prompt runs in groups of up to 8
  chunks layer by layer, each cold expert crossing PCIe once per layer per group; the streamed experts are repacked in
  one launch (+19% plain prefill on its own). Through the server, time to first token at 8k / 32k / 128k / 252k: 5.1 /
  17.6 / 131 / 356 s (6.7 / 25.5 / 167 / 442 s plain). `/etc/nixos` 276af231 sets `prefillGroup = 8`,
  `prefillGroupBusy = 4096` (48 experts a layer); it needs the eklow input at ecabbb1; the owner deploys.
- **The last check before the switch** (2026-11-02, `main`'s tree on fiona): `checkq ekk8` inside the gate (242-248
  of 256 top-1, chat 27/27); the server at the production shape with Ornith, the draft, the projector and EmbeddingGemma
  2 passed `serve_test.py basic` and `image_test.py` (a screenshot, a newspaper page as PNG with alpha and as a
  paletted GIF, a second turn reusing 1,130 of 1,153 tokens, two same-size pictures not confused, the refusals).
- **fiona's card**: the A770M is kept out of runtime suspend (`/etc/nixos` 432377be, deployed 2026-11-01): on xe a
  resume from it sometimes failed (forcewake timeouts, only while the card idled) and once lost the card until a reboot.

## The files

On Hugging Face (`ethanwtodd`, public; uploaded 2026-11-01; the model cards carry the KL tables):

| repo @ revision | file | sha256 |
|---|---|---|
| `Qwen3.6-35B-A3B-Q4_K-Q8_0-GGUF` @ 8db5016e | `Qwen3.6-35B-A3B-Q4_K-Q8_0.gguf` (20.7 GB) | 48cf98f4... |
| `Ornith-1.5-35B-A3B-Q4_K-Q8_0-GGUF` @ a89886e0 | `Ornith-1.5-35B-A3B-Q4_K-Q8_0.gguf` (20.7 GB, Ornith's own template) | 25a68129... |
| `Sage-A1.5-Q4_K-Q8_0-GGUF` @ d89116d8 (2026-11-07) | `Sage-A1.5-Q4_K-Q8_0.gguf` (21.2 GB, Sage's own template, MTP block kept) | 6d25f141... |
| `Qwen3.6-35B-A3B-DFlash2-Q8_0-GGUF` @ 65aaab95 | `Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf` (570 MB, Inco's draft) | 986b8f24... |

The mix (`EKK8`): routed experts Q4_K under bartowski's importance matrix, every other matrix Q8_0, head Q6_K; KL
against Q8_0 0.0134 (stock) and 0.0354 (Ornith) at 4k, better than the published Q4_K_M of each (DECISIONS
2026-10-31, "Qwen's weight formats", and 2026-11-01, "the draft for both models"). One draft serves both: Inco's
DFlash 2 (it ties the Ornith-trained community draft on Ornith at the short blocks this card runs, and beats it on
stock).

Working copies on fiona: `~/models/{stock,ornith}-EKK8.gguf`, `stock-EK8`, `ornith-EK8`, `stock-Q8_0` (the KL base),
`~/models/drafts/` (both DFlash 2 drafts, BF16 and Q8_0, Inco's block-16 variant, their safetensors). e-desktop keeps
only `~/eklow-qwen-data/{logs,results,hf-cards,imatrix,...}`: the owner had the model files deleted there (home was
at 99%) once their sha256 matched fiona's. The scripts that made them: `tools/weights/`.

## Numbers so far (production shape: 2 slots x 262,144, int8 KV)

Engine (`eklow-engine-bench`, stock EKK8, 71 of 256 experts a layer on the card; DECISIONS 2026-11-01, "the serving
engine", plus the attention segments): prefill 1,180 / 1,032 / 539 tok/s at 8k / 32k / 128k depth; decode 41.3 /
39.5 / 37.4 / 31.3 tok/s at 0 / 8k / 32k / 128k; a verify block of 8 at 8k 70.9 ms.

Server (`serve_perf.py`, stock EKK8 with Inco's draft at block 4, 52 experts a layer; measured before the attention
segments were merged, so prefill at depth is now higher; DECISIONS 2026-11-01, "eklow-server on Qwen3.6"):

| | Qwen3.6 |
|---|---|
| prefill at 8k / 32k / 128k / 252k (time to first token) | 1,001 / 977 / 602 / 392 tok/s |
| decode at 8k / 32k / 128k / 252k, with the draft | 44.1 / 40.5 / 33.1 / 24.3 tok/s |
| decode without the draft | 33.9 / 31.5 / 28.1 / 23.6 tok/s |
| GSM8K one / two at a time (with the draft) | 53.5 / 62.8 tok/s aggregate |
| next turn of a 10k-token conversation | 0.26-0.61 s |

Fidelity: the engine within the fp32 reference's band on every path (`checkq`: stock EKK8 242-248 of 256 top-1 on the
2,573-token prompt, chat 27/27); DFlash 2 lossless through the server (3,000 sampled completions with and without the
draft, chi-square p = 0.985); acceptance at block 8 5.75 / 5.43 / 5.02 on GSM8K / HumanEval / MT-Bench (llama.cpp:
5.24 / 4.98 / 4.53). The final numbers on the merged tree, for both models (59 experts a layer): README
"Performance" and `docs/bench/release/`.

## Where things are

- Source of truth: `github.com/ewtodd/eklow` (working copy `~/Software/eklow` on e-desktop, on `main`). The
  work runs on `qwen/<topic>` branches in worktrees `~/Software/eklow-wt/q*` (fiona mirrors `~/eklow-wt-q*`), merged
  into `qwen/main` and from there into `main`. `fiona:~/eklow-qmain` is a build of `main` as merged (2026-11-02).
- Toolchain: the fleet's nixpkgs (c3f15ff4): intel-llvm 7.1.1, level-zero 1.32.0, intel-compute-runtime 26.31. The
  oracle is llama.cpp at `ea0c4c2c` (`flake.nix`), which has qwen35moe and DFlash 2.
- fiona: A770M on PCIe 4.0 x8 (~11 GB/s), Iris Xe (titles, llama.cpp), i7-12700H (AVX-VNNI; its two DDR4 channels read
  38-43 GB/s: the cold experts' wall), 30 GB of RAM (~13 GiB of it the cold experts' pinned copy).
- Reference data on fiona: `~/eklow-oracle-qwen36/` (llama.cpp CPU and CUDA dumps) and `~/eklow-ref-qwen36/` (eklow's
  fp32 reference) for the `long` (2,573 tokens) and `chat` (27) prompts, `NOTES.txt` with the files' sha256;
  `~/eklow-qwen-data-dflash/` (the draft's checks); `~/eklow-bench` (the server benchmark's inputs).
- e-desktop's RTX 4090 runs llama.cpp CUDA for references (KL, acceptance): jobs under `~/eklow-qwen-data/4090.lock`.
  Its NVIDIA userspace was updated without a reboot on 2026-11-01 (module 595.104, libraries 595.112): until it
  reboots, start CUDA programs with the old libraries
  (`LD_LIBRARY_PATH=/nix/store/fprgxisprl0218i079npizclrgfgfja7-nvidia-x11-595.104.02/lib`); nvtop works the same way.

## How to work on it

- Build on e-desktop (`nix develop path:. -c bash -c 'cmake -S . -B build -G Ninja && ninja -C build'`; the host-only
  tools run there), run on fiona (rsync the tree without `.git` and `build`, build there).
- The harness, `fiona:~/eklow-perf` (edit `~/Software/eklow-perf` on e-desktop, scp to fiona): `bench.sh checkq
  [ek8|ekk8|oek8]` (the engine against the fp32 reference, six ways, with the gate), `engineq` (the production shape),
  `attn`, `kernels`; `bench.sh` takes `gpu.lock` itself (never wrap it in `flock`) and refuses to run while
  `eklow.service` is up; `AGENTS.md` has the workers' rules.
- Gates: `checkq` (DECISIONS 2026-10-31, "Qwen3.6's reference forward": stock >= 225/256 top-1 and median KL <= 1e-2 on
  `long`, chat >= 25/27; Ornith >= 195/256), `eklow-gdn-test`, `eklow-qfmt-test`, `eklow-cpu-qgemv-test`,
  `eklow-spec-test`, `bench.sh attn` and `kernels`, `tools/server/serve_test.py basic` against a server at the
  production shape, and the Qwen text checks (`tools/ref/qwen_*_compare.py`, `eklow-grammar-test-qwen`).
- Workers: Sonnet subagents, one per worktree, each with a `TASK.md`; they commit as son-of-anton-bot, never push,
  and report in their final message (the harness refuses report files). Sessions end (network, reboots, once a week
  for e-desktop): a worker dies with its session, its detached remote jobs survive; resume with a "Continuation"
  section in its TASK.md and a fresh worker. Long runs detached (`setsid nohup ... > log`), the card lock held only
  while a test runs, nothing kept under /tmp (a reboot wipes it).

## Open, in order

0. **The checker's policy on the features eklow serves.** Its heads and son-of-anton's cut-offs were fitted on bf16 features whose
   own noise moves unchecked by up to 0.036 (DECISIONS "vega's checker"); refitting them on eklow's (or fp32) features would put
   the cut-offs where the served probabilities are. The offtopic cut-off sits inside the mass of offtopic's answers (0.39-0.46
   after its temperature of 17.7), so any implementation flips some of its flags.
1. **What production shows** with Ornith-1.5 and the prefill groups.
2. **Prefill**, the owner's order (2026-11-02): layer-major groups (done), then a microbenchmark of the DMA tax (does a
   kernel reading pinned host memory slow the card less than copy-engine pushes?), int8 Q.K prefill attention at depth
   (needs a long-context fidelity check first), the placement gap below. A grouped 16k profile: the card's own GEMMs
   (stream 1.59 s, hot 0.78 s of 7.43 s), DeltaNet 1.4 s, attention 0.88 s, the fp32 router 0.53 s; copies 0.49 s.
   Context for experts (2 x 128k frees ~2.9 GiB, ~30 experts a layer) stays the owner's call.
3. **Placement**: the server places 59 experts a layer (52 before the draft's token table went to the host) against
   the engine bench's 71 (the embedding model, the draft, scratch): what is left to take back.
4. **Verify rows**: each row past the first costs ~8.5 ms of the CPU's cold experts, so blocks of 4 are best; cheaper
   rows would make the drafts' longer blocks pay (6.5 tokens a verify at block 16).
5. **Snapshots mid-generation**: Ornith's third turn reused 10,398 of 11,075 tokens because a sampled answer
   tokenized differently from its re-render; a snapshot every ~256 generated tokens would bound that.
6. **Decode launches**: ~25 a layer; a fused tail or SYCL graphs.
7. A routing profile from real traffic (today's are from the reference's recordings).

## History

Milestones 1-6 (foundation, the 12B's correct forward, serving, speed, the 26B A4B in production 2026-10-27; Gemma 4 is out of the tree since 2026-11-07) are in
DECISIONS.md from 2026-10-24 to 2026-10-27, with every measurement; the Qwen3.6 port from 2026-10-31.
