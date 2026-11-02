# STATUS

Live state, written as a handoff (2026-11-01). Design and targets: `DESIGN.md`; history and evidence:
`DECISIONS.md` (every entry from 2026-10-31 on is the Qwen3.6 port); what it is and the headline numbers: `README.md`.

## Now (2026-11-01)

- **eklow serves Qwen3.6-35B-A3B finetunes** (llama.cpp's `qwen35moe`: 30 Gated DeltaNet and 10 gated-attention
  layers, 256 routed experts of 512 + one shared, 8 used) with **DFlash 2** speculative decoding, in place of Gemma 4
  12B / 26B A4B, whose code is gone (EmbeddingGemma 2 stays, with Gemma's tokenizer and vision encoder). Two targets so
  far, same architecture: stock `Qwen/Qwen3.6-35B-A3B` and `ornith-ai/Ornith-1.5-35B-A3B`. All of it is on the branch
  `qwen/main`; `main` (and fiona's deployed eklow) still has the 26B until the switch.
- **Production on fiona is stopped**: the owner agreed to stop `eklow.service` (the 26B) for the port on 2026-10-31
  17:40. A deploy or reboot starts it again (the service is enabled): check `systemctl is-active eklow` before any card
  work, stop it, and say so. Clients that pick `eklow/gemma-4-26b-a4b` get nothing meanwhile.
- **The switch, still to do**: the release numbers (both models through the server, and the owner's `check-source`
  evals, branch `qwen/release`) decide which model fiona serves; then `qwen/main` goes into `main` and is pushed, and
  `/etc/nixos` gets the new eklow config (model, draft, profile, template; maitred's upstream id) together with the
  eklow input update, so a deploy never evaluates the new options against the old eklow. The owner deploys.
- **fiona's card**: the A770M is kept out of runtime suspend (`/etc/nixos` 432377be, deployed 2026-11-01): on xe a
  resume from it sometimes failed (forcewake timeouts, only while the card idled) and once lost the card until a reboot.

## The files

On Hugging Face (`ethanwtodd`, public; uploaded 2026-11-01; the model cards carry the KL tables):

| repo @ revision | file | sha256 |
|---|---|---|
| `Qwen3.6-35B-A3B-Q4_K-Q8_0-GGUF` @ 8db5016e | `Qwen3.6-35B-A3B-Q4_K-Q8_0.gguf` (20.7 GB) | 48cf98f4... |
| `Ornith-1.5-35B-A3B-Q4_K-Q8_0-GGUF` @ a89886e0 | `Ornith-1.5-35B-A3B-Q4_K-Q8_0.gguf` (20.7 GB, Ornith's own template) | 25a68129... |
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

| | Qwen3.6 | the 26B in production (2026-10-27) |
|---|---|---|
| prefill at 8k / 32k / 128k / 252k (time to first token) | 1,001 / 977 / 602 / 392 tok/s | 1,865 / 1,794 / 1,177 / 765 |
| decode at 8k / 32k / 128k / 252k, with the draft | 44.1 / 40.5 / 33.1 / 24.3 tok/s | 36.6 / 34.0 / 30.5 / 26.5 |
| decode without the draft | 33.9 / 31.5 / 28.1 / 23.6 tok/s | |
| GSM8K one / two at a time (with the draft) | 53.5 / 62.8 tok/s aggregate | 33.4 / 40.2 |
| next turn of a 10k-token conversation | 0.26-0.61 s | 0.2-0.7 s |

Fidelity: the engine within the fp32 reference's band on every path (`checkq`: stock EKK8 242-248 of 256 top-1 on the
2,573-token prompt, chat 27/27); DFlash 2 lossless through the server (3,000 sampled completions with and without the
draft, chi-square p = 0.985); acceptance at block 8 5.75 / 5.43 / 5.02 on GSM8K / HumanEval / MT-Bench (llama.cpp:
5.24 / 4.98 / 4.53). The final numbers on the merged tree, for both models, and the `check-source` scores are being
measured (`qwen/release`) and will replace this section and README's tables.

## Where things are

- Source of truth: `github.com/ewtodd/eklow` (working copy `~/Software/eklow` on e-desktop, now on `qwen/main`). The
  work runs on `qwen/<topic>` branches in worktrees `~/Software/eklow-wt/q*` (fiona mirrors `~/eklow-wt-q*`), merged
  into `qwen/main`. `fiona:~/eklow-qmain` is a build of `qwen/main` (2026-11-01).
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

1. **The switch** (above): release numbers, the model choice, `main`, `/etc/nixos`, the owner's deploy.
2. **Prefill**: half the 26B's at 8k (the experts are 18 GB and only 52-71 of 256 a layer fit beside 2 x 262k of KV;
   72% of a chunk's expert work is cold). The CPU's share is at its memory wall and streaming near its optimum
   (DECISIONS 2026-11-01, "the MoE's speed"). Levers: context for experts (2 x 128k frees ~2.9 GiB, ~30 more experts a
   layer: the owner's call), the Iris Xe as a second device for the pairs not streamed, the placement gap below.
3. **Placement**: the server places 52 experts a layer against the engine bench's 71 (the embedding model, the draft's
   reserve, scratch): account for it and take back what is wasted (`qwen/release` item 1).
4. **Verify rows**: each row past the first costs ~8.5 ms of the CPU's cold experts, so blocks of 4 are best; cheaper
   rows would make the drafts' longer blocks pay (6.5 tokens a verify at block 16).
5. **Snapshots mid-generation**: Ornith's third turn reused 10,398 of 11,075 tokens because a sampled answer
   tokenized differently from its re-render; a snapshot every ~256 generated tokens would bound that.
6. **Decode launches**: ~25 a layer; a fused tail or SYCL graphs.
7. **Image input for Qwen3.6** (its vision encoder is not ported: image parts are refused).
8. A routing profile from real traffic (today's are from the reference's recordings).

## History

Milestones 1-6 (foundation, the 12B's correct forward, serving, speed, the 26B A4B in production 2026-10-27) are in
DECISIONS.md from 2026-10-24 to 2026-10-27, with every measurement; the Qwen3.6 port from 2026-10-31.
