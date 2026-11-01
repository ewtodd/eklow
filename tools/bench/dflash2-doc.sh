#!/usr/bin/env bash
# dflash2-doc.sh OUTFILE: the published result file of the DFlash 2 acceptance matrix (header, tables from
# $DATA/results/dflash/*.jsonl, the exact server commands). Run from `nix-shell tools/weights/shell.nix`.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
R=${DATA:-$HOME/eklow-qwen-data}/results/dflash
{
cat <<'HDR'
Qwen3.6-35B-A3B finetunes with DFlash 2 drafts in llama.cpp (raw results; worker qwen/weights, 2026-11-01,
e-desktop, RTX 4090 24 GB, the rest of the model on the CPU via --fit on).

Server: llama-server of the pinned llama.cpp (ea0c4c2c, CUDA build, tools/weights/cuda.nix), --spec-type
draft-dflash, --spec-draft-n-max BLOCK-1, 4 slots (-np 4), 6,144 context per slot. Driver: tools/bench/dflash2_accept.py
through tools/bench/dflash2-llamacpp.sh; the whole matrix is tools/bench/dflash2-matrix.sh.
Protocol (Inco's, from z-lab's dflash/benchmark.py prompts): 32 requests each of GSM8K, HumanEval and MT-Bench (first
turn), thinking on, up to 4,096 new tokens. "greedy": temperature 0. "temp": temperature 1.0, top-p 0.95, top-k 20
(Inco's sampling). Acceptance = per-request mean of completion tokens / verification steps (tokens per verify step),
averaged over requests. Cell = acceptance (median per-request decode tok/s with 4 requests in flight, requests that
hit the 4,096-token cap / requests). Truncated requests count with the tokens they produced.
Targets: stock-EKK8.gguf for the stock model (Q4_K routed experts, Q8_0 elsewhere); Ornith rows use ornith-EK8.gguf
(Q4_0 routed experts, Q8_0 elsewhere, 20,673,845,568 bytes), not the EKK8 mix: the acceptance of the EK8 and EKK8
mixes of one model is expected to be the same within noise, and the Ornith EKK8 file was only built afterwards. The
block size of a draft trained at another one is set by rewriting dflash.block_size in a copy of the draft GGUF
(llama.cpp reads it from there). Drafts are the Q8_0 GGUFs of tools/weights/make-draft.sh.
Protocol check: row protocol-stock-EKK8-inco-b8/temp (stock + Inco, block 8, temperature 1.0): 5.24 / 4.98 / 4.53
on GSM8K / HumanEval / MT-Bench against Inco's published 5.21 / 4.90 / 4.41 (SGLang; MTP 4.92 / 4.44 / 4.26).
Rows named stock-* use the stock target, ornith-* the Ornith target; inco = incoai/Qwen3.6-35B-A3B-DFlash2 (trained
at block 8), jzinno = jzinno/Ornith-1.5-35B-A3B-DFlash2 (trained at block 16). Throughput is one slice of a shared
machine (the CPU did other work in places): compare acceptance, treat tok/s as indicative.

HDR
python3 -I "$here/dflash2_tables.py" "$R"
echo
echo "Server commands (one per run; file names under ~/eklow-qwen-data):"
for f in "$R"/*.cmd; do echo "  $(basename "$f" .cmd): $(sed "s#$HOME/eklow-qwen-data/##g; s#/tmp/claude-[^ ]*/work/#work/#g" "$f")"; done
} > "$1"
