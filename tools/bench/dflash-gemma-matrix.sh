#!/usr/bin/env bash
# dflash-gemma-matrix.sh: z-lab's plain DFlash draft for Gemma 4 26B A4B (Q8_0) against the 26B QAT Q4_0 target in the
# pinned llama.cpp's CUDA server on the 4090: blocks 16 (the draft's own), 8 and 4, greedy and at temperature 1.0 / top-p 0.95 /
# top-k 64 (Gemma's own sampling), GSM8K / HumanEval / MT-Bench, N requests each (16), thinking on. Resumable: a run whose
# summary has its six lines is skipped. Raw results (JSONL per request, summaries, server logs) in docs/bench/gemma-dflash/.
# Run from `nix-shell tools/weights/shell.nix`; WORK and DATA as lib.sh has them (the files are under ~/eklow-qwen-data/gemma).
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../weights/lib.sh"
G=$DATA/gemma
T=${TARGET:-$G/target/gemma-4-26B_q4_0-it.gguf}
D=${DRAFT:-$G/work/gguf/drafts/gemma-4-26B-A4B-it-DFlash-zlab-Q8_0.gguf}
export RESULTS=${RESULTS:-$here/../../docs/bench/gemma-dflash} TOPK=64 N=${N:-16} CONC=${CONC:-4}
mkdir -p "$RESULTS"
for B in ${BLOCKS:-16 8 4}; do
    label=gemma26b-zlab-b$B
    s=$RESULTS/$label.summary
    [ "$(grep -c acceptance_mean "$s" 2>/dev/null || true)" -ge 6 ] && { echo "skip $label"; continue; }
    rm -f "$RESULTS/$label.jsonl" "$s"
    bash "$here/dflash2-llamacpp.sh" "$T" "$D" "$B" "$label" || echo "run $label exited $? (the exit trap kills the server, so this is usual)"
done
