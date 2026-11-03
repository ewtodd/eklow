#!/usr/bin/env bash
# dflash-gemma-conc1.sh: the block-16 greedy runs of dflash-gemma-matrix.sh once more with one request at a time (CONC=1), to
# see whether the near-1.0 acceptances of single requests in the concurrent runs (the draft failing outright on a request) come
# from llama.cpp serving four slots at once. Run from `nix-shell tools/weights/shell.nix`.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../weights/lib.sh"
G=$DATA/gemma
export RESULTS=${RESULTS:-$here/../../docs/bench/gemma-dflash} TOPK=64 N=${N:-16} CONC=1 SAMPLINGS=greedy
label=gemma26b-zlab-b16-conc1
rm -f "$RESULTS/$label.jsonl" "$RESULTS/$label.summary"
bash "$here/dflash2-llamacpp.sh" "$G/target/gemma-4-26B_q4_0-it.gguf" "$G/work/gguf/drafts/gemma-4-26B-A4B-it-DFlash-zlab-Q8_0.gguf" 16 "$label" "gsm8k humaneval mt-bench" || echo "run $label exited $?"
