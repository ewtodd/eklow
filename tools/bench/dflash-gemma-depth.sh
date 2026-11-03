#!/usr/bin/env bash
# dflash-gemma-depth.sh: what the z-lab draft's full-attention layer is worth at depth, in llama.cpp (the 4090): prompts of
# about 8k and 24k tokens (slices of the Pile document in data/) answered greedily, with the draft as shipped (its layer 4 sees
# the whole context) and with every layer sliding at 2,048 (`dflash.attention.sliding_window_pattern` all true: the full layer
# capped to the last 2,048 positions, the harshest cap; the other four are unchanged). Block 16, thinking on, N requests.
# Raw results in docs/bench/gemma-dflash/depth-*. Run from `nix-shell tools/weights/shell.nix`.
here=$(cd "$(dirname "$0")" && pwd)
source "$here/../weights/lib.sh"
G=$DATA/gemma
export RESULTS=${RESULTS:-$here/../../docs/bench/gemma-dflash} TOPK=64 N=${N:-8} CONC=1 CTX_PER=32768 MAXTOK=${MAXTOK:-1024} SAMPLINGS=greedy
T=$G/target/gemma-4-26B_q4_0-it.gguf
for variant in full allswa2048; do
    D=$G/work/gguf/drafts/gemma-4-26B-A4B-it-DFlash-zlab-Q8_0.gguf
    [ $variant = full ] || D=$G/work/gguf/drafts/variants/zlab-Q8_0-allswa2048.gguf
    for chars in 30000 90000; do
        label=depth-$variant-$((chars / 1000))kchars
        s=$RESULTS/$label.summary
        [ "$(grep -c acceptance_mean "$s" 2>/dev/null || echo 0)" -ge 1 ] && { echo "skip $label"; continue; }
        rm -f "$RESULTS/$label.jsonl" "$s"
        DOC_CHARS=$chars bash "$here/dflash2-llamacpp.sh" "$T" "$D" 16 "$label" doc || echo "run $label exited $?"
    done
done
