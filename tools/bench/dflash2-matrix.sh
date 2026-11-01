#!/usr/bin/env bash
# dflash2-matrix.sh: the four target+draft pairs at block 8 and 16, greedy and at temperature 1.0 (see
# dflash2-llamacpp.sh for the protocol), then decode tok/s one stream at a time with and without the draft.
# The target is EKK8 (Q4_K routed experts, everything else as EK8), the best KL at 4.5 bits per expert weight.
# Run from `nix-shell tools/weights/shell.nix`; one server at a time, the 4090 lock held by each.
here=$(dirname "$0")
source "$here/../weights/lib.sh"
G=$DATA/gguf; DR=$G/drafts
TS=$G/stock/stock-EKK8.gguf
# Ornith: EKK8 when it exists, else its EK8 (Q4_0 experts); the target file name is in each .cmd file
TO=$G/ornith/ornith-EKK8.gguf; [ -s "$TO" ] || TO=$G/ornith/ornith-EK8.gguf
INCO=$DR/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf; JZ=$DR/Ornith-1.5-35B-A3B-DFlash2-jzinno-Q8_0.gguf
run() { bash "$here/dflash2-llamacpp.sh" "$@"; }
# native pairs first, then the crossed ones; the draft's own block size first
[ -s "$DATA/results/dflash/stock-inco-b8.summary" ] || run "$TS" "$INCO" 8  stock-inco-b8   # done in results/
run "$TO" "$JZ"  16  ornith-jzinno-b16
run "$TO" "$INCO" 8  ornith-inco-b8
run "$TS" "$JZ"  16  stock-jzinno-b16
run "$TS" "$INCO" 16 stock-inco-b16
run "$TO" "$JZ"  8   ornith-jzinno-b8
run "$TO" "$INCO" 16 ornith-inco-b16
run "$TS" "$JZ"  8   stock-jzinno-b8
# decode speed, one stream: without the draft, then with it at each pair's draft block
for m in stock ornith; do
    [ $m = stock ] && T=$TS || T=$TO
    N=8 CONC=1 run "$T" none 0 decode-$m-nodraft gsm8k
done
N=8 CONC=1 run "$TS" "$INCO" 8  decode-stock-inco-b8 gsm8k
N=8 CONC=1 run "$TO" "$JZ"  16  decode-ornith-jzinno-b16 gsm8k
N=8 CONC=1 run "$TO" "$INCO" 8  decode-ornith-inco-b8 gsm8k
N=8 CONC=1 run "$TS" "$JZ"  16  decode-stock-jzinno-b16 gsm8k
