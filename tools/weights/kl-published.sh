#!/usr/bin/env bash
# kl-published.sh MODEL [NAME...]: the published comparison files one at a time (disk): fetch, score with kl.sh
# against the Q8_0's saved logits, delete (their sha256 stays in $WORK/published/MODEL/SHA256SUMS and their
# tensor types come from published-types.sh).
source "$(dirname "$0")/lib.sh"
M=$1; shift
case $M in stock) P=Qwen_Qwen3.6-35B-A3B;; ornith) P=Ornith-1.5-35B-A3B;; esac
for q in ${*:-Q4_K_M Q4_K_S IQ4_XS Q4_0}; do
    f=$P-$q.gguf; [ "$q" = official ] && f=Ornith-1.5-35B-Q4_K_M.gguf
    bash "$(dirname "$0")/fetch-published.sh" "$M" "$q"
    bash "$(dirname "$0")/kl.sh" "$M" "$WORK/published/$M/$f"
    rm -f "$WORK/published/$M/$f"
done
