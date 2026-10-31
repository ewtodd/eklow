#!/usr/bin/env bash
# kl.sh MODEL FILE...: KL divergence of each GGUF against $WORK/gguf/MODEL/MODEL-Q8_0.gguf with the pinned
# llama.cpp's CUDA llama-perplexity on the 4090 (--fit puts what does not fit on the CPU: the Q8_0 is 35 GB).
# Two texts: wikitext-2-raw test at 4,096 tokens of context (WIKI_CHUNKS windows) and a Pile document at
# 32,768 (DOC_CHUNKS windows). The Q8_0's logits are saved once per text ($WORK/kl/, about 1 GB per scored
# 2,048 tokens: that is why the chunk counts are what they are) and every file is scored against them.
# Raw output: $DATA/logs/kl/MODEL-TEXT-FILE.log. Every run holds the 4090 lock only while it runs.
# Usage: kl.sh stock base            # make the base logits from the Q8_0
#        kl.sh stock path/to/x.gguf  # score files
source "$(dirname "$0")/lib.sh"
M=$1; shift
LLAMA_CUDA=${LLAMA_CUDA:-$DATA/llama-cuda/bin}
WIKI_CHUNKS=${WIKI_CHUNKS:-20}
DOC_CHUNKS=${DOC_CHUNKS:-2}
Q8=$WORK/gguf/$M/$M-Q8_0.gguf
mkdir -p "$WORK/kl" "$DATA/logs/kl"
declare -A TXT=( [wiki4k]="$DATA/data/wiki.test.raw 4096 $WIKI_CHUNKS" [doc32k]="$DATA/data/doc-131072.txt 32768 $DOC_CHUNKS" )
run() {   # run TEXT GGUF [extra args]
    local t=$1 g=$2; shift 2
    read -r file ctx chunks <<<"${TXT[$t]}"
    flock "$DATA/4090.lock" "$LLAMA_CUDA/llama-perplexity" -m "$g" -f "$file" -c "$ctx" --chunks "$chunks" \
        --fit on "$@"
}
for f in "$@"; do
    for t in wiki4k doc32k; do
        base=$WORK/kl/$M-$t.base
        if [ "$f" = base ]; then
            [ -s "$base" ] || run $t "$Q8" --kl-divergence-base "$base" > "$DATA/logs/kl/$M-$t-Q8_0-base.log" 2>&1
        else
            n=$(basename "$f" .gguf); n=${n#"$M"-}
            run $t "$f" --kl-divergence --kl-divergence-base "$base" > "$DATA/logs/kl/$M-$t-$n.log" 2>&1
            echo "$M $t $n"; grep -E "Mean PPL\(Q\)|Mean    KLD|99.0%|Same top p|Minimum|Mean PPL\(base\)" "$DATA/logs/kl/$M-$t-$n.log" | head -8
        fi
    done
done
