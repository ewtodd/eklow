#!/usr/bin/env bash
# dflash2-llamacpp.sh TARGET.gguf DRAFT.gguf BLOCK LABEL [datasets]: reference acceptance of one target+draft pair
# in the pinned llama.cpp's CUDA llama-server on the 4090 (run from `nix-shell tools/weights/shell.nix`).
#  - BLOCK is the draft's block size: if the draft GGUF was trained at another one, dflash.block_size is rewritten
#    in a copy (llama.cpp reads it from there, and drafts block-1 tokens: --spec-draft-n-max is set to BLOCK-1).
#  - BLOCK=0 serves the target alone (decode tok/s without the draft).
#  - per dataset and per sampling (greedy, then temperature 1.0 / top-p 0.95 / top-k $TOPK, 20 unless set): N requests (default 32),
#    thinking on, up to 4096 new tokens, CONC requests at a time (acceptance does not depend on it; run
#    CONC=1 for decode tok/s).
# Results: $DATA/results/dflash/LABEL.jsonl (one line per request), server log beside it; the server holds the 4090
# lock for its whole life.
source "$(dirname "$0")/../weights/lib.sh"
# e-desktop 2026-11-01: userspace libcuda (595.112) is newer than the loaded kernel module (595.104.02) until a reboot;
# use the libraries matching the module (harmless once they match)
NVLIB=/nix/store/fprgxisprl0218i079npizclrgfgfja7-nvidia-x11-595.104.02/lib
# (only while the loaded module is that old one: after the 2026-11-03 reboot it is 595.112.07, which the system libraries match)
grep -q 595.104.02 /proc/driver/nvidia/version 2>/dev/null && [ -d "$NVLIB" ] && export LD_LIBRARY_PATH=$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
T=$1 D=$2 B=$3 LABEL=$4; DS=${5:-gsm8k humaneval mt-bench}
N=${N:-32} CONC=${CONC:-4} PORT=${PORT:-8090} MAXTOK=${MAXTOK:-4096} CTX_PER=${CTX_PER:-6144}
SAMPLINGS=${SAMPLINGS:-"greedy temp"}
LLAMA_CUDA=${LLAMA_CUDA:-$DATA/llama-cuda/bin}
R=${RESULTS:-$DATA/results/dflash}; mkdir -p "$R"
TOPK=${TOPK:-20}   # temperature sampling's top-k (Qwen3.6: 20; Gemma 4: 64)
args=(-m "$T" --host 127.0.0.1 --port "$PORT" -np "$CONC" -c $((CONC * CTX_PER)) --fit on)
if [ "$B" != 0 ]; then
    trained=$(python3 - "$D" <<'PY'
import sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); print(int(r.fields["dflash.block_size"].contents()))
PY
)
    if [ "$trained" != "$B" ]; then
        V=$WORK/gguf/drafts/variants; mkdir -p "$V"
        VD=$V/$(basename "$D" .gguf)-block$B.gguf
        [ -s "$VD" ] || { cp "$D" "$VD"; python3 "$LLAMA_SRC/gguf-py/gguf/scripts/gguf_set_metadata.py" "$VD" dflash.block_size "$B" --force; }
        D=$VD
    fi
    args+=(--spec-draft-model "$D" --spec-type draft-dflash --spec-draft-n-max $((B - 1)) --spec-draft-ngl 99)
fi
cmd=("$LLAMA_CUDA/llama-server" "${args[@]}")
echo "${cmd[*]}" > "$R/$LABEL.cmd"
flock "$DATA/4090.lock" "${cmd[@]}" > "$R/$LABEL.server.log" 2>&1 &
SP=$!
trap 'pkill -P $SP 2>/dev/null; kill $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT   # flock forks the server
until curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 $SP 2>/dev/null || { echo "server died"; tail -20 "$R/$LABEL.server.log"; exit 1; }
    sleep 3
done
for s in $SAMPLINGS; do
    case $s in greedy) samp=(--temperature 0 --top-p 1.0 --top-k 0);; temp) samp=(--temperature 1.0 --top-p 0.95 --top-k "$TOPK");; esac
    for d in $DS; do
        python3 "$(dirname "$0")/dflash2_accept.py" --url "http://127.0.0.1:$PORT" --dataset "$d" --n "$N" "${samp[@]}" \
            --max-tokens "$MAXTOK" --concurrency "$CONC" --label "$LABEL/$s" --out "$R/$LABEL.jsonl" | tee -a "$R/$LABEL.summary"
    done
done
