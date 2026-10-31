# Shared bits of tools/weights: the model table, paths, the pinned llama.cpp, the Hub downloader.
# Sourced by make-gguf.sh and make-draft.sh; run those from `nix-shell tools/weights/shell.nix`.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
DATA=${DATA:-$HOME/eklow-qwen-data}       # finished files (gguf/), logs, build links
WORK=${WORK:-$DATA/work}                  # bulk intermediates: safetensors, BF16 GGUFs, KL logits
mkdir -p "$DATA/logs" "$DATA/gguf" "$WORK"

# llama.cpp at the revision flake.nix pins (the oracle). The CPU build gives llama-quantize,
# llama-perplexity, llama-imatrix; the CUDA build (tools/weights/cuda.nix) the 4090's tools.
llama_src() {
    nix eval --raw --impure --expr "(builtins.getFlake \"path:$REPO\").inputs.llama-cpp.outPath"
}
LLAMA_SRC=${LLAMA_SRC:-$(llama_src)}
if [ -z "${LLAMA_CPU:-}" ]; then
    [ -e "$DATA/llama-cpu" ] || nix build "path:$REPO#llamaCpuAvx2" -o "$DATA/llama-cpu"
    LLAMA_CPU=$DATA/llama-cpu/bin
fi

# model table: key -> repo, revision, imatrix repo, imatrix revision, imatrix file, imatrix sha256
model_info() {
    case $1 in
    stock)
        M_REPO=Qwen/Qwen3.6-35B-A3B; M_REV=995ad96eacd98c81ed38be0c5b274b04031597b0
        I_REPO=bartowski/Qwen_Qwen3.6-35B-A3B-GGUF; I_REV=5c2410d715
        I_FILE=Qwen_Qwen3.6-35B-A3B-imatrix.gguf; I_SHA=e860d71b ;;
    ornith)
        M_REPO=ornith-ai/Ornith-1.5-35B-A3B; M_REV=10fbf86fed7ecee4a061f8b499a618f46001cac1
        I_REPO=bartowski/Ornith-1.5-35B-A3B-GGUF; I_REV=64b0493d34
        I_FILE=Ornith-1.5-35B-A3B-imatrix.gguf; I_SHA=8d5b1693 ;;
    *) echo "model must be stock or ornith" >&2; return 1 ;;
    esac
}

# hub_fetch REPO REV DESTDIR [EXCLUDE-REGEX]: every file of the revision, resumable, four at a time;
# LFS files are checked against the Hub's sha256, the rest against the git blob id.
hub_fetch() {
    local repo=$1 rev=$2 dest=$3 excl=${4:-^$}
    mkdir -p "$dest"
    curl -fsS "https://huggingface.co/api/models/$repo/revision/$rev?blobs=true" > "$dest/.api.json"
    jq -r --arg x "$excl" '.siblings[] | select(.rfilename|test($x)|not)
        | [.rfilename, .size, (.lfs.sha256 // ""), .blobId] | @tsv' "$dest/.api.json" > "$dest/.files.tsv"
    export dest repo rev
    fetch_one() {
        local f=$1 size=$2 sha=$3 blob=$4 got
        mkdir -p "$dest/$(dirname "$f")"
        if [ "$(stat -c %s "$dest/$f" 2>/dev/null || echo 0)" != "$size" ]; then
            curl -fsSL --retry 8 --retry-delay 5 -C - -o "$dest/$f" \
                "https://huggingface.co/$repo/resolve/$rev/$f"
        fi
        if [ -n "$sha" ]; then got=$(sha256sum "$dest/$f" | cut -d' ' -f1)
        else got=$(git hash-object "$dest/$f"); sha=$blob; fi
        if [ "$got" != "$sha" ]; then echo "CHECKSUM MISMATCH $f: $got != $sha" >&2; return 1; fi
        echo "ok $f $got"
    }
    export -f fetch_one
    awk -F'\t' '{printf "%s\0%s\0%s\0%s\0",$1,$2,$3,$4}' "$dest/.files.tsv" \
        | xargs -0 -n4 -P4 bash -c 'fetch_one "$@"' _
}
