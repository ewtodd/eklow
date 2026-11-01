#!/usr/bin/env bash
# fetch-published.sh MODEL: the published GGUFs the mixes are compared with (pinned revisions), into
# $WORK/published/MODEL, with their sha256 checked against the Hub's and listed in SHA256SUMS.
source "$(dirname "$0")/lib.sh"
D=$WORK/published/$1; mkdir -p "$D"
get() {   # get REPO REV FILE
    local f=$3 meta size sha
    meta=$(curl -fsS "https://huggingface.co/api/models/$1/revision/$2?blobs=true" | jq -r --arg f "$f" '.siblings[]|select(.rfilename==$f)|"\(.size) \(.lfs.sha256)"')
    read -r size sha <<<"$meta"
    [ "$(stat -c %s "$D/$f" 2>/dev/null || echo 0)" = "$size" ] || curl -fsSL --retry 8 --retry-delay 5 -C - -o "$D/$f" "https://huggingface.co/$1/resolve/$2/$f"
    [ "$(sha256sum "$D/$f" | cut -d' ' -f1)" = "$sha" ] || { echo "MISMATCH $f" >&2; exit 1; }
    echo "$sha  $f  ($1 @ $2)" >> "$D/SHA256SUMS"; echo "ok $f"
}

case $1 in
stock) R=bartowski/Qwen_Qwen3.6-35B-A3B-GGUF; V=5c2410d715; P=Qwen_Qwen3.6-35B-A3B ;;
ornith) R=bartowski/Ornith-1.5-35B-A3B-GGUF; V=64b0493d34; P=Ornith-1.5-35B-A3B ;;
esac
shift
LIST=${*:-Q4_K_M Q4_K_S IQ4_XS Q4_0}   # names as in the file names; "official" = Ornith's own Q4_K_M
for q in $LIST; do
    if [ "$q" = official ]; then
        get ornith-ai/Ornith-1.5-35B-A3B-GGUF 12393612fd4f730ff5aadc23e9b8f9648aa49ceb Ornith-1.5-35B-Q4_K_M.gguf
    else
        get "$R" "$V" "$P-$q.gguf"
    fi
done
true
