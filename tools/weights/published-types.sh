#!/usr/bin/env bash
# published-types.sh REPO REV FILE: tensor-type counts and expert/other bytes of a published GGUF without
# downloading it: the header comes by a range request into a sparse file of the full size (tensor_types.py only
# reads the header).
source "$(dirname "$0")/lib.sh"
repo=$1 rev=$2 f=$3
size=$(curl -fsSIL "https://huggingface.co/$repo/resolve/$rev/$f" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
t=$(mktemp -p "$WORK" hdr.XXXXXX)
curl -fsSL -r 0-67108863 "https://huggingface.co/$repo/resolve/$rev/$f" -o "$t.h"
truncate -s "$size" "$t"; dd if="$t.h" of="$t" conv=notrunc status=none
python3 -I "$(dirname "$0")/tensor_types.py" "$t" | sed "1s|^[^:]*:|$f:|"
rm -f "$t" "$t.h"
