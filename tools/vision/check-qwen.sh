#!/usr/bin/env bash
# The Qwen3-VL encoder (eklow-vision-qwen) against llama.cpp's CPU encoder (eklow-oracle --encode-only), image by image.
#
#   check-qwen.sh OUT_DIR IMAGE...      (run on fiona from a tree with build/eklow-vision-qwen built)
#
# Environment: ORACLE (eklow-oracle), TEXT_MODEL (any GGUF with n_embd 2048: the oracle loads it only for mtmd),
# MMPROJ_F32 (the oracle's projector: F32 weights, so that ggml's bf16 activation rounding is out of the comparison, as
# for Gemma's, DECISIONS 2026-10-27), MMPROJ (eklow's: the BF16 file), TOKENS (token budgets, default "64 256 1024"),
# REF_MAX (largest budget the reference path runs, it is serial: default 256), PARTS ("ref fast", what to check).
# Prints, per budget and image, the oracle's token grid against eklow's and the rows' agreement (cosine min / mean,
# relative L2 worst row / total, from tools/ref/compare_rows.cpp, built next to this script's output).
set -e
OUT=$1; shift
ORACLE=${ORACLE:-/var/tmp/eklow-vision/oracle/bin/eklow-oracle}
TEXT_MODEL=${TEXT_MODEL:-$HOME/models/stock-EK8.gguf}
MMPROJ_F32=${MMPROJ_F32:-/var/tmp/eklow-vision/conv/mmproj-Ornith-1.5-35B-A3B-f32.gguf}
MMPROJ=${MMPROJ:-/var/tmp/eklow-vision/conv/mmproj-Ornith-1.5-35B-A3B-bf16.gguf}
TOKENS=${TOKENS:-"64 256 1024"}
REF_MAX=${REF_MAX:-256}
PARTS=${PARTS:-"ref fast"}
MIN=${MIN:-8}
mkdir -p "$OUT"
[ -x "$OUT/compare_rows" ] || c++ -O2 -std=c++17 "$(dirname "$0")/../ref/compare_rows.cpp" -o "$OUT/compare_rows"
imgs=(); for i in "$@"; do imgs+=(--image "$i"); done
for tok in $TOKENS; do
  d=$OUT/ora-$tok
  if [ ! -f "$d/images.json" ]; then
    mkdir -p "$d"
    "$ORACLE" --model "$TEXT_MODEL" --mmproj "$MMPROJ_F32" "${imgs[@]}" --image-tokens "$tok" --encode-only --out "$d" --threads 16 > "$d/log.txt" 2>&1
  fi
  for part in $PARTS; do
    [ "$part" = ref ] && [ "$tok" -gt "$REF_MAX" ] && continue
    e=$OUT/$part-$tok
    mkdir -p "$e"
    flags=(); [ "$part" = ref ] && flags=(--ref)
    # the card is shared: the lock (not bench.sh, this is our own program)
    flock -w 3600 ~/eklow-perf/gpu.lock ./build/eklow-vision-qwen --mmproj "$MMPROJ" "${imgs[@]}" --max-tokens "$tok" --min-tokens "$MIN" --out "$e" "${flags[@]}" > "$e/log.txt" 2>&1
    k=0
    for i in "$@"; do
      echo "== $part budget $tok $(basename "$i"): oracle $(grep -o '"nx": [0-9]*, "ny": [0-9]*' "$d/images.json" | sed -n "$((k+1))p"), eklow $(grep -o '"nx": [0-9]*, "ny": [0-9]*' "$e/images.json" | sed -n "$((k+1))p")"
      "$OUT/compare_rows" "$e/image_$k.f32" "$d/image_$k.f32" 2048 | head -3 | tr '\n' ' '; echo
      k=$((k+1))
    done
  done
done
