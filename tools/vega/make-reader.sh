#!/usr/bin/env bash
# make-reader.sh OUT.gguf: vega's reader as eklow loads it, Qwen/Qwen3.5-0.8B at 2fc06364 converted by the converter of the
# llama.cpp flake.nix pins, in BF16 (the weights as published; eklow makes fp16 tiles of them, exactly) and without the MTP
# block. Needs a Python with torch, transformers, safetensors and numpy (tools/weights/shell.nix has one); the checkpoint
# comes from the Hugging Face cache or the Hub. The result's sha256 was 4f4a6b43e0d3cc85... on 2026-11-10.
set -euo pipefail
OUT=$1
REPO=$(cd "$(dirname "$0")/../.." && pwd)
LLAMA_SRC=${LLAMA_SRC:-$(nix eval --raw --impure --expr "(builtins.getFlake \"path:$REPO\").inputs.llama-cpp.outPath")}
REV=2fc06364715b967f1860aea9cf38778875588b17
SRC=$(python3 -c "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen3.5-0.8B', revision='$REV', allow_patterns=['*.json', '*.safetensors', '*.jinja']))")
PYTHONPATH=$LLAMA_SRC/gguf-py python3 "$LLAMA_SRC/convert_hf_to_gguf.py" "$SRC" --outtype bf16 --no-mtp --outfile "$OUT.part"
mv "$OUT.part" "$OUT"
sha256sum "$OUT"
