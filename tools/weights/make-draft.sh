#!/usr/bin/env bash
# make-draft.sh inco|jzinno: a DFlash 2 draft as GGUFs (BF16 and Q8_0) with the pinned converter. The converter
# needs the target's config and tokenizer (--target-model-dir): stock for Inco's draft, Ornith for jzinno's,
# both left in $WORK/hf/MODEL by make-gguf.sh. Outputs $WORK/gguf/drafts/NAME-{BF16,Q8_0}.gguf.
source "$(dirname "$0")/lib.sh"
case $1 in
inco)   D_REPO=incoai/Qwen3.6-35B-A3B-DFlash2;   D_REV=51ef7b6923; T=stock;  N=Qwen3.6-35B-A3B-DFlash2-inco ;;
jzinno) D_REPO=jzinno/Ornith-1.5-35B-A3B-DFlash2; D_REV=9b4852c05f; T=ornith; N=Ornith-1.5-35B-A3B-DFlash2-jzinno ;;
*) echo "inco or jzinno" >&2; exit 1 ;;
esac
mkdir -p "$WORK/gguf/drafts"
hub_fetch "$D_REPO" "$D_REV" "$WORK/hf/draft-$1" '^assets/'
python3 "$LLAMA_SRC/convert_hf_to_gguf.py" "$WORK/hf/draft-$1" --target-model-dir "$WORK/hf/$T" \
    --outtype bf16 --outfile "$WORK/gguf/drafts/$N-BF16.gguf"
"$LLAMA_CPU/llama-quantize" "$WORK/gguf/drafts/$N-BF16.gguf" "$WORK/gguf/drafts/$N-Q8_0.gguf" Q8_0
(cd "$WORK/gguf/drafts" && sha256sum "$N"-*.gguf | tee "$N.SHA256SUMS")
python3 -I "$(dirname "$0")/tensor_types.py" "$WORK/gguf/drafts/$N-BF16.gguf"
