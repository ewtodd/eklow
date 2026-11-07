#!/usr/bin/env bash
# make-sage.sh: eklow's GGUF of InternScience/Sage-A1.5 (a Qwen3.6-35B-A3B finetune), the EKK8 mix (routed experts Q4_K under an
# importance matrix, every other matrix Q8_0, head Q6_K), made from InternScience's own Q8_0 GGUF rather than the BF16 checkpoint
# (DECISIONS 2026-11-07, "Sage-A1.5 on eklow"). Only the experts and the head are requantized; the other 629 tensors come out
# byte-identical to the Q8_0 (Q8_0 to Q8_0 is exact). Published as ethanwtodd/Sage-A1.5-Q4_K-Q8_0-GGUF.
#
#   make-sage.sh [OUTDIR]     (run from `nix-shell tools/weights/shell.nix`; needs the hf CLI with hf_xet on PATH)
#
# Two header keys InternScience's files lack are set: tokenizer.ggml.normalizer.nfc (Sage's tokenizer.json normalizes NFC; eklow's
# tokenizer refuses a Qwen GGUF without the key) and the model card's sampling defaults (eklow-server and llama.cpp read
# general.sampling.*). The MTP/NextN block (layer 40) is kept as InternScience's files have it; eklow skips it.
source "$(dirname "$0")/lib.sh"
OUT=${1:-$WORK/gguf/sage}
THREADS=${THREADS:-16}
SRC=$WORK/hf/sage-q8
mkdir -p "$OUT" "$SRC"
Q8_REPO=InternScience/Sage-A1.5-Q8_0-GGUF Q8_REV=1cbe2925983a13a6c09881b9bc088884dbfbc146
Q8_FILE=Sage-A1.5-Q8_0.gguf Q8_SHA=f7247f6faef3f6c616b953595e1fc7c3fbf257d02d9915af5fdf15e808ea8ee4
IM_REPO=mradermacher/Sage-A1.5-i1-GGUF IM_REV=6dd4c710e331ac64dee09c605b1625be7cce23c9
IM_FILE=Sage-A1.5.imatrix.gguf IM_SHA=41d264ff3e56152e3b217fd5f1d89264360d1181befb47bd5af26780352809ce
WANT=6d25f1417713e555aa5b731ab1cac37fe08a66bd1d0ed028476a7df3e8a46f0e

get() {  # get REPO REV FILE SHA: through hf (Xet storage; plain HTTP on these repos goes through a slow bridge), then the Hub's sha256
    [ -s "$SRC/$3" ] || HF_HUB_DISABLE_TELEMETRY=1 hf download "$1" "$3" --revision "$2" --local-dir "$SRC"
    local got; got=$(sha256sum "$SRC/$3" | cut -d' ' -f1)
    [ "$got" = "$4" ] || { echo "$3: sha256 $got, expected $4" >&2; exit 1; }
    echo "ok $3"
}
get "$Q8_REPO" "$Q8_REV" "$Q8_FILE" "$Q8_SHA"
get "$IM_REPO" "$IM_REV" "$IM_FILE" "$IM_SHA"

out=$(realpath "$OUT")/Sage-A1.5-Q4_K-Q8_0.gguf
# From the inputs' directory, with the imatrix by its bare name: the header's quantize.imatrix.file then reads
# "Sage-A1.5.imatrix.gguf" wherever this runs, and the output is the published file byte for byte.
cd "$SRC"
"$LLAMA_CPU/llama-quantize" --allow-requantize --imatrix "$IM_FILE" \
    --tensor-type 'ffn_(gate|up|down)_exps=q4_k' --tensor-type 'ffn_gate_inp_shexp=f32' \
    --output-tensor-type q6_k --token-embedding-type q8_0 \
    --override-kv tokenizer.ggml.normalizer.nfc=bool:true \
    --override-kv general.sampling.temp=float:1.0 --override-kv general.sampling.top_k=int:20 \
    --override-kv general.sampling.top_p=float:0.95 \
    "$Q8_FILE" "$out.part" Q8_0 "$THREADS"
mv "$out.part" "$out"
got=$(sha256sum "$out" | cut -d' ' -f1)
echo "$got  $(basename "$out")"
[ "$got" = "$WANT" ] && echo "the published file, byte for byte" || echo "NOT the published file ($WANT)"
