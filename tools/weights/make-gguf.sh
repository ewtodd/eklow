#!/usr/bin/env bash
# make-gguf.sh MODEL [STAGE...]: eklow's GGUFs of one Qwen3.6-35B-A3B finetune (MODEL = stock|ornith), with
# the pinned llama.cpp's converter and quantizer. Stages (default: all, in this order):
#   fetch    download the pinned checkpoint (sha256-checked) and the pinned bartowski imatrix
#   convert  safetensors -> BF16 GGUF, without the MTP/NextN block (eklow drafts with DFlash)
#   quant    BF16 -> Q8_0 and the mixes in $MIXES (EK8 EK4 EKK8 EKK6 EKI8)
#   check    tensor types of every output, sha256s -> $DATA/gguf/MODEL/SHA256SUMS
#   clean    delete the safetensors (stage "unhf") and the BF16 GGUF
# Run from `nix-shell tools/weights/shell.nix`. Outputs: $WORK/gguf/MODEL/MODEL-{Q8_0,EK8,EK4}.gguf.
source "$(dirname "$0")/lib.sh"
M=$1; shift
model_info "$M"
STAGES=${*:-fetch convert unhf quant check clean}
HF=$WORK/hf/$M
OUT=${OUT:-$WORK/gguf/$M}   # staging; move the keepers to $DATA/gguf
BF16=$WORK/$M-BF16.gguf
IM=$WORK/imatrix/$M.imatrix.gguf
THREADS=${THREADS:-32}
mkdir -p "$OUT" "$WORK/imatrix"

fetch() {
    hub_fetch "$M_REPO" "$M_REV" "$HF" '^assets/'
    local rev
    rev=$(curl -fsS "https://huggingface.co/api/models/$I_REPO/revision/$I_REV" | jq -r .sha)
    echo "imatrix $I_REPO @ $rev"
    mkdir -p "$WORK/imatrix"
    curl -fsSL --retry 8 -C - -o "$IM" "https://huggingface.co/$I_REPO/resolve/$rev/$I_FILE"
    local got; got=$(sha256sum "$IM" | cut -d' ' -f1)
    case $got in "$I_SHA"*) echo "imatrix sha256 $got ok";; *) echo "imatrix sha256 $got, expected $I_SHA..." >&2; exit 1;; esac
}

convert() {
    [ -s "$BF16" ] && { echo "$BF16 exists"; return; }
    # The converter prefers tokenizer_config.json's chat_template over chat_template.jinja (what transformers
    # uses). Ornith's repo ships stock Qwen3.6's tokenizer_config.json byte for byte, so convert from a view of
    # the checkpoint whose tokenizer_config.json has no chat_template: the converter then reads the .jinja.
    local src=$HF
    if [ -f "$HF/chat_template.jinja" ]; then
        src=$WORK/hf/$M-conv; rm -rf "$src"; mkdir -p "$src"
        for f in "$HF"/*; do ln -s "$f" "$src/"; done
        rm "$src/tokenizer_config.json"
        jq 'del(.chat_template)' "$HF/tokenizer_config.json" > "$src/tokenizer_config.json"
    fi
    python3 "$LLAMA_SRC/convert_hf_to_gguf.py" "$src" --outtype bf16 --no-mtp --outfile "$BF16.part"
    mv "$BF16.part" "$BF16"
    python3 -I "$(dirname "$0")/template_check.py" "$BF16" "$HF/chat_template.jinja"
}

# Tensor types by name (llama-quantize --tensor-type matches a regex anywhere in the tensor name, and the FIRST
# matching option wins, so a mix is built from its expert type and its dense type, never by appending):
#  routed experts (ffn_{gate,up,down}_exps): EXPERT type, with the importance matrix
#  "dense" matrices (GDN attn_qkv, attn_gate, ssm_out; attention attn_q, attn_output; shared expert
#       ffn_{gate,up,down}_shexp): DENSE type (q8_0 = the Q8_0 base type, no option needed)
#  attn_k, attn_v, ssm_alpha, ssm_beta, token_embd: Q8_0; output: Q6_K; norms, ssm_a, ssm_dt, ssm_conv1d and the
#       router as the converter wrote them (F32); the shared expert's gate (ffn_gate_inp_shexp, one row) F32.
#  name   experts   dense
#  EK8    q4_0      q8_0    what eklow's kernels read today
#  EK4    q4_0      q4_0
#  EKD6   q4_0      q6_k
#  EKK8   q4_k      q8_0
#  EKK6   q4_k      q6_k
#  EKI8   iq4_xs    q8_0
declare -A MIX_EXPERT=( [EK8]=q4_0 [EK4]=q4_0 [EKD6]=q4_0 [EKK8]=q4_k [EKK6]=q4_k [EKI8]=iq4_xs )
declare -A MIX_DENSE=(  [EK8]=q8_0 [EK4]=q4_0 [EKD6]=q6_k  [EKK8]=q8_0 [EKK6]=q6_k [EKI8]=q8_0 )
MIXES=${MIXES:-EK8 EK4 EKD6 EKK8 EKK6 EKI8}

mix_args() {   # mix_args NAME -> ARGS
    local e=${MIX_EXPERT[$1]} d=${MIX_DENSE[$1]}
    ARGS=(--imatrix "$IM" --tensor-type "ffn_(gate|up|down)_exps=$e" --tensor-type 'ffn_gate_inp_shexp=f32'
          --output-tensor-type q6_k --token-embedding-type q8_0)
    if [ "$d" != q8_0 ]; then
        for t in attn_qkv attn_gate ssm_out attn_q attn_output 'ffn_(gate|up|down)_shexp'; do
            ARGS+=(--tensor-type "$t=$d")
        done
    fi
}

mix_one() {   # mix_one NAME: quantize $BF16 as NAME (Q8_0 is the plain base)
    local name=$1 out=$OUT/$M-$1.gguf ARGS=()
    [ -s "$out" ] && return
    [ "$name" = Q8_0 ] || mix_args "$name"
    "$LLAMA_CPU/llama-quantize" "${ARGS[@]}" "$BF16" "$out.part" Q8_0 "$THREADS"
    mv "$out.part" "$out"
}

BASE=${BASE-Q8_0}   # BASE= (empty) skips the plain Q8_0
quant() { for x in $BASE $MIXES; do mix_one "$x"; done; }

check() {
    for x in $BASE $MIXES; do
        [ -s "$OUT/$M-$x.gguf" ] && { python3 -I "$(dirname "$0")/tensor_types.py" "$OUT/$M-$x.gguf" || true; }
    done
    (cd "$OUT" && sha256sum "$M"-*.gguf | tee SHA256SUMS)
}

unhf() { rm -f "$HF"/*.safetensors; }   # config and tokenizer stay: the DFlash drafts convert against them
clean() { rm -rf "$HF" "$BF16"; }

for s in $STAGES; do echo "== $s $(date -Is)"; "$s"; done
