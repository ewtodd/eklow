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
    python3 -I "$LLAMA_SRC/convert_hf_to_gguf.py" "$HF" --outtype bf16 --no-mtp --outfile "$BF16.part"
    mv "$BF16.part" "$BF16"
}

# Tensor types by name (llama-quantize --tensor-type matches a regex anywhere in the tensor name):
#  EK8: routed experts Q4_0 (imatrix); all other matrices Q8_0 (the Q8_0 base type); output Q6_K;
#       token_embd Q8_0. llama-quantize leaves norms, ssm_a, ssm_dt, ssm_conv1d and the router as converted;
#       the shared expert's gate (ffn_gate_inp_shexp, one row) is pinned to F32 like the router.
#  EK4: EK8 plus Q4_0 (imatrix) for GDN attn_qkv, attn_gate, ssm_out, attention attn_q, attn_output and
#       the shared expert (ffn_{gate,up,down}_shexp); attn_k, attn_v, ssm_alpha, ssm_beta stay Q8_0.
EK8_ARGS=(
    --tensor-type 'ffn_(gate|up|down)_exps=q4_0'
    --tensor-type 'ffn_gate_inp_shexp=f32'
    --output-tensor-type q6_k --token-embedding-type q8_0
)
EK4_ARGS=(
    "${EK8_ARGS[@]}"
    --tensor-type 'attn_qkv=q4_0' --tensor-type 'attn_gate=q4_0' --tensor-type 'ssm_out=q4_0'
    --tensor-type 'attn_q=q4_0' --tensor-type 'attn_output=q4_0'
    --tensor-type 'ffn_(gate|up|down)_shexp=q4_0'
)

# Mixes added 2026-10-31 (owner: quality per byte over every format the A770M can run, not only Q4_0):
#  EKK8: routed experts Q4_K (imatrix), everything else as EK8 (dense Q8_0)
#  EKK6: routed experts Q4_K, the dense matrices of EK4's list Q6_K, the rest as EK8
#  EKI8: routed experts IQ4_XS (imatrix), dense Q8_0
EKK8_ARGS=( "${EK8_ARGS[@]}" --tensor-type 'ffn_(gate|up|down)_exps=q4_k' )
EKK6_ARGS=(
    "${EKK8_ARGS[@]}"
    --tensor-type 'attn_qkv=q6_k' --tensor-type 'attn_gate=q6_k' --tensor-type 'ssm_out=q6_k'
    --tensor-type 'attn_q=q6_k' --tensor-type 'attn_output=q6_k'
    --tensor-type 'ffn_(gate|up|down)_shexp=q6_k'
)
EKI8_ARGS=( "${EK8_ARGS[@]}" --tensor-type 'ffn_(gate|up|down)_exps=iq4_xs' )
MIXES=${MIXES:-EK8 EK4 EKK8 EKK6 EKI8}

mix_one() {   # mix_one NAME: quantize $BF16 with NAME's arguments (Q8_0 is the plain base)
    local name=$1 out=$OUT/$M-$1.gguf args=()
    [ -s "$out" ] && return
    if [ "$name" != Q8_0 ]; then
        local v=${name}_ARGS[@]; args=(--imatrix "$IM" "${!v}")
    fi
    "$LLAMA_CPU/llama-quantize" "${args[@]}" "$BF16" "$out.part" Q8_0 "$THREADS"
    mv "$out.part" "$out"
}

quant() { for x in Q8_0 $MIXES; do mix_one "$x"; done; }

check() {
    for x in Q8_0 $MIXES; do
        python3 -I "$(dirname "$0")/tensor_types.py" "$OUT/$M-$x.gguf" || true
    done
    (cd "$OUT" && sha256sum "$M"-*.gguf | tee SHA256SUMS)
}

unhf() { rm -rf "$HF"; }
clean() { rm -rf "$HF" "$BF16"; }

for s in $STAGES; do echo "== $s $(date -Is)"; "$s"; done
