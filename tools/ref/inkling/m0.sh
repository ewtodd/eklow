#!/usr/bin/env bash
# Inkling 2 Mini's M0 ground truth from llama.cpp c74c71a6 on the 4090 (DECISIONS 2026-11-24).
#   m0.sh OUT_DIR GGUF ORACLE_BIN_DIR [NAMES]   (NAMES: the forward prompts to run, default all four)
# OUT_DIR must already hold m0.py's cases.json, strings.json and prompts/ (m0.py cases, m0.py prompts). Writes:
#   chat.json      the template, generation prompt, grammar and parsed outputs of every case (eklow-chat-oracle)
#   tokens.json    llama.cpp's tokens of every string (plain and with special tokens parsed)
#   fwd/NAME/      for each forward prompt: tokens.json, index.json, the dumped tensors, logits.f32 (the dumped
#                  positions), gen.json and logits_gen.f32 (a greedy continuation)
# The forward runs all on the card (--ngl 99), fp32 K/V, no flash attention; the long prompts prefill in chunks and
# dump only their last positions. llama.cpp reserves the attention scores for the whole context at the ubatch size
# (no flash attention), so the longer the prompt the smaller the ubatch that fits beside the 16 GB of weights. Holds
# ~/eklow-qwen-data/4090.lock for the forward passes.
set -euo pipefail
out=$1 gguf=$2 bin=$3 names=${4:-short tools long deep}
export LD_LIBRARY_PATH=/run/opengl-driver/lib
cd "$out"
"$bin/eklow-chat-oracle" --model "$gguf" --cases cases.json --out chat.json
"$bin/eklow-chat-oracle" --model "$gguf" --tokenize strings.json --out tokens.json

# every slot's stream and router; the detailed tensors on the slots named in $detail
# (the router picks as ffn_moe_argsort: ffn_moe_topk is a view of it that is not contiguous; its first 8 are the picks)
every="l_out-,ffn_inp-,ffn_out-,attn_out-,ffn_moe_argsort-,ffn_moe_weights_scaled-,result_norm,result_output"
detail_slots="0 1 5 6 11 17 18 23 24 29 30 35 41 47"
detail=""
for s in $detail_slots; do
  for t in Qcur Kcur Vcur Rcur kqv_out ffn_moe_logits ffn_moe_situ ffn_moe_out; do detail+=",$t-$s\$"; done
done

exec 9>"$HOME/eklow-qwen-data/4090.lock"
flock 9
run() {  # name ubatch dump_last generate tensors
  [[ " $names " == *" $1 "* ]] || return 0
  mkdir -p "fwd/$1"
  "$bin/eklow-oracle" --model "$gguf" --out "fwd/$1" --prompt-file "prompts/$1.txt" --ngl 99 \
    --ubatch "$2" --dump-last "$3" --generate "$4" --tensors "$5" 2>"fwd/$1/oracle.log"
  tail -2 "fwd/$1/oracle.log"
}
# short: every position, and the attention's bias, mask and probabilities on every slot
run short 0 0 64 "$every$detail,kq_b-,kq_soft_max-,ffn_moe_gate_up-"
run tools 1024 512 64 "$every$detail"
run long 256 256 32 "$every$detail"
run deep 128 128 32 "$every"
du -sh fwd/*
