#!/usr/bin/env bash
# Inkling 2 Mini's M2 checks on the card (DECISIONS 2026-11-24, "Inkling M2"): the serving engine over M0's four prompts and
# their greedy continuations, and the short prompt with every layer continued from the reference's stream.
#   m2_card.sh BUILD_DIR GGUF DATA OUT [PROFILE]
# DATA holds m0/NAME/{tokens,gen}.json and m1/short/l_out-*.f32 (eklow-ref-run's dump of every position); OUT gets one dump
# directory per run, for tools/ref/inkling/compare.py beside M0's and M1's on the machine that has them. PROFILE: a routing
# profile that places the experts (data/inkling-2-mini-experts.json: what does not fit beside the 33k prompt's K/V runs on the
# CPU). Takes
# ~/eklow-perf/gpu.lock around each run; never run it while eklow-bolt is active.
set -euo pipefail
build=$1 gguf=$2 data=$3 out=$4 profile=${5:-}
if systemctl is-active --quiet 'eklow*'; then echo "an eklow service is active: the card is production" >&2; exit 1; fi
run() {  # name out-subdir dump-last extra args...
  local name=$1 dir=$out/$2 last=$3
  shift 3
  mkdir -p "$dir"
  flock ~/eklow-perf/gpu.lock "$build/eklow-engine-run-inkling" --model "$gguf" --out "$dir" --tokens-json "$data/m0/$name/tokens.json" \
    --dump-last "$last" ${profile:+--experts "$profile"} "$@" 2>&1 | tail -3
}
run short short-tf 0 --teacher "$data/m1/short" \
  --tensors l_out-,ffn_inp-,ffn_out-,attn_out-,ffn_moe_topk-,result_norm,Qcur-,Kcur-,Vcur-,Rcur-,kqv_out-,ffn_moe_logits-
run short short 0 --gen-json "$data/m0/short/gen.json" --tensors l_out-,ffn_moe_topk-,result_norm
run tools tools 512 --gen-json "$data/m0/tools/gen.json" --tensors l_out-,ffn_moe_topk-,result_norm
run long long 256 --gen-json "$data/m0/long/gen.json" --tensors l_out-,ffn_moe_topk-,result_norm
run deep deep 128 --gen-json "$data/m0/deep/gen.json" --tensors l_out-,ffn_moe_topk-,result_norm
