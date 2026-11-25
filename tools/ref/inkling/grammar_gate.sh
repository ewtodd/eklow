#!/usr/bin/env bash
# Inkling 2 Mini's tool-call grammar against llama.cpp c74c71a6 (DECISIONS 2026-11-24, M4): the gate eklow-grammar-test-inkling.
#   grammar_gate.sh WORKDIR GGUF ORACLE_BIN_DIR TEST_BIN M0_DIR [FUZZ] [PER] [SEED]
# M0_DIR holds m0.py's cases.json and the chat oracle's chat.json (m0.sh). Host only, no GPU. Steps:
#   requests: M0's tool requests plus FUZZ fuzzed ones (random tool names with Unicode, quotes, escapes)
#   eklow-chat-oracle: the grammar llama-server builds for each
#   gen: PER sequences per grammar (valid outputs, mutations, random walks, M0's outputs, single tokens)
#   eklow-grammar-oracle (JOBS shards in parallel): which tokens llama.cpp's grammar sampler allows at every step
#   check: eklow's allowed sets, accept() and end-of-generation against them
set -euo pipefail
work=$1 gguf=$2 oracle=$3 test=$4 m0=$5 fuzz=${6:-150} per=${7:-60} seed=${8:-1}
jobs=${JOBS:-6}  # at the lowest priority: the machine is someone's desktop
export LD_LIBRARY_PATH=/run/opengl-driver/lib
mkdir -p "$work"
"$test" requests --m0-cases "$m0/cases.json" --out "$work" --fuzz "$fuzz" --seed "$seed"
"$oracle/eklow-chat-oracle" --model "$gguf" --cases "$work/requests.json" --out "$work/chat.json" > "$work/chat-oracle.log" 2>&1
"$oracle/eklow-grammar-oracle" --model "$gguf" --dump-vocab "$work/vocab.txt" > /dev/null 2>&1
"$test" check-vocab --model "$gguf" --vocab "$work/vocab.txt"
"$test" gen --model "$gguf" --chat "$work/chat.json" --meta "$work/meta.json" --m0-chat "$m0/chat.json" --out "$work" \
  --per "$per" --shards "$jobs" --seed "$seed"
start=$(date +%s)
for k in $(seq 0 $((jobs - 1))); do
  nice -n 19 "$oracle/eklow-grammar-oracle" --model "$gguf" --cases "$work/cases-$k.txt" --out "$work/results-$k.txt" > "$work/oracle-$k.log" 2>&1 &
done
wait
echo "oracle: $(( $(date +%s) - start )) s on $jobs processes"
rc=0
for k in $(seq 0 $((jobs - 1))); do
  "$test" check --model "$gguf" --dir "$work" --shard "$k" > "$work/check-$k.txt" || rc=1
done
cat "$work"/check-*.txt | awk '/^SUMMARY/ {for (i = 2; i <= NF; i++) {split($i, a, "="); if (a[1] != "allowed_us_mean") s[a[1]] += a[2]}} /^  CAT/ {c[$2] += $3}
  END {printf "TOTAL"; for (k in s) printf " %s=%d", k, s[k]; printf "\n"; for (k in c) printf "  %s %d\n", k, c[k]}'
exit $rc
