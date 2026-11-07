#!/usr/bin/env bash
# Expert formats on the real engine (stock Qwen3.6; the routing does not depend on the finetune's weights much): decode, a verify
# block of 4 and prefill at 8k, at fixed experts on the card. Small K/V (2 x 16k) so Q8_0's cold experts fit in pinned RAM.
# Stops before any run if an eklow service is up: the card is the owner's then.
set -u
P=/nix/store/f7401lw4042szq2fn758aii6iwyk8zwl-eklow-0.0.1
B=$HOME/eklow-bench-2026-11-07; M=$HOME/models
PROF=$P/share/eklow/qwen36-experts.json
busy() { systemctl is-active --quiet eklow-ornith || systemctl is-active --quiet eklow-gemma; }
bench() {  # bench NAME FILE R
  if busy; then echo "an eklow service is up: stopping before $1"; exit 1; fi
  echo "== $1 R=$3 $(date +%T)"
  "$P/bin/eklow-engine-bench" --model "$2" --slots 2 --max-ctx 16384 --kv-int8 --expert-profile "$PROF" --experts-on-card "$3" \
    --prefill 8192 --decode-ctx 8192 --decode-slots 1 --decode-steps 128 --verify 4 --verify-ctx 8192 --verify-steps 24 \
    --json "$B/$1-R$3.json" > "$B/$1-R$3.log" 2>&1
  echo "exit $? $(date +%T)"; grep -iE 'tok/s|ms a step|pinned|error|what' "$B/$1-R$3.log" | tail -8 | cut -c1-200
}
bench q4k "$M/stock-EKK8.gguf" 48
bench q4k "$M/stock-EKK8.gguf" 92
bench q8 "$M/stock-Q8_0.gguf" 48
bench q8 "$M/stock-Q8_0.gguf" 64
Q=$(nix build nixpkgs#llama-cpp --no-link --print-out-paths 2>/dev/null)/bin/llama-quantize
for f in q6_k q5_k; do
  out=$M/stock-${f^^}E.gguf
  [ -s "$out" ] || { echo "== quantize $f $(date +%T)"; nice "$Q" --allow-requantize --tensor-type "ffn_(gate|up|down)_exps=$f" --output-tensor-type q6_k \
     "$M/stock-Q8_0.gguf" "$out.part" Q8_0 12 > "$B/quant-$f.log" 2>&1 && mv "$out.part" "$out"; echo "exit $? $(date +%T)"; tail -2 "$B/quant-$f.log"; }
done
bench q6k "$M/stock-Q6_KE.gguf" 63
bench q6k "$M/stock-Q6_KE.gguf" 32
bench q5k "$M/stock-Q5_KE.gguf" 75
bench q5k "$M/stock-Q5_KE.gguf" 39
echo "all done $(date +%T)"
