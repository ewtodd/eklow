#!/usr/bin/env bash
# serve_test.py basic on the pool build (--kv-pool 262144, production shape) for: Sage with --verification on (after the developer fix),
# Sage with it off, and stock Qwen3.6 EKK8 (stock template): which failures are the server's and which Sage's longer thinking.
set -u
cd ~/eklow-wt-sagepool
B=~/eklow-sage-2026-11-07; M=~/models
busy() { systemctl is-active --quiet eklow-ornith || systemctl is-active --quiet eklow-gemma; }
PROD=(--host 127.0.0.1 --port 8093 --kv-int8 --slots 2 --ctx 262144 --kv-pool 262144 --snapshot-mib 4096 --group 8 --group-busy 4096
      --draft $M/drafts/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf --mmproj $M/mmproj-Ornith-1.5-35B-A3B-bf16.gguf
      --embed-model $M/embeddinggemma-2-Q8_0.gguf --embed-port 8094 --embed-alias embeddinggemma-2 --embed-mmproj $M/mmproj-embeddinggemma-2-Q8_0.gguf --image-tokens 280
      --expert-profile data/qwen36-experts.json)
run() {  # run NAME args...
  local name=$1; shift
  busy && { echo "an eklow service is up: stopping"; exit 1; }
  echo "== $name $(date +%T)"
  setsid flock ~/eklow-perf/gpu.lock nix develop path:. -c ./build/eklow-server "${PROD[@]}" "$@" > $B/srv2-$name.log 2>&1 < /dev/null &
  local srv=$!
  for i in $(seq 1 120); do curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null | grep -q 200 && break; sleep 3; done
  nix develop path:. -c python3 tools/server/serve_test.py --url http://127.0.0.1:8093 basic > $B/servetest2-$name.txt 2>&1
  grep -E "^FAIL|^FAILED|all passed|^ok .*developer" $B/servetest2-$name.txt | cut -c1-200
  pkill -TERM -g $srv 2>/dev/null; sleep 8; pkill -KILL -g $srv 2>/dev/null; wait $srv 2>/dev/null; sleep 3
}
run sage-verify --model $M/Sage-A1.5-Q4_K-Q8_0.gguf --alias sage-a1.5 --verification on
run sage-plain --model $M/Sage-A1.5-Q4_K-Q8_0.gguf --alias sage-a1.5
run stock --model $M/stock-EKK8.gguf --alias qwen3.6-35b-a3b
echo "== done $(date +%T)"
