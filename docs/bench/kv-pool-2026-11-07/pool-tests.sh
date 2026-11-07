#!/usr/bin/env bash
# The shared K/V pool's card tests, after Sage's replay (test/sage-pool = sage/template + serve/kv-pool, built in ~/eklow-wt-sagepool):
#  1. checkq ekk8 (the engine at the default pool must be unchanged; bench.sh takes gpu.lock itself)
#  2. kv_pool_test.py: reference server at --slots 2 --ctx 8192, then the pool --kv-pool 8192 (Sage EKK8 + draft)
#  3. production shape with --kv-pool 262144: startup (experts on the card), serve_test.py basic, serve_perf depth 8k/32k and GSM8K
#  4. the same speed runs at the default pool, for the comparison on the same build
set -u
cd ~/eklow-wt-sagepool
B=~/eklow-sage-2026-11-07; M=~/models
log() { echo "== $* $(date +%T)"; }
busy() { systemctl is-active --quiet eklow-ornith || systemctl is-active --quiet eklow-gemma; }
busy && { echo "an eklow service is up: stopping"; exit 1; }
log checkq; bash ~/eklow-perf/bench.sh checkq ekk8 > $B/checkq.txt 2>&1; tail -20 $B/checkq.txt
COMMON=(--model $M/Sage-A1.5-Q4_K-Q8_0.gguf --alias sage-a1.5 --host 127.0.0.1 --port 8093 --kv-int8 --expert-profile data/qwen36-experts.json
        --draft $M/drafts/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf --chat-template sage --verification on)
PROD=(--slots 2 --ctx 262144 --snapshot-mib 4096 --group 8 --group-busy 4096 --mmproj $M/mmproj-Ornith-1.5-35B-A3B-bf16.gguf
      --embed-model $M/embeddinggemma-2-Q8_0.gguf --embed-port 8094 --embed-alias embeddinggemma-2 --embed-mmproj $M/mmproj-embeddinggemma-2-Q8_0.gguf --image-tokens 280)
serve() {  # serve NAME args...: start under the lock, wait for health; the caller stops it with stop
  local name=$1; shift
  busy && { echo "an eklow service is up: stopping"; exit 1; }
  setsid flock ~/eklow-perf/gpu.lock nix develop path:. -c ./build/eklow-server "${COMMON[@]}" "$@" > $B/srv-$name.log 2>&1 < /dev/null &
  SRV=$!
  for i in $(seq 1 120); do curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8093/health 2>/dev/null | grep -q 200 && break; sleep 3; done
  grep -E "experts on the card|K/V pool|weights" $B/srv-$name.log | cut -c1-260
}
stop() { pkill -TERM -g $SRV 2>/dev/null; kill $SRV 2>/dev/null; sleep 8; pkill -KILL -g $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 3; }
log kv_pool_test reference; serve ref --slots 2 --ctx 8192
nix develop path:. -c python3 tools/server/kv_pool_test.py --url http://127.0.0.1:8093 record --out $B/kvpool-ref.json > $B/kvpool-ref.txt 2>&1; tail -5 $B/kvpool-ref.txt; stop
log kv_pool_test pool; serve pool --slots 2 --ctx 8192 --kv-pool 8192
nix develop path:. -c python3 tools/server/kv_pool_test.py --url http://127.0.0.1:8093 check --ref $B/kvpool-ref.json --log $B/srv-pool.log > $B/kvpool-check.txt 2>&1; tail -12 $B/kvpool-check.txt; stop
for shape in pool262k default; do
  log production shape $shape
  if [ $shape = pool262k ]; then serve $shape "${PROD[@]}" --kv-pool 262144; else serve $shape "${PROD[@]}"; fi
  [ $shape = pool262k ] && { nix develop path:. -c python3 tools/server/serve_test.py --url http://127.0.0.1:8093 basic > $B/servetest-$shape.txt 2>&1; tail -4 $B/servetest-$shape.txt; }
  nix develop path:. -c python3 tools/bench/serve_perf.py --url http://127.0.0.1:8093 --model sage-a1.5 depth --docs ~/eklow-qrelease-data/docs --depths 8192,32768 --repeats 5 > $B/perf-depth-$shape.txt 2>&1; tail -4 $B/perf-depth-$shape.txt
  nix develop path:. -c python3 tools/bench/serve_perf.py --url http://127.0.0.1:8093 --model sage-a1.5 gsm --requests ~/eklow-qrelease-data/gsm48.jsonl --concurrency 1 > $B/perf-gsm-$shape.txt 2>&1; tail -3 $B/perf-gsm-$shape.txt
  stop
done
log all done
