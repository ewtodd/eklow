#!/usr/bin/env bash
# kv_pool_test.py at a small pool on production's other flags, with the card's memory polled: the reference server (default pool, two slots of
# 8,192) records, the pooled one (two slots over 8,192 positions) checks. Run on fiona from the worktree:  pool.sh OUTDIR [extra server flags]
set -u
B=$1; shift
T=$PWD; S=/nix/store
for u in eklow-sage eklow-ornith; do systemctl is-active --quiet $u && { echo "$u is active: not starting"; exit 1; }; done
mkdir -p $B
PY="nix develop path:. -c python3"
run() {  # name, extra flags
  local name=$1; shift
  : > $B/$name.vram
  setsid flock ~/eklow-perf/gpu.lock timeout 3600 nix develop path:. -c ./build/eklow-server \
    --model $S/hs1g9p0q2gpk8bwizkmv6mwrpr4asrjc-Sage-A1.5-Q4_K-Q8_0.gguf --alias sage-a1.5 --host 127.0.0.1 --port 8097 --slots 2 --ctx 8192 \
    --kv-int8 --expert-profile data/qwen36-experts.json --snapshot-mib 1024 --draft $S/pfrc3qzdjv0jlsfpcky5mrz55qlfffkg-Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf \
    --group 8 --group-busy 4096 --verification on --mmproj $S/b23r16sm3j0pq7lr0lyyva2gw2swkkl6-mmproj-Ornith-1.5-35B-A3B-bf16.gguf \
    --embed-model $S/rxga3ywxisxmvmblmaqd9i40cr472wfr-embeddinggemma-2-Q8_0.gguf --embed-port 8098 --embed-alias embeddinggemma-2 \
    --embed-mmproj $S/ns0n6xhg02xrc8q5dqv6ck25yavlvm4v-mmproj-embeddinggemma-2-Q8_0.gguf --image-tokens 280 "$@" > $B/$name.log 2>&1 < /dev/null &
  for i in $(seq 1 300); do grep -aq "listening on 127.0.0.1:8097" $B/$name.log && break; sleep 3; done
  P=$(pgrep -f "^./build/eklow-server --model" | head -1)
  [ -n "$P" ] || { echo "no server"; return 1; }
  nix develop path:. -c python3 -I tools/margin/poll.py $P $B/$name.vram &
  sleep 5; grep -aE "experts on the card" $B/$name.log | cut -c1-250
  if [ $name = ref ]; then $PY tools/server/kv_pool_test.py --url http://127.0.0.1:8097 record --out $B/ref.json; else $PY tools/server/kv_pool_test.py --url http://127.0.0.1:8097 check --ref $B/ref.json --log $B/$name.log; fi
  echo "kv_pool_test $name exit $?"
  kill -TERM $P; for i in $(seq 1 30); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -KILL $P 2>/dev/null; sleep 3
  echo "$name card MiB: first $(head -1 $B/$name.vram | cut -d' ' -f2), max $(sort -k2 -n $B/$name.vram | tail -1 | cut -d' ' -f2)"
}
run ref "$@"
run pool --kv-pool 8192 "$@"
pgrep -f "^./build/eklow-server --model" && echo "WARNING: a server is still running"
