#!/usr/bin/env bash
# What eklow-server's card memory does after startup at production's shape (the deployed eklow-sage flags on a test port), stage by stage.
#   measure.sh OUTDIR "STAGE STAGE ..." [extra server flags]      (run on fiona from the worktree; OUTDIR is created)
# SERVER=path runs another build of the server; MODEL=file ALIAS=name another model (servetest: serve_test.py basic); perf is the decode at 8k and 32k, 5 repeats.
# Stages: deep (one 252k prompt, 3 decodes), deep1 (the same, 1 decode), block16 (the same at verify blocks of 16), twolong (88k + 167k at once),
# twodeep (a 100k and a 131k prompt at once, both generating), images (each test picture, then an embedding of each), embed8k (an embedding of
# 8,000 tokens), gsm2 (GSM8K two at a time, 16 requests), d8k / d32k (decode at 8k / 32k, 3 repeats).
# The server holds ~/eklow-perf/gpu.lock for its life; PRELOAD=libzetrace.so traces the allocations (tools/margin/zetrace.c) into OUTDIR/ze.txt.
set -u
B=$1; STAGES=$2; shift 2
T=$PWD; S=/nix/store; D=~/eklow-qrelease-data; V=~/eklow-vision-data
for u in eklow-sage eklow-ornith; do systemctl is-active --quiet $u && { echo "$u is active: not starting"; exit 1; }; done
mkdir -p $B; : > $B/marks.txt; : > $B/vram.txt
mark() { echo "$(date +%s) $*" >> $B/marks.txt; echo "== $* $(date +%T)"; }
PRE=(); [ -n "${PRELOAD:-}" ] && PRE=(env EKLOW_ZETRACE=$B/ze.txt LD_PRELOAD=$PRELOAD)
SERVER=${SERVER:-./build/eklow-server}; ALIAS=${ALIAS:-sage-a1.5}
setsid flock ~/eklow-perf/gpu.lock timeout 7200 nix develop path:. -c "${PRE[@]}" $SERVER \
  --model ${MODEL:-$S/hs1g9p0q2gpk8bwizkmv6mwrpr4asrjc-Sage-A1.5-Q4_K-Q8_0.gguf} --alias $ALIAS --host 127.0.0.1 --port 8095 --slots 2 --ctx 262144 --kv-int8 \
  --expert-profile data/qwen36-experts.json --snapshot-mib 4096 --draft $S/pfrc3qzdjv0jlsfpcky5mrz55qlfffkg-Qwen3.6-35B-A3B-DFlash2-Q8_0.gguf --group 8 --group-busy 4096 \
  --kv-pool 262144 --verification on --mmproj $S/b23r16sm3j0pq7lr0lyyva2gw2swkkl6-mmproj-Ornith-1.5-35B-A3B-bf16.gguf \
  --embed-model $S/rxga3ywxisxmvmblmaqd9i40cr472wfr-embeddinggemma-2-Q8_0.gguf --embed-port 8096 --embed-alias embeddinggemma-2 \
  --embed-mmproj $S/ns0n6xhg02xrc8q5dqv6ck25yavlvm4v-mmproj-embeddinggemma-2-Q8_0.gguf --image-tokens 280 "$@" > $B/server.log 2>&1 < /dev/null &
SRV=$!
for i in $(seq 1 300); do grep -aq "listening on 127.0.0.1:8095" $B/server.log && break; sleep 3; done
P=$(pgrep -f "^$SERVER --model" | head -1)
[ -n "$P" ] || { echo "the server did not come up"; pkill -TERM -g $SRV; exit 1; }
echo "server pid $P"; grep -aE "experts on the card|vision|projector|draft" $B/server.log | cut -c1-260
nix develop path:. -c python3 -I tools/margin/poll.py $P $B/vram.txt &
mark ready; sleep 15
PY="nix develop path:. -c python3"; U=http://127.0.0.1:8095; E=http://127.0.0.1:8096
sig() { [ -n "${PRELOAD:-}" ] && kill -USR1 $P; }
for st in $STAGES; do
  mark $st; sig
  case $st in
    deep)   $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 252000 --repeats 3 2>&1 | tail -3 ;;
    deep1)  $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 252000 --repeats 1 2>&1 | tail -3 ;;
    block16) $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 252000 --repeats 2 --block 16 2>&1 | tail -3 ;;
    twolong) $PY -I tools/margin/load.py $U $D/docs 88000 167000 2>&1 | cut -c1-200 ;;
    twodeep) $PY -I tools/margin/load.py $U $D/docs 100000 131072 2>&1 | cut -c1-200 ;;
    images) for f in $V/img/*; do mark image-$(basename $f); $PY tools/server/image_test.py $U $f --embed-url $E 2>&1 | tail -2; done ;;
    embed8k) $PY -I tools/margin/embed_long.py $E $D/docs/qwen-10000.txt embeddinggemma-2 8000 ;;
    gsm2)   head -16 $D/gsm48.jsonl > $B/gsm16.jsonl
            $PY tools/bench/serve_perf.py --url $U --model $ALIAS gsm --requests $B/gsm16.jsonl --concurrency 2 2>&1 | tail -1 ;;
    perf)   $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 8192,32768 --repeats 5 2>&1 | tail -4 ;;
    perfplain) $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 8192,32768 --repeats 5 --plain 5 2>&1 | tail -6 ;;
    servetest) $PY tools/server/serve_test.py --url $U basic 2>&1 | tee $B/servetest.txt | tail -45 ;;
    d8k)    $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 8192 --repeats 3 2>&1 | tail -2 ;;
    d32k)   $PY tools/bench/serve_perf.py --url $U --model $ALIAS depth --docs $D/docs --depths 32768 --repeats 3 2>&1 | tail -2 ;;
    *) echo "unknown stage $st" ;;
  esac
done
mark end; sleep 5
kill -TERM $P; for i in $(seq 1 30); do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -KILL $P 2>/dev/null
pkill -KILL -g $SRV 2>/dev/null; sleep 2
pgrep -f "^$SERVER --model" && echo "WARNING: a server is still running"
echo "== done $(date +%T)"
nix develop path:. -c python3 -I tools/margin/stages.py $B
