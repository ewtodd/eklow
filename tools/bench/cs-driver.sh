#!/usr/bin/env bash
# check-source on fiona, both models, alternating bench by bench. Detached, resumable: run it again after any stop.
#   cd ~/eklow-wt-qrelease && setsid nohup tools/bench/cs-driver.sh tools/bench/cs-plan.txt ~/eklow-qrelease-out/cs > ~/eklow-qrelease-out/cs-driver.log 2>&1 < /dev/null &
# PLAN lines: MODEL(stock|ornith) BENCH YEARS OFFSET LIMIT (YEARS only for aime; '-' otherwise), in the order to run. A slice
# whose summary JSON (OUTDIR/MODEL/TAG.json) exists is done and skipped; the one that was running starts over (the harness
# rewrites its rows file at start), which is why the slices are small. The server (cs-serve.sh, under the GPU lock) is
# restarted when the model changes and stopped at the end. Refuses to run while eklow.service is active.
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
plan=${1:?}; outdir=${2:?}
mkdir -p "$outdir"; outdir=$(realpath "$outdir")
tree=$(cd "$(dirname "$0")/../.." && pwd)
cs=$HOME/check-source
port=18181
cur=
up() { curl -sf --max-time 5 localhost:$port/health >/dev/null; }
stop_server() {
  pkill -f '[c]s-serve.sh' 2>/dev/null
  for i in $(seq 120); do flock -n ~/eklow-perf/gpu.lock true && return 0; sleep 2; done
  echo "$(date +%T) the card lock is still held"; return 1
}
start_server() {
  case $1 in
    stock) gguf=stock-EKK8.gguf; prof=data/qwen36-experts.json; alias=qwen3.6-35b-a3b ;;
    ornith) gguf=ornith-EKK8.gguf; prof=data/ornith-1.5-experts.json; alias=ornith-1.5-35b-a3b ;;
    *) echo "model: stock or ornith"; return 1 ;;
  esac
  systemctl is-active --quiet eklow && { echo "eklow.service is active; stopping"; return 1; }
  pgrep -f '[e]klow-server --model' >/dev/null && stop_server
  echo "$(date +%T) starting the server for $1"
  (cd "$tree" && setsid nohup flock ~/eklow-perf/gpu.lock bash tools/bench/cs-serve.sh "$outdir/serve-$1.log" ~/models/$gguf $prof \
     ~/models/drafts/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf $alias </dev/null >/dev/null 2>&1 &)
  for i in $(seq 1800); do up && return 0; sleep 2; done
  echo "$(date +%T) the server did not come up"; return 1
}
while read -r which bench years offset limit; do
  [ -z "$which" ] && continue
  case $which in \#*) continue ;; esac
  tag=$bench${years/-/}-$offset-$limit
  out=$outdir/$which/$tag.json
  [ -s "$out" ] && continue
  mkdir -p "$outdir/$which"
  if [ "$cur" != "$which" ] || ! up; then
    start_server $which || { echo "$(date +%T) stopping"; exit 1; }
    cur=$which
  fi
  case $which in stock) alias=qwen3.6-35b-a3b ;; ornith) alias=ornith-1.5-35b-a3b ;; esac
  args=(--base-url http://127.0.0.1:$port --model $alias --level medium --seed 1234 --concurrency 2 --timeout 7200
        --offset $offset --limit $limit --out "$out" --weights "EKK8 (Q4_K experts, Q8_0 dense, Q6_K head)" --activations "int8 small batches, fp16 large" --kv-cache int8)
  case $bench in
    gsm8k-chat|math500) args+=(--max-tokens 65536) ;;
    aime) args+=(--max-tokens 32768 --years $years) ;;
  esac
  echo "$(date +%T) $which $tag"
  (cd /tmp && nix develop "path:$cs" --command bash -c "PYTHONPATH=$cs/src python -m check_source $bench \"\$@\"" _ "${args[@]}") </dev/null >"$outdir/$which/$tag.log" 2>&1
  tail -1 "$outdir/$which/$tag.log"
done < "$plan"
stop_server
echo "$(date +%T) all slices done"
