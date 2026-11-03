#!/usr/bin/env bash
# check-source (~/Software/check-source, run as it is) against eklow-server on fiona, for one model, in slices that
# resume: the harness rewrites its rows file when it starts, so a stopped run is resumed by running this again (a slice with
# a summary JSON is done and skipped; the one that was running starts over).
#   tools/bench/cs-run.sh MODEL(stock|ornith) PLAN OUTDIR
# PLAN lines: BENCH YEARS OFFSET LIMIT (YEARS only for aime; '-' otherwise). Runs on e-desktop, through an ssh tunnel to the
# server that cs-serve.sh starts on fiona (and that stops itself when idle).
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
which=${1:?}; plan=${2:?}; outdir=${3:?}
case $which in
  stock) gguf=stock-EKK8.gguf; prof=data/qwen36-experts.json; alias=qwen3.6-35b-a3b ;;
  ornith) gguf=ornith-EKK8.gguf; prof=data/ornith-1.5-experts.json; alias=ornith-1.5-35b-a3b ;;
  *) echo "model: stock or ornith" >&2; exit 2 ;;
esac
mkdir -p "$outdir/$which"
cs=$HOME/Software/check-source
fiona_tree=eklow-wt-qrelease
tunnel=
stop() { [ -n "$tunnel" ] && kill $tunnel 2>/dev/null; }
trap stop EXIT
up() { curl -sf --max-time 5 localhost:18181/health >/dev/null; }
ensure() {
  up && return 0
  [ -n "$tunnel" ] && kill $tunnel 2>/dev/null
  # a server may be up on fiona with the tunnel dead
  tunnel=
  ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N -L 18181:127.0.0.1:18181 fiona & tunnel=$!
  sleep 3
  up && return 0
  echo "$(date +%T) starting the server for $which"
  ssh fiona "cd ~/$fiona_tree && systemctl is-active 'eklow*.service' >/dev/null && exit 3; pgrep -f 'eklow-server --model' >/dev/null && exit 4
    setsid nohup flock ~/eklow-perf/gpu.lock bash tools/bench/cs-serve.sh ~/eklow-qrelease-out/cs-$which.serve.log ~/models/$gguf $prof ~/models/drafts/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf $alias </dev/null >/dev/null 2>&1 &"
  rc=$?; if [ $rc -ne 0 ]; then echo "cannot start the server on fiona (rc $rc: 3 = an eklow service active, 4 = a server is already running)"; return 1; fi
  for i in $(seq 1800); do up && return 0; sleep 1; done
  return 1
}
while read -r bench years offset limit; do
  [ -z "$bench" ] && continue
  case $bench in \#*) continue ;; esac
  tag=$bench${years/-/}-$offset-$limit
  out=$outdir/$which/$tag.json
  [ -s "$out" ] && continue
  ensure || { echo "no server; stopping"; exit 1; }
  args=(--base-url http://127.0.0.1:18181 --model $alias --level medium --seed 1234 --concurrency 2 --timeout 7200
        --offset $offset --limit $limit --out "$out" --weights "EKK8 (Q4_K experts, Q8_0 dense, Q6_K head)" --activations "int8 small batches, fp16 large" --kv-cache int8)
  case $bench in
    gsm8k-chat|math500) args+=(--max-tokens 65536) ;;
    aime) args+=(--max-tokens 32768 --years $years) ;;
  esac
  echo "$(date +%T) $which $tag"
  (cd /tmp && nix develop "path:$cs" --command bash -c "PYTHONPATH=$cs/src python -m check_source $bench \"\$@\"" _ "${args[@]}") >"$outdir/$which/$tag.log" 2>&1
  tail -1 "$outdir/$which/$tag.log"
done < "$plan"
ssh fiona "pkill -f '[c]s-serve.sh'" 2>/dev/null
echo "$(date +%T) $which done"
