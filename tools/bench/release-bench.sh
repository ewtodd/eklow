#!/usr/bin/env bash
# The release numbers through eklow-server at the production shape, one model: the depth runs (prefill time to first token,
# decode with the draft and without, tokens a verify) and GSM8K with and without the draft, one and two at a time.
# Run on fiona from a built tree, detached; it takes the GPU lock itself and holds it for its whole run:
#   setsid nohup flock ~/eklow-perf/gpu.lock bash tools/bench/release-bench.sh OUT MODEL_GGUF PROFILE DRAFT ALIAS > OUT.log 2>&1 &
# Steps already done (their file in OUT exists and is non-empty) are skipped, so a stopped run resumes by running it again.
set -euo pipefail
out=${1:?out dir}; model=${2:?}; profile=${3:?}; draft=${4:?}; alias=${5:?}
docs=${DOCS:-$HOME/eklow-qrelease-data}
read -ra extra <<<"${EXTRA:-}"  # further eklow-server arguments: the deployed flags (--group, --mmproj, ...)
port=18181
mkdir -p "$out"
if systemctl is-active --quiet 'eklow*.service'; then echo "release-bench.sh: an eklow service is running" >&2; exit 1; fi
cd "$(dirname "$0")/../.."
nix develop path:. -c ./build/eklow-server --model "$model" --alias "$alias" --port $port --slots 2 --ctx 262144 --kv-int8 \
  --expert-profile "$profile" --draft "$draft" --snapshot-mib 4096 \
  --embed-model ~/models/embeddinggemma-2-Q8_0.gguf --embed-port $((port + 1)) "${extra[@]}" >"$out/serve.log" 2>&1 &
srv=$!
trap "pkill -f 'eklow-server --model.*--port 18181' 2>/dev/null; kill $srv 2>/dev/null; wait $srv 2>/dev/null" EXIT
for i in $(seq 900); do curl -sf localhost:$port/health >/dev/null && break; sleep 1; done
grep "eklow-server:" "$out/serve.log" >"$out/startup.txt" || true
run() { nix develop path:. -c python3 tools/bench/serve_perf.py --url http://127.0.0.1:$port --model "$alias" "$@"; }
step() { local f=$1; shift; if [ -s "$out/$f" ]; then echo "skip $f"; else echo "run $f"; run "$@" | tee "$out/$f.part"; mv "$out/$f.part" "$out/$f"; fi; }
step gsm-draft.txt gsm --requests "$docs/gsm48.jsonl" --concurrency 1,2
step gsm-plain.txt gsm --requests "$docs/gsm48.jsonl" --concurrency 1,2 --plain
for d in 8192 32768 131072 252000; do step depth-$d.txt depth --docs "$docs/docs" --depths $d --repeats 5 --plain 3; done
echo done
