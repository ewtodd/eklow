#!/usr/bin/env bash
# The README's performance numbers: eklow-server with production flags under the GPU lock, then serve_perf.py's depth
# and GSM8K runs against it; the server's startup lines and both results land in OUT. Run on fiona from a built tree.
#   tools/bench/serve-perf.sh OUT MODEL_GGUF ALIAS [eklow-server flags]
#   tools/bench/serve-perf.sh ~/eklow-bench/26b ~/models/gemma-4-26b-a4b-it-qat-q4_0.gguf gemma-4-26b-a4b \
#     --expert-profile data/gemma-4-26b-a4b-experts.json
# DOCS (default ~/eklow-bench/data) holds gemma-DEPTH.txt and gsm48.jsonl; DEPTHS overrides serve_perf.py's depths.
set -euo pipefail
# A second server beside fiona's production eklow.service starves it into UR_RESULT_ERROR_OUT_OF_RESOURCES on
# every request until it restarts (2026-10-27). To measure production itself, run serve_perf.py against it.
if systemctl is-active --quiet eklow; then
  echo "serve-perf.sh: eklow.service is running; stop it first, or point serve_perf.py at it" >&2
  exit 1
fi
out=${1:?out dir}; model=${2:?model gguf}; alias=${3:?alias}; shift 3
docs=${DOCS:-$HOME/eklow-bench/data}
port=18181
mkdir -p "$out"
flock ~/eklow-perf/gpu.lock bash -c '
  set -euo pipefail
  out=$1; model=$2; alias=$3; docs=$4; port=$5; shift 5
  nix develop path:. -c ./build/eklow-server --model "$model" --alias "$alias" --port $port --slots 2 --ctx 262144 \
    --kv-int8 --embed-model ~/models/embeddinggemma-2-Q8_0.gguf --embed-port $((port + 1)) "$@" >"$out/serve.log" 2>&1 &
  srv=$!
  trap "kill $srv 2>/dev/null; wait $srv 2>/dev/null" EXIT
  for i in $(seq 600); do curl -sf localhost:$port/health >/dev/null && break; sleep 1; done
  run() { nix develop path:. -c python3 tools/bench/serve_perf.py --url http://127.0.0.1:$port --model "$alias" "$@"; }
  run depth --docs "$docs" ${DEPTHS:+--depths $DEPTHS} | tee "$out/depth.txt"
  run gsm --requests "$docs/gsm48.jsonl" | tee "$out/gsm.txt"
' _ "$out" "$model" "$alias" "$docs" "$port" "$@"
grep "eklow-server:" "$out/serve.log" >"$out/startup.txt" || true
