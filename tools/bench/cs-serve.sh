#!/usr/bin/env bash
# eklow-server for a check-source session (the production shape, draft on), run on fiona under the GPU lock:
#   setsid nohup flock ~/eklow-perf/gpu.lock bash tools/bench/cs-serve.sh LOG MODEL_GGUF PROFILE DRAFT ALIAS > /dev/null 2>&1 &
# It listens on 127.0.0.1:18181 and exits (freeing the lock) when LOG has not been written for 30 minutes: the server logs
# every 10 s while it has traffic, so a client that died leaves no idle server on the card.
set -u
log=${1:?}; model=${2:?}; profile=${3:?}; draft=${4:?}; alias=${5:?}
if systemctl is-active --quiet eklow; then echo "cs-serve.sh: eklow.service is running" >&2; exit 1; fi
cd "$(dirname "$0")/../.."
nix develop path:. -c ./build/eklow-server --model "$model" --alias "$alias" --port 18181 --slots 2 --ctx 262144 --kv-int8 \
  --expert-profile "$profile" --draft "$draft" --snapshot-mib 4096 \
  --embed-model ~/models/embeddinggemma-2-Q8_0.gguf --embed-port 18182 >"$log" 2>&1 &
srv=$!
trap 'pkill -P $srv 2>/dev/null; kill $srv 2>/dev/null; wait $srv 2>/dev/null' EXIT
while kill -0 $srv 2>/dev/null; do
  sleep 30
  if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt 1800 ]; then echo "idle for 30 min: stopping" >> "$log"; break; fi
  if systemctl is-active --quiet eklow; then echo "eklow.service started: stopping" >> "$log"; break; fi
done
