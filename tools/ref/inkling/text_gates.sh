#!/bin/sh
# Inkling 2 Mini's text layer gates (DECISIONS 2026-11-24, "the text layer"): the tokenizer, the chat template and the
# output parser against llama.cpp c74c71a6 (M0's ground truth and fresh oracle runs) and Hugging Face. Prints each gate's
# counts; exits 1 if any differs.
#
#   text_gates.sh [BUILD_DIR GGUF DATA_DIR]
#
# BUILD_DIR holds eklow-tok, eklow-chat-inkling and eklow-parse-inkling (ninja -C build eklow-tok eklow-chat-inkling
# eklow-parse-inkling); DATA_DIR is ~/eklow-qwen-data/inkling-port (m0/, hf/, chat_template.jinja, py/, oracle-inkling/bin).
# The chat gate's random requests are 10,000 and each of the parser's 14 requests gets 1,000 random outputs; the code point
# check of the pre-tokenizer (1.1 million code points, two minutes) runs with CLASSES=1.
set -u
BUILD=${1:-build}
GGUF=${2:-$HOME/eklow-qwen-data/fiona-eval/gguf/Inkling-2-Mini-UD-Q4_K_M.gguf}
DATA=${3:-$HOME/eklow-qwen-data/inkling-port}
HERE=$(dirname "$0")
PY="$DATA/py/bin/python3 -I"
ORACLE="$DATA/oracle-inkling/bin/eklow-chat-oracle"
WORK="$DATA/text-worker"
mkdir -p "$WORK"
rc=0
$PY "$HERE/tok_compare.py" "$BUILD/eklow-tok" "$GGUF" "$DATA/hf/tokenizer.json" "$DATA/m0" --repo . || rc=1
if [ "${CLASSES:-0}" = 1 ]; then
  $PY "$HERE/classes_check.py" "$BUILD/eklow-tok" "$GGUF" "$DATA/hf/tokenizer.json" || rc=1
fi
$PY "$HERE/chat_compare.py" "$BUILD/eklow-chat-inkling" "$GGUF" "$DATA/m0" "$DATA/chat_template.jinja" \
  --oracle "$ORACLE" --fuzz 10000 --seed 7 --work "$WORK" || rc=1
$PY "$HERE/parse_compare.py" "$BUILD/eklow-chat-inkling" "$BUILD/eklow-parse-inkling" "$GGUF" "$DATA/m0" \
  --oracle "$ORACLE" --fuzz 1000 --seed 11 --work "$WORK" || rc=1
exit $rc
