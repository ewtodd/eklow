#!/usr/bin/env bash
# fetch.sh MODEL: download and verify the pinned HF checkpoint (stock|ornith) into $WORK/hf/MODEL.
source "$(dirname "$0")/lib.sh"
model_info "$1"
hub_fetch "$M_REPO" "$M_REV" "$WORK/hf/$1" '^assets/'
