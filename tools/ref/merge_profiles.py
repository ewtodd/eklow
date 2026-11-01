#!/usr/bin/env python3
# Sums routing recordings written by `eklow-engine-run --record-routing` (per-layer selection counts) into an expert profile for
# `--expert-profile` (the same JSON as expert_profile.py makes from eklow-ref-run's routing.bin).
#
#   merge_profiles.py OUT.json MODEL_NAME IN.json [IN.json ...]
import json
import sys

if len(sys.argv) < 4:
    raise SystemExit("usage: merge_profiles.py OUT.json MODEL_NAME IN.json [IN.json ...]")
out, name, files = sys.argv[1], sys.argv[2], sys.argv[3:]
docs = [json.load(open(f)) for f in files]
counts = [[sum(d["counts"][l][e] for d in docs) for e in range(len(docs[0]["counts"][l]))] for l in range(len(docs[0]["counts"]))]
doc = {
    "model": name,
    "layers": len(counts),
    "experts": len(counts[0]),
    "tokens": sum(d["tokens"] for d in docs),
    "sources": [f.rsplit("/", 1)[-1].removesuffix(".json") for f in files],
    "counts": counts,
}
json.dump(doc, open(out, "w"), separators=(",", ":"))
print(f"{out}: {doc['layers']} layers x {doc['experts']} experts over {doc['tokens']} tokens from {', '.join(doc['sources'])}")
