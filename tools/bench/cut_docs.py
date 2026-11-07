#!/usr/bin/env python3
"""Cuts unholy's Pile documents (~/eklow-bench/data/doc-N.txt, unique text per depth) to a number of Qwen3.6 tokens, for
serve_perf.py: writes OUT/qwen-DEPTH.txt holding the first DEPTH - SLACK tokens of the smallest doc-N.txt big enough, so that
the chat request built around it (template, question) is DEPTH tokens.

  cut_docs.py --eklow-tok build/eklow-tok --model GGUF --data ~/eklow-bench/data --out DIR [--depths 8192,32768,131072,252000]
Stdlib only; counts tokens with eklow-tok (the tokenizer under test, 26M tokens/s).
"""
import argparse
import glob
import os
import re
import subprocess


def count(tok, model, text):
    p = subprocess.run([tok, model, "encode"], input=text.encode(), capture_output=True, check=True)
    return len(p.stdout.split())


ap = argparse.ArgumentParser()
ap.add_argument("--eklow-tok", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--depths", default="8192,32768,131072,252000")
ap.add_argument("--slack", type=int, default=80)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
docs = sorted(glob.glob(f"{a.data}/doc-*.txt"), key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0]))
for depth in [int(x) for x in a.depths.split(",")]:
    want = depth - a.slack
    for path in docs:
        text = open(path, encoding="utf-8", errors="ignore").read()
        if count(a.eklow_tok, a.model, text) >= want:
            break
    else:
        raise SystemExit(f"no document of {want} tokens")
    lo, hi = 0, len(text)  # the longest prefix with at most `want` tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(a.eklow_tok, a.model, text[:mid]) <= want:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    open(f"{a.out}/qwen-{depth}.txt", "w").write(cut)
    print(f"{depth}: {os.path.basename(path)} cut to {len(cut)} bytes, {count(a.eklow_tok, a.model, cut)} tokens")
