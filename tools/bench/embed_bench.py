#!/usr/bin/env python3
"""Embedding throughput and agreement through an OpenAI-compatible /v1/embeddings endpoint.

  embed_bench.py run --url http://127.0.0.1:18182 --text FILE --out vecs.json [--chunk-chars 1800] [--batch 16] [--n 128]
  embed_bench.py compare a.json b.json

run cuts FILE into chunks of about --chunk-chars characters (Open WebUI's document chunks), prefixes them as
EmbeddingGemma 2's documents ("title: none | text: "), sends them --batch at a time, and prints tokens a second over
the wall time; compare prints the cosine of each pair of vectors (both files from the same inputs).
"""
import argparse
import json
import math
import sys
import time
import urllib.request


def post(url, body):
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def run(a):
    text = open(a.text, encoding="utf-8", errors="replace").read()
    chunks = [text[i:i + a.chunk_chars] for i in range(0, len(text), a.chunk_chars)][: a.n]
    inputs = ["title: none | text: " + c for c in chunks]
    vecs, tokens = [], 0
    t0 = time.monotonic()
    for i in range(0, len(inputs), a.batch):
        r = post(a.url, {"model": "embeddinggemma-2", "input": inputs[i:i + a.batch]})
        vecs += [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
        tokens += r["usage"]["prompt_tokens"]
    wall = time.monotonic() - t0
    json.dump(vecs, open(a.out, "w"))
    print(f"{len(inputs)} inputs, {tokens} tokens in {wall:.2f} s: {tokens / wall:.0f} tok/s ({a.batch} a request)")


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    cos = []
    for u, v in zip(x, y):
        dot = sum(p * q for p, q in zip(u, v))
        cos.append(dot / math.sqrt(sum(p * p for p in u) * sum(q * q for q in v)))
    cos.sort()
    print(f"{len(cos)} pairs: cosine min {cos[0]:.6f}, median {cos[len(cos) // 2]:.6f}, max {cos[-1]:.6f}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("run")
p.add_argument("--url", required=True)
p.add_argument("--text", required=True)
p.add_argument("--out", required=True)
p.add_argument("--chunk-chars", type=int, default=1800)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--n", type=int, default=128)
p = sub.add_parser("compare")
p.add_argument("a")
p.add_argument("b")
a = ap.parse_args()
run(a) if a.cmd == "run" else compare(a)
