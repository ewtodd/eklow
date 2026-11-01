"""Where two logit dumps disagree: per-position KL(a || b) over the last N positions, as quantiles and the worst
positions with the tokens there.

Usage: python3 tools/ref/kl_positions.py DIR_A DIR_B [--last N] [--tokens FILE]

A mean KL of 1e-1 over 256 positions can be 250 positions at 1e-3 and three at 10; this tells which.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def rows(d, last):
    idx = json.load(open(Path(d) / "index.json"))
    n, V, r = idx["n_tokens"], idx["n_vocab"], idx.get("logits_rows", idx["n_tokens"])
    m = np.fromfile(Path(d) / "logits.f32", dtype=np.float32).reshape(r, V)
    return n, m[r - min(r, last):].astype(np.float64)


def logp(l):
    l = l - l.max(axis=1, keepdims=True)
    return l - np.log(np.exp(l).sum(axis=1, keepdims=True))


ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--last", type=int, default=256)
a = ap.parse_args()
n, la = rows(a.a, a.last)
_, lb = rows(a.b, a.last)
k = min(len(la), len(lb))
la, lb = la[-k:], lb[-k:]
pa, pb = logp(la), logp(lb)
kl = (np.exp(pa) * (pa - pb)).sum(axis=1)
top1 = la.argmax(1) == lb.argmax(1)
pos = n - k + np.arange(k)
ent = -(np.exp(pa) * pa).sum(axis=1)
toks = json.load(open(Path(a.a) / "tokens.json"))["pieces"]
print(f"{k} positions: mean KL {kl.mean():.3e}, median {np.median(kl):.3e}, p90 {np.quantile(kl, .9):.3e}, "
      f"max {kl.max():.3e}; top-1 {top1.sum()}/{k}; mean KL without the worst 3: {np.sort(kl)[:-3].mean():.3e}")
print("worst positions (pos, KL, entropy of a, token at pos, a's top-1 vs b's top-1 logit gap):")
for i in np.argsort(-kl)[:6]:
    print(f"  {pos[i]:5d} KL {kl[i]:8.3f} H {ent[i]:6.3f} {toks[pos[i]]!r:16} a-top {la[i].argmax()} b-top {lb[i].argmax()}")
