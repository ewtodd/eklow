"""Logits of two runs over the same prompt with images (eklow-oracle --mmproj, or eklow-engine-run --images-dir), on the
text positions: top-1 agreement and KL(ref || test), as mean, median and worst, for the positions after the first image
(where the images' influence is) and for the text before it. The positions of image rows are left out (the model's
prediction there is not a text prediction; the tokens.json of the reference has id -1 for them).

Usage: python3 tools/ref/image_logits_compare.py REF_DIR TEST_DIR [--label NAME]

Both directories hold index.json, logits.f32 and tokens.json (REF's names the images); both must hold the logits of every
position (oracle without --logits-last, engine-run with --logits-last N for N = the prompt's length).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def logp(l):
    l = l - l.max(axis=1, keepdims=True)
    return l - np.log(np.exp(l).sum(axis=1, keepdims=True))


def load(d):
    idx = json.load(open(Path(d) / "index.json"))
    n, V, r = idx["n_tokens"], idx["n_vocab"], idx.get("logits_rows", idx["n_tokens"])
    m = np.fromfile(Path(d) / "logits.f32", dtype=np.float32).reshape(r, V)
    return n, r, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    n, rr, lr = load(a.ref)
    n2, tr, lt = load(a.test)
    if n != n2 or rr != n or tr != n:
        raise SystemExit(f"need the logits of all {n} positions in both (ref {rr}, test {tr} of {n2})")
    ids = json.load(open(Path(a.ref) / "tokens.json"))["ids"]
    first = ids.index(-1) if -1 in ids else n
    text = np.array([i >= 0 for i in ids])
    pr, pt = logp(lr.astype(np.float64)), logp(lt.astype(np.float64))
    kl = (np.exp(pr) * (pr - pt)).sum(axis=1)
    agree = lr.argmax(1) == lt.argmax(1)
    pos = np.arange(n)
    for name, mask in (("after the first image", text & (pos >= first)), ("before it", text & (pos < first))):
        k = int(mask.sum())
        if not k:
            continue
        print(f"{a.label:24s} {name:22s} {k:5d} positions: top-1 {int(agree[mask].sum())}/{k}  KL mean {kl[mask].mean():.3e}  "
              f"median {np.median(kl[mask]):.3e}  max {kl[mask].max():.3e}")


main()
