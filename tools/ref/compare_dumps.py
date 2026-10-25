"""Compare two tensor dumps (eklow-oracle's format: index.json + NAME.f32, logits.f32).

Usage: python3 tools/ref/compare_dumps.py REF_DIR TEST_DIR [--prefix P]

For every tensor present in both (in REF's order): relative L2 error ||t - r|| / ||r||, max
absolute difference and cosine similarity. Then, over the logits of every position: top-1
agreement and the mean KL divergence KL(ref || test).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(d):
    idx = json.load(open(Path(d) / "index.json"))
    return idx, {t["name"]: t for t in idx["tensors"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--prefix", default="")
    a = ap.parse_args()
    ri, rt = load(a.ref)
    ti, tt = load(a.test)
    print(f"{'tensor':32s} {'rel L2':>10s} {'max abs':>10s} {'cosine':>10s}")
    for t in ri["tensors"]:
        name = t["name"]
        if not name.startswith(a.prefix) or name not in tt:
            continue
        if tt[name]["ne"] != t["ne"]:
            print(f"{name:32s} shape {t['ne']} vs {tt[name]['ne']}")
            continue
        r = np.fromfile(Path(a.ref) / t["file"], dtype=np.float32).astype(np.float64)
        x = np.fromfile(Path(a.test) / tt[name]["file"], dtype=np.float32).astype(np.float64)
        rel = np.linalg.norm(x - r) / max(np.linalg.norm(r), 1e-30)
        cos = float(r @ x / max(np.linalg.norm(r) * np.linalg.norm(x), 1e-30))
        print(f"{name:32s} {rel:10.3e} {np.abs(x - r).max():10.3e} {cos:10.6f}")

    n, V = ri["n_tokens"], ri["n_vocab"]
    if ti["n_tokens"] == n and ti["n_vocab"] == V:
        lr = np.fromfile(Path(a.ref) / "logits.f32", dtype=np.float32).reshape(n, V).astype(np.float64)
        lt = np.fromfile(Path(a.test) / "logits.f32", dtype=np.float32).reshape(n, V).astype(np.float64)

        def logp(l):
            l = l - l.max(axis=1, keepdims=True)
            return l - np.log(np.exp(l).sum(axis=1, keepdims=True))

        pr, pt = logp(lr), logp(lt)
        kl = (np.exp(pr) * (pr - pt)).sum(axis=1)
        top1 = (lr.argmax(1) == lt.argmax(1)).mean()
        print(f"logits: top-1 agreement {top1:.3f}, mean KL {kl.mean():.3e}, max KL {kl.max():.3e} over {n} positions")


if __name__ == "__main__":
    main()
