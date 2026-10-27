#!/usr/bin/env python3
# How skewed is Gemma 4 26B A4B's expert routing, layer by layer?
#
# Reads routing.bin files written by `eklow-ref-run --routing` (raw int32 little-endian expert ids,
# layer-major: layer, token, `used` ids per token, with a sidecar .json {"layers", "tokens",
# "used"}). For each layer, ranks that layer's experts by how often they were selected and prints
# the share of the layer's selections the busiest N take, for N = 16, 32, 64, 96, 112 of 128: what
# keeping those N experts of every layer on the card would cover. Expert ids are per layer (expert 5
# of layer 0 has nothing to do with expert 5 of layer 20), so the summary is the mean over layers,
# not a pool. With two or more files, each file is also measured against a ranking from the others
# only: how well a profile of some text predicts other text.
#
#   routing_stats.py FILE [FILE ...]
import json
import sys
from pathlib import Path

import numpy as np

NS = [16, 32, 64, 96, 112]
N_EXPERT = 128


def load(path):
    p = Path(path)
    meta = json.loads((p.parent / (p.name + ".json")).read_text())
    a = np.fromfile(p, dtype="<i4")
    L, T, U = meta["layers"], meta["tokens"], meta["used"]
    if a.size != L * T * U:
        raise SystemExit(f"{p}: expected {L * T * U} ids, got {a.size}")
    return a.reshape(L, T, U)


def counts(arrs):
    """[layers, N_EXPERT] selection counts summed over the given recordings."""
    return np.stack([np.bincount(np.concatenate([a[l].ravel() for a in arrs]), minlength=N_EXPERT) for l in range(arrs[0].shape[0])])


def cover(rank, meas):
    """Per layer, the % of `meas`'s selections taken by the busiest N experts of `rank`."""
    out = []
    for l in range(rank.shape[0]):
        order = np.argsort(rank[l])[::-1]
        out.append([meas[l][order[:n]].sum() / meas[l].sum() for n in NS])
    return 100.0 * np.array(out)


def row(name, values):
    print(f"{name:>14}" + "".join(f"  {v:6.1f}" for v in values))


def main():
    files = sys.argv[1:]
    if not files:
        raise SystemExit("usage: routing_stats.py FILE [FILE ...]")
    arrs = [load(f) for f in files]
    if len({a.shape[0] for a in arrs}) != 1:
        raise SystemExit("layers disagree between files")
    c = counts(arrs)
    cv = cover(c, c)
    print("all inputs, ranked and measured on themselves (% of each layer's selections)")
    print(" " * 14 + "".join(f"  N={n:<4}" for n in NS))
    for l in range(cv.shape[0]):
        row(f"layer {l}", cv[l])
    row("mean", cv.mean(0))
    row("min", cv.min(0))
    if len(arrs) >= 2:
        print("held out: ranked on the other files, measured on each (mean over layers)")
        for i, f in enumerate(files):
            others = [a for j, a in enumerate(arrs) if j != i]
            row(Path(f).parent.name, cover(counts(others), counts([arrs[i]])).mean(0))


if __name__ == "__main__":
    main()
