#!/usr/bin/env python3
# Turns routing.bin recordings (`eklow-ref-run --routing`; see routing_stats.py for the format) into
# the engine's expert profile: a JSON of per-layer selection counts, 30 layers x 128 experts, summed
# over the recordings. The engine keeps each layer's N most-selected experts on the card
# (`--expert-profile FILE --experts-on-card N`) and computes the rest on the CPU.
#
#   expert_profile.py OUT.json FILE [FILE ...]
import json
import sys
from pathlib import Path

import numpy as np

N_EXPERT = 128


def load(path):
    p = Path(path)
    meta = json.loads((p.parent / (p.name + ".json")).read_text())
    a = np.fromfile(p, dtype="<i4")
    L, T, U = meta["layers"], meta["tokens"], meta["used"]
    if a.size != L * T * U:
        raise SystemExit(f"{p}: expected {L * T * U} ids, got {a.size}")
    return a.reshape(L, T, U)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: expert_profile.py OUT.json FILE [FILE ...]")
    out, files = sys.argv[1], sys.argv[2:]
    arrs = [load(f) for f in files]
    if len({a.shape[0] for a in arrs}) != 1:
        raise SystemExit("layers disagree between files")
    n_layer = arrs[0].shape[0]
    counts = [
        np.bincount(np.concatenate([a[l].ravel() for a in arrs]), minlength=N_EXPERT).tolist() for l in range(n_layer)
    ]
    doc = {
        "model": "gemma-4-26b-a4b",
        "layers": n_layer,
        "experts": N_EXPERT,
        "sources": [Path(f).parent.name for f in files],
        "counts": counts,
    }
    Path(out).write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{out}: {n_layer} layers x {N_EXPERT} experts, {sum(map(sum, counts))} selections from {len(files)} recordings")


if __name__ == "__main__":
    main()
