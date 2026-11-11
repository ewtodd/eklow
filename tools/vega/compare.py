"""eklow's vega checker against the reference (tools/vega/ref_vega.py).

Usage:
  compare.py engine REF_DIR EKLOW_ENGINE.json
      The engine on identical vectors: the largest |p_eklow - p_ref| over every item and option.
  compare.py features REF_DIR EKLOW_DIR [REF_DIR2 ...]
      eklow's pooled vectors (eklow-vega features) against each reference run's: per-vector cosine (each of the 6 item
      vectors, each option, each valid memory chunk), the worst and the 1st percentile, per kind; and the engine's
      probabilities on eklow's own vectors against each reference's.
  compare.py predict REF.json EKLOW.json [CUTOFFS]
      Two System One answers ({key: {qid: noul or {noul}}} or the private replay set's [{key, probs}]): the largest |delta| per
      question, and every flag that differs at the cut-offs (default son-of-anton's: claim 0.7325, unchecked 0.5916,
      offtopic 0.4387) with its margin.
"""

import json
import os
import sys

import numpy as np

CUTOFFS = {"claim": 0.7325, "unchecked": 0.5916, "offtopic": 0.4387}


def cos(a, b):
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def engine(ref, ek):
    idx = json.load(open(os.path.join(ref, "index.json")))["items"]
    got = json.load(open(ek))
    worst, at = 0.0, None
    for it, g in zip(idx, got):
        d = max(abs(a - b) for a, b in zip(it["P"], g["P"]))
        if d > worst:
            worst, at = d, (it["key"], it["qid"])
    print("engine on identical vectors: %d items, max |dp| %.3g at %s" % (len(idx), worst, at))


def features(ek, refs):
    eidx = json.load(open(os.path.join(ek, "index.json")))["items"]
    print("prompts equal to the reference's: %d of %d" % (sum(it["ids_equal"] for it in eidx), len(eidx)))
    for ref in refs:
        ridx = json.load(open(os.path.join(ref, "index.json")))
        cs = {"state": [], "question": [], "last": [], "option": [], "chunk": []}
        dp, at = 0.0, None
        for i, (r, e) in enumerate(zip(ridx["items"], eidx)):
            z = np.load(os.path.join(ref, "%04d.npz" % i))
            f = np.load(os.path.join(ek, "%04d-f.npy" % i))
            o = np.load(os.path.join(ek, "%04d-o.npy" % i))
            c = np.load(os.path.join(ek, "%04d-c.npy" % i))
            for l in range(f.shape[0]):
                for s, name in enumerate(("state", "question", "last")):
                    cs[name].append((cos(f[l, s], z["f"][l, s]), r["key"], r["qid"], l))
            for k in range(o.shape[0]):
                cs["option"].append((cos(o[k], z["o"][k]), r["key"], r["qid"], k))
            n = int(round(z["cm"].sum()))
            if n != e["n_chunks"]:
                print("  %s %s: %d chunks against the reference's %d" % (r["key"], r["qid"], e["n_chunks"], n))
            for k in range(min(n, e["n_chunks"])):
                cs["chunk"].append((cos(c[k], z["c"][k]), r["key"], r["qid"], k))
            d = max(abs(a - b) for a, b in zip(r["P"], e["P"]))
            if d > dp:
                dp, at = d, (r["key"], r["qid"])
        print("against %s (%s):" % (ref, ridx.get("dtype")))
        for name, v in cs.items():
            if not v:
                continue
            vals = np.array([x[0] for x in v])
            w = min(v)
            print("  %-8s n=%4d  min cos %.7f (%s %s %s)  p1 %.7f  median %.7f  (1 - min %.1e)" % (
                name, len(v), w[0], w[1], w[2], w[3], np.percentile(vals, 1), np.median(vals), 1 - w[0]))
        print("  engine probabilities on eklow's vectors: max |dp| %.4g at %s" % (dp, at))


def load_answers(path):
    d = json.load(open(path))
    if isinstance(d, list):
        return {r["key"]: r["probs"] for r in d}
    return {k: {q: (v["noul"] if isinstance(v, dict) else v) for q, v in a.items() if q in CUTOFFS} for k, a in d.items()}


def predict(ref, ek, cut):
    a, b = load_answers(ref), load_answers(ek)
    worst = {q: (0.0, None) for q in cut}
    flips = []
    for k in a:
        if k not in b:
            continue
        for q in cut:
            d = abs(a[k][q] - b[k][q])
            if d > worst[q][0]:
                worst[q] = (d, k)
            if (a[k][q] >= cut[q]) != (b[k][q] >= cut[q]):
                flips.append((k, q, a[k][q], b[k][q], min(abs(a[k][q] - cut[q]), abs(b[k][q] - cut[q]))))
    print("%d states" % len([k for k in a if k in b]))
    for q in cut:
        print("  %-9s max |delta| %.6f (%s)" % (q, worst[q][0], worst[q][1]))
    print("  flag flips at %s: %d" % (cut, len(flips)))
    for f in flips:
        print("    %s %s: reference %.6f, eklow %.6f (nearest to the cut-off by %.6f)" % f)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "engine":
        engine(sys.argv[2], sys.argv[3])
    elif cmd == "features":
        features(sys.argv[3], [sys.argv[2]] + sys.argv[4:])
    elif cmd == "predict":
        predict(sys.argv[2], sys.argv[3], json.loads(sys.argv[4]) if len(sys.argv) > 4 else CUTOFFS)
