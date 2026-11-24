#!/usr/bin/env python3
"""Inkling 2 Mini: an eklow dump (eklow-ref-run, later the engine) against an M0 oracle dump, slot by slot in the order
the forward runs them (0-5, 6-17, 24-35, 36-47, 18-23).

  compare.py REF_DIR TEST_DIR [--tensors l_out,ffn_inp,...] [--all]

For each tensor kind and slot: relative L2 error |t - r| / |r| and cosine. Router picks (ffn_moe_topk, or the first 8
of an oracle's ffn_moe_argsort): the share of (position, slot) whose 8 experts are the same set. Logits: top-1 agreement and KL(ref || test) over the positions both
hold, mean and max. --all prints every slot; otherwise slots whose error is below 1.5x the previous printed one are
folded.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ORDER = list(range(0, 6)) + list(range(6, 18)) + list(range(24, 36)) + list(range(36, 48)) + list(range(18, 24))


def index(d):
    idx = json.load(open(Path(d) / "index.json"))
    return idx, {t["name"]: t for t in idx["tensors"]}


def load(d, t):
    dt = np.int32 if t["file"].endswith(".i32") else np.float32
    return np.fromfile(Path(d) / t["file"], dtype=dt)


def picks(d, idx, s):
    """The 8 experts of each position at slot s: eklow dumps ffn_moe_topk, the oracle the full ffn_moe_argsort."""
    if f"ffn_moe_topk-{s}" in idx:
        return load(d, idx[f"ffn_moe_topk-{s}"]).astype(np.int64).reshape(-1, 8)
    if f"ffn_moe_argsort-{s}" in idx:
        t = idx[f"ffn_moe_argsort-{s}"]
        return load(d, t).reshape(-1, t["ne"][0])[:, :8]
    return None


def logp(l):
    l = l - l.max(axis=1, keepdims=True)
    return l - np.log(np.exp(l).sum(axis=1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--tensors", default="Qcur,Kcur,Vcur,Rcur,kqv_out,attn_out,ffn_inp,ffn_moe_logits,ffn_moe_out,ffn_out,l_out")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    ri, rt = index(a.ref)
    ti, tt = index(a.test)
    kinds = a.tensors.split(",")
    print(f"{'slot':>4s} " + " ".join(f"{k[:10]:>10s}" for k in kinds) + "  router")
    for s in ORDER:
        row, any_ = [], False
        for k in kinds:
            n = f"{k}-{s}"
            if n in rt and n in tt and rt[n]["ne"] == tt[n]["ne"]:
                r = load(a.ref, rt[n]).astype(np.float64)
                x = load(a.test, tt[n]).astype(np.float64)
                row.append(f"{np.linalg.norm(x - r) / (np.linalg.norm(r) + 1e-30):10.2e}")
                any_ = True
            else:
                row.append(f"{'-':>10s}")
        router = ""
        r, x = picks(a.ref, rt, s), picks(a.test, tt, s)
        if r is not None and x is not None and len(r) == len(x):
            same = np.mean([set(u) == set(v) for u, v in zip(r.tolist(), x.tolist())])
            router = f"{same:7.3f}"
        if any_:
            print(f"{s:4d} " + " ".join(row) + "  " + router)
    for n in ("result_norm",):
        if n in rt and n in tt:
            r = load(a.ref, rt[n]).astype(np.float64)
            x = load(a.test, tt[n]).astype(np.float64)
            print(f"{n}: rel L2 {np.linalg.norm(x - r) / np.linalg.norm(r):.2e}")
    # logits: the ref's rows cover positions [n - rows, n); the test's likewise
    V = ri["n_vocab"]
    rr, tr = ri.get("logits_rows", ri["n_tokens"]), ti.get("logits_rows", ti["n_tokens"])
    rl = np.fromfile(Path(a.ref) / "logits.f32", dtype=np.float32).reshape(rr, V)
    tl = np.fromfile(Path(a.test) / "logits.f32", dtype=np.float32).reshape(tr, V)
    k = min(rr, tr)
    rp, tp = logp(rl[rr - k:].astype(np.float64)), logp(tl[tr - k:].astype(np.float64))
    kl = (np.exp(rp) * (rp - tp)).sum(axis=1)
    top = (rp.argmax(1) == tp.argmax(1)).sum()
    print(f"logits over the last {k} positions: top-1 {top}/{k}, KL mean {kl.mean():.2e}, max {kl.max():.2e}, "
          f"median {np.median(kl):.2e}")


if __name__ == "__main__":
    main()
