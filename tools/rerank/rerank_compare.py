"""Rerank answers of two servers (llama.cpp's and eklow's /v1/rerank) on the same requests, and the gate between them.

Usage:
  rerank_compare.py fetch URL PAIRS.json OUT.json [--repeat N]
      POST each request of PAIRS.json (synthetic_pairs.py's) to URL (…/v1/rerank) with top_n = all; OUT.json holds
      {name: {"scores": [score of document i], "ms": [wall time of each repeat]}}.
  rerank_compare.py gate REF.json EKLOW.json
      Per document: |score - reference| within 0.005 where the reference is under 0.1 and within 0.02 above; per request:
      the same order of documents. Prints the worst of each and every failure; exits 1 on any.
"""

import json
import sys
import time
import urllib.request


def fetch(url, pairs, out, repeat):
    res = {}
    for r in json.load(open(pairs)):
        body = json.dumps({"model": "rerank", "query": r["query"], "documents": r["documents"],
                           "top_n": len(r["documents"])}).encode()
        ms, scores = [], None
        for _ in range(repeat):
            t0 = time.perf_counter()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            got = json.load(urllib.request.urlopen(req, timeout=120))
            ms.append(1000 * (time.perf_counter() - t0))
            scores = [None] * len(r["documents"])
            for x in got["results"]:
                scores[x["index"]] = x["relevance_score"]
        res[r["name"]] = {"scores": scores, "ms": ms}
        print(r["name"], ["%.4f" % s for s in scores], "%.0f ms" % min(ms), flush=True)
    json.dump(res, open(out, "w"), indent=1)


def gate(ref, ek):
    a, b = json.load(open(ref)), json.load(open(ek))
    bad, low, high = 0, 0.0, 0.0
    for name, r in a.items():
        sa, sb = r["scores"], b[name]["scores"]
        for i, (x, y) in enumerate(zip(sa, sb)):
            d = abs(x - y)
            lim = 0.005 if x < 0.1 else 0.02
            if x < 0.1:
                low = max(low, d)
            else:
                high = max(high, d)
            if d > lim:
                bad += 1
                print("  %s doc %d: reference %.5f, eklow %.5f (|d| %.5f > %.3f)" % (name, i, x, y, d, lim))
        oa = sorted(range(len(sa)), key=lambda i: -sa[i])
        ob = sorted(range(len(sb)), key=lambda i: -sb[i])
        if oa != ob:
            bad += 1
            # the widest reference gap between two documents the other puts the other way round
            gap = max(abs(sa[i] - sa[j]) for i in range(len(sa)) for j in range(len(sa)) if (sa[i] - sa[j]) * (sb[i] - sb[j]) < 0)
            print("  %s: order differs (pairs swapped at most %.2g apart in the reference): reference %s, eklow %s" % (name, gap, oa, ob))
    print("largest |d|: %.5f under 0.1, %.5f above; %d failures" % (low, high, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        rep = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 1
        fetch(sys.argv[2], sys.argv[3], sys.argv[4], rep)
    else:
        sys.exit(gate(sys.argv[2], sys.argv[3]))
