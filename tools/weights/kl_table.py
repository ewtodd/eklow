#!/usr/bin/env python3
"""kl_table.py LOGDIR MODEL: markdown table of the llama-perplexity --kl-divergence logs (kl.sh) of one model."""
import re, sys, glob, os

d, model = sys.argv[1], sys.argv[2]
rows = {}
for p in sorted(glob.glob(f"{d}/{model}-*-*.log")):
    m = re.match(rf"{re.escape(model)}-(wiki4k|doc32k)-(.*)\.log", os.path.basename(p))
    text, name = m.groups()
    if name.endswith("-base"):
        continue
    s = open(p, errors="replace").read()
    g = lambda pat: (re.search(pat, s) or [None, None])[1]
    if g(r"Mean    KLD:\s+(\S+)") is None:
        continue
    rows.setdefault(name, {})[text] = dict(
        ppl=g(r"Mean PPL\(Q\)\s+:\s+(\S+)"), base=g(r"Mean PPL\(base\)\s+:\s+(\S+)"),
        kld=g(r"Mean    KLD:\s+(\S+)"), kld99=g(r"99.0%   KLD:\s+(\S+)"), top=g(r"Same top p:\s+(\S+)"))
print("| file | text | PPL | base PPL | mean KL | 99% KL | same top p % |")
print("|---|---|---|---|---|---|---|")
for name, t in rows.items():
    for text in ("wiki4k", "doc32k"):
        if text in t:
            r = t[text]
            print(f"| {name} | {text} | {float(r['ppl']):.4f} | {float(r['base']):.4f} | {float(r['kld']):.5f} | "
                  f"{float(r['kld99']):.4f} | {r['top']} |")
