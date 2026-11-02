#!/usr/bin/env python3
"""dflash2_tables.py RESULTDIR: markdown tables from the *.jsonl of dflash2-llamacpp.sh (their summary lines)."""
import glob, json, os, sys

rows = {}
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.jsonl"))):
    for line in open(p):
        s = json.loads(line).get("summary")
        if s:
            rows.setdefault(s["label"], {})[s["dataset"]] = s
ds = ["gsm8k", "humaneval", "mt-bench"]
print("| run (pair-block / sampling) | " + " | ".join(f"{d} acceptance (tok/s, truncated)" for d in ds) + " |")
print("|---|" + "---|" * len(ds))
for label, t in rows.items():
    cells = []
    for d in ds:
        s = t.get(d)
        if not s:
            cells.append("")
        else:
            a = f"{s['acceptance_mean']:.2f}" if s["acceptance_mean"] else "n/a"
            cells.append(f"{a} ({s['decode_tps_median']:.0f}, {s['truncated']}/{s['requests']})")
    print(f"| {label} | " + " | ".join(cells) + " |")
