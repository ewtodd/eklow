"""Check eklow-dequant-test's dumps against the gguf Python package's dequantize.

Usage: python3 tools/ref/dequant_compare.py GGUF DIR

For every tensor in DIR/rows.json: the same rows dequantized by gguf.quants.dequantize (ggml's
dequantize_row_*, in numpy) against the reference kernels' fp32 values; prints the max absolute
difference and the largest weight so it reads as a relative figure. Both are fp32 from the same
fp16 scales, so the expected difference is 0 up to the order of one multiply-subtract (1e-7 relative).
"""

import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    reader = GGUFReader(sys.argv[1])
    by_name = {t.name: t for t in reader.tensors}
    d = Path(sys.argv[2])
    bad = 0
    for e in json.load(open(d / "rows.json")):
        t = by_name[e["name"]]
        k = e["k"]
        ours = np.fromfile(d / (e["name"] + ".f32"), dtype=np.float32).reshape(len(e["rows"]), k)
        ref = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1, k)[e["rows"]]
        diff = float(np.abs(ours - ref).max())
        scale = float(np.abs(ref).max())
        ok = diff <= 1e-6 * max(scale, 1e-30)
        bad += not ok
        print(f"{e['name']:40s} {e['type']:7s} max |diff| {diff:.3e}  max |w| {scale:.3e}  {'ok' if ok else 'MISMATCH'}")
    sys.exit(1 if bad else 0)


main()
