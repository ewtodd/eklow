"""Two mmproj GGUFs, tensor by tensor: names, types, shapes and the bytes; metadata keys that differ.

Usage: python3 mmproj_compare.py A.gguf B.gguf
For tensors of the same type and shape whose bytes differ it prints the worst element difference (BF16 / F16 / F32).
"""

import hashlib
import sys

import numpy as np
from gguf import GGUFReader


def as_f32(t):
    d = np.asarray(t.data)
    if t.tensor_type.name == "BF16":
        return (d.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return d.astype(np.float64)


def meta(r):
    out = {}
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        try:
            v = f.contents()
        except Exception as e:  # noqa: BLE001
            v = f"<{e}>"
        out[k] = v
    return out


a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
print(f"tensors: {len(ta)} vs {len(tb)}")
for n in sorted(set(ta) ^ set(tb)):
    print("only in", "A" if n in ta else "B", n)
same = diff = 0
for n in sorted(set(ta) & set(tb)):
    x, y = ta[n], tb[n]
    if x.tensor_type != y.tensor_type or list(x.shape) != list(y.shape):
        print(f"DIFF {n}: {x.tensor_type.name}{list(x.shape)} vs {y.tensor_type.name}{list(y.shape)}")
        diff += 1
        continue
    if hashlib.sha256(bytes(x.data)).digest() == hashlib.sha256(bytes(y.data)).digest():
        same += 1
        continue
    diff += 1
    fx, fy = as_f32(x), as_f32(y)
    d = np.abs(fx - fy)
    print(f"DIFF {n}: {x.tensor_type.name} max |a-b| {d.max():.3e} (max |b| {np.abs(fy).max():.3e}), {np.count_nonzero(d)} of {d.size} elements differ")
print(f"identical bytes: {same}, different: {diff}")
ma, mb = meta(a), meta(b)
for k in sorted(set(ma) | set(mb)):
    if str(ma.get(k, "<absent>")) != str(mb.get(k, "<absent>")):
        print(f"meta {k}: A={str(ma.get(k, '<absent>'))[:100]} B={str(mb.get(k, '<absent>'))[:100]}")
