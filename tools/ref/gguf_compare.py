"""Check eklow-gguf's reading of a GGUF file against the gguf Python package (ggml's own reader).

Usage: python3 tools/ref/gguf_compare.py FILE EKLOW_JSON
       (EKLOW_JSON from `eklow-gguf FILE --json`)

Compares the data offset, every metadata key (type and value; arrays by type, length and the first
and last 8 elements) and every tensor (type, shape, offset, size). Exits 1 on any difference.
"""

import json
import math
import sys

from gguf import GGUFReader, GGUFValueType
from gguf.constants import GGML_QUANT_SIZES

KIND = {
    GGUFValueType.UINT8: "u8", GGUFValueType.INT8: "i8", GGUFValueType.UINT16: "u16",
    GGUFValueType.INT16: "i16", GGUFValueType.UINT32: "u32", GGUFValueType.INT32: "i32",
    GGUFValueType.FLOAT32: "f32", GGUFValueType.BOOL: "bool", GGUFValueType.STRING: "string",
    GGUFValueType.ARRAY: "array", GGUFValueType.UINT64: "u64", GGUFValueType.INT64: "i64",
    GGUFValueType.FLOAT64: "f64",
}


def scalar(field, part_index):
    part = field.parts[part_index]
    t = field.types[-1]
    if t == GGUFValueType.STRING:
        return bytes(part).decode("utf-8", errors="surrogateescape")
    v = part.tolist()[0]
    if t == GGUFValueType.BOOL:
        return bool(v)
    if t in (GGUFValueType.FLOAT32, GGUFValueType.FLOAT64):
        return float(v)
    return int(v)


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=0.0) or (math.isnan(a) and math.isnan(b))
    return a == b


def main():
    path, ours_path = sys.argv[1], sys.argv[2]
    ours = json.load(open(ours_path, encoding="utf-8"))
    ref = GGUFReader(path)
    errors = []

    if ours["data_offset"] != ref.data_offset:
        errors.append(f"data offset {ours['data_offset']} vs {ref.data_offset}")

    ref_keys = {k for k in ref.fields if not k.startswith("GGUF.")}
    if set(ours["metadata"]) != ref_keys:
        errors.append(f"metadata keys differ: only ours {sorted(set(ours['metadata']) - ref_keys)[:5]}, "
                      f"only theirs {sorted(ref_keys - set(ours['metadata']))[:5]}")
    for key in sorted(ref_keys & set(ours["metadata"])):
        field, mine = ref.fields[key], ours["metadata"][key]
        kind = KIND[field.types[0]]
        if mine["type"] != kind:
            errors.append(f"{key}: type {mine['type']} vs {kind}")
            continue
        if kind == "array":
            vals = [scalar(field, i) for i in field.data]
            got = mine["value"]
            if got["type"] != KIND[field.types[-1]] or got["length"] != len(vals):
                errors.append(f"{key}: array {got['type']} x {got['length']} vs {KIND[field.types[-1]]} x {len(vals)}")
                continue
            head, tail = vals[:8], vals[-8:] if len(vals) > 8 else vals
            if not all(same(a, b) for a, b in zip(got["head"], head)) or len(got["head"]) != len(head):
                errors.append(f"{key}: array head differs")
            if not all(same(a, b) for a, b in zip(got["tail"], tail)) or len(got["tail"]) != len(tail):
                errors.append(f"{key}: array tail differs")
        else:
            want = scalar(field, field.data[0])
            if not same(mine["value"], want):
                errors.append(f"{key}: {mine['value']!r} vs {want!r}")

    theirs = {t.name: t for t in ref.tensors}
    if len(ours["tensors"]) != len(ref.tensors):
        errors.append(f"tensor count {len(ours['tensors'])} vs {len(ref.tensors)}")
    for t in ours["tensors"]:
        r = theirs.get(t["name"])
        if r is None:
            errors.append(f"tensor {t['name']} missing from the reference")
            continue
        block, size = GGML_QUANT_SIZES[r.tensor_type]
        want = {"type": r.tensor_type.name, "ne": [int(x) for x in r.shape],
                "offset": int(r.data_offset) - ref.data_offset, "nbytes": int(r.n_bytes)}
        got = {k: t[k] for k in want}
        if got != want:
            errors.append(f"tensor {t['name']}: {got} vs {want}")
        if t["nbytes"] != math.prod(t["ne"]) // block * size:
            errors.append(f"tensor {t['name']}: size does not follow from shape")

    for e in errors[:40]:
        print("DIFF", e)
    print(f"{len(ours['metadata'])} metadata keys, {len(ours['tensors'])} tensors: "
          f"{'identical' if not errors else f'{len(errors)} differences'}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
