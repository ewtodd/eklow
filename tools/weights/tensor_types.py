#!/usr/bin/env python3
"""tensor_types.py FILE [--only T1,T2]: tensor counts by type, bytes of the routed experts and of
everything else; with --only, exit 1 if any other type is present."""
import sys, collections
from gguf import GGUFReader

path = sys.argv[1]
only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[2] == "--only" else None
r = GGUFReader(path)
by = collections.defaultdict(collections.Counter)
exp = oth = 0
for t in r.tensors:
    kind = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
    by[t.tensor_type.name][kind] += 1
    if "_exps" in t.name: exp += int(t.n_bytes)
    else: oth += int(t.n_bytes)
print(f"{path}: {len(r.tensors)} tensors, {(exp+oth)/2**30:.2f} GiB (routed experts {exp/2**30:.2f} GiB, everything else {oth/2**30:.2f} GiB)")
for ty, c in sorted(by.items()):
    print(f"  {ty}: {sum(c.values())}  " + ", ".join(f"{k} x{n}" for k, n in sorted(c.items())))
if only:
    bad = set(by) - only
    if bad:
        print(f"  FAIL: types outside {sorted(only)}: {sorted(bad)}"); sys.exit(1)
