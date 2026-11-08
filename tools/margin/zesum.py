# zesum.py BINARY ZE.TXT [MIN_MIB]: the device and shared allocations in a zetrace log (tools/margin/zetrace.c), by stage (the log's "mark N" lines,
# written on SIGUSR1), grouped by size and call site (the first executable frames, symbolized with addr2line; the executable is not PIE,
# loaded at 0x400000). Allocations of the stage's start (mark 0) are the startup; the others are what the margin pays for.
import collections, subprocess, sys
binary, path = sys.argv[1], sys.argv[2]
floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
stages = collections.defaultdict(list)  # mark -> [(kind, size, frames)]
cur = 0
for line in open(path):
    f = line.split()
    if not f: continue
    if f[0] == "mark": cur = int(f[1]); continue
    if f[0] in ("dev", "shared"): stages[cur].append((f[0], int(f[2]), tuple(f[4:])))
addrs = sorted({a for ev in stages.values() for _, _, fr in ev for a in fr})
sym = {}
if addrs:
    out = subprocess.run(["addr2line", "-f", "-C", "-e", binary] + ["0x%x" % (int(a, 16) + 0x400000) for a in addrs], capture_output=True, text=True).stdout.split("\n")
    for i, a in enumerate(addrs): sym[a] = out[2 * i] if 2 * i < len(out) else "?"
def site(fr):
    names = [sym.get(a, "?") for a in fr if sym.get(a, "?") != "??"]
    return " < ".join(n[:60] for n in names[:3]) or "(driver)"
for m in sorted(stages):
    ev = stages[m]
    tot = sum(s for _, s, _ in ev)
    print(f"== mark {m}: {len(ev)} allocations, {tot / 1048576:.1f} MiB")
    g = collections.Counter(); n = collections.Counter()
    for k, s, fr in ev: g[(k, s, site(fr))] += s; n[(k, s, site(fr))] += 1
    for (k, s, st), tot_s in sorted(g.items(), key=lambda x: -x[1]):
        if tot_s / 1048576 >= floor: print(f"  {n[(k, s, st)]:>5} x {s / 1048576:9.3f} MiB = {tot_s / 1048576:9.2f}  {k}  {st}")
