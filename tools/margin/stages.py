# stages.py OUTDIR: the card memory growth per stage, from OUTDIR/vram.txt (poll.py) and OUTDIR/marks.txt ("unix-time name" lines).
# For each stage: the memory when it began, its peak, the growth over the stage, and the growth since "ready".
import sys
d = sys.argv[1]
v = [tuple(map(float, l.split())) for l in open(f"{d}/vram.txt") if l.strip()]
m = [(int(l.split()[0]), l.split(None, 1)[1].strip()) for l in open(f"{d}/marks.txt") if l.strip()]
ready = next((t for t, n in m if n == "ready"), m[0][0])
base = next((x[1] for x in v if x[0] >= ready), v[0][1])
print(f"{'stage':<26}{'start MiB':>10}{'peak MiB':>10}{'grew':>7}{'since ready':>13}")
peak_all = base
for i, (t, name) in enumerate(m):
    t_end = m[i + 1][0] if i + 1 < len(m) else v[-1][0] + 1
    seg = [x[1] for x in v if t <= x[0] < t_end]
    if not seg: continue
    start = next((x[1] for x in v if x[0] >= t), seg[0])
    peak = max(seg); peak_all = max(peak_all, peak)
    print(f"{name:<26}{start:>10.0f}{peak:>10.0f}{peak - start:>7.0f}{peak_all - base:>13.0f}")
print(f"ready {base:.0f} MiB, final peak {peak_all:.0f} MiB, residual growth {peak_all - base:.0f} MiB")
