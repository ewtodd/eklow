# poll.py PID OUT: every second, the process's card memory from DRM fdinfo (xe: drm-total-vram0 / drm-resident-vram0, KiB), summed over its
# DRM fds, one line per sample: unix time, total MiB, resident MiB. Stops when the process exits.
import os, sys, time
pid, out = sys.argv[1], sys.argv[2]
def sample():
    tot = res = 0; seen = set()
    for fd in os.listdir(f"/proc/{pid}/fdinfo"):
        try: txt = open(f"/proc/{pid}/fdinfo/{fd}").read()
        except OSError: continue
        kv = dict(l.split(":", 1) for l in txt.splitlines() if ":" in l)
        cid = kv.get("drm-client-id", "").strip()
        if not cid or cid in seen or "drm-total-vram0" not in kv: continue
        seen.add(cid)
        tot += int(kv["drm-total-vram0"].split()[0]); res += int(kv.get("drm-resident-vram0", "0 KiB").split()[0])
    return tot / 1024, res / 1024
with open(out, "a") as fh:
    while os.path.exists(f"/proc/{pid}"):
        try: t, r = sample()
        except FileNotFoundError: break
        fh.write(f"{time.time():.0f} {t:.0f} {r:.0f}\n"); fh.flush(); time.sleep(1)
