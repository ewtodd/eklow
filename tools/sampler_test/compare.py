"""eklow's sampler against llama.cpp's on the same rows (eklow-sampler-test and llama-sampler-ref, tools/sampler_test):
for each case, the candidates after the chain (deterministic chains: the same ids, probabilities within 1e-6) and a
chi-square of DRAWS draws from each.

Usage: compare.py EKLOW_BIN LLAMA_BIN [--draws N]   (LD_LIBRARY_PATH must reach the llama.cpp build's lib/)
"""
import argparse, math, subprocess, sys

CASES = [  # name, settings, deterministic
    ("llama.cpp's defaults", "", True),
    ("Bolt's GGUF (temp 1, top-k 64, top-p 0.95)", "temp=1,top_k=64", True),
    ("Sage-like: temp 0.6, top-k 20, min-p 0", "temp=0.6,top_k=20,top_p=0.95,min_p=0", True),
    ("penalties: repeat 1.3, freq 0.4, present 0.6 over 64", "repeat=1.3,freq=0.4,present=0.6", True),
    ("repeat 1.1 over the last 8", "repeat=1.1,last_n=8", True),
    ("typical 0.7", "typ=0.7", True),
    ("top-n-sigma 1.5", "sigma=1.5", True),
    ("top-k 0, top-p 0.9", "top_k=0,top_p=0.9", True),
    ("min-p 0.2, min_keep 5", "min_p=0.2,min_keep=5", True),
    ("dynatemp 0.5, exponent 1.5", "dyn=0.5,dyn_exp=1.5", True),
    ("logit bias", "bias=81:-100;90:3.5;124:false", True),
    ("greedy (temp 0)", "temp=0", True),
    ("order: temperature, top-k, min-p", "order=tkm,temp=1.5", True),
    ("XTC 0.5 / 0.05", "xtc_p=0.5,xtc_t=0.05", False),
    ("mirostat 2", "mirostat=2,tau=4,eta=0.1,temp=1", False),
    ("mirostat 1", "mirostat=1,tau=4,eta=0.1,temp=1", False),
]


def run(binary, V, seed, settings, draws):
    out = subprocess.run([binary, str(V), str(seed), settings, str(draws)], capture_output=True, text=True, check=True).stdout
    cand, hist = {}, {}
    part = cand
    for line in out.splitlines():
        if line == "draws":
            part = hist
            continue
        a, b = line.split()
        part[int(a)] = float(b)
    return cand, hist


def chi2(a, b):
    na, nb = sum(a.values()), sum(b.values())
    keys = sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) + b.get(k, 0)))
    pooled, rest = [], [0, 0]
    for k in keys:
        x, y = a.get(k, 0), b.get(k, 0)
        if x + y >= 10: pooled.append((x, y))
        else: rest[0] += x; rest[1] += y
    if sum(rest) > 0: pooled.append(tuple(rest))
    st = 0.0
    for x, y in pooled:
        e = x + y
        ex, ey = e * na / (na + nb), e * nb / (na + nb)
        st += (x - ex) ** 2 / ex + (y - ey) ** 2 / ey
    df = max(len(pooled) - 1, 1)
    z = (st - df) / math.sqrt(2 * df)
    return st, df, z


ap = argparse.ArgumentParser()
ap.add_argument("eklow")
ap.add_argument("llama")
ap.add_argument("--draws", type=int, default=20000)
a = ap.parse_args()
bad = 0
for name, settings, det in CASES:
    line = f"{name:52s}"
    if det:
        for V in (262144, 4096):
            ce, _ = run(a.eklow, V, 7, settings, 0)
            cl, _ = run(a.llama, V, 7, settings, 0)
            same = set(ce) == set(cl)
            dp = max((abs(ce[k] - cl[k]) for k in ce if k in cl), default=0.0)
            ok = same and dp < 1e-6
            bad += not ok
            line += f" | V {V}: {len(ce)} vs {len(cl)} candidates, {'same ids' if same else 'IDS DIFFER'}, max |dp| {dp:.1e}"
    _, he = run(a.eklow, 4096, 11, settings, a.draws)
    _, hl = run(a.llama, 4096, 11, settings, a.draws)
    st, df, z = chi2(he, hl)
    bad += z >= 4
    line += f" | draws: chi2 {st:.1f} df {df} z {z:+.2f} {'ok' if z < 4 else 'FAIL'}"
    print(line, flush=True)
print("all ok" if bad == 0 else f"{bad} failures")
