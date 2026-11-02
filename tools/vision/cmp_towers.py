"""Ornith's vision tensors (local shard 1) against stock Qwen3.6's, read from Hugging Face with range requests (only the
visual tensors' bytes are fetched). Prints per tensor identical / differs, with the worst element difference."""
import hashlib, json, struct, sys
import numpy as np, requests

REPO, REV = "Qwen/Qwen3.6-35B-A3B", "995ad96e"
def url(f): return f"https://huggingface.co/{REPO}/resolve/{REV}/{f}"
idx = requests.get(url("model.safetensors.index.json")).json()["weight_map"]
vis = {k: v for k, v in idx.items() if k.startswith("model.visual.")}
print("stock visual tensors:", len(vis), "in shards", sorted(set(vis.values())))
def header_remote(f):
    n = struct.unpack("<Q", requests.get(url(f), headers={"Range": "bytes=0-7"}).content)[0]
    h = json.loads(requests.get(url(f), headers={"Range": f"bytes=8-{7+n}"}).content)
    return h, 8 + n
def header_local(p):
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n
oh, ooff = header_local(sys.argv[1])
cache = {}
same = diff = 0
for k, shard in sorted(vis.items()):
    if shard not in cache: cache[shard] = header_remote(shard)
    sh, soff = cache[shard]
    s, o = sh[k], oh[k]
    if s["dtype"] != o["dtype"] or s["shape"] != o["shape"]:
        print("DIFF shape/dtype", k, s["dtype"], s["shape"], o["dtype"], o["shape"]); diff += 1; continue
    a, b = s["data_offsets"], o["data_offsets"]
    sb = requests.get(url(shard), headers={"Range": f"bytes={soff+a[0]}-{soff+a[1]-1}"}).content
    with open(sys.argv[1], "rb") as fh:
        fh.seek(ooff + b[0]); ob = fh.read(b[1] - b[0])
    if sb == ob: same += 1; continue
    diff += 1
    f = lambda x: (np.frombuffer(x, np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64) if s["dtype"] == "BF16" else np.frombuffer(x, np.float32 if s["dtype"]=="F32" else np.float16).astype(np.float64)
    fa, fb = f(sb), f(ob); d = np.abs(fa - fb)
    print(f"DIFF {k} {s['dtype']}: max|d| {d.max():.3e} (max|ornith| {np.abs(fb).max():.3e}), rel L2 {np.linalg.norm(fa-fb)/(np.linalg.norm(fb)+1e-30):.3e}, {np.count_nonzero(d)}/{d.size} differ")
print("identical:", same, "different:", diff)
