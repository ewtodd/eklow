# The PyTorch reference of the DFlash 2 draft: z-lab's dflash/model.py (DFlash2DraftModel, run in fp32 on the CPU) on the
# target's real hidden states, for several blocks along a text. It writes what eklow-dflash-test compares against: per
# block the draft's final hidden rows, the target head's logits' top 16 (ids sorted ascending, as C++ sorts them), the
# selector's whole lattice, and model.py's own greedy path (CandidateSelector.select).
#
#   python3 -I ref_torch.py DRAFT_HF_DIR TARGET_GGUF ORACLE_DIR TOKENS.txt OUT.bin B WINDOW P0,P1,...
#
# ORACLE_DIR holds eklow-oracle's l_out-N.f32 ([n_tokens][2048] fp32) for the draft's target layers; TOKENS.txt the
# prompt's token ids, one line; B the block size; WINDOW overrides the draft's sliding window (0: the draft's own).
import importlib.util
import json
import struct
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

import gguf

MODEL_PY = "/home/e-play/eklow-qwen-data/dflash-src/dflash/model.py"
draft_dir, target_gguf, oracle_dir, tokens_txt, out_path, B, window, ps = sys.argv[1:9]
B, window = int(B), int(window)
ps = [int(x) for x in ps.split(",")]
torch.set_grad_enabled(False)

spec = importlib.util.spec_from_file_location("dflash_model", MODEL_PY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

cfg = json.load(open(f"{draft_dir}/config.json"))
config = Qwen3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "dtype")})
config.dflash_config = cfg["dflash_config"]
config.num_target_layers = cfg["num_target_layers"]
if window:
    config.sliding_window = window
config._attn_implementation = "sdpa"
model = m.DFlash2DraftModel(config)
sd = load_file(f"{draft_dir}/model.safetensors")
sd = {(k + ".weight" if k.startswith("candidate_selector.") and k.endswith("codebook") else k): v.float() for k, v in sd.items()}
print(model.load_state_dict(sd, strict=True))
model = model.float().eval()
layer_ids = model.target_layer_ids
print("target layers", layer_ids, "window", config.sliding_window, "block", B)

tokens = [int(x) for x in open(tokens_txt).read().split()]
n = len(tokens)
feats = np.concatenate(
    [np.fromfile(f"{oracle_dir}/l_out-{i}.f32", dtype=np.float32).reshape(n, -1) for i in layer_ids], axis=1)
print("features", feats.shape)

r = gguf.GGUFReader(target_gguf)
tens = {t.name: t for t in r.tensors}
emb_t, out_t = tens["token_embd.weight"], tens["output.weight"]
emb = gguf.quants.dequantize(np.asarray(emb_t.data), emb_t.tensor_type)
head_w = torch.from_numpy(gguf.quants.dequantize(np.asarray(out_t.data), out_t.tensor_type).astype(np.float32))
print("head", head_w.shape)
head = torch.nn.Linear(head_w.shape[1], head_w.shape[0], bias=False)
head.weight = torch.nn.Parameter(head_w, requires_grad=False)
mask_id = model.mask_token_id

recs = []


def put(name, a):
    recs.append((name, np.ascontiguousarray(a)))


put("meta", np.array([B, config.sliding_window, n, len(ps)], dtype=np.int32))
for p in ps:
    ids = [tokens[p]] + [mask_id] * (B - 1)
    noise = torch.from_numpy(emb[ids].astype(np.float32))[None]
    th = torch.from_numpy(feats[:p])[None]
    pos = torch.arange(p + B)[None]
    hid = model(target_hidden=th, noise_embedding=noise, position_ids=pos)[:, 1 - B:, :]  # [1][B-1][2048]
    logits = head(hid)
    sel = model.candidate_selector
    unary, cands = torch.topk(logits, sel.top_k, dim=-1, sorted=False)
    order = torch.argsort(cands, dim=-1)
    cands, unary = cands.gather(-1, order), unary.gather(-1, order)
    gate = sel.hidden_projection(hid)  # [1][B-1][256]
    lat = torch.zeros(B - 1, 16, 16)
    for i in range(B - 1):
        prev_ids = torch.tensor([tokens[p]]) if i == 0 else cands[0, i - 1]
        cond = sel.predecessor_codebook(prev_ids) * gate[0, i]  # [npred][256]
        lat[i] = unary[0, i][None] + cond @ sel.successor_codebook(cands[0, i]).T
    path, _, _ = sel.select(hid, logits, torch.tensor([tokens[p]]), 0.0)
    # the walk on the lattice as llama.cpp does it, to check the lattice itself
    prev, walk = 0, []
    for i in range(B - 1):
        prev = int(torch.argmax(lat[i, prev if i else 0]))
        walk.append(int(cands[0, i, prev]))
    match = [int(a == b) for a, b in zip(walk, path[0].tolist())]
    truth = tokens[p + 1:p + B]
    agree = 0
    for a, b in zip(path[0].tolist(), truth):
        if a != b:
            break
        agree += 1
    print(f"p={p} anchor={tokens[p]} path={path[0].tolist()} lattice-walk-equal={all(match)} text-prefix-agree={agree}/{len(truth)}")
    put(f"hid{p}", hid[0].numpy())
    put(f"cand{p}", cands[0].numpy().astype(np.int32))
    put(f"unary{p}", unary[0].numpy())
    put(f"lat{p}", lat.numpy())
    put(f"path{p}", path[0].numpy().astype(np.int32))

with open(out_path, "wb") as f:
    for name, a in recs:
        nb = name.encode()
        dt = 0 if a.dtype == np.float32 else 1
        f.write(struct.pack("<I", len(nb)) + nb + struct.pack("<BI", dt, a.ndim))
        f.write(struct.pack(f"<{a.ndim}I", *a.shape))
        f.write(a.tobytes())
print("wrote", out_path)
