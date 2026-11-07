# Teacher-forced acceptance of a plain DFlash draft (z-lab's kind; built for Gemma 4 26B A4B, since removed from eklow) on the target's real hidden states, in z-lab's
# model.py (fp32, CPU), for the masks the layers may have:
#   model.py   sliding layers causal inside the block, the full layer not (z-lab's code, vLLM's: how the draft was trained)
#   all-open   every layer not causal (how llama.cpp at the pin runs it: its GGUF has no per-layer causal flag)
#   capW       model.py with the full layer seeing only the last W positions (cap2048, cap8192, ...)
# (VARIANTS=model.py,cap2048 picks them; default model.py,all-open)
# At every STRIDE-th position p of the text a block [token p, mask x 15] is drafted from the features of positions < p; the
# accepted length is how many of its greedy tokens (the head of the target's embedding table, softcapped) agree with the text
# that follows, i.e. the acceptance if the target said the text (greedy: the text is the target's own greedy output). The
# acceptance reported, like the llama.cpp runs', is tokens per step: 1 + the mean accepted length.
#
#   python3 -I accept_torch_plain.py DRAFT_HF_DIR TARGET_GGUF STRIDE B NAME=ORACLE_DIR:TOKENS.txt:FIRST ...
# ORACLE_DIR has eklow-oracle's l_out-N.f32 for the draft's target layers; FIRST the first position to draft from (the answer).
import importlib.util
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

import gguf

MODEL_PY = "/home/e-play/eklow-qwen-data/dflash-src/dflash/model.py"
draft_dir, target_gguf, stride, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
sets = [a.split("=", 1) for a in sys.argv[5:]]
torch.set_grad_enabled(False)
torch.set_num_threads(10)
spec = importlib.util.spec_from_file_location("dflash_model", MODEL_PY)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cfg = json.load(open(f"{draft_dir}/config.json"))
softcap = float(cfg["final_logit_softcapping"])
sd = {k: v.float() for k, v in load_file(f"{draft_dir}/model.safetensors").items()}


def build(is_causal):
    config = Qwen3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "dtype", "dflash_config", "block_size", "num_target_layers", "final_logit_softcapping")})
    config.dflash_config = cfg["dflash_config"]
    config.num_target_layers = cfg["num_target_layers"]
    config.block_size = cfg["block_size"]
    config._attn_implementation = "sdpa"
    if is_causal is not None:
        config.is_causal = is_causal
    model = m.DFlashDraftModel(config)
    model.load_state_dict(sd, strict=True)
    return model.float().eval(), config


def variant(name):
    if name == "model.py":
        return build(None)
    if name == "all-open":
        return build(False)
    if name.startswith("cap"):  # the full layer sees only the last W positions (a symmetric window mask, still not causal)
        model, config = build(None)
        for i, t in enumerate(config.layer_types):
            if t == "full_attention":
                model.layers[i].self_attn.sliding_window = int(name[3:])
        return model, config
    raise SystemExit("unknown variant " + name)


variants = {n: variant(n) for n in os.environ.get("VARIANTS", "model.py,all-open").split(",")}
r = gguf.GGUFReader(target_gguf)
emb_t = next(t for t in r.tensors if t.name == "token_embd.weight")
W = torch.from_numpy(gguf.quants.dequantize(np.asarray(emb_t.data), emb_t.tensor_type).astype(np.float32))
scale = float(cfg["hidden_size"]) ** 0.5
mask_id = cfg["dflash_config"]["mask_token_id"]
layer_ids = cfg["dflash_config"]["target_layer_ids"]
res = {v: [] for v in variants}
for name, spec_s in sets:
    odir, toks, first = spec_s.split(":")
    tokens = [int(x) for x in open(toks).read().split()]
    n = len(tokens)
    feats = np.concatenate([np.fromfile(f"{odir}/l_out-{i}.f32", dtype=np.float32).reshape(n, -1) for i in layer_ids], axis=1)
    out = {v: [] for v in variants}
    for p in range(int(first), n - B, stride):
        ids = torch.tensor([tokens[p]] + [mask_id] * (B - 1))
        noise = (W[ids] * scale)[None]
        th = torch.from_numpy(feats[:p])[None]
        pos = torch.arange(p + B)[None]
        for v, (model, config) in variants.items():
            hid = model(target_hidden=th, noise_embedding=noise, position_ids=pos)[:, 1 - B:, :][0]
            logits = torch.tanh((hid @ W.T) / softcap) * softcap
            pred = logits.argmax(-1).tolist()
            truth = tokens[p + 1:p + B]
            k = 0
            while k < len(truth) and pred[k] == truth[k]:
                k += 1
            out[v].append(k)
            res[v].append(k)
    print(name, "positions", len(next(iter(out.values()))), " ".join(f"{v}: {1 + np.mean(a):.2f}" for v, a in out.items()), flush=True)
print("all", " ".join(f"{v}: {1 + np.mean(a):.2f} (+/- {np.std(a) / np.sqrt(len(a)):.2f}, n={len(a)})" for v, a in res.items()))
for v, a in res.items():
    print(v, "accepted-length histogram", np.bincount(a, minlength=B).tolist())
base = np.array(res[next(iter(res))])
for v, a in res.items():
    d = np.array(a) - base  # paired by position
    print(f"{v} minus {next(iter(res))}: {d.mean():+.3f} tokens per step (paired se {d.std() / np.sqrt(len(d)):.3f})")
