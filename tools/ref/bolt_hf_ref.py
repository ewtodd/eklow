"""Transformers' Gemma 4 text model in fp32 over one prompt: the logits of its last positions in eklow-oracle's dump
format (index.json + logits.f32), so tools/ref/kl_positions.py sets it beside eklow-ref-run's.

Usage: bolt_hf_ref.py --hf DIR --tokens tokens.json --out DIR [--gguf FILE] [--last N] [--chunk N] [--dtype bf16] [--check-gguf]

--hf: the BF16 checkpoint (config.json, model-*.safetensors). With --gguf the text weights are replaced by that GGUF's,
dequantized as ggml stores them (the gguf package), so both sides run the same numbers and what is left between them is
arithmetic; without it the checkpoint's own weights. --check-gguf: only compare every GGUF tensor with the checkpoint's
(relative L2), which shows the name mapping and that the converter folded nothing into the values. --dtype bf16 runs
the model as transformers usually serves it (weights and activations in bf16, the embedding scales rounded to bf16 as
Gemma4TextScaledWordEmbedding does): a yardstick for what reduced-precision activations alone cost.

The prompt runs in chunks through transformers' DynamicCache, as generate() does. The two embedding tables (11 GiB of
the per-layer one in fp32) and the head stay on the CPU, the layers go to the GPU; TF32 is off.
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

PREFIX = "model.language_model."


def hf_name(g):
    """The checkpoint's name for a GGUF tensor, or None for tensors the text model does not read."""
    top = {
        "token_embd.weight": "embed_tokens.weight",
        "per_layer_token_embd.weight": "embed_tokens_per_layer.weight",
        "per_layer_model_proj.weight": "per_layer_model_projection.weight",
        "per_layer_proj_norm.weight": "per_layer_projection_norm.weight",
        "output_norm.weight": "norm.weight",
    }
    if g in top:
        return top[g]
    m = re.match(r"blk\.(\d+)\.(.+)$", g)
    if not m:
        return None
    per = {
        "attn_norm.weight": "input_layernorm.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.weight": "pre_feedforward_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
        "inp_gate.weight": "per_layer_input_gate.weight",
        "proj.weight": "per_layer_projection.weight",
        "post_norm.weight": "post_per_layer_input_norm.weight",
        "layer_output_scale.weight": "layer_scalar",
    }
    return f"layers.{m.group(1)}.{per[m.group(2)]}" if m.group(2) in per else None


def gguf_tensors(path):
    import gguf

    r = gguf.GGUFReader(path)
    for t in r.tensors:
        name = hf_name(t.name)
        if name is None:
            continue
        a = gguf.quants.dequantize(t.data, t.tensor_type).astype(np.float32)
        shape = [int(x) for x in reversed(t.shape)]  # ggml's ne reversed is torch's shape
        yield name, t.name, a.reshape(shape)


def checkpoint(hf):
    files = sorted(Path(hf).glob("model*.safetensors"))
    handles = [safe_open(str(f), "pt") for f in files]
    where = {}
    for h in handles:
        for k in h.keys():
            where[k] = h
    return where


def build(hf, gguf_path, device, dtype):
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel, Gemma4TextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(hf).text_config
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        tm = Gemma4TextModel(cfg)
    on_cpu = {"embed_tokens.weight", "embed_tokens_per_layer.weight"}
    want = dict(tm.named_parameters())
    want.update({k: v for k, v in tm.named_buffers() if k.endswith("layer_scalar")})

    def place(name, t):
        mod_name, _, attr = name.rpartition(".")
        mod = tm.get_submodule(mod_name) if mod_name else tm
        t = t.to(torch.float32).to(dtype).to("cpu" if name in on_cpu else device)
        if isinstance(getattr(mod, attr), torch.nn.Parameter):
            setattr(mod, attr, torch.nn.Parameter(t, requires_grad=False))
        else:
            mod._buffers[attr] = t

    loaded = set()
    src = checkpoint(hf)
    for name in want:
        if PREFIX + name in src:
            place(name, src[PREFIX + name].get_tensor(PREFIX + name))
            loaded.add(name)
    missing = set(want) - loaded
    if missing:
        raise SystemExit(f"checkpoint lacks {sorted(missing)[:8]}")
    if gguf_path:
        replaced = set()
        for name, gname, a in gguf_tensors(gguf_path):
            if name not in want:
                continue  # the shared layers' K/V, which transformers never reads
            if tuple(a.shape) != tuple(want[name].shape):
                raise SystemExit(f"{gname}: shape {a.shape} against {tuple(want[name].shape)} for {name}")
            place(name, torch.from_numpy(a))
            replaced.add(name)
        left = set(want) - replaced
        if left:
            raise SystemExit(f"the GGUF lacks {sorted(left)[:8]}")
    # Buffers that are not in a checkpoint: the RoPE frequencies and the embeddings' scales.
    tm.rotary_emb = Gemma4TextRotaryEmbedding(cfg, device=device)
    tm.embed_tokens.embed_scale = torch.tensor(cfg.hidden_size**0.5).to(dtype)
    tm.embed_tokens_per_layer.embed_scale = torch.tensor(cfg.hidden_size_per_layer_input**0.5).to(dtype)
    tm.eval()
    return cfg, tm


def check_gguf(hf, gguf_path):
    src = checkpoint(hf)
    worst = []
    for name, gname, a in gguf_tensors(gguf_path):
        key = PREFIX + name
        if key not in src:
            print(f"{gname:40s} -> {name}: not in the checkpoint")
            continue
        b = src[key].get_tensor(key).to(torch.float32).numpy().reshape(a.shape)
        rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        worst.append((rel, gname))
        if not gname.startswith("blk.") or gname.startswith(("blk.0.", "blk.23.", "blk.24.", "blk.41.")):
            print(f"{gname:40s} -> {name}: rel L2 {rel:.4f}")
    worst.sort(reverse=True)
    print("largest:", ", ".join(f"{g} {r:.4f}" for r, g in worst[:6]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True)
    ap.add_argument("--tokens")
    ap.add_argument("--out")
    ap.add_argument("--gguf")
    ap.add_argument("--last", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--check-gguf", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    a = ap.parse_args()
    if a.check_gguf:
        check_gguf(a.hf, a.gguf)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    cfg, tm = build(a.hf, a.gguf, device, torch.bfloat16 if a.dtype == "bf16" else torch.float32)
    from transformers import DynamicCache

    ids = json.load(open(a.tokens))["ids"]
    T = len(ids)
    last = a.last if 0 < a.last <= T else T
    cache = DynamicCache(config=cfg)
    rows = []
    with torch.no_grad():
        for s in range(0, T, a.chunk):
            chunk = torch.tensor([ids[s : s + a.chunk]], dtype=torch.long)
            n = chunk.shape[1]
            emb = tm.embed_tokens(chunk).to(device)
            pli = tm.get_per_layer_inputs(chunk, None).to(device)
            pos = torch.arange(s, s + n, device=device)[None]
            out = tm(inputs_embeds=emb, per_layer_inputs=pli, position_ids=pos, past_key_values=cache, use_cache=True)
            h = out.last_hidden_state[0]
            lo = max(0, (T - last) - s)
            if lo < n:
                rows.append(h[lo:].cpu())
    hid = torch.cat(rows).float()
    logits = hid @ tm.embed_tokens.weight.float().T
    cap = cfg.final_logit_softcapping
    if cap:
        logits = torch.tanh(logits / cap) * cap
    outd = Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    logits.numpy().astype(np.float32).tofile(outd / "logits.f32")
    json.dump({"n_tokens": T, "n_vocab": int(logits.shape[1]), "logits_rows": int(logits.shape[0]), "tensors": []},
              open(outd / "index.json", "w"))
    from transformers import AutoTokenizer

    pieces = AutoTokenizer.from_pretrained(a.hf).convert_ids_to_tokens(ids)
    json.dump({"ids": ids, "pieces": pieces}, open(outd / "tokens.json", "w"))
    print(f"{T} tokens, {logits.shape[0]} logits rows, {'GGUF ' + a.gguf if a.gguf else 'checkpoint'} weights, {a.dtype}")


if __name__ == "__main__":
    main()
