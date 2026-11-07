# The PyTorch reference of a plain DFlash draft (z-lab's dflash/model.py, DFlashDraftModel, in fp32 on the CPU) for z-lab's plain draft kind (built for Gemma 4
# 26B A4B, since removed from eklow), on fixed features and tokens, and its comparison with eklow's reference forward (eklow-dflash-host).
# Fixed inputs rather than the target's real hidden states: the checks are of the draft's arithmetic (the sliding/full layer
# pattern with its masks, the embedding scale, the head, the softcap), which random features exercise as well; the features
# are N(0, scale^2) from a numpy seed. The embedding table and the head are the target's token_embd.weight (tied), dequantized.
#
#   python3 -I ref_torch_plain.py DRAFT_HF_DIR TARGET_GGUF OUTDIR CTX B WINDOW SEED SCALE ANCHORS
#       writes OUTDIR/feats.f32 and OUTDIR/torch.npz (per block: hid, cand, unary, logits' top-16 by descending value)
#   python3 -I ref_torch_plain.py --compare OUTDIR HOST.bin CTX B
#       compares eklow-dflash-host's output (same inputs: --feats OUTDIR/feats.f32 --anchors ...) with torch.npz
#
# WINDOW overrides the draft's sliding window (0: its own, 2048) so that the sliding layers' mask matters at short contexts.
# DRAFT_HF_DIR is z-lab/gemma-4-26B-A4B-it-DFlash's files (config.json, model.safetensors).
import importlib.util
import json
import struct
import sys

import numpy as np

MODEL_PY = "/home/e-play/eklow-qwen-data/dflash-src/dflash/model.py"


def compare(outdir, host_bin, ctx, B):
    t = np.load(f"{outdir}/torch.npz")
    data = open(host_bin, "rb").read()
    off, b = 0, 0
    P = B - 1
    tot_h = [0.0, 0.0]
    worst_u = 0.0
    cand_diff = 0
    while off < len(data):
        anchor, pos, bb, K = struct.unpack_from("<4i", data, off)
        off += 16
        cand = np.frombuffer(data, np.int32, P * K, off).reshape(P, K); off += P * K * 4
        unary = np.frombuffer(data, np.float32, P * K, off).reshape(P, K); off += P * K * 4
        score = np.frombuffer(data, np.float32, P * K * K, off).reshape(P, K, K); off += P * K * K * 4
        E = t[f"hid{b}"].shape[-1]
        hid = np.frombuffer(data, np.float32, P * E, off).reshape(P, E); off += P * E * 4
        th = t[f"hid{b}"]
        rel = np.linalg.norm(hid - th) / np.linalg.norm(th)
        cd = int((cand != t[f"cand{b}"]).sum())
        # candidates can differ in order only on near-ties: compare the scores of the same ranks
        mu = float(np.abs(unary - t[f"unary{b}"]).max())
        rep = bool((score == unary[:, None, :]).all())
        print(f"block {b} anchor {anchor} pos {pos}: hidden rel err {rel:.2e}, candidates differing {cd}/{cand.size}, unary max abs {mu:.2e}, lattice rows replicate unary {rep}, "
              f"greedy tokens {'equal' if (cand[:, 0] == t[f'cand{b}'][:, 0]).all() else 'DIFFERENT'}")
        tot_h[0] += float(((hid - th) ** 2).sum()); tot_h[1] += float((th ** 2).sum())
        worst_u = max(worst_u, mu); cand_diff += cd
        b += 1
    print(f"total: hidden rel {np.sqrt(tot_h[0] / tot_h[1]):.2e}, unary max abs {worst_u:.2e}, candidates differing {cand_diff}")


def main():
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
        return
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3Config
    import gguf

    draft_dir, target_gguf, outdir, ctx, B, window, seed, scale, anchors = sys.argv[1:10]
    ctx, B, window, seed, scale = int(ctx), int(B), int(window), int(seed), float(scale)
    anchors = [int(x) for x in anchors.split(",")]
    torch.set_grad_enabled(False)
    spec = importlib.util.spec_from_file_location("dflash_model", MODEL_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    cfg = json.load(open(f"{draft_dir}/config.json"))
    config = Qwen3Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "dtype", "dflash_config", "block_size", "num_target_layers", "final_logit_softcapping")})
    config.dflash_config = cfg["dflash_config"]
    config.num_target_layers = cfg["num_target_layers"]
    config.block_size = cfg["block_size"]
    softcap = float(cfg["final_logit_softcapping"])
    if window:
        config.sliding_window = window
    config._attn_implementation = "sdpa"
    model = m.DFlashDraftModel(config)
    sd = {k: v.float() for k, v in load_file(f"{draft_dir}/model.safetensors").items()}
    print(model.load_state_dict(sd, strict=True))
    model = model.float().eval()
    layer_ids = model.target_layer_ids
    F = len(layer_ids) * config.hidden_size
    nb = len(anchors)
    n_rows = ctx + 3 * nb
    rng = np.random.default_rng(seed)
    feats = (rng.standard_normal((n_rows, F)) * scale).astype(np.float32)
    feats.tofile(f"{outdir}/feats.f32")

    r = gguf.GGUFReader(target_gguf)
    emb_t = next(t for t in r.tensors if t.name == "token_embd.weight")
    W = torch.from_numpy(gguf.quants.dequantize(np.asarray(emb_t.data), emb_t.tensor_type).astype(np.float32))  # [V][E], tied head
    print("token_embd", tuple(W.shape), emb_t.tensor_type.name)
    emb_scale = float(config.hidden_size) ** 0.5
    mask_id = model.mask_token_id
    out = {}
    for b, a in enumerate(anchors):
        p = ctx + 3 * b
        ids = torch.tensor([a] + [mask_id] * (B - 1))
        noise = (W[ids] * emb_scale)[None]
        th = torch.from_numpy(feats[:p])[None]
        pos = torch.arange(p + B)[None]
        hid = model(target_hidden=th, noise_embedding=noise, position_ids=pos)[:, 1 - B:, :][0]  # [B-1][E]
        logits = hid @ W.T
        logits = torch.tanh(logits / softcap) * softcap
        vals, idx = torch.topk(logits, 16, dim=-1)
        out[f"hid{b}"] = hid.numpy()
        out[f"cand{b}"] = idx.numpy().astype(np.int32)
        out[f"unary{b}"] = vals.numpy()
        print(f"block {b}: anchor {a} pos {p} greedy {idx[:, 0].tolist()}")
    np.savez(f"{outdir}/torch.npz", **out)


main()
