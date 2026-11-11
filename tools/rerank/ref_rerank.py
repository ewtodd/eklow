"""The reranker's scores in fp32 from the GGUF's own (Q8_0) weights: transformers' Qwen3 with the GGUF dequantized into it,
and the yes/no head (cls.output) on the last token's normed state, as llama.cpp's rank pooling reads it. No activation is
quantized, so this is what both llama.cpp (int8 activations against Q8_0 weights, a q8_0 or f16 K/V cache) and eklow
(fp16 activations, an fp16 cache) approximate.

Usage: ref_rerank.py GGUF TOKENS.json PAIRS.json OUT.json [--device cuda|cpu] [--dtype fp32|fp64]
  TOKENS.json: {name: [[prompt ids of document i], ...]} (eklow-rerank tokens writes it; it equals llama.cpp's /tokenize).
  OUT.json:    {name: {"scores": [...]}} as rerank_compare.py fetch writes it.
Needs torch, transformers and llama.cpp's gguf-py on the path.
"""

import argparse
import json

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gguf")
    p.add_argument("tokens")
    p.add_argument("pairs")
    p.add_argument("out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp32", "fp64"], default="fp32")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import Qwen3Config, Qwen3ForCausalLM

    r = GGUFReader(a.gguf)
    f = {k: v.contents() for k, v in r.fields.items() if k.startswith("qwen3.")}
    t = {x.name: x for x in r.tensors}

    def w(name):
        x = t[name]
        return torch.from_numpy(np.ascontiguousarray(dequantize(x.data, x.tensor_type).reshape([int(d) for d in reversed(x.shape)])).astype(np.float64))

    cfg = Qwen3Config(vocab_size=int(t["token_embd.weight"].shape[1]), hidden_size=f["qwen3.embedding_length"],
                      intermediate_size=f["qwen3.feed_forward_length"], num_hidden_layers=f["qwen3.block_count"],
                      num_attention_heads=f["qwen3.attention.head_count"], num_key_value_heads=f["qwen3.attention.head_count_kv"],
                      head_dim=f["qwen3.attention.key_length"], rms_norm_eps=f["qwen3.attention.layer_norm_rms_epsilon"],
                      rope_theta=f["qwen3.rope.freq_base"], max_position_embeddings=f["qwen3.context_length"], tie_word_embeddings=True)
    dt = torch.float64 if a.dtype == "fp64" else torch.float32
    m = Qwen3ForCausalLM(cfg).to(dt)
    sd = {"model.embed_tokens.weight": w("token_embd.weight"), "model.norm.weight": w("output_norm.weight")}
    names = {"attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm", "attn_q": "self_attn.q_proj",
             "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm", "ffn_gate": "mlp.gate_proj",
             "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
    for l in range(cfg.num_hidden_layers):
        for g, h in names.items():
            sd["model.layers.%d.%s.weight" % (l, h)] = w("blk.%d.%s.weight" % (l, g))
    sd = {k: v.to(dt) for k, v in sd.items()}
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    m.load_state_dict(sd, strict=True)
    m = m.to(a.device).eval()
    cls = w("cls.output.weight").to(dt).to(a.device)  # [2][E]: yes, no
    toks = json.load(open(a.tokens))
    res = {}
    with torch.no_grad():
        for req in json.load(open(a.pairs)):
            scores = []
            for ids in toks[req["name"]]:
                x = torch.tensor([ids], device=a.device)
                h = m.model(input_ids=x).last_hidden_state[0, -1]  # after the final norm
                logits = cls @ h
                scores.append(float(torch.softmax(logits.double(), -1)[0]))
            res[req["name"]] = {"scores": scores}
            print(req["name"], ["%.5f" % s for s in scores], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
