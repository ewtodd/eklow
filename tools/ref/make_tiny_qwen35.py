"""A tiny random qwen35moe GGUF, to check eklow's reference forward against llama.cpp without the 20 GB file.

Usage: python3 tools/ref/make_tiny_qwen35.py VOCAB_GGUF OUT_GGUF [--split] [--quant q8_0|f32]

The tokenizer and everything else that is not the model comes from VOCAB_GGUF (a vocab-only qwen35moe file),
so token ids mean what they mean in the real model. The model is 4 layers (3 Gated DeltaNet, 1 attention), hidden
256, 8 experts (2 used), the real file's tensor names, types per tensor kind and shapes' structure. Weights are
random with scales that keep the activations finite and the routing undecided enough to matter. --split
writes ffn_gate_exps / ffn_up_exps separately instead of a merged ffn_gate_up_exps.
"""

import argparse

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
from gguf.quants import quantize

ap = argparse.ArgumentParser()
ap.add_argument("vocab")
ap.add_argument("out")
ap.add_argument("--split", action="store_true")
ap.add_argument("--quant", default="q8_0")
a = ap.parse_args()

rng = np.random.default_rng(7)
E, NL, H, HKV, D, NROT = 256, 4, 4, 2, 64, 16
NE, NU, FE, FS = 8, 2, 64, 64
K, HK, HV, S = 4, 2, 4, 32
INNER = HV * S
CONV = 2 * HK * S + INNER

r = GGUFReader(a.vocab)
n_vocab = len(r.fields["tokenizer.ggml.tokens"].data)
w = GGUFWriter(a.out, "qwen35moe")
for key, f in r.fields.items():
    if key.startswith("GGUF.") or key.startswith("qwen35moe.") or key in ("general.architecture", "general.file_type"):
        continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        w.add_array(key, f.contents())
    else:
        w.add_key_value(key, f.contents(), vt)
p = "qwen35moe."
for k, v in {
    "block_count": NL, "context_length": 4096, "embedding_length": E, "attention.head_count": H,
    "attention.head_count_kv": HKV, "attention.key_length": D, "attention.value_length": D,
    "rope.dimension_count": NROT, "expert_count": NE, "expert_used_count": NU, "expert_feed_forward_length": FE,
    "expert_shared_feed_forward_length": FS, "ssm.conv_kernel": K, "ssm.inner_size": INNER, "ssm.state_size": S,
    "ssm.group_count": HK, "ssm.time_step_rank": HV, "full_attention_interval": 4, "nextn_predict_layers": 0,
}.items():
    w.add_uint32(p + k, v)
w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-6)
w.add_float32(p + "rope.freq_base", 1e7)
w.add_array(p + "rope.dimension_sections", [3, 3, 2, 0])

qt = {"q8_0": GGMLQuantizationType.Q8_0, "f32": GGMLQuantizationType.F32}[a.quant]


def mat(name, *shape, scale=0.05, quant=True):
    x = rng.normal(0, scale, shape).astype(np.float32)
    if quant and qt != GGMLQuantizationType.F32:
        w.add_tensor(name, quantize(x, qt), raw_dtype=qt)
    else:
        w.add_tensor(name, x)


def vec(name, n, mean=1.0, scale=0.1):
    w.add_tensor(name, (mean + rng.normal(0, scale, n)).astype(np.float32))


mat("token_embd.weight", n_vocab, E, scale=0.3)
vec("output_norm.weight", E)
mat("output.weight", n_vocab, E, scale=0.1)
for l in range(NL):
    b = f"blk.{l}."
    vec(b + "attn_norm.weight", E)
    vec(b + "post_attention_norm.weight", E)
    if (l + 1) % 4:
        mat(b + "attn_qkv.weight", CONV, E)
        mat(b + "attn_gate.weight", INNER, E)
        w.add_tensor(b + "ssm_conv1d.weight", rng.normal(0, 0.4, (CONV, K)).astype(np.float32))
        w.add_tensor(b + "ssm_dt.bias", rng.normal(0.5, 0.3, HV).astype(np.float32))
        w.add_tensor(b + "ssm_a", (-np.exp(rng.uniform(0, 1.5, HV))).astype(np.float32))
        mat(b + "ssm_beta.weight", HV, E, quant=False)
        mat(b + "ssm_alpha.weight", HV, E, quant=False)
        vec(b + "ssm_norm.weight", S, 1.0, 0.1)
        mat(b + "ssm_out.weight", E, INNER)
    else:
        mat(b + "attn_q.weight", 2 * D * H, E)
        mat(b + "attn_k.weight", D * HKV, E)
        mat(b + "attn_v.weight", D * HKV, E)
        mat(b + "attn_output.weight", E, D * H)
        vec(b + "attn_q_norm.weight", D)
        vec(b + "attn_k_norm.weight", D)
    mat(b + "ffn_gate_inp.weight", NE, E, scale=0.3, quant=False)
    w.add_tensor(b + "ffn_gate_inp_shexp.weight", rng.normal(0, 0.1, E).astype(np.float32))
    mat(b + "ffn_gate_shexp.weight", FS, E)
    mat(b + "ffn_up_shexp.weight", FS, E)
    mat(b + "ffn_down_shexp.weight", E, FS)
    if a.split:
        mat(b + "ffn_gate_exps.weight", NE, FE, E)
        mat(b + "ffn_up_exps.weight", NE, FE, E)
    else:
        mat(b + "ffn_gate_up_exps.weight", NE, 2 * FE, E)
    mat(b + "ffn_down_exps.weight", NE, E, FE)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", a.out, "vocab", n_vocab)
