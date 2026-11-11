"""The reference for eklow's vega checker: vegaml 0.16.1's reader and engine, and laya's System One on top.

Usage:
  ref_vega.py features STATES.json OUT_DIR [--dtype bf16|fp32] [--questions checker|all] [--device cuda]
      For every state and question: the prompt vegaml builds (ids, spans), Perception.pooled's vectors (state,
      question and last token at layers 13 and 19, options at 19, the 16 memory chunks and their mask) and the
      universal engine's output on them (logits, the thermal features, the calibrated probabilities). One
      OUT_DIR/NNNN.npz per item plus OUT_DIR/index.json. bf16 is what vegaml runs on a CUDA card (and what a policy
      fitted there saw); fp32 loads the reader in float32 with TF32 off.
  ref_vega.py engine FEATS_DIR OUT.json [--device cuda]
      The engine alone, in float32, on features read from FEATS_DIR (eklow's or this script's): what eklow's engine
      is held to given identical features.
  ref_vega.py predict STATES.json POLICY_DIR OUT.json [--dtype bf16|fp32]
      laya.Agent(vegaml=True) with the policy loaded (laya's load_calibration), POST /v1/systemone's answer for each
      state: {key: {qid: noul}} plus the readout that served each question.

STATES.json is a list of {"key", "state"} (synthetic_states.py writes one). The checker's questions are son-of-anton's
(daemon/internal/check). Needs vegaml 0.16.1 and torch; `predict` needs laya on the path (LAYA_SRC).
"""

import argparse
import json
import os
import sys

import numpy as np

CHECKER = {
    "claim": "Does the assistant's draft reply say it has done something (written, logged, updated, saved, created, "
             "removed, scheduled, checked, looked up, sent, noted or remembered) that none of this turn's tool calls did?",
    "unchecked": "The user's latest message asks about the current state of something (a scheduled job, a file, a log, a "
                 "worker) or why something happened. Does the draft reply answer it without any tool call in this turn "
                 "that checked that state? Answer no if the message asks no such thing.",
    "offtopic": "Does the draft reply fail to respond to the user's latest message (it answers something else, such as "
                "an earlier topic)?",
}
# Beyond the checker: the other question types and a yes/no with its two sides named, so every branch of the
# prompt builder and the engine (choice wells, the score rail) is exercised.
EXTRA = {
    "kind": {"type": "choice", "instructions": "What kind of request is the user's latest message?",
             "criteria": {"question": "asks for information", "task": "asks for something to be done",
                          "chat": None, "complaint": "reports a problem"}},
    "effort": {"type": "score", "instructions": "How much work did the assistant do in this turn?",
               "criteria": ["none", "a little", "some", "a lot"]},
    "polite": {"type": "boolean", "instructions": "Is the draft reply polite?",
               "criteria": {"false": "rude or curt", "true": "polite"}},
}


def questions(which):
    qs = {k: {"type": "boolean", "instructions": v} for k, v in CHECKER.items()}
    if which == "all":
        qs.update(EXTRA)
    return qs


def load_vega(dtype, device):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import vegaml
    import vegaml._common as common
    if dtype == "fp32":
        common.backbone_dtype = lambda dev: torch.float32
    return vegaml.load(device=device)


def cmd_features(a):
    import torch
    from vegaml._common import QTYPES, build_prompt, option_texts, thermal_features, thermal_tau, softmax_np
    v = load_vega(a.dtype, a.device)
    eng = v.engine
    cfg = eng.cfg
    states = json.load(open(a.states))
    os.makedirs(a.out, exist_ok=True)
    index = []
    for s in states:
        text = s["state"]
        st_ids = eng.perc.tok(text, add_special_tokens=False)["input_ids"]
        for qid, q in questions(a.questions).items():
            qi = eng._internal(q)
            k = len(option_texts(qi))
            ids, spans = build_prompt(eng.perc.tok, text, qi, cfg["max_len"], option_order=list(range(k)),
                                      opt_max_tokens=cfg["opt_max_tokens"],
                                      query_first_max_state=cfg.get("query_first_max_state", 0), state_ids=st_ids)
            item = {"ids": np.asarray(ids, dtype=np.int32), "spans": spans}
            C = eng.engine.memory_chunks
            with torch.no_grad():
                f, o, ch, cm = eng.perc.pooled([item], chunks=C, prefix_min=cfg.get("prefix_min_tokens", 4096),
                                               chunk_min_tokens=cfg.get("memory_min_tokens", 2048),
                                               chunk_tokens=cfg.get("memory_chunk_tokens", 128))
                opts = o[None].float()
                mask = torch.ones((1, k), dtype=torch.bool, device=eng.device)
                qtype = torch.tensor([QTYPES[qi["t"]]], device=eng.device)
                out = eng.engine(f.float(), opts, mask, qtype, halt=eng.halt, return_traj=False, chunks=ch, chunk_mask=cm)
                logits = out["logits"].float().cpu().numpy()
                phys = thermal_features(out, mask, eng.engine.max_steps).cpu().numpy()
            P = softmax_np(logits.astype(np.float64), thermal_tau(eng.theta, np.array([QTYPES[qi["t"]]]), phys))
            i = len(index)
            np.savez(os.path.join(a.out, "%04d.npz" % i), f=f[0].float().cpu().numpy(), o=o.float().cpu().numpy(),
                     c=ch[0].float().cpu().numpy(), cm=cm[0].float().cpu().numpy(), logits=logits[0], phys=phys[0],
                     P=P[0].astype(np.float32), nearest=out["nearest"].float().cpu().numpy(),
                     steps=out["steps_used"].cpu().numpy().astype(np.float32),
                     kinetic=out["kinetic"].float().cpu().numpy())
            index.append({"key": s["key"], "qid": qid, "question": q, "qtype": QTYPES[qi["t"]], "k": k, "ids": ids,
                          "spans": {"state": list(spans["state"]), "question": list(spans["question"]),
                                    "options": [list(x) for x in spans["options"]]},
                          "P": [float(x) for x in P[0]]})
            print("%s %s: %d tokens, P %s" % (s["key"], qid, len(ids), np.round(P[0], 6).tolist()), flush=True)
    json.dump({"dtype": a.dtype, "items": index}, open(os.path.join(a.out, "index.json"), "w"))


def cmd_engine(a):
    import torch
    from vegaml._common import thermal_features, thermal_tau, softmax_np
    v = load_vega("bf16", a.device)  # the reader is loaded but unused: only the engine runs here
    eng = v.engine
    index = json.load(open(os.path.join(a.feats, "index.json")))
    res = []
    for i, it in enumerate(index["items"]):
        z = np.load(os.path.join(a.feats, "%04d.npz" % i))
        dev = eng.device
        f = torch.tensor(z["f"], device=dev)[None]
        o = torch.tensor(z["o"], device=dev)[None]
        k = o.shape[1]
        mask = torch.ones((1, k), dtype=torch.bool, device=dev)
        qtype = torch.tensor([it["qtype"]], device=dev)
        ch = torch.tensor(z["c"], device=dev)[None]
        cm = torch.tensor(z["cm"] > 0.5, device=dev)[None]
        with torch.no_grad():
            out = eng.engine(f, o, mask, qtype, halt=eng.halt, return_traj=False, chunks=ch, chunk_mask=cm)
        logits = out["logits"].float().cpu().numpy()
        phys = thermal_features(out, mask, eng.engine.max_steps).cpu().numpy()
        P = softmax_np(logits.astype(np.float64), thermal_tau(eng.theta, np.array([it["qtype"]]), phys))
        res.append({"key": it["key"], "qid": it["qid"], "P": P[0].tolist(), "logits": logits[0].tolist(),
                    "phys": phys[0].tolist(), "steps": int(out["steps_used"][0])})
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_predict(a):
    sys.path.insert(0, os.environ.get("LAYA_SRC", "."))
    load_vega(a.dtype, "cuda")  # applies the dtype patch before laya loads its own copy
    import laya
    agent = laya.Agent(vegaml=True, device="cuda")
    agent.load_calibration(a.policy)
    qs = {k: {"type": "noul", "instructions": v} for k, v in CHECKER.items()}
    out = {}
    for s in json.load(open(a.states)):
        r = agent.predict(s["state"], qs)
        out[s["key"]] = {q: {"noul": r["answers"][q]["noul"], "readout": r["answers"][q]["readout"]} for q in qs}
        print(s["key"], {q: out[s["key"]][q]["noul"] for q in qs}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features")
    f.add_argument("states")
    f.add_argument("out")
    f.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    f.add_argument("--questions", choices=["checker", "all"], default="checker")
    f.add_argument("--device", default="cuda")
    e = sub.add_parser("engine")
    e.add_argument("feats")
    e.add_argument("out")
    e.add_argument("--device", default="cuda")
    r = sub.add_parser("predict")
    r.add_argument("states")
    r.add_argument("policy")
    r.add_argument("out")
    r.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    a = p.parse_args()
    {"features": cmd_features, "engine": cmd_engine, "predict": cmd_predict}[a.cmd](a)
