#!/usr/bin/env python3
"""Check eklow's Inkling 2 output parser against llama.cpp c74c71a6's (common/parsers/inkling.cpp, through llama-server's
own path: eklow-chat-oracle feeds each output to the session one token at a time).

  parse_compare.py EKLOW_CHAT EKLOW_PARSE GGUF M0_DIR --oracle BIN [--fuzz N --seed S --work DIR]

For every output of M0_DIR/cases.json, and for N random outputs under each of a set of requests (tools auto, no parallel
calls, required, tool_choice none, a response format, reasoning_format none, continued assistant messages, odd tool
names), the final message of llama.cpp (reasoning, content, each tool call's name and arguments text, whether finishing
raised) must equal eklow's, and eklow's deltas must add up to its message (the prefill's message plus what was streamed).
The end tokens are checked on their own: an output followed by <|content_model_end_sampling|> or <|endoftext|> parses as
the output does and reports the turn done.
"""
import argparse
import collections
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_fuzz  # noqa: E402
from chat_compare import run_eklow, run_oracle  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": n, "description": "d", "parameters": {"type": "object", "properties": {}}}}
         for n in ["read_file", "get_time", "run_command", "é", 'a"b', ""]]
NAMES = ["read_file", "get_time", "run_command", "é", 'a"b', ""]
U = {"role": "user", "content": "go"}
REQUESTS = {
    "plain": {"messages": [U]},
    "tools": {"messages": [U], "tools": TOOLS},
    "tools-serial": {"messages": [U], "tools": TOOLS, "parallel_tool_calls": False},
    "tools-required": {"messages": [U], "tools": TOOLS, "tool_choice": "required"},
    "tools-required-serial": {"messages": [U], "tools": TOOLS[:3], "tool_choice": "required", "parallel_tool_calls": False},
    "tools-none": {"messages": [U], "tools": TOOLS, "tool_choice": "none"},
    "response-format": {"messages": [U], "response_format": {"type": "json_object"}},
    "response-format-tools": {"messages": [U], "tools": TOOLS, "response_format": {"type": "json_object"}},
    "reasoning-none": {"messages": [U], "reasoning_format": "none"},
    "reasoning-none-tools": {"messages": [U], "tools": TOOLS, "reasoning_format": "none"},
    "continue-text": {"messages": [U, {"role": "assistant", "content": "The answer is"}]},
    "continue-text-tools": {"messages": [U, {"role": "assistant", "content": "The answer is"}], "tools": TOOLS},
    "continue-thinking": {"messages": [U, {"role": "assistant", "content": "", "reasoning_content": "Let me think"}], "tools": TOOLS},
    "continue-both": {"messages": [U, {"role": "assistant", "content": "partial", "reasoning_content": "thought"}], "tools": TOOLS, "reasoning_format": "none"},
}


def llama_calls(m):
    return [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("eklow_parse")
    ap.add_argument("gguf")
    ap.add_argument("m0")
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default=".")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)

    cases = json.load(open(os.path.join(a.m0, "cases.json"), encoding="utf-8"))
    for name, req in REQUESTS.items():
        n = a.fuzz
        cases.append({"id": name, "request": req, "outputs": chat_fuzz.outputs(n, a.seed, NAMES if "tools" in req else [], "response_format" in req)})
    reqs = [c["request"] for c in cases]
    cpath, opath = os.path.join(a.work, "parse-%d.cases.json" % a.seed), os.path.join(a.work, "parse-%d.out.json" % a.seed)
    json.dump([{"id": c["id"], "request": c["request"], "outputs": c["outputs"]} for c in cases], open(cpath, "w", encoding="utf-8"), ensure_ascii=False)
    subprocess.run([a.oracle, "--model", a.gguf, "--cases", cpath, "--out", opath], check=True, capture_output=True,
                   env=dict(os.environ, LD_LIBRARY_PATH="/run/opengl-driver/lib"))
    ll = json.load(open(opath, encoding="utf-8"))
    ek_chat = run_eklow(a.eklow_chat, a.gguf, reqs)

    lines, index = [], []
    for ci, (c, r, e) in enumerate(zip(cases, ll, ek_chat)):
        if "error" in r or "error" in e:
            continue
        m = e["mode"]
        for oi, p in enumerate(r["parsed"]):
            # llama.cpp's streaming diff throws when a call it showed is dropped ("Invalid diff: now finding less tool calls"):
            # its message is then the one before the token that did it, so eklow is checked on the tokens up to there
            ids = p["tokens"][:len(p["deltas"])] if p.get("parse_error", "").startswith("Invalid diff") else p["tokens"]
            lines.append(json.dumps({"ids": ids, "tools": m["tools"], "tool_choice": m["tool_choice"], "parallel_tool_calls": m["parallel_tool_calls"],
                                     "response_format": m["response_format"], "extract_reasoning": m["extract_reasoning"], "prefill": e["generation_ids"]}))
            index.append((ci, oi))
    out = subprocess.run([a.eklow_parse, a.gguf], input=("\n".join(lines) + "\n").encode(), capture_output=True, check=True).stdout.decode().split("\n")
    results = [json.loads(x) for x in out[:len(lines)]]

    bad, n_trunc, n_err_llama, n_diffthrow = [], 0, 0, 0
    cats = collections.Counter()
    for (ci, oi), res in zip(index, results):
        c, r = cases[ci], ll[ci]["parsed"][oi]
        lm = r["message"]
        em = res["message"]
        want = (lm.get("reasoning_content", ""), lm.get("content") or "", llama_calls(lm))
        got = (em["reasoning"], em["content"], [(t["name"], t["arguments_text"]) for t in em["tool_calls"]])
        why = None
        if want != got:
            why = "message differs"
        elif r.get("parse_error", "").startswith("Invalid diff"):
            n_diffthrow += 1
        elif ("parse_error" in r) != bool(em["errors"]):
            why = "parse_error %s in llama.cpp, errors %s in eklow" % ("parse_error" in r, em["errors"])
        else:
            pm, sm = res["prefilled"], res["streamed"]
            if pm["reasoning"] + sm["reasoning"] != em["reasoning"] or pm["content"] + sm["content"] != em["content"] \
                    or len(pm["tool_calls"]) + len(sm["tool_calls"]) != len(em["tool_calls"]):
                why = "deltas do not add up to the message"
        n_trunc += any(t["truncated"] for t in em["tool_calls"])
        n_err_llama += "parse_error" in r
        cats["ok" if not why else why] += 1
        if why:
            bad.append((ci, oi, why, want, got))
    total = len(index)
    print("parser: %d outputs under %d requests (%d from M0), %d differences; llama.cpp raised in %d, %d messages hold a cut-off call"
          % (total, len(cases), len(json.load(open(os.path.join(a.m0, "cases.json")))), len(bad), n_err_llama, n_trunc))
    print("  (%d of them end in llama.cpp's own diff exception, checked on the tokens before it)" % n_diffthrow)
    for ci, oi, why, want, got in bad[:int(os.environ.get("SHOW", 12))]:
        print("  %s #%d: %s\n    output: %r\n    llama.cpp: %r\n    eklow:     %r" % (cases[ci]["id"], oi, why, cases[ci]["outputs"][oi][:400], want, got))

    # end tokens: an output followed by one is the output, done
    sample = random.Random(3).sample(range(len(index)), min(400, len(index)))
    lines2 = []
    for k in sample:
        d = json.loads(lines[k])
        for end in (200006, 199999):
            d2 = dict(d, ids=d["ids"] + [end, 1735, 200010])
            lines2.append(json.dumps(d2))
    out2 = subprocess.run([a.eklow_parse, a.gguf], input=("\n".join(lines2) + "\n").encode(), capture_output=True, check=True).stdout.decode().split("\n")
    nbad = 0
    for j, line in enumerate(lines2):
        res2 = json.loads(out2[j])
        res = results[sample[j // 2]]
        if res2["message"] != res["message"] or res2["end"] != json.loads(line)["ids"][-3]:
            nbad += 1
    print("end tokens: %d turns followed by an end token and more tokens: %d differ from the turn alone or miss the end" % (len(lines2), nbad))
    sys.exit(1 if bad or nbad else 0)


if __name__ == "__main__":
    main()
