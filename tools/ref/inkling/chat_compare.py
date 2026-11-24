#!/usr/bin/env python3
"""Check eklow's Inkling 2 chat template port against llama.cpp c74c71a6 (the M0 ground truth and fresh oracle runs) and
report where Python's Jinja2, in Hugging Face's environment, differs from llama.cpp.

  chat_compare.py EKLOW_CHAT GGUF M0_DIR TEMPLATE.jinja [--oracle BIN --fuzz N --seed S --work DIR]

  1. M0: every request of M0_DIR/cases.json against chat.json: the rendered prompt byte for byte, the same cases raising
     (and the error text the template raised), generation_prompt, and the token ids (eklow tokenizes request text without
     special tokens: llama.cpp parses them everywhere, so a request that holds "<|" is only checked to decode back to the
     prompt, and counted).
  2. Fuzz (with --oracle, the eklow-chat-oracle binary): N random requests (chat_fuzz.py) through the oracle and eklow.
  3. Python: the same requests rendered with jinja2 as transformers' apply_chat_template sets it up (tojson =
     json.dumps(ensure_ascii=False, ...), raise_exception, `generation` a no-op), from the messages llama.cpp hands the
     template (null content as "", role developer as system, tool-call arguments parsed with Python's json); every case
     where it differs from llama.cpp is listed by kind.
Exits 1 if eklow differs from llama.cpp anywhere (except a request with special-token text, whose ids differ on purpose).
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_fuzz  # noqa: E402

import jinja2  # noqa: E402
import jinja2.ext  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402


def run_eklow(tool, gguf, requests):
    data = "\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n"
    out = subprocess.run([tool, "render", "--tokens", gguf], input=data.encode(), capture_output=True, check=True).stdout
    return [json.loads(line) for line in out.decode().split("\n")[:len(requests)]]


def run_oracle(oracle, gguf, requests, path_prefix):
    cases = [{"id": str(i), "request": r} for i, r in enumerate(requests)]
    cpath, opath = path_prefix + ".cases.json", path_prefix + ".out.json"
    with open(cpath, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)
    env = dict(os.environ, LD_LIBRARY_PATH="/run/opengl-driver/lib")
    subprocess.run([oracle, "--model", gguf, "--cases", cpath, "--out", opath], check=True, capture_output=True, env=env)
    return json.load(open(opath, encoding="utf-8"))


def template_error(err):
    """The message a template's raise_exception gave, from llama.cpp's error text; None if the error is another."""
    m = re.search(r"Jinja Exception: (.*)", err, re.S)
    return m.group(1).strip() if m else None


def deliberate(request):
    """eklow ignores chat_template_kwargs other than reasoning_effort; llama.cpp lets "messages" or "tools" replace the request's."""
    kw = request.get("chat_template_kwargs")
    return isinstance(kw, dict) and ("messages" in kw or "tools" in kw)


def has_special_text(request):
    return "<|" in json.dumps(request, ensure_ascii=False)


# ---- Python's Jinja2 in Hugging Face's environment ------------------------------------------------------------------------

def make_python_template(path):
    src = open(path, encoding="utf-8").read()
    src = src.replace("{%- generation -%}", "{#- -#}").replace("{%- endgeneration -%}", "{#- -#}")

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(src)


def python_input(request):
    """The values llama.cpp hands the template, with tool-call arguments parsed by Python's json. None where the request
    is one llama-server refuses before the template (the case is not compared) or one this does not cover (a continued
    message)."""
    msgs = []
    if request.get("continue_final_message"):
        return None
    for m in request.get("messages", []):
        if not isinstance(m, dict) or "role" not in m or not isinstance(m["role"], str):
            return None
        j = {"role": "system" if m["role"] == "developer" else m["role"]}
        c = m.get("content")
        if isinstance(c, list):
            if any(not isinstance(p, dict) for p in c):
                return None
            j["content"] = [dict(p) for p in c] if c else ""
        elif c is None or c == "":
            j["content"] = ""
        elif isinstance(c, str):
            j["content"] = c
        else:
            return None
        for k in ("reasoning_content", "name", "tool_call_id"):
            if m.get(k):
                j[k] = m[k]
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                a = tc["function"]["arguments"]
                try:
                    a = json.loads(a) if isinstance(a, str) else a
                except ValueError:
                    return None
                c2 = {"type": "function", "function": {"name": tc["function"]["name"], "arguments": a}}
                if tc.get("id"):
                    c2["id"] = tc["id"]
                calls.append(c2)
            j["tool_calls"] = calls
        msgs.append(j)
    if msgs and msgs[-1]["role"] == "assistant":
        return None
    tools = None
    if request.get("tools"):
        tools = []
        for t in request["tools"]:
            f = t["function"]
            tools.append({"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                           "parameters": f.get("parameters") if f.get("parameters") is not None else {}}})
    ctx = {"messages": msgs, "add_generation_prompt": False}
    if tools:
        ctx["tools"] = tools
    kw = request.get("chat_template_kwargs")
    if isinstance(kw, dict) and "reasoning_effort" in kw:
        ctx["reasoning_effort"] = kw["reasoning_effort"]
    if isinstance(request.get("reasoning_effort"), str) and request["reasoning_effort"]:
        ctx["reasoning_effort"] = request["reasoning_effort"]
    return ctx


def llama_ints(v):
    """Python's value as llama.cpp reads it: an integer above INT64_MAX within UINT64 wraps, a larger one is a float."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if -2 ** 63 <= v < 2 ** 63:
            return v
        if 2 ** 63 <= v < 2 ** 64:
            return v - 2 ** 64
        return float(v)
    if isinstance(v, list):
        return [llama_ints(x) for x in v]
    if isinstance(v, dict):
        return {k: llama_ints(x) for k, x in v.items()}
    return v


EFFORT = re.compile(r"Thinking effort level: [^<]*")


def classify(tmpl, request, llama_prompt, py_prompt):
    """Why Python's prompt differs from llama.cpp's, when it is one of the known reasons (a list of them)."""
    reasons = []
    ctx = python_input(request)
    if ctx is not None:
        ctx2 = dict(ctx, messages=llama_ints(ctx["messages"]))
        if "tools" in ctx2:
            ctx2["tools"] = llama_ints(ctx2["tools"])
        try:
            py2 = tmpl.render(**ctx2)
        except jinja2.exceptions.TemplateError:
            py2 = None
        if py2 is not None and py2 != py_prompt:
            reasons.append("an integer beyond 64 bits in a tool call or schema (Python keeps its digits; llama.cpp wraps one up to 2^64 and makes a larger one a float)")
            py_prompt = py2
    e1, e2 = EFFORT.search(py_prompt), EFFORT.search(llama_prompt)
    if e1 and e2 and e1.group(0) != e2.group(0):
        reasons.append("the effort number is written by Python as str(float), by llama.cpp as %f without trailing zeros (0.123456789 is 0.123457, 1e-05 is 0.00001, 5e-324 is 0.0)")
    if reasons and EFFORT.sub("", py_prompt) == EFFORT.sub("", llama_prompt):
        return " + ".join(reasons)
    return None


def python_render(tmpl, request):
    ctx = python_input(request)
    if ctx is None:
        return None
    try:
        return {"prompt": tmpl.render(**ctx)}
    except jinja2.exceptions.TemplateError as e:
        return {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__ + ": " + str(e)}


# ---- comparison -----------------------------------------------------------------------------------------------------------

def compare_eklow(requests, llama, eklow, tok_decode=None, label=""):
    """Returns (differences, special_cases). A difference: prompt, error presence or text, generation prompt, ids."""
    bad, special = [], 0
    for i, (req, ll, ek) in enumerate(zip(requests, llama, eklow)):
        why = None
        if deliberate(req):
            continue
        if "error" in ll:
            if "error" not in ek:
                why = "llama.cpp raised %r, eklow rendered" % ll["error"][-80:]
            else:
                tmsg = template_error(ll["error"])
                if tmsg is not None and tmsg not in ek["error"]:
                    why = "error text: llama %r eklow %r" % (tmsg, ek["error"])
        elif "error" in ek:
            why = "eklow raised %r, llama.cpp rendered" % ek["error"]
        elif ll["prompt"] != ek["prompt"]:
            k = next((j for j, (a, b) in enumerate(zip(ll["prompt"], ek["prompt"])) if a != b), min(len(ll["prompt"]), len(ek["prompt"])))
            why = "prompt differs at %d: llama %r eklow %r" % (k, ll["prompt"][max(0, k - 30):k + 40], ek["prompt"][max(0, k - 30):k + 40])
        elif ll.get("generation_prompt", "") != ek["generation_prompt"]:
            why = "generation_prompt: llama %r eklow %r" % (ll.get("generation_prompt"), ek["generation_prompt"])
        elif has_special_text(req):
            special += 1  # ids differ on purpose; the ids must still decode to the prompt (checked by the caller)
        elif ll["prompt_tokens"] != ek["ids"]:
            why = "ids differ (llama %d tokens, eklow %d)" % (len(ll["prompt_tokens"]), len(ek["ids"]))
        if why:
            bad.append((i, why))
    return bad, special


def special_text_gate(a, n=300):
    """Requests whose text holds special-token strings: llama.cpp reads them as the tokens, eklow as text. For a single user
    message the expected ids are llama.cpp's own: the prompt up to the message's text block, llama.cpp's tokens of the text
    with special tokens not parsed (--tokenize "plain"), and the block's end."""
    import random
    rng = random.Random(5)
    names = [t for t in chat_fuzz.SPECIALS] + ["<|unused_200007|>", "<|begin_of_text|>", "<|content_image|>", "<|not_special|>", "<|", "|>"]
    texts = []
    for _ in range(n):
        texts.append("".join(rng.choice(names + chat_fuzz.WORDS + ["é", "ü", "ß", "—"] + [" ", "\n", "  "]) for _ in range(rng.randint(1, 12))))
    reqs = [{"messages": [{"role": "user", "content": t}]} for t in texts]
    base = [{"messages": [{"role": "user", "content": ""}]}]
    ll = run_oracle(a.oracle, a.gguf, base, os.path.join(a.work, "special-base"))[0]["prompt_tokens"]
    prefix = ll[:-1]  # up to <|message_user|><|content_text|>; the last token is <|end_message|>
    spath = os.path.join(a.work, "special-strings.json")
    json.dump(texts, open(spath, "w", encoding="utf-8"), ensure_ascii=False)
    opath = os.path.join(a.work, "special-tokens.json")
    subprocess.run([a.oracle, "--model", a.gguf, "--tokenize", spath, "--out", opath], check=True, capture_output=True,
                   env=dict(os.environ, LD_LIBRARY_PATH="/run/opengl-driver/lib"))
    # (the oracle writes each token's piece too, and fails on a piece that is half a character: the texts avoid such tokens)
    plain = json.load(open(opath, encoding="utf-8"))
    ek = run_eklow(a.eklow_chat, a.gguf, reqs)
    bad = sum(1 for p, e in zip(plain, ek) if e["ids"] != prefix + p["plain"] + [ll[-1]] or not e["roundtrip"])
    print("special-token text in a user message: %d requests, eklow's ids = the prompt around llama.cpp's plain tokens of the text in all but %d" % (n, bad))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("gguf")
    ap.add_argument("m0")
    ap.add_argument("template")
    ap.add_argument("--oracle")
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--work", default=".")
    a = ap.parse_args()
    failed = False

    # 1. M0
    cases = json.load(open(os.path.join(a.m0, "cases.json"), encoding="utf-8"))
    ref = json.load(open(os.path.join(a.m0, "chat.json"), encoding="utf-8"))
    reqs = [c["request"] for c in cases]
    ek = run_eklow(a.eklow_chat, a.gguf, reqs)
    bad, special = compare_eklow(reqs, ref, ek)
    n_err = sum("error" in r for r in ref)
    print("M0 cases: %d requests (%d raise in llama.cpp, %d carry special-token text), %d differences" % (len(reqs), n_err, special, len(bad)))
    for i, why in bad[:10]:
        print("  %s: %s" % (cases[i]["id"], why))
    failed |= bool(bad)

    # 2. fuzz
    if a.fuzz:
        if not a.oracle:
            sys.exit("--fuzz needs --oracle")
        os.makedirs(a.work, exist_ok=True)
        reqs = chat_fuzz.requests(a.fuzz, a.seed)
        n_random = len(reqs)
        reqs += chat_fuzz.edge_requests()
        ll = run_oracle(a.oracle, a.gguf, reqs, os.path.join(a.work, "fuzz-%d" % a.seed))
        ek = run_eklow(a.eklow_chat, a.gguf, reqs)
        bad, special = compare_eklow(reqs, ll, ek)
        nrt = sum(1 for r in ek if "error" not in r and not r["roundtrip"])
        if nrt:
            print("  %d prompts whose ids do not decode back to the prompt" % nrt)
        failed |= bool(nrt)
        failed |= bool(special_text_gate(a))
        kinds = collections.Counter()
        for r in ll:
            kinds["rendered" if "error" not in r else ("template raised" if template_error(r["error"]) is not None else "refused before the template")] += 1
        print("fuzz seed %d: %d random requests and %d hand-written edge cases (%s; %d carry special-token text), %d differences" % (
            a.seed, n_random, len(reqs) - n_random, ", ".join("%d %s" % (v, k) for k, v in sorted(kinds.items())), special, len(bad)))
        for i, why in bad[:15]:
            print("  #%d: %s\n      %s" % (i, why, json.dumps(reqs[i], ensure_ascii=False)[:300]))
        failed |= bool(bad)

        # 3. Python
        tmpl = make_python_template(a.template)
        same, diff, skipped = 0, [], 0
        for i, (req, r) in enumerate(zip(reqs, ll)):
            if deliberate(req):
                skipped += 1
                continue
            if "error" in r and template_error(r["error"]) is None and "Unknown (built-in) filter" not in r["error"]:
                skipped += 1
                continue
            py = python_render(tmpl, req)
            if py is None:
                skipped += 1
                continue
            if "error" in r:
                lm = template_error(r["error"]) or r["error"][-100:]
                if "error" in py and (lm in py["error"] or py["error"] in lm):
                    same += 1
                else:
                    diff.append((i, "llama.cpp raised %r; python %s" % (lm, "raised %r" % py["error"] if "error" in py else "rendered")))
            elif "error" in py:
                diff.append((i, "python raised %r; llama.cpp rendered" % py["error"]))
            elif py["prompt"] != r["prompt"]:
                k = next((j for j, (x, y) in enumerate(zip(py["prompt"], r["prompt"])) if x != y), 0)
                why = classify(tmpl, req, r["prompt"], py["prompt"])
                diff.append((i, why or "UNEXPLAINED prompt differs at %d: llama %r python %r" % (k, r["prompt"][max(0, k - 25):k + 45], py["prompt"][max(0, k - 25):k + 45])))
            else:
                same += 1
        print("python (HF's Jinja2 environment) against llama.cpp: %d compared, %d identical, %d different (%d not covered: refused before the template, a continued message)" % (
            same + len(diff), same, len(diff), skipped))
        by_kind = collections.defaultdict(list)
        for i, why in diff:
            kind = re.sub(r"\(.*\)", "", why) if why.startswith("the effort") else why[:150]
            if why.startswith("llama.cpp raised 'Unknown reasoning_effort"):
                kind = "Unknown reasoning_effort where Python's trim strips more than ASCII white space (U+00A0)"
            if "Unknown (built-in) filter 'float'" in why:
                kind = "reasoning_effort is a list or object: llama.cpp has no float filter for it, Python's returns 0.0"
            by_kind[kind].append((i, why))
        for kind, items in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
            print("  %d x %s\n      e.g. #%d: %s" % (len(items), kind, items[0][0], items[0][1][:260]))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
