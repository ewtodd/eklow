#!/usr/bin/env python3
"""Tests of eklow-server through its HTTP API (stdlib only), on Qwen3.6 or on Gemma 4 E4B (Nemotron-3.5-Bolt-E4B; one binary
serves both).
Start the server first (srv.sh in ~/eklow-qwen-data/qserve on fiona, or by hand), then:

  serve_test.py --url http://127.0.0.1:8091 [--family auto|qwen|gemma] basic [--save FILE]
                                                                   whole and streamed chat, reasoning_content, a tool call
                                                                   (the grammar), stop strings, seeds, two at once, a
                                                                   cancelled request, prefix reuse; with --save the
                                                                   temperature-0 outputs go to FILE
  serve_test.py compare A.json B.json                              the saved outputs of two servers (with and without
                                                                   the draft) against each other
  serve_test.py --url ... sample --n 1500 --out FILE               N short completions at temperature 1 of one prompt
  serve_test.py chi2 A.json B.json                                 two-sample chi-square of the sampled completions
                                                                   (the acceptance's test through the server path)

--family says which server it is talking to; auto (the default) reads /v1/models and takes "gemma" or "bolt" in the model
id for Gemma 4, anything else for Qwen. Qwen's checks are unchanged. Gemma 4's tool-call grammar fixes the call's structure
and the tool's name but not the parameters' types (Qwen's does, from the schema), so for Gemma the typed-arguments
check is reported without failing the run.
"""
import argparse
import concurrent.futures
import http.client
import json
import math
import sys
import threading
import time
import urllib.parse


def post(url, body, stream=False, timeout=3600):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=stream)), {"content-type": "application/json"})
    r = c.getresponse()
    if not stream:
        data = r.read()
        c.close()
        return r.status, json.loads(data)
    return r.status, (c, r)


def chat(url, messages, **kw):
    status, j = post(url, dict(messages=messages, **kw))
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {j}")
    return j


def stream_chat(url, messages, **kw):
    """Streamed request; returns (reasoning, content, tool_calls, finish_reason, usage, chunks)."""
    status, (c, r) = post(url, dict(messages=messages, stream_options={"include_usage": True}, **kw), stream=True)
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {r.read()[:300]!r}")
    reasoning = content = ""
    calls, finish, usage, chunks = [], None, None, 0
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        chunks += 1
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            calls += d.get("tool_calls") or []
            finish = ch.get("finish_reason") or finish
    c.close()
    return reasoning, content, calls, finish, usage, chunks


FAILED = []
FAMILY = "qwen"


def detect_family(url, wanted):
    """qwen or gemma: as asked, or from the model id /v1/models gives."""
    if wanted != "auto":
        return wanted
    u = urllib.parse.urlparse(url)
    try:
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        c.request("GET", "/v1/models")
        ids = [m.get("id", "") for m in json.loads(c.getresponse().read()).get("data", [])]
        c.close()
    except Exception as e:
        print(f"(/v1/models unreadable: {e}; assuming qwen)")
        return "qwen"
    return "gemma" if any("gemma" in i.lower() or "bolt" in i.lower() for i in ids) else "qwen"


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail and not ok else (f"  {detail}" if detail else "")), flush=True)
    if not ok:
        FAILED.append(name)


WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "units": {"type": "string", "enum": ["metric", "imperial"]}},
    "required": ["city"]}}}

# Prompts whose temperature-0 outputs are compared between servers.
GREEDY = [
    [{"role": "user", "content": "Write a haiku about the sea, then explain it in one sentence."}],
    [{"role": "user", "content": "List the first twelve prime numbers, then their sum."}],
    [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is the capital of Australia and why is it not Sydney?"}],
    [{"role": "user", "content": "Write a Python function that merges two sorted lists, with a short docstring."}],
]


def basic(a):
    global FAMILY
    url = a.url
    FAMILY = detect_family(url, a.family)
    print(f"server family: {FAMILY}")
    saved = {}
    # Whole and streamed give the same text at temperature 0; reasoning_content is separate from content.
    msgs = [{"role": "user", "content": "In one sentence, why is the sky blue?"}]
    j = chat(url, msgs, max_tokens=700, temperature=0)
    m = j["choices"][0]["message"]
    check("whole: finish_reason stop", j["choices"][0]["finish_reason"] == "stop", j["choices"][0]["finish_reason"])
    check("whole: reasoning_content and content", bool(m.get("reasoning_content")) and bool(m.get("content")), str(m)[:200])
    check("whole: no think tags in content", all(t not in m["content"] for t in ("<think>", "</think>", "<|channel>", "<channel|>")))
    u = j["usage"]
    check("whole: usage adds up", u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"] and "cached_tokens" in u["prompt_tokens_details"])
    r, c, calls, fin, usage, chunks = stream_chat(url, msgs, max_tokens=700, temperature=0)
    check("stream: same reasoning and content as whole", r == m.get("reasoning_content") and c == m["content"], f"{len(r)} vs {len(m.get('reasoning_content', ''))}")
    check("stream: finish and usage", fin == "stop" and usage and usage["completion_tokens"] == u["completion_tokens"], str((fin, usage)))
    check("stream: many chunks", chunks > 20, str(chunks))
    # Thinking off.
    j = chat(url, msgs, max_tokens=200, temperature=0, reasoning_effort="none")
    m2 = j["choices"][0]["message"]
    check("reasoning_effort none: no reasoning", not m2.get("reasoning_content") and bool(m2["content"]), str(m2)[:200])
    # Tool call through the grammar, whole and streamed.
    tm = [{"role": "user", "content": "What is the weather in Lisbon for the next 3 days, in metric units?"}]
    j = chat(url, tm, tools=[WEATHER], max_tokens=1500, temperature=0)
    ch = j["choices"][0]
    calls = ch["message"].get("tool_calls") or []
    ok = len(calls) >= 1 and calls[0]["function"]["name"] == "get_weather"
    args = {}
    if ok:
        try:
            args = json.loads(calls[0]["function"]["arguments"])
        except Exception as e:
            ok = False
    check("tool call: get_weather with JSON arguments", ok and args.get("city") == "Lisbon", str(calls)[:300])
    typed = isinstance(args.get("days"), int) and args.get("units") in ("metric", "imperial")
    if FAMILY == "gemma":
        print(f"{'ok  ' if typed else 'note'} tool call: typed arguments (Gemma's grammar does not enforce the schema)  {args}")
    else:
        check("tool call: typed arguments", typed, str(args))
    check("tool call: finish_reason tool_calls", ch["finish_reason"] == "tool_calls", ch["finish_reason"])
    r, c, scalls, fin, usage, _ = stream_chat(url, tm, tools=[WEATHER], max_tokens=1500, temperature=0)
    check("tool call streamed: same call", len(scalls) == len(calls) and scalls[0]["function"]["arguments"] == calls[0]["function"]["arguments"] and fin == "tool_calls", str(scalls)[:200])
    # The tool's answer back (a second turn with tool_calls and a tool message).
    if ok:
        conv = tm + [{"role": "assistant", "content": None, "tool_calls": calls}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "Sunny, 24 C"}]
        j = chat(url, conv, tools=[WEATHER], max_tokens=1500, temperature=0, reasoning_effort="none")
        check("tool result turn answers in content", bool(j["choices"][0]["message"].get("content")), str(j["choices"][0])[:200])
    # Stop strings: content ends before the stop string, finish_reason stop. The stop word is one from the model's own
    # unstopped answer (a model may refuse to echo a fixed phrase), and the stopped output is that answer's prefix.
    sq = [{"role": "user", "content": "Write the numbers from one to twenty as words, separated by spaces, and nothing else."}]
    t0 = chat(url, sq, max_tokens=100, temperature=0, reasoning_effort="none")["choices"][0]["message"]["content"]
    words = t0.split()
    stopw = words[len(words) // 2] if len(words) > 4 else "x"
    j = chat(url, sq, max_tokens=100, temperature=0, reasoning_effort="none", stop=[stopw])
    t = j["choices"][0]["message"]["content"]
    check("stop string cuts the content", stopw not in t and t0.startswith(t.rstrip()) and len(t.strip()) > 3 and j["choices"][0]["finish_reason"] == "stop", repr((t, stopw, t0)))
    r, c, _, fin, _, _ = stream_chat(url, sq, max_tokens=100, temperature=0, reasoning_effort="none", stop=[stopw, "zzz"])
    check("stop string streamed", stopw not in c and t0.startswith(c.rstrip()) and len(c.strip()) > 3 and fin == "stop", repr(c))
    # max_tokens.
    j = chat(url, msgs, max_tokens=7, temperature=0)
    check("max_tokens ends the request exactly", j["usage"]["completion_tokens"] == 7 and j["choices"][0]["finish_reason"] == "length", str(j["usage"]))
    # Seeds: the same seed gives the same text at temperature 1.
    sm = [{"role": "user", "content": "Invent a name for a lighthouse and describe it in a sentence."}]
    x = chat(url, sm, max_tokens=60, temperature=1.0, seed=42, reasoning_effort="none")["choices"][0]["message"]["content"]
    y = chat(url, sm, max_tokens=60, temperature=1.0, seed=42, reasoning_effort="none")["choices"][0]["message"]["content"]
    z = chat(url, sm, max_tokens=60, temperature=1.0, seed=43, reasoning_effort="none")["choices"][0]["message"]["content"]
    check("seed: same seed, same output", x == y, f"{x!r} vs {y!r}")
    check("seed: another seed, another output", x != z)
    # A broken image part is refused with a clear error: no projector (--mmproj), or bytes that are no picture.
    status, j = post(url, {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})
    check("bad image part refused", status == 400 and ("not supported" in json.dumps(j) or "not a JPEG" in json.dumps(j)), f"{status} {j}")
    # A developer message is a system one; both give the same prompt.
    d1 = chat(url, [{"role": "developer", "content": "Answer in French."}, {"role": "user", "content": "Say hello."}], max_tokens=1, temperature=0)["usage"]["prompt_tokens"]
    d2 = chat(url, [{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Say hello."}], max_tokens=1, temperature=0)["usage"]["prompt_tokens"]
    check("developer role is system", d1 == d2, f"{d1} {d2}")
    # Two requests at once give what they give alone (temperature 0, near-ties aside: compared by prefix).
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(chat, url, GREEDY[0], max_tokens=300, temperature=0)
        f2 = ex.submit(chat, url, GREEDY[1], max_tokens=300, temperature=0)
        both = [f1.result(), f2.result()]
    solo = [chat(url, GREEDY[0], max_tokens=300, temperature=0), chat(url, GREEDY[1], max_tokens=300, temperature=0)]
    for k in range(2):
        a_, b_ = both[k]["choices"][0]["message"], solo[k]["choices"][0]["message"]
        ta = (a_.get("reasoning_content") or "") + (a_.get("content") or "")
        tb = (b_.get("reasoning_content") or "") + (b_.get("content") or "")
        n = next((i for i in range(min(len(ta), len(tb))) if ta[i] != tb[i]), min(len(ta), len(tb)))
        check(f"two at once: request {k} complete and coherent", len(ta) > 200 and both[k]["choices"][0]["finish_reason"] in ("length", "stop"), "")
        print(f"     (equal to its solo run for {n} of {len(tb)} characters: the int8 rounding depends on the step's shape, so near-ties go the other way)")
    # A cancelled stream frees its slot: three requests in a row on two slots all complete.
    status, (c, r) = post(url, {"messages": GREEDY[3], "max_tokens": 3000, "temperature": 1.0}, stream=True)
    got = 0
    for raw in r:
        got += 1
        if got > 30:
            break
    c.close()
    time.sleep(1.0)
    t0 = time.time()
    j = chat(url, [{"role": "user", "content": "Say hi."}], max_tokens=5, temperature=0, reasoning_effort="none")
    check("cancel: the next request is served promptly", time.time() - t0 < 20, f"{time.time() - t0:.1f} s")
    # Prefix reuse: the second turn of a conversation caches the first turn's prompt.
    conv = [{"role": "user", "content": "Tell me a fact about octopuses. " * 200}]
    j1 = chat(url, conv, max_tokens=60, temperature=0, reasoning_effort="none")
    conv2 = conv + [j1["choices"][0]["message"] | {"role": "assistant"}] + [{"role": "user", "content": "Another one."}]
    conv2[1].pop("reasoning_content", None)
    j2 = chat(url, conv2, max_tokens=30, temperature=0, reasoning_effort="none")
    c2 = j2["usage"]["prompt_tokens_details"]["cached_tokens"]
    check("prefix reuse: the next turn is mostly cached", c2 >= j1["usage"]["prompt_tokens"] - 10, f"cached {c2} of {j2['usage']['prompt_tokens']}, first prompt {j1['usage']['prompt_tokens']}")
    j3 = chat(url, conv, max_tokens=5, temperature=0, reasoning_effort="none")
    check("prefix reuse: the same prompt again is cached", j3["usage"]["prompt_tokens_details"]["cached_tokens"] >= j1["usage"]["prompt_tokens"] - 10, str(j3["usage"]))
    # A document and then a question, then another question: the paragraph break before the question is a snapshot point.
    doc = "\n\n".join(f"Paragraph {i}: the harbour at dusk, number {i * 7}, smells of tar and salt and old rope." for i in range(120))
    d1 = chat(url, [{"role": "user", "content": doc + "\n\nHow many paragraphs mention rope?"}], max_tokens=20, temperature=0, reasoning_effort="none")
    d2 = chat(url, [{"role": "user", "content": doc + "\n\nWhich paragraph is the last one?"}], max_tokens=20, temperature=0, reasoning_effort="none")
    check("document then another question: the document is cached", d2["usage"]["prompt_tokens_details"]["cached_tokens"] >= d1["usage"]["prompt_tokens"] - 40,
          f"cached {d2['usage']['prompt_tokens_details']['cached_tokens']} of {d2['usage']['prompt_tokens']}")
    if a.save:
        for i, m in enumerate(GREEDY):
            j = chat(url, m, max_tokens=400, temperature=0)
            saved[str(i)] = j["choices"][0]["message"]
        json.dump(saved, open(a.save, "w"))
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all ok")
    return 1 if FAILED else 0


def compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    bad = 0
    for k in x:
        tx = (x[k].get("reasoning_content") or "") + "\n=====\n" + (x[k].get("content") or "")
        ty = (y[k].get("reasoning_content") or "") + "\n=====\n" + (y[k].get("content") or "")
        n = next((i for i in range(min(len(tx), len(ty))) if tx[i] != ty[i]), min(len(tx), len(ty)))
        same = tx == ty
        bad += not same
        print(f"prompt {k}: {'identical' if same else f'differ after {n} of {len(tx)} / {len(ty)} characters'}")
    return 1 if bad else 0


def sample(a):
    """N completions of a short, high-entropy prompt at temperature 1 (thinking off): the first A.tokens tokens by
    logprob-free means, i.e. the text itself; the distribution of (first words) is compared between servers. No
    truncation unless asked (min_p 0 over the server's default): a hard cut-off meets the engine's step-shape noise (the int8
    activations), so a token near it is kept in one batch shape and dropped in another, and a draft changes the shapes."""
    msgs = [{"role": "user", "content": a.prompt}]
    out = []
    lock = threading.Lock()

    def one(i):
        j = chat(a.url, msgs, max_tokens=a.tokens, temperature=1.0, top_p=1.0, top_k=0, min_p=a.min_p, reasoning_effort="none", seed=1000 + i)
        t = j["choices"][0]["message"]["content"]
        with lock:
            out.append(t)

    with concurrent.futures.ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, range(a.n)))
    json.dump(out, open(a.out, "w"))
    print(f"{len(out)} samples, {len(set(out))} distinct")


def chi2(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    cx, cy = {}, {}
    for t in x:
        cx[t] = cx.get(t, 0) + 1
    for t in y:
        cy[t] = cy.get(t, 0) + 1
    # Categories with fewer than 10 samples in the pool are merged into one.
    keys = sorted(set(cx) | set(cy), key=lambda k: -(cx.get(k, 0) + cy.get(k, 0)))
    big = [k for k in keys if cx.get(k, 0) + cy.get(k, 0) >= 10]
    rest = [k for k in keys if k not in big]
    rows = [(cx.get(k, 0), cy.get(k, 0)) for k in big] + ([(sum(cx.get(k, 0) for k in rest), sum(cy.get(k, 0) for k in rest))] if rest else [])
    nx, ny = len(x), len(y)
    stat = 0.0
    for p, q in rows:
        if p + q == 0:
            continue
        e = (p + q) / (nx + ny)
        stat += (p - nx * e) ** 2 / (nx * e) + (q - ny * e) ** 2 / (ny * e)
    dof = len(rows) - 1
    # Wilson-Hilferty approximation of the chi-square tail.
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    print(f"chi-square {stat:.1f} on {dof} degrees of freedom over {len(rows)} categories ({len(big)} common), p = {p:.3f}; {nx} and {ny} samples")
    for (pa, pb), k in list(zip(rows, big))[:8]:
        print(f"  {pa:5d} {pb:5d}  {k!r}")
    return 0 if p > 0.001 else 1


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8091")
ap.add_argument("--family", choices=["auto", "qwen", "gemma"], default="auto")
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("basic")
p.add_argument("--save")
p = sub.add_parser("compare")
p.add_argument("a")
p.add_argument("b")
p = sub.add_parser("sample")
p.add_argument("--n", type=int, default=1500)
p.add_argument("--conc", type=int, default=2)
p.add_argument("--tokens", type=int, default=4)
p.add_argument("--prompt", default="Name a random animal, one word.")
p.add_argument("--min-p", type=float, default=0.0)
p.add_argument("--out", required=True)
p = sub.add_parser("chi2")
p.add_argument("a")
p.add_argument("b")
a = ap.parse_args()
sys.exit({"basic": basic, "compare": compare, "sample": sample, "chi2": chi2}[a.cmd](a))
