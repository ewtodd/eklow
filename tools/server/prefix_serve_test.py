#!/usr/bin/env python3
"""Card checks of the disk prefix cache through the server (DECISIONS 2026-11-22). Standard library only.

  prefix_serve_test.py --url URL --docs DIR rewind     a slot that changes conversation past a shared stretch: B fresh, then A
                                                       (B's first ~3k tokens, then more), then B again; greedy, plain decode
  prefix_serve_test.py --url URL --docs DIR cold       references on a server without the cache: A's second turn cold (text,
                                                       time to first token, the draft's acceptance), a 32k prefill, decode at 8k
  prefix_serve_test.py --url URL --docs DIR cache      the same on a server with --prefix-cache, A's second turn after two other
                                                       conversations took both slots; then prefill-only requests: before a
                                                       turn, beside a request, and given up to one
  prefix_serve_test.py --url URL --docs DIR overhead   the 32k prefill and decode at 8k alone (for a server with or without it)

DIR holds the documents (qwen-32768.txt, qwen-61440.txt). Every line printed is "name: value" so that runs line up.
"""
import argparse
import json
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--docs", required=True)
ap.add_argument("what", choices=["rewind", "cold", "cache", "overhead"])
a = ap.parse_args()
URL = a.url.rstrip("/") + "/v1/chat/completions"
doc_a = open(f"{a.docs}/qwen-32768.txt").read()
doc_b = open(f"{a.docs}/qwen-61440.txt").read()
CH = 4  # characters a token, roughly, in these documents


def post(body, timeout=3600):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def greedy(messages, max_tokens=96, speculative=False, **kw):
    b = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "reasoning_effort": "none", "speculative": speculative}
    b.update(kw)
    return b


def stream(body):
    """Streams a request: (text, time to the first text, total time, usage)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text, usage = "", None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            j = json.loads(line[6:])
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece and first is None:
                    first = time.time() - t0
                text += d.get("content") or ""
    return text, first, time.time() - t0, usage


def say(name, value):
    print(f"{name}: {value}", flush=True)


def common(x, y):
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    return n


def user(text):
    return [{"role": "user", "content": text}]


def rewind():
    for v in range(3):
        shared = doc_a[v * 40000 : v * 40000 + 3000 * CH]
        A = user(doc_a[v * 40000 : v * 40000 + 8192 * CH] + "\n\nSummarize the code above in one sentence.")
        B = user(shared + doc_b[v * 60000 : v * 60000 + 3000 * CH] + "\n\nWhat does the last function above do? One sentence.")
        fresh = post(greedy(B))
        post(greedy(A))
        again = post(greedy(B))
        f, g = fresh["choices"][0]["message"]["content"], again["choices"][0]["message"]["content"]
        say(f"rewind {v}: prompt / cached (fresh, again)", f"{fresh['usage']['prompt_tokens']} / {fresh['usage']['prompt_tokens_details']['cached_tokens']}, "
            f"{again['usage']['prompt_tokens_details']['cached_tokens']}")
        say(f"rewind {v}: same text", f == g)
        say(f"rewind {v}: common prefix (chars of fresh)", f"{common(f, g)} of {len(f)}")
        if f != g:
            say(f"rewind {v}: fresh", repr(f[:160]))
            say(f"rewind {v}: again", repr(g[:160]))


# A's conversation: a 20k-token document, a first question, a fixed answer, a second question (the prompt of its second turn
# extends the first one's).
DOC = doc_b[:20000 * CH]
Q1 = "\n\nWhat language is most of the code above in? Answer in one line."
ANS1 = "Most of it is JavaScript, with some Python and shell scripts mixed in."
Q2 = "List three function names that appear above, one per line, nothing else."
ANS2 = "init\nrun\nmain"
Q3 = "Which of those three is defined last? One word."
A1 = user(DOC + Q1)
A2 = A1 + [{"role": "assistant", "content": ANS1}, {"role": "user", "content": Q2}]
A3 = A2 + [{"role": "assistant", "content": ANS2}, {"role": "user", "content": Q3}]


def others():
    """Two other conversations at once, so that both slots go to them."""
    out = {}

    def run(k, text):
        out[k] = post(greedy(user(text + "\n\nOne sentence on the above."), max_tokens=16))

    ts = [threading.Thread(target=run, args=(k, doc_a[off : off + 8000 * CH])) for k, off in (("o1", 0), ("o2", 60000))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def second_turn(label):
    text, ttft, total, u = stream(greedy(A2, max_tokens=64))
    d = u["prompt_tokens_details"]
    say(f"{label}: prompt / cached / restored", f"{u['prompt_tokens']} / {d['cached_tokens']} / {d.get('restored_tokens', 0)}")
    say(f"{label}: ttft s", f"{ttft:.3f}")
    say(f"{label}: text", repr(text))
    return text


def spec_turn(label):
    j = post(greedy(A2, max_tokens=256, speculative=True))
    s = j["usage"].get("speculation", {})
    v, acc = s.get("verifies", 0), s.get("accepted", 0)
    say(f"{label}: draft tokens/verify", f"{(acc + v) / max(v, 1):.2f} ({acc} accepted in {v} verifies, {j['usage']['completion_tokens']} tokens)")
    return j


def overhead():
    # A cold 32k prefill (distinct text each run), and decode at 8k: plain and with the draft.
    t0 = time.time()
    tag = str(time.time())
    j = post(greedy(user(tag + doc_b[100000 : 100000 + 32000 * CH] + "\n\nOne word."), max_tokens=1))
    dt = time.time() - t0
    say("prefill 32k: tokens, s, tok/s", f"{j['usage']['prompt_tokens']}, {dt:.2f}, {j['usage']['prompt_tokens'] / dt:.0f}")
    base = user(tag + doc_a[:8000 * CH] + "\n\nRewrite the first function above in Python, with comments.")
    post(greedy(base, max_tokens=1))  # the prompt into a slot first: decode alone is timed below
    for spec in (False, True):
        text, ttft, total, u = stream(greedy(base, max_tokens=512, speculative=spec))
        n = u["completion_tokens"]
        say(f"decode 8k {'draft' if spec else 'plain'}: tokens, tok/s", f"{n}, {(n - 1) / max(total - ttft, 1e-6):.1f}")


def cold():
    t0 = time.time()
    post(greedy(A1, max_tokens=8))
    say("A1 cold: s", f"{time.time() - t0:.2f}")
    others()
    second_turn("A2 cold (after two others took the slots)")
    others()
    spec_turn("A2 cold, draft")
    overhead()


def cache():
    t0 = time.time()
    post(greedy(A1, max_tokens=8))
    say("A1: s", f"{time.time() - t0:.2f}")
    others()
    second_turn("A2 restored")
    others()
    spec_turn("A2 restored, draft")

    # A prefill-only request before a turn: A3's prompt goes in while nothing else runs, then A3 itself.
    others()
    t0 = time.time()
    w = post({"messages": A3, "max_tokens": 0, "reasoning_effort": "none"})
    say("prefill-only A3: s", f"{time.time() - t0:.2f}")
    say("prefill-only A3: finish / usage", f"{w['choices'][0]['finish_reason']} / {json.dumps(w['usage'])}")
    text, ttft, total, u = stream(greedy(A3, max_tokens=16))
    say("A3 after it: prompt / cached", f"{u['prompt_tokens']} / {u['prompt_tokens_details']['cached_tokens']}")
    say("A3 after it: ttft s", f"{ttft:.3f}")

    # Beside a request: a cold 30k prefill-only request, then a short request 1 s later.
    res = {}

    def warm(k, text):
        t = time.time()
        res[k] = post({"messages": user(text), "max_tokens": 0})
        res[k + "_s"] = time.time() - t

    big1 = doc_b[150000 : 150000 + 30000 * CH]
    th = threading.Thread(target=warm, args=("w1", big1))
    th.start()
    time.sleep(1.0)
    text, ttft, total, u = stream(greedy(user("Name a prime number above 100. Just the number."), max_tokens=8))
    say("a request during a prefill-only one: ttft s", f"{ttft:.3f} (prompt {u['prompt_tokens']}, cached {u['prompt_tokens_details']['cached_tokens']})")
    th.join()
    say("the prefill-only one: s / usage.prefill", f"{res['w1_s']:.2f} / {json.dumps(res['w1']['usage'].get('prefill'))}")

    # Given up: a cold 30k prefill-only request, a long request beside it, then another request finds no free slot.
    big2 = doc_a[20000 : 20000 + 30000 * CH]
    th = threading.Thread(target=warm, args=("w2", big2))
    th.start()
    time.sleep(1.0)
    long_res = {}

    def long_req():
        long_res["r"] = stream(greedy(user("Write a 300-word story about a lighthouse."), max_tokens=400))

    tl = threading.Thread(target=long_req)
    tl.start()
    time.sleep(1.0)
    text, ttft, total, u = stream(greedy(user("What is 17 times 23? Just the number."), max_tokens=8))
    say("a request with both slots busy: ttft s", f"{ttft:.3f}")
    th.join()
    tl.join()
    say("the prefill-only one given up: s / usage.prefill", f"{res['w2_s']:.2f} / {json.dumps(res['w2']['usage'].get('prefill'))} (prompt {res['w2']['usage']['prompt_tokens']})")
    say("the long request beside it: tokens", long_res["r"][3]["completion_tokens"])
    overhead()


{"rewind": rewind, "cold": cold, "cache": cache, "overhead": overhead}[a.what]()
