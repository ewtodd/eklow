#!/usr/bin/env python3
"""Prefill at depth, one-stream decode, concurrent throughput and prefix reuse through eklow-server, measured the way
unholy's tools/box/perf-depth.sh and eb.sh measure, so the two READMEs compare.

  serve_perf.py --url http://127.0.0.1:8081 --model ALIAS depth --docs DIR [--depths 8192,32768,...] [--repeats 5]
  serve_perf.py --url http://127.0.0.1:8081 --model ALIAS gsm --requests gsm48.jsonl [--concurrency 1,2]
  serve_perf.py --url http://127.0.0.1:8081 --model ALIAS reuse --docs DIR [--doc-depth 100000]

depth: for each DIR/qwen-DEPTH.txt (unique Pile text, unholy's documents cut to DEPTH Qwen tokens by cut_docs.py), one
request on it with nothing cached (its time to first token gives the prefill rate), then REPEATS - 1 more with other
questions, the document coming from the prefix cache; max_tokens 256, temperature 1.0, top_p 0.95, seed 7, thinking on
(eklow's default). Decode is each request's tokens after the first over the time after the first, from the stream (the
server's per-request log line has the same); the summary takes the median of the REPEATS, and with a draft the mean
tokens a verify block gave (accepted drafted tokens + 1), from the stream's usage.speculation.
gsm: unholy's 48 GSM8K 5-shot chat requests (thinking on, temperature 1.0, top_p 0.95, top_k 20, up to 1024 tokens),
CONC at a time; aggregate tok/s = completion tokens / wall time.
reuse: prefix reuse over the recurrent state. (1) a conversation of about 10.5k tokens (the size of son-of-anton's
prompts: a system prompt with tools' worth of text and a first question) cold, then its next turn (the answer as the
client sends it back, content only, and a new question) and a third; (2) a DOC_DEPTH-token document with one question,
then another question on the same document. Run it against servers with each template (stock, ornith) for the
conversation. Prints time to first token, prompt and cached tokens.
Stdlib only.
"""
import argparse
import concurrent.futures
import http.client
import json
import statistics
import time
import urllib.parse

QUESTIONS = ["Summarize the text above in three sentences.", "List five named entities that appear above.",
             "What is the main topic of the last part above?", "Write one question the text above answers.",
             "Give the text above a short title."]


def stream(url, body, want_text=False):
    """POSTs a streamed chat request; returns (prompt tokens, cached tokens, completion tokens, TTFT s, decode tok/s,
    tokens per verify or None[, (reasoning, content)])."""
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
    body = dict(body, stream=True, stream_options={"include_usage": True})
    t0 = time.monotonic()
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    first = last = None
    usage = {}
    reasoning = content = ""
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                now = time.monotonic()
                first = first or now
                last = now
    c.close()
    n = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ttft = (first or time.monotonic()) - t0
    decode = (n - 1) / (last - first) if n > 1 and last > first else 0.0
    sp = usage.get("speculation")
    per = (sp["accepted"] + sp["verifies"]) / sp["verifies"] if sp and sp["verifies"] else None
    out = (usage.get("prompt_tokens", 0), cached, n, ttft, decode, per)
    return out + ((reasoning, content),) if want_text else out


def depth(a):
    rows = []
    for d in [int(x) for x in a.depths.split(",")]:
        text = open(f"{a.docs}/qwen-{d}.txt").read()
        decs, pers = [], []
        for i in range(a.repeats):
            body = {"model": a.model, "messages": [{"role": "user", "content": text + "\n\n" + QUESTIONS[i % len(QUESTIONS)]}],
                    "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "seed": 7}
            if a.block:
                body["speculative"] = a.block
            p, cached, n, ttft, dec, per = stream(a.url, body)
            print(f"request | prompt {p} ({cached} cached), completion {n} | ttft {ttft:.2f} s, "
                  f"prefill {(p - cached) / ttft:.1f} tok/s, decode {dec:.1f} tok/s" + (f", {per:.2f} tokens/verify" if per else ""), flush=True)
            if i == 0:
                first = (p, ttft)
            decs.append(dec)
            if per:
                pers.append(per)
        plain = []
        for i in range(a.plain):  # the same document from the cache, decoded without the draft (the request's "speculative": false)
            body = {"model": a.model, "messages": [{"role": "user", "content": text + "\n\n" + QUESTIONS[(i + 1) % len(QUESTIONS)]}],
                    "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "seed": 7, "speculative": False}
            p, cached, n, ttft, dec, per = stream(a.url, body)
            print(f"plain request | prompt {p} ({cached} cached), completion {n} | ttft {ttft:.2f} s, decode {dec:.1f} tok/s", flush=True)
            plain.append(dec)
        rows.append((d, first[0] / first[1], first[1], statistics.median(decs), statistics.mean(pers) if pers else None, statistics.median(plain) if plain else None))
    print("\ndepth    prefill tok/s (TTFT)    decode tok/s, median of", a.repeats, "   tokens/verify")
    for d, rate, ttft, dec, per, plain in rows:
        print(f"{d // 1024}k{'':<6}{rate:7.0f} ({ttft:.1f} s){'':<8}{dec:.1f}" + (f"{'':<14}{per:.2f}" if per else "") + (f"{'':<8}plain {plain:.1f}" if plain else ""))


def gsm(a):
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()]
    for conc in [int(x) for x in a.concurrency.split(",")]:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda q: stream(a.url, dict(q, model=a.model)), reqs))
        wall = time.monotonic() - t0
        toks = sum(r[2] for r in res)
        pers = [r for r in res if r[5]]
        tail = f", {sum(r[2] for r in pers) / sum(r[2] / r[5] for r in pers):.2f} tokens/verify" if pers else ""
        print(f"gsm, {len(reqs)} requests, {conc} at a time: {toks} completion tokens in {wall:.1f} s, "
              f"{toks / wall:.1f} tok/s aggregate, decode per stream median {statistics.median(r[4] for r in res):.1f} tok/s{tail}",
              flush=True)


def reuse(a):
    docs = a.docs
    # (1) the conversation: about 10.5k tokens, a system prompt of text (unholy's doc-8192 cut) and a first question.
    system = open(f"{docs}/qwen-{a.conv_depth}.txt").read()
    msgs = [{"role": "system", "content": "You are an assistant working with the following reference material.\n\n" + system},
            {"role": "user", "content": "Summarize the reference material in two sentences."}]
    turns = ["Which person or organization is mentioned first?", "Now say the same in one short sentence."]
    print(f"conversation of about {a.conv_depth} tokens:")
    for t in range(3):
        body = {"model": a.model, "messages": msgs, "max_tokens": 200, "temperature": 1.0, "top_p": 0.95, "seed": 7}
        p, cached, n, ttft, dec, per, (reasoning, content) = stream(a.url, body, want_text=True)
        print(f"  turn {t + 1}: prompt {p}, cached {cached}, completion {n}, ttft {ttft:.2f} s ({(p - cached) / max(ttft, 1e-9):.0f} tok/s over the uncached part)", flush=True)
        if t < 2:
            msgs = msgs + [{"role": "assistant", "content": content}, {"role": "user", "content": turns[t]}]
    # (2) a long document, two questions.
    doc = open(f"{docs}/qwen-{a.doc_depth}.txt").read()
    print(f"document of about {a.doc_depth} tokens:")
    for i, q in enumerate(QUESTIONS[:3]):
        body = {"model": a.model, "messages": [{"role": "user", "content": doc + "\n\n" + q}], "max_tokens": 64, "temperature": 0, "reasoning_effort": "none"}
        p, cached, n, ttft, dec, per = stream(a.url, body)
        print(f"  question {i + 1}: prompt {p}, cached {cached}, ttft {ttft:.2f} s ({(p - cached) / max(ttft, 1e-9):.0f} tok/s over the uncached part)", flush=True)


def accept(a):
    """z-lab's dflash benchmark prompts (GSM8K, HumanEval, MT-Bench first turn: the first N of Random(42)'s shuffle, as
    ~/eklow-qwen-data/results/dflash/ was measured with llama.cpp), temperature 1.0 / top-p 0.95 / top-k 20, thinking on, up to
    4096 tokens; per dataset the mean over requests of tokens a verify block gave."""
    prompts = json.load(open(a.prompts))
    for name in a.datasets.split(","):
        qs = prompts[name][: a.n]

        def one(q):
            body = {"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": 4096, "temperature": a.temperature, "top_p": 0.95, "top_k": 20}
            if a.temperature == 0:
                body["seed"] = 0
            return stream(a.url, body)

        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(a.concurrency) as ex:
            res = list(ex.map(one, qs))
        pers = [r[5] for r in res if r[5]]
        toks = sum(r[2] for r in res)
        print(f"{name} temperature {a.temperature:g}: {len(qs)} requests, {toks} tokens in {time.monotonic() - t0:.0f} s, "
              f"{statistics.mean(pers):.3f} tokens a verify (mean over requests), decode per stream median {statistics.median(r[4] for r in res):.1f} tok/s", flush=True)


def blocks(a):
    """One stream at a time, the same N GSM8K requests with each block size B (the request's "speculative": B; 0 is plain
    decode): decode tok/s over the requests' streams and the tokens a verify block gave."""
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()][: a.n]
    for b in [int(x) for x in a.blocks.split(",")]:
        res = [stream(a.url, dict(q, model=a.model, max_tokens=a.max_tokens, speculative=b if b else False)) for q in reqs]
        toks = sum(r[2] for r in res)
        t = sum(r[2] / r[4] for r in res if r[4] > 0)  # seconds, taking the tokens after the first as the decode rate says
        pers = [r for r in res if r[5]]
        tail = f", {sum(r[2] for r in pers) / sum(r[2] / r[5] for r in pers):.2f} tokens/verify" if pers else ""
        print(f"block {b:2d}: {toks} tokens, decode {toks / t:.1f} tok/s (median stream {statistics.median(r[4] for r in res):.1f}){tail}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("depth")
p.add_argument("--docs", required=True)
p.add_argument("--depths", default="8192,32768,131072,252000")
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--plain", type=int, default=0, help="further requests per depth with \"speculative\": false")
p.add_argument("--block", type=int, default=0, help="the draft's block size for these requests (0: the server's)")
p = sub.add_parser("gsm")
p.add_argument("--requests", required=True)
p.add_argument("--concurrency", default="1,2")
p = sub.add_parser("reuse")
p.add_argument("--docs", required=True)
p.add_argument("--conv-depth", type=int, default=10000)
p.add_argument("--doc-depth", type=int, default=100000)
p = sub.add_parser("blocks")
p.add_argument("--requests", required=True)
p.add_argument("--n", type=int, default=6)
p.add_argument("--max-tokens", type=int, default=600)
p.add_argument("--blocks", default="0,4,6,8,12,16")
p = sub.add_parser("accept")
p.add_argument("--prompts", required=True)
p.add_argument("--datasets", default="gsm8k,humaneval,mt-bench")
p.add_argument("--n", type=int, default=16)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--concurrency", type=int, default=2)
a = ap.parse_args()
{"depth": depth, "gsm": gsm, "reuse": reuse, "accept": accept, "blocks": blocks}[a.cmd](a)
