#!/usr/bin/env python3
"""Prefill at depth, one-stream decode and concurrent throughput through eklow-server, measured the way
unholy's tools/box/perf-depth.sh and eb.sh measure, so the two READMEs compare.

  serve_perf.py --url http://127.0.0.1:8081 --model ALIAS depth --docs DIR [--depths 8192,32768,...] [--repeats 5]
  serve_perf.py --url http://127.0.0.1:8081 --model ALIAS gsm --requests gsm48.jsonl [--concurrency 1,2]

depth: for each DIR/gemma-DEPTH.txt (unique Pile text, unholy's documents cut to DEPTH Gemma tokens), one request on
it with nothing cached (its time to first token gives the prefill rate), then REPEATS - 1 more with other questions,
the document coming from the prefix cache; max_tokens 256, temperature 1.0, top_p 0.95, seed 7, thinking on (eklow's
default). Decode is each request's tokens after the first over the time after the first, from the stream (eklow-server
logs no per-request rates); the summary takes the median of the REPEATS.
gsm: unholy's 48 GSM8K 5-shot chat requests (thinking on, temperature 1.0, top_p 0.95, top_k 20, up to 1024 tokens),
CONC at a time; aggregate tok/s = completion tokens / wall time.
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


def stream(url, body):
    """POSTs a streamed chat request; returns (prompt tokens, cached tokens, completion tokens, TTFT s, decode tok/s)."""
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
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ev = json.loads(line[5:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                now = time.monotonic()
                first = first or now
                last = now
    c.close()
    n = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    ttft = (first or time.monotonic()) - t0
    decode = (n - 1) / (last - first) if n > 1 and last > first else 0.0
    return usage.get("prompt_tokens", 0), cached, n, ttft, decode


def depth(a):
    rows = []
    for d in [int(x) for x in a.depths.split(",")]:
        text = open(f"{a.docs}/gemma-{d}.txt").read()
        decs = []
        for i in range(a.repeats):
            body = {"model": a.model, "messages": [{"role": "user", "content": text + "\n\n" + QUESTIONS[i % len(QUESTIONS)]}],
                    "max_tokens": 256, "temperature": 1.0, "top_p": 0.95, "seed": 7}
            p, cached, n, ttft, dec = stream(a.url, body)
            print(f"request | prompt {p} ({cached} cached), completion {n} | ttft {ttft:.2f} s, "
                  f"prefill {(p - cached) / ttft:.1f} tok/s, decode {dec:.1f} tok/s", flush=True)
            if i == 0:
                first = (p, ttft)
            decs.append(dec)
        rows.append((d, first[0] / first[1], first[1], statistics.median(decs)))
    print("\ndepth    prefill tok/s (TTFT)    decode tok/s, median of", a.repeats)
    for d, rate, ttft, dec in rows:
        print(f"{d // 1024}k{'':<6}{rate:7.0f} ({ttft:.1f} s){'':<8}{dec:.1f}")


def gsm(a):
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()]
    for conc in [int(x) for x in a.concurrency.split(",")]:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(conc) as ex:
            res = list(ex.map(lambda q: stream(a.url, dict(q, model=a.model)), reqs))
        wall = time.monotonic() - t0
        toks = sum(r[2] for r in res)
        print(f"gsm, {len(reqs)} requests, {conc} at a time: {toks} completion tokens in {wall:.1f} s, "
              f"{toks / wall:.1f} tok/s aggregate, decode per stream median {statistics.median(r[4] for r in res):.1f} tok/s",
              flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--model", required=True)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("depth")
p.add_argument("--docs", required=True)
p.add_argument("--depths", default="8192,32768,131072,258048")
p.add_argument("--repeats", type=int, default=5)
p = sub.add_parser("gsm")
p.add_argument("--requests", required=True)
p.add_argument("--concurrency", default="1,2")
a = ap.parse_args()
depth(a) if a.cmd == "depth" else gsm(a)
