#!/usr/bin/env python3
"""What a long prefill costs a stream that is already generating (layer-major groups stall the other slot's decode for a whole
group): one request streams a long answer; DELAY seconds in, a second request with a DEPTH-token document arrives.

  serve_stall.py --url http://127.0.0.1:8081 --model ALIAS --docs DIR [--depth 32768] [--delay 4]

Prints the first stream's tokens/s before the document arrives, its longest gap between tokens after, and the second request's
time to first token. Stdlib only."""
import argparse
import http.client
import json
import threading
import time
import urllib.parse


def stream(url, body, stamps):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=7200)
    c.request("POST", "/v1/chat/completions", json.dumps(dict(body, stream=True)), {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:300]!r}")
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        for ch in json.loads(line[5:]).get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                stamps.append(time.monotonic())
    c.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--depth", type=int, default=32768)
    ap.add_argument("--delay", type=float, default=4.0)
    a = ap.parse_args()
    text = open(f"{a.docs}/qwen-{a.depth}.txt").read()
    gen = []
    t_first = threading.Thread(target=stream, args=(a.url, {"model": a.model, "max_tokens": 1500, "temperature": 1.0, "seed": 3,
                                                         "messages": [{"role": "user", "content": "Count from 1 to 400, one number per line, no other text."}]}, gen))
    t_first.start()
    time.sleep(a.delay)
    doc = []
    t_arrive = time.monotonic()
    stream(a.url, {"model": a.model, "max_tokens": 1, "temperature": 1.0, "seed": 7,
                   "messages": [{"role": "user", "content": text + "\n\nGive the text above a short title."}]}, doc)
    ttft = (doc[0] if doc else time.monotonic()) - t_arrive
    t_first.join()
    before = [t for t in gen if t < t_arrive]
    after = [t for t in gen if t >= t_arrive]
    rate = (len(before) - 1) / (before[-1] - before[0]) if len(before) > 1 else 0.0
    gaps = [b - a_ for a_, b in zip([t_arrive] + after, after)]
    print(f"stream: {len(gen)} tokens; {rate:.1f} tok/s before the document; document {a.depth} tokens: ttft {ttft:.2f} s; "
          f"longest gap in the stream after it arrived {max(gaps) if gaps else 0:.2f} s; {len(after)} tokens came out during/after")


main()
