#!/usr/bin/env python3
# Embedding latency while a long chat prompt prefills: one chat request, and an embedding request
# every few seconds until it ends.
#   embed_contend.py CHAT_URL EMBED_URL PROMPT_FILE [BYTES] [INTERVAL_S] [INPUTS]
import json, sys, threading, time, urllib.request

chat_url, embed_url, path = sys.argv[1], sys.argv[2], sys.argv[3]
n_bytes = int(sys.argv[4]) if len(sys.argv) > 4 else 190_000
interval = float(sys.argv[5]) if len(sys.argv) > 5 else 5.0
n_inputs = int(sys.argv[6]) if len(sys.argv) > 6 else 1
prompt = open(path, encoding="utf-8", errors="ignore").read()[:n_bytes] + "\n\nSummarize the code above in one sentence."
t_start = time.monotonic()


def log(rec):
    rec["at_s"] = round(time.monotonic() - t_start, 1)
    print(json.dumps(rec), flush=True)


def chat(done):
    body = {"model": "gemma-4-12b", "max_tokens": 32, "temperature": 0, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(chat_url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        u = json.loads(r.read())["usage"]
    log({"event": "chat", "s": round(time.monotonic() - t0, 1), "prompt_tokens": u["prompt_tokens"]})
    done.set()


done = threading.Event()
threading.Thread(target=chat, args=(done,), daemon=True).start()
time.sleep(2)  # let the prefill start
lat = []
while not done.is_set():
    body = json.dumps({"model": "embeddinggemma-2", "input": [f"what does tool {i} do?" for i in range(n_inputs)]}).encode()
    req = urllib.request.Request(embed_url + "/v1/embeddings", body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            r.read()
        ms = round((time.monotonic() - t0) * 1000)
        lat.append(ms)
        log({"event": "embed", "ms": ms, "during_prefill": not done.is_set()})
    except Exception as e:  # noqa: BLE001
        log({"event": "embed", "error": repr(e)})
    done.wait(interval)
lat.sort()
if lat:
    log({"event": "summary", "n": len(lat), "median_ms": lat[len(lat) // 2], "max_ms": lat[-1]})
