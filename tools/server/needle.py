#!/usr/bin/env python3
# Two near-full-window needle requests at once against eklow-server, with embedding requests
# timed while they prefill.
#   needle.py build HAY_A HAY_B BYTES_A BYTES_B OUTDIR    write req-a.json, req-b.json (+ .txt for counting)
#   needle.py run CHAT_URL EMBED_URL OUTDIR               send both, log to OUTDIR/needle.jsonl
import json, sys, threading, time, urllib.request

NEEDLES = {"a": (0.02, "plum-anchor-4417"), "b": (0.55, "cobalt-heron-2093")}
QUESTION = ("What is the passphrase for the eklow vault, given in a note somewhere in the code above? "
            "Reply with the passphrase alone on the first line, then describe in about 250 words what "
            "the code above does.")


def build(hay_a, hay_b, bytes_a, bytes_b, out):
    for key, path, n in (("a", hay_a, bytes_a), ("b", hay_b, bytes_b)):
        text = open(path, encoding="utf-8", errors="ignore").read()[: int(n)]
        frac, phrase = NEEDLES[key]
        cut = text.index("\n", int(len(text) * frac)) + 1
        note = f"# NOTE FOR THE READER: the passphrase for the eklow vault is {phrase}.\n"
        text = text[:cut] + note + text[cut:]
        content = text + "\n\n" + QUESTION
        open(f"{out}/req-{key}.txt", "w").write(content)
        body = {"model": "gemma-4-12b", "stream": True, "max_tokens": 700, "temperature": 0,
                "reasoning_effort": "none", "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": content}]}
        json.dump(body, open(f"{out}/req-{key}.json", "w"))


def log(out, rec):
    rec["t"] = round(time.time(), 1)
    line = json.dumps(rec)
    print(line, flush=True)
    with open(f"{out}/needle.jsonl", "a") as f:
        f.write(line + "\n")


def chat(url, out, key):
    body = open(f"{out}/req-{key}.json", "rb").read()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    usage, text, err = None, "", None
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            for line in r:
                line = line.strip()
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if ev.get("usage"):
                    usage = ev["usage"]
                for c in ev.get("choices", []):
                    piece = c.get("delta", {}).get("content") or ""
                    if piece:
                        if first is None:
                            first = time.monotonic()
                            log(out, {"slot": key, "event": "first_token", "ttft_s": round(first - t0, 1)})
                        text += piece
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        if hasattr(e, "read"):
            err += " " + e.read().decode(errors="replace")[:500]
    t1 = time.monotonic()
    rec = {"slot": key, "event": "done", "error": err, "ttft_s": round(first - t0, 1) if first else None,
           "total_s": round(t1 - t0, 1), "usage": usage, "expected": NEEDLES[key][1],
           "found": NEEDLES[key][1] in text.split("\n", 1)[0], "first_line": text.split("\n", 1)[0][:200],
           "text": text[:3000]}
    if first and usage and usage.get("completion_tokens", 0) > 1:
        rec["decode_tok_s"] = round((usage["completion_tokens"] - 1) / (t1 - first), 2)
    log(out, rec)


def embed_probe(url, out, stop):
    while not stop.is_set():
        body = json.dumps({"model": "embeddinggemma-2", "input": ["what does the relay tool do?"]}).encode()
        req = urllib.request.Request(url + "/v1/embeddings", body, {"Content-Type": "application/json"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                n = len(json.loads(r.read())["data"][0]["embedding"])
            log(out, {"event": "embed", "ms": round((time.monotonic() - t0) * 1000), "dim": n})
        except Exception as e:  # noqa: BLE001
            log(out, {"event": "embed", "error": repr(e)})
        stop.wait(120)


def run(chat_url, embed_url, out):
    stop = threading.Event()
    threads = [threading.Thread(target=chat, args=(chat_url, out, k)) for k in ("a", "b")]
    probe = threading.Thread(target=embed_probe, args=(embed_url, out, stop))
    log(out, {"event": "start"})
    for t in threads:
        t.start()
    probe.start()
    for t in threads:
        t.join()
    stop.set()
    probe.join()
    log(out, {"event": "end"})


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:7])
    else:
        run(*sys.argv[2:5])
