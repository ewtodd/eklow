#!/usr/bin/env python3
# Time to first token and decode rate for a long system prompt, cold then with the prefix cached.
#   ttft.py URL SYSTEM_FILE [MODEL]
import json, sys, time, urllib.request

url, sysfile = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "gemma-4-12b"
system = open(sysfile).read()
users = [
    "In one sentence, what is eklow?",
    "In one sentence, why does eklow quantize prefill activations differently from decode?",
    "List the milestones by number only.",
]


def run(user):
    body = {
        "model": model, "stream": True, "max_tokens": 256, "temperature": 0,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    n_delta = 0
    usage = None
    text = ""
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    if first is None:
                        first = time.monotonic()
                    n_delta += 1
                    text += d.get("content") or ""
    t1 = time.monotonic()
    out = {
        "user": user, "ttft_s": round(first - t0, 3) if first else None, "total_s": round(t1 - t0, 3),
        "prompt_tokens": usage and usage.get("prompt_tokens"),
        "cached_tokens": usage and (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "completion_tokens": usage and usage.get("completion_tokens"),
    }
    if first and usage and usage.get("completion_tokens", 0) > 1:
        out["decode_tok_s"] = round((usage["completion_tokens"] - 1) / (t1 - first), 1)
    out["content"] = text.strip()[:160]
    print(json.dumps(out), flush=True)


for u in users:
    run(u)
