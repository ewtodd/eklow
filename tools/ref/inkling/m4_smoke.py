#!/usr/bin/env python3
"""Inkling 2 Mini through eklow-server (M4): the API's paths on synthetic requests.

  m4_smoke.py BASE   (e.g. http://127.0.0.1:8091)

Checks, each printed with what came back: a plain answer; a tool call, whole and streamed; tool_choice required; the
turn after the tool's result (its prompt continues the slot: cached tokens); the same prompt again (it resumes at the
checkpoint before the prompt's last token); two requests at once; a stop string; max_tokens 0 (prefill only); a prompt
longer than the context (400). Greedy where the answer is checked.
"""
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "The current weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}},
         {"type": "function", "function": {"name": "add_note", "description": "Save a short note",
                                            "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                                                           "required": ["text"]}}}]
failures = []


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def stream(body, timeout=1800):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks


def check(name, ok, detail):
    print(("ok   " if ok else "FAIL ") + name + ": " + detail, flush=True)
    if not ok:
        failures.append(name)


def chat(messages, **kw):
    t0 = time.time()
    status, r = post("/v1/chat/completions", dict({"messages": messages}, **kw))
    return status, r, time.time() - t0


def brief(r):
    c = r["choices"][0]
    m = c["message"]
    return json.dumps({"finish": c["finish_reason"], "content": (m.get("content") or "")[:160], "reasoning": len(m.get("reasoning_content") or ""),
                       "tool_calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
                       "usage": r.get("usage")}, ensure_ascii=False)


status, models = 0, None
with urllib.request.urlopen(BASE + "/v1/models", timeout=10) as r:
    models = json.load(r)
check("models", bool(models.get("data")), json.dumps(models)[:200])

q1 = [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}]
status, r, dt = chat(q1, temperature=0)
check("plain", status == 200 and "391" in (r["choices"][0]["message"].get("content") or ""), f"{dt:.1f}s " + brief(r))

q2 = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's the weather in Lisbon right now?"}]
status, r, dt = chat(q2, tools=TOOLS, temperature=0)
calls = r["choices"][0]["message"].get("tool_calls") or [] if status == 200 else []
check("tool call", status == 200 and r["choices"][0]["finish_reason"] == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather"
      and json.loads(calls[0]["function"]["arguments"]).get("city") == "Lisbon", f"{dt:.1f}s " + brief(r))

chunks = stream({"messages": q2, "tools": TOOLS, "temperature": 0})
finish = [c["choices"][0]["finish_reason"] for c in chunks if c.get("choices") and c["choices"][0].get("finish_reason")]
scalls = [t for c in chunks if c.get("choices") for t in (c["choices"][0]["delta"].get("tool_calls") or [])]
usage = [c["usage"] for c in chunks if c.get("usage")]
check("tool call streamed", finish == ["tool_calls"] and scalls and scalls[0]["function"]["name"] == "get_weather" and usage,
      f"{len(chunks)} chunks, finish {finish}, calls {[(t['function']['name'], t['function']['arguments']) for t in scalls]}, usage {usage[-1] if usage else None}")
# the same prompt a third time: it resumes before its last token
check("retry resumes", bool(usage) and usage[-1].get("prompt_tokens_details", {}).get("cached_tokens", 0) >= usage[-1]["prompt_tokens"] - 1,
      f"usage {usage[-1] if usage else None}")

q3 = [{"role": "user", "content": "Save a note saying the plants need water on Friday."}]
status, r, dt = chat(q3, tools=TOOLS, tool_choice="required", temperature=0)
calls = r["choices"][0]["message"].get("tool_calls") or [] if status == 200 else []
check("tool_choice required", status == 200 and calls and calls[0]["function"]["name"] in ("add_note", "get_weather"), f"{dt:.1f}s " + brief(r))

if status == 200 and calls:
    turn = q3 + [{"role": "assistant", "content": r["choices"][0]["message"].get("content") or "",
                  "reasoning_content": r["choices"][0]["message"].get("reasoning_content"),
                  "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"], "content": "saved"}]
    status, r2, dt = chat(turn, tools=TOOLS, temperature=0)
    cached = (r2.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens", 0) if status == 200 else 0
    check("turn after the tool", status == 200 and r2["choices"][0]["finish_reason"] == "stop" and cached > 0, f"{dt:.1f}s " + brief(r2))

out = {}


def worker(k, msgs):
    out[k] = chat(msgs, max_tokens=300)


ths = [threading.Thread(target=worker, args=(k, [{"role": "user", "content": f"Write two sentences about the number {k}."}])) for k in (7, 11)]
t0 = time.time()
for t in ths:
    t.start()
for t in ths:
    t.join()
check("two at once", all(out[k][0] == 200 for k in out), f"{time.time() - t0:.1f}s " + " | ".join(brief(out[k][1]) for k in out))

status, r, dt = chat([{"role": "user", "content": "Count from 1 to 20, separated by commas, nothing else."}], temperature=0, stop=["7"],
                     reasoning_effort="none")
content = r["choices"][0]["message"].get("content") or "" if status == 200 else ""
check("stop string", status == 200 and "7" not in content and r["choices"][0]["finish_reason"] == "stop", f"{dt:.1f}s " + brief(r))

status, r, dt = chat([{"role": "user", "content": "A prompt to keep in a slot."}], max_tokens=0)
check("prefill only", status == 200 and r["usage"]["completion_tokens"] == 0, f"{dt:.1f}s " + json.dumps(r.get("usage")))

status, r, dt = chat([{"role": "user", "content": "word " * 70000}], max_tokens=4)
check("too long", status == 400, f"{status} {json.dumps(r)[:200]}")

print("failures:", failures or "none")
sys.exit(1 if failures else 0)
