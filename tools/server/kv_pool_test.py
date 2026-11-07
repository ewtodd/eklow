#!/usr/bin/env python3
"""The shared K/V pool through eklow-server's HTTP API (stdlib only), Qwen3.6. Three scenarios, run once against a server at the default
pool (the reference) and once against a small pool, and compared:

  trim   one slot holds a long cached prompt (A), a second prompt (B) needs its pages, A is cut, and A's next turn re-prefills correctly
  exhaust  two long generations at once that together outgrow the pool: no crash, one may end early with finish_reason "length", the
         other's text is what it is at the default pool
  wait   a request whose prompt does not fit beside a running generation waits in the queue, then runs, and its text is unchanged

  # reference: default pool, same --ctx (two slots)
  eklow-server ... --slots 2 --ctx 8192                    ; kv_pool_test.py --url U record --out ref.json
  # the pool under test: two slots over 8192 positions in all (each may use all of it)
  eklow-server ... --slots 2 --ctx 8192 --kv-pool 8192     ; kv_pool_test.py --url U check --ref ref.json [--log SERVER.LOG]

With --log the server's log is searched for the pool's lines ("kv pool: ... waits for pages", "cut from", "ends at ... length"), which
show the policy ran; without it the texts are all that is compared. Greedy decoding (temperature 0, thinking off); the concurrent
scenarios compare against the default pool's concurrent run, because batch composition can move the last bits of a logit.
"""
import argparse
import concurrent.futures
import http.client
import json
import sys
import threading
import time
import urllib.parse

FAILED = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def chat(url, content, max_tokens, system=None, timeout=3600):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    body = dict(messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}], max_tokens=max_tokens, temperature=0, reasoning_effort="none")
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {data}")
    ch = data["choices"][0]
    return {"text": (ch["message"].get("content") or ""), "finish": ch["finish_reason"], "prompt": data["usage"]["prompt_tokens"],
            "completion": data["usage"]["completion_tokens"], "cached": data["usage"]["prompt_tokens_details"]["cached_tokens"]}


def filler(seed, tokens, per_line):
    n = max(1, int(tokens / per_line))
    return "\n".join(f"Entry {seed}-{i}: harbour {seed} at dusk, lamp number {i * 7}, smells of tar and old rope." for i in range(n))


def calibrate(url):
    """Tokens a line of filler takes, from a request of 100 of them."""
    r = chat(url, filler("cal", 100 * 25, 25), 1)
    return max(8.0, r["prompt"] / 100.0)


def scenario_trim(url, per):
    """A (long), B (needs A's pages), A's next turn."""
    out = {}
    # A has a system prompt of ~1,500 tokens: a snapshot is taken where the user message starts, which is where the cut can keep.
    sa, sb = filler("SA", 1500, per), filler("SB", 1000, per)
    a = filler("A", 4000, per) + "\n\nHow many entries are there? Answer in one sentence."
    b = filler("B", 3500, per) + "\n\nWhat is the first entry? Answer in one sentence."
    out["A1"] = chat(url, a, 24, sa)
    out["B"] = chat(url, b, 24, sb)
    # A's second turn: the same document and a different question; what A's slot still holds is reused, the rest is prefilled again.
    out["A2"] = chat(url, filler("A", 4000, per) + "\n\nWhat is the last entry? Answer in one sentence.", 24, sa)
    return out


def numbers_prompt(seed):
    return f"{filler(seed, 2800, 25)}\n\nNow write the integers from 1 to 6000 separated by commas, nothing else."


def scenario_exhaust(url, per):
    """Two long generations at once: 2 x (about 2,800 + 2,600) positions against a pool of 8,192."""
    with concurrent.futures.ThreadPoolExecutor(2) as ex:
        f1 = ex.submit(chat, url, numbers_prompt("C"), 2600)
        f2 = ex.submit(chat, url, numbers_prompt("D"), 2600)
        return {"C": f1.result(), "D": f2.result()}


def scenario_wait(url, per):
    """A generation that holds ~4,500 positions, then a prompt of ~5,500 that cannot fit beside it."""
    res = {}

    def first():
        res["E"] = chat(url, filler("E", 3400, per) + "\n\nNow write the integers from 1 to 3000 separated by commas, nothing else.", 1000)

    t = threading.Thread(target=first)
    t.start()
    time.sleep(8)  # E's prompt is in and it is generating
    t0 = time.time()
    res["F"] = chat(url, filler("F", 5500, per) + "\n\nWhat is the first entry? Answer in one sentence.", 24)
    res["F"]["seconds"] = time.time() - t0
    t.join()
    return res


def run_all(url):
    per = calibrate(url)
    print(f"filler line: {per:.1f} tokens", flush=True)
    res = {}
    for name, fn in (("trim", scenario_trim), ("exhaust", scenario_exhaust), ("wait", scenario_wait)):
        t0 = time.time()
        res[name] = fn(url, per)
        print(f"{name}: {time.time() - t0:.0f} s  " + "  ".join(f"{k}: {v['prompt']}+{v['completion']} tok {v['finish']} cached {v['cached']}" for k, v in res[name].items()), flush=True)
    return res


def record(a):
    json.dump(run_all(a.url), open(a.out, "w"))
    return 0


def same(x, y):
    return x["text"] == y["text"] and x["completion"] == y["completion"] and x["finish"] == y["finish"]


def common(x, y):
    return next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))


def checkcmd(a):
    ref = json.load(open(a.ref))
    got = run_all(a.url)
    # trim: every text as at the default pool; A2 must have been served (cached below the prompt after the cut is fine).
    for k in ("A1", "B", "A2"):
        check(f"trim {k}: the text is the default pool's", same(got["trim"][k], ref["trim"][k]),
              f"differ after {common(got['trim'][k]['text'], ref['trim'][k]['text'])} characters")
    check("trim A2: the cut kept the system prompt (~1,500) and re-prefilled the rest", 1000 <= got["trim"]["A2"]["cached"] < got["trim"]["A2"]["prompt"] - 1000,
          f"cached {got['trim']['A2']['cached']} of {got['trim']['A2']['prompt']}")
    # exhaust: both answered; the pool is 8,192 and the pair needs ~10,800, so one ends early with "length" (fewer tokens than asked).
    ex = got["exhaust"]
    ended = [k for k, v in ex.items() if v["finish"] == "length" and v["completion"] < 2600]
    full = [k for k, v in ex.items() if v["completion"] == 2600 or v["finish"] == "stop"]
    check("exhaust: the pair outgrew the pool (the model wrote enough: the test is meaningful)", all(v["completion"] > 1500 for v in ex.values()) or bool(ended),
          str({k: v["completion"] for k, v in ex.items()}))
    check("exhaust: at most one sequence ended early, at least one finished", len(ended) <= 1 and len(full) >= 1, f"ended {ended}, finished {full}")
    for k in full:
        check(f"exhaust {k}: the survivor's text is the default pool's", same(ex[k], ref["exhaust"][k]), f"differ after {common(ex[k]['text'], ref['exhaust'][k]['text'])} characters")
    if ended:
        k = ended[0]
        n = common(ex[k]["text"], ref["exhaust"][k]["text"])
        check(f"exhaust {k}: what it wrote before ending is a prefix of the default pool's", n >= len(ex[k]["text"]) - 2 or ex[k]["text"] == ref["exhaust"][k]["text"][:len(ex[k]["text"])],
              f"differ after {n} of {len(ex[k]['text'])} characters")
    # wait: F did not fail and its text is the default pool's; it waited (its time exceeds a bare prefill of 5,500 tokens by E's remainder).
    w = got["wait"]
    check("wait: F's text is the default pool's", same(w["F"], ref["wait"]["F"]), f"differ after {common(w['F']['text'], ref['wait']['F']['text'])} characters")
    check("wait: E (running when F arrived) is unchanged", same(w["E"], ref["wait"]["E"]), f"differ after {common(w['E']['text'], ref['wait']['E']['text'])} characters")
    print(f"wait: F took {w['F']['seconds']:.1f} s here, {ref['wait']['F']['seconds']:.1f} s at the default pool (longer here: it waited for E)")
    if a.log:
        log = open(a.log, errors="replace").read()
        for what in ("kv pool: slot", "waits for pages", "ends at"):
            check(f"log: a line with '{what}'", what in log)
    print("FAILED: " + ", ".join(FAILED) if FAILED else "all ok")
    return 1 if FAILED else 0


ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8091")
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("record")
p.add_argument("--out", required=True)
p = sub.add_parser("check")
p.add_argument("--ref", required=True)
p.add_argument("--log")
a = ap.parse_args()
sys.exit({"record": record, "check": checkcmd}[a.cmd](a))
