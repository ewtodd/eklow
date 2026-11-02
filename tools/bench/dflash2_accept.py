#!/usr/bin/env python3
"""dflash2_accept.py: z-lab's dflash/benchmark.py protocol against a llama-server with a DFlash 2 draft.

Prompts are built as benchmark.py builds them (same dataset formats, the same Random(42) shuffle, the first
N rows; MT-Bench's first turn), sent to /v1/chat/completions so the server applies the target's own chat
template with enable_thinking on. Acceptance is per request, completion tokens per verification step; the
server reports drafted and accepted tokens (timings.draft_n, draft_n_accepted), and every drafted block yields
its accepted tokens plus one sampled token, so with n tokens generated and a accepted:
    steps = n - 1 - a   (the first token comes out of the prefill, as in benchmark.py)
    acceptance = (n - 1) / steps
Raw per-request fields go to the JSONL file, so the definition can be recomputed.

  dflash2_accept.py --url http://127.0.0.1:8090 --dataset gsm8k --n 32 --temperature 1.0 --top-p 0.95 \
      --top-k 20 --max-tokens 4096 --concurrency 4 --out results.jsonl
"""
import argparse, json, random, statistics, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

DATASETS = {   # copied from z-lab/dflash @ 07ebd93d dflash/benchmark.py
    "gsm8k": (("openai/gsm8k", "main"), {"split": "test"},
              lambda x: "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(**x)),
    "humaneval": (("openai/openai_humaneval",), {"split": "test"},
                  lambda x: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```".format(**x)),
    "mt-bench": (("HuggingFaceH4/mt_bench_prompts",), {"split": "train"}, lambda x: x["prompt"][0]),
}


def load(name, n):
    from datasets import load_dataset
    args, kw, fmt = DATASETS[name]
    rows = [fmt(r) for r in load_dataset(*args, **kw)]
    order = list(range(len(rows)))
    random.Random(42).shuffle(order)
    return [rows[order[i]] for i in range(min(n, len(order)))]


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = load(a.dataset, a.n)
    lock = threading.Lock()
    out = open(a.out, "a")

    def one(i):
        body = {"messages": [{"role": "user", "content": prompts[i]}], "max_tokens": a.max_tokens,
                "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "stream": False,
                "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": not a.no_thinking}}
        if a.temperature == 0:
            body["seed"] = 0
        t0 = time.time()
        r = post(a.url, body)
        t = r.get("timings", {})
        n = t.get("predicted_n", r["usage"]["completion_tokens"])
        acc = t.get("draft_n_accepted", 0)
        steps = n - 1 - acc
        rec = {"label": a.label, "dataset": a.dataset, "i": i, "prompt_n": t.get("prompt_n"), "predicted_n": n,
               "draft_n": t.get("draft_n", 0), "draft_n_accepted": acc, "steps": steps,
               "acceptance": (n - 1) / steps if steps > 0 and "draft_n" in t else None,
               "predicted_per_second": t.get("predicted_per_second"), "finish": r["choices"][0]["finish_reason"],
               "wall_s": time.time() - t0}
        with lock:
            out.write(json.dumps(rec) + "\n"); out.flush()
        return rec

    with ThreadPoolExecutor(a.concurrency) as ex:
        recs = list(ex.map(one, range(len(prompts))))
    acc = [r["acceptance"] for r in recs if r["acceptance"] is not None]
    tps = [r["predicted_per_second"] for r in recs if r["predicted_per_second"]]
    summ = {"label": a.label, "dataset": a.dataset, "requests": len(recs),
            "acceptance_mean": statistics.mean(acc) if acc else None,
            "tokens": sum(r["predicted_n"] for r in recs),
            "truncated": sum(r["finish"] == "length" for r in recs),
            "decode_tps_median": statistics.median(tps) if tps else None}
    print(json.dumps(summ))
    out.write(json.dumps({"summary": summ}) + "\n"); out.close()


if __name__ == "__main__":
    main()
