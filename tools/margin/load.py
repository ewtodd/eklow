# load.py URL DOCS: two long prompts at once in the two slots (88k and 167k, the 2026-10-27 margin test), each followed by 300 tokens of decode
# with the draft; prints each request's usage.
import concurrent.futures as cf, json, sys, time, urllib.request
url, docs = sys.argv[1], sys.argv[2]
def ask(depth):
    text = open(f"{docs}/qwen-{depth}.txt").read()
    body = {"model": "sage-a1.5", "messages": [{"role": "system", "content": "You are a careful reader."}, {"role": "user", "content": text + "\n\nSummarize the passage above in three sentences."}],
            "max_tokens": 300, "temperature": 1.0, "seed": 7}
    t = time.time()
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600) as r:
        j = json.load(r)
    return depth, round(time.time() - t), j.get("usage"), j["choices"][0]["finish_reason"]
with cf.ThreadPoolExecutor(2) as ex:
    for res in ex.map(ask, sys.argv[3:]): print(res, flush=True)
