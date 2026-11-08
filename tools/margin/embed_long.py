# embed_long.py URL TEXTFILE MODEL [TOKENS]: one embedding request of about TOKENS (8,000) tokens of the text, found by scaling the
# character count from the server's own prompt_tokens; prints the count and the time.
import json, sys, time, urllib.request
url, path, model = sys.argv[1], sys.argv[2], sys.argv[3]
want = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
text = open(path).read()
n = 4 * want
for _ in range(6):
    body = {"model": model, "input": text[:n]}
    t = time.time()
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/v1/embeddings", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600) as r:
            j = json.load(r)
    except Exception as e:
        print("error", n, e); n = int(n * 0.8); continue
    tok = j["usage"]["prompt_tokens"]
    print(f"{n} chars -> {tok} tokens, {len(j['data'][0]['embedding'])} dims, {time.time() - t:.2f} s", flush=True)
    if want * 0.95 <= tok <= want * 1.02: break
    n = int(n * want / max(tok, 1) * 0.98)
