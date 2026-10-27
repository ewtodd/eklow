#!/usr/bin/env python3
"""Embeddings of image inputs, for comparing eklow-server with llama.cpp's llama-server (both take
{"input": [{"content": [parts]}]}); writes the vectors in embed_bench.py's form, so that
  embed_bench.py compare eklow.json llama.json
prints the cosines.
  embed_image_test.py URL OUT.json IMAGE...
Cases: each image alone; text then image; image then text; the first two images in one input.
"""
import base64, json, sys, urllib.request


def part_image(path):
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(open(path, 'rb').read()).decode()}"}}


def text(s):
    return {"type": "text", "text": s}


url, out, images = sys.argv[1], sys.argv[2], sys.argv[3:]
cases = [[part_image(p)] for p in images]
cases.append([text("title: none | text: "), part_image(images[0])])
cases.append([part_image(images[0]), text(" front page of a newspaper")])
if len(images) > 1:
    cases.append([text("task: search result | query: "), part_image(images[0]), text(" and "), part_image(images[1])])
vecs = []
for parts in cases:
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps({"input": [{"content": parts}]}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    vecs.append(d["data"][0]["embedding"])
    print(len(vecs), "tokens", d["usage"]["prompt_tokens"], flush=True)
json.dump(vecs, open(out, "w"))
