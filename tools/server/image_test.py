#!/usr/bin/env python3
# Image input through eklow-server, end to end: descriptions of pictures, a second turn that reuses the prefix with its
# image, two different pictures of the same size not confused, the refusals, and (with --embed-url) an embedding of a
# picture. Prints what came back; judge the descriptions by eye.
#   image_test.py CHAT_URL IMAGE... [--embed-url URL] [--out DIR] [--max-tokens N]
# Needs numpy (nix develop path:. -c python3) for the two same-size test pictures (BMP, written by hand).
import argparse, base64, json, struct, sys, time, urllib.error, urllib.request

import numpy as np


def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def data_url(path_or_bytes, mime=None):
    if isinstance(path_or_bytes, bytes):
        raw = path_or_bytes
    else:
        raw = open(path_or_bytes, "rb").read()
        mime = mime or ("image/png" if path_or_bytes.endswith(".png") else "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def bmp(rgb):
    h, w, _ = rgb.shape
    row = (w * 3 + 3) // 4 * 4
    pix = b"".join(rgb[y, :, ::-1].tobytes().ljust(row, b"\0") for y in range(h - 1, -1, -1))
    return b"BM" + struct.pack("<IHHI", 54 + len(pix), 0, 0, 54) + struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(pix), 2835, 2835, 0, 0) + pix


def ask(url, messages, max_tokens, label):
    t0 = time.time()
    code, r = post(url, "/v1/chat/completions", {"model": "m", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
                                                  "reasoning_effort": "none"})
    dt = time.time() - t0
    if code != 200:
        print(f"[{label}] HTTP {code}: {r.get('error', r).get('message', r)}")
        return None, r
    msg = r["choices"][0]["message"]
    u = r["usage"]
    print(f"[{label}] {dt:.1f} s, prompt {u['prompt_tokens']} (cached {u['prompt_tokens_details']['cached_tokens']}), completion {u['completion_tokens']}\n  {msg['content'].strip()}")
    return msg["content"], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--embed-url")
    ap.add_argument("--out")
    ap.add_argument("--max-tokens", type=int, default=160)
    a = ap.parse_args()
    q = "Describe this picture in a few sentences."

    for path in a.images:
        ask(a.url, [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(path)}}, {"type": "text", "text": q}]}],
            a.max_tokens, path.split("/")[-1])

    if a.images:
        # A second turn: the conversation so far, the answer, and a question; the first turn is in the slot already.
        first = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url(a.images[0])}}, {"type": "text", "text": q}]}]
        ans, r1 = ask(a.url, first, a.max_tokens, "turn 1")
        if ans is not None:
            ask(a.url, first + [{"role": "assistant", "content": ans}, {"role": "user", "content": "What is the most prominent colour in it?"}],
                a.max_tokens, "turn 2 (cached tokens should cover turn 1)")

    # Two pictures of the same size, different content: the second must not reuse the first's image rows.
    yy, xx = np.mgrid[0:240, 0:240]
    top_red = np.zeros((240, 240, 3), np.uint8)
    top_red[:120] = (230, 20, 20)
    top_red[120:] = (20, 20, 230)
    green_yellow = np.zeros((240, 240, 3), np.uint8)
    green_yellow[:, :120] = (20, 200, 40)
    green_yellow[:, 120:] = (240, 230, 20)
    sq = "What colours are in this picture and where? One sentence."
    for name, img in (("red above blue", top_red), ("green left of yellow", green_yellow), ("red above blue again", top_red)):
        ask(a.url, [{"role": "user", "content": [{"type": "text", "text": sq}, {"type": "image_url", "image_url": {"url": data_url(bmp(img), "image/bmp")}}]}],
            60, "same size: " + name)

    # Refusals.
    for label, url in (("http url", "http://example.com/a.png"), ("webp", data_url(b"RIFF\0\0\0\0WEBPVP8 ", "image/webp")),
                       ("garbage", data_url(b"\xff\xd8not a jpeg", "image/jpeg"))):
        code, r = post(a.url, "/v1/chat/completions", {"model": "m", "max_tokens": 1, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]})
        print(f"[refusal: {label}] HTTP {code}: {r.get('error', {}).get('message')}")

    if a.embed_url and a.images:
        part = [{"type": "text", "text": "title: none | text: "}, {"type": "image_url", "image_url": {"url": data_url(a.images[0])}}]
        t0 = time.time()
        code, r = post(a.embed_url, "/v1/embeddings", {"input": [part]})
        if code == 200:
            v = r["data"][0]["embedding"]
            print(f"[embedding of {a.images[0].split('/')[-1]}] {time.time() - t0:.1f} s, {len(v)} dims, prompt_tokens {r['usage']['prompt_tokens']}")
            if a.out:
                json.dump([v], open(f"{a.out}/image-embedding.json", "w"))
        else:
            print(f"[embedding] HTTP {code}: {r}")


main()
