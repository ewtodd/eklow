"""Check eklow's tokenizer against Hugging Face's on Gemma 4.

Usage: python3 tools/ref/tok_compare.py EKLOW_TOK GGUF TOKENIZER_JSON [--random N] [--files FILE ...]

Encodes a corpus with both (special tokens parsed, no BOS from either), compares ids, then decodes
Hugging Face's ids with both and compares text. The corpus: hand-written edge cases (whitespace,
newlines, scripts, emoji, byte fallback, partial and adjacent special tokens), the chat template
rendered around them, random strings over a mixed alphabet, and the contents of any --files (whole
and line by line). Exits 1 on any difference.
"""

import argparse
import random
import subprocess
import sys

from tokenizers import Tokenizer

EDGE = [
    "", " ", "  ", "   ", "\t", "\n", "\n\n", "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n",
    " \n ", "\r\n", "a\r\nb", "Hello world", "Hello  world", " leading", "trailing ", "  both  ",
    "a\tb\t\tc", "naïve café résumé", "Straße", "İstanbul", "日本語のテキスト", "中文测试，标点。",
    "한국어 문장", "Русский текст", "العربية", "हिन्दी", "ภาษาไทย", "עברית",
    "é", "​‌‍", "👍", "👍🏽", "👨‍👩‍👧", "🏳️‍🌈", "\U0001F9EA\U0001FAE0",
    "\x01\x02\x7f", " nbsp ", "  ", "﻿bom", "▁", "▁▁ ▁", " ▁ ",
    "def f(x):\n    return x ** 2\n", "if (a && b) { c(); }", '{"key": "value", "n": [1, 2.5, -3e10]}',
    "https://example.com/path?q=1&r=2#frag", "1234567890", "3.14159265358979", "0x7fff_ffff",
    "<bos>", "<eos>", "<|turn>", "<turn|>", "<|turn>user\nhi<turn|>\n", "<|tur", "<|turn", "<|turn>>",
    "<<|turn>", "<bos><bos>", "x<|channel>thought\n<channel|>y", '<|"|>quoted<|"|>', "<unused0>",
    "<0x41>", "<|tool_call>call:f{a:<|\"|>b<|\"|>}<tool_call|>", "<pad><mask><unk>", "< bos >",
    "a" * 1000, " " * 257, "ab" * 300, "\n" * 70, "é" * 100,
]


def chat(text):
    return f"<bos><|turn>user\n{text}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def random_strings(n, seed=7):
    rng = random.Random(seed)
    alphabet = (list("abcxyzABC019 .,;:!?'\"()[]{}<>|/\\_-+=*&^%$#@~`") + [" "] * 8 + ["\n"] * 4 + ["\t", "\r"] +
                list("éüßøñçşğłžőčřΩπλжщюאבגدشم中文日本語한국어🙂👍🏽❤️▁") + ["‍", "́", "\x00"[:0] + "\x7f"] +
                ["<bos>", "<|turn>", "<turn|>", "<|channel>", "<|\"|>", "<|tool_call>", "<|"])
    out = []
    for _ in range(n):
        k = rng.choice([1, 2, 3, 5, 8, 13, 40, 120, 400])
        out.append("".join(rng.choice(alphabet) for _ in range(k)))
    return out


def run(cmd, data):
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_tok")
    ap.add_argument("gguf")
    ap.add_argument("tokenizer_json")
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()

    corpus = EDGE + [chat(s) for s in EDGE] + random_strings(a.random)
    for path in a.files:
        text = open(path, encoding="utf-8", errors="replace").read()
        corpus.append(text)
        corpus.extend(line for line in text.splitlines(keepends=True) if line)
    corpus = [s for s in corpus if "\x00" not in s]

    hf = Tokenizer.from_file(a.tokenizer_json)
    want = [hf.encode(s, add_special_tokens=False).ids for s in corpus]
    out = run([a.eklow_tok, a.gguf, "encode"], "\x00".join(corpus).encode("utf-8") + b"\x00").decode().split("\n")
    got = [[int(x) for x in line.split()] for line in out[: len(corpus)]]
    if len(got) != len(corpus) or out[len(corpus):] != [""]:
        sys.exit(f"eklow-tok returned {len(out) - 1} lines for {len(corpus)} strings")
    bad = [(s, w, g) for s, w, g in zip(corpus, want, got) if w != g]
    for s, w, g in bad[:15]:
        k = next((i for i, (x, y) in enumerate(zip(w, g)) if x != y), min(len(w), len(g)))
        print(f"ENCODE DIFF {s[:60]!r}: at token {k}, hf {w[k:k + 6]} eklow {g[k:k + 6]}")
    n_tok = sum(len(w) for w in want)

    dec_in = "\n".join(" ".join(map(str, w)) for w in want) + "\n"
    ours = run([a.eklow_tok, a.gguf, "decode"], dec_in.encode()).split(b"\x00")
    if len(ours) != len(want) + 1:
        sys.exit(f"eklow-tok decoded {len(ours) - 1} strings for {len(want)}")
    ours = ours[: len(want)]
    theirs = [hf.decode(w, skip_special_tokens=False) for w in want]
    dbad = [(s, t, o.decode("utf-8", errors="surrogateescape")) for s, t, o in zip(corpus, theirs, ours)
            if t != o.decode("utf-8", errors="surrogateescape")]
    for s, t, o in dbad[:10]:
        print(f"DECODE DIFF {s[:60]!r}: hf {t[:60]!r} eklow {o[:60]!r}")
    print(f"{len(corpus)} strings, {n_tok} tokens: {len(bad)} encode and {len(dbad)} decode differences")
    sys.exit(1 if bad or dbad else 0)


if __name__ == "__main__":
    main()
