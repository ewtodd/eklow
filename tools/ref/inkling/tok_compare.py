#!/usr/bin/env python3
"""Check eklow's Inkling 2 tokenizer against llama.cpp c74c71a6 (the M0 ground truth) and Hugging Face's tokenizers.

  tok_compare.py EKLOW_TOK GGUF TOKENIZER_JSON M0_DIR [--repo DIR] [--random N] [--seed S] [--extra FILE ...]

  1. llama.cpp: the strings of M0_DIR/strings.json, ids from tokens.json, in both modes (`plain`: special tokens as text,
     `special`: parsed), and the four rendered prompts of M0_DIR/prompts against fwd/*/tokens.json.
  2. Hugging Face (tokenizer.json, added tokens parsed, no post-processor): every git-tracked text file of --repo, whole
     and as random slices (valid UTF-8 only: a slice that would cut a character is moved to a boundary), line by line,
     the edge cases below, and N random strings mixing scripts, digits, white space runs, emoji, combining marks,
     apostrophes and the added tokens. eklow's decode is compared with Hugging Face's on the ids it produced and on
     random id sequences (single-byte tokens, which make ill-formed UTF-8, and added tokens).
Exits 1 on any difference and prints the counts either way.
"""

import argparse
import json
import os
import random
import subprocess
import sys

from tokenizers import Tokenizer

ADDED = ["<|unused|>", "<|endoftext|>", "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
         "<|content_text|>", "<|content_image|>", "<|content_model_end_sampling|>", "<|unused_200007|>",
         "<|content_thinking|>", "<|end_message|>", "<|content_tool_error|>", "<|content_xml|>", "<|begin_of_text|>",
         "<|content_invoke_tool_json|>", "<|unused_200054|>", "<|content_invoke_tool_text|>"]

EDGE = [
    "", " ", "  ", "   ", "\t", "\n", "\n\n", "\n" * 33, " \n ", " \n", "\n ", "\r", "\r\n", "\r\n\r\n", "\r\n" * 20, "\r\r\n\n",
    "a\r\nb", "a \r\n b", "a  \n\n  b", "x\n \n \ny", "  hello", "   hello world", "\t\thello", "\n  \n x", "hello  \n", "hello   ",
    "a b", "a b", "a　b", "a b c", "a\x0bb\x0cc", "a\x85b", " x", "​x", "﻿x",
    "Hello world", "Hello  world", " leading", "trailing ", "  both  ", "a\tb\t\tc", "tabs\t\t\t",
    "I'm", "I'M", "i'm", "don't", "DON'T", "Don'T", "they're", "THEY'RE", "They'Re", "we've", "WE'VE", "you'll", "YOU'LL",
    "he'd", "HE'D", "it's", "IT'S", "it'S", "itſ", "it'ſ", "'ſ", "'ſx", "it'ſx", "'sx", "'Sx", "x'ſſ",
    "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D", "'", "''", "'''s", "'s'", "'ss", "'tis", "'twas", "rock'n'roll", "'ve", "x'",
    "x'y", "'re'd", "'l", "'ll'", "'lL", "'Ll", "'rE", "'vE", "can’t", "‘s", "`s", "\"s", "HELLO's", "ABC'd", "ABCd'S", "aBC'T",
    "CamelCaseWord", "camelCase", "HTTPServer", "HTTPServerError", "XMLHttpRequest", "ABCdef", "ABC", "ABC def", "abc DEF", "ABC1def",
    "ǅǆ Ǆ", "ǅ", "Aǅb", "ǅb", "aBǅ", "ʰa", "Aʰ", "ʰA", "aʰ", "Aʰb", "Aー", "アー", "Á", "Áb", "́ABC", "́abc", "́ABc",
    "ab́C", "Á́B", " ́ABC", "́́", "!́", " ́", "1́",
    "中A", "A中", "A中B", "中文abc", "ABC中", "ABC中d", "ΑαΣς", "ΣΣΣ",
    "ßSS", "İstanbul", "ıi", "KÅ", "ⅠⅡ", "Ⓐⓐ", "Აა",
    "naïve café résumé", "Straße", "日本語のテキスト", "한국어 문장", "Русский текст",
    "العربية", "हिन्दी", "ภาษาไทย", "עברית", "ქართული", "ελληνικά",
    "é", "Ẹ́", "ẹ́", "å", "Å", "Å", "ṩ", "ཱི", "ཱི",
    "1234567890", "٠١٢٣٤٥٦٧٨٩", "０１２３", "x²³ ½ ¼ ① ⑩ Ⅻ ⅷ ৪ ๕", "3.14159265358979",
    "0x7fff_ffff", "1,000,000.00", "12abc34", "a1b2", "1" * 7, "12 345", "½½½½", "ⅠⅡⅢⅣ",
    "\U0001f44d", "\U0001f44d\U0001f3fd", "\U0001f468‍\U0001f469‍\U0001f467", "\U0001f3f3️‍\U0001f308", "\U0001f1fa\U0001f1f8\U0001f1ef\U0001f1f5", "1️⃣", "❤️", "‍", "‍‍",
    "\x01\x02\x7f", "\x1f", "a\x1cb", "def f(x):\n    return x ** 2\n", "if (a && b) { c(); }", '{"key": "value", "n": [1, 2.5, -3e10]}',
    "https://example.com/path?q=1&r=2#frag", "!!!", "...", "?!", "--", "a-b_c", "[]{}()", "<>", "<<>>", "$$$", "a\n\n\n!", "!\n\n\nb", "!\r\n", "!\n\r",
    "a/b", "//", "a//\n/b", "!/", "!//\n", "/\n/", " /", " //\n\n", "x /\r\n y", ".\n/\r",
    "▁", "▁▁ ▁", "Ġ", "ĠĠ", "Ċ", "Ġa",
    "<|", "|>", "<||>", "<|>", "<|a|>", "<|message_user|", "<|message_user", "<message_user|>", "<<|message_user|>>", "<|message_user|>>",
    "<|<|message_user|>", "<|message_user|><|message_user|>", "<|MESSAGE_USER|>", "< |message_user|>", "<|message_user |>", "<|unused_200007|>",
    "<|unused_200058|>", "<|unused_199999|>", "<|unused_|>", "<|unused|>", "<|unused|", "<|content_xml|>x<|content_xml|>",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
    "a" * 1000, " " * 257, "ab" * 300, "\n" * 70, "é" * 100, "日" * 300, "a b " * 200, "z" * 49, "z" * 48, "z" * 47, "ab" * 24, "ab" * 25,
    "Aa" * 300, "aA" * 300, "1" * 100, "!" * 100, "'s" * 100,
] + ADDED + [a + b for a in ADDED[:6] for b in ADDED[-6:]] + [f"x{a}y" for a in ADDED] + [f" {a} " for a in ADDED] + [f"\n{a}\n" for a in ADDED]


def random_strings(n, seed):
    rng = random.Random(seed)
    alphabet = (list("abcxyzABCXYZ019 .,;:!?'\"()[]{}<>|/\\_-+=*&^%$#@~`") + [" "] * 8 + ["\n"] * 4 + ["\t", "\r", "\r\n", "/", "/\n"] +
                list("éüßøñçşğłžőčřΩπλжщюאבגدشم中文日本語한국어\U0001f642\U0001f44d\U0001f3fd❤️▁ǆŞİıſK") +
                ["‍", "́", "̣", "̈", "゙", "ᅡ", "ᆨ", " ", "　", " ", "\x7f", "\x85", "ʰ", "ー", "ǅ", "Ⅰ", "½", "٣", "\U0001d7d8", "ͅ", "ั"] +
                ["'s", "'T", "'re", "'LL", "'d", "'M", "'ve", "'ſ"] + ADDED[:8] + ADDED[-4:] + ["<|", "|>", "<|unused_2000"] + ["12", "345", "6", "٣", "½", "Ⅻ"] +
                ["Hello", "world", "HELLO", "CamelCase", "ABc", "aBC", "ABC"])
    out = []
    for _ in range(n):
        k = rng.choice([1, 2, 3, 5, 8, 13, 40, 120, 400])
        out.append("".join(rng.choice(alphabet) for _ in range(k)))
    return out


def slices(text, rng, n):
    out = []
    b = text.encode("utf-8")
    for _ in range(n):
        lo = rng.randrange(len(b) + 1)
        hi = min(len(b), lo + rng.choice([1, 2, 5, 17, 100, 500, 3000, 20000]))
        # a slice must be valid UTF-8: move both cuts to character boundaries
        while lo < len(b) and (b[lo] & 0xC0) == 0x80:
            lo += 1
        while hi < len(b) and (b[hi] & 0xC0) == 0x80:
            hi += 1
        out.append(b[lo:hi].decode("utf-8"))
    return out


def run(cmd, data, check=True):
    return subprocess.run(cmd, input=data, capture_output=True, check=check).stdout


def eklow_encode(tok, gguf, strings, special=True):
    """ids per string. The protocol is NUL-separated, so a string with NUL is dropped (and counted by the caller)."""
    data = b"".join(s.encode("utf-8") + b"\0" for s in strings)
    cmd = [tok, gguf, "encode"] + ([] if special else ["--no-special"])
    lines = run(cmd, data).decode().split("\n")
    return [[int(x) for x in ln.split()] for ln in lines[:len(strings)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_tok")
    ap.add_argument("gguf")
    ap.add_argument("tokenizer_json")
    ap.add_argument("m0")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--random", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    bad_total = 0

    # 1. llama.cpp
    ref = json.load(open(os.path.join(a.m0, "tokens.json"), encoding="utf-8"))
    strings = [r["text"] for r in ref]
    for mode, special in (("plain", False), ("special", True)):
        got = eklow_encode(a.eklow_tok, a.gguf, strings, special)
        bad = [i for i, r in enumerate(ref) if got[i] != r[mode]]
        for i in bad[:5]:
            print(f"  LLAMA {mode} DIFF {strings[i]!r}: llama {ref[i][mode]} eklow {got[i]}")
        print(f"llama.cpp strings, {mode}: {len(ref)} strings, {len(bad)} differences")
        bad_total += len(bad)
    prompts = []
    for name in ("short", "tools", "long", "deep"):
        text = open(os.path.join(a.m0, "prompts", name + ".txt"), encoding="utf-8").read()
        want = json.load(open(os.path.join(a.m0, "fwd", name, "tokens.json")))["ids"]
        got = eklow_encode(a.eklow_tok, a.gguf, [text])[0]
        prompts.append(text)
        ok = got == want
        print(f"llama.cpp prompt {name}: {len(want)} tokens, {'identical' if ok else 'DIFFERENT (%d tokens)' % len(got)}")
        bad_total += not ok

    # 2. Hugging Face
    hf = Tokenizer.from_file(a.tokenizer_json)
    hf.no_padding()
    hf.no_truncation()
    rng = random.Random(a.seed)
    files = subprocess.run(["git", "-C", a.repo, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    corpus = list(EDGE)
    nfiles, nbytes = 0, 0
    for f in files + a.extra:
        path = f if os.path.isabs(f) else os.path.join(a.repo, f)
        if not f or not os.path.isfile(path) or os.path.getsize(path) > 3_000_000:
            continue
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        nfiles += 1
        nbytes += len(text.encode())
        corpus.append(text)
        corpus += slices(text, rng, 6)
        lines = text.split("\n")
        corpus += ["\n".join(lines[i:i + 20]) for i in range(0, len(lines), 400)]
    decisions = os.path.join(a.repo, "DECISIONS.md")
    if os.path.exists(decisions):
        corpus.append(open(decisions, encoding="utf-8").read())
    corpus += prompts
    corpus += random_strings(a.random, a.seed)
    corpus = [s for s in corpus if "\0" not in s]
    got_all = eklow_encode(a.eklow_tok, a.gguf, corpus)
    bad = 0
    for s, got in zip(corpus, got_all):
        want = hf.encode(s, add_special_tokens=False).ids
        if got != want:
            bad += 1
            if bad <= 8:
                k = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), min(len(got), len(want)))
                print(f"  HF DIFF at token {k} of {len(s)} chars: {s[:80]!r}: hf {want[max(0, k - 2):k + 4]} eklow {got[max(0, k - 2):k + 4]}")
    print(f"Hugging Face encode: {len(corpus)} strings ({nfiles} files, {nbytes / 1e6:.1f} MB of text, slices, edge cases, {a.random} random), {bad} differences")
    bad_total += bad

    # decode
    ids_corpus = [hf.encode(s, add_special_tokens=False).ids for s in corpus[:300]]
    for _ in range(300):
        k = rng.choice([1, 2, 3, 5, 20, 100])
        ids_corpus.append([rng.choice([rng.randrange(256), rng.randrange(200058), rng.randrange(199998, 200058)]) for _ in range(k)])
    ids_corpus = [ids for ids in ids_corpus if "\0" not in hf.decode(ids)]  # the output is NUL-separated
    bad = 0
    for skip in (False, True):
        data = "\n".join(" ".join(map(str, ids)) for ids in ids_corpus).encode() + b"\n"
        out = run([a.eklow_tok, a.gguf, "decode"] + (["--skip-special"] if skip else []), data).split(b"\0")
        for ids, o in zip(ids_corpus, out):
            want = hf.decode(ids, skip_special_tokens=skip)
            if o.decode("utf-8") != want:
                bad += 1
                if bad <= 5:
                    print(f"  DECODE DIFF skip={skip} {ids[:8]}: hf {want[:60]!r} eklow {o.decode()[:60]!r}")
    print(f"Hugging Face decode: {len(ids_corpus)} id sequences x 2 modes, {bad} differences")
    bad_total += bad
    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
