#!/usr/bin/env python3
"""Check, code point by code point, that eklow's pre-tokenizer for Inkling 2 (GPT-4o's regex over classes from Python's
unicodedata, tools/gen/gen_unicode_tables.py and gen_unicode_case.py) splits as Hugging Face's does, whose regex engine
(Oniguruma) carries its own Unicode version.

  classes_check.py EKLOW_TOK GGUF TOKENIZER_JSON

For every code point c (surrogates and NUL excepted) the strings in TEMPLATES (c alone and beside lower-case and
upper-case letters, digits, punctuation, white space, apostrophes and contractions, line breaks and slashes) are split by
Hugging Face's pre-tokenizer (the Split step of tokenizer.json) and by `eklow-tok pretok`; the piece lengths in bytes
must agree. Exits 1 on any difference.
"""

import json
import subprocess
import sys
import unicodedata

from tokenizers import Regex, pre_tokenizers

TEMPLATES = ["{c}", "a{c}", "{c}a", "A{c}", "{c}A", "Aa{c}", "{c}Aa", "aA{c}", "{c}aA", "{c}AA", "AA{c}b", "1{c}", "{c}1", "!{c}",
             "{c}!", " {c}", "{c} ", "{c}{c}", "\n{c}", "{c}\n", " {c}a", " {c}A", "'{c}", "'{c}x", "a'{c}x", "a'r{c}x", "a'l{c}",
             "{c}'s", "{c}A'S", "a{c}'s", "!{c}a", "!{c}A", "{c}/", "{c}\n/", " {c}\r\n", "{c}́a", "{c}́A", "A{c}́", "́{c}a"]


def main():
    tok, gguf, tj = sys.argv[1:4]
    spec = json.load(open(tj, encoding="utf-8"))
    regex = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    split = pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False)
    print("python unicodedata", unicodedata.unidata_version)
    cps = [cp for cp in range(1, 0x110000) if not 0xD800 <= cp <= 0xDFFF]
    bad, bad_cps = 0, set()
    chunk = 20000
    for lo in range(0, len(cps), chunk):
        strings = [t.format(c=chr(cp)) for cp in cps[lo:lo + chunk] for t in TEMPLATES]
        want = [" ".join(str(len(p.encode())) for p, _ in split.pre_tokenize_str(s)) for s in strings]
        data = "\x00".join(strings).encode() + b"\x00"
        out = subprocess.run([tok, gguf, "pretok"], input=data, capture_output=True, check=True).stdout.decode().split("\n")
        for i, (s, w) in enumerate(zip(strings, want)):
            if out[i] != w:
                bad += 1
                if bad <= 10:
                    print(f"SPLIT DIFF {s!r}: hf {w} eklow {out[i]}")
                bad_cps.add(cps[lo + i // len(TEMPLATES)])
    print(f"pre-tokenizer: {len(cps)} code points x {len(TEMPLATES)} contexts = {len(cps) * len(TEMPLATES)} strings, {bad} differences"
          + (f" at {len(bad_cps)} code points, first {[hex(c) for c in sorted(bad_cps)[:20]]}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
