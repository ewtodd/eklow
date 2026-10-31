"""Check, code point by code point, that eklow's NFC and pre-tokenizer (built on tables from Python's unicodedata,
tools/gen/gen_unicode_tables.py) agree with Hugging Face's, whose Rust crates carry their own Unicode version.

Usage: python3 tools/ref/qwen_classes_check.py EKLOW_TOK GGUF TOKENIZER_JSON

For every code point c (surrogates excepted) the strings c, "a"+c, c+"a", "1"+c, c+"1", "!"+c, c+"!", " "+c, c+" ",
c+c, "\\n"+c, c+"\\n", " "+c+"a", "'"+c, "'"+c+"x" (contractions), "'r"+c+"x", "'l"+c+"x" are NFC-normalized by both and split by
Hugging Face's pre-tokenizer regex (the Split step of tokenizer.json, on the normalized string) and by eklow-tok
`pretok`; the piece lengths must agree. Then NFC on c followed by combining marks of several classes, on "a"+c+marks,
on Hangul jamo and syllable sequences, and on 200,000 random short sequences of marks, starters and Hangul.
Exits 1 on any difference and prints the code points that differ.
"""

import json
import random
import subprocess
import sys
import unicodedata

from tokenizers import Regex, Tokenizer, normalizers, pre_tokenizers

TEMPLATES = ["{c}", "a{c}", "{c}a", "1{c}", "{c}1", "!{c}", "{c}!", " {c}", "{c} ", "{c}{c}", "\n{c}", "{c}\n", " {c}a",
             "'{c}", "'{c}x", "'r{c}x", "'l{c}x"]


def run(cmd, data):
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def main():
    eklow_tok, gguf, tj = sys.argv[1:4]
    spec = json.load(open(tj, encoding="utf-8"))
    regex = spec["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    split = pre_tokenizers.Split(Regex(regex), behavior="isolated", invert=False)
    nfc = normalizers.NFC()
    print("python unicodedata", unicodedata.unidata_version)

    cps = [cp for cp in range(1, 0x110000) if not 0xD800 <= cp <= 0xDFFF]  # NUL cannot cross the NUL-separated protocol
    bad_split, bad_cps = 0, set()
    chunk = 20000
    for lo in range(0, len(cps), chunk):
        strings = [t.format(c=chr(cp)) for cp in cps[lo:lo + chunk] for t in TEMPLATES]
        want = [" ".join(str(len(p.encode())) for p, _ in split.pre_tokenize_str(nfc.normalize_str(s))) for s in strings]
        out = run([eklow_tok, gguf, "pretok"], "\x00".join(strings).encode() + b"\x00").decode().split("\n")
        for i, (s, w) in enumerate(zip(strings, want)):
            if out[i] != w:
                bad_split += 1
                if bad_split <= 10:
                    print(f"SPLIT DIFF {s!r}: hf {w} eklow {out[i]}")
                bad_cps.add(cps[lo + i // len(TEMPLATES)])
    print(f"pre-tokenizer: {len(cps)} code points x {len(TEMPLATES)} contexts = {len(cps) * len(TEMPLATES)} strings, {bad_split} differences"
          + (f" at {len(bad_cps)} code points, first {[hex(c) for c in sorted(bad_cps)[:20]]}" if bad_split else ""))

    marks = [0x300, 0x315, 0x323, 0x327, 0x345, 0x5B0, 0x64B, 0x654, 0x3099, 0x302E, 0x1161, 0x11A8, 0x9BE, 0xBBE]
    strings = []
    for cp in cps:
        c = chr(cp)
        strings.append(c)
        for m in marks[:6]:
            strings.append(c + chr(m))
        strings.append("a" + c + chr(0x301))
        strings.append(c + chr(0x323) + chr(0x301))
    rng = random.Random(3)
    pool = [chr(m) for m in marks] + [chr(c) for c in range(0x41, 0x5B)] + [chr(c) for c in (0x1100, 0x1112, 0x1161, 0x1175, 0xAC00, 0xAC01, 0xD7A3, 0x11A8, 0x11C2, 0xB7, 0x0B47, 0x0B3E, 0x0F71, 0x0F72, 0x0F80, 0x1B05, 0x1B35, 0x1D15E, 0x1D157, 0x1D165, 0x2ADC, 0x387, 0x37E, 0x212B, 0x2126, 0xFB1D, 0x958)]
    for _ in range(200000):
        strings.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 8))))
    strings = [s for s in strings if "\x00" not in s]
    want = [nfc.normalize_str(s) for s in strings]
    out = run([eklow_tok, gguf, "nfc"], "\x00".join(strings).encode() + b"\x00").split(b"\x00")[:len(strings)]
    got = [o.decode("utf-8", errors="surrogateescape") for o in out]
    bad = [(s, w, g) for s, w, g in zip(strings, want, got) if w != g]
    for s, w, g in bad[:10]:
        print(f"NFC DIFF {[hex(ord(x)) for x in s]}: hf {[hex(ord(x)) for x in w]} eklow {[hex(ord(x)) for x in g]}")
    print(f"NFC: {len(strings)} strings, {len(bad)} differences")
    sys.exit(1 if bad or bad_split else 0)


if __name__ == "__main__":
    main()
