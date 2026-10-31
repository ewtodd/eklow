"""Check eklow's Qwen3.6 tokenizer against Hugging Face's on tokenizer.json.

Usage: python3 tools/ref/qwen_tok_compare.py EKLOW_TOK GGUF TOKENIZER_JSON [--random N] [--decode-random N] [--files FILE ...]

Encodes a corpus with both (added tokens parsed, no BOS from either) and compares ids, then decodes Hugging Face's ids
with both (added tokens kept and skipped) and compares text. The corpus: hand-written edge cases (CJK and other
scripts, combining marks, NFC/NFD pairs and singletons, Hangul jamo, digits of every kind, contractions in every
case, \\r\\n runs, white space runs before letters, emoji, every added token alone, adjacent, partial and in the
chat format), the chat format wrapped around each, random strings over a mixed alphabet, and the contents of any
--files, whole and line by line. Decoding is also run on random id sequences (single-byte tokens, which make
ill-formed UTF-8, added tokens and ids the tokenizer does not know). Exits 1 on any difference.
"""

import argparse
import random
import subprocess
import sys

from tokenizers import Tokenizer

ADDED = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
         "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
         "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
         "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
         "<think>", "</think>"]

EDGE = [
    "", " ", "  ", "   ", "\t", "\n", "\n\n", "\n" * 33, " \n ", " \n", "\n ", "\r", "\r\n", "\r\n\r\n", "\r\n" * 20, "\r\r\n\n",
    "a\r\nb", "a \r\n b", "a  \n\n  b", "x\n \n \ny", "  hello", "   hello world", "\t\thello", "\n  \n x", "hello  \n", "hello   ",
    "a\u00a0b", "a\u2003b", "a\u3000b", "a\u2028b\u2029c", "a\x0bb\x0cc", "a\x85b", "\u1680x", "\u200bx", "\ufeffx",
    "Hello world", "Hello  world", " leading", "trailing ", "  both  ", "a\tb\t\tc", "tabs\t\t\t",
    "I'm", "I'M", "i'm", "don't", "DON'T", "Don'T", "they're", "THEY'RE", "They'Re", "we've", "WE'VE", "you'll", "YOU'LL", "he'd", "HE'D",
    "it's", "IT'S", "it'S", "it\u017f", "it'\u017f", "'\u017f", "'\u017fx", "it'\u017fx", "'sx", "'Sx", "x'\u017f\u017f", "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D", "'", "''", "'''s", "'s'", "'ss", "'tis", "'twas",
    "rock'n'roll", "'ve", "x'", "x'y", "'re'd", "'l", "'ll'", "'lL", "'Ll", "'rE", "'vE", "can’t", "‘s", "`s", "\"s",
    "naïve café résumé", "Straße", "İstanbul", "ǅ ǆ Ǆ", "ﬁ ﬂ", "日本語のテキスト", "中文测试，标点。", "한국어 문장", "ひらがなカタカナ", "𠮷野家", "\U00020000\U0002FA1D",
    "Русский текст", "العربية", "ह\u093fन\u094dद\u0940", "ภาษาไทย", "עברית", "ქართული", "ελληνικά", "Ελληνικά ΣΑΣ", "త\u0c46ల\u0c41గ\u0c41", "ᏣᎳᎩ", "ꦄꦏ\ua9c0ꦱꦫ",
    "e\u0301", "E\u0301\u0323", "e\u0323\u0301", "e\u0301\u0323", "a\u030a", "\u212b", "\u00c5", "A\u030a", "ṩ", "s\u0323\u0307", "s\u0307\u0323",
    "क़", "क\u093c", "\u212a", "\u2126", "\u0344", "\u0340", "\u0f73", "\u0f71\u0f72", "각", "각", "한", "한" + "한",
    "\u0b47\u0b3e", "\u09c7\u09be", "ᬅ\u1b35", "\U0001d157\U0001d165", "\U0001d15e", "ǆ\u0301", "יִ", "שּׁ", "⫝̸", "·;",
    "a\u0300\u0301\u0302\u0303\u0304", "\u0301", "\u0301a", "\u0301\u0301", "ก\u0e48\u0e49", "ก\u0e49", "ا\u064b\u0651", "\u0651\u064b",
    "\u07fd", "a\u07fd\u0301", "\u0897\u0323", "\u0c3c", "\u0d3b\u0301", "\u1ac1\u0323", "\U00011070", "\U0001e5ee\u0323",
    "ą\u0301", "x\u0328\u0301", "x\u0301\u0328", "Ǫ\u0304", "Ạ\u0302",
    "1234567890", "٠١٢٣٤٥٦٧٨٩", "０１２３", "x²³ ½ ¼ ① ⑩ Ⅻ ⅷ ৪ ๕ ໙ ᠑ 𝟏𝟐", "3.14159265358979", "0x7fff_ffff", "1,000,000.00", "12abc34", "a1b2",
    "👍", "👍🏽", "👨\u200d👩\u200d👧", "🏳\ufe0f\u200d🌈", "🇺🇸🇯🇵", "\U0001F9EA\U0001FAE0", "1\ufe0f\u20e3", "❤\ufe0f", "©\ufe0f", "\u200d", "\u200d\u200d",
    "\x01\x02\x7f", "\x00"[:0] + "\x1f", "a\x1cb", "def f(x):\n    return x ** 2\n", "if (a && b) { c(); }", '{"key": "value", "n": [1, 2.5, -3e10]}',
    "https://example.com/path?q=1&r=2#frag", "!!!", "...", "?!", "--", "a-b_c", "[]{}()", "<>", "<<>>", "$$$", "a\n\n\n!", "!\n\n\nb", "!\r\n", "!\n\r",
    "▁", "▁▁ ▁", "Ġ", "ĠĠ", "Ċ", "Ġa", "ðŁĺĢ",
    "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>", "<think>\nhmm\n</think>\n\nanswer",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n", "<|im_sta", "<|im_start", "<|im_start|", "<|im_start|>>", "<<|im_start|>", "<<think>>",
    "<|im_start|><|im_start|>", "<think><think>", "<think", "think>", "< think>", "<Think>", "<THINK>", "<|IM_START|>", "</Think>",
    "<|audio_start|>", "<|audio_end|>", "<tts_pad>", "<|audio_pad|>", "<tts_text_bos>", "[PAD248077]", "<|reserved_special_token_0|>", "<unk>", "<s>", "</s>", "<pad>",
    "a" * 1000, " " * 257, "ab" * 300, "\n" * 70, "é" * 100, "日" * 300, "a b " * 200, "z" * 49, "z" * 48, "z" * 47, "ab" * 24, "ab" * 25,
] + ADDED + [a + b for a in ADDED[:6] for b in ADDED[-6:]] + [f"x{a}y" for a in ADDED] + [f" {a} " for a in ADDED] + [f"\n{a}\n" for a in ADDED]


def chat(text):
    return f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def random_strings(n, seed=7):
    rng = random.Random(seed)
    alphabet = (list("abcxyzABC019 .,;:!?'\"()[]{}<>|/\\_-+=*&^%$#@~`") + [" "] * 8 + ["\n"] * 4 + ["\t", "\r", "\r\n"] +
                list("éüßøñçşğłžőčř\u03a9πλжщюאבגدشم中文日本語한국어🙂👍🏽❤\ufe0f▁ǆŞİı\u017fK") + ["\u200d", "\u0301", "\u0323", "\u0308", "\u3099", "ᅡ", "ᆨ", "\u00a0", "\u3000", "\u2028", "\x7f", "\x85"] +
                ["'s", "'T", "'re", "'LL", "'d", "'M", "'ve"] + ADDED[:8] + ADDED[-4:] + ["<|", "<think", "<tool"] + ["12", "345", "6", "٣", "½", "Ⅻ"])
    out = []
    for _ in range(n):
        k = rng.choice([1, 2, 3, 5, 8, 13, 40, 120, 400])
        out.append("".join(rng.choice(alphabet) for _ in range(k)))
    return out


def run(cmd, data):
    return subprocess.run(cmd, input=data, capture_output=True, check=True).stdout


def byte_tokens(hf):
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for x in range(256):
        if x not in bs:
            bs.append(x)
            cs.append(256 + n)
            n += 1
    v = hf.get_vocab()
    return {b: v[chr(c)] for b, c in zip(bs, cs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_tok")
    ap.add_argument("gguf")
    ap.add_argument("tokenizer_json")
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--decode-random", type=int, default=20000)
    ap.add_argument("--files", nargs="*", default=[])
    a = ap.parse_args()

    corpus = EDGE + [chat(s) for s in EDGE] + random_strings(a.random)
    n_files = 0
    for path in a.files:
        text = open(path, encoding="utf-8", errors="replace").read()
        corpus.append(text)
        corpus.extend(line for line in text.splitlines(keepends=True) if line)
        n_files += 1
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

    # Decoding: Hugging Face's ids, then random sequences (ill-formed UTF-8 among them).
    rng = random.Random(9)
    btd = byte_tokens(hf)
    bt = [t for b, t in btd.items() if b != 0]  # NUL cannot cross the NUL-separated output
    seqs = list(want)
    for _ in range(a.decode_random):
        k = rng.choice([1, 2, 3, 4, 6, 10, 30])
        pool = rng.choice([bt, bt, bt + list(range(248044, 248070)), bt + [rng.randrange(0, 248044) for _ in range(5)],
                           [btd[b] for b in (0xC3, 0xA9, 0xE2, 0x82, 0xAC, 0xF0, 0x9F, 0x91, 0x8D, 0xED, 0xA0, 0xC0, 0x80, 0xFF, ord("a"))]
                           + [248045, 248068, 248070, 248300]])
        seqs.append([rng.choice(pool) for _ in range(k)])
    bad_dec = []
    for skip in (False, True):
        flag = ["--skip-special"] if skip else []
        ours = run([a.eklow_tok, a.gguf, "decode"] + flag, ("\n".join(" ".join(map(str, w)) for w in seqs) + "\n").encode()).split(b"\x00")
        if len(ours) != len(seqs) + 1:
            sys.exit(f"eklow-tok decoded {len(ours) - 1} strings for {len(seqs)}")
        theirs = [hf.decode(w, skip_special_tokens=skip) for w in seqs]
        bad_dec += [(w, t, o.decode("utf-8", errors="surrogateescape")) for w, t, o in zip(seqs, theirs, ours) if t != o.decode("utf-8", errors="surrogateescape")]
    for w, t, o in bad_dec[:10]:
        print(f"DECODE DIFF {w[:8]}: hf {t[:60]!r} eklow {o[:60]!r}")
    print(f"{len(corpus)} strings ({n_files} files), {n_tok} tokens: {len(bad)} encode differences; "
          f"{2 * len(seqs)} decodes ({len(seqs)} id sequences, with and without skipping special tokens): {len(bad_dec)} differences")
    sys.exit(1 if bad or bad_dec else 0)


if __name__ == "__main__":
    main()
