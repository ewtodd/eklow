"""Generates src/text/unicode_tables.hpp from Python's unicodedata: what Qwen's tokenizer needs of Unicode.

Usage: python3 tools/gen/gen_unicode_tables.py > src/text/unicode_tables.hpp

  - the classes the pre-tokenizer's regex names, as ranges: \\p{L} (Lu Ll Lt Lm Lo), \\p{M} (Mn Mc Me), \\p{N} (Nd Nl No)
    and \\s (Unicode White_Space, which is what Hugging Face's regex engine, Oniguruma, takes \\s to be; Python's own
    str.isspace also counts U+001C-U+001F and is not used for this);
  - NFC: the canonical combining class of every code point that has one, the full canonical decomposition of every code
    point that has one (Hangul syllables are algorithmic and left out), the primary composites (pairs that compose,
    after the composition exclusions), and the code points that a string must be free of for NFC to leave it unchanged
    (a non-zero combining class, anything that decomposes unstably, anything that can be the second half of a
    composition).

The classes are Python's Unicode version (printed in the header): Hugging Face's regex engine (Oniguruma) agrees with
it on every code point (tools/ref/qwen_classes_check.py). Its NFC does not: the Rust crate behind the tokenizers'
normalizer is Unicode 9 (it has the Adlam nukta, not 10.0's Malayalam vertical bar virama), so it neither orders nor composes
anything assigned since (a Telugu nukta, the
Quranic marks added in 14 and 16, Kirat Rai ...) and knows no decomposition of the characters Unicode 10-16 added
(Dives Akuru, Tulu-Tigalari, Todhri ...). eklow's tokenizer must produce Hugging Face's ids, so this script asks the
installed tokenizers (its NFC and NFD normalizers) which code points, decompositions and pairs it knows and leaves out
the rest; what it left out is printed to stderr. Without the tokenizers package the tables are Python's in full.
"""

import sys
import unicodedata


def ranges(pred_value):
    """Maximal runs [lo, hi] of code points with the same non-zero value."""
    out, start, cur = [], None, 0
    for cp in range(0x110000):
        v = pred_value(cp) if not (0xD800 <= cp <= 0xDFFF) else 0
        if v != cur:
            if cur:
                out.append((start, cp - 1, cur))
            start, cur = cp, v
    if cur:
        out.append((start, 0x10FFFF, cur))
    return out


WHITE_SPACE = set(range(0x09, 0x0E)) | {0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000} | set(range(0x2000, 0x200B))


def cls(cp):
    c = unicodedata.category(chr(cp))
    return (1 if c[0] == "L" else 0) | (2 if c[0] == "M" else 0) | (4 if c[0] == "N" else 0) | (8 if cp in WHITE_SPACE else 0)


def canonical_full(cp):
    d = unicodedata.decomposition(chr(cp))
    if not d or d.startswith("<"):
        return None
    out = []
    for h in d.split():
        sub = int(h, 16)
        out.extend(canonical_full(sub) or [sub])
    return out


def hf_normalizers():
    try:
        import tokenizers
        from tokenizers import normalizers
    except ImportError:
        return None
    return tokenizers.__version__, normalizers.NFC(), normalizers.NFD()


def main():
    hf = hf_normalizers()
    ccc_all = ranges(lambda cp: unicodedata.combining(chr(cp)))
    dropped = {"ccc": [], "decomp": [], "pairs": []}
    decomp = {}
    for cp in range(0x110000):
        if 0xAC00 <= cp <= 0xD7A3 or 0xD800 <= cp <= 0xDFFF:
            continue
        d = canonical_full(cp)
        if d:
            decomp[cp] = d
    # Primary composites: a two-code-point canonical decomposition that is not excluded (NFC maps the composite to
    # itself; an excluded one, or one whose first half is not a starter, is mapped to its decomposition).
    pairs = []
    for cp in decomp:
        d = unicodedata.decomposition(chr(cp))
        if d.startswith("<"):
            continue
        parts = [int(h, 16) for h in d.split()]
        if len(parts) == 2 and unicodedata.normalize("NFC", chr(cp)) == chr(cp):
            pairs.append((parts[0], parts[1], cp))
    pairs.sort()

    ccc = ccc_all
    if hf:
        _, nfc, nfd = hf[0], hf[1], hf[2]
        # Decompositions HF's NFD knows.
        for cp in sorted(decomp):
            if nfd.normalize_str(chr(cp)) == chr(cp):
                dropped["decomp"].append(cp)
                del decomp[cp]
        # Pairs HF's NFC composes.
        keep = []
        for a, b, c in pairs:
            if c in decomp and nfc.normalize_str(chr(a) + chr(b)) == chr(c):
                keep.append((a, b, c))
            else:
                dropped["pairs"].append((a, b, c))
        pairs = keep

        # Combining classes HF's NFD acts on: a mark of class > 1 is moved before U+0334 (class 1); a mark of class 1
        # is moved before U+0300 (class 230) by... no: U+0300 after it stays, so it is tested behind U+0300 instead.
        def known(cp):
            c = chr(cp)
            return nfd.normalize_str(c + "\u0334") != c + "\u0334" or nfd.normalize_str("\u0300" + c) != "\u0300" + c

        def hf_ccc(cp):
            v = unicodedata.combining(chr(cp))
            if v and cp not in decomp and not known(cp):
                dropped["ccc"].append(cp)
                return 0
            return v

        ccc = ranges(hf_ccc)
    second = {b for _, b, _ in pairs} | set(range(0x1161, 0x1176)) | set(range(0x11A8, 0x11C3))

    composites = {c for _, _, c in pairs}
    ccc_of = {}
    for lo, hi, v in ccc:
        for cp in range(lo, hi + 1):
            ccc_of[cp] = v

    def unsafe(cp):
        # A non-starter; a decomposable character that does not compose back (a singleton or an excluded one); the
        # second half of a composition.
        return int(cp in ccc_of or (cp in decomp and cp not in composites) or cp in second)

    flags = ranges(unsafe)

    o = sys.stdout.write
    o("// Generated by tools/gen/gen_unicode_tables.py from Python %s's unicodedata (Unicode %s)%s. Do not edit.\n"
      % (sys.version.split()[0], unicodedata.unidata_version,
         ", with the NFC data Hugging Face's tokenizers %s lacks left out" % hf[0] if hf else ""))
    o("#pragma once\n\n#include <cstdint>\n\nnamespace eklow::text::unicode {\n\n")
    o('inline constexpr const char* kVersion = "%s";\n\n' % unicodedata.unidata_version)

    def table(name, rows, doc):
        o("// %s\n" % doc)
        o("struct %sRange { uint32_t lo, hi; uint8_t v; };\n" % name)
        o("inline constexpr %sRange k%sRanges[] = {\n" % (name, name))
        for i in range(0, len(rows), 4):
            o("    " + " ".join("{0x%X,0x%X,%d}," % r for r in rows[i:i + 4]) + "\n")
        o("};\n\n")

    table("Class", ranges(cls), "Runs of code points by class: bit 0 \\p{L}, bit 1 \\p{M}, bit 2 \\p{N}, bit 3 \\s.")
    table("Ccc", ccc, "Runs of code points with the same non-zero canonical combining class.")
    table("Unsafe", flags, "Runs of code points a string must be free of to be left unchanged by NFC.")
    o("// Full canonical decompositions: kDecompCp sorted; code point i decomposes to kDecompPool[kDecompOff[i], kDecompOff[i+1]).\n")
    cps = sorted(decomp)
    o("inline constexpr uint32_t kDecompCp[] = {\n")
    for i in range(0, len(cps), 12):
        o("    " + ",".join("0x%X" % c for c in cps[i:i + 12]) + ",\n")
    o("};\ninline constexpr uint16_t kDecompOff[] = {\n")
    off, offs, pool = 0, [], []
    for c in cps:
        offs.append(off)
        pool.extend(decomp[c])
        off += len(decomp[c])
    offs.append(off)
    for i in range(0, len(offs), 16):
        o("    " + ",".join(str(x) for x in offs[i:i + 16]) + ",\n")
    o("};\ninline constexpr uint32_t kDecompPool[] = {\n")
    for i in range(0, len(pool), 12):
        o("    " + ",".join("0x%X" % c for c in pool[i:i + 12]) + ",\n")
    o("};\n\n// Primary composites, sorted by (first, second): first + second composes to the third.\n")
    o("struct Pair { uint32_t first, second, composite; };\ninline constexpr Pair kPairs[] = {\n")
    for i in range(0, len(pairs), 4):
        o("    " + " ".join("{0x%X,0x%X,0x%X}," % p for p in pairs[i:i + 4]) + "\n")
    o("};\n\n}  // namespace eklow::text::unicode\n")
    if hf:
        sys.stderr.write("left out for tokenizers %s: %d combining classes, %d decompositions, %d compositions\n"
                         % (hf[0], len(dropped["ccc"]), len(dropped["decomp"]), len(dropped["pairs"])))
        sys.stderr.write("  first left out: ccc %s, decomp %s\n" % ([hex(c) for c in dropped["ccc"][:6]], [hex(c) for c in dropped["decomp"][:6]]))


if __name__ == "__main__":
    main()
