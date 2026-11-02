"""Check eklow's Qwen3.6 tool-call grammar (eklow-grammar-test-qwen --lines) against the chat template and tokenizer.

Usage: python3 tools/ref/qwen_grammar_compare.py EKLOW_CHAT EKLOW_GRAMMAR_TEST GGUF MODEL_DIR VARIANT [--random N] [--mutations M] [--seed S]

Valid calls: the random assistant turns of qwen_parse_compare.py (reasoning, content, 1-3 calls with arguments of every
JSON kind), rendered by the template port and tokenized by Hugging Face (the text after the generation prompt, added
tokens only where the template wrote them), with the tools the request would carry (each key declared, some required,
null written only where the schema allows it). Every token of every call must be allowed by the grammar, as the
tokenizer splits it; tokens outside calls are not constrained.

Mutated calls: one call of a turn is written again by this script (not the template), the way the template writes it,
with one change whose first bad byte is known by construction: a tool name or parameter name altered, one added at
the end of a name, a parameter written twice, a required parameter left out, a value of the wrong JSON form for its
declared kind (the form decides the byte), an enum member with something added, a "</parameter>" in a string without a
newline before it, text before the call's first line or after its last. The call is tokenized by Hugging Face's
tokenizer and the grammar must accept every token before the one holding the bad byte (or, when the fault is the end of
the call, the close token) and refuse that one. Exits 1 on any difference.
"""

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys

from tokenizers import Tokenizer

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pc", os.path.join(here, "qwen_parse_compare.py"))
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)

S, I, N, B, O, A, Z = 1, 2, 4, 8, 16, 32, 64
NAMES = {"string": S, "str": S, "text": S, "integer": I, "int": I, "number": N, "float": N, "boolean": B, "bool": B,
         "object": O, "dict": O, "array": A, "list": A, "null": Z}


def kinds_of(schema):
    def of_type(t):
        ts = t if isinstance(t, list) else [t]
        return sum(NAMES.get(str(x).lower(), 0) for x in ts) if t is not None else 0
    k = of_type(schema.get("type"))
    if not k:
        for b in schema.get("anyOf", []):
            k |= of_type(b.get("type"))
    return k or S


def byte_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for x in range(256):
        if x not in bs:
            bs.append(x)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def free_text(schema):
    """Whether the grammar writes this parameter as free text (a string kind and no enum of scalars it can honour)."""
    return bool(kinds_of(schema) & S) and not honoured_enum(schema)


def honoured_enum(schema):
    e = schema.get("enum")
    if not e:
        return None
    texts = []
    for m in e:
        if isinstance(m, (dict, list)):
            return None
        t = m if isinstance(m, str) else "null" if m is None else "true" if m is True else "false" if m is False else json.dumps(m)
        if "\n" in t or "</parameter>" in t:
            return None
        texts.append(pc.NFC(t))  # the tokenizer (and the grammar) see NFC
    return texts


def prepare(turns):
    """The turns' tools, made declarable: every key declared, some required, null only where it can be written."""
    rng = random.Random(1)
    out = []
    for t, tools in turns:
        calls = t.get("tool_calls", [])
        by_name = {x["function"]["name"]: x for x in tools}
        ok = True
        for c in calls:
            fn = c["function"]
            tool = by_name[fn["name"]]
            props = tool["function"]["parameters"]["properties"]
            for k, v in fn["arguments"].items():
                if isinstance(v, str) and "</parameter>" in v:
                    ok = False  # the grammar writes that only as the end of the value
                if k not in props:
                    props[k] = {"type": "string"}
                if v is None and not kinds_of(props[k]) & (Z | S):
                    props[k] = {"type": "null"}
            if rng.random() < 0.5:
                tool["function"]["parameters"]["required"] = [k for k in fn["arguments"] if rng.random() < 0.6]
        out.append((t, tools, ok))
    return out


def body_text(name, blocks):
    return "\n<function=" + name + ">\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in blocks) + "</function>\n"


def value_text(v):
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def first_bad(text, candidates):
    """The first byte of text (a str, bytes in UTF-8) that is not a prefix of any of the candidates, else None."""
    tb = text.encode()
    cb = [c.encode() for c in candidates]
    for i in range(len(tb)):
        if not any(c.startswith(tb[:i + 1]) for c in cb):
            return i
    return None


# Wrong forms of a value for a kind mask, with the offset of the first bad byte in the value's text (the text is
# followed by "\n</parameter>\n"). Each applies only where the mask makes it wrong at that byte.
def bad_values(mask):
    out = [("a", 0)]
    if not mask & (I | N):
        out += [("5", 0), ("-1", 0)]
    if mask & (I | N) and not mask & N:
        out += [("5.5", 1), ("1e5", 1), ("05", 1), ("-", 1), ("5x", 1), ("0.0", 1)]
    if mask & N:
        out += [("1e", 2), (".5", 0), ("+1", 0), ("1.", 2), ("0x10", 1), ("1e+", 3), ("-.5", 1), ("00", 1)]
    if mask & B:
        out += [("True", 0), ("tru", 3), ("truee", 4), ("yes", 0), ("fals", 4), ("falsy", 4)]
    else:
        out += [("true", 0), ("false", 0)]
    if mask & Z:
        out += [("nil", 1), ("None", 0), ("nul", 3)]
    else:
        out += [("null", 0)]
    if mask & O:
        out += [("{a: 1}", 1), ('{"a" 1}', 5), ('{"a": }', 6), ('{"a": 1,}', 8), ("{'a': 1}", 1), ('{"a": 1} x', 8), ('{"a": tru}', 9), ('{"a": 01}', 7), ('{"a": "x\ty"}', 8)]
    else:
        out += [("{}", 0), ('{"a": 1}', 0)]
    if mask & A:
        out += [("[1,]", 3), ("[1 2]", 3), ("[,1]", 1), ("[1", 3), ("[1]]", 3), ('["a\\x"]', 4), ("[nul]", 4), ('["a" "b"]', 5)]
    else:
        out += [("[]", 0), ("[1]", 0)]
    return out


def make_mutation(rng, tool, call, plain_names, tools_by_name):
    """(mutated body, byte offset of the first bad byte or None for the close token, kind) or None."""
    name = call["function"]["name"]
    blocks = [(k, value_text(v)) for k, v in call["function"]["arguments"].items()]
    props = tool["function"]["parameters"]["properties"]
    required = tool["function"]["parameters"].get("required", [])
    declared = list(props)
    names = [t["function"]["name"] for t in tools_by_name.values()]
    pre = len("\n<function=")
    kind = rng.choice(["name_char", "name_extra", "param_char", "dup_param", "missing_required", "bad_value", "closing_tag", "enum", "tail", "head", "no_final_newline", "unknown_tool"])

    def unused_candidates(written):
        cands = [f"<parameter={k}>\n" for k in declared if k not in written]
        if all(r in written for r in required):
            cands.append("</function>\n")
        return cands

    if kind == "name_char":
        j = rng.randrange(len(name))
        for x in "xqz!":
            m = name[:j] + x + name[j + 1:]
            if not any(n.startswith(m[:j + 1]) for n in names) and name[j] != x:
                return body_text(m, blocks), len(name[:j].encode()) + pre, kind
        return None
    if kind == "name_extra":
        for x in "xqz!":
            if not any(n.startswith(name + x) for n in names):
                return "\n<function=" + name + x + ">\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in blocks) + "</function>\n", len(name.encode()) + pre, kind
        return None
    if kind == "unknown_tool":
        return body_text("nonexistent_tool", blocks), pre, kind
    if kind == "param_char" and blocks:
        i = rng.randrange(len(blocks))
        k = blocks[i][0]
        j = rng.randrange(len(k))
        written = [b[0] for b in blocks[:i]]
        for x in "xqz!":
            m = k[:j] + x + k[j + 1:]
            if k[j] == x:
                continue
            bad = first_bad("<parameter=" + m + ">\n", unused_candidates(written))
            if bad is not None and bad == len("<parameter=" + k[:j]):
                nb = blocks[:i] + [(m, blocks[i][1])] + blocks[i + 1:]
                prefix = "\n<function=" + name + ">\n" + "".join(f"<parameter={a}>\n{b}\n</parameter>\n" for a, b in blocks[:i])
                return body_text(name, nb), len(prefix.encode()) + bad, kind
        return None
    if kind == "dup_param" and blocks:
        i = rng.randrange(len(blocks))
        nb = blocks[:i + 1] + [blocks[i]] + blocks[i + 1:]
        written = [b[0] for b in blocks[:i + 1]]
        bad = first_bad(f"<parameter={blocks[i][0]}>\n", unused_candidates(written))
        if bad is None:
            return None
        prefix = "\n<function=" + name + ">\n" + "".join(f"<parameter={a}>\n{b}\n</parameter>\n" for a, b in blocks[:i + 1])
        return body_text(name, nb), len(prefix.encode()) + bad, kind
    if kind == "missing_required" and blocks and blocks[-1][0] in required:
        nb = blocks[:-1]
        written = [b[0] for b in nb]
        bad = first_bad("</function>\n", unused_candidates(written))
        if bad is None:
            return None
        prefix = "\n<function=" + name + ">\n" + "".join(f"<parameter={a}>\n{b}\n</parameter>\n" for a, b in nb)
        return body_text(name, nb), len(prefix.encode()) + bad, kind
    if kind == "bad_value" and blocks:
        i = rng.randrange(len(blocks))
        k, v = blocks[i]
        schema = props[k]
        if free_text(schema) or honoured_enum(schema):
            return None
        mask = kinds_of(schema)
        cands = bad_values(mask)
        if not cands:
            return None
        text, off = rng.choice(cands)
        nb = blocks[:i] + [(k, text)] + blocks[i + 1:]
        prefix = "\n<function=" + name + ">\n" + "".join(f"<parameter={a}>\n{b}\n</parameter>\n" for a, b in blocks[:i]) + f"<parameter={k}>\n"
        return body_text(name, nb), len(prefix.encode()) + off, kind
    if kind == "closing_tag" and blocks:
        i = rng.randrange(len(blocks))
        k, v = blocks[i]
        if not free_text(props[k]):
            return None
        text = rng.choice(["ab</parameter>cd", "x</parameter>", "</parameter>"])
        # The tag may follow a newline only at the very start, where the header's newline counts: avoid that.
        j = text.index("</parameter>")
        if j == 0:
            text = "q" + text
            j = 1
        nb = blocks[:i] + [(k, text)] + blocks[i + 1:]
        prefix = "\n<function=" + name + ">\n" + "".join(f"<parameter={a}>\n{b}\n</parameter>\n" for a, b in blocks[:i]) + f"<parameter={k}>\n"
        return body_text(name, nb), len(prefix.encode()) + j + len("</parameter>") - 1, kind
    if kind == "enum" and blocks:
        i = rng.randrange(len(blocks))
        k, v = blocks[i]
        texts = honoured_enum(props[k])
        if not texts:
            return None
        member = rng.choice(texts)
        for x in "ZQ#":
            text = member + x
            if not any(t.startswith(text) for t in texts):
                nb = blocks[:i] + [(k, text)] + blocks[i + 1:]
                prefix = "\n<function=" + name + ">\n" + "".join(f"<parameter={a}>\n{b}\n</parameter>\n" for a, b in blocks[:i]) + f"<parameter={k}>\n"
                return body_text(name, nb), len(prefix.encode()) + len(member.encode()), kind
        return None
    full = body_text(name, blocks)
    if kind == "tail":
        return full + "x", len(full.encode()), kind
    if kind == "head":
        return full[1:], 0, kind
    if kind == "no_final_newline":
        return full[:-1], None, kind
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("eklow_grammar_test")
    ap.add_argument("gguf")
    ap.add_argument("model_dir")
    ap.add_argument("variant", choices=["stock", "ornith"])
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--mutations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()

    turns, runs, skipped = pc.build_runs(a)
    prepared = prepare([(t, tools) for t, tools in turns])
    spec_json = json.load(open(f"{a.model_dir}/tokenizer.json", encoding="utf-8"))
    added = {t["content"]: t["id"] for t in spec_json["added_tokens"]}
    spec_json["added_tokens"] = []
    plain = Tokenizer.from_str(json.dumps(spec_json))
    bm = byte_map()
    call_open, call_close = added["<tool_call>"], added["</tool_call>"]

    def run(lines):
        out = subprocess.run([a.eklow_grammar_test, "--model", a.gguf, "--lines"], input=("\n".join(lines) + "\n").encode(), capture_output=True, check=True).stdout.decode()
        r = out.split("\n")[:-1]
        if len(r) != len(lines):
            sys.exit("eklow-grammar-test-qwen returned the wrong number of lines")
        return r

    # Valid calls.
    lines, idx = [], []
    n_calls = n_tokens = 0
    for think, i, ids, _ in runs:
        t, tools, ok = prepared[i]
        if not ok or not t.get("tool_calls"):
            continue
        lines.append(json.dumps(tools, ensure_ascii=False) + "\t" + " ".join(map(str, ids)))
        n_calls += len(t["tool_calls"])
        n_tokens += len(ids)
    got = run(lines)
    bad_valid = [(l, g) for l, g in zip(lines, got) if g != "ok"]
    for l, g in bad_valid[:5]:
        print("VALID REFUSED", g, l[:300])
    print(f"{a.variant}: {len(lines)} turns with calls ({n_calls} calls, {n_tokens} tokens): {len(lines) - len(bad_valid)} accepted token by token, {len(bad_valid)} refused")

    # Mutated calls.
    rng = random.Random(a.seed + 1)
    cases = []
    counts = {}
    pool = [(t, tools) for t, tools, ok in prepared if ok and t.get("tool_calls")]
    attempts = 0
    while len(cases) < a.mutations and attempts < a.mutations * 20:
        attempts += 1
        t, tools = rng.choice(pool)
        t = {**t, "tool_calls": [{"function": {"name": pc.NFC(c["function"]["name"]), "arguments": pc.nfc_value(c["function"]["arguments"])}} for c in t["tool_calls"]]}
        by_name = {x["function"]["name"]: x for x in tools}
        by_name = {pc.NFC(k): v for k, v in by_name.items()}
        call = rng.choice(t["tool_calls"])
        tool = by_name[call["function"]["name"]]
        # NFC keys of the declared properties, as the grammar sees the names the request carries (it reads them as sent).
        m = make_mutation(rng, tool, call, None, by_name)
        if m is None:
            continue
        body, bad, kind = m
        ids = [call_open] + plain.encode(body, add_special_tokens=False).ids + [call_close]
        # Byte spans of the body's tokens.
        pos = 0
        expect = len(ids) - 1
        if bad is not None:
            for j, tid in enumerate(ids[1:-1], start=1):
                n = len(bytes(bm[ch] for ch in plain.id_to_token(tid)))
                if pos <= bad < pos + n:
                    expect = j
                    break
                pos += n
        cases.append((json.dumps(tools, ensure_ascii=False) + "\t" + " ".join(map(str, ids)), expect, kind, body))
    got = run([c[0] for c in cases])
    bad_mut = 0
    for (line, expect, kind, body), g in zip(cases, got):
        counts[kind] = counts.get(kind, 0) + 1
        if g != f"refused {expect}":
            bad_mut += 1
            if bad_mut <= 8:
                print(f"MUTATION {kind}: expected refused {expect}, got {g}\n  body {body[:300]!r}")
    print(f"{a.variant}: {len(cases)} mutated calls ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))}): {len(cases) - bad_mut} refused at the first bad token, {bad_mut} differences")
    sys.exit(1 if bad_valid or bad_mut else 0)


if __name__ == "__main__":
    main()
