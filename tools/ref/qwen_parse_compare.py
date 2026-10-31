"""Round-trip check of eklow's Qwen3.6 output parser through the chat template.

Usage: python3 tools/ref/qwen_parse_compare.py EKLOW_CHAT EKLOW_PARSE GGUF MODEL_DIR VARIANT [--random N] [--seed S]

Random assistant turns (reasoning, 1-3 tool calls with arguments of every JSON kind, content) are rendered after a
user turn by the template port (identical to Hugging Face's, see qwen_chat_compare.py); the text after the generation
prompt of that user turn (which ends "<think>\\n", or with thinking off, with the empty think block) is what the model
would have generated, tokenized by Hugging Face's tokenizer (MODEL_DIR's tokenizer.json) with the added tokens only in
template text. eklow-parse reads them back one token at a time, with the tool schemas
the request would carry, and must recover the reasoning, the content (as the template trims them), and each call's
name and arguments (compared as values, JSON types included: 1 is not 1.0, "5" is not 5). Turns without reasoning are
also run with thinking off. A turn is checked against the template's own transformation, so content that the template
would split at "</think>" is not generated without reasoning_content, and string values holding "</parameter>" followed
by "</function>" or another "<parameter=" are left out (no parser can tell them from the end of the value). Exits 1
on any difference.
"""

import argparse
import json
import random
import subprocess
import sys

from tokenizers import Tokenizer, normalizers

NFC = normalizers.NFC().normalize_str


def nfc_value(v):
    """The tokenizer normalizes text to NFC, so what the model writes (and the parser reads) is NFC."""
    if isinstance(v, str):
        return NFC(v)
    if isinstance(v, list):
        return [nfc_value(x) for x in v]
    if isinstance(v, dict):
        return {NFC(k): nfc_value(x) for k, x in v.items()}
    return v

TEXTS = ["hello", "  padded  ", "multi\nline", "tabs\tand nbsp ", "naïve 日本語 🙂", "<|im_end|>not special",
         "a<tool_call>b</tool_call>c", "quote ' and \"", "back\\slash", "x" * 120, "trailing newline\n", "{braces}, [brackets]: colons",
         "\U0001F9EA\U0001FAE0", "é", "é", "<function=f>", "<parameter=k>", "</function>", "a</parameter>b", "</parameter>x",
         "\n\nleading and trailing\n\n", "null", "true", "123", "1.5", "[1, 2]", '{"a": 1}', "", " ", "\n", "<think>", "</think>",
         "<tool_response>x</tool_response>", "```py\nprint(1)\n```", "line1\r\nline2", "　", "10 > 5 < 7", "<|im_start|>"]
KEYS = ["a", "B", "path", "query", "limit", "x_y", "Zeta", "日本", "under_score", "with space", "k-1", "é"]
NAMES = ["web_search", "read_file", "cronjob", "f", "get_weather", "名前", "tool.with.dots", "tool-with-dashes"]


def rnd_text(rng):
    return rng.choice(TEXTS) if rng.random() < 0.7 else "".join(rng.choice("ab ,:{}[]\n\"'é🙂<>/=") for _ in range(rng.randint(1, 20)))


def rnd_value(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.3:
        return rnd_text(rng)
    if r < 0.4:
        return rng.choice([0, 1, -7, 42, 10**20, -(10**25)])
    if r < 0.5:
        return rng.choice([0.0, 0.5, 1.0, 1e-5, 1e-4, 3.14159, 1e16, 1e15, 2.5e-300, 123456789.125, -1.5])
    if r < 0.57:
        return rng.choice([True, False])
    if r < 0.62:
        return None
    if r < 0.8:
        return [rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    return {rng.choice(KEYS): rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


def schema_for(rng, v):
    """A parameter schema the value conforms to, or None (the key is not declared)."""
    r = rng.random()
    if r < 0.08:
        return None
    if isinstance(v, bool):
        return rng.choice([{"type": "boolean"}, {"type": ["boolean", "null"]}, {"anyOf": [{"type": "boolean"}, {"type": "null"}]}])
    if isinstance(v, int):
        return rng.choice([{"type": "integer"}, {"type": "number"}, {"type": ["integer", "null"]}, {"type": "int"}, {"type": "INTEGER"}])
    if isinstance(v, float):
        return rng.choice([{"type": "number"}, {"type": ["number", "null"]}, {"type": "float"}])
    if v is None:
        return rng.choice([{"type": "null"}, {"type": ["string", "null"]}, {"type": ["integer", "null"]}, {"type": "object"}, {"type": "array"}, {"type": "number"}, {"type": "boolean"}])
    if isinstance(v, list):
        return rng.choice([{"type": "array", "items": {"type": "string"}}, {"type": "array"}, {"type": ["array", "null"]}, {"type": "list"}])
    if isinstance(v, dict):
        return rng.choice([{"type": "object", "properties": {}}, {"type": "object"}, {"type": ["object", "null"]}, {"type": "dict"}])
    return rng.choice([{"type": "string"}, {"type": "string", "description": "d"}, {"type": "string", "enum": [v, "other"]}, {"type": "STRING"}, {"type": "str"}, {}, {"description": "no type"}])


def ambiguous(v):
    """A string value no parser can tell from the end of the value."""
    if isinstance(v, str):
        i = v.find("</parameter>")
        while i >= 0:
            rest = v[i + 12:].lstrip(" \t\r\n")
            if rest.startswith("<parameter=") or rest.startswith("</function>") or rest == "":
                return True
            i = v.find("</parameter>", i + 1)
        return False
    if isinstance(v, list):
        return any(ambiguous(x) for x in v)
    if isinstance(v, dict):
        return any(ambiguous(x) for x in v.values())
    return False


def rnd_turn(rng):
    m = {"role": "assistant"}
    tools = []
    if rng.random() < 0.5:
        m["reasoning_content"] = rnd_text(rng) or "thinking"
    if rng.random() < 0.6:
        calls = []
        for k, name in enumerate(rng.sample(NAMES, rng.randint(1, 3))):
            args = {}
            props = {}
            for _ in range(rng.randint(0, 4)):
                key = rng.choice(KEYS)
                v = rnd_value(rng)
                if ambiguous(v):
                    continue
                s = schema_for(rng, v)
                if isinstance(v, str) and v == "null":
                    # A string "null" is only a string where the schema says string and nothing else (convert_param).
                    s = {"type": "string"}
                elif s is None and not isinstance(v, str):
                    # Undeclared: the parser can only return text, so a declared kind is needed for the others.
                    s = {"type": "object"} if isinstance(v, dict) else {"type": "array"} if isinstance(v, list) else {"type": "boolean"} if isinstance(v, bool) else {"type": "number"} if isinstance(v, (int, float)) else {"type": ["string", "null"]}
                args[key] = v
                if s is None:
                    props.pop(key, None)
                else:
                    props[key] = s
            calls.append({"id": f"c{k}", "type": "function", "function": {"name": name, "arguments": args}})
            tools.append({"type": "function", "function": {"name": name, "description": "d", "parameters": {"type": "object", "properties": props}}})
        m["tool_calls"] = calls
    if rng.random() < 0.7 or "tool_calls" not in m:
        c = rnd_text(rng)
        if "reasoning_content" not in m:
            c = c.replace("</think>", "").replace("<think>", "")
        m["content"] = c
    return m, tools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("eklow_parse")
    ap.add_argument("gguf")
    ap.add_argument("model_dir")
    ap.add_argument("variant", choices=["stock", "ornith"])
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    turns = [rnd_turn(rng) for _ in range(a.random)]
    user = {"role": "user", "content": "go"}

    def render(cases):
        lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
        out = subprocess.run([a.eklow_chat, "render", "--variant", a.variant, "--tokens", a.gguf], input=lines.encode(), capture_output=True, check=True).stdout.decode()
        r = [json.loads(x) for x in out.split("\n") if x]
        if len(r) != len(cases):
            sys.exit("eklow-chat returned the wrong number of results")
        return r

    spec = json.load(open(f"{a.model_dir}/tokenizer.json", encoding="utf-8"))
    added = {t["content"]: t["id"] for t in spec["added_tokens"]}
    spec["added_tokens"] = []
    plain = Tokenizer.from_str(json.dumps(spec))
    specials = dict(sorted(added.items(), key=lambda kv: -len(kv[0])))

    def hf_ids(parts):
        ids, buf = [], []

        def flush():
            if buf:
                ids.extend(plain.encode("".join(buf), add_special_tokens=False).ids)
                buf.clear()

        for text, is_input in parts:
            if is_input:
                buf.append(text)
                continue
            i = 0
            while i < len(text):
                hit = next((x for x in specials if text.startswith(x, i)), None) if text[i] == "<" else None
                if hit:
                    flush()
                    ids.append(specials[hit])
                    i += len(hit)
                else:
                    buf.append(text[i])
                    i += 1
        flush()
        return ids

    prompts = render([{"messages": [user], "add_generation_prompt": True, "enable_thinking": t} for t in (True, False)])
    prompt_text = {True: prompts[0]["text"], False: prompts[1]["text"]}
    full = render([{"messages": [user, t], "add_generation_prompt": False} for t, _ in turns])

    # (thinking, turn index, generated ids, tools); turns without reasoning are also run with thinking off.
    runs = []
    skipped = 0
    for i, ((t, tools), r) in enumerate(zip(turns, full)):
        modes = [True] + ([False] if not t.get("reasoning_content") else [])
        for think in modes:
            p = prompt_text[think]
            if not r["text"].startswith(p):
                skipped += 1  # the thinking-off prompt of a turn that has reasoning is not its prefix
                continue
            rest, pos = [], 0
            for text, is_input in r["parts"]:
                if pos + len(text) > len(p):
                    rest.append((text[max(0, len(p) - pos):], is_input))
                pos += len(text)
            runs.append((think, i, hf_ids(rest), tools))
    bad = 0
    n_calls = 0
    for think in (True, False):
        sel = [x for x in runs if x[0] == think]
        if not sel:
            continue
        inp = "\n".join(json.dumps(tools) + "\t" + " ".join(map(str, g)) for _, _, g, tools in sel) + "\n"
        out = subprocess.run([a.eklow_parse, a.gguf] + (["--thinking"] if think else []), input=inp.encode(), capture_output=True, check=True).stdout.decode()
        got = [json.loads(x) for x in out.split("\n") if x]
        if len(got) != len(sel):
            sys.exit("eklow-parse returned the wrong number of results")
        for (_, i, _, _), g in zip(sel, got):
            t = turns[i][0]
            want_calls = [{"name": NFC(c["function"]["name"]), "arguments": nfc_value(c["function"]["arguments"])} for c in t.get("tool_calls", [])]
            n_calls += len(want_calls)
            want = {"reasoning": NFC((t.get("reasoning_content") or "").strip()) if think else "",
                    "content": NFC(t.get("content", "").strip()), "tool_calls": want_calls}
            have = {"reasoning": g["reasoning"], "content": g["content"], "tool_calls": g["tool_calls"]}
            # Compare with JSON types (1 vs 1.0 vs true) kept apart: dumps distinguishes them.
            if json.dumps(want, ensure_ascii=False) != json.dumps(have, ensure_ascii=False) or g["errors"]:
                bad += 1
                if bad <= 6:
                    print(f"TURN (thinking={think})", json.dumps(t, ensure_ascii=False)[:500])
                    print("  want", json.dumps(want, ensure_ascii=False)[:500])
                    print("  got ", json.dumps(have, ensure_ascii=False)[:500], g["errors"])
    print(f"{a.variant}: {len(turns)} turns ({len(runs)} runs, {n_calls} tool calls; {skipped} skipped): {len(runs) - bad} recovered exactly, {bad} differences")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
