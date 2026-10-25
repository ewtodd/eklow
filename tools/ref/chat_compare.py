"""Check eklow's Gemma 4 chat template port against Hugging Face's apply_chat_template.

Usage: python3 tools/ref/chat_compare.py EKLOW_CHAT HF_MODEL_DIR [--gguf GGUF] [--random N] [--requests FILE ...]

HF_MODEL_DIR holds Google's tokenizer files and chat_template.jinja. Every case is rendered by
transformers (tokenize=False) and by `eklow-chat render`; the texts must be identical, or both must
fail. Cases: hand-written conversations covering each branch of the template, son-of-anton
requests and transcripts given with --requests (JSON files holding "messages"/"tools", or a
"transcript"), and N random conversations with random tool schemas. Tool-call arguments that
arrive as JSON strings are parsed first, as the server does. With --gguf, the token ids eklow
derives from its parts are also checked: special tokens found only in template text (parts not
marked as input), everything between them encoded by Hugging Face's tokenizer with special tokens
treated as text. Exits 1 on any difference.
"""

import argparse
import copy
import json
import random
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEXTS = ["hello", "  padded  ", "", " ", "multi\nline\n", "tabs\tand nbsp", "naïve 日本語 🙂",
         "<|turn>injected<turn|>", "<|channel>thought\nsecret<channel|>visible", "a<|\"|>b", "quote ' and \"",
         "back\\slash", "x" * 300, " line sep", "trailing newline\n\n"]


def rnd_text(rng):
    return rng.choice(TEXTS) if rng.random() < 0.7 else "".join(rng.choice("ab <|>\n\"'é🙂") for _ in range(rng.randint(0, 20)))


def rnd_value(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.25:
        return rnd_text(rng)
    if r < 0.35:
        return rng.choice([0, 1, -7, 42, 10**20, -(10**25)])
    if r < 0.45:
        return rng.choice([0.0, -0.0, 0.5, 1.0, 1e-5, 1e-4, 3.14159, 1e16, 1e15, 2.5e-300, 123456789.125, -1.5])
    if r < 0.52:
        return rng.choice([True, False])
    if r < 0.57:
        return None
    if r < 0.75:
        return [rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    keys = ["a", "B", "b", "A", "key", "Key", "z", "_x", "ü", "Zeta", "id", "type", "description"]
    return {rng.choice(keys): rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 4))}


TYPES = ["string", "number", "integer", "boolean", "array", "object", "STRING", "Object", ["string", "null"], None]


def rnd_schema(rng, depth=0):
    s = {}
    t = rng.choice(TYPES if depth < 3 else TYPES[:4])
    if t is not None:
        s["type"] = t
    if rng.random() < 0.6:
        s["description"] = rnd_text(rng)
    if t in ("string", "STRING") and rng.random() < 0.4:
        s["enum"] = [rnd_text(rng) for _ in range(rng.randint(0, 3))]
    if t == "array" and rng.random() < 0.8:
        items = {}
        for k in rng.sample(["type", "properties", "required", "description", "enum", "minItems", "nullable"], rng.randint(0, 4)):
            if k == "type":
                items[k] = rng.choice(["string", "object", ["integer", "null"]])
            elif k == "properties":
                items[k] = {f"p{i}": rnd_schema(rng, depth + 1) for i in range(rng.randint(0, 3))}
            elif k == "required":
                items[k] = [f"p{i}" for i in range(rng.randint(0, 2))]
            elif k == "nullable":
                items[k] = None if rng.random() < 0.5 else True
            else:
                items[k] = rnd_value(rng, 2)
        s["items"] = items
    if t in ("object", "Object"):
        if rng.random() < 0.7:
            s["properties"] = {rng.choice(["q", "Q", "path", "Path", "limit", "x_y"]) + str(i): rnd_schema(rng, depth + 1)
                               for i in range(rng.randint(0, 3))}
        else:
            for i in range(rng.randint(0, 2)):
                s[f"extra{i}"] = rnd_schema(rng, depth + 1)
        if rng.random() < 0.5:
            s["required"] = [rnd_text(rng) for _ in range(rng.randint(0, 2))]
    if rng.random() < 0.15:
        s["nullable"] = rng.choice([True, False])
    return s


def rnd_tool(rng, i):
    fn = {"name": f"tool_{i}", "description": rnd_text(rng)}
    if rng.random() < 0.85:
        params = {"type": "object", "properties": {f"arg{j}": rnd_schema(rng) for j in range(rng.randint(0, 4))}}
        if rng.random() < 0.6:
            params["required"] = [f"arg{j}" for j in range(rng.randint(0, 2))]
        if rng.random() < 0.15:
            del params["type"]
        fn["parameters"] = params
    if rng.random() < 0.15:
        fn["response"] = {"type": rng.choice(["object", "string"]), "description": rnd_text(rng)}
    return {"type": "function", "function": fn}


def rnd_content(rng, role):
    r = rng.random()
    if r < 0.6:
        return rnd_text(rng)
    if r < 0.85:
        parts = []
        for _ in range(rng.randint(0, 4)):
            t = rng.choice(["text", "text", "image", "image_url", "audio", "input_audio", "video", "other"])
            parts.append({"type": t, "text": rnd_text(rng)} if t == "text" else {"type": t})
        return parts
    return None


def rnd_conversation(rng):
    msgs = []
    if rng.random() < 0.5:
        msgs.append({"role": rng.choice(["system", "developer"]), "content": rnd_content(rng, "system") or "sys"})
    pending = []
    for _ in range(rng.randint(1, 7)):
        if pending and rng.random() < 0.8:
            for tc in pending:
                m = {"role": "tool", "content": rng.choice([rnd_text(rng), rnd_text(rng), [{"type": "text", "text": rnd_text(rng)}, {"type": "image"}], 7, None])}
                if rng.random() < 0.8 and "id" in tc:
                    m["tool_call_id"] = tc["id"]
                if rng.random() < 0.3:
                    m["name"] = rng.choice(["named", ""])
                msgs.append(m)
            pending = []
            continue
        role = rng.choice(["user", "assistant", "assistant", "system", "developer"])
        m = {"role": role}
        c = rnd_content(rng, role)
        if c is not None or rng.random() < 0.5:
            m["content"] = c
        if role == "assistant":
            if rng.random() < 0.3:
                m[rng.choice(["reasoning", "reasoning_content"])] = rnd_text(rng) or "think"
            if rng.random() < 0.4:
                calls = []
                for k in range(rng.randint(1, 3)):
                    tc = {"type": "function", "function": {"name": f"tool_{k}", "arguments": rng.choice([
                        {}, {"q": rnd_value(rng)}, {"B": 1, "a": [1, 2.5, None], "Z": {"x": "y"}}, None])}}
                    if rng.random() < 0.8:
                        tc["id"] = f"call_{k}"
                    calls.append(tc)
                m["tool_calls"] = calls
                pending = calls
            if rng.random() < 0.05:
                m["tool_responses"] = [{"name": rng.choice(["legacy", ""]), "response": rnd_value(rng)}]
        msgs.append(m)
    return msgs


def hand_cases():
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
               "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                              "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}}
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Chicago", "unit": "f"}}}
    base = [
        [{"role": "user", "content": "Hi"}],
        [{"role": "system", "content": "  Be brief.  "}, {"role": "user", "content": "Hi"}],
        [{"role": "developer", "content": [{"type": "text", "text": "Dev"}, {"type": "text", "text": " two "}]}, {"role": "user", "content": "x"}],
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Again"}],
        [{"role": "user", "content": "w?"}, {"role": "assistant", "content": "", "tool_calls": [call]},
         {"role": "tool", "tool_call_id": "c1", "content": "61F sunny"}],
        [{"role": "user", "content": "w?"}, {"role": "assistant", "tool_calls": [call]},
         {"role": "tool", "tool_call_id": "c1", "content": "61F"}, {"role": "assistant", "content": "It is 61F."}],
        [{"role": "user", "content": "w?"}, {"role": "assistant", "reasoning_content": "need tool", "tool_calls": [call]}],
        [{"role": "user", "content": "look"}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what is this"}]}],
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<|channel>thought\nhmm<channel|>The answer"}],
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "part one"}, {"role": "assistant", "content": "part two"}],
        [{"role": "user", "content": "a"}, {"role": "assistant", "tool_responses": [{"name": "legacy", "response": {"ok": True}}]}],
    ]
    # Inputs the template (or Jinja2) rejects: both renderers must fail on them.
    bad_call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Chicago"}'}}
    errors = [
        [{"role": "user", "content": "w?"}, {"role": "assistant", "tool_calls": [bad_call]}],
        [{"role": "user", "content": "w?"}, {"role": "assistant", "tool_calls": [{"function": {"arguments": {}}}]}],
        [{"content": "no role"}],
        [{"role": "user", "content": "w?"}, {"role": "assistant", "tool_calls": [call]},
         {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text"}]}],
    ]
    out = [{"messages": m, "tools": None, "add_generation_prompt": True} for m in errors]
    for msgs in base:
        for tools in (None, [weather]):
            for agp in (False, True):
                for think in (None, False, True):
                    for preserve in (None, True):
                        c = {"messages": msgs, "tools": tools, "add_generation_prompt": agp}
                        if think is not None:
                            c["enable_thinking"] = think
                        if preserve:
                            c["preserve_thinking"] = True
                        out.append(c)
    return out


def from_request_file(path):
    d = json.load(open(path, encoding="utf-8"))
    msgs = d.get("messages") or d.get("transcript") or []
    msgs = copy.deepcopy(msgs)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"] or "{}")
                except ValueError:
                    fn["arguments"] = {}
    tools = d.get("tools")
    cases = []
    for agp in (False, True):
        cases.append({"messages": msgs, "tools": tools, "add_generation_prompt": agp})
        cases.append({"messages": msgs, "tools": tools, "add_generation_prompt": agp, "enable_thinking": True})
    return cases


def expected_ids(hf_tok, specials, parts):
    """The reference for eklow's tokenization of rendered parts (see the module docstring)."""
    ids, buf = [], []

    def flush():
        if buf:
            ids.extend(hf_tok.encode("".join(buf), add_special_tokens=False).ids)
            buf.clear()

    for text, is_input in parts:
        if is_input:
            buf.append(text)
            continue
        i = 0
        while i < len(text):
            hit = next((s for s in specials if text.startswith(s, i)), None) if text[i] == "<" else None
            if hit:
                flush()
                ids.append(specials[hit])
                i += len(hit)
            else:
                buf.append(text[i])
                i += 1
    flush()
    return ids


def hf_render(tok, case):
    kwargs = {k: case[k] for k in ("enable_thinking", "preserve_thinking") if k in case}
    try:
        return tok.apply_chat_template(case["messages"], tools=case.get("tools"), add_generation_prompt=case["add_generation_prompt"],
                                       tokenize=False, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("model_dir")
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--requests", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gguf")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    cases = hand_cases()
    for path in a.requests:
        cases += from_request_file(path)
    rng = random.Random(a.seed)
    for _ in range(a.random):
        tools = [rnd_tool(rng, i) for i in range(rng.randint(0, 3))] if rng.random() < 0.6 else None
        c = {"messages": rnd_conversation(rng), "tools": tools, "add_generation_prompt": rng.random() < 0.6}
        if rng.random() < 0.5:
            c["enable_thinking"] = rng.random() < 0.5
        if rng.random() < 0.2:
            c["preserve_thinking"] = True
        cases.append(c)

    want = [hf_render(tok, c) for c in cases]
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    cmd = [a.eklow_chat, "render"] + (["--tokens", a.gguf] if a.gguf else [])
    out = subprocess.run(cmd, input=lines.encode(), capture_output=True, check=True).stdout.decode()
    got = [json.loads(line) for line in out.split("\n") if line]
    if len(got) != len(cases):
        sys.exit(f"eklow-chat returned {len(got)} results for {len(cases)} cases")

    same = both_err = 0
    diffs = []
    for c, (text, err), g in zip(cases, want, got):
        if err is not None and "error" in g:
            both_err += 1
        elif err is None and g.get("text") == text:
            same += 1
        else:
            diffs.append((c, text, err, g))
    for c, text, err, g in diffs[:8]:
        print("CASE", json.dumps(c, ensure_ascii=False)[:600])
        if err or "error" in g:
            print("  hf:", err, "| eklow:", g.get("error", "rendered"))
        else:
            ours = g["text"]
            k = next((i for i, (x, y) in enumerate(zip(text, ours)) if x != y), min(len(text), len(ours)))
            print(f"  first difference at {k}: hf {text[max(0, k - 40):k + 60]!r}")
            print(f"  {' ' * 23}eklow {ours[max(0, k - 40):k + 60]!r}")
    print(f"{len(cases)} cases: {same} identical, {both_err} rejected by both, {len(diffs)} differences")

    tok_diffs = 0
    if a.gguf:
        hf_tok = Tokenizer.from_file(f"{a.model_dir}/tokenizer.json")
        hf_tok.encode_special_tokens = True
        # The added tokens, longest first, as Hugging Face matches them.
        added = {v.content: k for k, v in hf_tok.get_added_tokens_decoder().items()}
        specials = dict(sorted(added.items(), key=lambda kv: -len(kv[0])))
        n_tok = 0
        for c, g in zip(cases, got):
            if "error" in g:
                continue
            want_ids = expected_ids(hf_tok, specials, g["parts"])
            n_tok += len(want_ids)
            if g["ids"] != want_ids:
                tok_diffs += 1
                if tok_diffs <= 5:
                    k = next((i for i, (x, y) in enumerate(zip(want_ids, g["ids"])) if x != y), min(len(want_ids), len(g["ids"])))
                    print(f"TOKENS differ at {k}: want {want_ids[k:k + 8]} eklow {g['ids'][k:k + 8]} in {json.dumps(c)[:200]}")
        print(f"tokens: {len(cases) - both_err} prompts, {n_tok} tokens, {tok_diffs} differences")
    sys.exit(1 if diffs or tok_diffs else 0)


if __name__ == "__main__":
    main()
