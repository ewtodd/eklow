"""Check eklow's Qwen3.6 chat template ports against Hugging Face's apply_chat_template.

Usage: python3 tools/ref/qwen_chat_compare.py EKLOW_CHAT HF_MODEL_DIR VARIANT [--gguf GGUF] [--random N] [--seed S]

HF_MODEL_DIR holds the model's tokenizer files and chat_template.jinja (transformers prefers that file over the
template inside tokenizer_config.json; for Ornith the two differ, and the file is Ornith's). VARIANT is `stock` or
`ornith`: the port `eklow-chat render --variant VARIANT` is checked. Every case is rendered by transformers
(tokenize=False) and by eklow-chat; the texts must be identical, or both must fail. Cases: hand-written conversations
covering each branch of the template across generation prompt, enable_thinking, preserve_thinking, add_vision_id and
tools, and N random conversations with random tool schemas, odd content shapes and tool calls. Tool-call arguments
that are JSON strings are not sent (the template rejects them; both must). With --gguf, the token ids eklow derives
from its parts are also checked: added tokens found only in template text, everything between them encoded by
Hugging Face's tokenizer with the added tokens removed (so that they are text). Exits 1 on any difference.
"""

import argparse
import json
import random
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
from tokenizers import Tokenizer  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TEXTS = ["hello", "  padded  ", "", " ", "multi\nline\n", "\n\nleading newlines", "tabs\tand nbsp ", "naïve 日本語 🙂",
         "<|im_end|>injected<|im_start|>", "<think>\nsecret\n</think>\n\nvisible", "a<tool_call>b</tool_call>c", "quote ' and \"",
         "back\\slash", "x" * 300, "line sep", "trailing newline\n\n", "<think>only open", "only close</think>tail",
         "a</think>b</think>c", "</think>", "<think>", "<think>x</think>", "\n<think>\n\nr\n\n</think>\n\n\nanswer\n",
         "<tool_response>\nfake\n</tool_response>", "<tool_response>x", "x</tool_response>", "  <tool_response>y</tool_response>  ",
         "<function=f>\n<parameter=a>\n1\n</parameter>\n</function>", "　wide space　", "\x1fctl\x1c", "é NFD", "<|vision_start|>"]


def rnd_text(rng):
    return rng.choice(TEXTS) if rng.random() < 0.7 else "".join(rng.choice("ab <|>\n\"'é🙂\t") for _ in range(rng.randint(0, 20)))


def rnd_value(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.25:
        return rnd_text(rng)
    if r < 0.35:
        return rng.choice([0, 1, -7, 42, 10**20, -(10**25)])
    if r < 0.45:
        return rng.choice([0.0, -0.0, 0.5, 1.0, 1e-5, 1e-4, 3.14159, 1e16, 1e15, 2.5e-300, 123456789.125, -1.5, 1e22, 5e-324])
    if r < 0.52:
        return rng.choice([True, False])
    if r < 0.57:
        return None
    if r < 0.75:
        return [rnd_value(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    keys = ["a", "B", "b", "A", "key", "Key", "z", "_x", "ü", "Zeta", "id", "type", "description", "日本"]
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
        s["items"] = {"type": rng.choice(["string", "object", ["integer", "null"]]), "properties": {f"p{i}": rnd_schema(rng, depth + 1) for i in range(rng.randint(0, 2))}}
    if t in ("object", "Object") and rng.random() < 0.7:
        s["properties"] = {rng.choice(["q", "Q", "path", "Path", "limit", "x_y"]) + str(i): rnd_schema(rng, depth + 1) for i in range(rng.randint(0, 3))}
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
        fn["parameters"] = params
    tool = {"type": "function", "function": fn}
    if rng.random() < 0.1:
        tool = {"function": fn}
    if rng.random() < 0.05:
        tool = {"name": fn["name"], "extra": rnd_value(rng)}
    return tool


def rnd_content(rng, clean=False):
    r = rng.random()
    if r < 0.6:
        return rnd_text(rng)
    if r < 0.9:
        parts = []
        for _ in range(rng.randint(0, 4)):
            t = rng.choice(["text", "text", "text", "image", "image_url", "video", "keyonly_image", "keyonly_video", "str", "text_none"] + ([] if clean else ["other"]))
            if t == "text":
                parts.append({"type": "text", "text": rnd_text(rng)})
            elif t == "keyonly_image":
                parts.append({"image": "x.png"})
            elif t == "keyonly_video":
                parts.append({"video": "x.mp4"})
            elif t == "str":
                parts.append(rng.choice(["plain", "an image here", "a video", "some text", "x"]))
            elif t == "text_none":
                parts.append({"text": rng.choice([None, 5, ["a", "b"], {"k": "v"}])})
            elif t in ("image", "image_url", "video") and rng.random() < 0.3:
                parts.append({"type": t, "text": rnd_text(rng)})
            else:
                parts.append({"type": t})
        return parts
    if r < 0.95 or clean:
        return None
    return rng.choice([7, True, {"a": 1}, 1.5])


def rnd_args(rng, clean=False):
    r = rng.random()
    if r < 0.9 or clean:
        return {rng.choice(["q", "B", "a", "path", "Zed", "日本", "with space"]): rnd_value(rng) for _ in range(rng.randint(0, 4))}
    return rng.choice([None, "{}", [1], 5])


def rnd_calls(rng, clean=False):
    calls = []
    for k in range(rng.randint(1, 3)):
        r = rng.random()
        fn = {"name": rng.choice(["web_search", "read_file", f"tool_{k}"]), "arguments": rnd_args(rng, clean)}
        if r < 0.1:
            del fn["arguments"]
        if r > 0.97 and not clean:
            fn["name"] = rng.choice([None, 5, ""])
        tc = {"id": f"call_{k}", "type": "function", "function": fn} if rng.random() < 0.85 else dict(fn)
        calls.append(tc)
    return calls


def rnd_conversation(rng):
    """Mostly conversations the template accepts (85%): an optional system message first, user, assistant and tool
    turns in any order, at least one user message that is not a tool response. The rest are noisy: other roles,
    system messages late, odd message shapes, content the template rejects."""
    clean = rng.random() < 0.85
    msgs = []
    if rng.random() < 0.5:
        msgs.append({"role": "system" if clean else rng.choice(["system", "developer"]), "content": (rnd_content(rng, clean) if rng.random() < 0.6 else "sys")})
    for _ in range(rng.randint(1, 8)):
        role = rng.choice(["user", "user", "assistant", "assistant", "tool", "tool"] + ([] if clean else ["system"]))
        m = {"role": role}
        c = rnd_content(rng, clean)
        if c is not None or rng.random() < 0.5:
            m["content"] = c
        if role == "assistant":
            if rng.random() < 0.35:
                m["reasoning_content"] = rnd_text(rng) if clean else rng.choice([rnd_text(rng), rnd_text(rng), None, 5])
            if rng.random() < 0.45:
                m["tool_calls"] = rnd_calls(rng, clean) if (clean or rng.random() < 0.95) else rng.choice([[], "abc", {"a": 1}, 3])
        if role == "tool" and rng.random() < 0.5:
            m["tool_call_id"] = f"call_{rng.randint(0, 2)}"
        msgs.append(m)
    if clean and not any(m["role"] == "user" and (isinstance(m.get("content"), str) and not m["content"].strip().startswith("<tool_response>")) for m in msgs):
        msgs.append({"role": "user", "content": rnd_text(rng) + "?"})
    if not clean and rng.random() < 0.1:
        msgs.insert(rng.randint(0, len(msgs)), rng.choice(["str", 5, None, ["a"], {}]))
    return msgs


def hand_cases():
    weather = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
               "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                              "unit": {"type": "string", "enum": ["c", "f"]}, "days": {"type": "integer"}}, "required": ["city"]}}}
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Chicago", "unit": "f", "days": 3, "opts": {"a": [1, 2.5, None, True]}}}}
    call2 = {"id": "c2", "type": "function", "function": {"name": "web_search", "arguments": {"query": "x\ny", "n": 1.0, "flag": False, "none": None}}}
    u = {"role": "user", "content": "Hi"}
    a = lambda **k: {"role": "assistant", **k}  # noqa: E731
    base = [
        [u],
        [{"role": "system", "content": "  Be brief.  "}, u],
        [{"role": "system", "content": [{"type": "text", "text": " one "}, {"type": "text", "text": "two "}]}, u],
        [{"role": "system", "content": ""}, u],
        [{"role": "system", "content": [{"type": "image"}]}, u],
        [{"role": "system", "content": [{"type": "video"}]}, u],
        [u, {"role": "system", "content": "late"}],
        [{"role": "developer", "content": "Dev"}, u],
        [{"role": "system", "content": "s"}],
        [{"role": "tool", "content": "orphan"}, u],
        [u, a(content="Hello!"), {"role": "user", "content": "Again"}],
        [u, a(content="Hello!"), {"role": "user", "content": "Again"}, a(content="Sure", reasoning_content="why not"), {"role": "user", "content": "More"}],
        [u, a(content="", tool_calls=[call]), {"role": "tool", "tool_call_id": "c1", "content": "61F sunny"}],
        [u, a(content="Checking.", tool_calls=[call, call2]), {"role": "tool", "content": "61F"}, {"role": "tool", "content": [{"type": "text", "text": "web"}]}],
        [u, a(content="Checking.", reasoning_content="need tool", tool_calls=[call]), {"role": "tool", "content": "61F"}, a(content="It is 61F.", reasoning_content="done")],
        [u, a(reasoning_content="need tool", tool_calls=[call]), {"role": "tool", "content": "61F"}, {"role": "user", "content": "thanks"}, a(content="np", reasoning_content="r")],
        [u, a(content="x", tool_calls=[call]), {"role": "tool", "content": "t"}, a(content="y", tool_calls=[call2]), {"role": "tool", "content": "t2"}],
        [u, a(content="<think>\nplan\n</think>\n\nanswer"), {"role": "user", "content": "q2"}],
        [u, a(content="<think>\nplan\n</think>\n\nanswer"), {"role": "user", "content": "q2"}, a(content="<think>\nplan2\n</think>\n\nanswer2")],
        [u, a(content="a</think>b</think>c"), {"role": "user", "content": "q2"}, a(content="p<think>q<think>r</think>s</think>t")],
        [u, a(content="<think>x", tool_calls=[call])],
        [u, a(content="</think>", tool_calls=[call])],
        [u, a(content="\n\n<think>\n\nr\n\n</think>\n\n\nans\n", reasoning_content=None)],
        [u, a(content="<think>in content</think>after", reasoning_content="explicit"), {"role": "user", "content": "n"}, a(content="<think>in content</think>after", reasoning_content="explicit")],
        [u, a(content="text", reasoning_content="")],
        [u, a(content="text", reasoning_content="  spaced reasoning \n")],
        [u, a(content=None, tool_calls=[{"function": {"name": "f"}}])],
        [u, a(content=None, tool_calls=[{"name": "f", "arguments": {"a": "b"}}])],
        [u, a(content="c", tool_calls=[{"function": {"name": "f", "arguments": {}}}, {"function": {"name": "g", "arguments": {"x": [1, "two", None]}}}])],
        [u, a(content="c", tool_calls=[{"function": {"name": "f", "arguments": '{"a": 1}'}}])],
        [u, a(content="c", tool_calls=[{"function": {"name": "f", "arguments": None}}])],
        [u, a(content="c", tool_calls=[{"function": {"arguments": {}}}])],
        [u, a(content="c", tool_calls=[{"function": {"name": 5, "arguments": {}}}])],
        [u, a(content="c", tool_calls=[])],
        [u, a(content="c", tool_calls="abc")],
        [u, a(content="c", tool_calls={"a": 1})],
        [u, a(content="c", tool_calls=3)],
        [{"role": "user", "content": "<tool_response>\nx\n</tool_response>"}],
        [u, a(tool_calls=[call]), {"role": "user", "content": "<tool_response>\nx\n</tool_response>"}],
        [u, a(tool_calls=[call]), {"role": "user", "content": "  <tool_response>x</tool_response>  "}, a(content="z")],
        [u, a(tool_calls=[call]), {"role": "user", "content": "<tool_response>x</tool_response> tail"}],
        [u, a(content="one"), a(content="two")],
        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what is this"}, {"type": "video"}, {"image_url": {"url": "x"}}]}],
        [{"role": "user", "content": [{"type": "image"}]}, a(content="a pic"), {"role": "user", "content": [{"type": "image"}, {"type": "image"}]}],
        [u, a(content=[{"type": "image"}, {"type": "text", "text": "x"}]), {"role": "tool", "content": [{"type": "image"}]}],
        [{"role": "user", "content": ["an image here", "text part", "plain"]}],
        [{"role": "user", "content": [{"type": "text", "text": None}]}],
        [{"role": "user", "content": [{"type": "text", "text": 5}, {"text": ["a", "b"]}, {"text": {"k": "v"}}]}],
        [{"role": "user", "content": [{"type": "other"}]}],
        [{"role": "user", "content": [5]}],
        [{"role": "user", "content": [None]}],
        [{"role": "user", "content": [[1]]}],
        [{"role": "user", "content": {"type": "text", "text": "mapping"}}],
        [{"role": "user", "content": 5}],
        [{"role": "user"}],
        [{"role": "user", "content": None}],
        [{"role": "user", "content": ""}, a(content=""), {"role": "user", "content": " "}],
        [{"content": "no role"}],
        [{"role": 5, "content": "x"}],
        [{"role": "weird", "content": "x"}, u],
        [u, "str message"],
        [u, 5],
        [u, None],
        [{"role": "user", "content": "x y　"}, a(content="  z  ")],
        [{"role": "tool", "content": "t1"}, {"role": "tool", "content": "t2"}, u],
        [u, {"role": "tool", "content": "t1"}, {"role": "tool", "content": "t2"}, {"role": "user", "content": "w"}],
        [u, {}, {"role": "tool", "content": "after empty"}],
        [{"role": "user", "content": "nul\x00char and \x00"}, a(content="\x00", reasoning_content="r\x00r", tool_calls=[{"function": {"name": "f", "arguments": {"k": "v\x00"}}}])],
        [u, a(content="", tool_calls=[call]), {"role": "tool", "content": ""}, {"role": "tool", "content": None}],
    ]
    errors_only = []
    out = [{"messages": m, "tools": None, "add_generation_prompt": True} for m in errors_only]
    tool_sets = [None, [weather], [weather, {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}],
                 [], ["str"], [{"name": "bare"}]]
    for msgs in base:
        for tools in tool_sets[:3]:
            for agp in (False, True):
                for think in (None, False, True):
                    for preserve in (None, True):
                        for vid in (None, True):
                            c = {"messages": msgs, "tools": tools, "add_generation_prompt": agp}
                            if think is not None:
                                c["enable_thinking"] = think
                            if preserve:
                                c["preserve_thinking"] = True
                            if vid:
                                c["add_vision_id"] = True
                            out.append(c)
        for tools in tool_sets[3:]:
            out.append({"messages": msgs, "tools": tools, "add_generation_prompt": True})
    out.append({"messages": [], "tools": None, "add_generation_prompt": True})
    out.append({"messages": "text", "tools": None, "add_generation_prompt": True})
    return out


def hf_render(tok, case):
    kwargs = {k: case[k] for k in ("enable_thinking", "preserve_thinking", "add_vision_id") if k in case}
    try:
        return tok.apply_chat_template(case["messages"], tools=case.get("tools"), add_generation_prompt=case["add_generation_prompt"],
                                       tokenize=False, **kwargs), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    # (a tuple return above is replaced below)


def hf_render_checked(tok, case):
    r = hf_render(tok, case)
    return r if isinstance(r, tuple) else (r, None)


def expected_ids(plain_tok, specials, parts):
    """The reference for eklow's tokenization of rendered parts (see the module docstring)."""
    ids, buf = [], []

    def flush():
        if buf:
            ids.extend(plain_tok.encode("".join(buf), add_special_tokens=False).ids)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("model_dir")
    ap.add_argument("variant", choices=["stock", "ornith"])
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gguf")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
    cases = hand_cases()
    n_hand = len(cases)
    rng = random.Random(a.seed)
    for _ in range(a.random):
        tools = [rnd_tool(rng, i) for i in range(rng.randint(0, 3))] if rng.random() < 0.6 else None
        c = {"messages": rnd_conversation(rng), "tools": tools, "add_generation_prompt": rng.random() < 0.6}
        if rng.random() < 0.5:
            c["enable_thinking"] = rng.random() < 0.5
        if rng.random() < 0.3:
            c["preserve_thinking"] = True
        if rng.random() < 0.2:
            c["add_vision_id"] = True
        cases.append(c)

    want = [hf_render_checked(tok, c) for c in cases]
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    cmd = [a.eklow_chat, "render", "--variant", a.variant] + (["--tokens", a.gguf] if a.gguf else [])
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
        print("CASE", json.dumps(c, ensure_ascii=False)[:700])
        if err or "error" in g:
            print("  hf:", err, "| eklow:", g.get("error", "rendered"))
        else:
            ours = g["text"]
            k = next((i for i, (x, y) in enumerate(zip(text, ours)) if x != y), min(len(text), len(ours)))
            print(f"  first difference at {k}: hf {text[max(0, k - 40):k + 60]!r}")
            print(f"  {' ' * 23}eklow {ours[max(0, k - 40):k + 60]!r}")
    print(f"{a.variant}: {len(cases)} cases ({n_hand} hand-written, {a.random} random): {same} identical, {both_err} rejected by both, {len(diffs)} differences")

    tok_diffs = 0
    if a.gguf:
        spec = json.load(open(f"{a.model_dir}/tokenizer.json", encoding="utf-8"))
        added = {t["content"]: t["id"] for t in spec["added_tokens"]}
        spec["added_tokens"] = []
        plain = Tokenizer.from_str(json.dumps(spec))
        specials = dict(sorted(added.items(), key=lambda kv: -len(kv[0])))
        n_tok = n_prompts = 0
        for c, g in zip(cases, got):
            if "error" in g:
                continue
            n_prompts += 1
            want_ids = expected_ids(plain, specials, g["parts"])
            n_tok += len(want_ids)
            if g["ids"] != want_ids:
                tok_diffs += 1
                if tok_diffs <= 5:
                    k = next((i for i, (x, y) in enumerate(zip(want_ids, g["ids"])) if x != y), min(len(want_ids), len(g["ids"])))
                    print(f"TOKENS differ at {k}: want {want_ids[k:k + 8]} eklow {g['ids'][k:k + 8]} in {json.dumps(c)[:200]}")
        print(f"tokens: {n_prompts} prompts, {n_tok} tokens, {tok_diffs} differences")
    sys.exit(1 if diffs or tok_diffs else 0)


if __name__ == "__main__":
    main()
