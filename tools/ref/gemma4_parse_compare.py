"""Round-trip check of eklow's Gemma 4 output parser through the chat template.

Usage: python3 tools/ref/parse_compare.py EKLOW_CHAT EKLOW_PARSE GGUF [--random N]

Random assistant turns (reasoning, tool calls with arguments of every JSON kind, content) are
rendered after a user turn by the template port (identical to Hugging Face's, see chat_compare.py)
and tokenized; the tokens after the last "<|turn>model\\n" are what the model would have generated
for that turn. eklow-parse reads them back one token at a time and must recover the reasoning, the
tool calls (name and arguments, compared as values) and the content (as the template trims it).
Exits 1 on any difference.
"""

import argparse
import json
import random
import subprocess
import sys

TEXTS = ["hello", "  padded  ", "multi\nline", "tabs\tand nbsp", "naïve 日本語 🙂", "<|turn>not special",
         "a<|\"|>b", "quote ' and \"", "back\\slash", "x" * 120, "trailing newline\n", "{braces}, [brackets]: colons",
         "\U0001F9EA\U0001FAE0", "é"]
KEYS = ["a", "B", "path", "query", "limit", "x_y", "Zeta", "日本", "under_score", "with space"]


def rnd_text(rng):
    return rng.choice(TEXTS) if rng.random() < 0.7 else "".join(rng.choice("ab ,:{}[]\n\"'é🙂") for _ in range(rng.randint(1, 20)))


def rnd_value(rng, depth=0):
    r = rng.random()
    if depth > 3 or r < 0.3:
        return rnd_text(rng) if rng.random() < 0.9 else ""
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


def rnd_turn(rng):
    m = {"role": "assistant"}
    if rng.random() < 0.5:
        m["reasoning_content"] = rnd_text(rng) or "thinking"
    if rng.random() < 0.6:
        m["tool_calls"] = [{"id": f"c{k}", "type": "function", "function": {
            "name": rng.choice(["web_search", "read_file", "cronjob", "f"]),
            "arguments": {rng.choice(KEYS): rnd_value(rng) for _ in range(rng.randint(0, 4))}}}
            for k in range(rng.randint(1, 3))]
    if rng.random() < 0.7 or "tool_calls" not in m:
        m["content"] = rnd_text(rng)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eklow_chat")
    ap.add_argument("eklow_parse")
    ap.add_argument("gguf")
    ap.add_argument("--random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    turns = [rnd_turn(rng) for _ in range(a.random)]
    cases = [{"messages": [{"role": "user", "content": "go"}, t], "tools": None, "add_generation_prompt": False} for t in turns]
    lines = "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n"
    rendered = subprocess.run([a.eklow_chat, "render", "--tokens", a.gguf], input=lines.encode(), capture_output=True, check=True).stdout.decode()
    rendered = [json.loads(x) for x in rendered.split("\n") if x]
    if len(rendered) != len(cases):
        sys.exit("eklow-chat returned the wrong number of results")
    marker = [105, 4368, 107]  # <|turn> model \n
    gen = []
    for r in rendered:
        ids = r["ids"]
        k = max(i for i in range(len(ids) - 2) if ids[i:i + 3] == marker)
        gen.append(ids[k + 3:])
    out = subprocess.run([a.eklow_parse, a.gguf], input=("\n".join(" ".join(map(str, g)) for g in gen) + "\n").encode(),
                         capture_output=True, check=True).stdout.decode()
    got = [json.loads(x) for x in out.split("\n") if x]
    bad = 0
    for t, g, r in zip(turns, got, rendered):
        want_calls = [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in t.get("tool_calls", [])]
        want = {"reasoning": t.get("reasoning_content", ""), "content": t.get("content", "").strip(), "tool_calls": want_calls}
        have = {"reasoning": g["reasoning"], "content": g["content"], "tool_calls": g["tool_calls"]}
        if want != have or g["errors"]:
            bad += 1
            if bad <= 6:
                print("TURN ", json.dumps(t, ensure_ascii=False)[:400])
                print("  want", json.dumps(want, ensure_ascii=False)[:400])
                print("  got ", json.dumps(have, ensure_ascii=False)[:400], g["errors"])
    print(f"{len(turns)} turns: {len(turns) - bad} recovered exactly, {bad} differences")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
