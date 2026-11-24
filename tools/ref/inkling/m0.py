#!/usr/bin/env python3
"""Inkling 2 Mini's M0 ground truth from llama.cpp c74c71a6 (DECISIONS 2026-11-24): the chat template and output
parser cases, the tokenizer strings, and the forward prompts. Every case is synthetic or eklow's own public text.

  m0.py cases   OUT_DIR             write cases.json (template + parser) and strings.json (tokenizer)
  m0.py prompts OUT_DIR ORACLE GGUF render the forward requests with eklow-chat-oracle, cut to length, and
                                    write prompts/NAME.txt (the rendered prompt) and prompts.json (name, tokens wanted)

The forward prompts are chat requests rendered by llama-server's own path:
  short   one user question                                         ~70 tokens
  tools   a tool loop: tools, a call, a file's text as its result   ~1.6k, past the 1,024-position sliding window
  long    a document question                                       ~12k, past the 8,192-position log-scale floor
  deep    the same, longer                                          ~33k, where the log scale is 1.14
The documents are DECISIONS.md and source files of eklow at a fixed commit (DOC_REV).
"""
import json
import os
import subprocess
import sys

DOC_REV = "cc9e886"  # eklow's commit whose public files the long prompts quote
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))


def git_show(path):
    return subprocess.run(["git", "-C", REPO, "show", f"{DOC_REV}:{path}"], check=True, capture_output=True,
                          text=True).stdout


SYSTEM = ("You are a careful assistant working in a user's home directory. Use the tools to look at files before "
          "answering questions about them. Keep answers short.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file and return its contents with line numbers.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string", "description": "Absolute path of the file."},
            "offset": {"type": "integer", "description": "First line to read (1-based).", "minimum": 1},
            "limit": {"type": "integer", "description": "Number of lines to read."}},
            "required": ["file_path"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace one exact string in a file with another.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string", "description": "Text to find; must occur exactly once."},
            "new_string": {"type": "string"},
            "replace_all": {"type": "boolean", "default": False}},
            "required": ["file_path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Run a shell command and return its output (stdout and stderr).",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "number", "description": "Seconds before the command is killed."},
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
            "mode": {"type": "string", "enum": ["foreground", "background"]}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "get_time", "description": "The current local time.",
        "parameters": {"type": "object", "properties": {}}}},
]

TOOLS_ODD = [
    {"type": "function", "function": {"name": "no_params"}},
    {"type": "function", "function": {
        "name": "météo_lookup", "description": "Wetter für eine Stadt — «prévisions» 天気 🌦",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"},
            "days": {"type": "array", "items": {"type": "integer"}, "maxItems": 7},
            "units": {"type": ["string", "null"], "enum": ["metric", "imperial", None]},
            "nested": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "boolean"}}}},
            "required": ["city"]}}},
]


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def u(text):
    return {"role": "user", "content": text}


def a(text, reasoning=None, calls=None):
    m = {"role": "assistant", "content": text}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    if calls is not None:
        m["tool_calls"] = calls
    return m


def t(cid, text, name=None):
    m = {"role": "tool", "tool_call_id": cid, "content": text}
    if name is not None:
        m["name"] = name
    return m


S = {"role": "system", "content": SYSTEM}

ODD_ARGS = {"zeta": 1, "alpha": "quote \" backslash \\ newline \n tab \t", "Beta": [1, 2.5, -3e-05, 1e+20, None, True],
            "nested": {"z": {}, "a": [], "é": "ünïcödé ✓ 日本語 😀"}, "big": 12345678901234567890, "float1": 1.0,
            "slash": "a/b</script>", "ctrl": "\u0001\u001f", "emptystr": ""}

TOOL_OUT = (
    '<|message_model|><|content_thinking|>The user wants the size of the file. I should read it first.<|end_message|>'
    '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"file_path":"/home/u/notes.txt"}}'
    '<|end_message|>')


def cases():
    c = []

    def add(cid, request, outputs=()):
        c.append({"id": cid, "request": request, "outputs": list(outputs)})

    add("user-only", {"messages": [u("Hello!")]},
        ['<|message_model|><|content_thinking|>A greeting.<|end_message|><|message_model|><|content_text|>Hi! How can I help?<|end_message|>',
         '<|message_model|><|content_text|>Hi there.<|end_message|>',
         '<|message_model|><|content_thinking|>Unfinished thought, cut by the token limit',
         '<|message_model|><|content_thinking|>one<|end_message|><|message_model|><|content_thinking|>two<|end_message|><|message_model|><|content_text|>three<|end_message|>',
         '<|message_model|><|content_text|>Text with <|message_user|> inside? and more<|end_message|>',
         'no header at all',
         ''])
    add("system-user", {"messages": [S, u("What is 2+2?")]})
    add("two-systems", {"messages": [S, {"role": "system", "content": "Second system message."}, u("Hi")]})
    add("system-late", {"messages": [u("Hi"), a("Hello."), {"role": "system", "content": "Be brief."}, u("Bye")]})
    add("multi-turn", {"messages": [S, u("Name a prime."), a("7."), u("Another?"), a("11."), u("And one more.")]})
    add("history-reasoning", {"messages": [S, u("Is 91 prime?"), a("No: 91 = 7 × 13.", reasoning="91/7 = 13."),
                                           u("And 97?")]})
    add("history-reasoning-empty-content", {"messages": [u("Think only."), a("", reasoning="Thinking alone."), u("Now answer.")]})
    add("content-parts", {"messages": [{"role": "user", "content": [{"type": "text", "text": "Part one."},
                                                                     {"type": "text", "text": "Part two."}]}]})
    add("unicode", {"messages": [u("Ünïcödé — «quotes» 日本語 العربية हिन्दी 😀👍🏽 \t tabs\r\nCRLF \\ backslash {{ braces }} {% raw %}")]})
    add("specials-in-user", {"messages": [u("What does <|end_message|> mean, and <|message_model|>?")]})
    add("empty-user", {"messages": [u("")]})
    for eff in ["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
        add(f"effort-{eff}", {"messages": [u("Hi")], "reasoning_effort": eff})
    add("effort-kwarg-number", {"messages": [u("Hi")], "chat_template_kwargs": {"reasoning_effort": 0.5}})
    add("effort-kwarg-zero", {"messages": [u("Hi")], "chat_template_kwargs": {"reasoning_effort": 0}})
    add("effort-bad", {"messages": [u("Hi")], "reasoning_effort": "extreme"})
    add("thinking-off-kwarg", {"messages": [u("Hi")], "chat_template_kwargs": {"enable_thinking": False}})

    add("tools", {"messages": [S, u("How long is /home/u/notes.txt?")], "tools": TOOLS},
        [TOOL_OUT,
         '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"file_path":"/a b/ü.txt","offset":3,"limit":10}}<|end_message|>',
         '<|message_model|><|content_text|>Let me check.<|end_message|><|message_model|>run_command<|content_invoke_tool_json|>{"name":"run_command","args":{"command":"wc -l /home/u/notes.txt","env":{"LC_ALL":"C"},"mode":"foreground","timeout_s":2.5}}<|end_message|>',
         '<|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|><|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"file_path":"/x"}}<|end_message|>',
         '<|message_model|><|content_thinking|>No tool needed.<|end_message|><|message_model|><|content_text|>I cannot tell without reading it.<|end_message|>',
         '<|message_model|>edit_file<|content_invoke_tool_json|>{"name":"edit_file","args":{"file_path":"/f","old_string":"a\\nb \\"q\\"","new_string":"✓ \\u00e9","replace_all":true}}<|end_message|>',
         '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"file_path":"/trunc',
         '<|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"path":"/wrong-key"}}<|end_message|>',
         '<|message_model|>unknown_tool<|content_invoke_tool_json|>{"name":"unknown_tool","args":{}}<|end_message|>',
         '<|message_model|><|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>'])
    add("tools-parallel", {"messages": [S, u("Time, and read /x.")], "tools": TOOLS, "parallel_tool_calls": True},
        ['<|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|><|message_model|>read_file<|content_invoke_tool_json|>{"name":"read_file","args":{"file_path":"/x"}}<|end_message|>',
         '<|message_model|><|content_thinking|>Two calls.<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>'])
    add("tools-required", {"messages": [u("Time?")], "tools": TOOLS, "tool_choice": "required"},
        ['<|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>',
         '<|message_model|><|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>',
         '<|message_model|><|content_thinking|>Must call.<|end_message|><|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>'])
    add("tools-none", {"messages": [u("Time?")], "tools": TOOLS, "tool_choice": "none"},
        ['<|message_model|><|content_text|>I will not call a tool.<|end_message|>',
         '<|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>'])
    add("tools-odd", {"messages": [u("Weather?")], "tools": TOOLS_ODD},
        ['<|message_model|>météo_lookup<|content_invoke_tool_json|>{"name":"météo_lookup","args":{"city":"Zürich","days":[1,2],"units":null,"nested":{"a":1e-3,"b":false}}}<|end_message|>',
         '<|message_model|>no_params<|content_invoke_tool_json|>{"name":"no_params","args":{}}<|end_message|>'])
    loop = [S, u("How long is /home/u/notes.txt?"),
            a("", reasoning="Read it first.", calls=[call("c1", "read_file", {"file_path": "/home/u/notes.txt"})]),
            t("c1", "1\tfirst line\n2\tsecond line\n"),
            a("It has 2 lines.")]
    add("tool-loop", {"messages": loop + [u("Thanks. Now the time?")], "tools": TOOLS},
        ['<|message_model|>get_time<|content_invoke_tool_json|>{"name":"get_time","args":{}}<|end_message|>'])
    add("tool-loop-open", {"messages": loop[:4], "tools": TOOLS},
        ['<|message_model|><|content_text|>It has 2 lines.<|end_message|>'])
    add("tool-result-named", {"messages": [u("t?"), a(None, calls=[call("x9", "get_time", {})]),
                                           t("x9", "12:00", name="get_time")], "tools": TOOLS})
    add("tool-result-unmatched-id", {"messages": [u("t?"), a(None, calls=[call("x9", "get_time", {})]),
                                                  t("other", "12:00")], "tools": TOOLS})
    add("tool-calls-parallel-history", {"messages": [u("both"), a("Doing both.", calls=[
        call("p1", "get_time", {}), call("p2", "read_file", {"file_path": "/x", "limit": 5})]),
        t("p1", "12:00"), t("p2", "1\tx")], "tools": TOOLS})
    add("tool-call-odd-args", {"messages": [u("odd"), a(None, calls=[call("o1", "run_command", ODD_ARGS)]),
                                            t("o1", "ok")], "tools": TOOLS})
    add("tool-call-arg-order", {"messages": [u("order"), a(None, calls=[call("o2", "edit_file", {
        "new_string": "b", "file_path": "/f", "old_string": "a", "Zed": 1, "_u": 2})]), t("o2", "ok")], "tools": TOOLS})
    add("tool-error-content", {"messages": [u("x"), a(None, calls=[call("e1", "read_file", {"file_path": "/nope"})]),
                                            t("e1", "Error: no such file")], "tools": TOOLS})
    add("assistant-name", {"messages": [u("x"), {"role": "assistant", "name": "helper", "content": "named"}, u("y")]})
    add("continue-content", {"messages": [u("Count to five."), a("1, 2, 3")]},
        [', 4, 5.<|end_message|>'])
    add("continue-reasoning", {"messages": [u("Count to five."), a("", reasoning="Start counting")]})
    add("no-gen-prompt", {"messages": [u("x")], "add_generation_prompt": False})
    add("response-format", {"messages": [u("JSON please")], "response_format": {"type": "json_object"}},
        ['<|message_model|><|content_text|>{"a": 1}<|end_message|>'])
    add("tools-empty-array", {"messages": [u("x")], "tools": []})
    return c


STRINGS = [
    "", " ", "  ", "\n", "\n\n", " \n", "\t", "\r\n", "Hello world", "Hello, world!", " Hello", "hello  world",
    "HELLO World HeLLo", "I'm you're they'll we've she'd it's I'M YOU'RE", "don't won't can't", "ﬁ ligature",
    "1 12 123 1234 12345 123456 1234567", "3.14159 -2.5e-10 0x1F 1,000,000", "a1b2c3 x99y",
    "def f(x):\n    return x**2  # square\n\n\nclass A:\n\tpass\n",
    "for (int i = 0; i < n; ++i) { s += a[i]; }",
    "<html><body class=\"x\">Text &amp; more</body></html>", "{\"key\": [1, 2, {\"n\": null}]}",
    "https://example.com/path?q=1&r=2#frag", "user@example.com",
    "Ünïcödé àéîõü ñ ç ß ø å æ", "日本語のテキストです。漢字とかなカナ。", "中文文本，标点符号。",
    "한국어 텍스트입니다", "العربية نص تجريبي", "עברית טקסט", "हिन्दी पाठ यहाँ है", "ภาษาไทย ทดสอบ",
    "Ελληνικά κείμενο", "Русский текст, ё Ё", "Tiếng Việt có dấu", "e\u0301 combining", "😀👍🏽👨‍👩‍👧‍👦🏳️‍🌈",
    "\u200b zero width \u200d joiner \ufeff bom", "\u00a0nbsp\u2003em space\u3000ideographic",
    "    leading spaces", "trailing spaces    ", "mixed \t\n \r\n\n\n  whitespace", "a\n\n\n\nb",
    "...!!!???---___===***", "CamelCaseIdentifier snake_case_name kebab-case-name SCREAMING_CASE",
    "<|message_user|>hi<|end_message|>", "<|content_model_end_sampling|>", "<|unused_200054|>", "<|endoftext|>",
    "text <|message_model|><|content_thinking|>x<|end_message|> after", "<|not_a_token|>", "<|message_user",
    "Thinking effort level: 0.9", "tool_declare", "<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}",
    "ab" * 300, "x" * 2000, " " * 100, "\n" * 50, "1" * 50, "😀" * 30,
]


def build_cases(out):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "cases.json"), "w") as f:
        json.dump(cases(), f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "strings.json"), "w") as f:
        json.dump(STRINGS, f, ensure_ascii=False, indent=1)


def forward_requests():
    decisions = git_show("DECISIONS.md")
    src = git_show("src/core/json.hpp")
    tools_loop = [S, u("What does the JSON reader in src/core/json.hpp accept, and is there anything risky in it?"),
                  a("", reasoning="I should read the header first.",
                    calls=[call("r1", "read_file", {"file_path": "/home/u/eklow/src/core/json.hpp"})]),
                  t("r1", "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(src.splitlines())))]
    return {
        "short": ({"messages": [S, u("What is the capital of France, and why did it become the capital? Two sentences.")]}, None),
        "tools": ({"messages": tools_loop, "tools": TOOLS}, None),
        "long": ({"messages": [S, u("Here is an engineering log.\n\n" + "{DOC}" +
                                    "\n\nWhich three decisions in it mattered most for speed, and why?")]}, (decisions, 12000)),
        "deep": ({"messages": [S, u("Here is an engineering log.\n\n" + "{DOC}" +
                                    "\n\nList every model this log says was served in production, in order.")]}, (decisions, 33000)),
    }


def build_prompts(out, oracle, model):
    """Render each forward request with the chat oracle, cutting the document until the prompt has the wanted
    number of tokens (at a line boundary)."""
    reqs = forward_requests()
    os.makedirs(os.path.join(out, "prompts"), exist_ok=True)
    meta = {}
    for name, (req, doc) in reqs.items():
        def render(text):
            r = json.loads(json.dumps(req).replace("{DOC}", json.dumps(text)[1:-1])) if doc else req
            path = os.path.join(out, "prompts", f"{name}.case.json")
            with open(path, "w") as f:
                json.dump([{"id": name, "request": r}], f, ensure_ascii=False)
            res = os.path.join(out, "prompts", f"{name}.result.json")
            subprocess.run([oracle, "--model", model, "--cases", path, "--out", res], check=True, capture_output=True)
            with open(res) as f:
                return json.load(f)[0]
        if doc:
            text, want = doc
            lines = text.splitlines(keepends=True)
            lo, hi = 1, len(lines)
            while lo < hi:  # the most lines whose prompt is at most `want` tokens
                mid = (lo + hi + 1) // 2
                if len(render("".join(lines[:mid]))["prompt_tokens"]) <= want:
                    lo = mid
                else:
                    hi = mid - 1
            r = render("".join(lines[:lo]))
        else:
            r = render(None)
        with open(os.path.join(out, "prompts", f"{name}.txt"), "w") as f:
            f.write(r["prompt"])
        meta[name] = {"tokens": len(r["prompt_tokens"])}
        print(name, meta[name]["tokens"], "tokens")
    with open(os.path.join(out, "prompts.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "cases":
        build_cases(sys.argv[2])
    elif len(sys.argv) >= 5 and sys.argv[1] == "prompts":
        build_prompts(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit(__doc__)
