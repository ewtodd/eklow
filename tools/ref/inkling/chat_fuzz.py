"""Random chat requests for Inkling 2 Mini's template port, and random model outputs for its parser. Everything is synthetic
(words from a fixed list, random numbers and Unicode); nothing here comes from a chat.

  from chat_fuzz import requests, outputs

requests(n, seed): n request bodies (OpenAI chat completions): random roles in random orders, content as strings, null
and part lists, reasoning, names, tool calls with arguments as JSON text or objects (floats such as 1.0, 1e-07, 1e+21,
-0.0, integers up to and past 64 bits, Unicode, escapes, empty objects, arrays and strings), tool schemas, reasoning_effort
(as the OpenAI field and as a chat_template_kwarg: levels, numbers, bad values), tool_choice, continue_final_message and
a few malformed ones.
"""
import json
import random

WORDS = ["alpha", "beta", "gamma", "delta", "north", "river", "stone", "lamp", "paper", "seven", "quiet", "orbit", "maple",
         "ember", "pixel", "garden", "thread", "violet", "anchor", "bridge", "cobalt", "harbor", "ivory", "jungle"]
UNICODE = ["é", "ü", "ß", "日本語", "한국어", "العربية", "हिन्दी", "😀", "👍🏽", "👨‍👩‍👧", "—", "«»", "​", " ", " ", "ﬁ",
           "\U0001d11e", "Ω", "я"]
ESCAPES = ["\n", "\t", "\r\n", "\\", "\"", "'", "\u0001", "\u001f", "\u007f", "/", "</script>", "{{ x }}", "{% raw %}", "{#", "%}", "\x08", "\x0c"]
SPECIALS = ["<|message_user|>", "<|end_message|>", "<|content_text|>", "<|message_model|>", "<|content_thinking|>",
            "<|content_model_end_sampling|>", "<|content_invoke_tool_json|>", "<|endoftext|>", "<|unused_200054|>", "<|message_tool|>"]
FLOATS = [1.0, -0.0, 0.0, 0.1, 0.5, -2.5, 1e-7, 1e-5, 0.0001, 1e21, 1e22, 1e16, 1e15, 123456789.123456789, 1.5e300, 5e-324,
          1.7976931348623157e308, 2.0 ** 53, 2.0 ** 63, 3.141592653589793, 100.0, 1e100, -1e-10, 0.30000000000000004]
INTS = [0, 1, -1, 7, 42, 2 ** 31, 2 ** 53, 2 ** 53 + 1, 2 ** 63 - 1, 2 ** 63, 2 ** 64 - 1, 2 ** 64, -2 ** 63, -2 ** 63 - 1, 10 ** 30, -10 ** 20]


def text(rng, special=0.03):
    n = rng.choice([0, 1, 1, 2, 3, 5, 8, 20])
    out = []
    for _ in range(n):
        r = rng.random()
        if r < special:
            out.append(rng.choice(SPECIALS))
        elif r < 0.3:
            out.append(rng.choice(UNICODE))
        elif r < 0.45:
            out.append(rng.choice(ESCAPES))
        else:
            out.append(rng.choice(WORDS))
        out.append(rng.choice([" ", " ", "", "\n", "  "]))
    return "".join(out)


def key(rng):
    r = rng.random()
    if r < 0.5:
        return rng.choice(WORDS)
    if r < 0.6:
        return rng.choice(WORDS).upper()
    if r < 0.7:
        return rng.choice(["", "a b", "_u", "Zed", "zed", "10", "9", "é", "Z", "ä", "日", "😀", "éa", "a\"b", "\\", "\n"])
    return rng.choice(WORDS) + rng.choice(UNICODE + ["", "1", "_"])


def value(rng, depth=0):
    r = rng.random()
    if depth >= 3:
        r = r * 0.75
    if r < 0.15:
        return rng.choice(INTS)
    if r < 0.30:
        return rng.choice(FLOATS)
    if r < 0.45:
        return text(rng, 0.05)
    if r < 0.50:
        return rng.choice([True, False, None])
    if r < 0.58:
        return rng.choice(["", [], {}])
    if r < 0.67:
        return rng.randint(-1000, 1000)
    if r < 0.75:
        return rng.uniform(-1000, 1000)
    if r < 0.88:
        return [value(rng, depth + 1) for _ in range(rng.choice([0, 1, 2, 3, 5]))]
    return obj(rng, depth + 1)


def obj(rng, depth=0, n=None):
    d = {}
    for _ in range(rng.choice([0, 1, 2, 3, 4, 6]) if n is None else n):
        d[key(rng)] = value(rng, depth)
    return d


def schema(rng, depth=0):
    r = rng.random()
    if depth > 2 or r < 0.3:
        s = {"type": rng.choice(["string", "integer", "number", "boolean", "null", ["string", "null"]])}
        if rng.random() < 0.4:
            s["description"] = text(rng, 0.05)
        if rng.random() < 0.2:
            s["enum"] = [value(rng, 3) for _ in range(rng.choice([1, 2, 3]))]
        if rng.random() < 0.2:
            s[rng.choice(["minimum", "maximum", "default", "multipleOf"])] = rng.choice(INTS + FLOATS)
        return s
    if r < 0.5:
        return {"type": "array", "items": schema(rng, depth + 1), **({"maxItems": rng.randint(1, 9)} if rng.random() < 0.3 else {})}
    props = {key(rng): schema(rng, depth + 1) for _ in range(rng.choice([0, 1, 2, 3, 4]))}
    s = {"type": "object", "properties": props}
    if props and rng.random() < 0.6:
        s["required"] = rng.sample(sorted(props), rng.randint(0, len(props)))
    if rng.random() < 0.2:
        s["additionalProperties"] = rng.choice([False, True, {"type": "string"}])
    return s


def tools(rng):
    out = []
    names = []
    for _ in range(rng.choice([1, 1, 2, 3, 4])):
        name = rng.choice(WORDS) + rng.choice(["", "_" + rng.choice(WORDS), rng.choice(UNICODE)])
        names.append(name)
        fn = {"name": name}
        r = rng.random()
        if r < 0.7:
            fn["description"] = text(rng, 0.05)
        elif r < 0.75:
            fn["description"] = ""
        r = rng.random()
        if r < 0.75:
            fn["parameters"] = schema(rng) if rng.random() < 0.3 else {"type": "object", "properties": {key(rng): schema(rng, 1) for _ in range(rng.choice([0, 1, 2, 3]))}}
        elif r < 0.8:
            fn["parameters"] = {}
        elif r < 0.83:
            fn["parameters"] = None
        if rng.random() < 0.05:
            fn["defer_loading"] = rng.choice([True, False])
        t = {"type": "function", "function": fn}
        if rng.random() < 0.05:
            t["defer_loading"] = rng.choice([True, False])
        out.append(t)
    return out, names


def arguments(rng):
    r = rng.random()
    if r < 0.6:
        a = obj(rng)
    elif r < 0.7:
        a = {}
    elif r < 0.75:
        a = {key(rng): rng.choice(FLOATS + INTS) for _ in range(rng.randint(1, 4))}
    elif r < 0.8:
        a = rng.choice([None, [1, 2], "str", 5, 0, "", [], False, True, 1.5])
        return rng.choice([json.dumps(a, ensure_ascii=False), a])
    else:
        a = obj(rng)
    if rng.random() < 0.75:
        return json.dumps(a, ensure_ascii=rng.random() < 0.5)
    return a


def message(rng, role, state):
    m = {"role": role}
    r = rng.random()
    if role in ("user", "system", "developer", "tool"):
        if r < 0.7:
            m["content"] = text(rng)
        elif r < 0.9:
            m["content"] = [{"type": "text", "text": text(rng)} for _ in range(rng.choice([0, 1, 2, 3]))]
        elif r < 0.905:
            m["content"] = [{"type": "tool_reference", "name": "x"}, {"type": "text", "text": text(rng)}]
        elif r < 0.97:
            m["content"] = None
        # else: no content (an error for these roles, 3% of the rest)
        if role == "tool":
            if rng.random() < 0.5 and state["ids"]:
                m["tool_call_id"] = rng.choice(state["ids"])
            elif rng.random() < 0.4:
                m["tool_call_id"] = "id" + str(rng.randint(0, 3))
            if rng.random() < 0.25:
                m["name"] = rng.choice(WORDS)
        return m
    # assistant
    r = rng.random()
    if r < 0.5:
        m["content"] = text(rng)
    elif r < 0.6:
        m["content"] = None
    elif r < 0.7:
        m["content"] = ""
    elif r < 0.78:
        m["content"] = [{"type": "text", "text": text(rng)} for _ in range(rng.choice([1, 2]))]
    elif r < 0.85:
        pass  # no content
    else:
        m["content"] = text(rng)
    r = rng.random()
    if r < 0.3:
        m["reasoning_content"] = text(rng, 0.05)
    elif r < 0.33:
        m["reasoning_content"] = ""
    elif r < 0.34:
        m["reasoning_content"] = None
    if rng.random() < 0.1:
        m["name"] = rng.choice(WORDS)
    if rng.random() < 0.4:
        calls = []
        for _ in range(rng.choice([1, 1, 2, 3])):
            cid = rng.choice(["call_" + str(rng.randint(0, 5)), "id" + str(rng.randint(0, 3)), "c" + rng.choice(WORDS)])
            c = {"type": "function", "function": {"name": rng.choice(WORDS), "arguments": arguments(rng)}}
            if rng.random() < 0.9:
                c["id"] = cid
                state["ids"].append(cid)
            if rng.random() < 0.05:
                c["function"]["name"] = rng.choice(["", "é", "a\"b", "x\ny"])
            if rng.random() < 0.05 and "name" in m:
                c["function"]["name"] = m["name"]
            calls.append(c)
        m["tool_calls"] = calls
        if "content" not in m and rng.random() < 0.7:
            m["content"] = None
    elif rng.random() < 0.03:
        m["tool_calls"] = []
    if "content" not in m and "tool_calls" not in m:
        m["content"] = text(rng)
    return m


def request(rng):
    state = {"ids": []}
    n = rng.choice([0, 1, 1, 2, 3, 4, 5, 6, 8, 12])
    msgs = []
    last = None
    for _ in range(n):
        r = rng.random()
        if r < 0.15:
            role = "system"
        elif r < 0.45:
            role = "user"
        elif r < 0.75:
            role = "assistant"
        elif r < 0.95:
            role = "tool"
        elif r < 0.98:
            role = "developer"
        else:
            role = rng.choice(["function", "", "USER", "model"])
        if role == "assistant" and last == "assistant" and rng.random() < 0.7:
            role = "user"
        msgs.append(message(rng, role, state))
        last = role
    # most requests end where a model would answer: after a user or tool message (or a closed assistant one)
    if msgs and msgs[-1]["role"] == "assistant" and rng.random() < 0.85:
        msgs.append(message(rng, rng.choice(["user", "tool"]), state))
    req = {"messages": msgs}
    if rng.random() < 0.5:
        req["tools"], _ = tools(rng)
    elif rng.random() < 0.05:
        req["tools"] = []
    r = rng.random()
    if r < 0.2:
        req["reasoning_effort"] = rng.choice(["none", "minimal", "low", "medium", "high", "xhigh", "max", "extreme", "", " high ", "HIGH", "auto", "high ", "\thigh\n"])
    elif r < 0.25:
        req["reasoning_effort"] = rng.choice([0.5, 1, None, True])
    r = rng.random()
    if r < 0.2:
        k = {"reasoning_effort": rng.choice([0.2, 0.7, "max", "low", "none", 0.0, 0.9, 0.3333333333333333, 0.05, 0.99, "medium", "xhigh", "minimal", "none", "high", "low", "max", "0.5", "bogus", " medium", 0, 0.1, 0.5, 0.99, 0.989999, 0.9999, 1.0, -0.0, -0.5,
                                             1e-7, 1e-5, 0.0001, 0.00004, 0.123456789012, 0.99000001, 0.9899999, 5e-324, 2, 1, True, False, None, [1], {}, [], ""])}
        req["chat_template_kwargs"] = k
    elif r < 0.25:
        req["chat_template_kwargs"] = rng.choice([{"enable_thinking": False}, {"enable_thinking": True}, {"foo": [1, 2]}, {}, [], {"enable_thinking": False}, {"foo": 1}, {"enable_thinking": "yes"}])
    r = rng.random()
    if r < 0.1:
        req["tool_choice"] = rng.choice(["none", "required", "auto", "none", "required", "auto", "bogus", {"type": "function", "function": {"name": "x"}}])
    r = rng.random()
    if r < 0.08:
        req["continue_final_message"] = rng.choice([True, False, None, "content", "reasoning_content", True, "content", "reasoning_content", "other", 1])
        if rng.random() < 0.5:
            req["add_generation_prompt"] = False
    elif r < 0.09:
        req["add_generation_prompt"] = rng.choice([False, True, "no"])
    if rng.random() < 0.05:
        req["parallel_tool_calls"] = rng.choice([True, False])
    if rng.random() < 0.04:
        req["response_format"] = rng.choice([{"type": "json_object"}, {"type": "text"}, {"type": "json_schema", "json_schema": {"schema": {"type": "string"}}},
                                              {"type": "bogus"}, {"type": "json_object", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}])
    if rng.random() < 0.02:
        req["reasoning_format"] = rng.choice(["none", "auto", "deepseek", "bogus"])
    return req


def requests(n, seed):
    rng = random.Random(seed)
    return [request(rng) for _ in range(n)]


def deep(n, leaf="x"):
    v = leaf
    for _ in range(n):
        v = {"k": [v]}
    return v


def edge_requests():
    """Hand-written corner cases the random generator rarely reaches."""
    U = {"role": "user", "content": "hi"}

    def call(cid, name, args):
        return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}

    def a(calls, content=None):
        return {"role": "assistant", "content": content, "tool_calls": calls}

    reqs = []
    # raw argument texts nlohmann reads in its own ways
    for text_ in ['{"a":1e999}', '{"a":-1e999}', '{"a":"\\ud800"}', '{"a":"\\ud83d\\ude00"}', '{"a":1,"a":2}', '{"a":{"b":1,"b":{"c":2}}}',
                  '{"a":-0}', '{"a":-0.0}', '{"a":0.1e1}', '{"a":1E5}', '{"a":1e-400}', '{"a":123456789012345678901234567890e5}',
                  '{"a":9223372036854775807}', '{"a":9223372036854775808}', '{"a":18446744073709551615}', '{"a":18446744073709551616}',
                  '{"a":-9223372036854775808}', '{"a":-9223372036854775809}', ' {"a" : 1} ', '{"a":1}x', '{"a":}', '{', '', 'null', '[]', '{}', '0', '1',
                  '"s"', 'true', 'false', '[1]', '{"é":1,"E":2,"a":3,"Z":4,"z":5,"_":6,"":7,"\\u0000":8}', '{"b":1,"a":2,"B":3,"A":4,"aa":5,"a ":6,"ab":7}',
                  '{"a":"\\u0000\\u0008\\u000c\\u001f\\u007f\\u2028\\u00e9"}', '{"a":[1.5,2.50,3.0,1e2,1E+2,100e-2]}', '{"emoji😀":"\\ud83d\\ude00","日本":"x","a":null}',
                  '{"a":' + json.dumps(deep(20)) + '}', '{"a":' + json.dumps(deep(130)) + '}', '{"a":' + json.dumps(deep(300)) + '}']:
        reqs.append({"messages": [U, a([call("c", "f", text_)]), {"role": "tool", "tool_call_id": "c", "content": "ok"}]})
    # argument values that are not strings
    for v in [{}, {"b": 1, "a": [1.0]}, [], [1], None, 5, 0, "", True, False, 1.5, {"é": {"b": 1, "a": 2}}]:
        reqs.append({"messages": [U, a([call("c", "f", v)]), {"role": "tool", "tool_call_id": "c", "content": "ok"}]})
    # tool_call_id matching
    reqs.append({"messages": [U, a([call("x", "one", "{}"), call("x", "two", "{}")]), {"role": "tool", "tool_call_id": "x", "content": "r"}]})
    reqs.append({"messages": [a([call("x", "one", "{}")]), {"role": "tool", "tool_call_id": "x", "content": "r"}, a([call("x", "again", "{}")]),
                              {"role": "tool", "tool_call_id": "x", "content": "r2"}, U]})
    reqs.append({"messages": [{"role": "tool", "tool_call_id": "x", "content": "r"}, U, a([call("x", "late", "{}")]), U]})
    reqs.append({"messages": [U, a([{"type": "function", "function": {"name": "noid", "arguments": "{}"}}]), {"role": "tool", "tool_call_id": "", "content": "r"}, U]})
    reqs.append({"messages": [U, a([call("x", "one", "{}")], "text"), {"role": "tool", "tool_call_id": "x", "name": "chosen", "content": "r"}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "name": "bot", "content": "hi", "reasoning_content": "r"}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "name": "bot", "content": None, "tool_calls": [call("a", "bot", "{}")]}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "name": "bot", "content": None, "tool_calls": [call("a", "other", "{}")]}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "content": "", "reasoning_content": "only thought"}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "content": ""}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "content": [], "tool_calls": [call("a", "f", "{}")]}, U]})
    reqs.append({"messages": [U, {"role": "assistant", "content": [{"type": "text", "text": ""}], "reasoning_content": "t"}, U]})
    reqs.append({"messages": [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": "b"}]}]})
    reqs.append({"messages": [{"role": "user", "content": []}]})
    reqs.append({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})
    reqs.append({"messages": [{"role": "user", "content": [{"type": "text"}]}]})
    reqs.append({"messages": [{"role": "user", "content": [{"text": "no type"}]}]})
    reqs.append({"messages": [{"role": "user", "content": 5}]})
    reqs.append({"messages": [{"role": "user"}]})
    reqs.append({"messages": [{"content": "no role"}]})
    reqs.append({"messages": [{"role": 5, "content": "x"}]})
    reqs.append({"messages": ["x"]})
    reqs.append({"messages": []})
    reqs.append({"messages": [{"role": "system", "content": "only system"}]})
    reqs.append({"messages": [{"role": "developer", "content": "dev"}, {"role": "system", "content": "sys"}, U]})
    reqs.append({"messages": [{"role": "tool", "content": "t"}, {"role": "tool", "content": "t2", "name": "n"}]})
    reqs.append({"messages": {"role": "user"}})
    reqs.append({})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f"}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "description": None}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "description": 5}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": 5}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function"}]})
    reqs.append({"messages": [U], "tools": [{"function": {"name": "f"}}]})
    reqs.append({"messages": [U], "tools": [{"type": "custom", "function": {"name": "f"}}]})
    reqs.append({"messages": [U], "tools": "x"})
    reqs.append({"messages": [U], "tools": {}})
    reqs.append({"messages": [U], "tools": None})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"z": 1, "a": [1e-7, 1e21, -0.0], "m": {"y": 1.0, "b": 2}}}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": deep(60)}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": []}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": "str"}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": 0}}]})
    reqs.append({"messages": [U], "tools": [{"type": "function", "function": {"name": "f é\"\n", "description": "<|end_message|>"}}]})
    # effort in every shape (the OpenAI field, the kwarg, both)
    for kw in [0.99, 0.989999999, 0.9900000001, 1e-7, 0.5, 0.123456789, 1e-6, 4e-7, 5e-7, 6e-7, 0.0000005, 0.0000004999, 0.1, 0.30000000000000004,
               0.7, 0.2, 0.9, 0, 1, -1, -0.0, 0.0, 99, 1e308, 1e-320, True, False, None, "none", "max", " max ", "MAX", " max", "max\u0085", "\x0bmax\x0c", "\x1cmax", [], {}, [0.5], {"a": 1}]:
        reqs.append({"messages": [U], "chat_template_kwargs": {"reasoning_effort": kw}})
    for eff in ["none", "minimal", "low", "medium", "high", "xhigh", "max", "", " ", "NONE", 0.5, 1, None, True, [], {}, "unknown"]:
        reqs.append({"messages": [U], "reasoning_effort": eff, "chat_template_kwargs": {"reasoning_effort": 0.33}})
        reqs.append({"messages": [U], "reasoning_effort": eff})
    # continuation
    for cfm in [None, True, False, "content", "reasoning_content", "bogus", 0, "", [], {}]:
        for last in [{"role": "assistant", "content": "partial"}, {"role": "assistant", "content": "", "reasoning_content": "thought"},
                     {"role": "assistant", "content": "c", "reasoning_content": "r"}, {"role": "assistant", "content": ""}, {"role": "assistant", "content": None},
                     {"role": "assistant", "content": [{"type": "text", "text": "p1"}, {"type": "text", "text": "p2"}]},
                     {"role": "assistant", "content": [{"type": "text", "text": "p1"}, {"type": "tool_reference", "name": "n"}, {"type": "text", "text": "p2"}]},
                     {"role": "assistant", "content": "x", "tool_calls": [call("a", "f", "{}")]}, {"role": "assistant", "content": "", "name": "bot"},
                     {"role": "assistant", "content": "", "tool_call_id": "t"}, {"role": "user", "content": "u"}, {"role": "tool", "content": "t", "name": "x"}]:
            for agp in [None, False, True]:
                r = {"messages": [U, last]}
                if cfm is not None:
                    r["continue_final_message"] = cfm
                if agp is not None:
                    r["add_generation_prompt"] = agp
                reqs.append(r)
    reqs.append({"messages": [U, {"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]})
    reqs.append({"messages": [{"role": "assistant", "content": "only"}]})
    reqs.append({"messages": [{"role": "assistant", "content": "only"}], "continue_final_message": "content"})
    reqs.append({"messages": [], "continue_final_message": "content", "add_generation_prompt": False})
    reqs.append({"messages": [U, {"role": "assistant", "content": "a<|end_message|><|message_user|>b", "reasoning_content": "<|content_text|>"}]})
    # request-level options
    for extra in [{"tool_choice": "none"}, {"tool_choice": "required"}, {"tool_choice": "auto"}, {"tool_choice": {"type": "function", "function": {"name": "f"}}},
                  {"tool_choice": 5}, {"tool_choice": None}, {"tool_choice": "other"}, {"parallel_tool_calls": False}, {"parallel_tool_calls": "no"},
                  {"grammar": "root ::= x"}, {"grammar": ""}, {"grammar": 5}, {"json_schema": {"type": "string"}, "grammar": "x"},
                  {"json_schema": {}}, {"json_schema": 5}, {"response_format": "x"}, {"response_format": None}, {"response_format": {"type": "json_object"}},
                  {"response_format": {"type": "json_object", "schema": {}}}, {"response_format": {"type": "json_schema"}},
                  {"response_format": {"type": "json_schema", "json_schema": {"schema": 5}}}, {"response_format": {"type": 5}},
                  {"reasoning_format": "none"}, {"reasoning_format": 5}, {"reasoning_format": None}, {"reasoning_format": "deepseek-legacy"},
                  {"chat_template_kwargs": {"enable_thinking": "false"}}, {"chat_template_kwargs": {"enable_thinking": True}}, {"chat_template_kwargs": "x"},
                  {"chat_template_kwargs": None}, {"chat_template_kwargs": {"messages": [], "tools": []}}, {"stream": True}, {"add_generation_prompt": "x"}]:
        for with_tools in (False, True):
            r = {"messages": [U], **extra}
            if with_tools:
                r["tools"] = [{"type": "function", "function": {"name": "f"}}]
            reqs.append(r)
    return reqs
