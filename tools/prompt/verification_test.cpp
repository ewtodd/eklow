// Host test of the server defaults in parse_chat: --verification (Sage's template) and --preserve-thinking. eklow-verification-test
// VOCAB_GGUF, any Qwen3.6 GGUF for its tokenizer. Each request's prompt must be the ids of the template's own render with the options
// the rule gives; with preserve_thinking the next turn of an agentic conversation must share the prompt up to its generation prompt.
#include <algorithm>
#include <cstdio>
#include <string>

#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "serve/openai.hpp"
#include "text/chat_qwen36.hpp"
#include "text/output_parser_qwen.hpp"
#include "text/tokenizer_qwen.hpp"

using namespace eklow;

namespace {

int failures = 0, checks = 0;

void check(bool ok, const std::string& what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what.c_str());
  }
}

std::vector<int32_t> render_ids(const text::QwenTokenizer& tok, const json::Value& messages, std::optional<json::Value> verification) {
  text::QwenChatOptions o;
  o.add_generation_prompt = true;
  o.verification = std::move(verification);
  std::vector<text::QwenTokenizer::Piece> pieces;
  const auto parts = text::render_qwen36(text::QwenTemplate::Sage, messages, json::Value(), o);
  for (const auto& p : parts) pieces.push_back({p.text, !p.input});
  return tok.encode(pieces);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: eklow-verification-test VOCAB_GGUF\n");
    return 2;
  }
  const auto file = gguf::File::open(argv[1]);
  const text::QwenTokenizer tok = text::QwenTokenizer::from_gguf(file);
  // A conversation as JSON text (the request body's messages) and as a value (for the template's own render).
  struct Conv {
    std::string text;
    json::Value value;
  };
  const auto msgs = [](const char* system) {
    std::string s = "[";
    if (system) s += std::string("{\"role\":\"system\",\"content\":\"") + system + "\"},";
    s += "{\"role\":\"user\",\"content\":\"Hi\"}]";
    return Conv{s, json::parse(s)};
  };
  serve::ChatDefaults on{serve::SamplingParams{}, int64_t(1) << 30, text::QwenTemplate::Sage, true};
  serve::ChatDefaults off = on;
  off.verification = false;
  const json::Value T = json::Value::boolean(true), F = json::Value::boolean(false);

  const Conv m_sys = msgs(" Be brief. "), m_none = msgs(nullptr), m_blank = msgs("  \\n ");
  const auto prompt = [&](const Conv& m, const std::string& extra, const serve::ChatDefaults& d) {
    return serve::parse_chat(json::parse("{\"messages\":" + m.text + extra + "}"), tok, d).prompt;
  };
  const auto ids = [&](const Conv& m, std::optional<json::Value> v) { return render_ids(tok, m.value, std::move(v)); };

  // default on, non-empty system message: the trigger is appended
  check(prompt(m_sys, "", on) == ids(m_sys, T), "default on, system text: trigger");
  check(prompt(m_sys, "", on) != ids(m_sys, std::nullopt), "default on, system text: differs from no trigger");
  // default on, no system message / blank one: as if absent, no error
  check(prompt(m_none, "", on) == ids(m_none, std::nullopt), "default on, no system message: renders as without");
  check(prompt(m_blank, "", on) == ids(m_blank, std::nullopt), "default on, blank system message: no trigger");
  // a `developer` message is the system one (parse_chat maps the role before the template), so the default applies to it too
  // (serve_test.py basic on fiona, 2026-11-07: developer 22 tokens against system 27 before the fix)
  {
    std::string s = m_sys.text;
    s.replace(s.find("\"system\""), 8, "\"developer\"");
    const Conv m_dev{s, json::parse(s)};
    check(prompt(m_dev, "", on) == prompt(m_sys, "", on), "default on, developer message: the same prompt as a system one");
    check(prompt(m_dev, "", on) == ids(m_sys, T), "default on, developer message: trigger");
  }
  // default on, the request sends false (top level and in chat_template_kwargs): off
  check(prompt(m_sys, ",\"verification\":false", on) == ids(m_sys, F), "default on, request false");
  check(prompt(m_sys, ",\"chat_template_kwargs\":{\"verification\":false}", on) == ids(m_sys, F), "default on, kwargs false");
  check(prompt(m_sys, ",\"chat_template_kwargs\":{\"verification\":false}", on) == ids(m_sys, std::nullopt), "default on, kwargs false = no trigger");
  // the request's own true keeps the template's rules
  check(prompt(m_sys, ",\"verification\":true", off) == ids(m_sys, T), "default off, request true");
  for (const Conv* m : {&m_none, &m_blank}) {
    for (const char* extra : {",\"verification\":true", ",\"chat_template_kwargs\":{\"verification\":true}"}) {
      for (const auto* d : {&on, &off}) {
        bool threw = false;
        try {
          prompt(*m, extra, *d);
        } catch (const serve::ApiError& e) {
          threw = e.status == 400;
        }
        check(threw, std::string("request true without system text: 400 ") + extra + (d == &on ? " (default on)" : " (default off)"));
      }
    }
  }
  // the request's string "false" still raises, with the default on
  bool threw = false;
  try {
    prompt(m_sys, ",\"verification\":\"false\"", on);
  } catch (const serve::ApiError& e) {
    threw = e.status == 400;
  }
  check(threw, "request \"false\" string: 400");
  // default off: nothing sent, nothing appended
  check(prompt(m_sys, "", off) == ids(m_sys, std::nullopt), "default off, nothing sent");

  // --preserve-thinking: an agentic turn (a tool call with reasoning, its result), then the answer and a new user message. The
  // second request's prompt must share the first's up to its generation prompt (where the scheduler snapshots), or the whole turn
  // is prefilled again.
  {
    const std::string tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"ls\",\"description\":\"List files\","
                              "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
    const std::string turn = "[{\"role\":\"system\",\"content\":\"Be brief.\"},{\"role\":\"user\",\"content\":\"What is here?\"},"
                             "{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":\"I should list the files.\","
                             "\"tool_calls\":[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"ls\",\"arguments\":\"{}\"}}]},"
                             "{\"role\":\"tool\",\"tool_call_id\":\"c1\",\"content\":\"a.txt\"}";
    const std::string first = turn + "]";
    const std::string next = turn + ",{\"role\":\"assistant\",\"content\":\"There is a.txt.\",\"reasoning_content\":\"One file.\"},"
                                    "{\"role\":\"user\",\"content\":\"Thanks.\"}]";
    const auto req = [&](const std::string& m, const std::string& extra, const serve::ChatDefaults& d) {
      return serve::parse_chat(json::parse("{\"messages\":" + m + tools + extra + "}"), tok, d).prompt;
    };
    const auto shared = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
      size_t n = 0;
      while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
      return n;
    };
    const int32_t im_start = tok.id("<|im_start|>");
    const auto gen_start = [&](const std::vector<int32_t>& p) {
      size_t at = 0;
      for (size_t i = 0; i < p.size(); ++i)
        if (p[i] == im_start) at = i;
      return at;
    };
    serve::ChatDefaults keep = off, drop = off;
    keep.preserve_thinking = true;
    const auto a_keep = req(first, "", keep), b_keep = req(next, "", keep);
    const auto a_drop = req(first, "", drop), b_drop = req(next, "", drop);
    check(a_keep == a_drop, "preserve: the turn's own prompt is the same either way (its reasoning is after the last user query)");
    check(shared(a_keep, b_keep) >= gen_start(a_keep), "preserve on: the next turn shares the prompt up to its generation prompt");
    check(shared(a_drop, b_drop) < gen_start(a_drop), "preserve off: the next turn parts from the prompt inside the turn");
    // the request's own value wins either way, top level and in chat_template_kwargs
    check(req(next, ",\"preserve_thinking\":false", keep) == b_drop, "preserve default on, request false");
    check(req(next, ",\"chat_template_kwargs\":{\"preserve_thinking\":false}", keep) == b_drop, "preserve default on, kwargs false");
    check(req(next, ",\"preserve_thinking\":true", drop) == b_keep, "preserve default off, request true");
    check(req(next, ",\"chat_template_kwargs\":{\"preserve_thinking\":true}", drop) == b_keep, "preserve default off, kwargs true");
    std::printf("preserve: the next turn shares %zu of %zu tokens with it, %zu without (generation prompt at %zu)\n", shared(a_keep, b_keep),
                b_keep.size(), shared(a_drop, b_drop), gen_start(a_keep));

    // tool_choice "none" (a client's last step of a long tool loop): the same prompt as with tools allowed, so the prefix stays
    // cached, and no tool call can be sampled.
    const auto chat = [&](const std::string& extra) {
      return serve::parse_chat(json::parse("{\"messages\":" + first + tools + extra + "}"), tok, off);
    };
    const serve::ChatRequest c_auto = chat(""), c_none = chat(",\"tool_choice\":\"none\""), c_explicit = chat(",\"tool_choice\":\"auto\"");
    check(c_none.prompt == c_auto.prompt, "tool_choice none: the prompt keeps the tools (identical to tool_choice auto's)");
    check(c_auto.calls_allowed && c_auto.tool_specs, "no tool_choice: calls allowed, with the tools' specs");
    check(c_explicit.calls_allowed && c_explicit.prompt == c_auto.prompt, "tool_choice auto: as without");
    check(!c_none.calls_allowed && !c_none.tool_specs, "tool_choice none: calls not allowed, no specs to constrain them");
    const int32_t call_open = tok.id("<tool_call>"), im_end = tok.id("<|im_end|>");
    const auto bans = [](const text::QwenOutputParser& p, int32_t id) {
      return std::find(p.banned().begin(), p.banned().end(), id) != p.banned().end();
    };
    const text::QwenOutputParser p_on(tok, false, nullptr, true), p_off(tok, false, nullptr, false);
    check(!bans(p_on, call_open) && bans(p_off, call_open), "calls off: the parser bans <tool_call> in the content");
    check(!bans(p_on, im_end) && !bans(p_off, im_end), "calls off: the turn can still end");
    check(std::is_sorted(p_off.banned().begin(), p_off.banned().end()), "calls off: the banned ids stay ascending");
  }

  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
