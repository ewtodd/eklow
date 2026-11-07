// Host test of --verification's default in parse_chat (Sage's template): eklow-verification-test VOCAB_GGUF, any Qwen3.6
// GGUF for its tokenizer. Each request's prompt must be the ids of the template's own render with the options the rule gives.
#include <cstdio>
#include <string>

#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "serve/openai.hpp"
#include "text/chat_qwen36.hpp"
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

  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
