// eklow-grammar-test: the tool-call grammar (serve/tool_grammar.hpp) against the output parser,
// on Gemma 4's vocabulary.
//
//   eklow-grammar-test --model GGUF [--calls N] [--walks N] [--seed S]
//
// Calls: random well-formed calls (nested arguments of every kind, strings full of the
// punctuation the syntax uses), written as the template writes them and tokenized as the model
// would see them; every token must be allowed. Walks: random token sequences drawn from the
// allowed sets until the call closes; every one must parse into a call of a declared tool.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "serve/tool_grammar.hpp"
#include "text/output_parser.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

namespace {

const std::vector<std::string> kTools = {"get_weather", "web_search", "write_file", "todo", "terminal"};

struct Gen {
  std::mt19937_64 rng;
  std::string text_run() {
    static const std::vector<std::string> bits = {"a", "Paris", " ", ",", ":", "{", "}", "[", "]", "\n", "é", "日本", "x = 1", "\"", "\\", "'", "-3"};
    std::string s;
    for (int n = int(rng() % 6); n > 0; --n) s += bits[rng() % bits.size()];
    return s;
  }
  std::string key() {
    static const std::vector<std::string> keys = {"city", "query", "path", "content", "unit", "n", "items", "a_b", "deep-key"};
    return keys[rng() % keys.size()];
  }
  // Appends pieces (text, quote) for one value.
  void value(std::vector<std::pair<std::string, bool>>& out, int depth) {
    switch (rng() % (depth < 3 ? 8 : 6)) {
      case 0:
        out.push_back({"", true});
        out.push_back({text_run(), false});
        out.push_back({"", true});
        break;
      case 1: out.push_back({std::to_string(int64_t(rng() % 2000) - 1000), false}); break;
      case 2: {
        static const std::vector<std::string> floats = {"0.5", "-12.25", "1e-05", "1e+16", "3.0", "2.5E3"};
        out.push_back({floats[rng() % floats.size()], false});
        break;
      }
      case 3: out.push_back({"true", false}); break;
      case 4: out.push_back({"false", false}); break;
      case 5: out.push_back({"null", false}); break;
      case 6: object(out, depth + 1); break;
      default: {
        out.push_back({"[", false});
        for (int n = int(rng() % 4), i = 0; i < n; ++i) {
          if (i) out.push_back({",", false});
          value(out, depth + 1);
        }
        out.push_back({"]", false});
      }
    }
  }
  void object(std::vector<std::pair<std::string, bool>>& out, int depth) {
    out.push_back({"{", false});
    for (int n = int(rng() % 4), i = 0; i < n; ++i) {
      if (i) out.push_back({",", false});
      out.push_back({key() + ":", false});
      value(out, depth);
    }
    out.push_back({"}", false});
  }
};

}  // namespace

int main(int argc, char** argv) {
  std::string path;
  int calls = 2000, walks = 2000;
  uint64_t seed = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") path = next();
    else if (a == "--calls") calls = std::atoi(next().c_str());
    else if (a == "--walks") walks = std::atoi(next().c_str());
    else if (a == "--seed") seed = std::strtoull(next().c_str(), nullptr, 10);
    else {
      std::fprintf(stderr, "usage: eklow-grammar-test --model GGUF [--calls N] [--walks N] [--seed S]\n");
      return 2;
    }
  }
  try {
    auto f = gguf::File::open(path);
    const text::Tokenizer tok = text::Tokenizer::from_gguf(f);
    auto vocab = std::make_shared<const serve::ToolVocab>(tok);
    Gen g{std::mt19937_64(seed)};

    int bad_calls = 0;
    for (int c = 0; c < calls; ++c) {
      std::vector<std::pair<std::string, bool>> pieces{{"call:" + kTools[g.rng() % kTools.size()], false}};
      g.object(pieces, 0);
      // The model sees text runs tokenized together and <|"|> as its own token.
      std::vector<int32_t> ids{vocab->call_open};
      std::string run;
      auto flush = [&] {
        for (int32_t id : tok.encode(run, false)) ids.push_back(id);
        run.clear();
      };
      for (const auto& [t, q] : pieces) {
        if (q) {
          flush();
          ids.push_back(vocab->quote);
        } else {
          run += t;
        }
      }
      flush();
      ids.push_back(vocab->call_close);
      serve::ToolGrammar gr(vocab, kTools);
      for (size_t i = 0; i < ids.size(); ++i) {
        const bool constrained = gr.active();
        bool listed = !constrained;
        if (constrained)
          for (int32_t a : gr.allowed()) listed |= a == ids[i];
        if (!listed || !gr.accept(ids[i])) {
          if (++bad_calls <= 5) {
            std::string shown;
            for (const auto& [t, q] : pieces) shown += q ? "<|\"|>" : t;
            std::fprintf(stderr, "refused token %zu (%s) of %s\n", i, tok.piece(ids[i]).c_str(), shown.c_str());
          }
          break;
        }
      }
    }

    int closed = 0, bad_walks = 0, capped = 0;
    for (int w = 0; w < walks; ++w) {
      serve::ToolGrammar gr(vocab, kTools);
      text::OutputParser parser(tok);
      std::vector<int32_t> seq{vocab->call_open};
      gr.accept(vocab->call_open);
      bool done = false;
      for (int step = 0; step < 400 && !done; ++step) {
        const auto& allowed = gr.allowed();
        if (allowed.empty()) {
          ++bad_walks;
          std::fprintf(stderr, "walk %d: nothing allowed after %zu tokens\n", w, seq.size());
          break;
        }
        // Leaning towards what closes things, so that walks reach the end: the closing token
        // when allowed, else often a quote or a single structural character.
        int32_t pick = allowed[g.rng() % allowed.size()];
        std::vector<int32_t> closers;
        for (int32_t c : allowed) {
          const std::string& b = vocab->bytes[size_t(c)];
          if (c == vocab->call_close || c == vocab->quote || (b.size() == 1 && std::string("}],:0").find(b[0]) != std::string::npos))
            closers.push_back(c);
        }
        if (!closers.empty() && g.rng() % 3 != 0) pick = closers[g.rng() % closers.size()];
        for (int32_t c : closers)
          if (c == vocab->call_close && g.rng() % 10 != 0) pick = c;
        if (!gr.accept(pick)) {
          ++bad_walks;
          std::fprintf(stderr, "walk %d: an allowed token was refused\n", w);
          break;
        }
        seq.push_back(pick);
        done = pick == vocab->call_close;
      }
      if (!done) {
        ++capped;
        continue;
      }
      ++closed;
      text::OutputDelta d;
      for (int32_t id : seq) {
        auto x = parser.push(id);
        for (auto& c : x.tool_calls) d.tool_calls.push_back(std::move(c));
        for (auto& e : x.errors) d.errors.push_back(std::move(e));
      }
      bool known = d.tool_calls.size() == 1;
      if (known) {
        known = false;
        for (const auto& t : kTools) known |= t == d.tool_calls[0].name;
      }
      if (!d.errors.empty() || !known) {
        if (++bad_walks <= 5) {
          std::string text;
          for (int32_t id : seq) text += id == vocab->quote ? "<|\"|>" : tok.piece(id);
          std::fprintf(stderr, "walk %d does not parse (%s): %s\n", w, d.errors.empty() ? "unknown tool" : d.errors[0].c_str(), text.c_str());
        }
      }
    }
    std::printf("calls: %d of %d accepted token by token\n", calls - bad_calls, calls);
    std::printf("walks: %d closed (%d hit the length cap), %d failed to parse or got stuck\n", closed, capped, bad_walks);
    return bad_calls || bad_walks ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-grammar-test: %s\n", e.what());
    return 1;
  }
}
