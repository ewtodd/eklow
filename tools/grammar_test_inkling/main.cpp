// eklow-grammar-test-inkling: the gate for Inkling 2 Mini's tool-call grammar (serve/tool_grammar_inkling.hpp) against
// llama.cpp's grammar sampler (c74c71a6), through eklow-grammar-oracle (tools/ref/oracle/grammar_oracle.cpp). Host only.
//
//   requests --m0-cases CASES.json --out DIR [--fuzz N] [--seed S]
//       DIR/requests.json: chat requests for eklow-chat-oracle (M0's tool requests, and N fuzzed ones: random tool names, Unicode,
//       quotes, escapes, prefixes of each other; auto and required, parallel and not), DIR/meta.json: each one's names and choice.
//   gen --model GGUF --chat CHAT.json --meta DIR/meta.json --m0-chat M0_CHAT.json --out DIR [--per N] [--shards K] [--seed S]
//       CHAT.json is the chat oracle's answer to requests.json (the grammars). Writes DIR/g/ID.gbnf and DIR/cases-K.txt: for each
//       grammar token sequences of every kind (see below) for the grammar oracle.
//   check --model GGUF --dir DIR [--shard K]
//       reads DIR/cases-K.txt and the oracle's DIR/results-K.txt; at every step of every sequence eklow's allowed set must equal
//       llama.cpp's, accept() must agree on the next token, and the turn may end exactly where llama.cpp lets it. Prints counts.
//   check-vocab --model GGUF --vocab VOCAB.txt
//       the oracle's --dump-vocab against the grammar's view of every token's text and the end-of-generation tokens.
//   bench --model GGUF --dir DIR
//       replays the sequences with eklow alone and times allowed() and accept().
//
// Sequences, per grammar: (valid) model-like outputs written as text (blocks of thinking and text, calls with the spacing, the
// escapes, the numbers, the nesting and the Unicode JSON allows, and calls the grammar must refuse) and tokenized by eklow's
// tokenizer, some with their multi-byte characters split into single-byte tokens; (mutated) the same with tokens replaced,
// inserted, deleted, swapped, cut; (walk) random walks through eklow's allowed sets, steered by such an output and by pools of
// JSON-ish, byte and special tokens; (m0) the model outputs of M0's chat cases. Every request, name and value is synthetic.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "serve/tool_grammar_inkling.hpp"
#include "text/tokenizer_inkling.hpp"

using namespace eklow;
namespace fs = std::filesystem;
namespace ink = text::inkling;

namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void spit(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary);
  f << data;
}

int64_t as_int(const json::Value& v) { return std::stoll(v.int_digits()); }

struct Meta {
  std::string id;
  bool required = false;
  int parallel = -1;  // -1: unknown, read it off the grammar
  std::vector<std::string> names;
};

std::vector<Meta> read_meta(const std::string& path) {
  std::vector<Meta> out;
  const json::Value doc = json::parse(slurp(path));
  for (const auto& m : doc.items()) {
    Meta x;
    x.id = m.find("id")->as_string();
    x.required = m.find("required")->as_bool();
    x.parallel = m.find("parallel")->is_null() ? -1 : int(m.find("parallel")->as_bool());
    for (const auto& n : m.find("names")->items()) x.names.push_back(n.as_string());
    out.push_back(std::move(x));
  }
  return out;
}

// ---- random text ----

struct Rng {
  std::mt19937_64 g;
  explicit Rng(uint64_t seed) : g(seed) {}
  size_t below(size_t n) { return n ? size_t(g() % n) : 0; }
  bool chance(double p) { return double(g() >> 11) * (1.0 / 9007199254740992.0) < p; }
  template <class C>
  const typename C::value_type& pick(const C& c) { return c[below(c.size())]; }
};

// nlohmann's dump of a string, between the quotes (what llama.cpp's grammar and the model's calls write for a name).
std::string escape_body(const std::string& s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          o += buf;
        } else {
          o += char(c);
        }
    }
  }
  return o;
}

const std::vector<std::string> kWords = {"read", "file", "path", "run", "cmd", "get", "time", "edit", "x", "a", "ab", "abc", "tool", "q",
                                         "list", "dir", "search", "write", "do", "it", "call", "go", "id", "key", "val"};
const std::vector<std::string> kOddBits = {" ", "\"", "\\", "/", "\n", "\t", "\x01", "\x7f", "{", "}", "[", "]", ":", ",", "<", ">", "|", "-", ".",
                                           "\xc3\xa9", "\xe6\x97\xa5\xe6\x9c\xac", "\xf0\x9f\x98\x80", "\xe2\x82\xac", "e\xcc\x81",
                                           "<|end_message|>", "<|content_invoke_tool_json|>", "\xd0\xb6", "0", "1", "_"};

std::string random_name(Rng& r, bool odd) {
  std::string n;
  const int parts = 1 + int(r.below(3));
  for (int i = 0; i < parts; ++i) {
    if (i) n += r.chance(0.8) ? "_" : "";
    n += r.pick(kWords);
  }
  if (odd) {
    const int k = 1 + int(r.below(3));
    for (int i = 0; i < k; ++i) {
      const std::string& b = r.pick(kOddBits);
      if (r.chance(0.2)) n.insert(0, b);
      else n += b;
    }
  }
  return n;
}

// ---- requests ----

void write_tool(json::Writer& w, const std::string& name) {
  w.begin_object();
  w.kv("type", "function");
  w.key("function").begin_object();
  w.kv("name", name);
  w.kv("description", "A synthetic tool.");
  w.key("parameters").begin_object();
  w.kv("type", "object");
  w.key("properties").begin_object().end_object();
  w.end_object();
  w.end_object();
  w.end_object();
}

int cmd_requests(const std::string& m0, const std::string& dir, int fuzz, uint64_t seed) {
  fs::create_directories(dir);
  json::Writer req(false), meta(false);
  req.begin_array();
  meta.begin_array();
  int n_m0 = 0;
  const json::Value doc = json::parse(slurp(m0));
  for (const auto& c : doc.items()) {
    const json::Value* r = c.find("request");
    const json::Value* tools = r ? r->find("tools") : nullptr;
    if (!tools || !tools->is_array() || tools->items().empty()) continue;
    const json::Value* tc = r->find("tool_choice");
    if (tc && tc->is_string() && tc->as_string() == "none") continue;
    if (r->find("response_format")) continue;
    req.raw(json::dump(c));
    meta.begin_object();
    meta.kv("id", c.find("id")->as_string());
    meta.kv("required", tc && tc->is_string() && tc->as_string() == "required");
    meta.key("parallel").null();
    meta.key("names").begin_array();
    for (const auto& t : tools->items()) meta.value(t.find("function")->find("name")->as_string());
    meta.end_array();
    meta.end_object();
    ++n_m0;
  }
  Rng r(seed);
  for (int i = 0; i < fuzz; ++i) {
    const bool required = r.chance(0.45), parallel = r.chance(0.5), odd = r.chance(0.5);
    const int ntools = r.chance(0.15) ? 1 + int(r.below(30)) : 1 + int(r.below(5));
    std::set<std::string> names;
    while (int(names.size()) < ntools) {
      std::string nm = random_name(r, odd && r.chance(0.7));
      if (r.chance(0.2) && !names.empty()) nm = *names.begin() + r.pick(kWords);  // a name that starts with another
      names.insert(nm);
    }
    const std::string id = "fz" + std::to_string(i);
    req.begin_object();
    req.kv("id", id);
    req.key("request").begin_object();
    req.key("messages").begin_array().begin_object().kv("role", "user").kv("content", "hi").end_object().end_array();
    req.key("tools").begin_array();
    for (const auto& nm : names) write_tool(req, nm);
    req.end_array();
    if (required) req.kv("tool_choice", "required");
    req.kv("parallel_tool_calls", parallel);
    req.end_object();
    req.end_object();
    meta.begin_object();
    meta.kv("id", id);
    meta.kv("required", required);
    meta.kv("parallel", parallel);
    meta.key("names").begin_array();
    for (const auto& nm : names) meta.value(nm);
    meta.end_array();
    meta.end_object();
  }
  req.end_array();
  meta.end_array();
  spit(dir + "/requests.json", req.str());
  spit(dir + "/meta.json", meta.str());
  std::printf("requests: %d from M0, %d fuzzed -> %s/requests.json\n", n_m0, fuzz, dir.c_str());
  return 0;
}

// ---- sequences ----

struct Gen {
  Rng r;
  const text::InklingTokenizer& tok;
  std::vector<int32_t> byte_tok;      // the single-byte token of each byte
  std::vector<int32_t> structural;    // tokens of JSON's punctuation, digits, letters and spaces
  std::vector<int32_t> partial_ends;  // tokens that end inside a character
  std::vector<int32_t> multibyte;     // tokens of non-ASCII text
  std::vector<int32_t> specials;      // the 60 added tokens

  Gen(uint64_t seed, const text::InklingTokenizer& t, const serve::InklingToolVocab& v) : r(seed), tok(t), byte_tok(256, -1) {
    static const std::string js = "{}[]:,\" \n\t0123456789.-+eE\\/ntrufalsb";
    for (size_t id = 0; id < v.n; ++id) {
      if (!tok.is_special(int32_t(id))) {
        const std::string raw = tok.bytes(int32_t(id));
        if (raw.size() == 1) byte_tok[uint8_t(raw[0])] = int32_t(id);
      }
      const std::string& p = v.piece[id];
      if (p.empty()) continue;
      if (!tok.is_special(int32_t(id))) {
        bool all_js = true;
        for (char c : p) all_js = all_js && js.find(c) != std::string::npos;
        if (all_js && p.size() <= 4) structural.push_back(int32_t(id));
        if (v.end_pu[id].n_remain > 0) partial_ends.push_back(int32_t(id));
        if (uint8_t(p[0]) >= 0x80) multibyte.push_back(int32_t(id));
      }
    }
    specials = tok.special_ids();
  }

  std::string sp() {
    switch (r.below(40)) {
      case 0: return " ";
      case 1: return "\n";
      case 2: return "\n\n";
      case 3: return "\n  ";
      case 4: return "\t";
      case 5: return "\n\n\t\t";
      case 6: return std::string(1 + r.below(24), ' ');
      case 7: return "\n\n\n";
      case 8: return "  ";
      case 9: return std::string(20, ' ');
      case 10: return "\n" + std::string(r.below(23), '\t');
      default: return "";
    }
  }

  std::string str_body() {
    std::string s;
    for (int n = int(r.below(6)); n > 0; --n) {
      switch (r.below(16)) {
        case 0: s += "\\n"; break;
        case 1: s += "\\\""; break;
        case 2: s += "\\\\"; break;
        case 3: s += "\\u00e9"; break;
        case 4: s += r.chance(0.5) ? "\\u12G4" : "\\uD83D\\uDE00"; break;
        case 5: s += r.chance(0.5) ? "\\ " : "\\x"; break;
        case 6: s += "\n"; break;
        case 7: s += "\t"; break;
        case 8: {
          const std::string& b = r.pick(kOddBits);
          s += b == "\"" ? "q" : b;
          break;
        }
        case 9: s += "\xf0\x9f\x98\x80\xe6\x97\xa5\xe6\x9c\xac\xd0\xb6"; break;
        case 10: s += "<|end_message|>"; break;
        case 11: s += "/"; break;
        default: s += r.pick(kWords) + (r.chance(0.5) ? " " : ""); break;
      }
    }
    // an unescaped quote or a lone backslash now and then (the grammar must end or refuse the string there)
    if (r.chance(0.03)) s += "\"";
    if (r.chance(0.02)) s += "\\";
    return s;
  }

  std::string number() {
    static const std::vector<std::string> n = {"0", "-0", "12", "-3", "0.5", "1e5", "1E+5", "2.5e-3", "01", "1.", "-", "+1", ".5", "1e", "1e+",
                                               "123456789012345678901234567890", "-0.0", "7", "100", "3.14159", "9e-9", "0e0", "-12.5E+10"};
    return r.chance(0.5) ? std::to_string(int(r.below(1000)) - 500) : r.pick(n);
  }

  std::string value(int depth, int max) {
    const size_t k = r.below(depth < max ? 9 : 6);
    switch (k) {
      case 0:
      case 1:
      case 4: return "\"" + str_body() + "\"";
      case 2:
      case 5: return number();
      case 3: return r.pick(std::vector<std::string>{"true", "false", "null", "true", "false", "null", "tru", "True", "nul", "nulll"});
      case 6:
      case 7: {  // object
        std::string s = "{" + sp();
        const int n = int(r.below(4));
        for (int i = 0; i < n; ++i) {
          if (i) s += sp() + "," + sp();
          s += "\"" + str_body() + "\"" + sp() + ":" + sp() + value(depth + 1, max);
        }
        return s + sp() + "}";
      }
      default: {  // array
        std::string s = "[" + sp();
        const int n = int(r.below(4));
        for (int i = 0; i < n; ++i) {
          if (i) s += sp() + "," + sp();
          s += value(depth + 1, max);
        }
        return s + sp() + "]";
      }
    }
  }

  std::string deep_nest() {
    std::string open, close;
    const int d = 5 + int(r.below(30));
    for (int i = 0; i < d; ++i) {
      if (r.chance(0.5)) {
        open += "[" + sp();
        close = "]" + close;
      } else {
        open += "{" + sp() + "\"k\"" + sp() + ":" + sp();
        close = "}" + close;
      }
    }
    return open + "1" + close;
  }

  // The text of one call block; flaws now and then.
  std::string call(const std::vector<std::string>& names, bool name_text) {
    const std::string name = r.chance(0.93) ? r.pick(names) : random_name(r, true);
    std::string t = "<|message_model|>";
    if (name_text && r.chance(0.7)) t += name;
    else if (r.chance(0.1)) t += r.pick(kWords);
    t += "<|content_invoke_tool_json|>";
    const int max = r.chance(0.08) ? 40 : 4;
    std::string args = r.chance(0.7) ? value(0, max) : "{" + sp() + "\"" + str_body() + "\"" + sp() + ":" + sp() + value(1, max) + sp() + "}";
    if (r.chance(0.04)) args = deep_nest();
    switch (r.below(25)) {
      case 0: t += "{\"args\":" + args + ",\"name\":\"" + escape_body(name) + "\"}"; break;
      case 1: t += "{\"name\":\"" + escape_body(name) + "\"}"; break;
      case 2: t += "{\"name\":\"" + escape_body(name) + "\",\"args\":" + args + ",\"x\":1}"; break;
      case 3: t += "{\"name\":\"" + escape_body(name) + "\",\"arguments\":" + args + "}"; break;
      default:
        t += "{" + sp() + "\"name\"" + sp() + ":" + sp() + "\"" + escape_body(name) + "\"" + sp() + "," + sp() + "\"args\"" + sp() + ":" + sp() +
             args + sp() + "}";
    }
    if (!r.chance(0.04)) t += "<|end_message|>";
    return t;
  }

  std::string block(bool think) {
    std::string t = std::string("<|message_model|>") + (think ? "<|content_thinking|>" : "<|content_text|>");
    for (int n = int(r.below(5)); n > 0; --n) t += r.chance(0.2) ? r.pick(kOddBits) : r.pick(kWords) + " ";
    if (!r.chance(0.03)) t += "<|end_message|>";
    return t;
  }

  // A model-like output as text.
  std::string output(const std::vector<std::string>& names, bool required, bool parallel) {
    std::string t;
    if (!required && r.chance(0.25)) {  // plain text first, no header
      for (int n = int(r.below(8)); n > 0; --n) t += r.pick(kWords) + " ";
    }
    const int blocks = int(r.below(3));
    for (int i = 0; i < blocks; ++i) t += block(r.chance(0.5));
    if (r.chance(0.85)) {
      const int n = parallel || r.chance(0.2) ? 1 + int(r.below(3)) : 1;
      for (int i = 0; i < n; ++i) t += call(names, required);
    }
    if (r.chance(0.5)) t += "<|content_model_end_sampling|>";
    return t;
  }

  std::vector<int32_t> encode(const std::string& t) {
    auto ids = tok.encode(t, true);
    if (r.chance(0.2)) ids = split_bytes(ids, 0.5);
    return ids;
  }

  // Multi-byte tokens as their single-byte tokens, each with probability p.
  std::vector<int32_t> split_bytes(const std::vector<int32_t>& ids, double p) {
    std::vector<int32_t> out;
    for (int32_t id : ids) {
      const std::string b = tok.is_special(id) ? "" : tok.bytes(id);
      bool wide = false;
      for (char c : b) wide = wide || uint8_t(c) >= 0x80;
      if (wide && r.chance(p)) {
        for (char c : b) out.push_back(byte_tok[uint8_t(c)]);
      } else {
        out.push_back(id);
      }
    }
    return out;
  }

  int32_t random_token() {
    switch (r.below(8)) {
      case 0: return r.pick(specials);
      case 1: return r.pick(byte_tok);
      case 2: return r.pick(partial_ends);
      case 3: return r.pick(multibyte);
      case 4:
      case 5: return r.pick(structural);
      default: return int32_t(r.below(tok.size()));
    }
  }

  void mutate(std::vector<int32_t>& s) {
    for (int k = 1 + int(r.below(3)); k > 0 && !s.empty(); --k) {
      const size_t i = r.below(s.size());
      switch (r.below(7)) {
        case 0: s[i] = random_token(); break;
        case 1: s.insert(s.begin() + long(i), random_token()); break;
        case 2: s.erase(s.begin() + long(i)); break;
        case 3:
          if (i + 1 < s.size()) std::swap(s[i], s[i + 1]);
          break;
        case 4: s.resize(i); break;
        case 5: s.insert(s.begin() + long(i), s[i]); break;
        default: s = split_bytes(s, 0.7); break;
      }
    }
  }

  // A random walk through the grammar's own allowed sets, steered by a guide sequence.
  std::vector<int32_t> walk(serve::InklingToolGrammar g, const std::vector<int32_t>& guide, size_t max_len) {
    std::vector<int32_t> out;
    size_t gi = 0;
    while (out.size() < max_len) {
      const std::vector<int32_t>& al = g.allowed();
      if (al.empty()) break;
      int32_t t = -1;
      auto allowed_has = [&](int32_t x) { return std::binary_search(al.begin(), al.end(), x); };
      if (gi < guide.size() && r.chance(0.8)) {
        if (allowed_has(guide[gi])) {
          t = guide[gi];
          ++gi;
        } else if (r.chance(0.5)) {
          ++gi;
        }
      }
      if (t < 0) {
        for (int tries = 0; tries < 24 && t < 0; ++tries) {
          const int32_t c = random_token();
          if (allowed_has(c)) t = c;
        }
        if (t < 0) t = al[r.below(al.size())];
      }
      if (!g.accept(t)) break;  // cannot happen: t is allowed
      out.push_back(t);
      if (g.ended()) break;
      if (g.can_end() && r.chance(0.12)) {
        out.push_back(r.chance(0.8) ? ink::kModelEndSampling : ink::kEndOfText);
        break;
      }
    }
    return out;
  }
};

std::string seq_line(const std::vector<int32_t>& s) {
  std::string l = "S " + std::to_string(s.size());
  for (int32_t t : s) l += " " + std::to_string(t);
  return l;
}

// A grammar's parallel flag, as its text shows it: auto has the rule tool-call; required's root ends with a repeat.
bool grammar_parallel(const std::string& g, bool required) {
  if (!required) return g.find("tool-call ::=") != std::string::npos;
  const size_t at = g.find("\nroot ::=");
  const size_t end = g.find('\n', at + 1);
  const std::string line = g.substr(at + 1, end - at - 1);
  return line.size() >= 2 && line.compare(line.size() - 2, 2, ")*") == 0 && line.find(" (<[200001]> (tool") != std::string::npos;
}

int cmd_gen(const std::string& model, const std::string& chat_path, const std::string& meta_path, const std::string& m0chat_path,
            const std::string& dir, int per, int shards, uint64_t seed) {
  auto f = gguf::File::open(model);
  const auto tok = text::InklingTokenizer::from_gguf(f);
  auto vocab = std::make_shared<const serve::InklingToolVocab>(tok);
  Gen gen(seed, tok, *vocab);

  std::map<std::string, std::string> grammars;
  const json::Value chat = json::parse(slurp(chat_path));
  for (const auto& c : chat.items()) {
    const json::Value* g = c.find("grammar");
    if (g && g->is_string() && !g->as_string().empty()) grammars[c.find("id")->as_string()] = g->as_string();
  }
  std::vector<std::vector<int32_t>> m0_outputs;
  const json::Value m0chat = json::parse(slurp(m0chat_path));
  for (const auto& c : m0chat.items()) {
    const json::Value* p = c.find("parsed");
    if (!p || !p->is_array()) continue;
    for (const auto& o : p->items()) {
      const json::Value* t = o.find("tokens");
      if (!t || !t->is_array()) continue;
      std::vector<int32_t> ids;
      for (const auto& x : t->items()) ids.push_back(int32_t(as_int(x)));
      if (!ids.empty()) m0_outputs.push_back(std::move(ids));
    }
  }
  fs::create_directories(dir + "/g");
  std::vector<std::ofstream> out;
  for (int k = 0; k < shards; ++k) out.emplace_back(dir + "/cases-" + std::to_string(k) + ".txt");
  std::map<std::string, int> cats;
  int ng = 0;
  for (const Meta& m : read_meta(meta_path)) {
    auto it = grammars.find(m.id);
    if (it == grammars.end()) {
      std::fprintf(stderr, "no grammar for %s\n", m.id.c_str());
      continue;
    }
    const std::string& gtext = it->second;
    const bool parallel = grammar_parallel(gtext, m.required);
    if (m.parallel >= 0 && bool(m.parallel) != parallel)
      std::fprintf(stderr, "note: %s asked parallel=%d, the grammar has parallel=%d\n", m.id.c_str(), m.parallel, int(parallel));
    const int shard = ng++ % shards;
    spit(dir + "/g/" + m.id + ".gbnf", gtext);
    std::ofstream& o = out[size_t(shard)];
    o << "M " << m.id << ' ' << int(m.required) << ' ' << int(parallel) << '\n';
    o << "G " << dir << "/g/" << m.id << ".gbnf\n";
    const serve::InklingToolGrammar proto(vocab, m.required ? serve::InklingToolGrammar::Choice::Required : serve::InklingToolGrammar::Choice::Auto,
                                          m.names, parallel);
    auto emit = [&](const char* cat, const std::vector<int32_t>& s) {
      o << "C " << cat << '\n' << seq_line(s) << '\n';
      ++cats[cat];
    };
    const bool is_m0 = m.id.rfind("fz", 0) != 0;
    for (int i = 0; i < per; ++i) {
      std::string txt = gen.output(m.names, m.required, parallel);
      std::vector<int32_t> ids = gen.encode(txt);
      if (ids.size() > 600) ids.resize(600);
      const size_t kind = gen.r.below(10);
      if (kind < 3) {
        emit("valid", ids);
      } else if (kind < 5) {
        gen.mutate(ids);
        emit("mutated", ids);
      } else {
        if (gen.r.chance(0.3)) gen.mutate(ids);
        emit("walk", gen.walk(proto, ids, 500));
      }
    }
    for (size_t i = 0; i < m0_outputs.size(); ++i)
      if (is_m0 || gen.r.chance(0.15)) emit("m0", m0_outputs[i]);
    // Whole-vocabulary probes: each of the special tokens alone, and single random tokens.
    for (int32_t s : gen.specials) emit("single", {s});
    for (int i = 0; i < 20; ++i) emit("single", {int32_t(gen.r.below(tok.size()))});
  }
  size_t total = 0;
  for (const auto& [k, v] : cats) {
    std::printf("  %-8s %d sequences\n", k.c_str(), v);
    total += size_t(v);
  }
  std::printf("gen: %d grammars, %zu sequences in %d shards -> %s\n", ng, total, shards, dir.c_str());
  return 0;
}

// ---- reading the oracle's results ----

struct Step {
  bool complement = false;
  std::vector<int32_t> ids;
};

struct OracleSeq {
  std::vector<Step> steps;
  std::string end;  // "ok", "rej J", "eog J", "err J ..."
};

std::vector<OracleSeq> read_results(const std::string& path) {
  std::vector<OracleSeq> out;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::string line;
  OracleSeq cur;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == 'A' || line[0] == 'N') {
      std::istringstream is(line.substr(2));
      size_t k = 0;
      is >> k;
      Step s;
      s.complement = line[0] == 'N';
      s.ids.resize(k);
      for (auto& x : s.ids) is >> x;
      cur.steps.push_back(std::move(s));
    } else if (line[0] == 'Z') {
      cur.end = line.substr(2);
      out.push_back(std::move(cur));
      cur = OracleSeq{};
    }
  }
  return out;
}

struct CaseSeq {
  std::string grammar_id;
  bool required = false, parallel = false;
  std::string category;
  std::vector<int32_t> tokens;
};

std::vector<CaseSeq> read_cases(const std::string& path) {
  std::vector<CaseSeq> out;
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::string line, id, cat;
  bool req = false, par = false;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == 'M') {
      std::istringstream is(line.substr(2));
      int a, b;
      is >> id >> a >> b;
      req = a, par = b;
    } else if (line[0] == 'C') {
      cat = line.substr(2);
    } else if (line[0] == 'S') {
      std::istringstream is(line.substr(2));
      size_t n;
      is >> n;
      CaseSeq c{id, req, par, cat, std::vector<int32_t>(n)};
      for (auto& t : c.tokens) is >> t;
      out.push_back(std::move(c));
    }
  }
  return out;
}

std::string hexpiece(const serve::InklingToolVocab& v, int32_t id) {
  static const char* hx = "0123456789abcdef";
  std::string s;
  for (unsigned char c : v.piece[size_t(id)]) {
    s += hx[c >> 4];
    s += hx[c & 15];
  }
  return s;
}

using Clock = std::chrono::steady_clock;

int cmd_check(const std::string& model, const std::string& dir, int shard) {
  auto f = gguf::File::open(model);
  const auto tok = text::InklingTokenizer::from_gguf(f);
  auto vocab = std::make_shared<const serve::InklingToolVocab>(tok);
  std::map<std::string, Meta> metas;
  for (auto& m : read_meta(dir + "/meta.json")) metas[m.id] = m;
  auto cases = read_cases(dir + "/cases-" + std::to_string(shard) + ".txt");
  const auto results = read_results(dir + "/results-" + std::to_string(shard) + ".txt");
  // A run that was stopped leaves the sequences it finished (each ends in a Z line): those are compared, the rest ignored.
  if (results.size() < cases.size()) cases.resize(results.size());
  if (cases.size() != results.size()) {
    std::fprintf(stderr, "shard %d: %zu sequences, %zu results\n", shard, cases.size(), results.size());
    return 1;
  }
  size_t steps = 0, mismatched_sets = 0, mismatched_accept = 0, bad_end = 0, bad_unconstrained = 0, rejected_seq = 0, eog_seq = 0, err_seq = 0,
         tokens_fed = 0, ids_compared = 0;
  std::map<std::string, size_t> by_cat;
  double allowed_ns = 0;
  size_t allowed_calls = 0;
  std::ofstream log(dir + "/mismatch-" + std::to_string(shard) + ".log");
  int logged = 0;
  const size_t n = vocab->n;
  std::set<std::string> grammars_seen;
  for (size_t q = 0; q < cases.size(); ++q) {
    const CaseSeq& c = cases[q];
    const OracleSeq& o = results[q];
    const Meta& m = metas.at(c.grammar_id);
    grammars_seen.insert(c.grammar_id);
    serve::InklingToolGrammar g(vocab, c.required ? serve::InklingToolGrammar::Choice::Required : serve::InklingToolGrammar::Choice::Auto,
                                m.names, c.parallel);
    ++by_cat[c.category];
    bool failed = false;
    auto report = [&](size_t j, const std::string& what) {
      failed = true;
      if (logged++ < 40) {
        log << "grammar " << c.grammar_id << " category " << c.category << " seq " << q << " step " << j << ": " << what << "\n  tokens:";
        for (size_t i = 0; i < std::min<size_t>(j, c.tokens.size()); ++i) log << ' ' << c.tokens[i];
        log << "\n";
      }
    };
    for (size_t j = 0; j < o.steps.size() && !failed; ++j) {
      const auto t0 = Clock::now();
      const std::vector<int32_t> mine = g.allowed();
      allowed_ns += std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
      ++allowed_calls;
      ++steps;
      const Step& s = o.steps[j];
      bool same;
      std::vector<int32_t> only_mine, only_theirs;
      if (!s.complement) {
        same = mine == s.ids;
        if (!same) {
          std::set_difference(mine.begin(), mine.end(), s.ids.begin(), s.ids.end(), std::back_inserter(only_mine));
          std::set_difference(s.ids.begin(), s.ids.end(), mine.begin(), mine.end(), std::back_inserter(only_theirs));
        }
      } else {
        // theirs = everything but s.ids
        same = mine.size() == n - s.ids.size();
        for (int32_t x : s.ids)
          if (std::binary_search(mine.begin(), mine.end(), x)) {
            same = false;
            only_mine.push_back(x);
          }
        if (!same) {
          for (int32_t x = 0; size_t(x) < n && only_theirs.size() < 20; ++x) {
            if (std::binary_search(s.ids.begin(), s.ids.end(), x)) continue;
            if (!std::binary_search(mine.begin(), mine.end(), x)) only_theirs.push_back(x);
          }
        }
      }
      ids_compared += mine.size();
      if (!same) {
        ++mismatched_sets;
        std::string w = "allowed sets differ: eklow " + std::to_string(mine.size()) + " ids, llama.cpp " +
                        (s.complement ? std::to_string(n - s.ids.size()) : std::to_string(s.ids.size())) + "; only eklow:";
        for (size_t i = 0; i < std::min<size_t>(only_mine.size(), 8); ++i) w += " " + std::to_string(only_mine[i]) + "(" + hexpiece(*vocab, only_mine[i]) + ")";
        w += "; only llama.cpp:";
        for (size_t i = 0; i < std::min<size_t>(only_theirs.size(), 8); ++i) w += " " + std::to_string(only_theirs[i]) + "(" + hexpiece(*vocab, only_theirs[i]) + ")";
        report(j, w);
        break;
      }
      // unconstrained(): the whole vocabulary but what llama.cpp never accepts
      if (g.unconstrained()) {
        size_t want = 2;  // the end-of-generation tokens
        for (size_t id = 0; id < n; ++id) want += !vocab->dead[id] && vocab->end_pu[id].n_remain >= 0 && id != 199999 && id != 200006;
        if (mine.size() != want) {
          ++bad_unconstrained;
          report(j, "unconstrained() but the set is not the whole vocabulary");
          break;
        }
      }
      if (j < c.tokens.size()) {
        const int32_t t = c.tokens[j];
        const bool theirs = s.complement ? !std::binary_search(s.ids.begin(), s.ids.end(), t) : std::binary_search(s.ids.begin(), s.ids.end(), t);
        const bool is_end = t == ink::kModelEndSampling || t == ink::kEndOfText;
        const bool before_can_end = g.can_end();
        const bool mine_ok = g.accept(t);
        if (mine_ok != theirs) {
          ++mismatched_accept;
          report(j, "accept(" + std::to_string(t) + ") = " + std::to_string(mine_ok) + ", llama.cpp allows " + std::to_string(theirs));
          break;
        }
        if (mine_ok) {
          ++tokens_fed;
          if (is_end && (!g.ended() || !before_can_end)) {
            ++bad_end;
            report(j, "end token accepted but ended()/can_end() disagree");
            break;
          }
        } else if (g.allowed() != mine) {
          ++mismatched_accept;
          report(j, "a refused token changed the state");
          break;
        }
      }
    }
    if (!failed) {
      const std::string& e = o.end;
      if (e.rfind("rej", 0) == 0) ++rejected_seq;
      else if (e.rfind("eog", 0) == 0) ++eog_seq;
      else if (e.rfind("err", 0) == 0) {
        ++err_seq;
        report(0, "llama.cpp threw: " + e);
      }
    }
  }
  const size_t bad = mismatched_sets + mismatched_accept + bad_end + bad_unconstrained + err_seq;
  std::printf("SUMMARY shard=%d grammars=%zu sequences=%zu steps=%zu tokens_accepted=%zu ids_compared=%zu rejected_sequences=%zu "
              "ended_sequences=%zu set_mismatches=%zu accept_mismatches=%zu end_errors=%zu unconstrained_errors=%zu oracle_errors=%zu "
              "allowed_us_mean=%.2f\n",
              shard, grammars_seen.size(), cases.size(), steps, tokens_fed, ids_compared, rejected_seq, eog_seq, mismatched_sets,
              mismatched_accept, bad_end, bad_unconstrained, err_seq, allowed_calls ? allowed_ns / double(allowed_calls) / 1000.0 : 0.0);
  for (const auto& [k, v] : by_cat) std::printf("  CAT %s %zu\n", k.c_str(), v);
  return bad ? 1 : 0;
}

int cmd_check_vocab(const std::string& model, const std::string& vocab_path) {
  auto f = gguf::File::open(model);
  const auto tok = text::InklingTokenizer::from_gguf(f);
  serve::InklingToolVocab v(tok);
  std::ifstream in(vocab_path);
  std::string line;
  size_t n = 0, bad = 0;
  while (std::getline(in, line)) {
    std::istringstream is(line);
    int64_t id;
    int eog;
    std::string hex;
    is >> id >> eog;
    is >> hex;
    ++n;
    const bool mine_eog = id == v.eog[0] || id == v.eog[1];
    // llama.cpp reads the piece as a C string, and rejects a piece that is empty or starts with NUL
    const std::string mine = hexpiece(v, int32_t(id));
    std::string theirs = hex;
    for (size_t i = 0; i + 1 < theirs.size(); i += 2)
      if (theirs.compare(i, 2, "00") == 0) {
        theirs.resize(i);
        break;
      }
    const bool theirs_dead = hex.empty() || hex.compare(0, 2, "00") == 0;
    if (mine_eog != bool(eog) || mine != theirs || bool(v.dead[size_t(id)]) != theirs_dead) {
      if (bad++ < 10)
        std::printf("token %ld: eklow eog=%d piece=%s dead=%d, llama.cpp eog=%d piece=%s\n", long(id), int(mine_eog), mine.c_str(), int(v.dead[size_t(id)]), eog,
                    hex.c_str());
    }
  }
  std::printf("check-vocab: %zu tokens, %zu differ (piece bytes as the grammar reads them, dead, end-of-generation)\n", n, bad);
  return bad || n != tok.size() ? 1 : 0;
}

int cmd_bench(const std::string& model, const std::string& dir) {
  auto f = gguf::File::open(model);
  const auto tok = text::InklingTokenizer::from_gguf(f);
  auto vocab = std::make_shared<const serve::InklingToolVocab>(tok);
  std::map<std::string, Meta> metas;
  for (auto& m : read_meta(dir + "/meta.json")) metas[m.id] = m;
  std::vector<CaseSeq> cases;
  for (int k = 0; fs::exists(dir + "/cases-" + std::to_string(k) + ".txt"); ++k) {
    auto c = read_cases(dir + "/cases-" + std::to_string(k) + ".txt");
    cases.insert(cases.end(), c.begin(), c.end());
  }
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<double> allowed_us, accept_us;
    size_t steps = 0;
    for (const CaseSeq& c : cases) {
      serve::InklingToolGrammar g(vocab, c.required ? serve::InklingToolGrammar::Choice::Required : serve::InklingToolGrammar::Choice::Auto,
                                  metas.at(c.grammar_id).names, c.parallel);
      for (int32_t t : c.tokens) {
        auto t0 = Clock::now();
        const auto& al = g.allowed();
        (void)al.size();
        auto t1 = Clock::now();
        const bool ok = g.accept(t);
        auto t2 = Clock::now();
        allowed_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        accept_us.push_back(std::chrono::duration<double, std::micro>(t2 - t1).count());
        ++steps;
        if (!ok) break;
      }
    }
    auto pct = [](std::vector<double> v, double p) {
      std::sort(v.begin(), v.end());
      return v.empty() ? 0.0 : v[size_t(p * double(v.size() - 1))];
    };
    auto mean = [](const std::vector<double>& v) {
      double s = 0;
      for (double x : v) s += x;
      return v.empty() ? 0.0 : s / double(v.size());
    };
    std::printf("pass %d (%s cache): %zu steps; allowed() mean %.2f us, p50 %.2f, p99 %.2f, p99.9 %.1f, max %.0f; accept() mean %.2f us, p99 %.2f, max %.0f\n",
                pass, pass == 0 ? "cold" : "warm", steps, mean(allowed_us), pct(allowed_us, 0.5), pct(allowed_us, 0.99), pct(allowed_us, 0.999),
                pct(allowed_us, 1.0), mean(accept_us), pct(accept_us, 0.99), pct(accept_us, 1.0));
    std::printf("  cache: %zu sets, %zu ids\n", vocab->cache.size(), vocab->cached_ids);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: eklow-grammar-test-inkling requests|gen|check|check-vocab|bench ... (see the file's header)\n");
    return 2;
  }
  const std::string cmd = argv[1];
  std::map<std::string, std::string> o;
  for (int i = 2; i + 1 < argc; i += 2) o[argv[i]] = argv[i + 1];
  auto get = [&](const char* k, const char* d = "") { return o.count(k) ? o[k] : std::string(d); };
  try {
    if (cmd == "requests")
      return cmd_requests(get("--m0-cases"), get("--out"), std::atoi(get("--fuzz", "0").c_str()), std::strtoull(get("--seed", "1").c_str(), nullptr, 10));
    if (cmd == "gen")
      return cmd_gen(get("--model"), get("--chat"), get("--meta"), get("--m0-chat"), get("--out"), std::atoi(get("--per", "40").c_str()),
                     std::atoi(get("--shards", "16").c_str()), std::strtoull(get("--seed", "1").c_str(), nullptr, 10));
    if (cmd == "check") return cmd_check(get("--model"), get("--dir"), std::atoi(get("--shard", "0").c_str()));
    if (cmd == "check-vocab") return cmd_check_vocab(get("--model"), get("--vocab"));
    if (cmd == "bench") return cmd_bench(get("--model"), get("--dir"));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-grammar-test-inkling: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr, "unknown command %s\n", cmd.c_str());
  return 2;
}
