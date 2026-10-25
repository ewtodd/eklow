#include "serve/tool_grammar.hpp"

#include <stdexcept>

namespace eklow::serve {

namespace {

constexpr int kQuote = -1;

bool ws(int c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

// A word value (true, false, null or a number) as the parser reads it: up to a terminator, with
// trailing spaces and newlines dropped. Open (not `complete`, no trailing space yet): it must be a
// prefix of a valid word.
bool word_ok(const std::string& w, bool complete) {
  size_t end = w.size();
  while (end > 0 && (w[end - 1] == ' ' || w[end - 1] == '\n')) --end;
  const std::string core = w.substr(0, end);
  if (core.empty()) return false;
  for (char c : core)
    if (ws(c)) return false;
  const bool open = !complete && end == w.size();
  for (const std::string lit : {"true", "false", "null"})
    if (core == lit || (open && lit.compare(0, core.size(), core) == 0)) return true;
  // Python's int and float notation, -?D+(.D+)?([eE][+-]?D+)?, as a DFA: 2, 4 and 7 accept.
  int st = 0;
  for (char c : core) {
    const bool d = c >= '0' && c <= '9';
    switch (st) {
      case 0: st = c == '-' ? 1 : d ? 2 : -1; break;
      case 1: st = d ? 2 : -1; break;
      case 2: st = d ? 2 : c == '.' ? 3 : (c == 'e' || c == 'E') ? 5 : -1; break;
      case 3: st = d ? 4 : -1; break;
      case 4: st = d ? 4 : (c == 'e' || c == 'E') ? 5 : -1; break;
      case 5: st = (c == '+' || c == '-') ? 6 : d ? 7 : -1; break;
      case 6: st = d ? 7 : -1; break;
      case 7: st = d ? 7 : -1; break;
    }
    if (st < 0) return false;
  }
  return open || st == 2 || st == 4 || st == 7;
}

bool key_char(int c) { return c >= 0 && c != ':' && c != ',' && c != '{' && c != '}' && c != '[' && c != ']'; }

}  // namespace

ToolVocab::ToolVocab(const text::Tokenizer& tok) : by_first(256) {
  call_open = tok.id("<|tool_call>");
  call_close = tok.id("<tool_call|>");
  quote = tok.id("<|\"|>");
  if (call_open < 0 || call_close < 0 || quote < 0) throw std::runtime_error("tool grammar: the vocabulary lacks Gemma 4's tool-call tokens");
  bytes.resize(tok.size());
  for (int32_t id = 0; id < int32_t(tok.size()); ++id) {
    if (tok.is_special(id)) continue;
    bytes[size_t(id)] = tok.bytes(id);
    if (bytes[size_t(id)].empty()) continue;
    by_first[uint8_t(bytes[size_t(id)][0])].push_back(id);
    string_ok.push_back(id);
  }
  string_ok.push_back(quote);
}

ToolGrammar::ToolGrammar(std::shared_ptr<const ToolVocab> vocab, std::vector<std::string> tool_names)
    : v_(std::move(vocab)), names_(std::move(tool_names)) {
  for (const auto& n : names_) names_key_ += n + '\0';
}

bool ToolGrammar::step(State& s, int c, const std::vector<std::string>& names) {
  auto close = [&](char kind) {
    if (s.stack.empty() || s.stack.back() != kind) return false;
    s.stack.pop_back();
    s.mode = s.stack.empty() ? Mode::Done : Mode::After;
    return true;
  };
  // A value starting with c, in Value or ValueOrEnd.
  auto value = [&]() {
    if (ws(c)) return true;
    if (c == kQuote) return s.mode = Mode::String, true;
    if (c == '{') return s.stack.push_back('o'), s.mode = Mode::KeyOrEnd, true;
    if (c == '[') return s.stack.push_back('a'), s.mode = Mode::ValueOrEnd, true;
    if (c == ',' || c == '}' || c == ']' || c == ':') return false;
    s.buf.assign(1, char(c));
    s.mode = Mode::Word;
    return word_ok(s.buf, false);
  };
  switch (s.mode) {
    case Mode::Out: return true;
    case Mode::Prefix:
      if (s.prefix == 0 && ws(c)) return true;
      if (c != "call:"[s.prefix]) return false;
      if (++s.prefix == 5) s.mode = Mode::Name, s.buf.clear();
      return true;
    case Mode::Name:
      if (c == '{') {
        for (const auto& n : names)
          if (n == s.buf) return s.stack = "o", s.mode = Mode::KeyOrEnd, true;
        return false;
      }
      if (c == kQuote) return false;
      s.buf += char(c);
      for (const auto& n : names)
        if (n.compare(0, s.buf.size(), s.buf) == 0) return true;
      return false;
    case Mode::KeyOrEnd:
      if (c == '}') return close('o');
      [[fallthrough]];
    case Mode::KeyStart:
      if (ws(c)) return true;
      if (c == kQuote) return s.mode = Mode::QuotedKey, true;
      if (!key_char(c)) return false;
      s.buf.assign(1, char(c));
      s.mode = Mode::Key;
      return true;
    case Mode::Key:
      if (c == ':') return s.mode = Mode::Value, true;
      if (!key_char(c) || c == kQuote) return false;
      s.buf += char(c);
      return true;
    case Mode::QuotedKey:
      if (c == kQuote) s.mode = Mode::Colon;
      return true;
    case Mode::Colon:
      if (ws(c)) return true;
      return c == ':' ? (s.mode = Mode::Value, true) : false;
    case Mode::ValueOrEnd:
      if (c == ']') return close('a');
      return value();
    case Mode::Value: return value();
    case Mode::String:
      if (c == kQuote) s.mode = Mode::After;
      return true;
    case Mode::Word:
      if (c == ',' || c == '}' || c == ']') {
        if (!word_ok(s.buf, true)) return false;
        s.mode = Mode::After;
        return step(s, c, names);
      }
      if (c == kQuote) return false;
      s.buf += char(c);
      return word_ok(s.buf, false);
    case Mode::After:
      if (ws(c)) return true;
      if (c == ',') {
        if (s.stack.empty()) return false;
        s.mode = s.stack.back() == 'o' ? Mode::KeyStart : Mode::Value;
        return true;
      }
      if (c == '}') return close('o');
      if (c == ']') return close('a');
      return false;
    case Mode::Done: return ws(c);
  }
  return false;
}

bool ToolGrammar::feed(State& s, int32_t id) const {
  if (s.mode == Mode::Out) {
    if (id == v_->call_open) s = State{Mode::Prefix, 0, "", ""};
    return true;
  }
  if (id == v_->call_close) {
    if (s.mode != Mode::Done) return false;
    s = State{};
    return true;
  }
  if (id == v_->quote) return step(s, kQuote, names_);
  const std::string& b = v_->bytes[size_t(id)];
  if (b.empty()) return false;  // another special token
  for (unsigned char c : b)
    if (!step(s, c, names_)) return false;
  return true;
}

const std::vector<int32_t>& ToolGrammar::allowed() {
  if (allowed_valid_) return allowed_;
  allowed_.clear();
  if (st_.mode == Mode::String) {
    allowed_valid_ = true;
    return allowed_ = v_->string_ok;
  }
  std::string key(1, char('A' + int(st_.mode)));
  key += st_.stack + '|';
  if (st_.mode == Mode::Prefix) key += char('0' + st_.prefix);
  if (st_.mode == Mode::Name) key += st_.buf + '|' + names_key_;
  if (st_.mode == Mode::Word) key += st_.buf;
  {
    std::lock_guard lk(v_->mu);
    if (auto it = v_->cache.find(key); it != v_->cache.end()) {
      allowed_valid_ = true;
      return allowed_ = it->second;
    }
  }
  for (int b = 0; b < 256; ++b) {
    State probe = st_;
    if (!step(probe, b, names_)) continue;
    for (int32_t id : v_->by_first[size_t(b)]) {
      State s = st_;
      if (feed(s, id)) allowed_.push_back(id);
    }
  }
  for (int32_t id : {v_->quote, v_->call_close}) {
    State s = st_;
    if (feed(s, id)) allowed_.push_back(id);
  }
  allowed_valid_ = true;
  std::lock_guard lk(v_->mu);
  if (v_->cache.size() < 4096) v_->cache.emplace(key, allowed_);
  return allowed_;
}

bool ToolGrammar::accept(int32_t id) {
  State s = st_;
  if (!feed(s, id)) return false;
  st_ = std::move(s);
  allowed_valid_ = false;
  return true;
}

}  // namespace eklow::serve
