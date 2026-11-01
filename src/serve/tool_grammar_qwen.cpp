#include "serve/tool_grammar_qwen.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "text/pyfmt.hpp"

namespace eklow::serve {

namespace {

constexpr std::string_view kPrefix = "\n<function=";
constexpr std::string_view kParamEnd = "</parameter>";
constexpr std::string_view kTerm = "\n</parameter>\n";  // what follows a value
constexpr std::string_view kFunctionEnd = "</function>\n";
constexpr size_t kMaxDepth = 48;
constexpr size_t kCacheBudget = 24u << 20;  // token ids (96 MB)

bool ws(int c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }
bool digit(int c) { return c >= '0' && c <= '9'; }
bool hex(int c) { return digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

// A name the parser can read back: it ends at the first '>' and a newline would be taken for layout.
bool plain_name(const std::string& s) { return !s.empty() && s.find('>') == std::string::npos && s.find('\n') == std::string::npos; }

// The text the template writes for a scalar enum member, or false if the member is not one.
bool enum_text(const json::Value& v, std::string& out) {
  switch (v.kind()) {
    case json::Value::Kind::String: out = v.as_string(); return true;
    case json::Value::Kind::Int: out = v.int_digits(); return true;
    case json::Value::Kind::Float: out = text::py::float_repr(v.as_double()); return std::isfinite(v.as_double());
    case json::Value::Kind::Bool: out = v.as_bool() ? "true" : "false"; return true;
    case json::Value::Kind::Null: out = "null"; return true;
    default: return false;
  }
}

}  // namespace

QwenToolVocab::QwenToolVocab(const text::QwenTokenizer& tok) : by_first(256) {
  call_open = tok.id("<tool_call>");
  call_close = tok.id("</tool_call>");
  if (call_open < 0 || call_close < 0) throw std::runtime_error("tool grammar: the vocabulary lacks Qwen's tool-call tokens");
  bytes.resize(tok.size());
  for (int32_t id = 0; id < int32_t(tok.size()); ++id) {
    if (tok.is_special(id)) continue;
    bytes[size_t(id)] = tok.bytes(id);
    if (!bytes[size_t(id)].empty()) by_first[uint8_t(bytes[size_t(id)][0])].push_back(id);
  }
}

QwenToolGrammar::QwenToolGrammar(std::shared_ptr<const QwenToolVocab> vocab, std::shared_ptr<const text::ToolSpecs> tools) : v_(std::move(vocab)) {
  if (tools)
    for (const auto& t : tools->tools()) {
      if (!plain_name(t.name)) continue;
      ToolPlan tp;
      tp.name = t.name;
      for (const auto& p : t.params) {
        if (!plain_name(p.name)) continue;
        tp.headers.push_back("<parameter=" + p.name + ">\n");
        tp.required.push_back(p.required);
        Plan plan;
        plan.kinds = p.kinds;
        // An enum of scalars (none holding a newline or the parameter's closing tag, which would make the end of the
        // value ambiguous) fixes the text; anything else follows the declared kinds.
        std::vector<std::string> texts;
        bool scalar = !p.enum_values.empty();
        for (const auto& e : p.enum_values) {
          std::string s;
          if (!enum_text(e, s) || s.find('\n') != std::string::npos || s.find(kParamEnd) != std::string::npos) scalar = false;
          texts.push_back(std::move(s));
        }
        if (scalar) {
          plan.kind = Plan::Kind::Enum;
          plan.enums = std::move(texts);
        } else {
          plan.kind = p.kinds & text::kString ? Plan::Kind::Free : Plan::Kind::Json;
        }
        tp.plans.push_back(std::move(plan));
      }
      tools_.push_back(std::move(tp));
    }
  // Everything an allowed set depends on besides the byte state: the names, which parameters are required, each one's mode.
  for (const auto& t : tools_) {
    fp_ += t.name + '\1';
    for (size_t i = 0; i < t.headers.size(); ++i) {
      fp_ += t.headers[i] + char('0' + t.required[i]) + char('0' + int(t.plans[i].kind)) + std::to_string(t.plans[i].kinds);
      for (const auto& e : t.plans[i].enums) fp_ += '\2' + e;
      fp_ += '\3';
    }
    fp_ += '\4';
  }
  fp_ = std::to_string(std::hash<std::string>()(fp_)) + ':' + std::to_string(fp_.size());
}

bool QwenToolGrammar::all_required(const State& s) const {
  const ToolPlan& t = tools_[size_t(s.tool)];
  for (size_t i = 0; i < t.required.size(); ++i)
    if (t.required[i] && s.used[i] != '1') return false;
  return true;
}

// The first byte after "<parameter=KEY>\n": the state in which the parameter's value is written.
void QwenToolGrammar::begin_value(State& s) const {
  const Plan& p = tools_[size_t(s.tool)].plans[size_t(s.param)];
  s.buf.clear();
  s.stack.clear();
  switch (p.kind) {
    case Plan::Kind::Free:
      s.mode = Mode::Free;
      s.r = 0;
      s.prenl = false;
      s.lastnl = true;  // the newline the header ends with may also be the one before "</parameter>" (an empty value)
      break;
    case Plan::Kind::Enum: s.mode = Mode::Enum; break;
    case Plan::Kind::Json:
      s.mode = Mode::Json;
      s.j = J::Start;
      s.kinds = p.kinds;
      break;
  }
}

bool QwenToolGrammar::step_json(State& s, int c) const {
  using text::kArray;
  using text::kBoolean;
  using text::kInteger;
  using text::kNull;
  using text::kNumber;
  using text::kObject;
  const bool top = s.stack.empty();
  auto done = [&] { s.j = s.stack.empty() ? J::Done : J::After; };
  auto close = [&](char kind) {
    if (s.stack.empty() || s.stack.back() != kind) return false;
    s.stack.pop_back();
    done();
    return true;
  };
  auto literal = [&](const char* word) {
    s.buf = word;
    s.idx = 1;
    s.j = J::Lit;
    return true;
  };
  // A value starting with c, nested (anything JSON) or at the top (only the declared kinds, nothing before it).
  auto start = [&](bool nested) {
    const unsigned k = nested ? ~0u : s.kinds;
    if (c == '{') {
      if (!(k & kObject) || s.stack.size() >= kMaxDepth) return false;
      s.stack += 'o';
      s.j = J::KeyOrEnd;
      return true;
    }
    if (c == '[') {
      if (!(k & kArray) || s.stack.size() >= kMaxDepth) return false;
      s.stack += 'a';
      s.j = J::ValueOrEnd;
      return true;
    }
    if (c == '"') {
      if (!nested) return false;
      s.j = J::String;
      s.key_string = false;
      return true;
    }
    if (c == 't') return (k & kBoolean) && literal("true");
    if (c == 'f') return (k & kBoolean) && literal("false");
    if (c == 'n') return (k & kNull) && literal("null");
    if (c == '-') return (k & (kInteger | kNumber)) && (s.j = J::Neg, true);
    if (c == '0') return (k & (kInteger | kNumber)) && (s.j = J::Zero, true);
    if (c >= '1' && c <= '9') return (k & (kInteger | kNumber)) && (s.j = J::Int, true);
    return false;
  };
  // The top-level number is an integer unless the declared kinds include number; a nested one is any JSON number.
  const bool int_only = top && !(s.kinds & kNumber);
  auto number_end = [&] {
    done();
    return step_json(s, c);
  };
  switch (s.j) {
    case J::Start: return start(false);
    case J::Value: return ws(c) || start(true);
    case J::ValueOrEnd: return ws(c) || (c == ']' ? close('a') : start(true));
    case J::KeyOrEnd:
      if (ws(c)) return true;
      if (c == '}') return close('o');
      [[fallthrough]];
    case J::Key:
      if (ws(c)) return true;
      if (c != '"') return false;
      s.j = J::String;
      s.key_string = true;
      return true;
    case J::Colon:
      if (ws(c)) return true;
      if (c != ':') return false;
      s.j = J::Value;
      return true;
    case J::String:
      if (c == '"') {
        if (s.key_string) s.j = J::Colon;
        else done();
        return true;
      }
      if (c == '\\') return s.j = J::Escape, true;
      return c >= 0x20;
    case J::Escape:
      if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') return s.j = J::String, true;
      if (c == 'u') return s.j = J::Hex, s.idx = 0, true;
      return false;
    case J::Hex:
      if (!hex(c)) return false;
      if (++s.idx == 4) s.j = J::String;
      return true;
    case J::Neg:
      if (c == '0') return s.j = J::Zero, true;
      if (c >= '1' && c <= '9') return s.j = J::Int, true;
      return false;
    case J::Zero:
    case J::Int:
      if (s.j == J::Int && digit(c)) return true;
      if (c == '.' && !int_only) return s.j = J::Frac0, true;
      if ((c == 'e' || c == 'E') && !int_only) return s.j = J::Exp0, true;
      return number_end();
    case J::Frac0:
      if (!digit(c)) return false;
      return s.j = J::Frac, true;
    case J::Frac:
      if (digit(c)) return true;
      if (c == 'e' || c == 'E') return s.j = J::Exp0, true;
      return number_end();
    case J::Exp0:
      if (c == '+' || c == '-') return s.j = J::ExpSign, true;
      if (!digit(c)) return false;
      return s.j = J::Exp, true;
    case J::ExpSign:
      if (!digit(c)) return false;
      return s.j = J::Exp, true;
    case J::Exp:
      if (digit(c)) return true;
      return number_end();
    case J::Lit:
      if (c != s.buf[s.idx]) return false;
      if (++s.idx == s.buf.size()) done();
      return true;
    case J::After:
      if (ws(c)) return true;
      if (c == ',') return s.j = s.stack.back() == 'o' ? J::Key : J::Value, true;
      if (c == '}') return close('o');
      if (c == ']') return close('a');
      return false;
    case J::Done:
      // The value is complete: the next byte starts "\n</parameter>\n".
      if (c != '\n') return false;
      s.mode = Mode::Term;
      s.idx = 1;
      return true;
  }
  return false;
}

bool QwenToolGrammar::step(State& s, int c) const {
  switch (s.mode) {
    case Mode::Out: return true;
    case Mode::Prefix:
      if (c != kPrefix[s.idx]) return false;
      if (++s.idx == kPrefix.size()) s.mode = Mode::Name, s.buf.clear();
      return true;
    case Mode::Name: {
      if (c == '>') {
        for (size_t i = 0; i < tools_.size(); ++i)
          if (tools_[i].name == s.buf) return s.tool = int(i), s.mode = Mode::NameEnd, true;
        return false;
      }
      s.buf += char(c);
      for (const auto& t : tools_)
        if (t.name.compare(0, s.buf.size(), s.buf) == 0) return true;
      return false;
    }
    case Mode::NameEnd:
      if (c != '\n') return false;
      s.mode = Mode::Item;
      s.buf.clear();
      s.used.assign(tools_[size_t(s.tool)].headers.size(), '0');
      return true;
    case Mode::Item: {
      s.buf += char(c);
      const ToolPlan& t = tools_[size_t(s.tool)];
      bool ok = false;
      int whole = -2;  // the parameter whose header is complete, or -1 for </function>
      for (size_t i = 0; i < t.headers.size(); ++i) {
        if (s.used[i] == '1' || t.headers[i].compare(0, s.buf.size(), s.buf) != 0) continue;
        ok = true;
        if (t.headers[i].size() == s.buf.size()) whole = int(i);
      }
      if (all_required(s) && kFunctionEnd.compare(0, s.buf.size(), s.buf) == 0) {
        ok = true;
        if (kFunctionEnd.size() == s.buf.size()) whole = -1;
      }
      if (!ok) return false;
      if (whole == -1) {
        s.mode = Mode::Closed;
      } else if (whole >= 0) {
        s.used[size_t(whole)] = '1';
        s.param = whole;
        begin_value(s);
      }
      return true;
    }
    case Mode::Free:
      // The value is text, but "</parameter>" may appear in it only as its end, after a newline.
      if (s.r > 0 && c == kParamEnd[s.r]) {
        if (++s.r == kParamEnd.size()) {
          if (!s.prenl) return false;
          s.mode = Mode::Term;
          s.idx = uint8_t(kTerm.size() - 1);  // the newline after the tag is all that is left
          return true;
        }
      } else if (c == '<') {
        s.prenl = s.r == 0 && s.lastnl;
        s.r = 1;
      } else {
        s.r = 0;
      }
      s.lastnl = c == '\n';
      return true;
    case Mode::Enum: {
      const auto& texts = tools_[size_t(s.tool)].plans[size_t(s.param)].enums;
      if (c == '\n') {
        // The members hold no newline, so one after a whole member is the start of "\n</parameter>\n".
        if (std::find(texts.begin(), texts.end(), s.buf) == texts.end()) return false;
        s.mode = Mode::Term;
        s.idx = 1;
        return true;
      }
      s.buf += char(c);
      for (const auto& e : texts)
        if (e.compare(0, s.buf.size(), s.buf) == 0) return true;
      return false;
    }
    case Mode::Json: return step_json(s, c);
    case Mode::Term:
      if (c != kTerm[s.idx]) return false;
      if (++s.idx == kTerm.size()) s.mode = Mode::Item, s.buf.clear();
      return true;
    case Mode::Closed: return false;
  }
  return false;
}

bool QwenToolGrammar::feed(State& s, int32_t id) const {
  if (s.mode == Mode::Out) {
    if (id == v_->call_open) {
      s = State{};
      s.mode = Mode::Prefix;
    }
    return true;
  }
  if (id == v_->call_close) {
    if (s.mode != Mode::Closed) return false;
    s = State{};
    return true;
  }
  const std::string& b = v_->bytes[size_t(id)];
  if (b.empty()) return false;  // another special token
  for (unsigned char c : b)
    if (!step(s, c)) return false;
  return true;
}

std::string QwenToolGrammar::state_key(const State& s) const {
  // Every key carries the tools' fingerprint: a token can run from almost any state (the end of a value, the literal
  // before a name) on into the parts that depend on the tools.
  std::string k(1, char('A' + int(s.mode)));
  k += fp_;
  k += '|';
  switch (s.mode) {
    case Mode::Prefix: k += char('0' + s.idx); break;
    case Mode::Name: k += s.buf; break;
    case Mode::Item: k += std::to_string(s.tool) + '|' + s.used + '|' + s.buf; break;
    case Mode::Free:
      k += char('0' + s.r);
      k += s.prenl ? '1' : '0';
      k += s.lastnl ? '1' : '0';
      k += std::to_string(s.tool) + '|' + s.used;
      break;
    case Mode::Enum: k += std::to_string(s.tool) + '|' + std::to_string(s.param) + '|' + s.buf + '|' + s.used; break;
    case Mode::Json:
      k += char('a' + int(s.j));
      k += char('0' + s.idx);
      k += s.key_string ? '1' : '0';
      k += s.stack + '|' + s.buf + '|' + std::to_string(s.tool) + '|' + s.used;
      if (s.stack.empty()) k += '|' + std::to_string(s.kinds);
      break;
    case Mode::Term: k += char('0' + s.idx); k += std::to_string(s.tool) + '|' + s.used; break;
    case Mode::NameEnd: break;
    default: break;
  }
  return k;
}

const std::vector<int32_t>& QwenToolGrammar::allowed() {
  if (allowed_valid_) return allowed_;
  allowed_.clear();
  allowed_valid_ = true;
  if (st_.mode == Mode::Out) return allowed_;
  if (st_.mode == Mode::Closed) return allowed_ = {v_->call_close};
  const std::string key = state_key(st_);
  {
    std::lock_guard lk(v_->mu);
    if (auto it = v_->cache.find(key); it != v_->cache.end()) return allowed_ = it->second;
  }
  for (int b = 0; b < 256; ++b) {
    State probe = st_;
    if (!step(probe, b)) continue;
    for (int32_t id : v_->by_first[size_t(b)]) {
      State s = st_;
      if (feed(s, id)) allowed_.push_back(id);
    }
  }
  std::lock_guard lk(v_->mu);
  // The sets of the states inside a value hold most of the vocabulary: bound the cache by what it holds, not by how many.
  if (v_->cached_ints + allowed_.size() > kCacheBudget) {
    v_->cache.clear();
    v_->cached_ints = 0;
  }
  if (v_->cache.emplace(key, allowed_).second) v_->cached_ints += allowed_.size();
  return allowed_;
}

bool QwenToolGrammar::accept(int32_t id) {
  State s = st_;
  if (!feed(s, id)) return false;
  st_ = std::move(s);
  allowed_valid_ = false;
  return true;
}

std::string QwenToolGrammar::completion() const {
  using text::kArray;
  using text::kBoolean;
  using text::kInteger;
  using text::kNull;
  using text::kNumber;
  using text::kObject;
  State s = st_;
  std::string out;
  auto emit = [&](std::string_view t) {
    for (unsigned char c : t)
      if (!step(s, c)) throw std::logic_error("tool grammar: completion '" + std::string(t) + "' is not accepted");
    out += t;
  };
  for (int guard = 0; s.mode != Mode::Closed && s.mode != Mode::Out && guard < 100000; ++guard) {
    switch (s.mode) {
      case Mode::Prefix: emit(kPrefix.substr(s.idx)); break;
      case Mode::Name: {
        for (const auto& t : tools_)
          if (t.name.compare(0, s.buf.size(), s.buf) == 0) {
            emit(t.name.substr(s.buf.size()) + ">");
            break;
          }
        break;
      }
      case Mode::NameEnd: emit("\n"); break;
      case Mode::Item: {
        const ToolPlan& t = tools_[size_t(s.tool)];
        if (!s.buf.empty()) {
          // Partway through the next item: finish it.
          if (all_required(s) && kFunctionEnd.compare(0, s.buf.size(), s.buf) == 0) {
            emit(kFunctionEnd.substr(s.buf.size()));
          } else {
            for (size_t i = 0; i < t.headers.size(); ++i)
              if (s.used[i] != '1' && t.headers[i].compare(0, s.buf.size(), s.buf) == 0) {
                emit(t.headers[i].substr(s.buf.size()));
                break;
              }
          }
        } else if (all_required(s)) {
          emit(kFunctionEnd);
        } else {
          for (size_t i = 0; i < t.headers.size(); ++i)
            if (t.required[i] && s.used[i] != '1') {
              emit(t.headers[i]);
              break;
            }
        }
        break;
      }
      case Mode::Free: emit(kTerm.substr(0, kTerm.size() - 1)); break;  // "\n</parameter>", then Term's newline
      case Mode::Enum: {
        const auto& texts = tools_[size_t(s.tool)].plans[size_t(s.param)].enums;
        const std::string* best = nullptr;
        for (const auto& e : texts)
          if (e.compare(0, s.buf.size(), s.buf) == 0 && (!best || e.size() < best->size())) best = &e;
        emit(best->substr(s.buf.size()) + "\n");
        break;
      }
      case Mode::Term: emit(kTerm.substr(s.idx)); break;
      case Mode::Json: {
        const char close = s.stack.empty() ? '\n' : s.stack.back() == 'o' ? '}' : ']';
        switch (s.j) {
          case J::Start:
            emit(s.kinds & kNull ? "null" : s.kinds & kBoolean ? "true" : s.kinds & (kInteger | kNumber) ? "0" : s.kinds & kObject ? "{}" : "[]");
            break;
          case J::Value: emit("0"); break;
          case J::ValueOrEnd: emit("]"); break;
          case J::KeyOrEnd: emit("}"); break;
          case J::Key: emit("\"a\":0"); break;
          case J::Colon: emit(":0"); break;
          case J::String: emit(s.key_string ? "\":0" : "\""); break;
          case J::Escape: emit("n"); break;
          case J::Hex: emit(std::string(size_t(4 - s.idx), '0')); break;
          case J::Neg:
          case J::Frac0:
          case J::Exp0:
          case J::ExpSign: emit("0"); break;
          case J::Zero:
          case J::Int:
          case J::Frac:
          case J::Exp:
          case J::Done: emit(std::string(1, close)); break;
          case J::Lit: emit(s.buf.substr(s.idx)); break;
          case J::After: emit(std::string(1, close)); break;
        }
        break;
      }
      case Mode::Out:
      case Mode::Closed: break;
    }
  }
  return out;
}

}  // namespace eklow::serve
