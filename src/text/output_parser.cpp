#include "text/output_parser.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace eklow::text {

namespace {

// Length of the longest prefix of s that ends on a whole UTF-8 character.
size_t complete_prefix(const std::string& s) {
  size_t n = s.size();
  for (size_t back = 1; back <= 3 && back <= n; ++back) {
    unsigned char c = s[n - back];
    if ((c & 0xc0) == 0x80) continue;  // continuation byte: keep looking for the lead
    size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    return need > back ? n - back : n;
  }
  return n;
}

int32_t need_id(const Tokenizer& t, const char* piece) {
  int32_t id = t.id(piece);
  if (id < 0) throw std::runtime_error(std::string("tokenizer has no ") + piece);
  return id;
}

// The tool-call body as a symbol stream: bytes 0..255, and kQuote for each <|"|>.
constexpr int kQuote = -1;

struct BodyParser {
  std::vector<int> s;
  size_t i = 0;

  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error("tool call: " + what + " at symbol " + std::to_string(i));
  }
  bool at_end() const { return i >= s.size(); }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
  }
  void expect(char c) {
    ws();
    if (at_end() || s[i] != c) fail(std::string("expected '") + c + "'");
    ++i;
  }
  std::string quoted() {
    ++i;  // the opening quote
    std::string out;
    while (!at_end() && s[i] != kQuote) out += char(s[i++]);
    if (at_end()) fail("unterminated string");
    ++i;
    return out;
  }
  std::string key() {
    ws();
    if (!at_end() && s[i] == kQuote) return quoted();
    std::string k;
    while (!at_end() && s[i] != ':' && s[i] != kQuote && s[i] != '}' && s[i] != ',') k += char(s[i++]);
    while (!k.empty() && (k.back() == ' ' || k.back() == '\n')) k.pop_back();
    if (k.empty()) fail("empty key");
    return k;
  }
  json::Value object() {
    // After '{'.
    json::Value o = json::Value::object();
    ws();
    if (!at_end() && s[i] == '}') {
      ++i;
      return o;
    }
    for (;;) {
      std::string k = key();
      expect(':');
      o.set(std::move(k), value());
      ws();
      if (at_end()) fail("unterminated object");
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == '}') {
        ++i;
        return o;
      }
      fail("expected ',' or '}'");
    }
  }
  json::Value array() {
    json::Value a = json::Value::array();
    ws();
    if (!at_end() && s[i] == ']') {
      ++i;
      return a;
    }
    for (;;) {
      a.items().push_back(value());
      ws();
      if (at_end()) fail("unterminated list");
      if (s[i] == ',') {
        ++i;
        continue;
      }
      if (s[i] == ']') {
        ++i;
        return a;
      }
      fail("expected ',' or ']'");
    }
  }
  json::Value value() {
    ws();
    if (at_end()) fail("missing value");
    if (s[i] == kQuote) return json::Value::string(quoted());
    if (s[i] == '{') {
      ++i;
      return object();
    }
    if (s[i] == '[') {
      ++i;
      return array();
    }
    std::string w;
    while (!at_end() && s[i] != ',' && s[i] != '}' && s[i] != ']' && s[i] != kQuote) w += char(s[i++]);
    while (!w.empty() && (w.back() == ' ' || w.back() == '\n')) w.pop_back();
    if (w == "true") return json::Value::boolean(true);
    if (w == "false") return json::Value::boolean(false);
    if (w == "null") return json::Value::null();
    return number(w);
  }
  // Python's int and float notation (what {{ x }} wrote): -12, 0.5, 1e-05, 1e+16.
  json::Value number(const std::string& w) {
    size_t k = 0;
    if (k < w.size() && w[k] == '-') ++k;
    size_t digits = k;
    while (k < w.size() && w[k] >= '0' && w[k] <= '9') ++k;
    if (k == digits) fail("unquoted value '" + w + "'");
    bool is_float = false;
    if (k < w.size() && w[k] == '.') {
      is_float = true;
      size_t f = ++k;
      while (k < w.size() && w[k] >= '0' && w[k] <= '9') ++k;
      if (k == f) fail("bad number '" + w + "'");
    }
    if (k < w.size() && (w[k] == 'e' || w[k] == 'E')) {
      is_float = true;
      ++k;
      if (k < w.size() && (w[k] == '+' || w[k] == '-')) ++k;
      size_t e = k;
      while (k < w.size() && w[k] >= '0' && w[k] <= '9') ++k;
      if (k == e) fail("bad number '" + w + "'");
    }
    if (k != w.size()) fail("unquoted value '" + w + "'");
    if (!is_float) {
      // Python writes ints without leading zeros; normalize anything else the model writes.
      std::string d = w.substr(digits);
      size_t nz = d.find_first_not_of('0');
      d = nz == std::string::npos ? "0" : d.substr(nz);
      return json::Value::integer((digits && d != "0" ? "-" : "") + d);
    }
    return json::Value::number(std::strtod(w.c_str(), nullptr));
  }
};

}  // namespace

ToolCall parse_tool_call(const std::vector<std::pair<std::string, bool>>& pieces) {
  BodyParser p;
  for (const auto& [text, quote] : pieces) {
    if (quote) p.s.push_back(kQuote);
    else
      for (unsigned char c : text) p.s.push_back(c);
  }
  p.ws();
  for (char c : std::string("call:")) {
    if (p.at_end() || p.s[p.i] != c) p.fail("expected 'call:'");
    ++p.i;
  }
  ToolCall call;
  while (!p.at_end() && p.s[p.i] != '{' && p.s[p.i] != kQuote) call.name += char(p.s[p.i++]);
  if (call.name.empty()) p.fail("no function name");
  p.expect('{');
  call.arguments = p.object();
  p.ws();
  if (!p.at_end()) p.fail("text after the arguments");
  return call;
}

OutputParser::OutputParser(const Tokenizer& tok, bool in_thinking) : tok_(tok) {
  channel_open_ = need_id(tok, "<|channel>");
  channel_close_ = need_id(tok, "<channel|>");
  call_open_ = need_id(tok, "<|tool_call>");
  call_close_ = need_id(tok, "<tool_call|>");
  quote_ = need_id(tok, "<|\"|>");
  eos_ = need_id(tok, "<eos>");
  turn_end_ = need_id(tok, "<turn|>");
  tool_response_ = need_id(tok, "<|tool_response>");
  turn_open_ = need_id(tok, "<|turn>");
  if (in_thinking) state_ = State::Thinking;
  // What push() takes in each state besides text: the state's own markers, and the turn's ends except inside a
  // tool call (one not closed is an error).
  auto ban_all_but = [&](State s, std::vector<int32_t> ok) {
    for (int32_t id : tok.special_ids())
      if (std::find(ok.begin(), ok.end(), id) == ok.end()) banned_[size_t(s)].push_back(id);
  };
  ban_all_but(State::Content, {channel_open_, call_open_, eos_, turn_end_, tool_response_, turn_open_});
  ban_all_but(State::ChannelName, {eos_, turn_end_, tool_response_, turn_open_});
  ban_all_but(State::Thinking, {channel_close_, eos_, turn_end_, tool_response_, turn_open_});
  ban_all_but(State::ToolCall, {call_close_, quote_});
}

void OutputParser::emit_text(std::string& pending, std::string& out, bool flush) {
  size_t n = flush ? pending.size() : complete_prefix(pending);
  // Reasoning ends with the template's "\n" before <channel|>: hold a trailing newline back until
  // the next token shows whether the channel closes.
  if (state_ == State::Thinking && !flush && n > 0 && n == pending.size() && pending[n - 1] == '\n') --n;
  out.append(pending, 0, n);
  pending.erase(0, n);
}

void OutputParser::finish_call(OutputDelta& d) {
  std::vector<std::pair<std::string, bool>> pieces;
  for (auto& p : call_) pieces.emplace_back(p.text, p.quote);
  call_.clear();
  try {
    d.tool_calls.push_back(parse_tool_call(pieces));
  } catch (const std::exception& e) {
    d.errors.push_back(e.what());
  }
}

OutputDelta OutputParser::push(int32_t id) {
  OutputDelta d;
  if (state_ == State::Done) return d;
  const bool special = tok_.is_special(id);
  if (is_end(id) || id == turn_open_) {
    if (state_ == State::ToolCall) {
      d.errors.push_back("tool call: not closed before the end of the turn");
      call_.clear();
    } else if (state_ == State::Thinking) {
      emit_text(pending_, d.reasoning, true);
    } else {
      emit_text(pending_, d.content, true);
    }
    state_ = State::Done;
    d.done = true;
    d.end_token = id;
    return d;
  }
  switch (state_) {
    case State::Content:
      if (id == channel_open_) {
        emit_text(pending_, d.content, true);
        channel_name_.clear();
        state_ = State::ChannelName;
      } else if (id == call_open_) {
        emit_text(pending_, d.content, true);
        call_.clear();
        state_ = State::ToolCall;
      } else if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in content");
      } else {
        pending_ += tok_.bytes(id);
        emit_text(pending_, d.content, false);
      }
      break;
    case State::ChannelName: {
      if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in a channel name");
        break;
      }
      channel_name_ += tok_.bytes(id);
      size_t nl = channel_name_.find('\n');
      if (nl != std::string::npos) {
        pending_ = channel_name_.substr(nl + 1);
        channel_name_.resize(nl);
        state_ = State::Thinking;
        emit_text(pending_, d.reasoning, false);
      }
      break;
    }
    case State::Thinking:
      if (id == channel_close_) {
        if (!pending_.empty() && pending_.back() == '\n') pending_.pop_back();
        emit_text(pending_, d.reasoning, true);
        state_ = State::Content;
      } else if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in reasoning");
      } else {
        pending_ += tok_.bytes(id);
        emit_text(pending_, d.reasoning, false);
      }
      break;
    case State::ToolCall:
      if (id == call_close_) {
        finish_call(d);
        state_ = State::Content;
      } else if (id == quote_) {
        call_.push_back({"", true});
      } else if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in a tool call");
      } else {
        if (call_.empty() || call_.back().quote) call_.push_back({"", false});
        call_.back().text += tok_.bytes(id);
      }
      break;
    case State::Done: break;
  }
  return d;
}

OutputDelta OutputParser::finish() {
  OutputDelta d;
  if (state_ == State::ToolCall) d.errors.push_back("tool call: not closed");
  else if (state_ == State::Thinking) emit_text(pending_, d.reasoning, true);
  else if (state_ != State::Done) emit_text(pending_, d.content, true);
  state_ = State::Done;
  return d;
}

}  // namespace eklow::text
