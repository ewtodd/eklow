#include "text/output_parser_inkling.hpp"

#include <cctype>
#include <stdexcept>

#include "text/chat_inkling.hpp"
#include "text/unicode.hpp"

namespace eklow::text {

namespace {

constexpr int kMaxNesting = 64;  // llama.cpp's RULE_RECURSIVE_LIMIT: a value nested deeper stops the parser

// ---- what llama.cpp's PEG does with bytes ----------------------------------------------------------------------------------

enum class St { Fail, Ok, More };
struct R {
  St st;
  size_t end;
};

struct NestingError : std::runtime_error {
  NestingError() : std::runtime_error("input is nested more than 64 levels deep") {}
};

// common_parse_utf8_codepoint: no check for overlong forms, surrogates or the range; an invalid sequence consumes the bytes
// before the bad one (at least one).
struct Step {
  enum { Ok, Invalid, Incomplete } status;
  size_t n;
};
Step utf8_step(std::string_view s, size_t off) {
  if (off >= s.size()) return {Step::Incomplete, 0};
  const unsigned char c = s[off];
  if (!(c & 0x80)) return {Step::Ok, 1};
  if (!(c & 0x40)) return {Step::Invalid, 1};
  size_t len;
  if (!(c & 0x20)) len = 2;
  else if (!(c & 0x10)) len = 3;
  else if (!(c & 0x08)) len = 4;
  else return {Step::Invalid, 1};
  for (size_t i = 1; i < len; ++i) {
    if (off + i >= s.size()) return {Step::Incomplete, i};
    if ((static_cast<unsigned char>(s[off + i]) & 0xc0) != 0x80) return {Step::Invalid, i};
  }
  return {Step::Ok, len};
}

struct In {
  const std::string& s;
  const std::vector<int32_t>& tk;  // the token id at a token's first byte, -1 elsewhere
  size_t n() const { return s.size(); }
};

R fail(size_t at) { return {St::Fail, at}; }

// literal: a prefix of it at the end of the input is "more", a different byte is a failure
R lit(const In& in, size_t pos, std::string_view L) {
  size_t p = pos;
  for (char c : L) {
    if (p >= in.n()) return {St::More, p};
    if (in.s[p] != c) return fail(pos);
    ++p;
  }
  return {St::Ok, p};
}

size_t ws(const In& in, size_t pos) {
  while (pos < in.n() && std::isspace(static_cast<unsigned char>(in.s[pos]))) ++pos;
  return pos;
}

#define SEQ(r)                         \
  if ((r).st == St::Fail) return (r); \
  if ((r).st == St::More) return (r);

// string_content(delim): up to the closing delimiter (not consumed); a valid JSON escape, any other character as UTF-8
R string_content(const In& in, size_t pos, char delim) {
  size_t p = pos;
  while (p < in.n()) {
    const char c = in.s[p];
    if (c == delim) return {St::Ok, p};
    if (c == '\\') {
      const size_t save = p;
      ++p;
      if (p >= in.n()) return {St::More, save};
      const char e = in.s[p];
      if (e == delim || e == '\\' || e == '/' || e == 'b' || e == 'f' || e == 'n' || e == 'r' || e == 't') {
        ++p;
      } else if (e == 'u') {
        ++p;
        for (int i = 0; i < 4; ++i) {
          if (p >= in.n()) return {St::More, save};
          if (!std::isxdigit(static_cast<unsigned char>(in.s[p]))) return fail(pos);
          ++p;
        }
      } else {
        return fail(pos);
      }
    } else {
      const Step st = utf8_step(in.s, p);
      if (st.status == Step::Incomplete) return {St::More, p};
      if (st.status == Step::Invalid) return fail(pos);
      p += st.n;
    }
  }
  return {St::More, p};
}

R jvalue(const In& in, size_t pos, int depth);

R jstring(const In& in, size_t pos) {
  R r = lit(in, pos, "\"");
  SEQ(r)
  r = string_content(in, r.end, '"');
  SEQ(r)
  return lit(in, r.end, "\"");
}

// chars(class, 1, 1): one character of the class
R one_of(const In& in, size_t pos, std::string_view cls) {
  if (pos >= in.n()) return {St::More, pos};
  const Step st = utf8_step(in.s, pos);
  if (st.status == Step::Incomplete) return {St::More, pos};
  if (st.status == Step::Invalid) return fail(pos);
  if (st.n == 1 && cls.find(in.s[pos]) != std::string_view::npos) return {St::Ok, pos + 1};
  return fail(pos);
}

// chars("[0-9]") with `min` digits or more
R digits(const In& in, size_t pos, int min) {
  size_t p = pos;
  int count = 0;
  while (p < in.n() && in.s[p] >= '0' && in.s[p] <= '9') ++p, ++count;
  if (count >= min) return {St::Ok, p};
  if (p >= in.n()) return {St::More, p};
  const Step st = utf8_step(in.s, p);
  if (st.status == Step::Incomplete) return {St::More, p};
  return fail(pos);
}

R jnumber(const In& in, size_t pos) {
  size_t p = pos;
  // optional("-"): at the end of the input nothing is tried
  if (p < in.n()) {
    R r = lit(in, p, "-");
    if (r.st == St::More) return r;
    if (r.st == St::Ok) p = r.end;
  }
  // int_part: "0" | [1-9][0-9]*
  R r = lit(in, p, "0");
  if (r.st == St::More) return r;
  if (r.st == St::Ok) {
    p = r.end;
  } else {
    r = one_of(in, p, "123456789");
    SEQ(r)
    p = r.end;
    while (p < in.n() && in.s[p] >= '0' && in.s[p] <= '9') ++p;
  }
  // optional(frac): "." digits
  if (p < in.n()) {
    R f = lit(in, p, ".");
    if (f.st == St::Ok) {
      R d = digits(in, f.end, 1);
      if (d.st == St::More) return d;
      if (d.st == St::Ok) p = d.end;
    }
  }
  // optional(exp): [eE] [+-]? digits
  if (p < in.n()) {
    R e = one_of(in, p, "eE");
    if (e.st == St::More) return e;
    if (e.st == St::Ok) {
      size_t q = e.end;
      if (q < in.n()) {
        R sg = one_of(in, q, "+-");
        if (sg.st == St::Ok) q = sg.end;
      }
      R d = digits(in, q, 1);
      if (d.st == St::More) return d;
      if (d.st == St::Ok) p = d.end;
    }
  }
  // not [0-9.eE+-]: at the end of the input the number is not settled
  R nx = one_of(in, p, "0123456789.eE+-");
  if (nx.st == St::Ok) return fail(pos);
  if (nx.st == St::More) return {St::More, p};
  return {St::Ok, p};
}

R jobject(const In& in, size_t pos, int depth);
R jarray(const In& in, size_t pos, int depth);

R member(const In& in, size_t pos, int depth) {
  R r = jstring(in, pos);
  SEQ(r)
  size_t p = ws(in, r.end);
  r = lit(in, p, ":");
  SEQ(r)
  p = ws(in, r.end);
  return jvalue(in, p, depth + 1);
}

R jobject(const In& in, size_t pos, int depth) {
  R r = lit(in, pos, "{");
  SEQ(r)
  size_t p = ws(in, r.end);
  R close = lit(in, p, "}");
  if (close.st != St::Fail) return close;
  r = member(in, p, depth);
  SEQ(r)
  size_t q = r.end;
  for (;;) {
    if (q >= in.n()) break;
    size_t a = ws(in, q);
    R c = lit(in, a, ",");
    if (c.st == St::Fail) break;
    if (c.st == St::More) return c;
    a = ws(in, c.end);
    R m = member(in, a, depth);
    if (m.st == St::Fail) break;
    if (m.st == St::More) return m;
    q = m.end;
  }
  p = ws(in, q);
  return lit(in, p, "}");
}

R jarray(const In& in, size_t pos, int depth) {
  R r = lit(in, pos, "[");
  SEQ(r)
  size_t p = ws(in, r.end);
  R close = lit(in, p, "]");
  if (close.st != St::Fail) return close;
  r = jvalue(in, p, depth + 1);
  SEQ(r)
  size_t q = r.end;
  for (;;) {
    if (q >= in.n()) break;
    size_t a = ws(in, q);
    R c = lit(in, a, ",");
    if (c.st == St::Fail) break;
    if (c.st == St::More) return c;
    a = ws(in, c.end);
    R m = jvalue(in, a, depth + 1);
    if (m.st == St::Fail) break;
    if (m.st == St::More) return m;
    q = m.end;
  }
  p = ws(in, q);
  return lit(in, p, "]");
}

// json(): the first alternative that does not fail
R jvalue(const In& in, size_t pos, int depth) {
  if (depth >= kMaxNesting) throw NestingError();
  for (auto alt : {jobject, jarray}) {
    R r = alt(in, pos, depth);
    if (r.st != St::Fail) return r;
  }
  R r = jstring(in, pos);
  if (r.st != St::Fail) return r;
  r = jnumber(in, pos);
  if (r.st != St::Fail) return r;
  for (const char* w : {"true", "false", "null"}) {
    r = lit(in, pos, w);
    if (r.st != St::Fail) return r;
  }
  return fail(pos);
}

std::string_view trim_trailing(std::string_view sv) {
  while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) sv.remove_suffix(1);
  return sv;
}

int brace_depth(const std::string& s) {
  int depth = 0;
  bool in_string = false, escaped = false;
  for (char c : s) {
    if (escaped) {
      escaped = false;
      continue;
    }
    if (c == '\\' && in_string) {
      escaped = true;
      continue;
    }
    if (c == '"') {
      in_string = !in_string;
      continue;
    }
    if (!in_string) {
      if (c == '{') ++depth;
      else if (c == '}') --depth;
    }
  }
  return depth;
}

// A token starting at pos: its end (the next token's start, or the end of the input).
size_t token_end(const In& in, size_t pos) {
  size_t p = pos + 1;
  while (p < in.n() && in.tk[p] < 0) ++p;
  return p;
}

// token(id) at pos
R tok(const In& in, size_t pos, int32_t id) {
  if (pos >= in.n()) return {St::More, pos};
  if (in.tk[pos] != id) return fail(pos);
  return {St::Ok, token_end(in, pos)};
}

// The JSON text of a call after its name: the member "args" and the end, building the call's fields as the mapper does.
struct CallFields {
  bool named = false, closed = false;
  std::string name, args, buffer;
};

R call_body(const In& in, size_t pos, bool required, std::string_view esc_name, CallFields& f) {
  // tool_open("{") << "name" : '"' tool_name '"' , "args" : tool_args(json) << tool_close("}") + end
  R r = lit(in, pos, "{");
  SEQ(r)
  size_t p = ws(in, r.end);
  r = lit(in, p, "\"name\"");
  SEQ(r)
  p = ws(in, r.end);
  r = lit(in, p, ":");
  SEQ(r)
  p = ws(in, r.end);
  r = lit(in, p, "\"");
  SEQ(r)
  const size_t name_at = r.end;
  r = required ? lit(in, name_at, esc_name) : string_content(in, name_at, '"');
  SEQ(r)  // (the name is an atomic node: while it is unfinished there is none)
  f.named = true;
  f.name = std::string(trim_trailing(std::string_view(in.s).substr(name_at, r.end - name_at)));
  f.args = "{";
  p = r.end;
  r = lit(in, p, "\"");
  SEQ(r)
  p = ws(in, r.end);
  r = lit(in, p, ",");
  SEQ(r)
  p = ws(in, r.end);
  r = lit(in, p, "\"args\"");
  SEQ(r)
  p = ws(in, r.end);
  r = lit(in, p, ":");
  SEQ(r)
  p = ws(in, r.end);
  const R j = jvalue(in, p, 0);
  if (j.st == St::Fail) return j;
  {
    const std::string_view text = trim_trailing(std::string_view(in.s).substr(p, j.end - p));
    if (!text.empty() && text.front() == '{') (f.name.empty() ? f.buffer : f.args) = std::string(text);
  }
  if (j.st == St::More) return j;
  p = ws(in, j.end);
  r = lit(in, p, "}");
  SEQ(r)
  f.closed = true;
  if (f.name.empty() && !f.buffer.empty()) f.args = f.buffer;
  for (int d = brace_depth(f.args); d > 0; --d) f.args += '}';
  return tok(in, r.end, inkling::kEndMessage);
}

}  // namespace

// ---- the parser ---------------------------------------------------------------------------------------------------------

InklingOutputParser::InklingOutputParser(const InklingTokenizer& tok, const InklingMode& mode, std::span<const int32_t> prefill)
    : tok_(tok), mode_(mode) {
  rf_ = mode.response_format;
  tools_ = !rf_ && mode.use_tools();
  required_ = tools_ && mode.tool_choice == InklingToolChoice::Required;
  for (int32_t id : prefill) {
    OutputDelta sink;
    if (!done_) feed(id, sink);
  }
}

bool InklingOutputParser::committed() const {
  return !msg_.reasoning.empty() || !msg_.content.empty() || !msg_.tool_calls.empty();
}

void InklingOutputParser::emit_text(const std::string& text, OutputDelta& d) {
  if (text.empty()) return;
  if (phase_ == Phase::Think && mode_.extract_reasoning) {
    // reasoning of nothing but blanks is no reasoning (llama.cpp's mapper drops it)
    std::string t = text;
    if (msg_.reasoning.empty()) {
      held_ += text;
      if (held_.find_first_not_of(" \n\r\t") == std::string::npos) return;
      t = std::move(held_);
      held_.clear();
    }
    msg_.reasoning += t;
    d.reasoning += t;
  } else {
    msg_.content += text;
    d.content += text;
  }
}

// The block's bytes so far, as text: whole characters as they are, an invalid sequence as U+FFFD, an incomplete one
// waits (at the end of the block the end token's first byte, '<', shows it is invalid).
void InklingOutputParser::release_text(const std::string& bytes, bool at_end, OutputDelta& d) {
  tail_ += bytes;
  std::string probe = tail_;
  if (at_end) probe += '<';
  std::string out;
  size_t i = 0;
  while (i < tail_.size()) {
    const Step st = utf8_step(probe, i);
    if (st.status == Step::Incomplete) break;
    // (llama.cpp lets an overlong form, a surrogate or a code point past U+10FFFF through as it is; only byte tokens can make
    // one, and the message must be text: it is one U+FFFD here)
    if (st.status == Step::Ok && (st.n == 1 || unicode::valid_utf8(std::string_view(probe).substr(i, st.n)))) out.append(probe, i, st.n);
    else out += "\xEF\xBF\xBD";
    i += st.n;
  }
  tail_.erase(0, i);
  emit_text(out, d);
}

void InklingOutputParser::start_block(Phase kind, OutputDelta& d) {
  phase_ = kind;
  tail_.clear();
  if (kind == Phase::Think && !mode_.extract_reasoning) emit_text("<|content_thinking|>", d);
}

void InklingOutputParser::end_block(OutputDelta& d) {
  release_text("", true, d);
  tail_.clear();
  if (phase_ == Phase::Think && !mode_.extract_reasoning) emit_text("<|end_message|>", d);
  phase_ = Phase::Idle;
}

void InklingOutputParser::freeze(OutputDelta&) {
  phase_ = Phase::Frozen;
  failed_ = true;
}

void InklingOutputParser::add_call(const Call& c, bool truncated, OutputDelta& d) {
  ToolCall t;
  t.name = c.name;
  t.arguments_text = c.args;
  t.truncated = truncated;
  try {
    t.arguments = parse_llama_json(c.args);
    if (!t.arguments.is_object()) t.arguments = json::Value::object();
  } catch (const std::exception&) {
    t.arguments = json::Value::object();
  }
  msg_.tool_calls.push_back(t);
  d.tool_calls.push_back(std::move(t));
}

// One call from `pos` (its header): the header, free text up to <|content_invoke_tool_json|> (with tool_choice "required":
// nothing, or one tool's name), the token, the JSON.
InklingOutputParser::Call InklingOutputParser::parse_call(size_t pos) const {
  const In in{cb_, ct_};
  Call out;
  CallFields f;
  auto done = [&](R r) {
    out.status = r.st == St::Ok ? Parse::Ok : r.st == St::More ? Parse::More : Parse::Fail;
    out.end = r.end;
    if (r.st != St::Fail) {
      out.named = f.named;
      out.name = f.name;
      out.args = f.args;
    }
    return out;
  };
  R r = tok(in, pos, inkling::kMessageModel);
  if (r.st != St::Ok) return done(r);
  if (!required_) {
    // name = until(<|content_invoke_tool_json|>)
    size_t p = r.end;
    while (p < in.n() && in.tk[p] != inkling::kContentInvokeToolJson) p = token_end(in, p);
    if (p >= in.n()) return done({St::More, in.n()});
    r = tok(in, p, inkling::kContentInvokeToolJson);
    return done(call_body(in, r.end, false, "", f));
  }
  // choice over the tools, each: optional(literal(name)) token(<|content_invoke_tool_json|>) body(literal(name))
  R last = fail(pos);
  for (const std::string& name : mode_.tool_names) {
    f = CallFields{};
    size_t p = r.end;
    if (p < in.n()) {
      R l = lit(in, p, name);
      if (l.st == St::More) {
        last = l;
        break;
      }
      if (l.st == St::Ok) p = l.end;
    }
    R t = tok(in, p, inkling::kContentInvokeToolJson);
    if (t.st == St::Fail) continue;
    if (t.st == St::More) {
      last = t;
      break;
    }
    std::string esc;
    {
      const std::string q = jinja_tojson(json::Value::string(name), false);
      esc = q.substr(1, q.size() - 2);
    }
    const R b = call_body(in, t.end, true, esc, f);
    if (b.st == St::Fail) continue;
    last = b;
    break;
  }
  return done(last);
}

void InklingOutputParser::run_calls(OutputDelta& d) {
  Call c;
  try {
    c = parse_call(call_from_);
  } catch (const NestingError& e) {
    // llama.cpp's parser throws: nothing more is read, and the message stays the one of the token before (the call it showed)
    if (open_.named) add_call(open_, true, d);
    msg_.errors.push_back(e.what());
    d.errors.push_back(e.what());
    phase_ = Phase::Frozen;
    return;
  }
  switch (c.status) {
    case Parse::More: open_ = c; return;
    case Parse::Ok:
      add_call(c, false, d);
      ++n_calls_;
      open_ = Call{};
      call_from_ = c.end;
      if (!mode_.parallel_tool_calls) phase_ = Phase::Ignore;
      return;
    case Parse::Fail:
      // The call is not one. When it was all there was, llama.cpp's message is still the one of the token before, which had it.
      if (n_calls_ == 0 && open_.named && (required_ || !committed())) add_call(open_, true, d);
      open_ = Call{};
      if (required_ && n_calls_ == 0) failed_ = true;
      phase_ = required_ && n_calls_ == 0 ? Phase::Frozen : Phase::Ignore;
      return;
  }
}

void InklingOutputParser::feed(int32_t id, OutputDelta& d) {
  const std::string& piece = tok_.bytes(id);
  switch (phase_) {
    case Phase::Ignore:
    case Phase::Frozen: return;
    case Phase::Idle:
      if (id == inkling::kMessageModel) {
        phase_ = Phase::AfterHeader;
      } else if (required_ || rf_) {
        freeze(d);
      } else {
        phase_ = Phase::Ignore;
      }
      return;
    case Phase::AfterHeader:
      if (id == inkling::kContentThinking) return start_block(Phase::Think, d);
      if (id == inkling::kContentText && !rf_) return start_block(Phase::Text, d);
      if (id == inkling::kContentText) {
        phase_ = Phase::Json;
        cb_.clear();
        ct_.clear();
        json_sent_ = 0;
        return;
      }
      if (rf_) return freeze(d);
      if (!tools_) {
        phase_ = Phase::Ignore;
        return;
      }
      // the blocks end before this header: a call
      phase_ = Phase::Calls;
      {
        const std::string& h = tok_.bytes(inkling::kMessageModel);
        cb_ = h;
        ct_.assign(h.size(), -1);
        ct_[0] = inkling::kMessageModel;
        call_from_ = 0;
        n_calls_ = 0;
        open_ = Call{};
      }
      [[fallthrough]];
    case Phase::Calls: {
      const size_t at = cb_.size();
      cb_ += piece;
      ct_.resize(cb_.size(), -1);
      if (!piece.empty()) ct_[at] = id;
      if (call_from_ == cb_.size() - piece.size() && n_calls_ > 0) {
        // the next call after a finished one: its header is this token
      }
      return run_calls(d);
    }
    case Phase::Think:
    case Phase::Text:
      if (id == inkling::kEndMessage) return end_block(d);
      return release_text(piece, false, d);
    case Phase::Json: {
      const size_t at = cb_.size();
      cb_ += piece;
      ct_.resize(cb_.size(), -1);
      if (!piece.empty()) ct_[at] = id;
      R r;
      try {
        r = jvalue(In{cb_, ct_}, 0, 0);
      } catch (const NestingError& e) {
        msg_.errors.push_back(e.what());
        d.errors.push_back(e.what());
        phase_ = Phase::Frozen;
        return;
      }
      // The value ends where its text does; the end token must follow at once, and (a number is settled by the next
      // character) it may be this token itself.
      if (r.st == St::Fail || (r.st == St::Ok && r.end < cb_.size() && ct_[r.end] != inkling::kEndMessage)) return freeze(d);
      emit_text(cb_.substr(json_sent_, r.end - json_sent_), d);
      json_sent_ = r.end;
      if (r.st == St::Ok) phase_ = r.end < cb_.size() ? Phase::Ignore : Phase::JsonEnd;
      return;
    }
    case Phase::JsonEnd:
      if (id == inkling::kEndMessage) phase_ = Phase::Ignore;
      else freeze(d);
      return;
  }
}

OutputDelta InklingOutputParser::push(int32_t id) {
  OutputDelta d;
  if (done_) return d;
  if (is_end(id)) {
    d = finish();
    d.done = true;
    d.end_token = id;
    return d;
  }
  feed(id, d);
  return d;
}

OutputDelta InklingOutputParser::finish() {
  OutputDelta d;
  if (done_) return d;
  done_ = true;
  if (phase_ == Phase::Calls && open_.named) add_call(open_, true, d);
  if (failed_) {
    const char* e = "The model produced output that does not match the expected peg-native format";
    msg_.errors.push_back(e);
    d.errors.push_back(e);
  }
  return d;
}

}  // namespace eklow::text
