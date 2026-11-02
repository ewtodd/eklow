#include "text/output_parser_qwen.hpp"

#include <algorithm>
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

int32_t need_id(const QwenTokenizer& t, const char* piece) {
  int32_t id = t.id(piece);
  if (id < 0) throw std::runtime_error(std::string("tokenizer has no ") + piece);
  return id;
}

bool ws(char c) { return c == ' ' || c == '\n' || c == '\t' || c == '\r'; }

}  // namespace

ToolCall parse_qwen_tool_call(const std::string& body, const ToolSpecs* tools) {
  const size_t n = body.size();
  size_t i = 0;
  auto skip = [&] {
    while (i < n && ws(body[i])) ++i;
  };
  auto at = [&](const char* lit) { return body.compare(i, std::char_traits<char>::length(lit), lit) == 0; };
  auto fail = [&](const std::string& what) {
    throw std::runtime_error("tool call: " + what + " at byte " + std::to_string(i));
  };
  skip();
  if (!at("<function=")) fail("expected '<function='");
  i += 10;
  const size_t gt = body.find('>', i);
  if (gt == std::string::npos) fail("no '>' after the function name");
  ToolCall call;
  call.name = body.substr(i, gt - i);
  if (call.name.empty()) fail("no function name");
  call.arguments = json::Value::object();
  i = gt + 1;
  const ToolSpec* spec = tools ? tools->find(call.name) : nullptr;
  for (;;) {
    skip();
    if (i >= n) break;  // a call that never wrote </function> still ends here
    if (at("</function>")) {
      i += 11;
      skip();
      if (i < n) fail("text after </function>");
      break;
    }
    if (!at("<parameter=")) fail("unexpected text '" + body.substr(i, 20) + "'");
    i += 11;
    const size_t kgt = body.find('>', i);
    if (kgt == std::string::npos) fail("no '>' after a parameter name");
    const std::string key = body.substr(i, kgt - i);
    const size_t start = kgt + 1;
    // The value ends at the first </parameter> that is followed, past white space, by another parameter, </function>
    // or the end: a value that merely contains "</parameter>" is the model's own text.
    size_t end = std::string::npos;
    for (size_t p = body.find("</parameter>", start); p != std::string::npos; p = body.find("</parameter>", p + 1)) {
      size_t q = p + 12;
      while (q < n && ws(body[q])) ++q;
      if (q >= n || body.compare(q, 11, "<parameter=") == 0 || body.compare(q, 11, "</function>") == 0) {
        end = p;
        break;
      }
    }
    if (end == std::string::npos) {
      i = start;
      fail("parameter '" + key + "' is not closed");
    }
    std::string raw = body.substr(start, end - start);
    if (!raw.empty() && raw.front() == '\n') raw.erase(0, 1);
    if (!raw.empty() && raw.back() == '\n') raw.pop_back();
    call.arguments.set(key, convert_param(spec ? spec->param(key) : nullptr, raw));
    i = end + 12;
  }
  return call;
}

QwenOutputParser::QwenOutputParser(const QwenTokenizer& tok, bool in_thinking, const ToolSpecs* tools) : tok_(tok), tools_(tools) {
  think_close_ = need_id(tok, "</think>");
  call_open_ = need_id(tok, "<tool_call>");
  call_close_ = need_id(tok, "</tool_call>");
  im_end_ = need_id(tok, "<|im_end|>");
  endoftext_ = need_id(tok, "<|endoftext|>");
  if (in_thinking) state_ = State::Thinking;
  auto ban_all_but = [&](State s, std::vector<int32_t> ok) {
    for (int32_t id : tok.special_ids())
      if (std::find(ok.begin(), ok.end(), id) == ok.end()) banned_[size_t(s)].push_back(id);
  };
  ban_all_but(State::Thinking, {think_close_, im_end_, endoftext_});
  ban_all_but(State::Content, {call_open_, im_end_, endoftext_});
  ban_all_but(State::ToolCall, {call_close_});
}

void QwenOutputParser::begin_block() {
  pending_.clear();
  held_newlines_ = 0;
  block_started_ = false;
}

// Releases the whole characters of pending_ (all of it when flushing), leaving out newlines at the start of the block
// and withholding the ones at the end of what is released until more text follows (and dropping them at the end).
void QwenOutputParser::emit_text(std::string& out, bool flush) {
  const size_t n = flush ? pending_.size() : complete_prefix(pending_);
  for (size_t i = 0; i < n; ++i) {
    const char c = pending_[i];
    if (c == '\n') {
      if (block_started_) ++held_newlines_;
      continue;
    }
    if (!block_started_) {
      block_started_ = true;
      // Text after a tool call, with content before it, starts a new paragraph.
      if (state_ == State::Content && after_call_ && content_emitted_) out += "\n\n";
    } else if (held_newlines_) {
      out.append(held_newlines_, '\n');
      held_newlines_ = 0;
    }
    if (state_ == State::Content) content_emitted_ = true;
    out += c;
  }
  pending_.erase(0, n);
  if (flush) held_newlines_ = 0;
}

void QwenOutputParser::finish_call(OutputDelta& d) {
  const std::string body = std::move(call_);
  call_.clear();
  try {
    d.tool_calls.push_back(parse_qwen_tool_call(body, tools_));
  } catch (const std::exception& e) {
    d.errors.push_back(e.what());
  }
}

OutputDelta QwenOutputParser::push(int32_t id) {
  OutputDelta d;
  if (state_ == State::Done) return d;
  const bool special = tok_.is_special(id);
  if (is_end(id)) {
    if (state_ == State::ToolCall) {
      d.errors.push_back("tool call: not closed before the end of the turn");
      call_.clear();
    } else {
      emit_text(state_ == State::Thinking ? d.reasoning : d.content, true);
    }
    state_ = State::Done;
    d.done = true;
    d.end_token = id;
    return d;
  }
  switch (state_) {
    case State::Thinking:
      if (id == think_close_) {
        emit_text(d.reasoning, true);
        state_ = State::Content;
        begin_block();
      } else if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in reasoning");
      } else {
        pending_ += tok_.bytes(id);
        emit_text(d.reasoning, false);
      }
      break;
    case State::Content:
      if (id == call_open_) {
        emit_text(d.content, true);
        call_.clear();
        state_ = State::ToolCall;
      } else if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in content");
      } else {
        pending_ += tok_.bytes(id);
        emit_text(d.content, false);
      }
      break;
    case State::ToolCall:
      if (id == call_close_) {
        finish_call(d);
        state_ = State::Content;
        after_call_ = true;
        begin_block();
      } else if (special) {
        d.errors.push_back("unexpected " + tok_.piece(id) + " in a tool call");
      } else {
        call_ += tok_.bytes(id);
      }
      break;
    case State::Done: break;
  }
  return d;
}

OutputDelta QwenOutputParser::finish() {
  OutputDelta d;
  if (state_ == State::ToolCall) d.errors.push_back("tool call: not closed");
  else if (state_ != State::Done) emit_text(state_ == State::Thinking ? d.reasoning : d.content, true);
  state_ = State::Done;
  return d;
}

}  // namespace eklow::text
