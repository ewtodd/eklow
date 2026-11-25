#include "serve/tool_grammar_inkling.hpp"

#include <algorithm>
#include <stdexcept>

namespace eklow::serve {

namespace inkling = text::inkling;
using Pu = InklingToolVocab::Pu;

namespace {

constexpr int32_t kHeader = inkling::kMessageModel;          // 200001
constexpr int32_t kText = inkling::kContentText;             // 200004
constexpr int32_t kThinking = inkling::kContentThinking;     // 200008
constexpr int32_t kEnd = inkling::kEndMessage;               // 200010
constexpr int32_t kCall = inkling::kContentInvokeToolJson;   // 200049

// Where the grammar is.
enum Mode : uint8_t {
  M_Pre,     // auto, before a call: any token; kCall starts one (accepting)
  M_Until,   // auto, after a further block's header: any token; kCall starts the call
  M_Block,   // required, in a thinking or text block: any token; kEnd closes it
  M_Idle,    // required, between blocks: kHeader
  M_After,   // a call is complete (accepting); with parallel calls a header may follow
  M_Choice,  // required, after a header: a block's opening token, the tool's name as text, or kCall
  M_Body,    // the call's JSON
  M_Done,    // an end-of-generation token was accepted
};

// The call: "{" space "\"name\"" space ":" space NAME space "," space "\"args\"" space ":" JSON space "}" kEnd
enum Item : uint8_t {
  I_OPEN, I_SP1, I_NAMEKEY, I_SP2, I_COLON1, I_SP3, I_NAMEVAL, I_SP4, I_COMMA, I_SP5, I_ARGSKEY, I_SP6, I_COLON2,
  I_JSON, I_SP7, I_CLOSE, I_END,
};

const char* literal_of(uint8_t item) {
  switch (item) {
    case I_OPEN: return "{";
    case I_NAMEKEY: return "\"name\"";
    case I_COLON1:
    case I_COLON2: return ":";
    case I_COMMA: return ",";
    case I_ARGSKEY: return "\"args\"";
    case I_CLOSE: return "}";
  }
  return "";
}

// JSON, inside the call's arguments.
enum Jm : uint8_t {
  J_VAL,       // a value, after a space
  J_ARR0,      // after "[": "]" or a value
  J_OBJ0,      // after "{": "}" or a key
  J_KEYS,      // after ",": a key
  J_STR,       // inside a string value (sub: the string's state)
  J_KSTR,      // inside a key
  J_AFTERKEY,  // after a key: space ":"
  J_NUM,       // inside a number (sub: its state)
  J_LIT,       // inside true, false or null (sub: which, off: matched)
  J_AFTER,     // after a value inside a container: space then "," or the container's closer
};

// llama.cpp's `space ::= | " " | "\n"{1,2} [ \t]{0,20}`: the state after a character, or -1. 0 nothing yet, 1 after " " (nothing
// more), 2 after one newline, 3 after two, 3 + k after k spaces or tabs.
int space_next(int sp, uint32_t c) {
  switch (sp) {
    case 0: return c == ' ' ? 1 : c == '\n' ? 2 : -1;
    case 1: return -1;
    case 2:
      if (c == '\n') return 3;
      [[fallthrough]];
    case 3: return (c == ' ' || c == '\t') ? 4 : -1;
    default: return ((c == ' ' || c == '\t') && sp < 23) ? sp + 1 : -1;
  }
}

// A JSON string's characters, as `( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )* "\""` (the escape class holds a
// space, as the converter wrote it). sub 0 plain, 1 after a backslash, 2-5 hex digits owed (4 down to 1). 0 refuses, 1 goes on,
// 2 the closing quote.
int str_step(uint8_t& sub, uint32_t c) {
  if (sub == 0) {
    if (c == '"') return 2;
    if (c == '\\') return sub = 1, 1;
    return c != 0;
  }
  if (sub == 1) {
    if (c == '"' || c == '\\' || c == '/' || c == ' ' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') return sub = 0, 1;
    if (c == 'u') return sub = 2, 1;
    return 0;
  }
  const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
  if (!hex) return 0;
  sub = sub == 5 ? 0 : uint8_t(sub + 1);
  return 1;
}

// A number: -? (0 | [1-9][0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
enum Num : uint8_t { N_MINUS = 1, N_ZERO, N_INT, N_DOT, N_FRAC, N_E, N_ESIGN, N_EXP };

bool num_step(uint8_t& st, uint32_t c) {
  const bool d = c >= '0' && c <= '9';
  switch (st) {
    case N_MINUS:
      if (c == '0') return st = N_ZERO, true;
      if (d) return st = N_INT, true;
      return false;
    case N_ZERO:
    case N_INT:
      if (st == N_INT && d) return true;
      if (c == '.') return st = N_DOT, true;
      if (c == 'e' || c == 'E') return st = N_E, true;
      return false;
    case N_DOT: return d ? (st = N_FRAC, true) : false;
    case N_FRAC:
      if (d) return true;
      if (c == 'e' || c == 'E') return st = N_E, true;
      return false;
    case N_E:
      if (c == '+' || c == '-') return st = N_ESIGN, true;
      return d ? (st = N_EXP, true) : false;
    case N_ESIGN: return d ? (st = N_EXP, true) : false;
    case N_EXP: return d;
  }
  return false;
}

bool num_complete(uint8_t st) { return st == N_ZERO || st == N_INT || st == N_FRAC || st == N_EXP; }

const char* word_of(uint8_t kind) { return kind == 1 ? "true" : kind == 2 ? "false" : "null"; }

// llama.cpp's llama_grammar_decode_utf8 (src/llama-grammar.cpp), on a piece without NULs: the code points the piece completes, and
// the character it leaves begun. A piece that does not continue a begun character, or begins with a continuation byte or an
// invalid lead, leaves n_remain negative.
Pu decode(const std::string& src, Pu start, std::vector<uint32_t>& cps) {
  static const int lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};
  cps.clear();
  size_t pos = 0;
  const size_t n = src.size();
  uint32_t value = start.value;
  int n_remain = start.n_remain;
  while (pos < n && n_remain > 0) {
    const uint8_t b = uint8_t(src[pos]);
    if ((b >> 6) != 2) {
      cps.clear();
      return Pu{0, -1};
    }
    value = (value << 6) + (b & 0x3F);
    ++pos;
    --n_remain;
  }
  if (start.n_remain > 0 && n_remain == 0) cps.push_back(value);
  while (pos < n) {
    const uint8_t first = uint8_t(src[pos]);
    n_remain = lookup[first >> 4] - 1;
    if (n_remain < 0) {
      cps.clear();
      return Pu{0, n_remain};
    }
    const uint8_t mask = uint8_t((1 << (7 - n_remain)) - 1);
    value = first & mask;
    ++pos;
    while (pos < n && n_remain > 0) {
      value = (value << 6) + (uint8_t(src[pos]) & 0x3F);
      ++pos;
      --n_remain;
    }
    if (n_remain == 0) cps.push_back(value);
  }
  return Pu{n_remain == 0 ? 0 : value, n_remain};
}

// The code points a character begun can still complete to, as llama_grammar_match_partial_char computes them; false when the
// begun character cannot be valid.
bool partial_range(Pu p, uint32_t& low, uint32_t& high) {
  if (p.n_remain <= 0 || (p.n_remain == 1 && p.value < 2)) return false;
  low = p.value << (p.n_remain * 6);
  high = low | ((1u << (p.n_remain * 6)) - 1);
  if (low == 0) {
    if (p.n_remain == 2) low = 1u << 11;
    else if (p.n_remain == 3) low = 1u << 16;
  }
  return true;
}

std::vector<uint32_t> code_points(const std::string& s) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < s.size();) {
    const uint8_t b = uint8_t(s[i]);
    int len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 1;
    if (i + size_t(len) > s.size()) len = 1;
    uint32_t v = len == 1 ? b : b & (0xFFu >> (len + 1));
    for (int k = 1; k < len; ++k) v = (v << 6) | (uint8_t(s[i + size_t(k)]) & 0x3F);
    out.push_back(v);
    i += size_t(len);
  }
  return out;
}

// nlohmann's json(name).dump() between the quotes, which is what the grammar's JSON name holds in required mode.
std::string json_escaped(const std::string& s) {
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
          static const char* hx = "0123456789abcdef";
          o += "\\u00";
          o += hx[c >> 4];
          o += hx[c & 15];
        } else {
          o += char(c);
        }
    }
  }
  return o;
}

}  // namespace

// ---- the vocabulary ----

InklingToolVocab::InklingToolVocab(const text::InklingTokenizer& tok) {
  n = tok.size();
  piece.resize(n);
  dead.assign(n, 0);
  cp_off.assign(n + 1, 0);
  end_pu.resize(n);
  std::vector<uint32_t> cps;
  for (size_t id = 0; id < n; ++id) {
    std::string b = tok.bytes(int32_t(id));
    const bool empty_or_nul = b.empty() || b[0] == 0;
    if (const size_t z = b.find('\0'); z != std::string::npos) b.resize(z);  // llama.cpp reads the piece as a C string
    piece[id] = std::move(b);
    dead[id] = empty_or_nul;
    end_pu[id] = decode(piece[id], Pu{}, cps);
    cp_off[id] = uint32_t(cp_pool.size());
    cp_pool.insert(cp_pool.end(), cps.begin(), cps.end());
  }
  cp_off[n] = uint32_t(cp_pool.size());
  for (size_t id = 0; id < n; ++id)
    if (dead[id] || end_pu[id].n_remain < 0) never.push_back(int32_t(id));
}

// ---- the grammar ----

struct InklingToolGrammar::Tools {
  std::vector<std::vector<uint32_t>> raw;     // each tool's name: what may precede the call token
  std::vector<std::vector<uint32_t>> quoted;  // `"` + the JSON-escaped name + `"`: the call's "name"
  uint64_t hash = 0;
};

InklingToolGrammar::InklingToolGrammar(std::shared_ptr<const InklingToolVocab> v, Choice c, std::vector<std::string> tool_names,
                                       bool parallel)
    : v_(std::move(v)), choice_(c), parallel_(parallel) {
  auto tools = std::make_shared<Tools>();
  if (c == Choice::Required) {
    if (tool_names.empty()) throw std::invalid_argument("tool grammar: tool_choice required needs a tool");
    std::sort(tool_names.begin(), tool_names.end());
    tool_names.erase(std::unique(tool_names.begin(), tool_names.end()), tool_names.end());
    uint64_t h = 1469598103934665603ull;
    for (const auto& name : tool_names) {
      tools->raw.push_back(code_points(name));
      tools->quoted.push_back(code_points("\"" + json_escaped(name) + "\""));
      for (unsigned char ch : name) h = (h ^ ch) * 1099511628211ull;
      h = (h ^ 0xFF) * 1099511628211ull;
    }
    tools->hash = h;
  }
  tools_ = std::move(tools);
  prefix_ = c == Choice::Auto ? "A" : "R" + std::to_string(tools_->hash) + "/" + std::to_string(tools_->raw.size());
  prefix_ += parallel ? "p|" : "s|";
  st_.mode = c == Choice::Auto ? M_Pre : M_Idle;
}

// The tokens a rule of the grammar names: they match by id only (llama.cpp's opaque tokens). Required mode names the block tokens too.
bool InklingToolGrammar::opaque(int32_t id) const {
  if (id == kEnd || id == kCall) return true;
  if (id == kHeader) return choice_ == Choice::Required || parallel_;  // auto without parallel calls never names it
  return choice_ == Choice::Required && (id == kText || id == kThinking);
}

bool InklingToolGrammar::accepting(const State& s) const { return s.mode == M_Pre || s.mode == M_After; }
bool InklingToolGrammar::can_end() const { return accepting(st_); }
bool InklingToolGrammar::ended() const { return st_.mode == M_Done; }
bool InklingToolGrammar::unconstrained() const { return st_.mode == M_Pre && st_.pu.n_remain == 0; }

// Whether a character begun (n_remain more bytes to come) could be the next character the grammar wants.
bool InklingToolGrammar::partial_ok(const State& s, Pu pu) const {
  uint32_t low, high;
  if (!partial_range(pu, low, high)) return false;
  auto in_range = [&](const std::vector<uint32_t>& name) { return s.off < name.size() && low <= name[s.off] && name[s.off] <= high; };
  auto any_alive = [&](const std::vector<std::vector<uint32_t>>& names) {
    if (s.all) {
      for (const auto& nm : names)
        if (in_range(nm)) return true;
    } else {
      for (uint32_t i : s.alive)
        if (in_range(names[i])) return true;
    }
    return false;
  };
  if (s.mode == M_Choice) return any_alive(tools_->raw);
  if (s.mode != M_Body) return false;
  if (s.item == I_JSON) return (s.jm == J_STR || s.jm == J_KSTR) && s.sub == 0;  // [^"\\] takes any begun character
  if (s.item == I_NAMEVAL) {
    if (choice_ == Choice::Required) return any_alive(tools_->quoted);
    return s.off > 0 && s.sub == 0;
  }
  return false;
}

// One code point of the call (mode M_Body) or of a tool's name (mode M_Choice).
bool InklingToolGrammar::step_cp(State& s, uint32_t c) const {
  if (s.mode == M_Choice) {
    std::vector<uint32_t> next;
    auto test = [&](uint32_t i) {
      const auto& name = tools_->raw[i];
      if (s.off < name.size() && name[s.off] == c) next.push_back(i);
    };
    if (s.all) {
      for (uint32_t i = 0; i < tools_->raw.size(); ++i) test(i);
    } else {
      for (uint32_t i : s.alive) test(i);
    }
    if (next.empty()) return false;
    s.all = false;
    s.alive = std::move(next);
    ++s.off;
    return true;
  }
  if (s.mode != M_Body) return false;

  auto value_done = [&]() {
    s.sp = 0;
    s.sub = 0;
    if (s.stack.empty()) {
      s.item = I_SP7;
    } else {
      s.jm = J_AFTER;
    }
  };
  auto value_start = [&]() {
    s.sp = 0;
    switch (c) {
      case '"': s.jm = J_STR, s.sub = 0; return true;
      case '{': s.stack.push_back('o'); s.jm = J_OBJ0; return true;
      case '[': s.stack.push_back('a'); s.jm = J_ARR0; return true;
      case '-': s.jm = J_NUM, s.sub = N_MINUS; return true;
      case '0': s.jm = J_NUM, s.sub = N_ZERO; return true;
      case 't': s.jm = J_LIT, s.sub = 1, s.off = 1; return true;
      case 'f': s.jm = J_LIT, s.sub = 2, s.off = 1; return true;
      case 'n': s.jm = J_LIT, s.sub = 3, s.off = 1; return true;
    }
    if (c >= '1' && c <= '9') return s.jm = J_NUM, s.sub = N_INT, true;
    return false;
  };
  auto space = [&]() {
    const int nx = space_next(s.sp, c);
    if (nx < 0) return false;
    s.sp = uint8_t(nx);
    return true;
  };
  auto pop = [&]() {
    s.stack.pop_back();
    value_done();
    return true;
  };

  for (int guard = 0; guard < 8; ++guard) {  // a character that ends a number or a space is read again by what follows
    switch (s.item) {
      case I_OPEN:
      case I_NAMEKEY:
      case I_COLON1:
      case I_COMMA:
      case I_ARGSKEY:
      case I_COLON2:
      case I_CLOSE: {
        const char* lit = literal_of(s.item);
        if (c != uint8_t(lit[s.off])) return false;
        if (lit[++s.off] == 0) {
          ++s.item;
          s.off = 0;
          s.sp = 0;
          if (s.item == I_JSON) s.jm = J_VAL, s.sub = 0, s.stack.clear();
        }
        return true;
      }
      case I_SP1:
      case I_SP2:
      case I_SP3:
      case I_SP4:
      case I_SP5:
      case I_SP6:
      case I_SP7:
        if (space()) return true;
        ++s.item;
        s.sp = 0;
        s.off = 0;
        s.sub = 0;
        continue;
      case I_NAMEVAL: {
        if (choice_ == Choice::Auto) {
          if (s.off == 0) {
            if (c != '"') return false;
            s.off = 1;
            s.sub = 0;
            return true;
          }
          const int r = str_step(s.sub, c);
          if (r == 0) return false;
          if (r == 2) s.item = I_SP4, s.sp = 0, s.off = 0, s.sub = 0;
          return true;
        }
        std::vector<uint32_t> next;
        auto test = [&](uint32_t i) {
          const auto& name = tools_->quoted[i];
          if (s.off < name.size() && name[s.off] == c) next.push_back(i);
        };
        if (s.all) {
          for (uint32_t i = 0; i < tools_->quoted.size(); ++i) test(i);
        } else {
          for (uint32_t i : s.alive) test(i);
        }
        if (next.empty()) return false;
        ++s.off;
        s.all = false;
        s.alive = std::move(next);
        for (uint32_t i : s.alive)
          if (tools_->quoted[i].size() == s.off) {
            s.item = I_SP4;
            s.sp = 0;
            s.off = 0;
            s.all = true;
            s.alive.clear();
            break;
          }
        return true;
      }
      case I_JSON:
        switch (s.jm) {
          case J_VAL: return space() || value_start();
          case J_ARR0:
            if (space()) return true;
            if (c == ']') return pop();
            return value_start();
          case J_OBJ0:
            if (space()) return true;
            if (c == '}') return pop();
            if (c == '"') return s.sp = 0, s.jm = J_KSTR, s.sub = 0, true;
            return false;
          case J_KEYS:
            if (space()) return true;
            if (c == '"') return s.sp = 0, s.jm = J_KSTR, s.sub = 0, true;
            return false;
          case J_STR: {
            const int r = str_step(s.sub, c);
            if (r == 2) value_done();
            return r != 0;
          }
          case J_KSTR: {
            const int r = str_step(s.sub, c);
            if (r == 2) s.jm = J_AFTERKEY, s.sp = 0, s.sub = 0;
            return r != 0;
          }
          case J_AFTERKEY:
            if (space()) return true;
            if (c != ':') return false;
            s.jm = J_VAL;
            s.sp = 0;
            return true;
          case J_NUM:
            if (num_step(s.sub, c)) return true;
            if (!num_complete(s.sub)) return false;
            value_done();
            continue;
          case J_LIT: {
            const char* w = word_of(s.sub);
            if (c != uint8_t(w[s.off])) return false;
            if (w[++s.off] == 0) s.off = 0, value_done();
            return true;
          }
          case J_AFTER:
            if (space()) return true;
            if (c == ',') {
              s.jm = s.stack.back() == 'a' ? J_VAL : J_KEYS;
              s.sp = 0;
              return true;
            }
            if (c == ']' && s.stack.back() == 'a') return pop();
            if (c == '}' && s.stack.back() == 'o') return pop();
            return false;
        }
        return false;
      case I_END: return false;
    }
  }
  return false;
}

// The whole transition on one token, as llama.cpp's grammar sampler would take it (and refuse it: false, s unchanged only if
// the caller works on a copy).
bool InklingToolGrammar::feed(State& s, int32_t id) const {
  const InklingToolVocab& v = *v_;
  if (id < 0 || size_t(id) >= v.n) return false;
  if (id == v.eog[0] || id == v.eog[1]) {
    if (!accepting(s)) return false;
    s.mode = M_Done;
    return true;
  }
  if (s.mode == M_Done || v.dead[size_t(id)]) return false;

  // The token's code points and the character it leaves begun, given the one begun before it.
  std::vector<uint32_t> slow;
  const uint32_t* cps;
  size_t ncp;
  Pu npu;
  if (s.pu.n_remain == 0) {
    cps = v.cp_pool.data() + v.cp_off[size_t(id)];
    ncp = v.cp_off[size_t(id) + 1] - v.cp_off[size_t(id)];
    npu = v.end_pu[size_t(id)];
  } else {
    npu = decode(v.piece[size_t(id)], s.pu, slow);
    cps = slow.data();
    ncp = slow.size();
  }
  if (npu.n_remain < 0) return false;

  auto reset = [&]() {  // a state that holds no call: the same state whatever came before
    s.item = s.sp = s.sub = s.jm = 0;
    s.off = 0;
    s.stack.clear();
    s.all = true;
    s.alive.clear();
    s.blocks_ok = false;
  };
  auto by_id = [&]() {  // the token matched a rule that names it (or any token but one): characters do not count
    s.pu = npu;
    return true;
  };
  switch (s.mode) {
    case M_Pre:
    case M_Until:
      if (id == kCall) {
        reset();
        s.mode = M_Body;
      }
      return by_id();
    case M_Block:
      if (id == kEnd) reset(), s.mode = M_Idle;
      return by_id();
    case M_Idle:
      if (id != kHeader) return false;
      reset();
      s.mode = M_Choice;
      s.blocks_ok = true;
      return by_id();
    case M_After:
      if (!parallel_ || id != kHeader) return false;
      if (choice_ == Choice::Auto) {
        reset();
        s.mode = M_Until;
      } else {
        reset();
        s.mode = M_Choice;
      }
      return by_id();
    case M_Choice:
      if ((id == kThinking || id == kText) && s.blocks_ok && s.off == 0) {
        reset();
        s.mode = M_Block;
        return by_id();
      }
      if (id == kCall) {
        if (s.off > 0) {  // the tools whose name was written in full
          std::vector<uint32_t> done;
          for (uint32_t i : s.alive)
            if (tools_->raw[i].size() == s.off) done.push_back(i);
          if (done.empty()) return false;
          s.alive = std::move(done);
          s.all = false;
        }
        s.mode = M_Body;
        s.blocks_ok = false;
        s.item = s.sp = s.sub = s.jm = 0;
        s.off = 0;
        s.stack.clear();
        return by_id();
      }
      break;
    case M_Body:
      if (s.item == I_END) {
        if (id != kEnd) return false;
        reset();
        s.mode = M_After;
        return by_id();
      }
      break;
    default: return false;
  }
  // By its characters: a token the grammar names is no character.
  if (opaque(id)) return false;
  for (size_t i = 0; i < ncp; ++i)
    if (!step_cp(s, cps[i])) return false;
  if (npu.n_remain != 0 && !partial_ok(s, npu)) return false;
  s.pu = npu;
  return true;
}

// The state, as the part of the allowed set's key that the tokens do not already fix.
std::string InklingToolGrammar::key(const State& s) const {
  std::string k = prefix_;
  k += char('a' + s.mode);
  k += char('a' + s.item);
  k += char('a' + s.sp);
  k += char('a' + s.sub);
  k += char('a' + s.jm);
  k += s.blocks_ok ? 'b' : 'n';
  k += char('0' + s.pu.n_remain);
  // The pending character's value matters where characters do.
  if (s.pu.n_remain > 0 && (s.mode == M_Body || s.mode == M_Choice)) k.append(reinterpret_cast<const char*>(&s.pu.value), 4);
  k.append(reinterpret_cast<const char*>(&s.off), 4);
  k += '|';
  k += s.stack;
  k += '|';
  if (!s.all)
    for (uint32_t i : s.alive) k.append(reinterpret_cast<const char*>(&i), 4);
  return k;
}

InklingToolVocab::Set InklingToolGrammar::compute(const State& s) const {
  const InklingToolVocab& v = *v_;
  auto out = std::make_shared<std::vector<int32_t>>();
  if (s.mode == M_Done) return out;
  out->reserve(s.mode == M_Pre || s.mode == M_Until || s.mode == M_Block ? v.n : 1024);
  const bool char_mode = s.mode == M_Body ? s.item != I_END : s.mode == M_Choice;
  const bool pending = s.pu.n_remain > 0;

  // For a state that reads characters: which first code points get a token anywhere (ASCII ones exactly), so that most tokens are
  // refused without a copy of the state.
  bool ascii_ok[128] = {};
  bool other_ok = true;  // non-ASCII first code points: tried per token where names can hold them
  if (char_mode && !pending) {
    for (uint32_t c = 1; c < 128; ++c) {
      State t = s;
      ascii_ok[c] = step_cp(t, c);
    }
    other_ok = false;
    if (s.mode == M_Choice || (s.mode == M_Body && s.item == I_NAMEVAL && choice_ == Choice::Required)) {
      other_ok = true;  // names may hold any character
    } else {
      State t = s;
      other_ok = step_cp(t, 0x4E2D);  // string interiors take every character the same
    }
  }
  for (size_t id = 0; id < v.n; ++id) {
    if (!pending && char_mode && !v.dead[id]) {
      if (v.end_pu[id].n_remain < 0) continue;
      const int32_t i32 = int32_t(id);
      if (opaque(i32)) {
        State t = s;
        if (feed(t, i32)) out->push_back(i32);
        continue;
      }
      if (i32 == v.eog[0] || i32 == v.eog[1]) {
        if (accepting(s)) out->push_back(i32);
        continue;
      }
      const uint32_t b = v.cp_off[id], e = v.cp_off[id + 1];
      if (b != e) {
        const uint32_t c0 = v.cp_pool[b];
        if (c0 < 128 ? !ascii_ok[c0] : !other_ok) continue;
      }
      State t = s;
      if (feed(t, i32)) out->push_back(i32);
      continue;
    }
    State t = s;
    if (feed(t, int32_t(id))) out->push_back(int32_t(id));
  }
  return out;
}

const std::vector<int32_t>& InklingToolGrammar::allowed() {
  if (allowed_valid_) return *allowed_;
  const std::string k = key(st_);
  {
    std::lock_guard lk(v_->mu);
    if (auto it = v_->cache.find(k); it != v_->cache.end()) {
      allowed_ = it->second;
      allowed_valid_ = true;
      return *allowed_;
    }
  }
  allowed_ = compute(st_);
  allowed_valid_ = true;
  std::lock_guard lk(v_->mu);
  if (v_->cached_ids > (size_t(1) << 26)) {
    v_->cache.clear();
    v_->cached_ids = 0;
  }
  if (v_->cache.emplace(k, allowed_).second) v_->cached_ids += allowed_->size();
  return *allowed_;
}

bool InklingToolGrammar::accept(int32_t id) {
  State s = st_;
  if (!feed(s, id)) return false;
  st_ = std::move(s);
  allowed_valid_ = false;
  return true;
}

}  // namespace eklow::serve
