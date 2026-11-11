#include "vega/prompt.hpp"

#include <algorithm>
#include <stdexcept>

namespace eklow::vega {

const char* qtype_name(QType t) {
  switch (t) {
    case QType::Choice: return "choice";
    case QType::Score: return "score";
    case QType::Boolean: return "boolean";
  }
  return "?";
}

std::vector<std::string> option_texts(const Question& q) {
  std::vector<std::string> out;
  if (q.type == QType::Choice) {
    for (const auto& [k, v] : q.criteria) out.push_back(v && !v->empty() ? k + " - " + *v : k);
  } else if (q.type == QType::Score) {
    for (size_t i = 0; i < q.criteria.size(); ++i) out.push_back("level " + std::to_string(i) + " - " + q.criteria[i].second.value_or(""));
  } else {
    std::optional<std::string> no, yes;
    for (const auto& [k, v] : q.criteria) {
      if (k == "false") no = v;
      else if (k == "true") yes = v;
    }
    out.push_back("no - " + (no && !no->empty() ? *no : std::string("the statement does not hold")));
    out.push_back("yes - " + (yes && !yes->empty() ? *yes : std::string("the statement holds")));
  }
  return out;
}

std::vector<std::string> option_ids(const Question& q) {
  std::vector<std::string> out;
  if (q.type == QType::Boolean) return {"False", "True"};
  for (size_t i = 0; i < q.criteria.size(); ++i) out.push_back(q.type == QType::Score ? std::to_string(i) : q.criteria[i].first);
  return out;
}

namespace {

int64_t floordiv(int64_t a, int64_t b) { return a / b - ((a % b != 0) && ((a < 0) != (b < 0))); }

}  // namespace

Prompt build_prompt(const text::QwenTokenizer& tok, const std::vector<int32_t>& state_ids, const Question& q, const PromptConfig& c) {
  const std::string t = qtype_name(q.type);
  auto enc = [&](const std::string& s) { return tok.encode(s, true); };
  auto cut = [](std::vector<int32_t> v, int64_t n) {
    if (int64_t(v.size()) > n) v.resize(size_t(std::max<int64_t>(n, 0)));
    return v;
  };
  const std::vector<std::string> opts = option_texts(q);
  if (opts.size() < 2) throw std::runtime_error("a question needs at least two options");
  const std::vector<int32_t> head = enc("Observation:\n"), end = enc("\nOutcome:");
  std::vector<int32_t> qids = cut(enc("\n\nMeasurement (" + t + "): " + q.instructions + "\nPossible outcomes:"), c.ins_max_tokens);
  std::vector<std::vector<int32_t>> oids;
  for (const auto& o : opts) oids.push_back(cut(enc("\n* " + o), c.opt_max_tokens));
  auto opt_sum = [&] {
    int64_t s = 0;
    for (const auto& o : oids) s += int64_t(o.size());
    return s;
  };
  const int64_t H = int64_t(head.size()), En = int64_t(end.size());
  int64_t fixed = H + int64_t(qids.size()) + opt_sum() + En;
  if (fixed > c.max_len - 16) {  // too many or too long options: every option shrunk evenly, then the instructions
    const int64_t per = std::max<int64_t>(3, floordiv(c.max_len - 16 - H - En - std::min<int64_t>(int64_t(qids.size()), 64), int64_t(oids.size())));
    for (auto& o : oids) o = cut(o, per);
    qids = cut(qids, std::max<int64_t>(8, c.max_len - 16 - H - En - opt_sum()));
    fixed = H + int64_t(qids.size()) + opt_sum() + En;
  }
  std::vector<int32_t> pre;
  if (c.query_first_max_state && int64_t(state_ids.size()) < c.query_first_max_state) {
    pre = cut(enc("Question to keep in mind while reading (" + t + "): " + q.instructions + "\n\n"), std::min<int64_t>(128, c.ins_max_tokens));
    fixed += int64_t(pre.size());
  }
  const int64_t room = std::max<int64_t>(0, c.max_len - fixed);
  const int64_t ns = std::min<int64_t>(int64_t(state_ids.size()), room);
  Prompt p;
  p.ids = pre;
  p.ids.insert(p.ids.end(), head.begin(), head.end());
  const int32_t s0 = int32_t(p.ids.size());
  p.ids.insert(p.ids.end(), state_ids.begin(), state_ids.begin() + ns);
  // An empty state still needs a span: the header's tokens.
  if (ns > 0) p.s0 = s0, p.s1 = int32_t(p.ids.size());
  else p.s0 = 0, p.s1 = s0;
  p.q0 = int32_t(p.ids.size());
  p.ids.insert(p.ids.end(), qids.begin(), qids.end());
  p.q1 = int32_t(p.ids.size());
  for (const auto& o : oids) {
    const int32_t a = int32_t(p.ids.size());
    p.ids.insert(p.ids.end(), o.begin(), o.end());
    p.options.emplace_back(a, int32_t(p.ids.size()));
  }
  p.ids.insert(p.ids.end(), end.begin(), end.end());
  return p;
}

}  // namespace eklow::vega
