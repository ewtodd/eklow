#include "vega/checker.hpp"

#include <chrono>
#include <stdexcept>

#include "vega/files.hpp"

namespace eklow::vega {

namespace {

// vegaml's _crit_text: a description as written, anything else as Python's json.dumps(ensure_ascii=False) renders it.
std::optional<std::string> crit_text(const json::Value& v) {
  if (v.is_null()) return std::nullopt;
  if (v.is_string()) return v.as_string();
  return python_json(v, false, false, false);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

Question noul_question(const std::string& qid, const json::Value& def) {
  auto refuse = [&](const std::string& why) { throw Refusal(422, "question '" + qid + "': " + why); };
  if (qid.find_first_not_of(" \t\r\n") == std::string::npos) throw Refusal(422, "question id must be a non-empty string");
  if (!def.is_object()) refuse("definition must be an object");
  const json::Value* t = def.find("type");
  if (!t || !t->is_string()) refuse("no type; use one of choice, noul, score");
  if (t->as_string() == "choice" || t->as_string() == "score")
    refuse("type '" + t->as_string() + "' is not served here: eklow's System One answers noul (yes/no) questions only");
  if (t->as_string() != "noul") refuse("unknown type '" + t->as_string() + "'; use one of choice, noul, score");
  const json::Value* ins = def.find("instructions");
  if (!ins || ins->is_null()) refuse("no 'instructions'; add the text the model should answer");
  Question q;
  q.type = QType::Boolean;
  if (ins->is_string()) {
    if (ins->as_string().find_first_not_of(" \t\r\n") == std::string::npos) refuse("'instructions' must not be empty");
    q.instructions = ins->as_string();
  } else if ((ins->is_array() && ins->items().empty()) || (ins->is_object() && ins->members().empty())) {
    refuse("'instructions' must not be empty");
  } else if (ins->is_bool()) {
    refuse("'instructions' must be a string, dict, or list, got bool");
  } else {
    // vegaml renders non-string instructions with json.dumps (ensure_ascii, the default separators).
    q.instructions = python_json(*ins, false, false, true);
  }
  // laya: a noul's criteria as given, else its labels as the two sides ("no" / "yes" when a side is missing).
  if (const json::Value* c = def.find("criteria"); c && c->is_object()) {
    for (const auto& [k, v] : c->members()) q.criteria.emplace_back(k, crit_text(v));
  } else if (const json::Value* l = def.find("labels"); l && l->is_object()) {
    const json::Value* f = l->find("false");
    const json::Value* tr = l->find("true");
    q.criteria.emplace_back("false", f ? crit_text(*f) : std::optional<std::string>("no"));
    q.criteria.emplace_back("true", tr ? crit_text(*tr) : std::optional<std::string>("yes"));
  }
  return q;
}

Checker::Checker(sycl::queue& q, const std::string& reader, const std::string& vega_dir, const std::string& policy_dir, int64_t max_tokens,
                 int64_t chunk) {
  file_ = std::make_unique<gguf::File>(gguf::File::open(reader));
  tok_ = std::make_unique<text::QwenTokenizer>(text::QwenTokenizer::from_gguf(*file_));
  engine_ = std::make_unique<Engine>(vega_dir);
  if (!policy_dir.empty()) policy_ = std::make_unique<Policy>(Policy::load(policy_dir, engine_->sha256()));
  const EngineConfig& c = engine_->cfg();
  reader_ = std::make_unique<Reader>(q, *file_, c.feature_layers, c.option_layers, max_tokens, chunk);
  if (reader_->d_model() != c.d_model)
    throw std::runtime_error(reader + ": the reader's width is " + std::to_string(reader_->d_model()) + ", the engine's port takes " +
                             std::to_string(c.d_model));
}

PromptConfig Checker::prompt_config() const {
  const EngineConfig& c = engine_->cfg();
  PromptConfig p;
  p.max_len = c.max_len, p.opt_max_tokens = c.opt_max_tokens, p.query_first_max_state = c.query_first_max_state;
  return p;
}

PoolSpec Checker::pool_spec() const {
  const EngineConfig& c = engine_->cfg();
  PoolSpec s;
  s.chunks = int32_t(c.memory_chunks), s.chunk_min_tokens = int32_t(c.memory_min_tokens), s.chunk_tokens = int32_t(c.memory_chunk_tokens);
  s.prefix_min = int32_t(c.prefix_min_tokens);
  return s;
}

void Checker::warm() {
  const std::vector<std::pair<std::string, Question>> q{{"warm", Question{QType::Boolean, "Is this a warm-up?", {}}}};
  predict("A warm-up.", q);
  std::string longer;
  for (int i = 0; i < 400; ++i) longer += "The quick brown fox jumps over the lazy dog. ";
  predict(longer, q);
}

CheckerResult Checker::predict(const json::Value& body) {
  if (!body.is_object()) throw Refusal(400, "request body must be an object with a 'questions' field");
  const json::Value* qs = body.find("questions");
  if (!qs) throw Refusal(400, "request body must be an object with a 'questions' field");
  if (!qs->is_object()) throw Refusal(422, "'questions' must be an object of question id -> definition");
  const json::Value* st = body.find("state");
  if (!st || st->is_null()) throw Refusal(422, "predict takes a 'state'");
  // vegaml's serialize_state: text as it is, anything else as json.dumps writes it (default separators, ensure_ascii off).
  const std::string state = st->is_string() ? st->as_string() : python_json(*st, false, false, false);
  std::vector<std::pair<std::string, Question>> questions;
  for (const auto& [qid, def] : qs->members()) questions.emplace_back(qid, noul_question(qid, def));
  return predict(state, questions);
}

CheckerResult Checker::predict(const std::string& state, const std::vector<std::pair<std::string, Question>>& questions) {
  const auto t0 = std::chrono::steady_clock::now();
  CheckerResult r;
  if (questions.empty()) return r;
  const std::vector<int32_t> state_ids = tok_->encode(state, true);
  std::vector<Prompt> prompts;
  for (const auto& [qid, q] : questions) {
    prompts.push_back(build_prompt(*tok_, state_ids, q, prompt_config()));
    r.tokens += int64_t(prompts.back().ids.size());
  }
  const auto t1 = std::chrono::steady_clock::now();
  std::vector<Pooled> pooled;
  try {
    pooled = reader_->read(prompts, pool_spec());
  } catch (const std::length_error& e) {
    throw Refusal(413, std::string("state too long for this server's reader: ") + e.what());
  }
  r.read_ms = ms_since(t1);
  for (size_t i = 0; i < questions.size(); ++i) r.answers.push_back(answer(*engine_, policy_.get(), questions[i].first, questions[i].second, pooled[i]));
  r.total_ms = ms_since(t0);
  return r;
}

CheckerAnswer answer(const Engine& engine, const Policy* policy, const std::string& qid, const Question& q, const Pooled& pooled) {
  CheckerAnswer a;
  a.qid = qid;
  double p[2];
  if (const Head* h = policy ? policy->head_for(qid, {"false", "true"}) : nullptr) {
    const auto hp = h->predict(pooled.item);
    const double tot = hp[0] + hp[1];
    p[0] = hp[0] / (tot > 0 ? tot : 1.0), p[1] = hp[1] / (tot > 0 ? tot : 1.0);
    a.readout = "ttt";
  } else {
    const EngineAnswer e = engine.answer(pooled, q.type);
    p[0] = e.probs[0], p[1] = e.probs[1];
    a.readout = "engine";
  }
  if (const CalQuestion* c = policy ? policy->calibration(qid) : nullptr) {
    // vegaml's Calibration.apply, over the answer's probabilities keyed by the fitted labels (rounded to 6 places).
    if (c->labels.size() == 2 && ((c->labels[0] == "false" && c->labels[1] == "true") || (c->labels[0] == "true" && c->labels[1] == "false"))) {
      const bool swap = c->labels[0] == "true";
      const auto s = rescale({swap ? p[1] : p[0], swap ? p[0] : p[1]}, c->temperature);
      p[0] = round6(swap ? s[1] : s[0]), p[1] = round6(swap ? s[0] : s[1]);
      a.calibrated = true;
    }
  }
  a.probs[0] = p[0], a.probs[1] = p[1];
  a.noul = round6(p[1]);
  return a;
}

}  // namespace eklow::vega
