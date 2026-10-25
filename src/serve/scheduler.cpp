#include "serve/scheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace eklow::serve {

bool Request::next(Event& e, std::chrono::milliseconds wait) {
  std::unique_lock lk(mu_);
  if (!cv_.wait_for(lk, wait, [&] { return !events_.empty(); })) return false;
  e = std::move(events_.front());
  events_.pop_front();
  return true;
}

void Request::cancel() {
  std::lock_guard lk(mu_);
  cancelled_ = true;
}

void Request::push(Event e) {
  {
    std::lock_guard lk(mu_);
    events_.push_back(std::move(e));
  }
  cv_.notify_all();
}

bool Request::cancelled() const {
  std::lock_guard lk(mu_);
  return cancelled_;
}

Scheduler::Scheduler(engine::Engine& eng, const text::Tokenizer& tok, const SchedulerConfig& c)
    : eng_(eng), tok_(tok), vocab_(std::make_shared<ToolVocab>(tok)), c_(c), seqs_(size_t(eng.n_slots())) {}

void Scheduler::submit(std::shared_ptr<Request> r) {
  if (r->prompt.empty()) throw std::invalid_argument("empty prompt");
  if (int64_t(r->prompt.size()) >= c_.max_ctx)
    throw std::invalid_argument("the prompt is " + std::to_string(r->prompt.size()) + " tokens, the context " +
                                std::to_string(c_.max_ctx));
  {
    std::lock_guard lk(mu_);
    waiting_.push_back(std::move(r));
    ++stats_.waiting;
  }
  cv_.notify_all();
}

Scheduler::Stats Scheduler::stats() const {
  std::lock_guard lk(mu_);
  return stats_;
}

// Gives waiting requests free slots; true if any sequence is active afterwards.
bool Scheduler::admit() {
  std::lock_guard lk(mu_);
  while (!waiting_.empty() && waiting_.front()->cancelled()) {
    waiting_.pop_front();
    --stats_.waiting;
  }
  while (!waiting_.empty()) {
    auto& r = waiting_.front();
    int best = -1;
    size_t best_lcp = 0;
    for (size_t s = 0; s < seqs_.size(); ++s) {
      if (seqs_[s].req) continue;
      const auto& t = seqs_[s].tokens;
      size_t lcp = 0;
      while (lcp < t.size() && lcp < r->prompt.size() && t[lcp] == r->prompt[lcp]) ++lcp;
      if (best < 0 || lcp > best_lcp) best = int(s), best_lcp = lcp;
    }
    if (best < 0) break;
    Seq& q = seqs_[size_t(best)];
    // At least the prompt's last token is fed again, for its logits.
    int64_t keep = int64_t(std::min(best_lcp, r->prompt.size() - 1));
    if (int64_t(q.tokens.size()) - keep > eng_.ring_slack()) keep = std::max<int64_t>(0, keep - eng_.model().window);
    eng_.truncate(best, keep);
    q.tokens.resize(size_t(keep));
    q.fed = size_t(keep);
    q.cached = keep;
    q.next_token = -1;
    q.generated = 0;
    q.held.clear();
    q.called = false;
    q.parser = std::make_unique<text::OutputParser>(tok_, r->in_thinking);
    q.sampler = std::make_unique<Sampler>(r->sampling);
    q.grammar = r->tools.empty() ? nullptr : std::make_unique<ToolGrammar>(vocab_, r->tools);
    q.req = std::move(r);
    waiting_.pop_front();
    --stats_.waiting;
    ++stats_.active;
    stats_.cached_tokens += keep;
  }
  return stats_.active > 0;
}

void Scheduler::run(const std::atomic<bool>& stop) {
  while (!stop.load()) {
    if (!admit()) {
      std::unique_lock lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(100), [&] { return !waiting_.empty() || stop.load(); });
      continue;
    }
    try {
      step();
    } catch (const std::exception& e) {
      fail_all(e.what());
    }
  }
}

void Scheduler::step() {
  for (size_t s = 0; s < seqs_.size(); ++s)
    if (seqs_[s].req && seqs_[s].req->cancelled()) finish(int(s), "stop");
  // Decode tokens first, then prefill chunks in the batch that is left.
  std::vector<engine::Chunk> chunks;
  std::vector<int> owner;
  int64_t budget = c_.max_batch;
  for (size_t s = 0; s < seqs_.size(); ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token < 0) continue;
    chunks.push_back({int(s), int64_t(q.tokens.size()), {q.next_token}, 1});
    owner.push_back(int(s));
    --budget;
  }
  for (size_t s = 0; s < seqs_.size() && budget > 0; ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token >= 0 || q.fed >= q.req->prompt.size()) continue;
    const size_t n = size_t(std::min<int64_t>({c_.max_chunk, budget, int64_t(q.req->prompt.size() - q.fed)}));
    const bool last = q.fed + n == q.req->prompt.size();
    chunks.push_back({int(s), int64_t(q.tokens.size()),
                      std::vector<int32_t>(q.req->prompt.begin() + std::ptrdiff_t(q.fed), q.req->prompt.begin() + std::ptrdiff_t(q.fed + n)),
                      last ? 1 : 0});
    owner.push_back(int(s));
    budget -= int64_t(n);
  }
  if (chunks.empty()) return;
  const std::vector<float> logits = eng_.step(chunks);
  {
    std::lock_guard lk(mu_);
    ++stats_.steps;
  }
  const size_t V = size_t(eng_.model().n_vocab);
  size_t row = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    Seq& q = seqs_[size_t(owner[i])];
    q.tokens.insert(q.tokens.end(), chunks[i].tokens.begin(), chunks[i].tokens.end());
    if (q.next_token >= 0) q.next_token = -1;
    else q.fed += chunks[i].tokens.size();
    if (chunks[i].logits_rows == 0) continue;
    const std::span<const float> l(logits.data() + row * V, V);
    const int32_t t = q.grammar && q.grammar->active() ? q.sampler->sample(l, &q.grammar->allowed()) : q.sampler->sample(l);
    if (q.grammar) q.grammar->accept(t);
    ++row;
    on_token(owner[i], t);
  }
}

void Scheduler::on_token(int slot, int32_t token) {
  Seq& q = seqs_[size_t(slot)];
  q.sampler->accept(token);
  ++q.generated;
  {
    std::lock_guard lk(mu_);
    ++stats_.generated_tokens;
  }
  Event e;
  e.delta = q.parser->push(token);
  if (!e.delta.tool_calls.empty()) q.called = true;
  // Stop strings end the content; hold back a tail that could still become one.
  bool stopped = false;
  if (!q.req->stop.empty() && !e.delta.content.empty()) {
    q.held += e.delta.content;
    size_t cut = std::string::npos, longest = 0;
    for (const auto& s : q.req->stop) {
      longest = std::max(longest, s.size());
      if (const size_t at = q.held.find(s); at != std::string::npos) cut = std::min(cut, at);
    }
    if (cut != std::string::npos) {
      e.delta.content = q.held.substr(0, cut);
      q.held.clear();
      stopped = true;
    } else {
      size_t keep = std::min(q.held.size(), longest - 1);
      size_t emit = q.held.size() - keep;
      while (emit > 0 && emit < q.held.size() && (uint8_t(q.held[emit]) & 0xC0) == 0x80) --emit;  // whole characters
      e.delta.content = q.held.substr(0, emit);
      q.held.erase(0, emit);
    }
  }
  const bool any = !e.delta.reasoning.empty() || !e.delta.content.empty() || !e.delta.tool_calls.empty() || !e.delta.errors.empty();
  const bool ended = e.delta.done || stopped;
  if (any) q.req->push(std::move(e));
  if (ended) return finish(slot, q.called ? "tool_calls" : "stop");
  const int64_t cap = q.req->max_tokens > 0 ? q.req->max_tokens : c_.max_ctx;
  if (q.generated >= cap || int64_t(q.tokens.size()) + 1 >= c_.max_ctx) return finish(slot, "length");
  q.next_token = token;
}

void Scheduler::finish(int slot, const std::string& reason, const std::string& error) {
  Seq& q = seqs_[size_t(slot)];
  if (!q.req) return;
  Event e;
  e.delta = q.parser->finish();
  if (!q.held.empty()) e.delta.content = q.held + e.delta.content;
  e.done = true;
  e.finish_reason = reason;
  e.error = error;
  e.prompt_tokens = int64_t(q.req->prompt.size());
  e.cached_tokens = q.cached;
  e.completion_tokens = q.generated;
  const int64_t prompt = e.prompt_tokens;
  q.req->push(std::move(e));
  q.req.reset();
  q.parser.reset();
  q.sampler.reset();
  q.grammar.reset();
  q.held.clear();
  q.next_token = -1;
  std::lock_guard lk(mu_);
  --stats_.active;
  stats_.prompt_tokens += prompt;
}

void Scheduler::fail_all(const std::string& error) {
  for (size_t s = 0; s < seqs_.size(); ++s) {
    if (seqs_[s].req) finish(int(s), "stop", error);
    // What the engine holds for the slot is unknown after a failed step.
    eng_.release(int(s));
    seqs_[s].tokens.clear();
  }
}

}  // namespace eklow::serve
