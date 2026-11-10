#include "serve/scheduler_gemma4.hpp"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <span>
#include <stdexcept>

#include "serve/spec.hpp"

namespace eklow::serve::gemma4 {

namespace {

double seconds(std::chrono::steady_clock::duration d) { return std::chrono::duration<double>(d).count(); }

// The image rows of a prompt (the number of positions), and where a prefix of `pos` positions cuts one: back to its start.
int64_t image_tokens(const Request& r) {
  int64_t n = 0;
  for (const auto& im : r.images) n += im.n;
  return n;
}
int64_t cut_before_image(const Request& r, int64_t pos) {
  for (const auto& im : r.images)
    if (im.start < pos && pos < im.start + im.n) return im.start;
  return pos;
}

// Adds the wall time of its scope to a counter.
struct Timer {
  Timer(std::mutex& mu, double& total) : mu_(mu), total_(total), t0_(std::chrono::steady_clock::now()) {}
  ~Timer() {
    const double s = seconds(std::chrono::steady_clock::now() - t0_);
    std::lock_guard lk(mu_);
    total_ += s;
  }
  std::mutex& mu_;
  double& total_;
  std::chrono::steady_clock::time_point t0_;
};

// What a drafted row's constraint needs: the output parser and the tool grammar as they stand after the row's prefix.
struct RowState {
  text::OutputParser parser;
  std::unique_ptr<ToolGrammar> grammar;
};

}  // namespace

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

Scheduler::Scheduler(engine::gemma4::Engine& eng, const text::Tokenizer& tok, const SchedulerConfig& c)
    : eng_(eng), tok_(tok), vocab_(std::make_shared<ToolVocab>(tok)), c_(c), seqs_(size_t(eng.n_slots())) {
  if (c_.draft) {
    if (c_.draft_block < 2 || c_.draft_block > engine::gemma4::kMaxVerify)
      throw std::invalid_argument("the draft block must be 2 to " + std::to_string(engine::gemma4::kMaxVerify));
    if (eng.config().capture_layers.empty())
      throw std::invalid_argument("a draft needs an engine that captures the draft's layers");
  }
}

void Scheduler::submit(std::shared_ptr<Request> r) {
  if (r->prompt.empty()) throw std::invalid_argument("empty prompt");
  if (int64_t(r->prompt.size()) >= c_.max_ctx)
    throw std::invalid_argument("the prompt is " + std::to_string(r->prompt.size()) + " tokens, the context " +
                                std::to_string(c_.max_ctx));
  for (const auto& im : r->images)
    if (im.n <= 0 || im.n > c_.max_chunk || im.start < 0 || im.start + im.n > int64_t(r->prompt.size()) || !im.rows)
      throw std::invalid_argument("an image of " + std::to_string(im.n) + " rows does not fit a prefill chunk of " + std::to_string(c_.max_chunk));
  r->submitted = std::chrono::steady_clock::now();
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
    if (c_.on_finish) {
      RequestLog l;
      l.tag = waiting_.front()->tag, l.reason = "cancelled";
      l.prompt_tokens = int64_t(waiting_.front()->prompt.size());
      l.queue_s = seconds(std::chrono::steady_clock::now() - waiting_.front()->submitted);
      c_.on_finish(l);
    }
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
    const size_t r_prompt = r->prompt.size();
    // At least the prompt's last token is fed again, for its logits.
    // A prefix never ends inside an image (its rows are prefilled together), nor, after the rewind, does the window.
    int64_t keep = cut_before_image(*r, int64_t(std::min(best_lcp, r->prompt.size() - 1)));
    if (int64_t(q.tokens.size()) - keep > eng_.ring_slack()) keep = cut_before_image(*r, std::max<int64_t>(0, keep - eng_.model().window));
    // A slot whose live sequence parts from the prompt after a long shared stretch is a conversation whose last answer the
    // template rendered back differently from how it was sampled (a token boundary, stripped whitespace): say where (the Qwen
    // scheduler's line, 655209e; image rows, ids below zero, have no bytes).
    if (best_lcp >= 256 && best_lcp < q.tokens.size() && best_lcp < r->prompt.size() && q.tokens[best_lcp] >= 0 && r->prompt[best_lcp] >= 0)
      std::fprintf(stderr, "eklow-server: prefix: the slot's sequence (%zu tokens) and the prompt (%zu) part at token %zu: sampled %s, prompt %s; reusing %lld\n", q.tokens.size(),
                   r->prompt.size(), best_lcp, tok_.bytes(q.tokens[best_lcp]).c_str(), tok_.bytes(r->prompt[best_lcp]).c_str(), (long long)keep);
    eng_.truncate(best, keep);
    if (c_.draft) {
      if (keep == 0) c_.draft->reset(best);
      else c_.draft->truncate(best, keep);
    }
    q.tokens.resize(size_t(keep));
    q.fed = size_t(keep);
    q.cached = keep;
    q.admitted = std::chrono::steady_clock::now();
    q.next_token = -1;
    q.generated = q.verifies = q.drafted = q.accepted = 0;
    q.draft_rng.seed(r->sampling.seed * 0x9e3779b97f4a7c15ull + 1);
    q.held.clear();
    q.called = false;
    q.parser = std::make_unique<text::OutputParser>(tok_, r->in_thinking, r->calls_allowed);
    q.sampler = std::make_unique<Sampler>(r->sampling);
    q.grammar = r->tools.empty() ? nullptr : std::make_unique<ToolGrammar>(vocab_, r->tools);
    q.req = std::move(r);
    waiting_.pop_front();
    --stats_.waiting;
    ++stats_.active;
    stats_.prompt_tokens += int64_t(r_prompt);
    stats_.cached_tokens += keep;
    note_positions();
  }
  return stats_.active > 0;
}

Scheduler::Hold::Hold(Scheduler& s) : s_(s) {
  std::lock_guard lk(s_.hold_mu_);
  ++s_.holds_;
}

Scheduler::Hold::~Hold() {
  {
    std::lock_guard lk(s_.hold_mu_);
    --s_.holds_;
    ++s_.releases_;
  }
  s_.hold_cv_.notify_all();
}

void Scheduler::run(const std::atomic<bool>& stop) {
  uint64_t seen = 0;  // releases_ at the last step
  while (!stop.load()) {
    if (!admit()) {
      std::unique_lock lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(100), [&] { return !waiting_.empty() || stop.load(); });
      continue;
    }
    {
      std::unique_lock lk(hold_mu_);
      while (holds_ > 0 && releases_ == seen && !stop.load()) hold_cv_.wait_for(lk, std::chrono::milliseconds(100));
      seen = releases_;
    }
    if (stop.load()) break;
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
  const bool spec = c_.draft != nullptr;
  const int64_t W = spec ? c_.draft->reach() : 0;  // the prompt rows the draft's context can see (a full layer reaches past the window)
  int64_t cap_left = spec ? eng_.config().max_capture_rows : 0;  // rows of capture a step may ask for
  // One entry per chunk: what the step did for the slot.
  struct Item {
    enum Kind { Prefill, Decode, Verify } kind;
    int slot;
    int64_t rows;  // logits rows
    DraftBlock block;
  };
  std::vector<engine::gemma4::Chunk> chunks;
  std::vector<Item> items;
  int64_t budget = c_.max_batch;

  // Generating slots first: a verify block (the draft's) or a decode token each.
  std::vector<int> gen;
  for (size_t s = 0; s < seqs_.size(); ++s)
    if (seqs_[s].req && seqs_[s].next_token >= 0) gen.push_back(int(s));
  std::vector<DraftBlock> blocks(seqs_.size());
  // The block of a request: the server's, or its own (a request may ask for another, or 0 for plain decode).
  auto block_of = [&](const Seq& q) { return q.req->draft_block < 0 ? c_.draft_block : q.req->draft_block; };
  std::vector<int> speculate(seqs_.size(), 0);  // per slot: the block size B it verifies this step (0: a decode token)
  if (spec) {
    Timer timer(mu_, stats_.draft_s);
    // Slots with one block size draft in one pass (32 rows at most).
    for (int B = 2; B <= int(engine::gemma4::kMaxVerify); ++B) {
      std::vector<model::Dflash2::Request> reqs;
      std::vector<int> who;
      for (int s : gen) {
        const Seq& q = seqs_[size_t(s)];
        const int64_t pos = int64_t(q.tokens.size());
        // A block needs room in the context, and the draft's context must end where the block starts.
        if (block_of(q) == B && pos + B <= c_.max_ctx && c_.draft->context_end(s) == pos) reqs.push_back({s, q.next_token, pos}), who.push_back(s);
      }
      const size_t per = size_t(std::max(1, 32 / B));
      for (size_t i0 = 0; i0 < reqs.size(); i0 += per) {
        const std::vector<model::Dflash2::Request> sub(reqs.begin() + std::ptrdiff_t(i0), reqs.begin() + std::ptrdiff_t(std::min(reqs.size(), i0 + per)));
        const std::vector<Lattice> lats = c_.draft->lattices(sub, B);
        for (size_t k = 0; k < sub.size(); ++k) {
          Seq& q = seqs_[size_t(who[i0 + k])];
          blocks[size_t(who[i0 + k])] = walk(lats[k], std::max(0.0f, q.req->sampling.temperature), q.draft_rng);
          speculate[size_t(who[i0 + k])] = B;
        }
      }
    }
  }
  for (int s : gen) {
    Seq& q = seqs_[size_t(s)];
    engine::gemma4::Chunk c;
    c.slot = s, c.start_pos = int64_t(q.tokens.size());
    c.tokens.push_back(q.next_token);
    Item it{Item::Decode, s, 1, {}};
    if (const int B = speculate[size_t(s)]) {
      it.kind = Item::Verify, it.block = std::move(blocks[size_t(s)]), it.rows = B;
      c.tokens.insert(c.tokens.end(), it.block.tokens.begin(), it.block.tokens.end());
      c.verify = true, c.logits_rows = B, c.capture_rows = B;
    } else {
      c.logits_rows = 1, c.capture_rows = spec ? 1 : 0;
    }
    budget -= int64_t(c.tokens.size());
    cap_left -= c.capture_rows;
    chunks.push_back(std::move(c));
    items.push_back(std::move(it));
  }

  // Prefill chunks in what is left. With a draft the last `window` rows of the prompt are captured for it (a step captures a
  // limited number of rows: a chunk ends earlier if they do not fit).
  for (size_t s = 0; s < seqs_.size() && budget > 0; ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token >= 0 || q.fed >= q.req->prompt.size()) continue;
    const int64_t fed = int64_t(q.fed), P = int64_t(q.req->prompt.size());
    int64_t n = std::min<int64_t>({c_.max_chunk, budget, P - fed});
    // An image is never split: the chunk ends before one it would cut, and an image that does not fit what is left
    // of the step waits for the next.
    auto fit = [&](int64_t len) -> int64_t {
      const int64_t end = fed + len;
      for (const auto& im : q.req->images)
        if (im.start < end && end < im.start + im.n) return std::max<int64_t>(0, im.start - fed);
      return len;
    };
    n = fit(n);
    int64_t cap = 0;
    if (spec) {
      const int64_t from = std::max(fed, P - W);
      cap = std::max<int64_t>(0, fed + n - from);
      if (cap > cap_left) {
        n = fit(from + cap_left - fed);
        cap = std::max<int64_t>(0, fed + n - from);
      }
    }
    if (n <= 0) continue;
    const bool last = fed + n == P;
    engine::gemma4::Chunk chunk{int(s), int64_t(q.tokens.size()),
                        std::vector<int32_t>(q.req->prompt.begin() + std::ptrdiff_t(q.fed), q.req->prompt.begin() + std::ptrdiff_t(q.fed) + n),
                        last ? 1 : 0};
    chunk.capture_rows = cap;
    for (const auto& im : q.req->images)
      if (im.start >= int64_t(q.fed) && im.start + im.n <= int64_t(q.fed) + n) chunk.images.push_back({im.start - int64_t(q.fed), im.n, im.rows->data()});
    for (int32_t& t : chunk.tokens) t = std::max<int32_t>(t, 0);  // image rows carry ids below zero; the engine ignores them
    chunks.push_back(std::move(chunk));
    items.push_back({Item::Prefill, int(s), last ? 1 : 0, {}});
    budget -= n;
    cap_left -= cap;
  }
  if (chunks.empty()) return;

  const std::vector<float> logits = [&] {
    Timer timer(mu_, stats_.engine_s);
    return eng_.step(chunks);
  }();
  const engine::gemma4::Engine::Capture& capture = eng_.capture();
  {
    std::lock_guard lk(mu_);
    ++stats_.steps;
    for (size_t i = 0; i < items.size(); ++i)
      if (items[i].kind == Item::Prefill) stats_.prefill_tokens += int64_t(chunks[i].tokens.size());
    note_positions();
  }
  const size_t V = size_t(eng_.model().n_vocab);
  size_t row = 0;
  int64_t cap_row = 0;  // the next chunk's first row in the capture
  // Context rows for the draft: `count` rows of a chunk from its first captured row onward (consecutive positions of one slot;
  // an image's rows too: they are real hidden states).
  auto inject = [&](const engine::gemma4::Chunk& c, int64_t first_cap_row, int64_t count) {
    if (!spec || count <= 0) return;
    Timer timer(mu_, stats_.inject_s);
    c_.draft->inject(c.slot, capture.data + first_cap_row * capture.width, int64_t(capture.ids[size_t(first_cap_row)].pos), count);
  };
  auto logits_row = [&](size_t r) { return std::span<const float>(logits.data() + r * V, V); };
  for (size_t i = 0; i < chunks.size(); ++i) {
    const engine::gemma4::Chunk& c = chunks[i];
    const Item& it = items[i];
    Seq& q = seqs_[size_t(it.slot)];
    const size_t row0 = row;
    row += size_t(it.rows);
    const int64_t cap0 = cap_row;
    cap_row += c.capture_rows;
    switch (it.kind) {
      case Item::Prefill: {
        // The prompt's own ids (an image's rows are named by theirs), not the zeros the engine was given.
        q.tokens.insert(q.tokens.end(), q.req->prompt.begin() + std::ptrdiff_t(q.fed), q.req->prompt.begin() + std::ptrdiff_t(q.fed + c.tokens.size()));
        q.fed += c.tokens.size();
        inject(c, cap0, c.capture_rows);
        if (it.rows == 0) break;
        // Special tokens the output parser cannot take where it stands are never sampled (OutputParser::banned).
        const std::vector<int32_t>* banned = &q.parser->banned();
        Timer timer(mu_, stats_.sample_s);
        const int32_t t = q.grammar && q.grammar->active() ? q.sampler->sample(logits_row(row0), &q.grammar->allowed(), banned)
                                                           : q.sampler->sample(logits_row(row0), nullptr, banned);
        const std::string why = on_token(it.slot, t);
        if (!why.empty()) finish(it.slot, why);
        break;
      }
      case Item::Decode: {
        q.tokens.push_back(c.tokens[0]);
        inject(c, cap0, c.capture_rows);
        const std::vector<int32_t>* banned = &q.parser->banned();
        Timer timer(mu_, stats_.sample_s);
        const int32_t t = q.grammar && q.grammar->active() ? q.sampler->sample(logits_row(row0), &q.grammar->allowed(), banned)
                                                           : q.sampler->sample(logits_row(row0), nullptr, banned);
        const std::string why = on_token(it.slot, t);
        if (!why.empty()) finish(it.slot, why);
        break;
      }
      case Item::Verify: {
        const DraftBlock& block = it.block;
        const int n = block.size();
        // Row i of the target's logits follows the anchor and the first i drafted tokens: its parser and grammar are
        // the request's own advanced by them (clones, fed as rows are asked for), so a drafted token the grammar
        // forbids, or a structural token that does not parse where it stands, has probability zero there.
        std::vector<std::unique_ptr<RowState>> states(size_t(n) + 1);
        const ConstraintFn constraint = [&](int r, std::span<const int32_t> prefix) {
          if (r == 0) return RowConstraint{q.grammar && q.grammar->active() ? &q.grammar->allowed() : nullptr, &q.parser->banned()};
          if (!states[size_t(r)]) {
            std::unique_ptr<RowState> st;
            int from = r - 1;  // the row whose state the new one extends (-1: the request's own)
            while (from > 0 && !states[size_t(from)]) --from;
            if (from > 0) {
              const RowState& p = *states[size_t(from)];
              st = std::make_unique<RowState>(RowState{p.parser, p.grammar ? std::make_unique<ToolGrammar>(*p.grammar) : nullptr});
            } else {
              st = std::make_unique<RowState>(RowState{*q.parser, q.grammar ? std::make_unique<ToolGrammar>(*q.grammar) : nullptr});
            }
            for (int k = from; k < r; ++k) {
              st->parser.push(prefix[size_t(k)]);
              if (st->grammar) st->grammar->accept(prefix[size_t(k)]);
            }
            states[size_t(r)] = std::move(st);
          }
          RowState& st = *states[size_t(r)];
          return RowConstraint{st.grammar && st.grammar->active() ? &st.grammar->allowed() : nullptr, &st.parser.banned()};
        };
        std::optional<Timer> timer(std::in_place, mu_, stats_.sample_s);
        const VerifyResult res = verify(*q.sampler, block, [&](int r) { return logits_row(row0 + size_t(r)); }, constraint);
        timer.reset();
        {
          std::lock_guard lk(mu_);
          ++stats_.verifies;
          stats_.drafted += n;
          stats_.accepted += res.accepted;
        }
        ++q.verifies, q.drafted += n, q.accepted += res.accepted;
        // The kept tokens are d_0 .. d_{a-1} and the next one; the j-th of them follows the j-th token of the chunk. A stop
        // string, an end token or max_tokens ends the request at its token: the rest of the block is dropped.
        std::string why;
        int fed = 0;
        for (int j = 0; j <= res.accepted; ++j) {
          q.tokens.push_back(c.tokens[size_t(j)]);
          fed = j + 1;
          why = on_token(it.slot, j < res.accepted ? block.tokens[size_t(j)] : res.next);
          if (!why.empty()) break;
        }
        // The rewind: the engine holds the whole block; keep the anchor and the accepted drafts (the sliding rings restore
        // the window within ring_slack(), which a block fits: Engine::step checked it).
        eng_.truncate(it.slot, int64_t(q.tokens.size()));
        inject(c, cap0, fed);
        if (!why.empty()) finish(it.slot, why);
        break;
      }
    }
  }
}

std::string Scheduler::on_token(int slot, int32_t token) {
  Seq& q = seqs_[size_t(slot)];
  if (q.grammar) q.grammar->accept(token);
  q.sampler->accept(token);
  if (q.generated == 0) q.first = std::chrono::steady_clock::now();
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
  if (ended) return q.called ? "tool_calls" : "stop";
  const int64_t cap = q.req->max_tokens > 0 ? q.req->max_tokens : c_.max_ctx;
  if (q.generated >= cap || int64_t(q.tokens.size()) + 1 >= c_.max_ctx) return "length";
  q.next_token = token;
  return "";
}

void Scheduler::finish(int slot, const std::string& reason, const std::string& error) {
  Seq& q = seqs_[size_t(slot)];
  if (!q.req) return;
  // The draft's context is cut to the committed tokens (a block's rejected rows and the token not yet fed are not in it).
  if (c_.draft && error.empty()) c_.draft->truncate(slot, int64_t(q.tokens.size()));
  Event e;
  e.delta = q.parser->finish();
  if (!q.held.empty()) e.delta.content = q.held + e.delta.content;
  e.done = true;
  e.finish_reason = reason;
  e.error = error;
  e.prompt_tokens = int64_t(q.req->prompt.size());
  e.cached_tokens = q.cached;
  e.completion_tokens = q.generated;
  e.verifies = q.verifies, e.drafted = q.drafted, e.accepted = q.accepted;
  if (c_.on_finish) {
    const auto now = std::chrono::steady_clock::now();
    RequestLog l;
    l.tag = q.req->tag, l.reason = q.req->cancelled() ? "cancelled" : reason, l.error = error;
    l.prompt_tokens = e.prompt_tokens, l.cached_tokens = e.cached_tokens, l.completion_tokens = e.completion_tokens;
    l.image_tokens = image_tokens(*q.req);
    l.verifies = q.verifies, l.drafted = q.drafted, l.accepted = q.accepted;
    l.queue_s = seconds(q.admitted - q.req->submitted);
    if (q.generated > 0) {
      l.ttft_s = seconds(q.first - q.req->submitted);
      l.prefill_s = seconds(q.first - q.admitted);
      l.decode_s = seconds(now - q.first);
    }
    c_.on_finish(l);
  }
  q.req->push(std::move(e));
  q.req.reset();
  q.parser.reset();
  q.sampler.reset();
  q.grammar.reset();
  q.held.clear();
  q.next_token = -1;
  std::lock_guard lk(mu_);
  --stats_.active;
  note_positions();
}

// The positions the slots hold, all and those of slots with no request; with mu_ held, on the engine thread.
void Scheduler::note_positions() {
  stats_.positions = stats_.idle_positions = 0;
  for (size_t s = 0; s < seqs_.size(); ++s) {
    stats_.positions += eng_.length(int(s));
    if (!seqs_[s].req) stats_.idle_positions += eng_.length(int(s));
  }
}

void Scheduler::fail_all(const std::string& error) {
  for (size_t s = 0; s < seqs_.size(); ++s) {
    if (seqs_[s].req) finish(int(s), "stop", error);
    // What the engine holds for the slot is unknown after a failed step.
    eng_.release(int(s));
    if (c_.draft) c_.draft->reset(int(s));
    seqs_[s].tokens.clear();
  }
}

}  // namespace eklow::serve::gemma4
