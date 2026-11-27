#include "serve/scheduler_inkling.hpp"

#include <algorithm>
#include <cstdio>
#include <span>
#include <stdexcept>

namespace eklow::serve::inkling {

namespace {

double seconds(std::chrono::steady_clock::duration d) { return std::chrono::duration<double>(d).count(); }

// Adds the wall time of a scope to a counter (under the scheduler's lock).
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

Scheduler::Scheduler(engine::inkling::Engine& eng, const text::InklingTokenizer& tok, const SchedulerConfig& c)
    : eng_(eng), tok_(tok), vocab_(std::make_shared<InklingToolVocab>(tok)), c_(c), seqs_(size_t(eng.config().n_slots)) {
  stats_.kv_pool_positions = stats_.kv_free_positions = eng.pool_pages() * attn::kPage;
}

void Scheduler::submit(std::shared_ptr<Request> r) {
  if (r->prompt.empty()) throw std::invalid_argument("empty prompt");
  if (int64_t(r->prompt.size()) >= c_.max_ctx)
    throw std::invalid_argument("the prompt is " + std::to_string(r->prompt.size()) + " tokens, the context " + std::to_string(c_.max_ctx));
  r->submitted = std::chrono::steady_clock::now();
  std::shared_ptr<Request> dropped;
  {
    std::lock_guard lk(mu_);
    if (r->prefill_only) {
      if (prefill_only_.size() >= std::max<size_t>(1, c_.max_prefill_only_waiting)) {
        dropped = std::move(prefill_only_.front());
        prefill_only_.pop_front();
      }
      prefill_only_.push_back(std::move(r));
      stats_.prefill_only_waiting = int64_t(prefill_only_.size());
    } else {
      waiting_.push_back(std::move(r));
      ++stats_.waiting;
    }
  }
  if (dropped) {
    Event e;
    e.done = true, e.finish_reason = "length", e.prefill_only = true;
    e.prompt_tokens = int64_t(dropped->prompt.size());
    if (c_.on_finish) {
      RequestLog l;
      l.tag = dropped->tag, l.reason = "dropped", l.prefill_only = true, l.prompt_tokens = e.prompt_tokens;
      l.queue_s = seconds(std::chrono::steady_clock::now() - dropped->submitted);
      c_.on_finish(l);
    }
    dropped->push(std::move(e));
  }
  cv_.notify_all();
}

Scheduler::Stats Scheduler::stats() const {
  std::lock_guard lk(mu_);
  return stats_;
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

// Gives waiting requests free slots; true if any sequence is active afterwards. Requests first, in order; a prefill-only
// request only when no other request runs or waits; a request that finds no free slot takes a prefill-only request's.
bool Scheduler::admit() {
  std::unique_lock lk(mu_);
  for (auto* d : {&waiting_, &prefill_only_})
    while (!d->empty() && d->front()->cancelled()) {
      if (c_.on_finish) {
        RequestLog l;
        l.tag = d->front()->tag, l.reason = "cancelled", l.prefill_only = d->front()->prefill_only;
        l.prompt_tokens = int64_t(d->front()->prompt.size());
        l.queue_s = seconds(std::chrono::steady_clock::now() - d->front()->submitted);
        c_.on_finish(l);
      }
      d->pop_front();
      if (d == &waiting_) --stats_.waiting;
    }
  stats_.prefill_only_waiting = int64_t(prefill_only_.size());
  for (;;) {
    bool others = false;
    for (const Seq& q : seqs_) others = others || (q.req && !q.req->prefill_only);
    std::deque<std::shared_ptr<Request>>* from = !waiting_.empty() ? &waiting_ : (!others && !prefill_only_.empty() ? &prefill_only_ : nullptr);
    if (!from) break;
    const std::shared_ptr<Request> r = from->front();
    int best = -1;
    size_t best_lcp = 0;
    for (size_t s = 0; s < seqs_.size(); ++s) {
      if (seqs_[s].req) continue;
      const auto& t = seqs_[s].tokens;
      size_t lcp = 0;
      while (lcp < t.size() && lcp < r->prompt.size() && t[lcp] == r->prompt[lcp]) ++lcp;
      // the longest shared prefix; among equals the slot used longest ago
      if (best < 0 || lcp > best_lcp || (lcp == best_lcp && seqs_[s].used < seqs_[size_t(best)].used)) best = int(s), best_lcp = lcp;
    }
    if (best < 0) {
      int w = -1;
      for (size_t s = 0; s < seqs_.size() && from == &waiting_; ++s)
        if (seqs_[s].req && seqs_[s].req->prefill_only) w = int(s);
      if (w < 0) break;
      ++stats_.preempted;
      lk.unlock();
      finish(w, "length");
      lk.lock();
      continue;
    }
    from->pop_front();
    if (from == &waiting_) --stats_.waiting;
    stats_.prefill_only_waiting = int64_t(prefill_only_.size());
    seqs_[size_t(best)].used = ++admissions_;
    lk.unlock();
    start(best, r, int64_t(best_lcp));
    lk.lock();
  }
  return stats_.active > 0;
}

void Scheduler::start(int s, std::shared_ptr<Request> r, int64_t lcp) {
  Seq& q = seqs_[size_t(s)];
  const int64_t P = int64_t(r->prompt.size());
  // At least the prompt's last token is fed again, for its logits. The slot goes on from its end when the prompt continues it,
  // else from its latest checkpoint at or before the shared prefix, else from the start.
  int64_t keep = std::min(lcp, P - 1);
  const int64_t len = eng_.length(s);
  auto piece = [&](int32_t id) { return id < 0 ? std::string("<image>") : tok_.bytes(id); };  // image rows hold ids below zero
  if (keep < len) {
    int64_t point = -1;
    for (int64_t c : eng_.checkpoint_positions(s))
      if (c <= keep) point = std::max(point, c);
    if (point >= 0 && eng_.rewind(s, point)) {
      if (keep >= 256 && lcp < int64_t(q.tokens.size()) && lcp < P)
        std::fprintf(stderr, "eklow-server: prefix: the slot's sequence (%zu tokens) and the prompt (%lld) part at token %lld: sampled %s, prompt %s; resuming at %lld\n",
                     q.tokens.size(), (long long)P, (long long)lcp, piece(q.tokens[size_t(lcp)]).c_str(), piece(r->prompt[size_t(lcp)]).c_str(),
                     (long long)point);
      keep = point;
    } else {
      eng_.release(s);
      keep = 0;
    }
  }
  if (c_.draft) {
    if (keep == 0) c_.draft->reset(s);
    else if (c_.draft->context_end(s) > keep) c_.draft->truncate(s, keep);
  }
  q.tokens.resize(size_t(keep));
  q.fed = size_t(keep);
  q.cached = keep;
  q.admitted = std::chrono::steady_clock::now();
  q.next_token = -1;
  q.generated = 0;
  q.held.clear();
  q.called = false;
  q.first_pushed = false;
  q.parser = std::make_unique<text::InklingOutputParser>(tok_, r->mode, std::span<const int32_t>(r->prefill));
  q.sampler = std::make_unique<Sampler>(r->sampling);
  q.draft_rng.seed(r->sampling.seed * 0x9e3779b97f4a7c15ull + 1);
  q.verifies = q.drafted = q.accepted = 0;
  q.sampler->accept(std::span<const int32_t>(r->prompt));  // llama-server's sampler has seen the prompt
  q.grammar = r->grammar ? std::make_unique<InklingToolGrammar>(vocab_, r->choice, r->tools, r->parallel) : nullptr;
  q.req = std::move(r);
  std::lock_guard lk(mu_);
  ++stats_.active;
  stats_.prompt_tokens += P;
  stats_.cached_tokens += keep;
  note_positions();
}

bool Scheduler::release_idle() {
  int pick = -1;
  for (size_t s = 0; s < seqs_.size(); ++s)
    if (!seqs_[s].req && eng_.length(int(s)) > 0 && (pick < 0 || seqs_[s].used < seqs_[size_t(pick)].used)) pick = int(s);
  if (pick < 0) return false;
  eng_.release(pick);
  seqs_[size_t(pick)].tokens.clear();
  std::lock_guard lk(mu_);
  note_positions();
  return true;
}

void Scheduler::requeue(int s) {
  Seq& q = seqs_[size_t(s)];
  std::shared_ptr<Request> r = std::move(q.req);
  q.parser.reset();
  q.sampler.reset();
  q.grammar.reset();
  q.held.clear();
  q.next_token = -1;
  eng_.release(s);
  q.tokens.clear();
  std::lock_guard lk(mu_);
  --stats_.active;
  if (r->prefill_only) {
    prefill_only_.push_front(std::move(r));
  } else {
    waiting_.push_front(std::move(r));
    ++stats_.waiting;
  }
  note_positions();
}

void Scheduler::run(const std::atomic<bool>& stop) {
  uint64_t seen = 0;
  while (!stop.load()) {
    if (!admit()) {
      std::unique_lock lk(mu_);
      cv_.wait_for(lk, std::chrono::milliseconds(100), [&] { return !waiting_.empty() || !prefill_only_.empty() || stop.load(); });
      continue;
    }
    {
      std::unique_lock lk(hold_mu_);
      while (holds_ > 0 && releases_ == seen && !stop.load()) hold_cv_.wait_for(lk, std::chrono::milliseconds(100));
      seen = releases_;
    }
    if (stop.load()) break;
    const auto t0 = std::chrono::steady_clock::now();
    step_prefill_seqs_ = step_decode_seqs_ = 0;
    try {
      step();
    } catch (const std::exception& e) {
      fail_all(e.what());
    }
    const double dt = seconds(std::chrono::steady_clock::now() - t0);
    std::lock_guard lk(mu_);
    stats_.busy_s += dt;
    stats_.prefill_seq_s += dt * step_prefill_seqs_;
    stats_.decode_seq_s += dt * step_decode_seqs_;
  }
  // The server is stopping: what still runs or waits ends here with an error, so that no request thread waits on an engine
  // that has stopped (the HTTP server's stop joins them).
  for (size_t s = 0; s < seqs_.size(); ++s)
    if (seqs_[s].req) finish(int(s), "stop", "the server is stopping");
  std::deque<std::shared_ptr<Request>> left;
  {
    std::lock_guard lk(mu_);
    for (auto* d : {&waiting_, &prefill_only_}) {
      left.insert(left.end(), d->begin(), d->end());
      d->clear();
    }
    stats_.waiting = stats_.prefill_only_waiting = 0;
  }
  for (const auto& r : left) {
    Event e;
    e.done = true, e.finish_reason = "stop", e.error = "the server is stopping", e.prefill_only = r->prefill_only;
    e.prompt_tokens = int64_t(r->prompt.size());
    r->push(std::move(e));
  }
}

void Scheduler::step() {
  for (size_t s = 0; s < seqs_.size(); ++s)
    if (seqs_[s].req && seqs_[s].req->cancelled()) finish(int(s), "stop");
  struct Item {
    bool prefill;
    int slot;
    int64_t rows;
    bool verify = false;      // a speculative block: rows = its tokens
    serve::DraftBlock block;  // its drafted tokens
  };
  std::vector<engine::inkling::Chunk> chunks;
  std::vector<Item> items;
  int64_t budget = c_.max_batch;
  // The K/V pool the slots share (engine Config::pool_positions): this step's tokens take free pages, then idle slots'
  // cached sequences (the least recently used first). A generating sequence that still finds no page makes room by
  // restarting a request that is still reading its prompt (it has sent nothing yet), or else by ending the most recently
  // admitted other request; a prefill takes what fits and waits for the rest. One request alone always fits (the pool
  // holds max_ctx).
  std::vector<int64_t> counted(seqs_.size(), 0);  // pages this step takes for each slot
  auto avail = [&] {
    int64_t a = eng_.free_pages();
    for (int64_t c : counted) a -= c;
    return a;
  };
  auto take = [&](int s, int64_t positions) {
    const int64_t more = eng_.pages_needed(s, positions) - counted[size_t(s)];
    if (more <= 0) return true;
    while (more > avail() && release_idle()) {
    }
    if (more > avail()) return false;
    counted[size_t(s)] += more;
    return true;
  };
  for (size_t s = 0; s < seqs_.size(); ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token < 0 || take(int(s), int64_t(q.tokens.size()) + 1)) continue;
    int victim = -1;
    for (size_t t = 0; t < seqs_.size(); ++t)
      if (t != s && seqs_[t].req && seqs_[t].next_token < 0 && seqs_[t].generated == 0) victim = int(t);
    if (victim >= 0) {
      requeue(victim);
    } else {
      for (size_t t = 0; t < seqs_.size(); ++t)
        if (t != s && seqs_[t].req && (victim < 0 || seqs_[t].used > seqs_[size_t(victim)].used)) victim = int(t);
      if (victim >= 0) {
        finish(victim, "length", "the K/V pool is full: another conversation needed its room");
        eng_.release(victim);
        seqs_[size_t(victim)].tokens.clear();
      }
    }
    if (victim >= 0) counted[size_t(victim)] = 0;
    if (!take(int(s), int64_t(q.tokens.size()) + 1)) finish(int(s), "length", "the K/V pool is full");
  }
  // Generating slots first: a verify block of the draft's (in a step that reads no prompt: the engine keeps the
  // convolutions' states after each row of a step of up to 16) or a decode token each. With a draft every committed
  // token's features go into its context: a decode token's row is captured too.
  bool reading = false;
  for (const Seq& q : seqs_) reading = reading || (q.req && q.next_token < 0);
  std::vector<serve::DraftBlock> blocks(seqs_.size());
  std::vector<int> speculate(seqs_.size(), 0);  // per slot: the block it verifies this step (0: a decode token)
  int64_t cap_left = c_.draft ? eng_.config().max_capture_rows : 0;
  if (c_.draft && !reading) {
    Timer timer(mu_, stats_.draft_s);
    auto block_of = [&](const Seq& q) {
      if (!q.sampler->speculative()) return 0;  // a sampler whose next distribution cannot be stated decodes plain
      const int b = q.req->draft_block < 0 ? c_.draft_block : q.req->draft_block;
      return std::clamp(b, 0, 8);
    };
    int rows_left = int(engine::inkling::Engine::kVerifyRows);
    for (int B = 2; B <= 8; ++B) {
      std::vector<model::Dflash2::Request> reqs;
      std::vector<int> who;
      for (size_t s = 0; s < seqs_.size(); ++s) {
        Seq& q = seqs_[s];
        if (!q.req || q.next_token < 0 || block_of(q) != B) continue;
        const int64_t pos = int64_t(q.tokens.size());
        if (pos + B > c_.max_ctx || c_.draft->context_end(int(s)) != pos || B > rows_left || !take(int(s), pos + B)) continue;
        reqs.push_back({int(s), q.next_token, pos}), who.push_back(int(s));
        rows_left -= B;
      }
      if (reqs.empty()) continue;
      const std::vector<serve::Lattice> lats = c_.draft->lattices(reqs, B);
      for (size_t k = 0; k < reqs.size(); ++k) {
        Seq& q = seqs_[size_t(who[k])];
        blocks[size_t(who[k])] = serve::walk(lats[k], std::max(0.0f, q.req->sampling.temperature), q.draft_rng);
        speculate[size_t(who[k])] = B;
      }
    }
  }
  for (size_t s = 0; s < seqs_.size(); ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token < 0) continue;
    engine::inkling::Chunk c{int(s), int64_t(q.tokens.size()), {q.next_token}, 1};
    Item it{false, int(s), 1};
    if (const int B = speculate[s]) {
      it.verify = true, it.block = std::move(blocks[s]), it.rows = B;
      c.tokens.insert(c.tokens.end(), it.block.tokens.begin(), it.block.tokens.end());
      c.verify = true, c.logits_rows = B, c.capture_rows = B;
    } else {
      c.capture_rows = c_.draft ? 1 : 0;
    }
    budget -= int64_t(c.tokens.size());
    cap_left -= c.capture_rows;
    chunks.push_back(std::move(c));
    items.push_back(std::move(it));
  }
  // Prefill chunks in what is left; a prefill-only request gets none while another request runs, and smaller chunks.
  bool others = false;
  for (const Seq& q : seqs_) others = others || (q.req && !q.req->prefill_only);
  // A slot may take several consecutive chunks in a step (the engine runs a step of more than a sub-batch layer by layer,
  // so a long prompt's cold experts are copied onto the card once a step).
  for (size_t s = 0; s < seqs_.size() && budget > 0; ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token >= 0 || q.fed >= q.req->prompt.size()) continue;
    if (q.req->prefill_only && others) continue;
    const int64_t P = int64_t(q.req->prompt.size());
    // Only when the rest of the prompt fits beside the generating sequences, with 4 pages (1,024 positions) each for them
    // to grow: a prefill that half fits would be restarted whenever they need a page.
    int64_t reserve = 0;
    for (size_t t = 0; t < seqs_.size(); ++t)
      if (t != s && seqs_[t].req && seqs_[t].next_token >= 0) reserve += 4;
    while (eng_.pages_needed(int(s), P) - counted[s] > avail() - reserve && release_idle()) {
    }
    if (eng_.pages_needed(int(s), P) - counted[s] > avail() - reserve) continue;
    for (int64_t fed = int64_t(q.fed), at = int64_t(q.tokens.size()); budget > 0 && fed < P;) {
      int64_t n = std::min<int64_t>({q.req->prefill_only ? c_.prefill_only_chunk : c_.max_chunk, budget, P - fed});
      // The prompt's last token goes in a step of its own, after a checkpoint before it: a request whose prompt is this
      // one again (a retry) rewinds there and feeds that token for its logits. It costs a decode-sized step per request.
      if (!q.req->prefill_only && fed + n == P && n > 1) --n;
      // no more than the pool has room for (idle slots' caches given up first); the rest waits
      while ((eng_.pages_held(int(s)) + counted[s] + avail()) * attn::kPage - at < n && release_idle()) {
      }
      n = std::min<int64_t>(n, (eng_.pages_held(int(s)) + counted[s] + avail()) * attn::kPage - at);
      // with a draft, the prompt's rows within its reach of the end are captured for its context: a chunk ends where the
      // step's capture is full (the rest waits for the next step)
      int64_t cap = 0;
      if (c_.draft) {
        const int64_t from = std::max(fed, P - c_.draft->reach());
        cap = std::max<int64_t>(0, fed + n - from);
        if (cap > cap_left) {
          n = std::max<int64_t>(0, from + cap_left - fed);
          cap = std::max<int64_t>(0, fed + n - from);
        }
      }
      if (n <= 0) break;
      take(int(s), at + n);
      const bool last = fed + n == P && !q.req->prefill_only;
      chunks.push_back({int(s), at,
                        std::vector<int32_t>(q.req->prompt.begin() + std::ptrdiff_t(fed), q.req->prompt.begin() + std::ptrdiff_t(fed + n)),
                        last ? 1 : 0});
      chunks.back().capture_rows = cap;
      cap_left -= cap;
      // the images' rows in the chunk: an image decodes causally, as text, so a chunk may hold part of one (a long image,
      // a step's end, a rewind to a checkpoint inside one)
      for (const auto& im : q.req->images) {
        const int64_t lo = std::max(im.start, fed), hi = std::min(im.start + im.n, fed + n);
        if (lo < hi) chunks.back().images.push_back({lo - fed, hi - lo, im.rows->data() + (lo - im.start) * eng_.model().n_embd});
      }
      items.push_back({true, int(s), last ? 1 : 0});
      budget -= n, fed += n, at += n;
      // the checkpoint goes where the step ends: none of the slot's chunks after the one that ends before the last token
      if (last || (!q.req->prefill_only && fed + 1 == P)) break;
    }
  }
  if (chunks.empty()) return;
  for (const Item& it : items) ++(it.prefill ? step_prefill_seqs_ : step_decode_seqs_);

  const std::vector<float> logits = [&] {
    Timer timer(mu_, stats_.engine_s);
    return eng_.step(chunks);
  }();
  {
    std::lock_guard lk(mu_);
    ++stats_.steps;
    for (size_t i = 0; i < items.size(); ++i)
      if (items[i].prefill) stats_.prefill_tokens += int64_t(chunks[i].tokens.size());
    note_positions();
  }
  const size_t V = size_t(eng_.model().n_vocab_out);
  auto sample = [&](Seq& q, size_t row) {
    Timer timer(mu_, stats_.sample_s);
    const std::span<const float> l(logits.data() + row * V, V);
    if (!q.grammar) return q.sampler->sample(l, nullptr, nullptr);
    // Unconstrained (auto mode before a call), the grammar still never takes the tokens it can never accept.
    if (q.grammar->unconstrained()) return q.sampler->sample(l, nullptr, &vocab_->never);
    return q.sampler->sample(l, &q.grammar->allowed(), nullptr);
  };
  // The draft's context: a chunk's captured rows (consecutive positions of one slot) from its first in the capture.
  const auto& capture = eng_.capture();
  int64_t cap_row = 0;
  auto inject = [&](int64_t first, int64_t count) {
    if (!c_.draft || count <= 0) return;
    Timer timer(mu_, stats_.inject_s);
    c_.draft->inject(capture.ids[size_t(first)].slot, capture.data + first * capture.width, capture.ids[size_t(first)].pos, count);
  };
  size_t row = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const engine::inkling::Chunk& c = chunks[i];
    const Item& it = items[i];
    Seq& q = seqs_[size_t(it.slot)];
    const size_t row0 = row;
    row += size_t(it.rows);
    const int64_t cap0 = cap_row;
    cap_row += c.capture_rows;
    if (it.verify) {
      // Row r of the logits follows the anchor and the first r drafted tokens: the tool grammar there is the request's
      // advanced by them (clones, made as rows are asked for), so a drafted token it forbids has probability zero.
      const serve::DraftBlock& block = it.block;
      const int n = block.size();
      std::vector<std::unique_ptr<InklingToolGrammar>> states(size_t(n) + 1);
      const serve::ConstraintFn constraint = [&](int r, std::span<const int32_t> prefix) -> serve::RowConstraint {
        if (!q.grammar) return {};
        InklingToolGrammar* g = q.grammar.get();
        if (r > 0) {
          if (!states[size_t(r)]) {
            int from = r - 1;  // the row whose state the new one extends (0: the request's own)
            while (from > 0 && !states[size_t(from)]) --from;
            auto st = std::make_unique<InklingToolGrammar>(from > 0 ? *states[size_t(from)] : *q.grammar);
            for (int k = from; k < r; ++k) st->accept(prefix[size_t(k)]);
            states[size_t(r)] = std::move(st);
          }
          g = states[size_t(r)].get();
        }
        if (g->unconstrained()) return {nullptr, &vocab_->never};
        return {&g->allowed(), nullptr};
      };
      serve::VerifyResult res;
      {
        Timer timer(mu_, stats_.sample_s);
        res = serve::verify(*q.sampler, block, [&](int r) { return std::span<const float>(logits.data() + (row0 + size_t(r)) * V, V); }, constraint);
      }
      {
        std::lock_guard lk(mu_);
        ++stats_.verifies, stats_.drafted += n, stats_.accepted += res.accepted;
      }
      ++q.verifies, q.drafted += n, q.accepted += res.accepted;
      // The kept tokens are the anchor and d_0 .. d_{a-1}; what comes out is d_0 .. d_{a-1} and the next one (a stop ends
      // the request at its token, the rest of the block dropped). The engine goes back to the kept tokens.
      std::string why;
      int fed = 0;
      for (int j = 0; j <= res.accepted; ++j) {
        q.tokens.push_back(c.tokens[size_t(j)]);
        fed = j + 1;
        why = on_token(it.slot, j < res.accepted ? block.tokens[size_t(j)] : res.next);
        if (!why.empty()) break;
      }
      eng_.truncate(it.slot, int64_t(q.tokens.size()));
      inject(cap0, fed);
      if (!why.empty()) finish(it.slot, why);
      continue;
    }
    if (it.prefill) {
      q.tokens.insert(q.tokens.end(), c.tokens.begin(), c.tokens.end());
      q.fed += c.tokens.size();
      inject(cap0, c.capture_rows);
      // A checkpoint before the prompt's last token (at its end for a prefill-only request): where a next turn that departs
      // from this answer resumes, or the same prompt again.
      if (q.fed + (q.req->prefill_only ? 0 : 1) == q.req->prompt.size()) eng_.checkpoint(it.slot);
      if (it.rows == 0) {
        if (q.req->prefill_only && q.fed == q.req->prompt.size()) finish(it.slot, "length");
        continue;
      }
    } else {
      q.tokens.push_back(c.tokens[0]);
      inject(cap0, c.capture_rows);
    }
    const std::string why = on_token(it.slot, sample(q, row0));
    if (!why.empty()) finish(it.slot, why);
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
  // Stop strings end the content; a tail that could still become one is held back.
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
      while (emit > 0 && emit < q.held.size() && (uint8_t(q.held[emit]) & 0xC0) == 0x80) --emit;
      e.delta.content = q.held.substr(0, emit);
      q.held.erase(0, emit);
    }
  }
  const bool any = !e.delta.reasoning.empty() || !e.delta.content.empty() || !e.delta.tool_calls.empty() || !e.delta.errors.empty();
  const bool ended = e.delta.done || stopped;
  if (any) {
    q.req->push(std::move(e));
    if (!q.first_pushed) {
      q.first_pushed = true;
      const double t = seconds(std::chrono::steady_clock::now() - q.req->submitted);
      std::lock_guard lk(mu_);
      stats_.first_token_s += t, ++stats_.first_tokens;
    }
  }
  if (ended) return q.called ? "tool_calls" : "stop";
  const int64_t cap = q.req->max_tokens > 0 ? q.req->max_tokens : c_.max_ctx;
  if (q.generated >= cap || int64_t(q.tokens.size()) + 1 >= c_.max_ctx) return "length";
  q.next_token = token;
  return "";
}

void Scheduler::finish(int slot, const std::string& reason, const std::string& error) {
  Seq& q = seqs_[size_t(slot)];
  if (!q.req) return;
  Event e;
  e.delta = q.parser->finish();
  if (!q.held.empty()) e.delta.content = q.held + e.delta.content;
  if (!e.delta.tool_calls.empty()) q.called = true;
  e.done = true;
  e.finish_reason = reason == "stop" && q.called ? "tool_calls" : reason;
  e.error = error;
  e.prompt_tokens = int64_t(q.req->prompt.size());
  e.cached_tokens = q.cached;
  e.completion_tokens = q.generated;
  e.prefill_only = q.req->prefill_only;
  e.held = int64_t(q.fed);
  if (c_.on_finish) {
    const auto now = std::chrono::steady_clock::now();
    RequestLog l;
    l.tag = q.req->tag, l.reason = q.req->cancelled() ? "cancelled" : e.finish_reason, l.error = error;
    l.prompt_tokens = e.prompt_tokens, l.cached_tokens = e.cached_tokens, l.completion_tokens = e.completion_tokens;
    for (const auto& im : q.req->images) l.image_tokens += im.n;
    l.prefill_only = q.req->prefill_only, l.held = e.held;
    l.verifies = q.verifies, l.drafted = q.drafted, l.accepted = q.accepted;
    if (q.req->prefill_only) l.prefill_s = seconds(now - q.admitted);
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

void Scheduler::note_positions() {
  stats_.kv_pool_positions = eng_.pool_pages() * attn::kPage;
  stats_.kv_free_positions = eng_.free_pages() * attn::kPage;
  stats_.kv_held_positions = stats_.kv_pool_positions - stats_.kv_free_positions;
  stats_.positions = stats_.idle_positions = 0;
  for (size_t s = 0; s < seqs_.size(); ++s) {
    stats_.positions += eng_.length(int(s));
    if (!seqs_[s].req) stats_.idle_positions += eng_.length(int(s));
  }
}

void Scheduler::fail_all(const std::string& error) {
  for (size_t s = 0; s < seqs_.size(); ++s) {
    if (seqs_[s].req) finish(int(s), "stop", error);
    eng_.release(int(s));  // what the engine holds for the slot is unknown after a failed step
    seqs_[s].tokens.clear();
  }
}

}  // namespace eklow::serve::inkling
