#include "serve/scheduler_gemma4.hpp"

#include <algorithm>
#include <cstdio>
#include <future>
#include <optional>
#include <span>
#include <stdexcept>

#include "serve/spec.hpp"

namespace eklow::serve::gemma4 {

static_assert(kPrefixPage == attn::kPage, "the disk cache's pages are the engine's");

// Pinned host buffers of one size, for copies between the card and the disk cache.
class Scheduler::HostPool {
 public:
  HostPool(sycl::queue& q, size_t bytes, size_t count) : q_(q), bytes_(bytes) {
    for (size_t i = 0; i < count; ++i) {
      auto* p = sycl::malloc_host<uint8_t>(bytes, q_);
      if (!p) break;
      all_.push_back(p);
      free_.push_back(p);
    }
    if (all_.empty()) throw std::runtime_error("scheduler: no pinned host memory for the disk cache's buffers");
  }
  ~HostPool() {
    for (uint8_t* p : all_) sycl::free(p, q_);
  }
  // A free buffer: null when none is free and !wait.
  uint8_t* take(bool wait) {
    std::unique_lock lk(mu_);
    if (wait) cv_.wait(lk, [&] { return !free_.empty(); });
    if (free_.empty()) return nullptr;
    uint8_t* p = free_.back();
    free_.pop_back();
    return p;
  }
  void give(uint8_t* p) {
    {
      std::lock_guard lk(mu_);
      free_.push_back(p);
    }
    cv_.notify_one();
  }
  size_t bytes() const { return bytes_; }

 private:
  sycl::queue& q_;
  size_t bytes_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<uint8_t*> all_, free_;
};

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
  stats_.kv_pool_positions = stats_.kv_free_positions = eng.pool_pages() * attn::kPage;
  if (c_.store) {
    reach_pages_ = c_.draft ? (c_.draft->reach() + attn::kPage - 1) / attn::kPage : 0;
    const auto pb = eng_.page_bytes();
    const int64_t w = eng_.window_pages();
    store_pool_ = std::make_unique<HostPool>(eng_.queue(), pb.global, std::max<size_t>(1, c_.store_buffers));
    sliding_pool_ = std::make_unique<HostPool>(eng_.queue(), pb.sliding, size_t(2 * w * eng.n_slots()));
    restore_pool_ = std::make_unique<HostPool>(eng_.queue(), pb.global, 8);
    restore_sliding_pool_ = std::make_unique<HostPool>(eng_.queue(), pb.sliding, size_t(w));
    // A restore point's sliding parts lie reach + window pages back from the sequence's end, which the rings must still hold.
    if (eng_.ring_pages() < reach_pages_ + w + 1)
      std::fprintf(stderr, "eklow-server: prefix cache: the sliding rings hold %lld pages, fewer than the %lld a restore point needs (a smaller --chunk): only global "
                           "parts will be stored, and nothing restored\n",
                   (long long)eng_.ring_pages(), (long long)(reach_pages_ + w + 1));
  }
}

// The store writes from the pools' buffers: it finishes before they go.
Scheduler::~Scheduler() {
  if (c_.store) c_.store->flush();
}

void Scheduler::submit(std::shared_ptr<Request> r) {
  if (r->prompt.empty()) throw std::invalid_argument("empty prompt");
  if (int64_t(r->prompt.size()) >= c_.max_ctx)
    throw std::invalid_argument("the prompt is " + std::to_string(r->prompt.size()) + " tokens, the context " +
                                std::to_string(c_.max_ctx));
  for (const auto& im : r->images)
    if (im.n <= 0 || im.n > c_.max_chunk || im.start < 0 || im.start + im.n > int64_t(r->prompt.size()) || !im.rows)
      throw std::invalid_argument("an image of " + std::to_string(im.n) + " rows does not fit a prefill chunk of " + std::to_string(c_.max_chunk));
  // The keys are hashed here, on the request's thread, rather than on the engine's.
  if (c_.store) extend_chain(c_.store_root, r->page_keys, r->prompt, int64_t(r->prompt.size()) / attn::kPage);
  r->submitted = std::chrono::steady_clock::now();
  std::shared_ptr<Request> dropped;
  {
    std::lock_guard lk(mu_);
    if (r->prefill_only) {
      // The oldest waiting one goes: a newer one of the same conversation holds more of it.
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

// Gives waiting requests free slots; true if any sequence is active afterwards. Requests first, in order; a prefill-only
// request only when no other request runs or waits. A request that finds no free slot takes a prefill-only request's.
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
    bool others = false;  // a request other than a prefill-only one holds a slot
    for (const Seq& q : seqs_) others = others || (q.req && !q.req->prefill_only);
    std::deque<std::shared_ptr<Request>>* from = !waiting_.empty() ? &waiting_ : (!others && !prefill_only_.empty() ? &prefill_only_ : nullptr);
    if (!from) break;
    const std::shared_ptr<Request> r = from->front();
    const int64_t P = int64_t(r->prompt.size());
    int best = -1, oldest = -1;
    size_t best_lcp = 0;
    for (size_t s = 0; s < seqs_.size(); ++s) {
      if (seqs_[s].req) continue;
      const auto& t = seqs_[s].tokens;
      size_t lcp = 0;
      while (lcp < t.size() && lcp < r->prompt.size() && t[lcp] == r->prompt[lcp]) ++lcp;
      if (best < 0 || lcp > best_lcp) best = int(s), best_lcp = lcp;
      if (oldest < 0 || seqs_[s].used < seqs_[size_t(oldest)].used) oldest = int(s);
    }
    if (best < 0) {
      // No free slot: a request takes a prefill-only request's, which ends where it got to (its pages are stored).
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
    Seq& q = seqs_[size_t(best)];
    // At least the prompt's last token is fed again, for its logits. A prefix never ends inside an image (its rows are
    // prefilled together), and the sliding rings must still hold the window before it (ring_safe_keep).
    const int64_t keep = ring_safe_keep(cut_before_image(*r, int64_t(std::min(best_lcp, r->prompt.size() - 1))), int64_t(q.tokens.size()), eng_.ring_slack());
    // A slot whose live sequence parts from the prompt after a long shared stretch is a conversation whose last answer the
    // template rendered back differently from how it was sampled (a token boundary, stripped whitespace): say where (the Qwen
    // scheduler's line, 655209e; image rows, ids below zero, have no bytes).
    if (best_lcp >= 256 && best_lcp < q.tokens.size() && best_lcp < r->prompt.size() && q.tokens[best_lcp] >= 0 && r->prompt[best_lcp] >= 0)
      std::fprintf(stderr, "eklow-server: prefix: the slot's sequence (%zu tokens) and the prompt (%zu) part at token %zu: sampled %s, prompt %s; reusing %lld\n", q.tokens.size(),
                   r->prompt.size(), best_lcp, tok_.bytes(q.tokens[best_lcp]).c_str(), tok_.bytes(r->prompt[best_lcp]).c_str(), (long long)keep);
    // The disk cache: the furthest restore point the prompt's pages reach (never inside an image), against the slot's reuse.
    int slot = best;
    int64_t restore_pages = 0, slot_keep = keep;
    if (c_.store && !r->page_keys.empty() && eng_.ring_pages() >= reach_pages_ + eng_.window_pages() + 1) {
      const RestoreQuery rq{restore_limit(P, attn::kPage, reach_pages_), eng_.window_pages()};
      int64_t t = c_.store->restore_point(r->page_keys, rq);
      while (t > 0 && cut_before_image(*r, t * attn::kPage) != t * attn::kPage) t = c_.store->restore_point(r->page_keys, {t - 1, rq.window_pages});
      if (prefer_restore(keep, t * attn::kPage, c_.restore_cost)) {
        // The restore takes the free slot given a request least recently, rather than the one whose sequence agrees best.
        slot = oldest, restore_pages = t, slot_keep = 0;
      }
    }
    from->pop_front();
    if (from == &waiting_) --stats_.waiting;
    stats_.prefill_only_waiting = int64_t(prefill_only_.size());
    seqs_[size_t(slot)].used = ++admissions_;
    lk.unlock();
    start(slot, r, slot == best ? keep : slot_keep, restore_pages);
    lk.lock();
  }
  return stats_.active > 0;
}

void Scheduler::start(int s, std::shared_ptr<Request> r, int64_t keep, int64_t restore_pages) {
  Seq& q = seqs_[size_t(s)];
  // What the slot's sequence has not yet stored goes before the sequence is cut or replaced.
  store_pages(s, true);
  int64_t restored = 0;
  bool failed = false;
  if (restore_pages > 0) {
    if (restore(s, *r, restore_pages)) restored = restore_pages * attn::kPage;
    else failed = true, keep = 0;
  }
  if (restored > 0) {
    if (c_.draft) c_.draft->reset(s);
    q.tokens.assign(r->prompt.begin(), r->prompt.begin() + std::ptrdiff_t(restored));
    q.chain.assign(r->page_keys.begin(), r->page_keys.begin() + std::ptrdiff_t(restore_pages));
    q.chain_stored = restore_pages;
    keep = restored;
  } else {
    if (failed) eng_.release(s);  // a failed restore leaves the slot's sequence unknown
    eng_.truncate(s, std::min(keep, eng_.length(s)));
    if (c_.draft) {
      if (keep == 0) c_.draft->reset(s);
      else c_.draft->truncate(s, keep);
    }
    q.tokens.resize(size_t(keep));
    q.chain.resize(std::min(q.chain.size(), size_t(keep / attn::kPage)));
    q.chain_stored = std::min<int64_t>(q.chain_stored, int64_t(q.chain.size()));
  }
  q.point_stored = 0;
  q.restored = restored;
  q.fed = size_t(keep);
  q.cached = keep;
  q.admitted = std::chrono::steady_clock::now();
  q.next_token = -1;
  q.generated = q.verifies = q.drafted = q.accepted = 0;
  q.draft_rng.seed(r->sampling.seed * 0x9e3779b97f4a7c15ull + 1);
  q.held.clear();
  q.called = false;
  q.first_pushed = false;
  q.parser = std::make_unique<text::OutputParser>(tok_, r->in_thinking, r->calls_allowed);
  q.sampler = std::make_unique<Sampler>(r->sampling);
  // llama-server's sampler has seen the prompt: its penalties' window starts in it.
  q.sampler->accept(std::span<const int32_t>(r->prompt));
  q.grammar = r->tools.empty() ? nullptr : std::make_unique<ToolGrammar>(vocab_, r->tools);
  const int64_t P = int64_t(r->prompt.size());
  q.req = std::move(r);
  std::lock_guard lk(mu_);
  ++stats_.active;
  stats_.prompt_tokens += P;
  stats_.cached_tokens += keep;
  if (restored > 0) ++stats_.restores, stats_.restored_tokens += restored;
  if (failed) ++stats_.failed_restores;
  note_positions();
}

// Reads `pages` pages of the request's prompt back from the disk into slot `s`: the sliding parts first (a broken one leaves
// the slot as it was), then the global parts in batches read side by side. False if a page could not be read; past the
// sliding parts the slot then holds nothing.
bool Scheduler::restore(int s, const Request& r, int64_t pages) {
  const auto pb = eng_.page_bytes();
  const int64_t from = std::max<int64_t>(0, pages - eng_.window_pages());
  std::vector<uint8_t*> sliding;
  auto give_sliding = [&] {
    for (uint8_t* b : sliding) restore_sliding_pool_->give(b);
  };
  for (int64_t j = from; j < pages; ++j) {
    sliding.push_back(restore_sliding_pool_->take(true));
    if (!c_.store->read(r.page_keys[size_t(j)], Part::Sliding, sliding.back(), pb.sliding)) {
      give_sliding();
      return false;
    }
  }
  eng_.adopt(s, pages);
  constexpr int64_t kBatch = 8;
  bool ok = true;
  for (int64_t j0 = 0; j0 < pages && ok; j0 += kBatch) {
    const int64_t n = std::min(kBatch, pages - j0);
    std::vector<uint8_t*> bufs;
    for (int64_t i = 0; i < n; ++i) bufs.push_back(restore_pool_->take(true));
    std::vector<std::future<bool>> reads;
    for (int64_t i = 0; i < n; ++i)
      reads.push_back(std::async(std::launch::async, [&, i] { return c_.store->read(r.page_keys[size_t(j0 + i)], Part::Global, bufs[size_t(i)], pb.global); }));
    for (auto& f : reads) ok = f.get() && ok;
    if (ok) {
      sycl::event ev;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t j = j0 + i;
        ev = eng_.import_page(s, j, bufs[size_t(i)], j >= from ? sliding[size_t(j - from)] : nullptr);
      }
      ev.wait();
    }
    for (uint8_t* b : bufs) restore_pool_->give(b);
  }
  give_sliding();
  if (!ok) eng_.release(s);
  return ok;
}

int64_t Scheduler::store_pages(int s, bool wait) {
  if (!c_.store) return 0;
  Seq& q = seqs_[size_t(s)];
  const int64_t full = int64_t(q.tokens.size()) / attn::kPage;
  extend_chain(c_.store_root, q.chain, q.tokens, full);
  int64_t queued = 0, deferred = 0;
  while (q.chain_stored < full) {
    const int64_t j = q.chain_stored;
    if (!c_.store->has(q.chain[size_t(j)], Part::Global)) {
      uint8_t* buf = store_pool_->take(wait);
      if (!buf) {  // every buffer is on its way to the disk: the rest wait in the slot for a later call (store_waiting)
        ++deferred;
        break;
      }
      sycl::event ev = eng_.export_page(s, j, buf, nullptr);
      HostPool* pool = store_pool_.get();
      if (c_.store->put(q.chain[size_t(j)], Part::Global, j, buf, pool->bytes(), [ev]() mutable { ev.wait(); }, [pool, buf] { pool->give(buf); })) ++queued;
    }
    ++q.chain_stored;
  }
  if (queued || deferred) {
    std::lock_guard lk(mu_);
    stats_.stored_parts += queued, stats_.deferred_stores += deferred;
  }
  return queued;
}

int64_t Scheduler::store_waiting(bool wait) {
  if (!c_.store) return 0;
  int64_t queued = 0;
  for (size_t s = 0; s < seqs_.size(); ++s)
    if (int64_t(seqs_[s].tokens.size()) / attn::kPage > seqs_[s].chain_stored) queued += store_pages(int(s), wait);
  return queued;
}

bool Scheduler::pages_waiting() const {
  for (const Seq& q : seqs_)
    if (int64_t(q.tokens.size()) / attn::kPage > q.chain_stored) return true;
  return false;
}

void Scheduler::store_point(int s) {
  if (!c_.store) return;
  Seq& q = seqs_[size_t(s)];
  const SlidingPages sp = point_pages(int64_t(q.tokens.size()), attn::kPage, reach_pages_, eng_.window_pages());
  if (sp.to <= 0 || sp.to == q.point_stored) return;
  q.point_stored = sp.to;
  extend_chain(c_.store_root, q.chain, q.tokens, int64_t(q.tokens.size()) / attn::kPage);
  int64_t queued = 0, skipped = 0;
  for (int64_t j = sp.from; j < sp.to; ++j) {
    if (c_.store->has(q.chain[size_t(j)], Part::Sliding)) continue;
    if (!eng_.ring_holds(s, j)) {
      ++skipped;
      continue;
    }
    uint8_t* buf = sliding_pool_->take(true);
    sycl::event ev = eng_.export_page(s, j, nullptr, buf);
    HostPool* pool = sliding_pool_.get();
    if (c_.store->put(q.chain[size_t(j)], Part::Sliding, j, buf, pool->bytes(), [ev]() mutable { ev.wait(); }, [pool, buf] { pool->give(buf); })) ++queued;
  }
  if (queued || skipped) {
    std::lock_guard lk(mu_);
    stats_.stored_parts += queued, stats_.unstored_parts += skipped;
  }
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
  int64_t idle_stored = 0;  // pages that waited for a buffer, queued while idle since the backlog began
  while (!stop.load()) {
    if (!admit()) {
      // An idle slot's pages that waited for a buffer; one line once none is left.
      idle_stored += store_waiting(false);
      if (idle_stored > 0 && !pages_waiting()) {
        std::fprintf(stderr, "eklow-server: prefix cache: %lld pages that waited for a buffer stored while idle\n", (long long)idle_stored);
        idle_stored = 0;
      }
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
    // /metrics' accounting rule, the Qwen scheduler's: this iteration's wall dt is busy time, and counts once for every sequence
    // that contributed tokens to it, to the phase of those tokens (prefill if it was fed prompt tokens, else decode).
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
  // Pages still waiting for a buffer go to the disk before the server stops.
  try {
    if (const int64_t n = store_waiting(true); n > 0)
      std::fprintf(stderr, "eklow-server: prefix cache: %lld pages that waited for a buffer stored at the stop\n", (long long)n);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-server: prefix cache: pages not stored at the stop: %s\n", e.what());
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
  // A sampler whose next distribution cannot be stated (mirostat) decodes plain.
  auto block_of = [&](const Seq& q) {
    return !q.sampler->speculative() ? 0 : q.req->draft_block < 0 ? c_.draft_block : q.req->draft_block;
  };
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
  // limited number of rows: a chunk ends earlier if they do not fit). A prefill-only request gets none while another request
  // runs, and smaller chunks.
  bool others = false;
  for (const Seq& q : seqs_) others = others || (q.req && !q.req->prefill_only);
  for (size_t s = 0; s < seqs_.size() && budget > 0; ++s) {
    Seq& q = seqs_[s];
    if (!q.req || q.next_token >= 0 || q.fed >= q.req->prompt.size()) continue;
    if (q.req->prefill_only && others) continue;
    const int64_t fed = int64_t(q.fed), P = int64_t(q.req->prompt.size());
    int64_t n = std::min<int64_t>({q.req->prefill_only ? c_.prefill_only_chunk : c_.max_chunk, budget, P - fed});
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
    const bool last = fed + n == P && !q.req->prefill_only;  // the step that samples the first token (a prefill-only request samples none)
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
  for (const Item& it : items) ++(it.kind == Item::Prefill ? step_prefill_seqs_ : step_decode_seqs_);  // one chunk a sequence

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
        // The whole prompt is in: its restore point goes to the disk cache now, while the rings hold it.
        if (q.fed == q.req->prompt.size()) store_point(it.slot);
        if (it.rows == 0) {
          if (q.req->prefill_only && q.fed == q.req->prompt.size()) finish(it.slot, "length");
          break;
        }
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
  // Pages the step filled go to the disk cache, and an idle slot's that waited for a buffer (finish has done its slot's).
  store_waiting(false);
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
  // The draft's context is cut to the committed tokens (a block's rejected rows and the token not yet fed are not in it).
  if (c_.draft && error.empty()) c_.draft->truncate(slot, int64_t(q.tokens.size()));
  // The sequence as it ends goes to the disk cache, with a restore point at its end.
  if (error.empty()) {
    store_pages(slot, false);
    store_point(slot);
  }
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
  e.restored_tokens = q.restored;
  e.prefill_only = q.req->prefill_only;
  e.held = int64_t(q.fed);
  if (c_.on_finish) {
    const auto now = std::chrono::steady_clock::now();
    RequestLog l;
    l.tag = q.req->tag, l.reason = q.req->cancelled() ? "cancelled" : reason, l.error = error;
    l.prompt_tokens = e.prompt_tokens, l.cached_tokens = e.cached_tokens, l.completion_tokens = e.completion_tokens;
    l.image_tokens = image_tokens(*q.req);
    l.verifies = q.verifies, l.drafted = q.drafted, l.accepted = q.accepted;
    l.restored_tokens = q.restored, l.prefill_only = q.req->prefill_only, l.held = e.held;
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

// The positions the slots hold, all and those of slots with no request; with mu_ held, on the engine thread.
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
    // What the engine holds for the slot is unknown after a failed step.
    eng_.release(int(s));
    if (c_.draft) c_.draft->reset(int(s));
    seqs_[s].tokens.clear();
    seqs_[s].chain.clear();
    seqs_[s].chain_stored = seqs_[s].point_stored = 0;
  }
}

}  // namespace eklow::serve::gemma4
