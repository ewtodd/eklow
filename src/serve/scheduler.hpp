// Continuous batching over the engine's slots. Each step advances every sequence that is generating by one token, or by
// a verify block when a DFlash 2 draft is attached (the last sampled token and B - 1 drafted ones, accepted by
// serve::verify against the target's logits so that the output distribution is what plain decoding gives), and fills the
// rest of the batch with prefill chunks of the sequences still reading their prompts. Generated tokens go through
// Qwen3.6's output parser; the reader gets reasoning, content and tool calls as they complete. Inside a tool call,
// sampling is limited to the tokens the tool-call grammar allows, so a request with tools gets no malformed call.
//
// Prefix reuse over a recurrent state: a slot's K/V can be cut anywhere, its DeltaNet state only where a snapshot was
// taken (engine.hpp). The scheduler takes snapshots where the next request is likely to diverge (the end of a prompt's
// prefill before its generation prompt, every snapshot_every tokens of a long prefill, the end of a generation), keeps
// them in pinned host memory under a budget, and gives a new request the slot and snapshot (or live state) that leave
// the least to prefill.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "engine/engine.hpp"
#include "model/dflash2.hpp"
#include "serve/sampler.hpp"
#include "serve/tool_grammar_qwen.hpp"
#include "text/output_parser_qwen.hpp"
#include "text/tokenizer_qwen.hpp"
#include "text/tool_schema.hpp"

namespace eklow::serve {

// What a request's reader receives as generation goes.
struct Event {
  text::OutputDelta delta;
  bool done = false;
  std::string finish_reason;  // with done: "stop", "length" or "tool_calls"
  std::string error;          // with done: the request failed
  int64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;  // with done
  int64_t verifies = 0, drafted = 0, accepted = 0;  // with done: speculation (verify blocks, tokens drafted, of them accepted)
};

// An image in a prompt: its rows stand for `n` consecutive prompt positions from `start` (a grid of nx columns by ny rows, n = nx * ny).
// The prompt holds, at those positions, ids below zero that name the image and the row (image_row_id), so that prefix reuse tells
// images apart. In sequence terms the image takes n positions (K/V entries, DeltaNet steps); in rotary terms max(nx, ny) (engine.hpp).
struct ImageInput {
  int64_t start = 0, n = 0, nx = 0, ny = 0;
  std::shared_ptr<const std::vector<float>> rows;  // fp32 [n][n_embd]
};
// The prompt id of row `row` of the image whose content hashes to `hash`: negative, never a vocabulary id.
inline int32_t image_row_id(uint64_t hash, int64_t row) {
  uint64_t x = hash ^ (uint64_t(row) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2));
  x ^= x >> 33, x *= 0xff51afd7ed558ccdull, x ^= x >> 33;
  return -1 - int32_t(x & 0x3fffffff);
}

class Request {
 public:
  std::vector<int32_t> prompt;
  std::vector<ImageInput> images;  // in order, each within the prompt
  SamplingParams sampling;
  int64_t max_tokens = 0;          // 0: as far as the context allows
  std::vector<std::string> stop;   // strings that end the content (not included)
  bool in_thinking = false;        // the prompt ends inside the reasoning
  std::shared_ptr<const text::ToolSpecs> tool_specs;  // the request's tools (null: none); calls are constrained to them
  // Positions where a snapshot is taken in prefill (ascending): where the generation prompt starts (the prefix the next turn
  // shares, even when it drops this turn's reasoning), where the last message starts (an edited or retried message), and where
  // the first message after the system one starts (the system prompt and tools a whole family of conversations share).
  std::vector<int64_t> snapshot_points;
  int draft_block = -1;            // -1: the server's block size; 0: plain decode even with a draft; 2 to 16: this one (for measuring)
  std::string tag;                 // names the request in the log (the completion id)

  // Reader side: waits up to `wait` for the next event.
  bool next(Event& e, std::chrono::milliseconds wait);
  void cancel();
  // Scheduler side.
  void push(Event e);
  bool cancelled() const;
  std::chrono::steady_clock::time_point submitted;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Event> events_;
  bool cancelled_ = false;
};

// Sequence index to rotary position for a request with images: what the images before `seq` save (n rows take max(nx, ny)
// positions), and the image that holds `seq` (null when none).
inline int64_t rope_delta(const Request& r, int64_t seq) {
  int64_t d = 0;
  for (const ImageInput& im : r.images)
    if (im.start + im.n <= seq) d += im.n - std::max(im.nx, im.ny);
  return d;
}
inline const ImageInput* image_at(const Request& r, int64_t seq) {
  for (const ImageInput& im : r.images)
    if (seq >= im.start && seq < im.start + im.n) return &im;
  return nullptr;
}

// A request as it ended, for the server's log (unholy's per-request line). Times in seconds, -1 for what did not
// happen: queue from submission to a slot, ttft from submission to the first token, prefill from the slot to the
// first token (the uncached part of the prompt), decode from the first token to the end.
struct RequestLog {
  std::string tag, reason, error;  // reason: stop, length, tool_calls or cancelled
  int64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;
  int64_t image_tokens = 0;  // of the prompt, rows of images
  int64_t verifies = 0, drafted = 0, accepted = 0;  // speculation: verify blocks, tokens drafted, of them accepted
  double queue_s = -1, ttft_s = -1, prefill_s = -1, decode_s = -1;
};

struct SchedulerConfig {
  int64_t max_batch;  // tokens per step
  int64_t max_chunk;  // prefill tokens per sequence per step
  int64_t max_ctx;    // positions per slot
  std::function<void(const RequestLog&)> on_finish = {};  // on the engine thread, as each request ends
  // Speculative decoding: the draft (null: plain decode) and the block size B (anchor + B - 1 drafted tokens, at most 16).
  model::Dflash2* draft = nullptr;
  int draft_block = 8;
  // Snapshots of the recurrent state: the budget in bytes of pinned host memory (0: none, and then prefix reuse is limited
  // to a slot's live sequence), the spacing of those taken during a long prefill, and the shortest prefix worth one.
  uint64_t snapshot_bytes = 0;
  int64_t snapshot_every = 16384;
  int64_t snapshot_min = 256;
  // Layer-major prefill (engine::Config::max_group, at least as large): a step whose only work is one slot's prefill with more than
  // max_chunk tokens left takes up to max_group of them as one group. While another slot generates the group would stall its decode for
  // the whole group, so then it takes up to group_busy tokens (0: plain chunks, decode first).
  int64_t max_group = 0, group_busy = 0;
};

class Scheduler {
 public:
  Scheduler(engine::Engine& eng, const text::QwenTokenizer& tok, const SchedulerConfig& c);

  // Queues a request. Throws std::invalid_argument for one whose prompt does not fit a slot.
  void submit(std::shared_ptr<Request> r);
  // The engine thread: runs until `stop` is set.
  void run(const std::atomic<bool>& stop);

  // Counters since the start, and the state after the last step; the server's periodic log line takes differences.
  struct Stats {
    int64_t active = 0, waiting = 0, steps = 0;
    int64_t prompt_tokens = 0, cached_tokens = 0;      // of admitted prompts, and of those already in their slot
    int64_t prefill_tokens = 0, generated_tokens = 0;  // fed as prefill, sampled
    int64_t positions = 0, idle_positions = 0;         // held by the slots, and by those with no request (cached)
    int64_t verifies = 0, drafted = 0, accepted = 0;   // verify blocks, tokens drafted, of them accepted
    int64_t snapshots = 0, snapshot_bytes = 0;         // held now (bytes: pinned host memory)
    int64_t snapshots_taken = 0, restores = 0;         // since the start
    // Wall seconds of the engine thread in each part of a step, since the start: drafting, the engine step (the logits come
    // back to the host in it), acceptance and sampling, the draft's context injection, snapshots.
    double draft_s = 0, engine_s = 0, sample_s = 0, inject_s = 0, snapshot_s = 0;
  };
  Stats stats() const;

  // While a Hold lives the engine thread starts no new step (the one under way finishes), so other
  // work on the card, the embedding model, has it to itself instead of getting a timeslice now and
  // then behind a long prefill. After each Hold ends the thread takes at least one step, so holds
  // back to back cannot starve chat.
  class Hold {
   public:
    explicit Hold(Scheduler& s);
    ~Hold();
    Hold(const Hold&) = delete;
    Hold& operator=(const Hold&) = delete;

   private:
    Scheduler& s_;
  };

 private:
  // A snapshot of a slot's recurrent state at `position`: valid while the slot's K/V still holds the first `position`
  // positions of the tokens it was taken on.
  struct Snap {
    int64_t position;
    engine::Engine::Snapshot state;
    uint64_t used;  // tick of the last use (or of taking it): the least recent goes first
  };
  struct Seq {
    std::shared_ptr<Request> req;
    std::vector<int32_t> tokens;  // what the engine holds for the slot (committed)
    size_t fed = 0;               // prompt tokens in the engine
    int32_t next_token = -1;      // sampled, fed in the next step
    // >= 0: the engine holds a verify chunk's tokens pending for this slot, of which this many are accepted (what the next
    // chunk's `accept` or a settle() carries).
    int pending_accept = -1;
    std::unique_ptr<text::QwenOutputParser> parser;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<QwenToolGrammar> grammar;
    std::mt19937_64 draft_rng;
    int64_t generated = 0, cached = 0, verifies = 0, drafted = 0, accepted = 0;
    std::chrono::steady_clock::time_point admitted, first;  // the slot taken, the first token sampled
    std::string held;             // content held back while it could start a stop string
    bool called = false;          // a tool call came out
    std::vector<Snap> snaps;      // of the slot's content, by ascending position; outlives the request
    uint64_t used = 0;            // tick of the slot's last use
  };

  bool admit();
  void step();
  // Follows a sampled token; returns why the request ends at it ("" if it goes on).
  std::string on_token(int slot, int32_t token);
  void finish(int slot, const std::string& reason, const std::string& error = "");
  void fail_all(const std::string& error);
  void note_positions();
  // Snapshots.
  bool has_snapshot(const Seq& q, int64_t position) const;
  bool wants_snapshot(const Seq& q, int64_t pos) const;
  void take_snapshot(int slot);
  void drop_snapshots_beyond(Seq& q, int64_t position);
  void enforce_budget();

  engine::Engine& eng_;
  const text::QwenTokenizer& tok_;
  std::shared_ptr<const QwenToolVocab> vocab_;
  SchedulerConfig c_;
  std::vector<Seq> seqs_;
  uint64_t tick_ = 0;
  bool grouped_last_ = false;  // the last step was a layer-major group (a generating slot then gets a step before the next group)
  uint64_t snap_bytes_ = 0;  // held by all slots' snapshots (engine thread; mirrored into stats_)

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> waiting_;
  Stats stats_;

  std::mutex hold_mu_;
  std::condition_variable hold_cv_;
  int holds_ = 0;
  uint64_t releases_ = 0;  // Holds ended so far
};

}  // namespace eklow::serve
