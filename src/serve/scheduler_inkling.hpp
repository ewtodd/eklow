// Continuous batching of Inkling 2 Mini over the engine's slots (M4, DECISIONS 2026-11-24): each step decodes one token for
// every generating sequence and fills the rest of the batch with prefill chunks of the sequences still reading their
// prompts. Generated tokens go through Inkling's output parser (text/output_parser_inkling); with tools the sampling is
// constrained by the grammar llama-server applies (serve/tool_grammar_inkling).
//
// Prefix reuse: a new request takes the free slot whose sequence shares the longest prefix with its prompt. When the prompt
// continues the slot's sequence the slot goes on from its end; when they part earlier, the slot rewinds to its latest
// checkpoint at or before that point (Engine::rewind: the sliding rings and convolution states cannot be cut back by
// returning pages) or starts over. Every prompt's end is checkpointed, so a next turn that re-renders the last answer
// differently from how it was sampled resumes there.
//
// The request and event types, the request log and the Hold that side models take are the Gemma scheduler's
// (serve/scheduler_gemma4.hpp), so that the server and its response builder serve either model.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/engine_inkling.hpp"
#include "serve/sampler.hpp"
#include "serve/scheduler_gemma4.hpp"
#include "serve/tool_grammar_inkling.hpp"
#include "text/inkling_mode.hpp"
#include "text/output_parser_inkling.hpp"
#include "text/tokenizer_inkling.hpp"

namespace eklow::serve::inkling {

using gemma4::Event;
using gemma4::RequestLog;

class Request {
 public:
  std::vector<int32_t> prompt;
  std::vector<int32_t> prefill;    // the prompt's last tokens that open the assistant turn being continued (the parser's)
  text::InklingMode mode;          // for the output parser
  SamplingParams sampling;
  int64_t max_tokens = 0;          // 0: as far as the context allows
  bool prefill_only = false;       // the prompt into a slot, nothing generated: the lowest priority
  std::vector<std::string> stop;   // strings that end the content (not included)
  // Tools: the grammar (none without tools or with tool_choice none), their names and whether several calls may follow.
  bool grammar = false;
  InklingToolGrammar::Choice choice = InklingToolGrammar::Choice::Auto;
  std::vector<std::string> tools;
  bool parallel = true;
  std::string tag;                 // names the request in the log (the completion id)

  bool next(Event& e, std::chrono::milliseconds wait);
  void cancel();
  void push(Event e);
  bool cancelled() const;
  std::chrono::steady_clock::time_point submitted;

 private:
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Event> events_;
  bool cancelled_ = false;
};

struct SchedulerConfig {
  int64_t max_batch;  // tokens per step
  int64_t max_chunk;  // prefill tokens per sequence per step
  int64_t max_ctx;    // positions per slot
  std::function<void(const RequestLog&)> on_finish = {};
  int64_t prefill_only_chunk = 512;
  size_t max_prefill_only_waiting = 8;
};

class Scheduler {
 public:
  Scheduler(engine::inkling::Engine& eng, const text::InklingTokenizer& tok, const SchedulerConfig& c);
  // Queues a request. Throws std::invalid_argument for one whose prompt does not fit a slot.
  void submit(std::shared_ptr<Request> r);
  // The engine thread: runs until `stop` is set.
  void run(const std::atomic<bool>& stop);
  // The Gemma scheduler's counters (the draft's and the disk cache's stay zero), for the server's log and /metrics.
  using Stats = gemma4::Scheduler::Stats;
  Stats stats() const;

  // While a Hold lives the engine thread starts no new step, so a side model has the card to itself; after each Hold the
  // thread takes at least one step.
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
  struct Seq {
    std::shared_ptr<Request> req;
    std::vector<int32_t> tokens;  // what the engine holds for the slot
    uint64_t used = 0;
    size_t fed = 0;               // prompt tokens in the engine
    int32_t next_token = -1;      // sampled, fed in the next step
    std::unique_ptr<text::InklingOutputParser> parser;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<InklingToolGrammar> grammar;
    int64_t generated = 0, cached = 0;
    std::chrono::steady_clock::time_point admitted, first;
    bool first_pushed = false;
    std::string held;             // content held back while it could start a stop string
    bool called = false;          // a tool call came out
  };

  bool admit();
  void start(int s, std::shared_ptr<Request> r, int64_t lcp);
  void step();
  std::string on_token(int slot, int32_t token);
  void finish(int slot, const std::string& reason, const std::string& error = "");
  void fail_all(const std::string& error);
  void note_positions();

  engine::inkling::Engine& eng_;
  const text::InklingTokenizer& tok_;
  std::shared_ptr<const InklingToolVocab> vocab_;
  SchedulerConfig c_;
  std::vector<Seq> seqs_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> waiting_, prefill_only_;
  Stats stats_;
  uint64_t admissions_ = 0;
  std::mutex hold_mu_;
  std::condition_variable hold_cv_;
  int holds_ = 0;
  uint64_t releases_ = 0;
  int step_prefill_seqs_ = 0, step_decode_seqs_ = 0;
};

}  // namespace eklow::serve::inkling
