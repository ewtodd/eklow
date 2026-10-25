// Continuous batching over the engine's slots. Each step decodes one token for every sequence that
// is generating and fills the rest of the batch with prefill chunks of the sequences still reading
// their prompts. A new request takes the free slot whose cached sequence shares the longest prefix
// with its prompt and prefills only the rest. Generated tokens go through Gemma 4's output parser;
// the reader gets reasoning, content and tool calls as they complete. Inside a tool call, sampling
// is limited to the tokens the tool-call grammar allows, so a request with tools gets no malformed
// call.
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

#include "engine/engine.hpp"
#include "serve/sampler.hpp"
#include "serve/tool_grammar.hpp"
#include "text/output_parser.hpp"
#include "text/tokenizer.hpp"

namespace eklow::serve {

// What a request's reader receives as generation goes.
struct Event {
  text::OutputDelta delta;
  bool done = false;
  std::string finish_reason;  // with done: "stop", "length" or "tool_calls"
  std::string error;          // with done: the request failed
  int64_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0;  // with done
};

class Request {
 public:
  std::vector<int32_t> prompt;
  SamplingParams sampling;
  int64_t max_tokens = 0;          // 0: as far as the context allows
  std::vector<std::string> stop;   // strings that end the content (not included)
  bool in_thinking = false;        // the prompt ends inside the thought channel
  std::vector<std::string> tools;  // the request's tool names; tool calls are constrained to them

  // Reader side: waits up to `wait` for the next event.
  bool next(Event& e, std::chrono::milliseconds wait);
  void cancel();
  // Scheduler side.
  void push(Event e);
  bool cancelled() const;

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
};

class Scheduler {
 public:
  Scheduler(engine::Engine& eng, const text::Tokenizer& tok, const SchedulerConfig& c);

  // Queues a request. Throws std::invalid_argument for one whose prompt does not fit a slot.
  void submit(std::shared_ptr<Request> r);
  // The engine thread: runs until `stop` is set.
  void run(const std::atomic<bool>& stop);

  struct Stats {
    int64_t active = 0, waiting = 0, steps = 0, prompt_tokens = 0, cached_tokens = 0, generated_tokens = 0;
  };
  Stats stats() const;

 private:
  struct Seq {
    std::shared_ptr<Request> req;
    std::vector<int32_t> tokens;  // what the engine holds for the slot
    size_t fed = 0;               // prompt tokens in the engine
    int32_t next_token = -1;      // sampled, fed in the next step
    std::unique_ptr<text::OutputParser> parser;
    std::unique_ptr<Sampler> sampler;
    std::unique_ptr<ToolGrammar> grammar;
    int64_t generated = 0, cached = 0;
    std::string held;             // content held back while it could start a stop string
    bool called = false;          // a tool call came out
  };

  bool admit();
  void step();
  void on_token(int slot, int32_t token);
  void finish(int slot, const std::string& reason, const std::string& error = "");
  void fail_all(const std::string& error);

  engine::Engine& eng_;
  const text::Tokenizer& tok_;
  std::shared_ptr<const ToolVocab> vocab_;
  SchedulerConfig c_;
  std::vector<Seq> seqs_;

  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<Request>> waiting_;
  Stats stats_;
};

}  // namespace eklow::serve
