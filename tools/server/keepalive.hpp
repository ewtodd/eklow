// Keeps a streamed response from going silent. The server writes an SSE chunk only when a token adds text, and the clients in front of
// it give up on a stream that sends nothing for a while: son-of-anton after 10 minutes without a byte, maitred after its idle timeout (5
// minutes by default). A tool call goes out whole once it closes, so a long one (a file through write_file) is silent from its first token
// to its last: on 2026-11-07 one ran 17,911 tokens, 10 minutes, and son-of-anton hung up on it. A cold prefill of a long prompt is silent
// too (~6 minutes at 252k). A line starting ":" is an SSE comment: every client skips it (son-of-anton reads only "data: " lines), and any
// byte resets their timers (son-of-anton's idleReader, maitred's relay).
#pragma once

#include <httplib.h>

#include <chrono>

namespace eklow::serve {

class KeepAlive {
 public:
  using Clock = std::chrono::steady_clock;
  // every <= 0: never.
  explicit KeepAlive(std::chrono::milliseconds every, Clock::time_point now = Clock::now()) : every_(every), last_(now) {}
  // Something real went out at `now`.
  void wrote(Clock::time_point now = Clock::now()) { last_ = now; }
  // Writes a comment when nothing has gone out for `every`. False when the write failed: the client is gone.
  bool tick(httplib::DataSink& sink, Clock::time_point now = Clock::now()) {
    if (every_.count() <= 0 || now - last_ < every_) return true;
    if (!sink.write(kLine, sizeof(kLine) - 1)) return false;
    last_ = now;
    return true;
  }
  static constexpr char kLine[] = ": keep-alive\n\n";

 private:
  std::chrono::milliseconds every_;
  Clock::time_point last_;
};

}  // namespace eklow::serve
