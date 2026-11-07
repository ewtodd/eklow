// eklow-keepalive-test: the streams' keep-alive (keepalive.hpp), on the host, no card.
//
//   eklow-keepalive-test
//
// 1. The timer: nothing before the interval, one comment after it, reset by a real write and by its own comment, never at 0, false when
//    the write fails.
// 2. A real HTTP stream on loopback, built the way the servers build theirs (set_chunked_content_provider, events every 100 ms that add
//    no text for 3.5 s, as a tool call does until it closes): with a keep-alive of 1 s the longest gap between bytes the client sees must
//    stay under 1.5 s (without it, 3.5 s), and a parser that keeps only "data: " lines, as son-of-anton's does, must get exactly the
//    payloads sent.
#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "keepalive.hpp"

using eklow::serve::KeepAlive;
using Clock = KeepAlive::Clock;
using namespace std::chrono_literals;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
  std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}

struct Recorder {
  std::string got;
  bool fail = false;
  httplib::DataSink sink;
  Recorder() {
    sink.write = [this](const char* d, size_t n) {
      if (fail) return false;
      got.append(d, n);
      return true;
    };
  }
};

void unit() {
  std::printf("the timer\n");
  const Clock::time_point t0 = Clock::now();
  {
    Recorder r;
    KeepAlive ka(15s, t0);
    check(ka.tick(r.sink, t0 + 14s) && r.got.empty(), "nothing before the interval");
    check(ka.tick(r.sink, t0 + 15s) && r.got == KeepAlive::kLine, "one comment at the interval");
    r.got.clear();
    check(ka.tick(r.sink, t0 + 29s) && r.got.empty(), "its own comment restarts the interval");
    ka.wrote(t0 + 29s);
    check(ka.tick(r.sink, t0 + 43s) && r.got.empty(), "a real write restarts it too");
    check(ka.tick(r.sink, t0 + 44s) && r.got == KeepAlive::kLine, "and it fires again 15 s after that write");
  }
  {
    Recorder r;
    KeepAlive ka(0ms, t0);
    check(ka.tick(r.sink, t0 + 1h) && r.got.empty(), "an interval of 0 never writes");
  }
  {
    Recorder r;
    r.fail = true;
    KeepAlive ka(1s, t0);
    check(ka.tick(r.sink, t0 + 500ms), "a failed sink is not touched before the interval");
    check(!ka.tick(r.sink, t0 + 2s), "a failed write says so (the client is gone)");
  }
  check(std::string(KeepAlive::kLine).rfind(":", 0) == 0 && std::string(KeepAlive::kLine).find("\n\n") != std::string::npos,
        "the line is an SSE comment, ended by a blank line");
}

// One run of the stream: returns the longest gap between the client's received bytes, and the data payloads it parsed.
double stream(std::chrono::milliseconds every, std::vector<std::string>& payloads) {
  httplib::Server srv;
  srv.Get("/stream", [every](const httplib::Request&, httplib::Response& res) {
    auto ka = std::make_shared<KeepAlive>(every);
    auto step = std::make_shared<int>(0);
    res.set_chunked_content_provider("text/event-stream", [ka, step](size_t, httplib::DataSink& sink) {
      // the servers' loop: an event every 100 ms; the first and the last add text, the 35 between add none
      std::this_thread::sleep_for(100ms);
      const int i = (*step)++;
      std::string text;
      if (i == 0) text = "data: {\"delta\":\"first\"}\n\n";
      if (i == 36) text = "data: {\"delta\":\"the whole tool call\"}\n\ndata: [DONE]\n\n";
      if (!text.empty()) {
        if (!sink.write(text.data(), text.size())) return false;
        ka->wrote();
      } else if (!ka->tick(sink)) {
        return false;
      }
      if (i == 36) sink.done();
      return true;
    });
  });
  const int port = srv.bind_to_any_port("127.0.0.1");
  std::thread t([&] { srv.listen_after_bind(); });
  httplib::Client cli("127.0.0.1", port);
  cli.set_read_timeout(30, 0);
  Clock::time_point last = Clock::now();
  double worst = 0;
  std::string body;
  auto res = cli.Get("/stream", [&](const char* d, size_t n) {
    const Clock::time_point now = Clock::now();
    worst = std::max(worst, std::chrono::duration<double>(now - last).count());
    last = now;
    body.append(d, n);
    return true;
  });
  srv.stop();
  t.join();
  if (!res || res->status != 200) return -1;
  // son-of-anton's parser (daemon/internal/provider/provider.go): every line not starting "data: " is skipped
  payloads.clear();
  size_t at = 0;
  while (at < body.size()) {
    size_t end = body.find('\n', at);
    if (end == std::string::npos) end = body.size();
    const std::string line = body.substr(at, end - at);
    if (line.rfind("data: ", 0) == 0) payloads.push_back(line.substr(6));
    at = end + 1;
  }
  return worst;
}

void live() {
  std::printf("a stream on loopback, 3.5 s without text\n");
  std::vector<std::string> with, without;
  const double gap_on = stream(1000ms, with);
  const double gap_off = stream(0ms, without);
  std::printf("  longest gap between bytes: %.2f s with a keep-alive of 1 s, %.2f s without\n", gap_on, gap_off);
  check(gap_on > 0 && gap_on < 1.5, "with it, no gap of 1.5 s or more");
  check(gap_off > 3.0, "without it, the silence the clients time out on");
  const std::vector<std::string> want = {"{\"delta\":\"first\"}", "{\"delta\":\"the whole tool call\"}", "[DONE]"};
  check(with == want, "a data-only parser gets exactly the payloads, with it");
  check(without == want, "and without it");
}

}  // namespace

int main() {
  unit();
  live();
  std::printf("%s\n", g_fail ? "FAILED" : "all passed");
  return g_fail ? 1 : 0;
}
