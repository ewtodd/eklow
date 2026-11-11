// eklow-metrics-test: render_metrics (metrics.hpp) on the host, no card: every name once, numeric values, label escaping, the K/V ratio,
// for the Qwen scheduler's Stats and Gemma 4's.
#include <cstdio>
#include <map>
#include <sstream>

#include "metrics.hpp"
#include "serve/scheduler.hpp"
#include "serve/scheduler_gemma4.hpp"

using namespace eklow::serve;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
  std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}
// name (with labels) -> value text, and how many times each sample appeared.
struct Parsed {
  std::map<std::string, std::string> v;
  std::map<std::string, int> n;
  bool numeric = true;
};
Parsed parse(const std::string& text) {
  Parsed p;
  std::istringstream in(text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    const size_t sp = line.rfind(' ');
    const std::string name = line.substr(0, sp), val = line.substr(sp + 1);
    char* end = nullptr;
    std::strtod(val.c_str(), &end);
    if (val.empty() || *end) p.numeric = false;
    p.v[name] = val, ++p.n[name];
  }
  return p;
}
}  // namespace

int main() {
  Scheduler::Stats s;
  s.active = 2, s.waiting = 3, s.steps = 10, s.prompt_tokens = 100, s.cached_tokens = 40, s.prefill_tokens = 60, s.generated_tokens = 25;
  s.kv_pool_positions = 1000, s.kv_held_positions = 600, s.idle_positions = 200;
  s.verifies = 5, s.drafted = 35, s.accepted = 20, s.snapshots = 2, s.snapshot_bytes = 1 << 20;
  s.busy_s = 1.5, s.prefill_seq_s = 0.25, s.decode_seq_s = 2.5, s.first_token_s = 0.75, s.first_tokens = 4;
  const MetricsEmbed e{7, 9, 1234};
  const MetricsHttp h{12, 3};
  const std::string text = render_metrics(s, e, h, {"qwen \"x\"\\y\nz", 2, 262144});
  const Parsed p = parse(text);

  std::printf("names\n");
  const char* names[] = {"slots", "context_tokens", "running", "waiting", "kv_usage_ratio", "steps_total", "busy_seconds_total", "prompt_tokens_total",
                         "cached_prompt_tokens_total", "prefill_tokens_total", "decode_tokens_total", "prefill_sequence_seconds_total",
                         "decode_sequence_seconds_total", "first_token_seconds_total", "first_tokens_total", "spec_verify_steps_total",
                         "spec_drafted_total", "spec_accepted_total", "http_requests_total", "http_errors_total", "embedding_requests_total",
                         "embedding_inputs_total", "embedding_tokens_total", "snapshots", "snapshot_bytes"};
  bool once = true;
  for (const char* n : names) {
    auto it = p.n.find(std::string("eklow_") + n);
    if (it == p.n.end() || it->second != 1) {
      std::printf("    bad: %s\n", n);
      once = false;
    }
  }
  check(once, "every contract name appears exactly once");
  check(p.numeric, "every value parses as a number");
  check(p.v.at("eklow_slots") == "2" && p.v.at("eklow_context_tokens") == "262144", "slots and context");
  check(p.v.at("eklow_running") == "2" && p.v.at("eklow_waiting") == "3", "running and waiting");
  check(p.v.at("eklow_decode_tokens_total") == "25" && p.v.at("eklow_prefill_tokens_total") == "60", "token counters");
  check(p.v.at("eklow_busy_seconds_total") == "1.5" && p.v.at("eklow_decode_sequence_seconds_total") == "2.5", "second counters");
  check(p.v.at("eklow_first_tokens_total") == "4" && p.v.at("eklow_http_errors_total") == "3", "first tokens and http");
  check(p.v.at("eklow_embedding_tokens_total") == "1234" && p.v.at("eklow_snapshot_bytes") == "1048576", "embedding and snapshots");
  check(p.v.at("eklow_kv_usage_ratio") == "0.4", "kv ratio is (held - idle) / pool");

  std::printf("label escaping\n");
  check(text.find("eklow_info{model=\"qwen \\\"x\\\"\\\\y\\nz\"} 1\n") != std::string::npos, "backslash, quote and newline escaped in the model label");
  check(p.n.size() == 26, "info plus 25 samples, no stray lines");

  std::printf("kv ratio\n");
  Scheduler::Stats z;
  check(metrics_kv_ratio(z) == 0.0, "a zero-capacity pool gives 0");
  z.kv_held_positions = 50;
  check(metrics_kv_ratio(z) == 0.0, "zero capacity with pages held still gives 0");
  z.kv_pool_positions = 100, z.kv_held_positions = 150;
  check(metrics_kv_ratio(z) == 1.0, "clamped to 1");
  z.kv_held_positions = 10, z.idle_positions = 30;
  check(metrics_kv_ratio(z) == 0.0, "idle over held (page rounding) clamped to 0");
  z.kv_held_positions = 100, z.idle_positions = 100;
  check(metrics_kv_ratio(z) == 0.0, "all cached, none held by a request: 0");
  check(parse(render_metrics(z, {}, {}, {"m", 0, 0})).numeric, "an all-zero render is numeric");

  std::printf("gemma 4\n");
  gemma4::Scheduler::Stats g;
  g.active = 1, g.waiting = 0, g.steps = 7, g.prompt_tokens = 300, g.cached_tokens = 280, g.prefill_tokens = 20, g.generated_tokens = 90;
  g.kv_pool_positions = 131072, g.kv_held_positions = 4096, g.idle_positions = 1024;
  g.verifies = 30, g.drafted = 210, g.accepted = 60;
  g.busy_s = 2.5, g.prefill_seq_s = 0.5, g.decode_seq_s = 2.0, g.first_token_s = 0.25, g.first_tokens = 1;
  const std::string gt = render_metrics(g, e, h, {"nemotron-3.5-bolt-e4b", 2, 65536});
  const Parsed gp = parse(gt);
  bool gonce = true;
  for (const char* n : names) {
    const bool snap = std::string(n).rfind("snapshot", 0) == 0;
    auto it = gp.n.find(std::string("eklow_") + n);
    if (snap ? it != gp.n.end() : (it == gp.n.end() || it->second != 1)) {
      std::printf("    bad: %s\n", n);
      gonce = false;
    }
  }
  check(gonce, "every contract name exactly once, no snapshot gauges (Gemma 4 keeps none)");
  check(gp.numeric && gp.n.size() == 24, "numeric, info plus 23 samples");
  check(gp.v.at("eklow_kv_usage_ratio") == "0.0234375", "kv ratio from the global layers' pool: (4096 - 1024) / 131072");
  check(gp.v.at("eklow_spec_drafted_total") == "210" && gp.v.at("eklow_spec_accepted_total") == "60", "the draft's counters");
  check(gp.v.at("eklow_decode_sequence_seconds_total") == "2" && gp.v.at("eklow_first_token_seconds_total") == "0.25", "second counters");
  check(gt.find("eklow_info{model=\"nemotron-3.5-bolt-e4b\"} 1\n") != std::string::npos, "the served name");

  std::printf(g_fail ? "FAILED\n" : "all ok\n");
  return g_fail ? 1 : 0;
}
