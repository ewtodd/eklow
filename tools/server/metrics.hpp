// GET /metrics: the Prometheus text exposition (version 0.0.4) llmtop scrapes, prefix eklow. A pure function of the scheduler's Stats, the
// embedding counters and the config, so it is testable without a card (metrics_test.cpp). Names follow llmtop's contract for engines.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>

#include "serve/scheduler.hpp"

namespace eklow::serve {

// The embedding endpoint's counters since the start (the server's EmbedCounts).
struct MetricsEmbed {
  int64_t requests = 0, inputs = 0, tokens = 0;
};
// Chat requests handled and those answered with an error status.
struct MetricsHttp {
  int64_t requests = 0, errors = 0;
};
struct MetricsConfig {
  std::string model;     // the served name (/v1/models)
  int64_t slots = 0, context_tokens = 0;
};

// A label value in the text format: backslash, double quote and newline are escaped.
inline std::string metrics_escape(const std::string& v) {
  std::string out;
  for (char c : v) {
    if (c == '\\') out += "\\\\";
    else if (c == '"') out += "\\\"";
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  return out;
}

// Share of the K/V pool held by sequences that have a request, 0..1. Stats::kv_held_positions counts the pool's pages in use (a cached
// prefix in a slot with no request included) and Stats::idle_positions the positions of slots with no request, so held minus idle is what
// running requests hold; over kv_pool_positions. A zero-capacity pool gives 0, and rounding (held counts whole pages, idle exact positions)
// cannot push the value outside 0..1.
inline double metrics_kv_ratio(const Scheduler::Stats& s) {
  if (s.kv_pool_positions <= 0) return 0.0;
  const double r = double(s.kv_held_positions - s.idle_positions) / double(s.kv_pool_positions);
  return std::min(1.0, std::max(0.0, r));
}

inline std::string render_metrics(const Scheduler::Stats& s, const MetricsEmbed& e, const MetricsHttp& h, const MetricsConfig& c) {
  std::string out;
  char buf[96];
  auto put = [&](const char* name, const char* type, const std::string& value) {
    out += std::string("# TYPE eklow_") + name + " " + type + "\neklow_" + name + " " + value + "\n";
  };
  auto num = [&](int64_t v) { return std::to_string(v); };
  auto flt = [&](double v) {
    if (!std::isfinite(v)) v = 0;
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return std::string(buf);
  };
  out += "# TYPE eklow_info gauge\neklow_info{model=\"" + metrics_escape(c.model) + "\"} 1\n";
  put("slots", "gauge", num(c.slots));
  put("context_tokens", "gauge", num(c.context_tokens));
  put("running", "gauge", num(s.active));
  put("waiting", "gauge", num(s.waiting));
  put("kv_usage_ratio", "gauge", flt(metrics_kv_ratio(s)));
  put("steps_total", "counter", num(s.steps));
  put("busy_seconds_total", "counter", flt(s.busy_s));
  put("prompt_tokens_total", "counter", num(s.prompt_tokens));
  put("cached_prompt_tokens_total", "counter", num(s.cached_tokens));
  put("prefill_tokens_total", "counter", num(s.prefill_tokens));
  put("decode_tokens_total", "counter", num(s.generated_tokens));
  put("prefill_sequence_seconds_total", "counter", flt(s.prefill_seq_s));
  put("decode_sequence_seconds_total", "counter", flt(s.decode_seq_s));
  put("first_token_seconds_total", "counter", flt(s.first_token_s));
  put("first_tokens_total", "counter", num(s.first_tokens));
  put("spec_verify_steps_total", "counter", num(s.verifies));
  put("spec_drafted_total", "counter", num(s.drafted));
  put("spec_accepted_total", "counter", num(s.accepted));
  put("http_requests_total", "counter", num(h.requests));
  put("http_errors_total", "counter", num(h.errors));
  put("embedding_requests_total", "counter", num(e.requests));
  put("embedding_inputs_total", "counter", num(e.inputs));
  put("embedding_tokens_total", "counter", num(e.tokens));
  put("snapshots", "gauge", num(s.snapshots));
  put("snapshot_bytes", "gauge", num(s.snapshot_bytes));
  return out;
}

}  // namespace eklow::serve
