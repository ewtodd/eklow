// POST /v1/systemone: laya's System One (its request and response shape, laya serve.py) over vega's checker (src/vega),
// for son-of-anton's end-of-turn checks. Shared by eklow-server (--checker-*, beside a chat model) and eklow-vega serve
// (the checker alone).
//
//   request  {"state": "...", "questions": {id: {"type": "noul", "instructions": "...", "criteria"|"labels"?: {...}}}}
//   response {"model": ALIAS, "answers": {id: {"type": "noul", "noul": P(yes), "probabilities": {"false", "true"},
//             "readout": "engine"|"ttt", "calibrated": bool}}, "usage": {"input_tokens": N, "output_tokens": 0}}
// Errors are laya's: {"detail": "..."} with 400 (not an object with questions), 422 (a question laya would refuse, or one
// that is not noul: eklow answers yes/no questions only), 413 (the prompt is longer than the reader holds), 500. One
// request at a time reads the card; `hold`, when given, runs with the chat scheduler held, as the embeddings do, so the
// check is not queued behind chat steps.
#pragma once

#include <httplib.h>

#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "vega/checker.hpp"

namespace eklow::serve {

struct SystemOneConfig {
  std::string alias = "vega-u";
  std::function<void(const std::string&)> log;      // a line per request
  std::function<std::shared_ptr<void>()> hold;      // takes the card from chat until the returned handle is dropped
};

inline void serve_systemone(httplib::Server& http, vega::Checker& ck, std::mutex& mu, const SystemOneConfig& cfg) {
  http.Post("/v1/systemone", [&ck, &mu, cfg](const httplib::Request& req, httplib::Response& res) {
    auto fail = [&](int status, const std::string& detail) {
      if (cfg.log) cfg.log("systemone: error " + std::to_string(status) + ": " + detail);
      json::Writer w(false);
      w.begin_object().kv("detail", detail).end_object();
      res.status = status;
      res.set_content(w.str(), "application/json");
    };
    try {
      json::Value body;
      try {
        body = json::parse(req.body);
      } catch (const std::exception& e) {
        return fail(400, std::string("the body is not JSON: ") + e.what());
      }
      const auto t0 = std::chrono::steady_clock::now();
      vega::CheckerResult r;
      double wait = 0;
      {
        std::lock_guard lk(mu);
        const std::shared_ptr<void> held = cfg.hold ? cfg.hold() : nullptr;
        wait = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        r = ck.predict(body);
      }
      json::Writer w(false);
      w.begin_object().kv("model", cfg.alias).key("answers").begin_object();
      std::string line;
      for (const auto& a : r.answers) {
        w.key(a.qid).begin_object().kv("type", "noul").kv("noul", a.noul).key("probabilities").begin_object();
        w.kv("false", a.probs[0]).kv("true", a.probs[1]).end_object().kv("readout", a.readout).kv("calibrated", a.calibrated).end_object();
        char b[96];
        std::snprintf(b, sizeof b, " %s %.4f", a.qid.c_str(), a.noul);
        line += b;
      }
      w.end_object().key("usage").begin_object().kv("input_tokens", r.tokens).kv("output_tokens", int64_t(0)).end_object().end_object();
      if (cfg.log) {
        char b[160];
        std::snprintf(b, sizeof b, "systemone: %zu question%s, %lld tokens | wait %.2f s, %.3f s (read %.3f s) |", r.answers.size(),
                      r.answers.size() == 1 ? "" : "s", (long long)r.tokens, wait, r.total_ms / 1000.0, r.read_ms / 1000.0);
        cfg.log(b + line);
      }
      res.set_header("X-Inference-Time-Ms", std::to_string(r.total_ms));
      res.set_content(w.str(), "application/json");
    } catch (const vega::Refusal& e) {
      fail(e.status, e.what());
    } catch (const std::exception& e) {
      // The reason goes to the log, not to the client (laya's rule: no paths or allocator text in a response).
      if (cfg.log) cfg.log(std::string("systemone: inference failed: ") + e.what());
      fail(500, "inference failed");
    }
  });
}

}  // namespace eklow::serve
