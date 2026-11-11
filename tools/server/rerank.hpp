// POST /v1/rerank (and /rerank, /v1/reranking, /reranking, as llama.cpp's server answers them): Qwen3-Reranker
// (src/model/rerank_qwen3.hpp) in llama.cpp's request and answer shape, which maitred's rerank route and son-of-anton's
// recall send. Shared by eklow-server (--rerank-*, beside a chat model) and eklow-rerank serve (the reranker alone).
//
//   request  {"model"?, "query": "...", "documents": ["..." | {"text": "..."}], "top_n"?}   ("texts" for TEI's shape)
//   response {"model", "object": "list", "usage": {"prompt_tokens", "total_tokens"}, "results": [{"index", "relevance_score"}]}
//            sorted by score, the first top_n (all by default); TEI's shape is the bare array with "score".
// Errors are llama.cpp's kind: {"error": {"code", "message", "type"}}, 400 for a malformed request or a pair longer than
// the model takes. One request at a time reads the card; `hold`, when given, runs with the chat scheduler held.
#pragma once

#include <httplib.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/rerank_qwen3.hpp"
#include "models.hpp"

namespace eklow::serve {

struct RerankConfig {
  std::string alias = "qwen3-reranker-0.6b";
  std::function<void(const std::string&)> log;  // a line per request
  std::function<std::shared_ptr<void>()> hold;  // takes the card from chat until the returned handle is dropped
};

inline void serve_rerank(httplib::Server& http, model::RerankQwen3& m, std::mutex& mu, const RerankConfig& cfg) {
  auto handler = [&m, &mu, cfg](const httplib::Request& req, httplib::Response& res) {
    auto fail = [&](int status, const std::string& msg) {
      if (cfg.log) cfg.log("rerank: error " + std::to_string(status) + ": " + msg);
      json::Writer w(false);
      w.begin_object().key("error").begin_object().kv("code", int64_t(status)).kv("message", msg);
      w.kv("type", status == 400 ? "invalid_request_error" : "server_error").end_object().end_object();
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
      if (!body.is_object()) return fail(400, "the body must be an object");
      const json::Value* query = body.find("query");
      const bool tei = !body.find("documents") && body.find("texts");
      const json::Value* docs = tei ? body.find("texts") : body.find("documents");
      auto text_of = [](const json::Value& v, std::string& out) {
        if (v.is_string()) return out = v.as_string(), true;
        const json::Value* t = v.is_object() ? v.find("text") : nullptr;
        if (!t) return false;
        if (t->is_string()) return out = t->as_string(), true;
        if (!t->is_array()) return false;
        for (const auto& s : t->items()) {
          if (!s.is_string()) return false;
          out += s.as_string();
        }
        return true;
      };
      std::string q;
      if (!query || !text_of(*query, q)) return fail(400, "\"query\" must be provided as a string or {\"text\": ...}");
      if (!docs || !docs->is_array() || docs->items().empty()) return fail(400, "\"documents\" must be a non-empty array of strings or objects");
      std::vector<std::string> texts(docs->items().size());
      for (size_t i = 0; i < texts.size(); ++i)
        if (!text_of(docs->items()[i], texts[i])) return fail(400, "documents must be strings or {\"text\": ...}; images are not served");
      int64_t top_n = int64_t(texts.size());
      if (const json::Value* t = body.find("top_n"); t && !t->is_null()) {
        if (!t->is_int() || std::stoll(t->int_digits()) < 0) return fail(400, "\"top_n\" must be >= 0");
        top_n = std::min<int64_t>(top_n, std::stoll(t->int_digits()));
      }
      std::vector<std::vector<int32_t>> pairs;
      int64_t tokens = 0;
      for (const auto& d : texts) {
        pairs.push_back(m.prompt(q, d));
        tokens += int64_t(pairs.back().size());
        if (int64_t(pairs.back().size()) > m.max_tokens())
          return fail(400, "input is too large to process: a query and document of " + std::to_string(pairs.back().size()) +
                               " tokens, at most " + std::to_string(m.max_tokens()));
      }
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<double> scores;
      double wait = 0;
      {
        std::lock_guard lk(mu);
        const std::shared_ptr<void> held = cfg.hold ? cfg.hold() : nullptr;
        wait = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        scores = m.score(pairs);
      }
      const double busy = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() - wait;
      std::vector<size_t> order(scores.size());
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] > scores[b]; });
      order.resize(size_t(top_n));
      json::Writer w(false);
      if (!tei) {
        const json::Value* model = body.find("model");
        w.begin_object().kv("model", model && model->is_string() ? model->as_string() : cfg.alias).kv("object", "list");
        w.key("usage").begin_object().kv("prompt_tokens", tokens).kv("total_tokens", tokens).end_object().key("results");
      }
      w.begin_array();
      for (size_t i : order) w.begin_object().kv("index", int64_t(i)).kv(tei ? "score" : "relevance_score", scores[i]).end_object();
      w.end_array();
      if (!tei) w.end_object();
      if (cfg.log) {
        char b[160];
        std::snprintf(b, sizeof b, "rerank: %zu documents, %lld tokens | wait %.2f s, %.3f s, %.0f tok/s", texts.size(), (long long)tokens, wait,
                      busy, busy > 0 ? double(tokens) / busy : 0.0);
        cfg.log(b);
      }
      res.set_content(w.str(), "application/json");
    } catch (const std::exception& e) {
      if (cfg.log) cfg.log(std::string("rerank: failed: ") + e.what());
      fail(500, "rerank failed");
    }
  };
  for (const char* path : {"/v1/rerank", "/rerank", "/v1/reranking", "/reranking"}) http.Post(path, handler);
  serve_models(http, cfg.alias, m.max_tokens(), 1);
}

}  // namespace eklow::serve
