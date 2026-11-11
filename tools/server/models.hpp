// GET /v1/models for one model, on every port eklow listens on. maitred counts an upstream as up only while
// GET <base_url>/models answers 2xx, so a port without it reads as down (llmtop: standby) while it serves.
#pragma once

#include <httplib.h>

#include <chrono>
#include <cstdint>
#include <string>

#include "core/json.hpp"

namespace eklow::serve {

// The listing of `id` alone, with llama.cpp's meta (n_ctx, n_slots). context_length comes first: clients take the
// first context key they find.
inline std::string models_json(const std::string& id, int64_t ctx, int64_t slots) {
  const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  json::Writer w(false);
  w.begin_object().kv("object", "list").key("data").begin_array().begin_object().kv("id", id).kv("context_length", ctx);
  w.kv("object", "model").kv("created", now).kv("owned_by", "eklow");
  w.key("meta").begin_object().kv("n_ctx", ctx).kv("n_slots", slots).end_object();
  w.end_object().end_array().end_object();
  return w.str();
}

inline void serve_models(httplib::Server& http, const std::string& id, int64_t ctx, int64_t slots) {
  http.Get("/v1/models", [id, ctx, slots](const httplib::Request&, httplib::Response& res) {
    res.set_content(models_json(id, ctx, slots), "application/json");
  });
}

}  // namespace eklow::serve
