// eklow-server: Gemma 4 on the A770M behind an OpenAI-compatible HTTP API.
//
//   eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N]
//                [--batch N] [--chunk N]
//
// --ctx is the context of each slot (positions); --batch the tokens of one engine step and --chunk
// the prefill tokens of one sequence in a step. Endpoints: POST /v1/chat/completions (streamed or
// whole), GET /v1/models, GET /health.
#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"
#include "serve/openai.hpp"
#include "serve/scheduler.hpp"
#include "text/chat_gemma4.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

namespace {

std::atomic<bool> g_stop{false};

int64_t now_s() { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }

void reply_error(httplib::Response& res, const serve::ApiError& e) {
  res.status = e.status;
  res.set_content(serve::error_json(e), "application/json");
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, alias = "gemma-4-12b", host = "127.0.0.1";
  int port = 8081, slots = 2;
  int64_t ctx = 65536, batch = 2048, chunk = 1024;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model_path = next();
    else if (a == "--alias") alias = next();
    else if (a == "--host") host = next();
    else if (a == "--port") port = std::atoi(next().c_str());
    else if (a == "--slots") slots = std::atoi(next().c_str());
    else if (a == "--ctx") ctx = std::atoll(next().c_str());
    else if (a == "--batch") batch = std::atoll(next().c_str());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N] "
                           "[--batch N] [--chunk N]\n");
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "eklow-server: --model is required\n");
    return 2;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    auto file = gguf::File::open(model_path);
    if (const gguf::Value* t = file.find("tokenizer.chat_template"); !t || !text::is_gemma4_template(t->as_string()))
      throw std::runtime_error(model_path + ": not Gemma 4's chat template");
    const text::Tokenizer tok = text::Tokenizer::from_gguf(file);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::Config ec;
    ec.n_slots = slots;
    ec.max_ctx = (ctx + attn::kPage - 1) / attn::kPage * attn::kPage;
    ec.max_batch = batch;
    ec.max_chunk = chunk;
    ec.max_logits = slots;
    engine::Engine eng(q, file, ec);
    std::fprintf(stderr, "eklow-server: %s on %s, %.2f GiB of weights, %.2f GiB of KV cache (%d slots of %lld), %.1f s\n",
                 alias.c_str(), q.get_device().get_info<sycl::info::device::name>().c_str(), eng.weight_bytes() / 1073741824.0,
                 eng.cache_bytes() / 1073741824.0, slots, (long long)ec.max_ctx,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    serve::Scheduler sched(eng, tok, {batch, chunk, ec.max_ctx});
    std::thread engine_thread([&] { sched.run(g_stop); });
    const serve::ChatDefaults defaults{serve::SamplingParams{}, ec.max_ctx};

    httplib::Server http;
    http.new_task_queue = [] { return new httplib::ThreadPool(16); };
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
      json::Writer w(false);
      w.begin_object().kv("object", "list").key("data").begin_array().begin_object().kv("id", alias).kv("object", "model");
      w.kv("created", now_s()).kv("owned_by", "eklow").key("meta").begin_object().kv("n_ctx", ec.max_ctx).kv("n_slots", int64_t(slots));
      w.end_object().end_object().end_array().end_object();
      res.set_content(w.str(), "application/json");
    });
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
      serve::ChatRequest cr;
      try {
        json::Value body;
        try {
          body = json::parse(req.body);
        } catch (const std::exception& e) {
          throw serve::ApiError(400, "invalid_request_error", "", std::string("the body is not JSON: ") + e.what());
        }
        cr = serve::parse_chat(body, tok, defaults);
      } catch (const serve::ApiError& e) {
        return reply_error(res, e);
      }
      auto r = std::make_shared<serve::Request>();
      r->prompt = std::move(cr.prompt);
      r->sampling = cr.sampling;
      r->max_tokens = cr.max_tokens;
      r->stop = cr.stop;
      r->in_thinking = cr.in_thinking;
      try {
        sched.submit(r);
      } catch (const std::invalid_argument& e) {
        return reply_error(res, serve::ApiError(400, "invalid_request_error", "context_length_exceeded", e.what()));
      }
      auto out = std::make_shared<serve::ChatResponse>(serve::new_completion_id(), cr.model.empty() ? alias : cr.model, now_s());
      if (!cr.stream) {
        serve::Event e;
        try {
          do {
            while (!r->next(e, std::chrono::milliseconds(1000))) {
            }
            out->add(e);
          } while (!e.done);
        } catch (const serve::ApiError& err) {
          return reply_error(res, err);
        }
        return res.set_content(out->body(), "application/json");
      }
      const bool usage = cr.include_usage;
      res.set_header("Cache-Control", "no-cache");
      res.set_chunked_content_provider(
          "text/event-stream",
          [r, out, usage](size_t, httplib::DataSink& sink) {
            serve::Event e;
            while (!r->next(e, std::chrono::milliseconds(1000)))
              if (!sink.is_writable()) {
                r->cancel();
                return false;
              }
            const std::string text = out->chunks(e, usage);
            if (!text.empty() && !sink.write(text.data(), text.size())) {
              r->cancel();
              return false;
            }
            if (e.done) sink.done();
            return true;
          },
          [r](bool) { r->cancel(); });
    });

    std::signal(SIGINT, [](int) { g_stop = true; });
    std::signal(SIGTERM, [](int) { g_stop = true; });
    std::thread stopper([&] {
      while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
      http.stop();
    });
    std::fprintf(stderr, "eklow-server: listening on %s:%d\n", host.c_str(), port);
    if (!http.listen(host, port)) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    g_stop = true;
    stopper.join();
    engine_thread.join();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-server: %s\n", e.what());
    return 1;
  }
  return 0;
}
