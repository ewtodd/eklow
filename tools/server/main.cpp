// eklow-server: Gemma 4 on the A770M behind an OpenAI-compatible HTTP API.
//
//   eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N]
//                [--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--expert-profile JSON [--experts-on-card N] [--card-margin-mib N] [--stream-slots N] [[--stream-min-rows N]]
//                [--embed-model GGUF --embed-port N [--embed-alias NAME]] [--log-interval S]
//
// --ctx is the context of each slot (positions), Gemma's full 262,144 by default; --batch the
// tokens of one engine step and --chunk the prefill tokens of one sequence in a step (2,048: twice the rows per
// expert GEMM of 1,024 and half the steps, +14% prefill at 32k for ~0.4 GB of sliding rings, 4 experts a layer at
// 2 x 262k; DECISIONS 2026-10-27). The global layers' K/V are int8 by default, which is what lets 2 x 262k
// fit beside the weights; --kv-fp16 decodes a little faster at depth but cannot hold 2 x 262k.
// --expert-profile (the 26B): the experts of each layer that the routing profile says are busiest stay on
// the card and the rest are computed on the CPU. How many (--experts-on-card) is, by default, as many as fit
// in the card's free memory after the KV cache, the workspace and the embedding model, less
// --card-margin-mib (1024) for what the first requests allocate and the driver keeps.
// Endpoints: POST /v1/chat/completions (streamed or whole), GET /v1/models, GET /health; with an
// embedding model, POST /v1/embeddings, GET /v1/models and GET /health on its own port (the house
// runs chat and embeddings as separate upstreams).
// The log (stdout, unholy's form, each line prefixed with the local time): a line per chat request as it ends
// (finish reason; prompt, cached and completion tokens; queue, time to first token, prefill and decode tok/s), a
// line per embedding request, a line per rejected request, and every --log-interval seconds (10; 0 for none) while
// there is traffic the engine's rates, running and waiting requests, the slots' positions and the prefix hit rate.
#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <mutex>
#include <string>
#include <thread>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine.hpp"
#include "model/embed_gemma2.hpp"
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

double seconds(std::chrono::steady_clock::duration d) { return std::chrono::duration<double>(d).count(); }

// The log, in unholy's form: lines on stdout, each written whole under one lock and prefixed with the local time.
std::mutex g_log_mu;

void log_line(const std::string& line) {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  localtime_r(&t, &tm);
  char ts[16];
  std::strftime(ts, sizeof(ts), "%H:%M:%S", &tm);
  std::lock_guard lk(g_log_mu);
  std::printf("%s %s\n", ts, line.c_str());
  std::fflush(stdout);
}

template <class... A>
std::string fmt(const char* f, A... a) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), f, a...);
  return buf;
}

// A chat request as it ended: why, its tokens, and its times (prefill over the uncached part of the prompt).
std::string request_line(const serve::RequestLog& l) {
  std::string out = "request " + l.tag + ": " + l.reason;
  if (!l.error.empty()) out += " (" + l.error + ")";
  out += fmt(" | prompt %lld", (long long)l.prompt_tokens);
  if (l.cached_tokens > 0) out += fmt(" (%lld cached)", (long long)l.cached_tokens);
  out += fmt(", completion %lld | queue %.2f s", (long long)l.completion_tokens, l.queue_s);
  if (l.ttft_s >= 0) {
    out += fmt(", ttft %.2f s", l.ttft_s);
    if (l.prefill_s > 0) out += fmt(", prefill %.1f tok/s", double(l.prompt_tokens - l.cached_tokens) / l.prefill_s);
    if (l.completion_tokens > 1 && l.decode_s > 0) out += fmt(", decode %.1f tok/s", double(l.completion_tokens - 1) / l.decode_s);
  }
  return out;
}

// The embedding endpoint's counters since the start.
struct EmbedCounts {
  int64_t requests = 0, inputs = 0, tokens = 0;
};
struct EmbedStats {
  std::atomic<int64_t> requests{0}, inputs{0}, tokens{0};
  EmbedCounts snapshot() const { return {requests.load(), inputs.load(), tokens.load()}; }
};

// The server over the interval from `a` to `b` (unholy's, after vLLM's periodic stats line): prefill and generated
// tokens a second, requests running and waiting, the positions the slots hold (of slots x ctx; cached: in slots with
// no request, kept for prefix reuse), the share of admitted prompt tokens already in their slot, and embeddings.
std::string engine_line(const serve::Scheduler::Stats& a, const serve::Scheduler::Stats& b, const EmbedCounts& ea, const EmbedCounts& eb,
                        double dt, int64_t capacity) {
  std::string out = fmt("engine: prompt %.1f tok/s, generation %.1f tok/s | running %lld, waiting %lld",
                        double(b.prefill_tokens - a.prefill_tokens) / dt, double(b.generated_tokens - a.generated_tokens) / dt,
                        (long long)b.active, (long long)b.waiting);
  out += fmt(" | kv %.1f%% (%lld of %lld positions, %lld cached)", 100.0 * double(b.positions) / double(std::max<int64_t>(capacity, 1)),
             (long long)b.positions, (long long)capacity, (long long)b.idle_positions);
  if (b.prompt_tokens > a.prompt_tokens)
    out += fmt(" | prefix hit %.1f%%", 100.0 * double(b.cached_tokens - a.cached_tokens) / double(b.prompt_tokens - a.prompt_tokens));
  if (eb.inputs > ea.inputs)
    out += fmt(" | embeddings %lld requests, %lld inputs, %.1f tok/s", (long long)(eb.requests - ea.requests), (long long)(eb.inputs - ea.inputs),
               double(eb.tokens - ea.tokens) / dt);
  return out;
}

void reply_error(httplib::Response& res, const serve::ApiError& e) {
  log_line(fmt("error %d (%s): ", e.status, e.type.c_str()) + e.what());
  res.status = e.status;
  res.set_content(serve::error_json(e), "application/json");
}

std::string models_json(const std::string& id, int64_t ctx, int64_t slots) {
  // context_length first: clients take the first context key they find.
  json::Writer w(false);
  w.begin_object().kv("object", "list").key("data").begin_array().begin_object().kv("id", id).kv("context_length", ctx);
  w.kv("object", "model").kv("created", now_s()).kv("owned_by", "eklow");
  w.key("meta").begin_object().kv("n_ctx", ctx).kv("n_slots", slots).end_object();
  w.end_object().end_array().end_object();
  return w.str();
}

// The embedding endpoint: inputs one at a time through the model (each its own sequence,
// attended both ways), each at most max_tokens long, with the scheduler held so that no new chat
// step starts until they are done. Each request is logged (its wait for the model, behind another request,
// and its time) and counted in `stats`.
void serve_embeddings(httplib::Server& http, model::EmbedGemma2& m, const text::Tokenizer& tok, std::mutex& mu,
                      serve::Scheduler& sched, const std::string& alias, int64_t max_tokens, EmbedStats& stats) {
  http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
  http.Get("/v1/models", [alias, max_tokens](const httplib::Request&, httplib::Response& res) {
    res.set_content(models_json(alias, max_tokens, 1), "application/json");
  });
  http.Post("/v1/embeddings", [&m, &tok, &mu, &sched, &stats, alias, max_tokens](const httplib::Request& req, httplib::Response& res) {
    try {
      json::Value body;
      try {
        body = json::parse(req.body);
      } catch (const std::exception& e) {
        throw serve::ApiError(400, "invalid_request_error", "", std::string("the body is not JSON: ") + e.what());
      }
      const json::Value* input = body.is_object() ? body.find("input") : nullptr;
      if (!input) throw serve::ApiError(400, "invalid_request_error", "", "input is required");
      // A string, strings, token ids, or arrays of token ids.
      std::vector<std::vector<int32_t>> seqs;
      auto text = [&](const std::string& str) {
        std::vector<int32_t> ids{tok.bos()};
        for (int32_t id : tok.encode(str)) ids.push_back(id);
        ids.push_back(tok.eos());
        return ids;
      };
      auto ids_of = [&](const json::Value& a) {
        std::vector<int32_t> ids;
        for (const auto& v : a.items()) {
          if (!v.is_int()) throw serve::ApiError(400, "invalid_request_error", "", "token ids must be integers");
          ids.push_back(int32_t(std::stoll(v.int_digits())));
        }
        return ids;
      };
      if (input->is_string()) seqs.push_back(text(input->as_string()));
      else if (input->is_array() && !input->items().empty() && input->items()[0].is_int()) seqs.push_back(ids_of(*input));
      else if (input->is_array())
        for (const auto& v : input->items()) {
          if (v.is_string()) seqs.push_back(text(v.as_string()));
          else if (v.is_array()) seqs.push_back(ids_of(v));
          else throw serve::ApiError(400, "invalid_request_error", "", "input must hold strings or token ids");
        }
      else throw serve::ApiError(400, "invalid_request_error", "", "input must be a string or an array");
      if (seqs.empty()) throw serve::ApiError(400, "invalid_request_error", "", "input is empty");
      int64_t total = 0;
      for (const auto& q : seqs) {
        if (int64_t(q.size()) > max_tokens)
          throw serve::ApiError(400, "invalid_request_error", "",
                                "input is too large to process: " + std::to_string(q.size()) + " tokens, at most " + std::to_string(max_tokens));
        for (int32_t id : q)
          if (id < 0 || size_t(id) >= tok.size()) throw serve::ApiError(400, "invalid_request_error", "", "token id out of range");
        total += int64_t(q.size());
      }
      json::Writer w(false);
      w.begin_object().kv("object", "list").kv("model", alias).key("data").begin_array();
      std::vector<std::vector<float>> embs;
      const auto t0 = std::chrono::steady_clock::now();
      std::chrono::steady_clock::time_point t1;
      {
        std::lock_guard lk(mu);
        const serve::Scheduler::Hold hold(sched);
        t1 = std::chrono::steady_clock::now();
        embs = m.embed(seqs);
      }
      const double busy = seconds(std::chrono::steady_clock::now() - t1);
      ++stats.requests;
      stats.inputs += int64_t(seqs.size());
      stats.tokens += total;
      log_line(fmt("embeddings: %zu input%s, %lld tokens | wait %.2f s, %.3f s, %.0f tok/s", seqs.size(), seqs.size() == 1 ? "" : "s",
                   (long long)total, seconds(t1 - t0), busy, busy > 0 ? double(total) / busy : 0.0));
      for (size_t i = 0; i < embs.size(); ++i) {
        w.begin_object().kv("object", "embedding").kv("index", int64_t(i)).key("embedding").begin_array();
        for (float v : embs[i]) w.value(double(v));
        w.end_array().end_object();
      }
      w.end_array().key("usage").begin_object().kv("prompt_tokens", total).kv("total_tokens", total).end_object().end_object();
      res.set_content(w.str(), "application/json");
    } catch (const serve::ApiError& e) {
      reply_error(res, e);
    } catch (const std::exception& e) {
      reply_error(res, serve::ApiError(500, "server_error", "", e.what()));
    }
  });
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, alias = "gemma-4-12b", host = "127.0.0.1", embed_path, embed_alias = "embeddinggemma-2";
  int port = 8081, slots = 2, embed_port = 8082;
  int64_t ctx = 262144, batch = 2048, chunk = 2048;
  bool kv_int8 = true;
  std::string expert_profile;
  int64_t experts_on_card = -2;  // as many as fit, once there is a profile
  int64_t card_margin_mib = 1024;
  int stream_slots = 40, stream_min_rows = 16;
  double log_interval = 10;  // seconds between the engine's log lines while there is traffic; 0: none
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
    else if (a == "--expert-profile") expert_profile = next();
    else if (a == "--experts-on-card") experts_on_card = std::atoll(next().c_str());
    else if (a == "--card-margin-mib") card_margin_mib = std::atoll(next().c_str());
    else if (a == "--stream-slots") stream_slots = std::atoi(next().c_str());
    else if (a == "--stream-min-rows") stream_min_rows = std::atoi(next().c_str());
    else if (a == "--embed-model") embed_path = next();
    else if (a == "--embed-port") embed_port = std::atoi(next().c_str());
    else if (a == "--embed-alias") embed_alias = next();
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--kv-fp16") kv_int8 = false;
    else if (a == "--log-interval") log_interval = std::atof(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N] "
                           "[--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--expert-profile JSON [--experts-on-card N] [--card-margin-mib N]] [--embed-model GGUF --embed-port N [--embed-alias NAME]] "
                           "[--log-interval S]\n");
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
    ec.int8_global_kv = kv_int8;
    ec.expert_profile = expert_profile;
    ec.experts_on_card = expert_profile.empty() ? -1 : experts_on_card;
    ec.card_margin_bytes = uint64_t(card_margin_mib) << 20;
    ec.stream_slots = stream_slots;
    ec.stream_min_rows = stream_min_rows;
    // The embedding model: its own queue on the card, so its work does not wait behind steps. It goes on
    // first, with room for its longest input, so that the experts' placement sees what it leaves.
    std::unique_ptr<gguf::File> embed_file;
    std::unique_ptr<text::Tokenizer> embed_tok;
    std::unique_ptr<sycl::queue> embed_q;
    std::unique_ptr<model::EmbedGemma2> embed_model;
    std::mutex embed_mu;
    if (!embed_path.empty()) {
      embed_file = std::make_unique<gguf::File>(gguf::File::open(embed_path));
      embed_tok = std::make_unique<text::Tokenizer>(text::Tokenizer::from_gguf(*embed_file));
      embed_q = std::make_unique<sycl::queue>(q.get_device(), sycl::property::queue::in_order());
      embed_model = std::make_unique<model::EmbedGemma2>(*embed_q, *embed_file);
      embed_model->reserve(8192);  // the longest input: no allocation at request time
    }
    engine::Engine eng(q, file, ec);
    const double GiB = 1073741824.0;
    std::fprintf(stderr, "eklow-server: %s on %s, %.2f GiB of weights, %.2f GiB of KV cache (%d slots of %lld), %.1f s\n",
                 alias.c_str(), q.get_device().get_info<sycl::info::device::name>().c_str(), eng.weight_bytes() / GiB,
                 eng.cache_bytes() / GiB, slots, (long long)ec.max_ctx,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    if (eng.model().n_expert > 0)
      std::fprintf(stderr,
                   "eklow-server: experts on the card: N = %lld of %lld per layer%s (one expert of every layer is %.0f MiB; free card memory before them "
                   "%.2f GiB, margin %.2f GiB, free after %.2f GiB)\n",
                   (long long)eng.experts_on_card(), (long long)eng.model().n_expert, eng.experts_on_card() < eng.model().n_expert ? ", the rest on the CPU" : "",
                   eng.expert_set_bytes() / 1048576.0, eng.free_at_placement() / GiB, double(ec.card_margin_bytes) / GiB,
                   rt::free_memory(q.get_device()).value_or(0) / GiB);
    if (eng.model().n_expert > 0 && eng.stream_slots() > 0)
      std::fprintf(stderr, "eklow-server: prefill streams busy cold experts to the card: %d slots (%.0f MiB on the card), %.2f GiB of cold experts pinned in host memory\n",
                   eng.stream_slots(), eng.stream_slot_bytes() / 1048576.0, eng.pinned_bytes() / GiB);

    serve::Scheduler sched(eng, tok, {batch, chunk, ec.max_ctx, [](const serve::RequestLog& l) { log_line(request_line(l)); }});
    std::thread engine_thread([&] { sched.run(g_stop); });
    EmbedStats embed_stats;
    // The engine's line every log_interval seconds while there is traffic, and once more after it ends (rates
    // back to zero).
    std::thread stats_thread;
    if (log_interval > 0)
      stats_thread = std::thread([&, capacity = int64_t(slots) * ec.max_ctx] {
        const auto every = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(log_interval));
        auto prev = sched.stats();
        auto eprev = embed_stats.snapshot();
        auto t_prev = std::chrono::steady_clock::now();
        bool was_active = false;
        while (!g_stop.load()) {
          while (!g_stop.load() && std::chrono::steady_clock::now() < t_prev + every) std::this_thread::sleep_for(std::chrono::milliseconds(200));
          if (g_stop.load()) break;
          const auto cur = sched.stats();
          const auto ecur = embed_stats.snapshot();
          const auto now = std::chrono::steady_clock::now();
          const bool active = cur.steps != prev.steps || cur.active > 0 || cur.waiting > 0 || ecur.requests != eprev.requests;
          if (active || was_active) log_line(engine_line(prev, cur, eprev, ecur, seconds(now - t_prev), capacity));
          was_active = active, prev = cur, eprev = ecur, t_prev = now;
        }
      });
    const serve::ChatDefaults defaults{serve::SamplingParams{}, ec.max_ctx};

    httplib::Server http;
    http.new_task_queue = [] { return new httplib::ThreadPool(16); };
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
      res.set_content(models_json(alias, ec.max_ctx, slots), "application/json");
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
      r->tools = cr.tool_names;
      r->tag = serve::new_completion_id();
      try {
        sched.submit(r);
      } catch (const std::invalid_argument&) {
        return reply_error(res, serve::context_exceeded(int64_t(r->prompt.size()), ec.max_ctx));
      }
      auto out = std::make_shared<serve::ChatResponse>(r->tag, cr.model.empty() ? alias : cr.model, now_s());
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

    httplib::Server embed_http;
    std::thread embed_thread;
    if (embed_model) {
      embed_http.new_task_queue = [] { return new httplib::ThreadPool(4); };
      serve_embeddings(embed_http, *embed_model, *embed_tok, embed_mu, sched, embed_alias, 8192, embed_stats);
      std::fprintf(stderr, "eklow-server: %s, %.2f GiB, listening on %s:%d\n", embed_alias.c_str(), embed_model->bytes() / 1073741824.0,
                   host.c_str(), embed_port);
      embed_thread = std::thread([&] {
        if (!embed_http.listen(host, embed_port)) {
          std::fprintf(stderr, "eklow-server: cannot listen on %s:%d\n", host.c_str(), embed_port);
          g_stop = true;
        }
      });
    }

    std::signal(SIGINT, [](int) { g_stop = true; });
    std::signal(SIGTERM, [](int) { g_stop = true; });
    std::thread stopper([&] {
      while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
      http.stop();
      embed_http.stop();
    });
    std::fprintf(stderr, "eklow-server: listening on %s:%d\n", host.c_str(), port);
    if (!http.listen(host, port)) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    g_stop = true;
    stopper.join();
    engine_thread.join();
    if (embed_thread.joinable()) embed_thread.join();
    if (stats_thread.joinable()) stats_thread.join();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-server: %s\n", e.what());
    return 1;
  }
  return 0;
}
