// eklow-server: Gemma 4 on the A770M behind an OpenAI-compatible HTTP API.
//
//   eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N]
//                [--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--expert-profile JSON [--experts-on-card N] [--card-margin-mib N] [--stream-slots N] [[--stream-min-rows N]]
//                [--embed-model GGUF --embed-port N [--embed-alias NAME]]
//
// --ctx is the context of each slot (positions), Gemma's full 262,144 by default; --batch the
// tokens of one engine step and --chunk the prefill tokens of one sequence in a step. The global
// layers' K/V are int8 by default, which is what lets 2 x 262k fit beside the weights; --kv-fp16
// decodes a little faster at depth but cannot hold 2 x 262k.
// --expert-profile (the 26B): the experts of each layer that the routing profile says are busiest stay on
// the card and the rest are computed on the CPU. How many (--experts-on-card) is, by default, as many as fit
// in the card's free memory after the KV cache, the workspace and the embedding model, less
// --card-margin-mib (1024) for what the first requests allocate and the driver keeps.
// Endpoints: POST /v1/chat/completions (streamed or whole), GET /v1/models, GET /health; with an
// embedding model, POST /v1/embeddings, GET /v1/models and GET /health on its own port (the house
// runs chat and embeddings as separate upstreams).
#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
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

void reply_error(httplib::Response& res, const serve::ApiError& e) {
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
// step starts until they are done.
void serve_embeddings(httplib::Server& http, model::EmbedGemma2& m, const text::Tokenizer& tok, std::mutex& mu,
                      serve::Scheduler& sched, const std::string& alias, int64_t max_tokens) {
  http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
  http.Get("/v1/models", [alias, max_tokens](const httplib::Request&, httplib::Response& res) {
    res.set_content(models_json(alias, max_tokens, 1), "application/json");
  });
  http.Post("/v1/embeddings", [&m, &tok, &mu, &sched, alias, max_tokens](const httplib::Request& req, httplib::Response& res) {
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
      {
        std::lock_guard lk(mu);
        const serve::Scheduler::Hold hold(sched);
        embs = m.embed(seqs);
      }
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
  int64_t ctx = 262144, batch = 2048, chunk = 1024;
  bool kv_int8 = true;
  std::string expert_profile;
  int64_t experts_on_card = -2;  // as many as fit, once there is a profile
  int64_t card_margin_mib = 1024;
  int stream_slots = 32, stream_min_rows = 16;
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
    else {
      std::fprintf(stderr, "usage: eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N] "
                           "[--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--expert-profile JSON [--experts-on-card N] [--card-margin-mib N]] [--embed-model GGUF --embed-port N [--embed-alias NAME]]\n");
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

    serve::Scheduler sched(eng, tok, {batch, chunk, ec.max_ctx});
    std::thread engine_thread([&] { sched.run(g_stop); });
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
      try {
        sched.submit(r);
      } catch (const std::invalid_argument&) {
        return reply_error(res, serve::context_exceeded(int64_t(r->prompt.size()), ec.max_ctx));
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

    httplib::Server embed_http;
    std::thread embed_thread;
    if (embed_model) {
      embed_http.new_task_queue = [] { return new httplib::ThreadPool(4); };
      serve_embeddings(embed_http, *embed_model, *embed_tok, embed_mu, sched, embed_alias, 8192);
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
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-server: %s\n", e.what());
    return 1;
  }
  return 0;
}
