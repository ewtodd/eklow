// eklow-server: Qwen3.6-35B-A3B (stock or a finetune such as Ornith) with DFlash 2 speculative decoding on the A770M,
// behind an OpenAI-compatible HTTP API.
//
//   eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N]
//                [--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--expert-profile JSON [--experts-on-card N] [--card-margin-mib N] [--stream-slots N] [--stream-min-rows N]]
//                [--draft GGUF [--draft-block B]] [--chat-template stock|ornith] [--snapshot-mib N [--snapshot-every N]]
//                [--embed-model GGUF --embed-port N [--embed-alias NAME] [--embed-mmproj GGUF]] [--image-tokens N]
//                [--log-interval S]
//
// --ctx is the context of each slot (positions), 262,144 by default; --batch the tokens of one engine step and --chunk
// the prefill tokens of one sequence in a step (2,048). The attention layers' K/V are int8 by default, which is what lets
// 2 x 262k fit beside the weights; --kv-fp16 decodes a little faster at depth but cannot hold 2 x 262k.
// --expert-profile: the experts of each layer that the routing profile says are busiest stay on the card and the rest are
// computed on the CPU (the model's 256 experts a layer do not fit). How many (--experts-on-card) is, by default, as many as
// fit in the card's free memory after the KV cache, the workspace, the embedding model and the draft, less
// --card-margin-mib (1024) for what the first requests allocate and the driver keeps.
// --draft (a DFlash 2 draft GGUF) turns on speculative decoding: a verify block of --draft-block tokens (default the
// draft's trained block, at most 16) per step, accepted losslessly (the sampler's every feature, the tool grammar included).
// --chat-template says which of Qwen3.6's two templates the model's repo uses when the GGUF's is not one eklow knows (the
// official Ornith GGUFs carry Unsloth's edit): stock drops earlier turns' reasoning, ornith keeps it.
// --snapshot-mib is the pinned host memory kept for snapshots of the DeltaNet state, which make a conversation's or a
// document's prefix reusable (taken before each prompt's generation prompt, every --snapshot-every tokens of a long
// prefill, and at the end of a generation); 0 keeps none, and then only a slot's live sequence is reused.
// Image parts in chat are refused (the vision encoder is not ported); --embed-mmproj (EmbeddingGemma 2's) lets an
// embeddings input be an array of text and image_url parts, --image-tokens being its budget of rows per image
// (llama.cpp's image_max_tokens, 70 to 1,120, 280 by default). Each image is encoded on its request's thread with the
// scheduler held, one at a time per encoder, and counts in the prompt tokens.
// Sampling defaults come from the GGUF's general.sampling.* (temperature, top_k, top_p, min_p), each overridable per request.
// Endpoints: POST /v1/chat/completions (streamed or whole), GET /v1/models, GET /health; with an
// embedding model, POST /v1/embeddings, GET /v1/models and GET /health on its own port (the house
// runs chat and embeddings as separate upstreams).
// The log (stdout, unholy's form, each line prefixed with the local time): a line per chat request as it ends
// (finish reason; prompt, cached and completion tokens; queue, time to first token, prefill and decode tok/s; with a
// draft, the tokens a verify gave), a line per embedding request, a line per rejected request, and every --log-interval
// seconds (10; 0 for none) while there is traffic the engine's rates, running and waiting requests, the slots' positions,
// the prefix hit rate, the draft's acceptance and the snapshots held.
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine.hpp"
#include "image/gemma4_image.hpp"
#include "model/embed_gemma2.hpp"
#include "model/dflash2.hpp"
#include "model/gguf.hpp"
#include "model/vision_gemma4v.hpp"
#include "rt/device.hpp"
#include "serve/openai.hpp"
#include "serve/scheduler.hpp"
#include "text/chat_qwen36.hpp"
#include "text/tokenizer.hpp"
#include "text/tokenizer_qwen.hpp"

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

// A chat request as it ended: why, its tokens, and its times (prefill over the uncached part of the prompt). With a
// draft, the tokens each verify block gave (the accepted drafted ones and the one after them) and the share accepted.
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
  if (l.verifies > 0)
    out += fmt(" | draft %.2f tokens/verify, %lld of %lld accepted", double(l.accepted + l.verifies) / double(l.verifies), (long long)l.accepted,
               (long long)l.drafted);
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
// no request, kept for prefix reuse), the share of admitted prompt tokens already in their slot, the draft's tokens a
// verify and its acceptance, the snapshots held, and embeddings.
std::string engine_line(const serve::Scheduler::Stats& a, const serve::Scheduler::Stats& b, const EmbedCounts& ea, const EmbedCounts& eb,
                        double dt, int64_t capacity) {
  std::string out = fmt("engine: prompt %.1f tok/s, generation %.1f tok/s | running %lld, waiting %lld",
                        double(b.prefill_tokens - a.prefill_tokens) / dt, double(b.generated_tokens - a.generated_tokens) / dt,
                        (long long)b.active, (long long)b.waiting);
  out += fmt(" | kv %.1f%% (%lld of %lld positions, %lld cached)", 100.0 * double(b.positions) / double(std::max<int64_t>(capacity, 1)),
             (long long)b.positions, (long long)capacity, (long long)b.idle_positions);
  if (b.prompt_tokens > a.prompt_tokens)
    out += fmt(" | prefix hit %.1f%%", 100.0 * double(b.cached_tokens - a.cached_tokens) / double(b.prompt_tokens - a.prompt_tokens));
  if (b.verifies > a.verifies)
    out += fmt(" | draft %.2f tokens/verify, %.1f%% of %lld drafted accepted", double(b.accepted - a.accepted + b.verifies - a.verifies) / double(b.verifies - a.verifies),
               100.0 * double(b.accepted - a.accepted) / double(std::max<int64_t>(b.drafted - a.drafted, 1)), (long long)(b.drafted - a.drafted));
  if (b.steps > a.steps && b.verifies > a.verifies)
    out += fmt(" | per step: engine %.1f ms, draft %.1f, sample %.1f, inject %.1f, snapshot %.1f", 1e3 * (b.engine_s - a.engine_s) / double(b.steps - a.steps),
               1e3 * (b.draft_s - a.draft_s) / double(b.steps - a.steps), 1e3 * (b.sample_s - a.sample_s) / double(b.steps - a.steps),
               1e3 * (b.inject_s - a.inject_s) / double(b.steps - a.steps), 1e3 * (b.snapshot_s - a.snapshot_s) / double(b.steps - a.steps));
  if (b.snapshots > 0 || b.snapshots_taken > a.snapshots_taken)
    out += fmt(" | snapshots %lld (%.0f MiB), %lld taken, %lld restored", (long long)b.snapshots, double(b.snapshot_bytes) / 1048576.0,
               (long long)(b.snapshots_taken - a.snapshots_taken), (long long)(b.restores - a.restores));
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

// Images to rows for the text model: decoded and cut on the host, encoded on the card with the scheduler held (the
// encoder runs for seconds, and every wait on the card costs a turn behind a long prefill: DECISIONS 2026-10-26) and
// one at a time, the rows copied to host memory. A few encoded images are kept by the hash of their bytes: a
// conversation sends its images again every turn, and the rows are the same.
class ImageService {
 public:
  ImageService(sycl::queue& q, model::VisionGemma4V& v, int tokens) : q_(q), v_(v), tokens_(tokens) {}
  void bind(serve::Scheduler* s) { sched_ = s; }
  int64_t n_embd() const { return v_.n_embd(); }

  serve::EncodedImage operator()(std::span<const uint8_t> bytes) {
    uint64_t h = 0xcbf29ce484222325ull ^ uint64_t(tokens_);  // FNV-1a over the bytes, from the budget
    for (uint8_t b : bytes) h = (h ^ b) * 0x100000001b3ull;
    {
      std::lock_guard lk(cache_mu_);
      for (auto it = cache_.begin(); it != cache_.end(); ++it)
        if (it->hash == h && it->n_bytes == int64_t(bytes.size())) {
          cache_.splice(cache_.begin(), cache_, it);
          return {h, it->n, it->rows};
        }
    }
    image::Patches patches;
    try {
      patches = image::prepare(image::decode(bytes), v_.spec(tokens_));
    } catch (const std::invalid_argument& e) {
      throw serve::ApiError(400, "invalid_request_error", "unsupported_content", std::string("image_url: ") + e.what());
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard lk(enc_mu_);
    std::unique_ptr<serve::Scheduler::Hold> hold;
    if (sched_) hold = std::make_unique<serve::Scheduler::Hold>(*sched_);
    const auto t1 = std::chrono::steady_clock::now();
    model::DeviceRows rows;
    try {
      rows = v_.encode(patches);
    } catch (const std::invalid_argument& e) {
      throw serve::ApiError(400, "invalid_request_error", "unsupported_content", std::string("image_url: ") + e.what());
    }
    auto host = std::make_shared<std::vector<float>>(size_t(rows.n * rows.dim));
    q_.memcpy(host->data(), rows.data.get(), host->size() * sizeof(float)).wait();
    size_t bad = 0;
    for (float x : *host) bad += !std::isfinite(x);
    if (bad) log_line(fmt("image: %zu of %zu row values are not finite", bad, host->size()));
    log_line(fmt("image: %dx%d patches, %lld rows | wait %.2f s, encoded in %.2f s", patches.cols, patches.rows, (long long)rows.n,
                 seconds(t1 - t0), seconds(std::chrono::steady_clock::now() - t1)));
    {
      std::lock_guard ck(cache_mu_);
      cache_.push_front({h, int64_t(bytes.size()), rows.n, host});
      if (cache_.size() > 6) cache_.pop_back();
    }
    return {h, rows.n, host};
  }

 private:
  struct Entry {
    uint64_t hash;
    int64_t n_bytes, n;
    std::shared_ptr<const std::vector<float>> rows;
  };
  sycl::queue& q_;
  model::VisionGemma4V& v_;
  int tokens_;
  serve::Scheduler* sched_ = nullptr;
  std::mutex enc_mu_, cache_mu_;
  std::list<Entry> cache_;
};

// The embedding endpoint: inputs one at a time through the model (each its own sequence,
// attended both ways), each at most max_tokens long, with the scheduler held so that no new chat
// step starts until they are done. Each request is logged (its wait for the model, behind another request,
// and its time) and counted in `stats`.
void serve_embeddings(httplib::Server& http, model::EmbedGemma2& m, const text::Tokenizer& tok, std::mutex& mu,
                      serve::Scheduler& sched, const std::string& alias, int64_t max_tokens, EmbedStats& stats, ImageService* images) {
  http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
  http.Get("/v1/models", [alias, max_tokens](const httplib::Request&, httplib::Response& res) {
    res.set_content(models_json(alias, max_tokens, 1), "application/json");
  });
  http.Post("/v1/embeddings", [&m, &tok, &mu, &sched, &stats, alias, max_tokens, images](const httplib::Request& req, httplib::Response& res) {
    try {
      json::Value body;
      try {
        body = json::parse(req.body);
      } catch (const std::exception& e) {
        throw serve::ApiError(400, "invalid_request_error", "", std::string("the body is not JSON: ") + e.what());
      }
      const json::Value* input = body.is_object() ? body.find("input") : nullptr;
      if (!input) throw serve::ApiError(400, "invalid_request_error", "", "input is required");
      // A string, strings, token ids, arrays of token ids, or arrays of chat-style content parts (text and image_url),
      // which as llama.cpp's server tokenizes a prompt with media markers are BOS, the text before each image, <|image>,
      // the image's rows, <image|>, the text after, EOS (text parts joined with nothing between them, as it does). Also as
      // its server takes them: an item {"content": [parts]}. A top-level array of parts is one input.
      std::vector<std::vector<int32_t>> seqs;
      std::vector<std::vector<model::ImageRows>> spans;  // per input
      std::vector<serve::EncodedImage> encoded;          // the rows the spans point into
      int64_t image_rows = 0;
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
      auto bad = [](const std::string& m) { return serve::ApiError(400, "invalid_request_error", "", m); };
      auto parts_of = [&](const json::Value& a) {
        std::vector<int32_t> ids{tok.bos()};
        std::vector<model::ImageRows> sp;
        std::vector<serve::EncodedImage> mine;
        std::string pending;  // text since the last image
        auto flush = [&] {
          for (int32_t id : tok.encode(pending)) ids.push_back(id);
          pending.clear();
        };
        for (const auto& part : a.items()) {
          const json::Value* type = part.is_object() ? part.find("type") : nullptr;
          if (!type || !type->is_string()) throw bad("each content part needs a type");
          if (type->as_string() == "text") {
            const json::Value* t = part.find("text");
            if (!t || !t->is_string()) throw bad("a text part needs text");
            pending += t->as_string();
          } else if (type->as_string() == "image_url") {
            if (!images)
              throw serve::ApiError(400, "invalid_request_error", "unsupported_content",
                                    "image_url is not supported: this server was started without a vision projector for the embedding model (--embed-mmproj)");
            const json::Value* iu = part.find("image_url");
            const json::Value* url = iu && iu->is_object() ? iu->find("url") : (iu && iu->is_string() ? iu : nullptr);
            if (!url || !url->is_string()) throw bad("an image_url part needs image_url.url");
            const int32_t open = tok.id("<|image>"), close = tok.id("<image|>");
            if (open < 0 || close < 0) throw bad("the model's vocabulary has no image tokens");
            const std::vector<uint8_t> bytes = serve::image_bytes(url->as_string());
            flush();
            mine.push_back((*images)(bytes));
            ids.push_back(open);
            sp.push_back({int64_t(ids.size()), mine.back().n, nullptr});
            ids.insert(ids.end(), size_t(mine.back().n), 0);  // placeholders: the rows replace their lookups
            ids.push_back(close);
            image_rows += mine.back().n;
          } else throw bad("unsupported content[].type: only text and image_url parts are supported");
        }
        flush();
        ids.push_back(tok.eos());
        for (size_t k = 0; k < sp.size(); ++k) sp[k].rows = mine[k].rows->data();
        for (auto& e : mine) encoded.push_back(std::move(e));
        spans.resize(seqs.size());
        spans.push_back(std::move(sp));
        return ids;
      };
      if (input->is_string()) seqs.push_back(text(input->as_string()));
      else if (input->is_array() && !input->items().empty() && input->items()[0].is_int()) seqs.push_back(ids_of(*input));
      else if (input->is_object() && input->find("content") && input->find("content")->is_array()) seqs.push_back(parts_of(*input->find("content")));
      else if (input->is_array() && !input->items().empty() && input->items()[0].is_object() && input->items()[0].find("type")) seqs.push_back(parts_of(*input));
      else if (input->is_array())
        for (const auto& v : input->items()) {
          if (v.is_string()) seqs.push_back(text(v.as_string()));
          else if (v.is_array() && !v.items().empty() && v.items()[0].is_object()) seqs.push_back(parts_of(v));
          else if (v.is_object() && v.find("content") && v.find("content")->is_array()) seqs.push_back(parts_of(*v.find("content")));  // llama.cpp's shape
          else if (v.is_array()) seqs.push_back(ids_of(v));
          else throw serve::ApiError(400, "invalid_request_error", "", "input must hold strings, token ids or content parts");
        }
      else throw serve::ApiError(400, "invalid_request_error", "", "input must be a string or an array");
      spans.resize(seqs.size());
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
        embs = image_rows > 0 ? m.embed(seqs, &spans) : m.embed(seqs);
      }
      const double busy = seconds(std::chrono::steady_clock::now() - t1);
      ++stats.requests;
      stats.inputs += int64_t(seqs.size());
      stats.tokens += total;
      log_line(fmt("embeddings: %zu input%s, %lld tokens%s | wait %.2f s, %.3f s, %.0f tok/s", seqs.size(), seqs.size() == 1 ? "" : "s",
                   (long long)total, image_rows > 0 ? fmt(" (%lld image)", (long long)image_rows).c_str() : "", seconds(t1 - t0), busy,
                   busy > 0 ? double(total) / busy : 0.0));
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
  std::string model_path, alias = "qwen3.6-35b-a3b", host = "127.0.0.1", embed_path, embed_alias = "embeddinggemma-2", embed_mmproj_path;
  std::string draft_path, chat_template;
  int image_tokens = 280;
  int port = 8081, slots = 2, embed_port = 8082, draft_block = 0;
  int64_t ctx = 262144, batch = 2048, chunk = 2048;
  bool kv_int8 = true;
  std::string expert_profile;
  int64_t experts_on_card = -2;  // as many as fit, once there is a profile
  int64_t card_margin_mib = 1024;
  int64_t snapshot_mib = 4096, snapshot_every = 16384;
  int stream_slots = -1, stream_min_rows = -1;  // -1: the engine's
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
    else if (a == "--draft") draft_path = next();
    else if (a == "--draft-block") draft_block = std::atoi(next().c_str());
    else if (a == "--chat-template") chat_template = next();
    else if (a == "--snapshot-mib") snapshot_mib = std::atoll(next().c_str());
    else if (a == "--snapshot-every") snapshot_every = std::atoll(next().c_str());
    else if (a == "--embed-model") embed_path = next();
    else if (a == "--embed-port") embed_port = std::atoi(next().c_str());
    else if (a == "--embed-alias") embed_alias = next();
    else if (a == "--embed-mmproj") embed_mmproj_path = next();
    else if (a == "--image-tokens") image_tokens = std::atoi(next().c_str());
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--kv-fp16") kv_int8 = false;
    else if (a == "--log-interval") log_interval = std::atof(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N] "
                           "[--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--expert-profile JSON [--experts-on-card N] [--card-margin-mib N] [--stream-slots N] [--stream-min-rows N]] "
                           "[--draft GGUF [--draft-block B]] [--chat-template stock|ornith] [--snapshot-mib N [--snapshot-every N]] "
                           "[--embed-model GGUF --embed-port N [--embed-alias NAME] [--embed-mmproj GGUF]] [--image-tokens N] [--log-interval S]\n");
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "eklow-server: --model is required\n");
    return 2;
  }
  if (image_tokens < 70 || image_tokens > 1120) {
    std::fprintf(stderr, "eklow-server: --image-tokens must be 70 to 1120\n");
    return 2;
  }
  if (!chat_template.empty() && chat_template != "stock" && chat_template != "ornith") {
    std::fprintf(stderr, "eklow-server: --chat-template must be stock or ornith\n");
    return 2;
  }
  if (draft_block < 0 || draft_block > 16 || draft_block == 1) {
    std::fprintf(stderr, "eklow-server: --draft-block must be 2 to 16\n");
    return 2;
  }
  if (snapshot_every < 1024) {
    std::fprintf(stderr, "eklow-server: --snapshot-every must be at least 1024\n");
    return 2;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    auto file = gguf::File::open(model_path);
    // Which of Qwen3.6's two templates renders the conversation: the flag, else the GGUF's own when it is one of the two
    // published files exactly (a third, such as Unsloth's edit in the official Ornith GGUFs, needs the flag).
    text::QwenTemplate tmpl = text::QwenTemplate::Stock;
    if (!chat_template.empty()) {
      tmpl = chat_template == "ornith" ? text::QwenTemplate::Ornith : text::QwenTemplate::Stock;
    } else {
      const gguf::Value* t = file.find("tokenizer.chat_template");
      const auto known = t ? text::qwen36_template_of(t->as_string()) : std::nullopt;
      if (!known)
        throw std::runtime_error(model_path + ": its chat template is not one of Qwen3.6's two that eklow renders (stock, ornith); say which the model uses with --chat-template");
      tmpl = *known;
    }
    const text::QwenTokenizer tok = text::QwenTokenizer::from_gguf(file);
    // The sampling defaults the model's card gives, from the GGUF (llama.cpp's general.sampling.*).
    serve::SamplingParams sampling;
    if (const gguf::Value* v = file.find("general.sampling.temp")) sampling.temperature = float(v->as_float());
    if (const gguf::Value* v = file.find("general.sampling.top_k")) sampling.top_k = int32_t(v->as_int());
    if (const gguf::Value* v = file.find("general.sampling.top_p")) sampling.top_p = float(v->as_float());
    if (const gguf::Value* v = file.find("general.sampling.min_p")) sampling.min_p = float(v->as_float());
    std::unique_ptr<gguf::File> draft_file;
    model::Dflash2Config draft_cfg{};
    if (!draft_path.empty()) {
      draft_file = std::make_unique<gguf::File>(gguf::File::open(draft_path));
      draft_cfg = model::Dflash2Config::from_gguf(*draft_file);
      if (draft_block == 0) draft_block = std::min(draft_cfg.block_size, 16);
    }
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::Config ec;
    ec.n_slots = slots;
    ec.max_ctx = (ctx + attn::kPage - 1) / attn::kPage * attn::kPage;
    ec.max_batch = batch;
    ec.max_chunk = chunk;
    ec.max_logits = int64_t(slots) * (std::max(draft_block, 1) + 1);
    ec.int8_kv = kv_int8;
    ec.expert_profile = expert_profile;
    ec.experts_on_card = expert_profile.empty() ? -1 : experts_on_card;
    ec.card_margin_bytes = uint64_t(card_margin_mib) << 20;
    if (stream_slots >= 0) ec.stream_slots = stream_slots;
    if (stream_min_rows >= 0) ec.stream_min_rows = stream_min_rows;
    if (draft_file) {
      // The draft reads the target's hidden states at its layers; a step captures the rows of its verify blocks and of the
      // last `window` rows of a prompt.
      ec.capture_layers = draft_cfg.target_layers;
      ec.max_capture_rows = chunk + int64_t(slots) * draft_block;
      ec.max_pending = std::max(ec.max_pending, draft_block);
    } else {
      ec.capture_layers.clear();
      ec.draft_reserve_bytes = 0;
    }
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
    // EmbeddingGemma 2's vision encoder, on a queue of its own and loaded before the engine places its experts, so that its
    // weights are counted in what is left of the card. An encode's activations are allocated and freed per image.
    std::unique_ptr<gguf::File> embed_vision_file;
    std::unique_ptr<sycl::queue> embed_vision_q;
    std::unique_ptr<model::VisionGemma4V> embed_vision;
    std::unique_ptr<ImageService> embed_images;
    if (!embed_mmproj_path.empty()) {
      if (!embed_model) throw std::runtime_error("--embed-mmproj needs --embed-model");
      embed_vision_file = std::make_unique<gguf::File>(gguf::File::open(embed_mmproj_path));
      embed_vision_q = std::make_unique<sycl::queue>(q.get_device(), sycl::property::queue::in_order());
      embed_vision = std::make_unique<model::VisionGemma4V>(*embed_vision_q, *embed_vision_file);
      if (embed_vision->n_embd() != embed_model->n_embd())
        throw std::runtime_error(embed_mmproj_path + ": projects to " + std::to_string(embed_vision->n_embd()) + " but the embedding model's width is " +
                                 std::to_string(embed_model->n_embd()));
      // Not image_tokens: reserve(280) on this projector gave NaN embeddings for some images (reserve(400) and up, and no
      // reserve, did not; reported to perf/vision-fast), so it takes the full 1,120 rows (264 MiB instead of 183).
      embed_vision->reserve(1120);
      std::fprintf(stderr, "eklow-server: vision projector %s, %.0f MiB on the card for %d-row images\n", embed_mmproj_path.c_str(), double(embed_vision->bytes()) / 1048576.0, 1120);
      embed_images = std::make_unique<ImageService>(*embed_vision_q, *embed_vision, image_tokens);
    }
    engine::Engine eng(q, file, ec);
    const double GiB = 1073741824.0;
    std::fprintf(stderr, "eklow-server: %s on %s, %.2f GiB of weights, %.2f GiB of KV cache and state (%d slots of %lld), %.1f s\n",
                 alias.c_str(), q.get_device().get_info<sycl::info::device::name>().c_str(), eng.weight_bytes() / GiB,
                 eng.cache_bytes() / GiB, slots, (long long)ec.max_ctx,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::fprintf(stderr,
                 "eklow-server: experts on the card: N = %lld of %lld per layer%s (one expert of every layer is %.0f MiB; free card memory before them "
                 "%.2f GiB, margin %.2f GiB, free after %.2f GiB)\n",
                 (long long)eng.experts_on_card(), (long long)eng.model().n_expert, eng.experts_on_card() < eng.model().n_expert ? ", the rest on the CPU" : "",
                 eng.expert_set_bytes() / 1048576.0, eng.free_at_placement() / GiB, double(ec.card_margin_bytes) / GiB,
                 rt::free_memory(q.get_device()).value_or(0) / GiB);
    if (eng.experts_on_card() < eng.model().n_expert && eng.stream_slots() > 0)
      std::fprintf(stderr, "eklow-server: prefill streams busy cold experts to the card: %d slots, %.2f GiB of cold experts pinned in host memory\n",
                   eng.stream_slots(), eng.pinned_bytes() / GiB);
    // The draft goes on after the engine, into the memory the engine kept for it.
    std::unique_ptr<model::OwnedTarget> draft_target;
    std::unique_ptr<model::Dflash2> draft;
    if (draft_file) {
      draft_target = std::make_unique<model::OwnedTarget>(q, file);
      model::Dflash2::Options opt;
      opt.slots = slots;
      opt.max_block = draft_block;
      draft = std::make_unique<model::Dflash2>(q, *draft_file, draft_target->view(), opt);
      std::fprintf(stderr, "eklow-server: DFlash 2 draft %s: blocks of %d (trained for %d), window %d, %.0f MiB on the card with the target's head and embeddings; free card memory now %.2f GiB\n",
                   draft_path.c_str(), draft_block, draft_cfg.block_size, draft->window(), double(draft->bytes()) / 1048576.0,
                   rt::free_memory(q.get_device()).value_or(0) / GiB);
    }
    serve::SchedulerConfig sc{batch, chunk, ec.max_ctx, [](const serve::RequestLog& l) { log_line(request_line(l)); }};
    sc.draft = draft.get();
    sc.draft_block = draft_block;
    sc.snapshot_bytes = uint64_t(std::max<int64_t>(snapshot_mib, 0)) << 20;
    sc.snapshot_every = snapshot_every;
    serve::Scheduler sched(eng, tok, sc);
    std::fprintf(stderr, "eklow-server: chat template %s, sampling defaults temperature %g, top-k %d, top-p %g, min-p %g; snapshots of %.0f MiB each, up to %lld MiB\n",
                 text::qwen36_template_name(tmpl), double(sampling.temperature), int(sampling.top_k), double(sampling.top_p), double(sampling.min_p),
                 double(eng.snapshot_bytes()) / 1048576.0, (long long)snapshot_mib);
    if (embed_images) embed_images->bind(&sched);
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
    serve::ChatDefaults defaults{sampling, ec.max_ctx, tmpl};

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
      r->tool_specs = cr.tool_specs;
      r->snapshot_at = cr.snapshot_at;
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
      serve_embeddings(embed_http, *embed_model, *embed_tok, embed_mu, sched, embed_alias, 8192, embed_stats, embed_images.get());
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
