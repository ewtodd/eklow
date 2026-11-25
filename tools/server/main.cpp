// eklow-server: Gemma 4 E4B (NVIDIA's Nemotron-3.5-Bolt-E4B, the file NVIDIA ships in Q4_K_M) on the A770M behind an
// OpenAI-compatible HTTP API, with the other models of the card beside it on ports of their own: EmbeddingGemma 2, vega's
// end-of-turn checker and Qwen3-Reranker. A GGUF of any other architecture is refused (the Qwen3.6-35B-A3B server left on
// 2026-11-11; DECISIONS says where it is in the history). Bolt can run a DFlash draft (--draft: NVIDIA's
// Nemotron-3.5-Bolt-E4B-DFlash), accepted losslessly.
//
//   eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N]
//                [--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--preserve-thinking on|off]
//                [--mmproj GGUF] [--embed-model GGUF --embed-port N [--embed-alias NAME] [--embed-mmproj GGUF]] [--image-tokens N]
//                [--image-max-tokens N]
//                [--draft GGUF [--draft-block B] [--draft-full-context N] [--draft-sliding-causal -1|0|1]]
//                [--checker-reader GGUF --checker-vega DIR [--checker-policy DIR] --checker-port N [--checker-alias NAME] [--checker-max-tokens N]]
//                [--rerank-model GGUF --rerank-port N [--rerank-alias NAME] [--rerank-max-tokens N]] [--log-interval S] [--keepalive S]
//                [--prefix-cache DIR --prefix-cache-key FILE [--prefix-cache-gib N] [--prefix-cache-buffers N]]
//
// --ctx is the context of each slot (positions), Bolt's full 65,536 by default; --batch the tokens of one engine step
// and --chunk the prefill tokens of one sequence in a step (2,048). The whole model fits on the card (5.7 GiB with the
// per-layer table), and the K/V of two slots of 65,536 is 2 GiB in fp16 (only the 24 layers before the shared ones keep
// any, 20 of them sliding rings of window + chunk), so fp16 is the default; --kv-int8 halves the global layers'.
// --preserve-thinking on renders earlier assistant turns that made tool calls with their reasoning (the template's
// preserve_thinking, for a request that sends none), so the next turn shares the prompt the slot holds.
// --draft (a DFlash draft GGUF) turns on speculative decoding: a verify block of --draft-block tokens (default the
// draft's trained block, 2 to 16) per step, accepted losslessly (the sampler's every feature, the tool grammar
// included); a request may choose its own with "speculative": N (0 or false: plain decode). The draft shares the engine's
// head on the matrix units and reads the token table from the GGUF's mapping.
// --mmproj (a Gemma 4 projector GGUF, or Inkling 2's inkling2v) lets chat requests carry images (OpenAI image_url parts
// with base64 data: URLs), --embed-mmproj (EmbeddingGemma 2's) lets an embeddings input be an array of text and
// image_url parts; --image-tokens is the Gemma towers' budget of rows per image (llama.cpp's image_max_tokens, 70 to
// 1,120, 280 by default), --image-max-tokens Inkling's (1 to 4,096, 1,024 by default: a larger image is resized down, and
// the encoder's activations for it stay on the card). Each image is encoded on its request's thread with the scheduler
// held, one at a time per encoder, and counts in the prompt tokens.
// Endpoints: POST /v1/chat/completions (streamed or whole), GET /v1/models, GET /health, GET /metrics (Prometheus text, for llmtop; no
// auth; tools/server/metrics.hpp, K/V use is the global layers' pages); with an embedding model, POST /v1/embeddings, GET /v1/models
// and GET /health on its own port (the house runs chat and embeddings as separate upstreams). The checker (--checker-*: POST
// /v1/systemone, tools/server/systemone.hpp) and the reranker (--rerank-*: POST /v1/rerank, tools/server/rerank.hpp) are each on
// a port and queue of their own, with GET /v1/models and GET /health, a request at a time with the scheduler held, as the
// embeddings are.
// --prefix-cache keeps the K/V of prompts and replies on disk, page by page (serve/prefix_store.hpp), AES-256-GCM sealed under
// the key in --prefix-cache-key (32 bytes, or 64 hex digits) and at most --prefix-cache-gib GiB (200), so that a conversation
// whose slot went to another one is read back instead of prefilled. --prefix-cache-buffers is the pinned host buffers (4 MiB
// each for Bolt, 48 by default) that pages take to the disk; fewer make pages wait in their slot sooner (the checks use 2). A
// chat request with max_tokens 0 is prefill-only: its prompt goes into a slot and the disk, nothing is generated, and it
// waits for the engine to be otherwise idle.
// The log (stdout, unholy's form, each line prefixed with the local time): a line per chat request as it ends
// (finish reason; prompt, cached and completion tokens; queue, time to first token, prefill and decode tok/s), a
// line per embedding request, a line per rejected request, and every --log-interval seconds (10; 0 for none) while
// there is traffic the engine's rates, running and waiting requests, the slots' positions and the prefix hit rate.
#include <httplib.h>

#include "keepalive.hpp"
#include "metrics.hpp"
#include "models.hpp"
#include "rerank.hpp"
#include "systemone.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine_gemma4.hpp"
#include "engine/engine_inkling.hpp"
#include "image/gemma4_image.hpp"
#include "model/dflash2.hpp"
#include "model/embed_gemma2.hpp"
#include "model/gguf.hpp"
#include "model/vision_gemma4v.hpp"
#include "model/vision_inkling2v.hpp"
#include "image/inkling_image.hpp"
#include "rt/device.hpp"
#include "serve/openai_gemma4.hpp"
#include "serve/openai_inkling.hpp"
#include "serve/scheduler_gemma4.hpp"
#include "serve/scheduler_inkling.hpp"
#include "text/chat_gemma4.hpp"
#include "text/chat_inkling.hpp"
#include "text/tokenizer.hpp"
#include "text/tokenizer_inkling.hpp"


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
std::string request_line(const serve::gemma4::RequestLog& l) {
  std::string out = "request " + l.tag + ": " + (l.prefill_only ? "prefill-only " : "") + l.reason;
  if (!l.error.empty()) out += " (" + l.error + ")";
  out += fmt(" | prompt %lld", (long long)l.prompt_tokens);
  std::string notes;
  if (l.cached_tokens > 0) notes += fmt("%lld cached", (long long)l.cached_tokens);
  if (l.restored_tokens > 0) notes += fmt(", %lld of them from the disk", (long long)l.restored_tokens);
  if (l.image_tokens > 0) notes += (notes.empty() ? "" : ", ") + fmt("%lld image", (long long)l.image_tokens);
  if (!notes.empty()) out += " (" + notes + ")";
  if (l.prefill_only) {
    out += fmt(", %lld in the slot | queue %.2f s", (long long)l.held, l.queue_s);
    if (l.prefill_s > 0 && l.held > l.cached_tokens) out += fmt(", prefill %.1f tok/s", double(l.held - l.cached_tokens) / l.prefill_s);
    return out;
  }
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
// no request, kept for prefix reuse), the share of admitted prompt tokens already in their slot, and embeddings.
std::string engine_line(const serve::gemma4::Scheduler::Stats& a, const serve::gemma4::Scheduler::Stats& b, const EmbedCounts& ea, const EmbedCounts& eb,
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
    out += fmt(" | per step: engine %.1f ms, draft %.1f, sample %.1f, inject %.1f", 1e3 * (b.engine_s - a.engine_s) / double(b.steps - a.steps),
               1e3 * (b.draft_s - a.draft_s) / double(b.steps - a.steps), 1e3 * (b.sample_s - a.sample_s) / double(b.steps - a.steps),
               1e3 * (b.inject_s - a.inject_s) / double(b.steps - a.steps));
  if (eb.inputs > ea.inputs)
    out += fmt(" | embeddings %lld requests, %lld inputs, %.1f tok/s", (long long)(eb.requests - ea.requests), (long long)(eb.inputs - ea.inputs),
               double(eb.tokens - ea.tokens) / dt);
  return out;
}

// What names a file in the disk cache's keys: its resolved path (a Nix store path names its content), its size, and a hash of its
// first and last MiB (a file replaced in place under the same name).
std::string file_identity(const std::string& path) {
  std::error_code ec;
  const std::filesystem::path real = std::filesystem::canonical(path, ec);
  const uint64_t size = std::filesystem::file_size(path, ec);
  std::ifstream f(path, std::ios::binary);
  std::string ends(size_t(std::min<uint64_t>(size, 1 << 20)), '\0');
  f.read(ends.data(), std::streamsize(ends.size()));
  if (size > (1u << 21)) {
    std::string tail(1 << 20, '\0');
    f.seekg(std::streamoff(size - (1u << 20)));
    f.read(tail.data(), std::streamsize(tail.size()));
    ends += tail;
  }
  return real.string() + " " + std::to_string(size) + " " + serve::chain_root(ends).hex();
}

void reply_error(httplib::Response& res, const serve::gemma4::ApiError& e) {
  log_line(fmt("error %d (%s): ", e.status, e.type.c_str()) + e.what());
  res.status = e.status;
  res.set_content(serve::gemma4::error_json(e), "application/json");
}

// Images to rows for the text model: decoded and cut on the host, encoded on the card with the scheduler held (the
// encoder runs for seconds, and every wait on the card costs a turn behind a long prefill: DECISIONS 2026-10-26) and
// one at a time, the rows copied to host memory. A few encoded images are kept by the hash of their bytes: a
// conversation sends its images again every turn, and the rows are the same.
class ImageService {
 public:
  // Gemma 4's tower (Bolt's, EmbeddingGemma 2's): tokens is the soft-token budget.
  ImageService(sycl::queue& q, model::VisionGemma4V& v, int tokens) : tokens_(tokens), n_embd_(v.n_embd()) {
    prepare_ = [&v, tokens](const image::Rgb& rgb) { return image::prepare(rgb, v.spec(tokens)); };
    encode_ = [&q, &v](const image::Patches& p) {
      const model::DeviceRows rows = v.encode(p);
      auto host = std::make_shared<std::vector<float>>(size_t(rows.n * rows.dim));
      q.memcpy(host->data(), rows.data.get(), host->size() * sizeof(float)).wait();
      return host;
    };
  }
  // Inkling 2's tower: tokens is the most an image may take (a larger one is resized down, as mtmd's
  // image_max_tokens).
  ImageService(model::VisionInkling2V& v, int tokens) : tokens_(tokens), n_embd_(v.n_out()) {
    prepare_ = [tokens](const image::Rgb& rgb) { return image::prepare_inkling(rgb, tokens); };
    encode_ = [&v](const image::Patches& p) { return std::make_shared<std::vector<float>>(v.encode(p)); };
  }
  void bind(std::function<std::shared_ptr<void>()> hold) { hold_ = std::move(hold); }
  int64_t n_embd() const { return n_embd_; }

  serve::gemma4::EncodedImage operator()(std::span<const uint8_t> bytes) {
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
      patches = prepare_(image::decode(bytes));
    } catch (const std::invalid_argument& e) {
      throw serve::gemma4::ApiError(400, "invalid_request_error", "unsupported_content", std::string("image_url: ") + e.what());
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::lock_guard lk(enc_mu_);
    std::shared_ptr<void> hold;
    if (hold_) hold = hold_();
    const auto t1 = std::chrono::steady_clock::now();
    std::shared_ptr<std::vector<float>> host;
    try {
      host = encode_(patches);
    } catch (const std::invalid_argument& e) {
      throw serve::gemma4::ApiError(400, "invalid_request_error", "unsupported_content", std::string("image_url: ") + e.what());
    }
    const int64_t n = int64_t(host->size()) / n_embd_;
    hold.reset();  // the rows are on the host: the scheduler goes on while they are checked and cached
    size_t bad = 0;
    for (float x : *host) bad += !std::isfinite(x);
    if (bad) log_line(fmt("image: %zu of %zu row values are not finite", bad, host->size()));
    log_line(fmt("image: %dx%d patches, %lld rows | wait %.2f s, encoded in %.2f s", patches.cols, patches.rows, (long long)n,
                 seconds(t1 - t0), seconds(std::chrono::steady_clock::now() - t1)));
    {
      std::lock_guard ck(cache_mu_);
      cache_.push_front({h, int64_t(bytes.size()), n, host});
      if (cache_.size() > 6) cache_.pop_back();
    }
    return {h, n, host};
  }

 private:
  struct Entry {
    uint64_t hash;
    int64_t n_bytes, n;
    std::shared_ptr<const std::vector<float>> rows;
  };
  int tokens_;
  int64_t n_embd_;
  std::function<image::Patches(const image::Rgb&)> prepare_;
  std::function<std::shared_ptr<std::vector<float>>(const image::Patches&)> encode_;
  std::function<std::shared_ptr<void>()> hold_;
  std::mutex enc_mu_, cache_mu_;
  std::list<Entry> cache_;
};

// The embedding endpoint: inputs one at a time through the model (each its own sequence,
// attended both ways), each at most max_tokens long, with the scheduler held so that no new chat
// step starts until they are done. Each request is logged (its wait for the model, behind another request,
// and its time) and counted in `stats`.
void serve_embeddings(httplib::Server& http, model::EmbedGemma2& m, const text::Tokenizer& tok, std::mutex& mu,
                      std::function<std::shared_ptr<void>()> hold, const std::string& alias, int64_t max_tokens, EmbedStats& stats, ImageService* images) {
  http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
  serve::serve_models(http, alias, max_tokens, 1);
  http.Post("/v1/embeddings", [&m, &tok, &mu, hold, &stats, alias, max_tokens, images](const httplib::Request& req, httplib::Response& res) {
    try {
      json::Value body;
      try {
        body = json::parse(req.body);
      } catch (const std::exception& e) {
        throw serve::gemma4::ApiError(400, "invalid_request_error", "", std::string("the body is not JSON: ") + e.what());
      }
      const json::Value* input = body.is_object() ? body.find("input") : nullptr;
      if (!input) throw serve::gemma4::ApiError(400, "invalid_request_error", "", "input is required");
      // A string, strings, token ids, arrays of token ids, or arrays of chat-style content parts (text and image_url),
      // which as llama.cpp's server tokenizes a prompt with media markers are BOS, the text before each image, <|image>,
      // the image's rows, <image|>, the text after, EOS (text parts joined with nothing between them, as it does). Also as
      // its server takes them: an item {"content": [parts]}. A top-level array of parts is one input.
      std::vector<std::vector<int32_t>> seqs;
      std::vector<std::vector<model::ImageRows>> spans;  // per input
      std::vector<serve::gemma4::EncodedImage> encoded;          // the rows the spans point into
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
          if (!v.is_int()) throw serve::gemma4::ApiError(400, "invalid_request_error", "", "token ids must be integers");
          ids.push_back(int32_t(std::stoll(v.int_digits())));
        }
        return ids;
      };
      auto bad = [](const std::string& m) { return serve::gemma4::ApiError(400, "invalid_request_error", "", m); };
      auto parts_of = [&](const json::Value& a) {
        std::vector<int32_t> ids{tok.bos()};
        std::vector<model::ImageRows> sp;
        std::vector<serve::gemma4::EncodedImage> mine;
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
              throw serve::gemma4::ApiError(400, "invalid_request_error", "unsupported_content",
                                    "image_url is not supported: this server was started without a vision projector for the embedding model (--embed-mmproj)");
            const json::Value* iu = part.find("image_url");
            const json::Value* url = iu && iu->is_object() ? iu->find("url") : (iu && iu->is_string() ? iu : nullptr);
            if (!url || !url->is_string()) throw bad("an image_url part needs image_url.url");
            const int32_t open = tok.id("<|image>"), close = tok.id("<image|>");
            if (open < 0 || close < 0) throw bad("the model's vocabulary has no image tokens");
            const std::vector<uint8_t> bytes = serve::gemma4::image_bytes(url->as_string());
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
          else throw serve::gemma4::ApiError(400, "invalid_request_error", "", "input must hold strings, token ids or content parts");
        }
      else throw serve::gemma4::ApiError(400, "invalid_request_error", "", "input must be a string or an array");
      spans.resize(seqs.size());
      if (seqs.empty()) throw serve::gemma4::ApiError(400, "invalid_request_error", "", "input is empty");
      int64_t total = 0;
      for (const auto& q : seqs) {
        if (int64_t(q.size()) > max_tokens)
          throw serve::gemma4::ApiError(400, "invalid_request_error", "",
                                "input is too large to process: " + std::to_string(q.size()) + " tokens, at most " + std::to_string(max_tokens));
        for (int32_t id : q)
          if (id < 0 || size_t(id) >= tok.size()) throw serve::gemma4::ApiError(400, "invalid_request_error", "", "token id out of range");
        total += int64_t(q.size());
      }
      json::Writer w(false);
      w.begin_object().kv("object", "list").kv("model", alias).key("data").begin_array();
      std::vector<std::vector<float>> embs;
      const auto t0 = std::chrono::steady_clock::now();
      std::chrono::steady_clock::time_point t1;
      {
        std::lock_guard lk(mu);
        const std::shared_ptr<void> held = hold();
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
    } catch (const serve::gemma4::ApiError& e) {
      reply_error(res, e);
    } catch (const std::exception& e) {
      reply_error(res, serve::gemma4::ApiError(500, "server_error", "", e.what()));
    }
  });
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, alias = "nemotron-3.5-bolt-e4b", host = "127.0.0.1", embed_path, embed_alias = "embeddinggemma-2", mmproj_path, embed_mmproj_path;
  int image_tokens = 280;
  int image_max_tokens = 1024;  // Inkling 2's chat images (--image-max-tokens, as llama.cpp's)
  std::string draft_path;
  int draft_block = 0;
  int64_t draft_full_context = 0;  // a plain draft's full-attention layer: positions per slot it keeps (0: the draft code's 32,768)
  int draft_sliding_causal = -1;   // -1 the draft's own, 0 or 1: to measure the difference (Dflash2::Options::sliding_causal)
  int port = 8081, slots = 2, embed_port = 8082;
  int64_t ctx = 65536, batch = 0, chunk = 0;  // 0: the model's default (Bolt 2,048, Inkling 1,024)
  // Inkling's experts (engine/engine_inkling.hpp): the routing profile that ranks them for the card, the card memory left
  // free after them, and the CPU pool that runs the cold ones.
  std::string expert_profile;
  int64_t card_margin_mib = 1024;
  int cpu_threads = 8, checkpoints = 2;
  int64_t cpu_max_tokens = 128;
  // Inkling: tokens a step may carry, run layer by layer over sub-batches of --batch. 8,192: each layer application's
  // cold set is copied to the card once a step, and the copies cost the card ~61 ms a GB, so longer steps copy less a
  // token (prefill at 12k 1,032 -> 1,087 tok/s, decode 22.2 -> 21.9: the step's buffers take ~90 experts off the card;
  // DECISIONS 2026-11-25).
  int64_t step_tokens = 8192;
  int64_t kv_pool = 0;         // Inkling: the full layers' K/V positions the slots share (0: --ctx, one slot's room in all)
  double loopcd = 0;  // Inkling: LoopCD's contrast strength (0: off)
  bool kv_int8 = false, preserve_thinking = false;
  double log_interval = 10;  // seconds between the engine's log lines while there is traffic; 0: none
  double keepalive_s = 15;  // a stream silent this long gets an SSE comment (keepalive.hpp); 0: never
  std::string checker_reader, checker_vega, checker_policy, checker_alias = "vega-u";
  int checker_port = 0;
  int64_t checker_max_tokens = 16384;
  std::string rerank_path, rerank_alias = "qwen3-reranker-0.6b";
  int rerank_port = 0;
  int64_t rerank_max_tokens = 4096;
  std::string prefix_dir, prefix_key;
  int64_t prefix_gib = 200, prefix_buffers = 48;
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
    else if (a == "--embed-model") embed_path = next();
    else if (a == "--embed-port") embed_port = std::atoi(next().c_str());
    else if (a == "--embed-alias") embed_alias = next();
    else if (a == "--mmproj") mmproj_path = next();
    else if (a == "--embed-mmproj") embed_mmproj_path = next();
    else if (a == "--image-tokens") image_tokens = std::atoi(next().c_str());
    else if (a == "--image-max-tokens") image_max_tokens = std::atoi(next().c_str());
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--kv-fp16") kv_int8 = false;
    else if (a == "--preserve-thinking") {
      const std::string v = next();
      if (v != "on" && v != "off") {
        std::fprintf(stderr, "eklow-server: --preserve-thinking takes on or off\n");
        return 2;
      }
      preserve_thinking = v == "on";
    }
    else if (a == "--log-interval") log_interval = std::atof(next().c_str());
    else if (a == "--keepalive") keepalive_s = std::atof(next().c_str());
    else if (a == "--draft") draft_path = next();
    else if (a == "--draft-block") draft_block = std::atoi(next().c_str());
    else if (a == "--draft-full-context") draft_full_context = std::atoll(next().c_str());
    else if (a == "--draft-sliding-causal") draft_sliding_causal = std::atoi(next().c_str());
    else if (a == "--checker-reader") checker_reader = next();
    else if (a == "--checker-vega") checker_vega = next();
    else if (a == "--checker-policy") checker_policy = next();
    else if (a == "--checker-port") checker_port = std::atoi(next().c_str());
    else if (a == "--checker-alias") checker_alias = next();
    else if (a == "--checker-max-tokens") checker_max_tokens = std::atoll(next().c_str());
    else if (a == "--rerank-model") rerank_path = next();
    else if (a == "--rerank-port") rerank_port = std::atoi(next().c_str());
    else if (a == "--rerank-alias") rerank_alias = next();
    else if (a == "--rerank-max-tokens") rerank_max_tokens = std::atoll(next().c_str());
    else if (a == "--prefix-cache") prefix_dir = next();
    else if (a == "--prefix-cache-key") prefix_key = next();
    else if (a == "--prefix-cache-gib") prefix_gib = std::atoll(next().c_str());
    else if (a == "--prefix-cache-buffers") prefix_buffers = std::atoll(next().c_str());
    else if (a == "--expert-profile") expert_profile = next();
    else if (a == "--card-margin-mib") card_margin_mib = std::atoll(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
    else if (a == "--cpu-max-tokens") cpu_max_tokens = std::atoll(next().c_str());
    else if (a == "--checkpoints") checkpoints = std::atoi(next().c_str());
    else if (a == "--step-tokens") step_tokens = std::atoll(next().c_str());
    else if (a == "--kv-pool") kv_pool = std::atoll(next().c_str());
    else if (a == "--loopcd") loopcd = std::atof(next().c_str());
    else if (a == "--chat-template" || a == "--snapshot-mib" || a == "--snapshot-every" || a == "--group" || a == "--group-busy" ||
               a == "--image-rows-dir" || a == "--image-max-tokens" || a == "--vision-reserve-mib" ||
               a == "--experts-on-card" || a == "--verification" || a == "--draft-reserve-mib") {
      std::fprintf(stderr, "eklow-server: %s was the Qwen3.6-35B-A3B server's, which eklow no longer has (2026-11-11)\n", a.c_str());
      return 2;
    } else {
      std::fprintf(stderr, "usage: eklow-server --model GGUF [--alias NAME] [--host ADDR] [--port N] [--slots N] [--ctx N] "
                           "[--batch N] [--chunk N] [--kv-int8 | --kv-fp16] [--preserve-thinking on|off] [--mmproj GGUF] [--embed-model GGUF --embed-port N [--embed-alias NAME] [--embed-mmproj GGUF]] "
                           "[--draft GGUF [--draft-block B] [--draft-full-context N] [--draft-sliding-causal -1|0|1]] [--image-tokens N] [--image-max-tokens N] "
                           "[--checker-reader GGUF --checker-vega DIR [--checker-policy DIR] --checker-port N [--checker-alias NAME] [--checker-max-tokens N]] "
                           "[--rerank-model GGUF --rerank-port N [--rerank-alias NAME] [--rerank-max-tokens N]] [--log-interval S] [--keepalive S] "
                           "[--prefix-cache DIR --prefix-cache-key FILE [--prefix-cache-gib N] [--prefix-cache-buffers N]] "
                           "[--expert-profile JSON] [--card-margin-mib N] [--cpu-threads N] [--cpu-max-tokens N] [--checkpoints N] [--step-tokens N] [--kv-pool N] [--loopcd W]\n");
      return 2;
    }
  }
  if (model_path.empty()) {
    std::fprintf(stderr, "eklow-server: --model is required\n");
    return 2;
  }
  if (draft_block < 0 || draft_block > int(engine::gemma4::kMaxVerify) || draft_block == 1) {
    std::fprintf(stderr, "eklow-server: --draft-block must be 2 to %d\n", int(engine::gemma4::kMaxVerify));
    return 2;
  }
  if (draft_path.empty() && draft_block) {
    std::fprintf(stderr, "eklow-server: --draft-block needs --draft\n");
    return 2;
  }
  if (!checker_reader.empty() && (checker_vega.empty() || checker_port <= 0 || checker_max_tokens < 1024)) {
    std::fprintf(stderr, "eklow-server: --checker-reader needs --checker-vega, --checker-port and a --checker-max-tokens of 1024 or more\n");
    return 2;
  }
  if (!prefix_dir.empty() && (prefix_key.empty() || prefix_gib < 1 || prefix_buffers < 1)) {
    std::fprintf(stderr, "eklow-server: --prefix-cache needs --prefix-cache-key (the cache holds conversations: it is always encrypted), a --prefix-cache-gib "
                         "and a --prefix-cache-buffers of 1 or more\n");
    return 2;
  }
  if (!rerank_path.empty() && (rerank_port <= 0 || rerank_max_tokens < 256)) {
    std::fprintf(stderr, "eklow-server: --rerank-model needs --rerank-port and a --rerank-max-tokens of 256 or more\n");
    return 2;
  }
  if (image_tokens < 70 || image_tokens > 1120) {
    std::fprintf(stderr, "eklow-server: --image-tokens must be 70 to 1120\n");
    return 2;
  }
  if (image_max_tokens < 1 || image_max_tokens > 4096) {
    std::fprintf(stderr, "eklow-server: --image-max-tokens must be 1 to 4096\n");
    return 2;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    auto file = gguf::File::open(model_path);
    const gguf::Value* arch_v = file.find("general.architecture");
    const std::string arch = arch_v ? arch_v->as_string() : std::string("(none)");
    const bool inkling = arch == "inkling2_loop";
    if (arch != "gemma4" && !inkling)
      throw std::runtime_error(model_path + ": architecture " + arch +
                               "; eklow serves gemma4 (Gemma 4 E4B: Nemotron-3.5-Bolt-E4B) and inkling2_loop (Inkling 2 Mini), and no longer qwen35moe (2026-11-11)");
    if (batch <= 0) batch = inkling ? 1024 : 2048;
    if (chunk <= 0) chunk = inkling ? 1024 : 2048;
    const int64_t max_ctx = (ctx + attn::kPage - 1) / attn::kPage * attn::kPage;
    // The template is ported, not interpreted: a file carrying another one is refused rather than rendered wrong.
    const gguf::Value* tmpl = file.find("tokenizer.chat_template");
    std::optional<text::Gemma4Template> which;
    std::unique_ptr<text::Tokenizer> gtok;
    std::unique_ptr<text::InklingTokenizer> itok;
    if (inkling) {
      // M4 of the port (DECISIONS 2026-11-24): text and tools. Its vision is M5 and its MTP draft M6; it keeps no disk prefix
      // cache (its sliding rings and convolution states are rewound from checkpoints, engine_inkling.hpp).
      if (!draft_path.empty() || !prefix_dir.empty() || kv_int8 || preserve_thinking)
        throw std::runtime_error("Inkling 2 Mini is served without --draft, --prefix-cache, --kv-int8 and --preserve-thinking");
      if (!tmpl || !text::is_inkling_template(tmpl->as_string()))
        throw std::runtime_error(model_path + ": its chat template is not Inkling 2 Mini's, which eklow ports");
      if (const gguf::Value* n = file.find("inkling2_loop.context_length"); n && ctx > n->as_int())
        throw std::runtime_error("--ctx " + std::to_string(ctx) + " is beyond the model's context of " + std::to_string(n->as_int()));
      if (expert_profile.empty())
        throw std::runtime_error("Inkling 2 Mini needs --expert-profile (data/inkling-2-mini-experts.json): its experts do not all fit the card");
      itok = std::make_unique<text::InklingTokenizer>(text::InklingTokenizer::from_gguf(file));
    } else {
      which = tmpl ? text::gemma4_template_of(tmpl->as_string()) : std::nullopt;
      if (!which) throw std::runtime_error(model_path + ": its chat template is neither Bolt's nor Google's Gemma 4 template, which eklow ports");
      // Bolt was trained to 65,536 positions; beyond them its RoPE is untested, so a larger --ctx is refused, not served.
      if (const gguf::Value* n = file.find("gemma4.context_length"); n && ctx > n->as_int())
        throw std::runtime_error("--ctx " + std::to_string(ctx) + " is beyond the model's context of " + std::to_string(n->as_int()));
      if (!mmproj_path.empty() && (chunk < image_tokens || batch < image_tokens + slots))
        // An image's rows go in one chunk (its sliding layers see all of it at once), and a step holds the decode tokens too.
        throw std::runtime_error("--image-tokens " + std::to_string(image_tokens) + " needs --chunk >= " + std::to_string(image_tokens) + " and --batch >= " +
                                 std::to_string(image_tokens + slots));
      gtok = std::make_unique<text::Tokenizer>(text::Tokenizer::from_gguf(file));
    }
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::gemma4::Config ec;
    ec.n_slots = slots;
    ec.max_ctx = max_ctx;
    ec.max_batch = batch;
    ec.max_chunk = chunk;
    ec.max_logits = slots;
    ec.int8_global_kv = kv_int8;
    std::unique_ptr<gguf::File> draft_file;
    model::Dflash2Config draft_cfg{};
    if (!draft_path.empty()) {
      draft_file = std::make_unique<gguf::File>(gguf::File::open(draft_path));
      draft_cfg = model::Dflash2Config::from_gguf(*draft_file);
      const model::Gemma4Config mcfg = model::Gemma4Config::from_gguf(file);
      if (draft_cfg.n_embd != mcfg.n_embd || draft_cfg.n_vocab != mcfg.n_vocab)
        throw std::runtime_error(draft_path + ": the draft is for a target of hidden " + std::to_string(draft_cfg.n_embd) + " and vocabulary " +
                                 std::to_string(draft_cfg.n_vocab) + ", not this model's " + std::to_string(mcfg.n_embd) + " and " + std::to_string(mcfg.n_vocab));
      // The draft's trained block (8 for Bolt's): the model is dense, so a verify row past the first costs little. Bolt at
      // 8k / 32k on prose: 96.7 / 80.9 tok/s at 8 against 77.4 / 64.3 at 4.
      if (draft_block == 0) draft_block = std::min(draft_cfg.block_size, int(engine::gemma4::kMaxVerify));
      // The draft reads the target's hidden states at its layers; a step captures the rows of its verify blocks and of the
      // last `window` rows of a prompt. The head is the engine's, shared with the draft.
      ec.capture_layers = draft_cfg.target_layers;
      ec.max_capture_rows = std::min<int64_t>(chunk, draft_cfg.window) + int64_t(slots) * engine::gemma4::kMaxVerify;
      ec.max_logits = int64_t(slots) * (engine::gemma4::kMaxVerify + 1);  // a verify block of up to 16 rows each
    }
    // The embedding model: its own queue on the card, so its work does not wait behind steps. It goes on
    // first, with room for its longest input.
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
    // The vision encoders, each on a queue of its own and loaded before the engine, so that their
    // weights are counted in what is left of the card. An encode's activations are allocated and freed per image.
    std::unique_ptr<gguf::File> vision_file, embed_vision_file;
    std::unique_ptr<sycl::queue> vision_q, embed_vision_q;
    std::unique_ptr<model::VisionGemma4V> vision, embed_vision;
    std::unique_ptr<model::VisionInkling2V> ivision;
    std::unique_ptr<ImageService> images, embed_images;
    if (!mmproj_path.empty() && inkling) {
      vision_file = std::make_unique<gguf::File>(gguf::File::open(mmproj_path));
      vision_q = std::make_unique<sycl::queue>(q.get_device(), sycl::property::queue::in_order());
      // The weights in pinned host memory, copied in a layer at a time per encode, and the activations in the engine's
      // step buffers (idle while the scheduler is held): the experts keep the card (DECISIONS 2026-11-25, "Inkling M5").
      ivision = std::make_unique<model::VisionInkling2V>(*vision_q, *vision_file, true);
      std::fprintf(stderr, "eklow-server: vision projector %s, %.0f MiB on the card and %.0f MiB of weights in host memory, images of up to %d tokens\n",
                   mmproj_path.c_str(), double(ivision->bytes()) / 1048576.0, double(ivision->host_bytes()) / 1048576.0, image_max_tokens);
      images = std::make_unique<ImageService>(*ivision, image_max_tokens);
    } else if (!mmproj_path.empty()) {
      vision_file = std::make_unique<gguf::File>(gguf::File::open(mmproj_path));
      vision_q = std::make_unique<sycl::queue>(q.get_device(), sycl::property::queue::in_order());
      vision = std::make_unique<model::VisionGemma4V>(*vision_q, *vision_file);
      // The encoder's activations for the budget, held for good: reserved at startup so that they are
      // counted, and no allocation at request time (~280 MiB at 280 rows, DECISIONS 2026-10-27 vision); the rows an
      // encode returns are the caller's and fit the card margin.
      vision->reserve(image_tokens);
      std::fprintf(stderr, "eklow-server: vision projector %s, %.0f MiB on the card for %d-row images\n", mmproj_path.c_str(), double(vision->bytes()) / 1048576.0, image_tokens);
      images = std::make_unique<ImageService>(*vision_q, *vision, image_tokens);
    }
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
    // vega's checker and the reranker, each on a queue of its own and warmed before the engine, so that their weights,
    // activations and the driver's buffers for their kernels are counted in what is left of the card.
    std::unique_ptr<sycl::queue> checker_q;
    std::unique_ptr<vega::Checker> checker;
    std::mutex checker_mu;
    if (!checker_reader.empty()) {
      const auto t_ck = std::chrono::steady_clock::now();
      const double free0 = double(rt::free_memory(q.get_device()).value_or(0));
      checker_q = std::make_unique<sycl::queue>(q.get_device(), sycl::property::queue::in_order());
      checker = std::make_unique<vega::Checker>(*checker_q, checker_reader, checker_vega, checker_policy, checker_max_tokens);
      checker->warm();
      std::fprintf(stderr, "eklow-server: checker %s (%s, policy %s): %.0f MiB on the card with the driver's, prompts of up to %lld tokens, %.1f s\n",
                   checker_alias.c_str(), checker_reader.c_str(), checker->policy() ? checker->policy()->verified().c_str() : "none",
                   (free0 - double(rt::free_memory(q.get_device()).value_or(0))) / 1048576.0, (long long)checker_max_tokens,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ck).count());
    }
    std::unique_ptr<gguf::File> rerank_file;
    std::unique_ptr<sycl::queue> rerank_q;
    std::unique_ptr<model::RerankQwen3> reranker;
    std::mutex rerank_mu;
    if (!rerank_path.empty()) {
      const auto t_rr = std::chrono::steady_clock::now();
      const double free0 = double(rt::free_memory(q.get_device()).value_or(0));
      rerank_file = std::make_unique<gguf::File>(gguf::File::open(rerank_path));
      rerank_q = std::make_unique<sycl::queue>(q.get_device(), sycl::property::queue::in_order());
      reranker = std::make_unique<model::RerankQwen3>(*rerank_q, *rerank_file, rerank_max_tokens);
      reranker->score({reranker->prompt("A warm-up.", "A warm-up.")});
      std::fprintf(stderr, "eklow-server: reranker %s (%s): %.0f MiB on the card with the driver's, pairs of up to %lld tokens, %.1f s\n",
                   rerank_alias.c_str(), rerank_path.c_str(), (free0 - double(rt::free_memory(q.get_device()).value_or(0))) / 1048576.0,
                   (long long)rerank_max_tokens, std::chrono::duration<double>(std::chrono::steady_clock::now() - t_rr).count());
    }
    const double GiB = 1073741824.0;
    // What the rest of the server sees of the chat model: its scheduler's counters, a hold that keeps it from starting steps
    // while a side model has the card, and its engine thread.
    std::function<serve::gemma4::Scheduler::Stats()> chat_stats;
    std::function<std::shared_ptr<void>()> hold;
    std::function<void()> run_engine;
    std::unique_ptr<engine::gemma4::Engine> geng;
    std::unique_ptr<model::Dflash2> draft;
    std::unique_ptr<serve::PrefixStore> store;
    std::unique_ptr<serve::gemma4::Scheduler> gsched;
    std::unique_ptr<engine::inkling::Engine> ieng;
    std::unique_ptr<serve::inkling::Scheduler> isched;
    if (inkling) {
      // The side models are on the card already, so the experts' placement sees what they left (DECISIONS 2026-11-24, M4).
      engine::inkling::Config ic;
      ic.n_slots = slots;
      ic.max_ctx = max_ctx;
      ic.max_batch = batch;
      ic.max_chunk = chunk;
      ic.max_logits = slots;
      ic.expert_profile = expert_profile;
      ic.card_margin_bytes = uint64_t(std::max<int64_t>(0, card_margin_mib)) << 20;
      ic.cpu_threads = cpu_threads;
      ic.cpu_max_tokens = cpu_max_tokens;
      ic.checkpoints = checkpoints;
      ic.max_step = std::max(step_tokens, batch);
      // One K/V pool for the slots (the owner's choice, 2026-11-24): by default a slot's room in all, so that two long
      // conversations share it and the card's memory goes to experts (DECISIONS "Inkling speed: the shared pool ...").
      ic.pool_positions = (kv_pool > 0 ? (kv_pool + attn::kPage - 1) / attn::kPage * attn::kPage : max_ctx);
      ic.loopcd = float(loopcd);
      ieng = std::make_unique<engine::inkling::Engine>(q, file, ic);
      if (ivision) {
        if (ivision->n_out() != ieng->model().n_embd)
          throw std::runtime_error(mmproj_path + ": projects to " + std::to_string(ivision->n_out()) + " but the model's width is " +
                                   std::to_string(ieng->model().n_embd));
        ivision->use_workspace(ieng->idle_workspace(), [e = ieng.get()] { e->drain(); });
        if (!ivision->fits(image_max_tokens))
          throw std::runtime_error("--image-max-tokens " + std::to_string(image_max_tokens) + ": the encoder's activations for an image that large do not fit the engine's step buffers at --step-tokens " +
                                   std::to_string(ic.max_step) + "; lower it");
      }
      std::fprintf(stderr, "eklow-server: %s on %s, %.2f GiB of weights on the card, %lld cold experts (%.2f GiB) on the CPU with %d threads, %.2f GiB of caches (%d slots of %lld), %.1f s\n",
                   alias.c_str(), q.get_device().get_info<sycl::info::device::name>().c_str(), ieng->weight_bytes() / GiB, (long long)ieng->cold_experts(),
                   ieng->host_expert_bytes() / GiB, cpu_threads, ieng->cache_bytes() / GiB, slots, (long long)max_ctx,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      serve::inkling::SchedulerConfig sc{ic.max_step, chunk, max_ctx, [](const serve::gemma4::RequestLog& l) { log_line(request_line(l)); }};
      isched = std::make_unique<serve::inkling::Scheduler>(*ieng, *itok, sc);
      chat_stats = [&] { return isched->stats(); };
      hold = [&] { return std::static_pointer_cast<void>(std::make_shared<serve::inkling::Scheduler::Hold>(*isched)); };
      run_engine = [&] { isched->run(g_stop); };
    } else {
      geng = std::make_unique<engine::gemma4::Engine>(q, file, ec);
      auto& eng = *geng;
      if (vision && vision->n_embd() != eng.model().n_embd)
        throw std::runtime_error(mmproj_path + ": projects to " + std::to_string(vision->n_embd()) + " but the model's width is " +
                                 std::to_string(eng.model().n_embd));
      std::fprintf(stderr, "eklow-server: %s on %s, %.2f GiB of weights, %.2f GiB of KV cache (%d slots of %lld), %.1f s\n",
                   alias.c_str(), q.get_device().get_info<sycl::info::device::name>().c_str(), eng.weight_bytes() / GiB,
                   eng.cache_bytes() / GiB, slots, (long long)ec.max_ctx,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

      // The draft goes on after the engine.
      if (draft_file) {
        // The token table stays in the mapped file (a pass copies the rows it embeds); the head is the engine's own.
        const model::Dflash2Target target{eng.embedding_table(), eng.head_weight(), true};
        model::Dflash2::Options opt;
        opt.slots = slots;
        opt.max_block = int(engine::gemma4::kMaxVerify);  // a request may ask for a larger block than the default
        opt.full_context = draft_full_context;
        opt.sliding_causal = draft_sliding_causal;
        draft = std::make_unique<model::Dflash2>(q, *draft_file, target, opt);
        std::fprintf(stderr, "eklow-server: DFlash draft %s: blocks of %d (trained for %d), window %d, %.0f MiB on the card (its token table stays on the host, its head is the engine's); free card memory now %.2f GiB\n",
                     draft_path.c_str(), draft_block, draft_cfg.block_size, draft->window(), double(draft->bytes()) / 1048576.0,
                     rt::free_memory(q.get_device()).value_or(0) / GiB);
      }
      serve::gemma4::SchedulerConfig sc{batch, chunk, ec.max_ctx, [](const serve::gemma4::RequestLog& l) { log_line(request_line(l)); }};
      sc.draft = draft.get();
      if (draft) sc.draft_block = draft_block;
      // The disk prefix cache, before the scheduler, which writes into it until it stops.
      if (!prefix_dir.empty()) {
        sc.store_buffers = size_t(prefix_buffers);
        serve::PrefixStore::Options so;
        so.dir = prefix_dir;
        so.max_bytes = uint64_t(prefix_gib) << 30;
        so.key = serve::read_key_file(prefix_key);
        so.max_queued = sc.store_buffers + size_t(2 * eng.window_pages() * slots) + 8;  // the scheduler's buffers, so a put never finds the queue full
        store = std::make_unique<serve::PrefixStore>(so);
        // The keys' root: what the K/V depends on besides the tokens (the image rows' too, when the prompt has images).
        std::string fp = "model " + file_identity(model_path) + " | " + eng.kv_fingerprint();
        if (!mmproj_path.empty()) fp += " | mmproj " + file_identity(mmproj_path) + " image-tokens " + std::to_string(image_tokens);
        sc.store = store.get();
        sc.store_root = serve::chain_root(fp);
        const auto pb = eng.page_bytes();
        std::fprintf(stderr, "eklow-server: prefix cache %s, up to %lld GiB, AES-256-GCM: a page (%lld positions) is %.1f MiB of global K/V, %.1f MiB of sliding K/V at a restore point\n",
                     prefix_dir.c_str(), (long long)prefix_gib, (long long)attn::kPage, double(pb.global) / 1048576.0, double(pb.sliding) / 1048576.0);
      }
      gsched = std::make_unique<serve::gemma4::Scheduler>(eng, *gtok, sc);
      chat_stats = [&] { return gsched->stats(); };
      hold = [&] { return std::static_pointer_cast<void>(std::make_shared<serve::gemma4::Scheduler::Hold>(*gsched)); };
      run_engine = [&] { gsched->run(g_stop); };
    }
    if (images) images->bind(hold);
    if (embed_images) embed_images->bind(hold);
    std::thread engine_thread(run_engine);
    EmbedStats embed_stats;
    // The engine's line every log_interval seconds while there is traffic, and once more after it ends (rates
    // back to zero).
    std::thread stats_thread;
    if (log_interval > 0)
      stats_thread = std::thread([&, capacity = int64_t(slots) * max_ctx] {
        const auto every = std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(log_interval));
        auto prev = chat_stats();
        auto eprev = embed_stats.snapshot();
        auto t_prev = std::chrono::steady_clock::now();
        bool was_active = false;
        while (!g_stop.load()) {
          while (!g_stop.load() && std::chrono::steady_clock::now() < t_prev + every) std::this_thread::sleep_for(std::chrono::milliseconds(200));
          if (g_stop.load()) break;
          const auto cur = chat_stats();
          const auto ecur = embed_stats.snapshot();
          const auto now = std::chrono::steady_clock::now();
          const bool active = cur.steps != prev.steps || cur.active > 0 || cur.waiting > 0 || ecur.requests != eprev.requests;
          if (active || was_active) {
            std::string line = engine_line(prev, cur, eprev, ecur, seconds(now - t_prev), capacity);
            if (store) {
              const auto st = store->stats();
              line += fmt(" | disk %.1f GiB in %llu files, %lld restores (%lld positions), %lld parts stored", double(st.bytes) / 1073741824.0,
                          (unsigned long long)st.files, (long long)(cur.restores - prev.restores), (long long)(cur.restored_tokens - prev.restored_tokens),
                          (long long)(cur.stored_parts - prev.stored_parts));
              if (st.queued > 0) line += fmt(", %llu queued", (unsigned long long)st.queued);
              if (cur.deferred_stores > prev.deferred_stores)
                line += fmt(", %lld calls found no buffer free (their pages waited)", (long long)(cur.deferred_stores - prev.deferred_stores));
              if (cur.unstored_parts > prev.unstored_parts) line += fmt(", %lld sliding parts not stored", (long long)(cur.unstored_parts - prev.unstored_parts));
              if (cur.failed_restores > prev.failed_restores) line += fmt(", %lld failed restores", (long long)(cur.failed_restores - prev.failed_restores));
            }
            log_line(line);
          }
          was_active = active, prev = cur, eprev = ecur, t_prev = now;
        }
      });
    // The sampling defaults: llama.cpp's, under the GGUF's general.sampling.* (as llama-server takes them).
    serve::SamplingParams sampling;
    serve::apply_gguf_sampling(file, sampling);
    serve::gemma4::ChatDefaults defaults{sampling, max_ctx, which.value_or(text::Gemma4Template::Bolt), preserve_thinking};
    const serve::inkling::ChatDefaults idefaults{sampling};
    const serve::gemma4::ImageEncoder encode_image = [&](std::span<const uint8_t> bytes) { return (*images)(bytes); };
    std::fprintf(stderr, "eklow-server: sampling defaults: temperature %.2f, top_k %d, top_p %.2f, min_p %.2f, penalty_last_n %d, repeat %.2f\n",
                 double(sampling.temperature), int(sampling.top_k), double(sampling.top_p), double(sampling.min_p), int(sampling.penalty_last_n),
                 double(sampling.repeat_penalty));
    if (which)
      std::fprintf(stderr, "eklow-server: %s's chat template; earlier turns' reasoning %s by default\n", text::gemma4_template_name(*which),
                   preserve_thinking ? "kept (--preserve-thinking on)" : "as the template drops it");
    if (images) defaults.encode_image = [&](std::span<const uint8_t> bytes) { return (*images)(bytes); };

    httplib::Server http;
    http.new_task_queue = [] { return new httplib::ThreadPool(16); };
    std::atomic<int64_t> http_requests{0}, http_errors{0};  // chat requests, and those answered with an error status (/metrics)
    http.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    http.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
      const EmbedCounts ec_ = embed_stats.snapshot();
      res.set_content(serve::render_metrics(chat_stats(), {ec_.requests, ec_.inputs, ec_.tokens}, {http_requests.load(), http_errors.load()}, {alias, slots, max_ctx}),
                      "text/plain; version=0.0.4; charset=utf-8");
    });
    http.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
      res.set_content(serve::models_json(alias, max_ctx, slots), "application/json");
    });
    // A submitted request's answer, whole or streamed: its events through ChatResponse (either model's Request).
    auto respond = [keepalive_s](auto r, httplib::Response& res, std::shared_ptr<serve::gemma4::ChatResponse> out, bool stream, bool usage,
                                 const std::function<void(const serve::gemma4::ApiError&)>& fail) {
      if (!stream) {
        serve::gemma4::Event e;
        try {
          do {
            while (!r->next(e, std::chrono::milliseconds(1000))) {
            }
            out->add(e);
          } while (!e.done);
        } catch (const serve::gemma4::ApiError& err) {
          return fail(err);
        }
        return res.set_content(out->body(), "application/json");
      }
      res.set_header("Cache-Control", "no-cache");
      auto ka = std::make_shared<serve::KeepAlive>(std::chrono::milliseconds(int64_t(keepalive_s * 1000)));
      res.set_chunked_content_provider(
          "text/event-stream",
          [r, out, usage, ka](size_t, httplib::DataSink& sink) {
            serve::gemma4::Event e;
            while (!r->next(e, std::chrono::milliseconds(1000)))
              if (!sink.is_writable() || !ka->tick(sink)) {
                r->cancel();
                return false;
              }
            const std::string text = out->chunks(e, usage);
            if (!text.empty()) {
              if (!sink.write(text.data(), text.size())) {
                r->cancel();
                return false;
              }
              ka->wrote();
            } else if (!ka->tick(sink)) {  // a token that adds no text yet (a tool call is sent whole when it closes)
              r->cancel();
              return false;
            }
            if (e.done) sink.done();
            return true;
          },
          [r](bool) { r->cancel(); });
    };
    http.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
      ++http_requests;
      const std::function<void(const serve::gemma4::ApiError&)> fail = [&](const serve::gemma4::ApiError& e) {
        ++http_errors;
        reply_error(res, e);
      };
      if (inkling) {
        serve::inkling::ChatRequest cr;
        try {
          json::Value body;
          try {
            body = text::parse_llama_json(req.body);  // as llama-server reads it (nlohmann's numbers and refusals)
          } catch (const std::exception& e) {
            throw serve::gemma4::ApiError(400, "invalid_request_error", "", std::string("the body is not JSON: ") + e.what());
          }
          cr = serve::inkling::parse_chat(body, *itok, idefaults, images ? &encode_image : nullptr);
        } catch (const serve::gemma4::ApiError& e) {
          return fail(e);
        }
        auto r = std::make_shared<serve::inkling::Request>();
        r->prompt = std::move(cr.prompt);
        r->images = std::move(cr.images);
        r->prefill = std::move(cr.prefill);
        r->sampling = cr.sampling;
        r->max_tokens = cr.max_tokens;
        r->prefill_only = cr.prefill_only;
        r->stop = cr.stop;
        // llama-server samples a request with tools (tool_choice not none) under Inkling's tool grammar from its first token.
        r->grammar = cr.mode.use_tools();
        r->choice = cr.mode.tool_choice == text::InklingToolChoice::Required ? serve::InklingToolGrammar::Choice::Required
                                                                             : serve::InklingToolGrammar::Choice::Auto;
        r->tools = cr.mode.tool_names;
        r->parallel = cr.mode.parallel_tool_calls;
        r->mode = std::move(cr.mode);
        r->tag = serve::gemma4::new_completion_id();
        try {
          isched->submit(r);
        } catch (const std::invalid_argument&) {
          return fail(serve::gemma4::context_exceeded(int64_t(r->prompt.size()), max_ctx));
        }
        return respond(r, res, std::make_shared<serve::gemma4::ChatResponse>(r->tag, cr.model.empty() ? alias : cr.model, now_s()), cr.stream,
                       cr.include_usage, fail);
      }
      serve::gemma4::ChatRequest cr;
      try {
        json::Value body;
        try {
          body = json::parse(req.body);
        } catch (const std::exception& e) {
          throw serve::gemma4::ApiError(400, "invalid_request_error", "", std::string("the body is not JSON: ") + e.what());
        }
        cr = serve::gemma4::parse_chat(body, *gtok, defaults);
      } catch (const serve::gemma4::ApiError& e) {
        return fail(e);
      }
      auto r = std::make_shared<serve::gemma4::Request>();
      r->prompt = std::move(cr.prompt);
      r->images = std::move(cr.images);
      r->sampling = cr.sampling;
      r->max_tokens = cr.max_tokens;
      r->prefill_only = cr.prefill_only;
      r->stop = cr.stop;
      r->in_thinking = cr.in_thinking;
      r->tools = cr.tool_names;
      r->draft_block = cr.draft_block;
      r->tag = serve::gemma4::new_completion_id();
      try {
        gsched->submit(r);
      } catch (const std::invalid_argument&) {
        return fail(serve::gemma4::context_exceeded(int64_t(r->prompt.size()), max_ctx));
      }
      respond(r, res, std::make_shared<serve::gemma4::ChatResponse>(r->tag, cr.model.empty() ? alias : cr.model, now_s()), cr.stream, cr.include_usage,
              fail);
    });

    httplib::Server embed_http;
    std::thread embed_thread;
    if (embed_model) {
      embed_http.new_task_queue = [] { return new httplib::ThreadPool(4); };
      serve_embeddings(embed_http, *embed_model, *embed_tok, embed_mu, hold, embed_alias, 8192, embed_stats, embed_images.get());
      std::fprintf(stderr, "eklow-server: %s, %.2f GiB, listening on %s:%d\n", embed_alias.c_str(), embed_model->bytes() / 1073741824.0,
                   host.c_str(), embed_port);
      embed_thread = std::thread([&] {
        if (!embed_http.listen(host, embed_port)) {
          std::fprintf(stderr, "eklow-server: cannot listen on %s:%d\n", host.c_str(), embed_port);
          g_stop = true;
        }
      });
    }

    httplib::Server checker_http;
    std::thread checker_thread;
    if (checker) {
      checker_http.new_task_queue = [] { return new httplib::ThreadPool(4); };
      checker_http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
      serve::SystemOneConfig c1;
      c1.alias = checker_alias;
      c1.log = log_line;
      c1.hold = hold;
      serve::serve_systemone(checker_http, *checker, checker_mu, c1);
      std::fprintf(stderr, "eklow-server: checker %s listening on %s:%d (POST /v1/systemone)\n", checker_alias.c_str(), host.c_str(), checker_port);
      checker_thread = std::thread([&] {
        if (!checker_http.listen(host, checker_port)) {
          std::fprintf(stderr, "eklow-server: cannot listen on %s:%d\n", host.c_str(), checker_port);
          g_stop = true;
        }
      });
    }
    httplib::Server rerank_http;
    std::thread rerank_thread;
    if (reranker) {
      rerank_http.new_task_queue = [] { return new httplib::ThreadPool(4); };
      rerank_http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
      serve::RerankConfig rc;
      rc.alias = rerank_alias;
      rc.log = log_line;
      rc.hold = hold;
      serve::serve_rerank(rerank_http, *reranker, rerank_mu, rc);
      std::fprintf(stderr, "eklow-server: reranker %s listening on %s:%d (POST /v1/rerank)\n", rerank_alias.c_str(), host.c_str(), rerank_port);
      rerank_thread = std::thread([&] {
        if (!rerank_http.listen(host, rerank_port)) {
          std::fprintf(stderr, "eklow-server: cannot listen on %s:%d\n", host.c_str(), rerank_port);
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
      checker_http.stop();
      rerank_http.stop();
    });
    std::fprintf(stderr, "eklow-server: listening on %s:%d\n", host.c_str(), port);
    if (!http.listen(host, port)) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
    g_stop = true;
    stopper.join();
    engine_thread.join();
    if (embed_thread.joinable()) embed_thread.join();
    if (checker_thread.joinable()) checker_thread.join();
    if (rerank_thread.joinable()) rerank_thread.join();
    if (stats_thread.joinable()) stats_thread.join();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-server: %s\n", e.what());
    return 1;
  }
  return 0;
}
