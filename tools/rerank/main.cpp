// eklow-rerank: Qwen3-Reranker on the card, alone.
//
//   eklow-rerank serve --model GGUF [--host ADDR] --port N [--alias NAME] [--max-tokens N]
//       POST /v1/rerank (tools/server/rerank.hpp) and GET /health; eklow-server serves the same beside a chat model
//       (--rerank-*).
//   eklow-rerank score --model GGUF --pairs PAIRS.json --out OUT.json [--repeat N] [--max-tokens N]
//       Each request of PAIRS.json (tools/rerank/synthetic_pairs.py's) scored without HTTP: {name: {"scores": [...],
//       "ms": [...]}} as rerank_compare.py fetch writes it, the first run a warm-up.
//   eklow-rerank tokens --model GGUF --pairs PAIRS.json --out OUT.json
//       Every pair's prompt as tokens (no card): {name: [[ids of document 0], ...]}, to hold against llama.cpp's /tokenize.
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "../server/rerank.hpp"
#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"
#include "text/tokenizer_qwen.hpp"

using namespace eklow;

namespace {

std::atomic<bool> g_stop{false};

std::string read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::ostringstream s;
  s << f.rdbuf();
  return s.str();
}

int cmd_serve(const std::map<std::string, std::string>& a) {
  const auto f = gguf::File::open(a.at("model"));
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
  model::RerankQwen3 m(q, f, std::stoll(a.count("max-tokens") ? a.at("max-tokens") : "4096"));
  m.score({m.prompt("warm-up", "warm-up")});
  const std::string host = a.count("host") ? a.at("host") : "127.0.0.1";
  const int port = std::stoi(a.at("port"));
  std::mutex mu, log_mu;
  httplib::Server http;
  http.new_task_queue = [] { return new httplib::ThreadPool(4); };
  http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
  serve::RerankConfig cfg;
  if (a.count("alias")) cfg.alias = a.at("alias");
  cfg.log = [&log_mu](const std::string& line) {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    char ts[16];
    std::strftime(ts, sizeof ts, "%H:%M:%S", &tm);
    std::lock_guard lk(log_mu);
    std::printf("%s %s\n", ts, line.c_str());
    std::fflush(stdout);
  };
  serve::serve_rerank(http, m, mu, cfg);
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });
  std::thread stopper([&] {
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    http.stop();
  });
  std::fprintf(stderr, "eklow-rerank: %s, %.0f MiB on the card, pairs of up to %lld tokens, listening on %s:%d\n", cfg.alias.c_str(),
               double(m.bytes()) / 1048576.0, (long long)m.max_tokens(), host.c_str(), port);
  const bool ok = http.listen(host, port);
  g_stop = true;
  stopper.join();
  if (!ok) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
  return 0;
}

int cmd_tokens(const std::map<std::string, std::string>& a) {
  const auto f = gguf::File::open(a.at("model"));
  const auto tok = text::QwenTokenizer::from_gguf(f);
  const std::string tmpl = f.at("tokenizer.chat_template.rerank").as_string();
  const json::Value reqs = json::parse(read_all(a.at("pairs")));
  json::Writer w(false);
  w.begin_object();
  for (const auto& r : reqs.items()) {
    w.key(r.find("name")->as_string()).begin_array();
    for (const auto& d : r.find("documents")->items()) {
      std::string s;
      for (size_t pos = 0; pos < tmpl.size();) {
        if (tmpl.compare(pos, 7, "{query}") == 0) s += r.find("query")->as_string(), pos += 7;
        else if (tmpl.compare(pos, 10, "{document}") == 0) s += d.as_string(), pos += 10;
        else s += tmpl[pos++];
      }
      w.begin_array();
      for (int32_t id : tok.encode(s, true)) w.value(int64_t(id));
      w.end_array();
    }
    w.end_array();
  }
  w.end_object();
  std::ofstream(a.at("out")) << w.str();
  return 0;
}

int cmd_score(const std::map<std::string, std::string>& a) {
  const auto f = gguf::File::open(a.at("model"));
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
  model::RerankQwen3 m(q, f, std::stoll(a.count("max-tokens") ? a.at("max-tokens") : "4096"));
  const int repeat = std::stoi(a.count("repeat") ? a.at("repeat") : "3");
  const json::Value reqs = json::parse(read_all(a.at("pairs")));
  json::Writer w;
  w.begin_object();
  for (const auto& r : reqs.items()) {
    std::vector<std::vector<int32_t>> pairs;
    int64_t tokens = 0;
    for (const auto& d : r.find("documents")->items()) {
      pairs.push_back(m.prompt(r.find("query")->as_string(), d.as_string()));
      tokens += int64_t(pairs.back().size());
    }
    std::vector<double> scores = m.score(pairs), ms;
    for (int i = 0; i < repeat; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<std::vector<int32_t>> again;
      for (const auto& d : r.find("documents")->items()) again.push_back(m.prompt(r.find("query")->as_string(), d.as_string()));
      scores = m.score(again);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const std::string name = r.find("name")->as_string();
    w.key(name).begin_object().key("scores").begin_array();
    for (double s : scores) w.value(s);
    w.end_array().key("ms").begin_array();
    for (double v : ms) w.value(v);
    w.end_array().kv("tokens", tokens).end_object();
    std::fprintf(stderr, "%s: %zu documents, %lld tokens, %.1f ms (best of %d):", name.c_str(), pairs.size(), (long long)tokens,
                 ms.empty() ? 0.0 : ms[0], repeat);
    for (double s : scores) std::fprintf(stderr, " %.5f", s);
    std::fprintf(stderr, "\n");
  }
  w.end_object();
  std::ofstream(a.at("out")) << w.str();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: eklow-rerank serve|score --flag value ... (see tools/rerank/main.cpp)\n");
    return 2;
  }
  const std::string cmd = argv[1];
  std::map<std::string, std::string> a;
  for (int i = 2; i + 1 < argc; i += 2) a[std::string(argv[i]).substr(2)] = argv[i + 1];
  try {
    if (cmd == "serve") return cmd_serve(a);
    if (cmd == "score") return cmd_score(a);
    if (cmd == "tokens") return cmd_tokens(a);
    std::fprintf(stderr, "eklow-rerank: unknown command %s\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-rerank: %s\n", e.what());
    return 1;
  }
}
