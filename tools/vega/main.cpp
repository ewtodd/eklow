// eklow-vega: vega's reader, engine and laya's System One on the card, against vegaml's reference (tools/vega/ref_vega.py).
//
//   eklow-vega features --reader GGUF --vega DIR --states STATES.json --index REF_DIR --out DIR [--max-tokens N] [--chunk N]
//       For every item of a reference run (REF_DIR/index.json: a state's key and a question), builds the prompt itself and
//       checks its ids and spans against the reference's, reads every question of a state in one pass of the reader (as a
//       request does), and writes the pooled vectors (DIR/NNNN-f.npy [2][3][d], -o.npy [K][1][d], -c.npy [C][d]) and
//       DIR/index.json (per item: ids_equal, n_chunks, the engine's probabilities on eklow's vectors; per state: ms).
//   eklow-vega check --reader GGUF --vega DIR --states STATES.json [--golden tools/vega/golden.json] [--tolerance P]
//       The regression gate without Python: the synthetic states (synthetic_states.py's) through the reader and the engine,
//       every question's probabilities against the reference's in the golden file (fp32 vegaml), each prompt's length too.
//       Exits 1 past the tolerance (default 2e-3; eklow measured 5.7e-4 on 2026-11-10).
//   eklow-vega prompts --reader GGUF --vega DIR --states STATES.json --index REF_DIR
//       Only the prompts (no card): every item's ids and spans against the reference's.
//   eklow-vega engine --vega DIR --feats REF_DIR --out OUT.json [--policy DIR]
//       The engine alone on the reference's own pooled vectors (REF_DIR/NNNN.npz): probabilities, logits, the thermal
//       features, steps; with a policy also each item's noul as the System One gives it from those vectors (the head
//       when its question has one worth using, the calibration).
//   eklow-vega predict --reader GGUF --vega DIR [--policy DIR] --states STATES.json --out OUT.json [--repeat N] [--max-tokens N]
//       POST /v1/systemone's answer to son-of-anton's three questions for every state ({key: {qid: {noul, readout}}}), and
//       the latency of each (the median of N runs after one warm-up).
//   eklow-vega serve --reader GGUF --vega DIR [--policy DIR] [--host ADDR] --port N [--alias NAME] [--max-tokens N]
//       POST /v1/systemone (tools/server/systemone.hpp) and GET /health, the checker alone; eklow-server serves the same
//       beside a chat model (--checker-*).
//
// STATES.json is a list of {"key", "state"}.
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <mutex>
#include <thread>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "rt/device.hpp"
#include "vega/checker.hpp"
#include "vega/files.hpp"
#include "../server/systemone.hpp"

using namespace eklow;
using namespace eklow::vega;

namespace {

// son-of-anton's end-of-turn questions (daemon/internal/check/check.go), as the policy's heads were fitted on them.
const std::vector<std::pair<std::string, std::string>> kCheckerQuestions = {
    {"claim",
     "Does the assistant's draft reply say it has done something (written, logged, updated, saved, created, removed, scheduled, "
     "checked, looked up, sent, noted or remembered) that none of this turn's tool calls did?"},
    {"unchecked",
     "The user's latest message asks about the current state of something (a scheduled job, a file, a log, a worker) or why "
     "something happened. Does the draft reply answer it without any tool call in this turn that checked that state? Answer no if "
     "the message asks no such thing."},
    {"offtopic", "Does the draft reply fail to respond to the user's latest message (it answers something else, such as an earlier topic)?"},
};

std::map<std::string, std::string> load_states(const std::string& path) {
  std::map<std::string, std::string> out;
  const json::Value v = json::parse(read_file(path));
  for (const auto& s : v.items()) out[s.find("key")->as_string()] = s.find("state")->as_string();
  return out;
}

std::vector<std::string> state_order(const std::string& path) {
  std::vector<std::string> out;
  const json::Value v = json::parse(read_file(path));
  for (const auto& s : v.items()) out.push_back(s.find("key")->as_string());
  return out;
}

// A question as ref_vega.py writes it (vegaml's vocabulary).
Question vega_question(const json::Value& d) {
  Question q;
  const std::string t = d.find("type")->as_string();
  q.type = t == "choice" ? QType::Choice : t == "score" ? QType::Score : QType::Boolean;
  q.instructions = d.find("instructions")->as_string();
  if (const json::Value* c = d.find("criteria")) {
    if (c->is_object())
      for (const auto& [k, v] : c->members()) q.criteria.emplace_back(k, v.is_null() ? std::nullopt : std::optional<std::string>(v.as_string()));
    else
      for (const auto& v : c->items()) q.criteria.emplace_back("", v.as_string());
  }
  return q;
}

std::string pad4(size_t i) {
  char b[16];
  std::snprintf(b, sizeof b, "%04zu", i);
  return b;
}

int cmd_features(const std::map<std::string, std::string>& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
  Checker ck(q, a.at("reader"), a.at("vega"), "", std::stoll(a.count("max-tokens") ? a.at("max-tokens") : "16384"),
             std::stoll(a.count("chunk") ? a.at("chunk") : "1024"));
  const auto states = load_states(a.at("states"));
  const json::Value index = json::parse(read_file(a.at("index") + "/index.json"));
  const auto& items = index.find("items")->items();
  std::filesystem::create_directories(a.at("out"));
  json::Writer w;
  w.begin_object().key("items").begin_array();
  std::map<std::string, std::vector<size_t>> by_state;
  std::vector<std::string> order;
  for (size_t i = 0; i < items.size(); ++i) {
    const std::string k = items[i].find("key")->as_string();
    if (!by_state.count(k)) order.push_back(k);
    by_state[k].push_back(i);
  }
  std::vector<json::Value> per_state;
  int bad = 0;
  for (const std::string& key : order) {
    const std::vector<int32_t> st = ck.tokenizer().encode(states.at(key), true);
    std::vector<Prompt> prompts;
    std::vector<Question> qs;
    for (size_t i : by_state[key]) {
      qs.push_back(vega_question(*items[i].find("question")));
      prompts.push_back(build_prompt(ck.tokenizer(), st, qs.back(), ck.prompt_config()));
    }
    const auto t0 = std::chrono::steady_clock::now();
    const auto pooled = ck.reader().read(prompts, ck.pool_spec());
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "%s: %zu prompts, %zu tokens each at most, %.1f ms\n", key.c_str(), prompts.size(), prompts[0].ids.size(), ms);
    for (size_t j = 0; j < by_state[key].size(); ++j) {
      const size_t i = by_state[key][j];
      const json::Value& it = items[i];
      const Prompt& p = prompts[j];
      std::vector<int32_t> ref;
      for (const auto& v : it.find("ids")->items()) ref.push_back(int32_t(std::stoll(v.int_digits())));
      const auto& sp = *it.find("spans");
      auto pair_eq = [](const json::Value& v, int32_t a, int32_t b) {
        return std::stoll(v.items()[0].int_digits()) == a && std::stoll(v.items()[1].int_digits()) == b;
      };
      bool eq = ref == p.ids && pair_eq(*sp.find("state"), p.s0, p.s1) && pair_eq(*sp.find("question"), p.q0, p.q1) &&
                sp.find("options")->items().size() == p.options.size();
      for (size_t k = 0; eq && k < p.options.size(); ++k) eq = pair_eq(sp.find("options")->items()[k], p.options[k].first, p.options[k].second);
      if (!eq) ++bad, std::fprintf(stderr, "  %s %s: prompt differs from the reference's\n", key.c_str(), it.find("qid")->as_string().c_str());
      const Pooled& po = pooled[j];
      const int64_t d = ck.reader().d_model(), K = int64_t(p.options.size());
      write_npy(a.at("out") + "/" + pad4(i) + "-f.npy", {2, 3, d}, po.item.data());
      write_npy(a.at("out") + "/" + pad4(i) + "-o.npy", {K, 1, d}, po.opts.data());
      write_npy(a.at("out") + "/" + pad4(i) + "-c.npy", {int64_t(po.chunks.size()) / d, d}, po.chunks.data());
      const EngineAnswer e = ck.engine().answer(po, qs[j].type);
      w.begin_object().kv("key", key).kv("qid", it.find("qid")->as_string()).kv("ids_equal", eq).kv("tokens", int64_t(p.ids.size()));
      w.kv("n_chunks", int64_t(po.n_chunks)).key("P").begin_array();
      for (double v : e.probs) w.value(v);
      w.end_array().kv("state_ms", ms).end_object();
    }
  }
  w.end_array().kv("prompt_mismatches", int64_t(bad)).end_object();
  std::ofstream(a.at("out") + "/index.json") << w.str();
  return bad ? 1 : 0;
}

int cmd_check(const std::map<std::string, std::string>& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
  Checker ck(q, a.at("reader"), a.at("vega"), "", 16384);
  const auto states = load_states(a.at("states"));
  const json::Value g = json::parse(read_file(a.count("golden") ? a.at("golden") : "tools/vega/golden.json"));
  const double tol = std::stod(a.count("tolerance") ? a.at("tolerance") : "2e-3");
  std::map<std::string, Question> qs;
  for (const auto& [qid, d] : g.find("questions")->members()) qs[qid] = vega_question(d);
  std::map<std::string, std::vector<const json::Value*>> by_state;
  std::vector<std::string> order;
  for (const auto& it : g.find("items")->items()) {
    const std::string k = it.find("key")->as_string();
    if (!by_state.count(k)) order.push_back(k);
    by_state[k].push_back(&it);
  }
  double worst = 0;
  int bad = 0;
  for (const std::string& key : order) {
    const std::vector<int32_t> st = ck.tokenizer().encode(states.at(key), true);
    std::vector<Prompt> prompts;
    for (const json::Value* it : by_state[key]) prompts.push_back(build_prompt(ck.tokenizer(), st, qs.at(it->find("qid")->as_string()), ck.prompt_config()));
    const auto pooled = ck.reader().read(prompts, ck.pool_spec());
    for (size_t j = 0; j < prompts.size(); ++j) {
      const json::Value& it = *by_state[key][j];
      const std::string qid = it.find("qid")->as_string();
      const EngineAnswer e = ck.engine().answer(pooled[j], qs.at(qid).type);
      double d = 0;
      const auto& ref = it.find("P")->items();
      for (size_t k = 0; k < ref.size() && k < e.probs.size(); ++k) d = std::max(d, std::abs(e.probs[k] - ref[k].as_double()));
      const bool len_ok = int64_t(prompts[j].ids.size()) == std::stoll(it.find("tokens")->int_digits());
      worst = std::max(worst, d);
      if (d > tol || !len_ok) ++bad, std::fprintf(stderr, "%s %s: |dp| %.3g%s\n", key.c_str(), qid.c_str(), d, len_ok ? "" : ", prompt length differs");
    }
  }
  std::fprintf(stderr, "check: %d of %zu answers past %.3g (largest |dp| %.3g)\n", bad, g.find("items")->items().size(), tol, worst);
  return bad ? 1 : 0;
}

int cmd_prompts(const std::map<std::string, std::string>& a) {
  const auto f = gguf::File::open(a.at("reader"));
  const auto tok = text::QwenTokenizer::from_gguf(f);
  const EngineConfig c = Engine(a.at("vega")).cfg();
  PromptConfig pc;
  pc.max_len = c.max_len, pc.opt_max_tokens = c.opt_max_tokens, pc.query_first_max_state = c.query_first_max_state;
  const auto states = load_states(a.at("states"));
  const json::Value index = json::parse(read_file(a.at("index") + "/index.json"));
  int bad = 0, n = 0;
  for (const auto& it : index.find("items")->items()) {
    const Prompt p = build_prompt(tok, tok.encode(states.at(it.find("key")->as_string()), true), vega_question(*it.find("question")), pc);
    std::vector<int32_t> ref;
    for (const auto& v : it.find("ids")->items()) ref.push_back(int32_t(std::stoll(v.int_digits())));
    const auto& sp = *it.find("spans");
    auto at = [](const json::Value& v, int i) { return int32_t(std::stoll(v.items()[size_t(i)].int_digits())); };
    bool eq = ref == p.ids && at(*sp.find("state"), 0) == p.s0 && at(*sp.find("state"), 1) == p.s1 && at(*sp.find("question"), 0) == p.q0 &&
              at(*sp.find("question"), 1) == p.q1 && sp.find("options")->items().size() == p.options.size();
    for (size_t k = 0; eq && k < p.options.size(); ++k)
      eq = at(sp.find("options")->items()[k], 0) == p.options[k].first && at(sp.find("options")->items()[k], 1) == p.options[k].second;
    ++n;
    if (!eq) {
      ++bad;
      size_t d = 0;
      while (d < ref.size() && d < p.ids.size() && ref[d] == p.ids[d]) ++d;
      std::fprintf(stderr, "%s %s: differs (%zu vs %zu tokens, first difference at %zu)\n", it.find("key")->as_string().c_str(),
                   it.find("qid")->as_string().c_str(), p.ids.size(), ref.size(), d);
    }
  }
  std::fprintf(stderr, "prompts: %d of %d equal to the reference's\n", n - bad, n);
  return bad ? 1 : 0;
}

int cmd_engine(const std::map<std::string, std::string>& a) {
  Engine eng(a.at("vega"));
  std::unique_ptr<Policy> pol;
  if (a.count("policy")) {
    pol = std::make_unique<Policy>(Policy::load(a.at("policy"), eng.sha256()));
    std::fprintf(stderr, "policy: %s; %zu heads\n", pol->verified().c_str(), pol->heads().size());
  }
  const json::Value index = json::parse(read_file(a.at("feats") + "/index.json"));
  json::Writer w;
  w.begin_array();
  const auto& items = index.find("items")->items();
  for (size_t i = 0; i < items.size(); ++i) {
    const auto z = parse_npz(read_file(a.at("feats") + "/" + pad4(i) + ".npz"), pad4(i) + ".npz");
    Pooled p;
    p.item = z.at("f").floats(), p.opts = z.at("o").floats(), p.chunks = z.at("c").floats();
    for (float m : z.at("cm").floats()) p.n_chunks += m > 0.5f;
    const QType t = QType(int32_t(std::stoll(items[i].find("qtype")->int_digits())));
    const EngineAnswer e = eng.answer(p, t);
    w.begin_object().kv("key", items[i].find("key")->as_string()).kv("qid", items[i].find("qid")->as_string()).key("P").begin_array();
    for (double v : e.probs) w.value(v);
    w.end_array().key("logits").begin_array();
    for (double v : e.logits) w.value(v);
    w.end_array().key("phys").begin_array();
    for (double v : e.phys) w.value(v);
    w.end_array().kv("steps", int64_t(e.steps));
    if (pol && t == QType::Boolean) {
      const CheckerAnswer an = answer(eng, pol.get(), items[i].find("qid")->as_string(), Question{t, "", {}}, p);
      w.kv("noul", an.noul).kv("readout", an.readout);
    }
    w.end_object();
  }
  w.end_array();
  std::ofstream(a.at("out")) << w.str();
  std::fprintf(stderr, "engine %s on %zu items\n", eng.sha256().substr(0, 12).c_str(), items.size());
  return 0;
}

int cmd_predict(const std::map<std::string, std::string>& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
  const auto t0 = std::chrono::steady_clock::now();
  Checker ck(q, a.at("reader"), a.at("vega"), a.count("policy") ? a.at("policy") : "",
             std::stoll(a.count("max-tokens") ? a.at("max-tokens") : "16384"), std::stoll(a.count("chunk") ? a.at("chunk") : "1024"));
  std::fprintf(stderr, "loaded in %.1f s: reader %.0f MiB on the card; policy %s\n",
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), double(ck.reader().bytes()) / 1048576.0,
               ck.policy() ? ck.policy()->verified().c_str() : "none");
  const int repeat = std::stoi(a.count("repeat") ? a.at("repeat") : "1");
  const auto states = load_states(a.at("states"));
  std::vector<std::pair<std::string, Question>> qs;
  for (const auto& [id, ins] : kCheckerQuestions) qs.emplace_back(id, Question{QType::Boolean, ins, {}});
  json::Writer w;
  w.begin_object();
  std::vector<double> lat;
  for (const std::string& key : state_order(a.at("states"))) {
    CheckerResult r = ck.predict(states.at(key), qs);  // warm-up, and the answer
    std::vector<double> ms;
    for (int i = 0; i < repeat; ++i) ms.push_back(ck.predict(states.at(key), qs).total_ms);
    std::sort(ms.begin(), ms.end());
    const double med = ms[ms.size() / 2];
    lat.push_back(med);
    w.key(key).begin_object();
    for (const auto& an : r.answers)
      w.key(an.qid).begin_object().kv("noul", an.noul).kv("readout", an.readout).kv("calibrated", an.calibrated).end_object();
    w.kv("tokens", r.tokens).kv("ms", med).end_object();
    std::fprintf(stderr, "%s: %lld tokens, %.1f ms (read %.1f):", key.c_str(), (long long)r.tokens, med, r.read_ms);
    for (const auto& an : r.answers) std::fprintf(stderr, " %s %.6f", an.qid.c_str(), an.noul);
    std::fprintf(stderr, "\n");
  }
  w.end_object();
  std::ofstream(a.at("out")) << w.str();
  std::sort(lat.begin(), lat.end());
  if (!lat.empty())
    std::fprintf(stderr, "latency over %zu states: p50 %.1f ms, p90 %.1f ms, max %.1f ms\n", lat.size(), lat[lat.size() / 2],
                 lat[std::min(lat.size() - 1, size_t(double(lat.size()) * 0.9))], lat.back());
  return 0;
}

std::atomic<bool> g_stop{false};

int cmd_serve(const std::map<std::string, std::string>& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
  Checker ck(q, a.at("reader"), a.at("vega"), a.count("policy") ? a.at("policy") : "",
             std::stoll(a.count("max-tokens") ? a.at("max-tokens") : "16384"), std::stoll(a.count("chunk") ? a.at("chunk") : "1024"));
  ck.warm();
  const std::string host = a.count("host") ? a.at("host") : "127.0.0.1";
  const int port = std::stoi(a.at("port"));
  std::mutex mu, log_mu;
  httplib::Server http;
  http.new_task_queue = [] { return new httplib::ThreadPool(4); };
  http.Get("/health", [](const httplib::Request&, httplib::Response& res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
  serve::SystemOneConfig cfg;
  cfg.alias = a.count("alias") ? a.at("alias") : "vega-u";
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
  serve::serve_systemone(http, ck, mu, cfg);
  std::signal(SIGINT, [](int) { g_stop = true; });
  std::signal(SIGTERM, [](int) { g_stop = true; });
  std::thread stopper([&] {
    while (!g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    http.stop();
  });
  std::fprintf(stderr, "eklow-vega: %s (policy %s), reader %.0f MiB on the card, listening on %s:%d\n", cfg.alias.c_str(),
               ck.policy() ? ck.policy()->verified().c_str() : "none", double(ck.reader().bytes()) / 1048576.0, host.c_str(), port);
  const bool ok = http.listen(host, port);
  g_stop = true;
  stopper.join();
  if (!ok) throw std::runtime_error("cannot listen on " + host + ":" + std::to_string(port));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: eklow-vega check|features|prompts|engine|predict|serve --flag value ... (see tools/vega/main.cpp)\n");
    return 2;
  }
  const std::string cmd = argv[1];
  std::map<std::string, std::string> a;
  for (int i = 2; i + 1 < argc; i += 2) {
    std::string k = argv[i];
    if (k.rfind("--", 0) != 0) {
      std::fprintf(stderr, "eklow-vega: unexpected %s\n", k.c_str());
      return 2;
    }
    a[k.substr(2)] = argv[i + 1];
  }
  try {
    if (cmd == "features") return cmd_features(a);
    if (cmd == "engine") return cmd_engine(a);
    if (cmd == "prompts") return cmd_prompts(a);
    if (cmd == "check") return cmd_check(a);
    if (cmd == "predict") return cmd_predict(a);
    if (cmd == "serve") return cmd_serve(a);
    std::fprintf(stderr, "eklow-vega: unknown command %s\n", cmd.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-vega: %s\n", e.what());
    return 1;
  }
}
