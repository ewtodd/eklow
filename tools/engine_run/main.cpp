// eklow-engine-run: the serving engine over one prompt, written in eklow-oracle's dump format so
// tools/ref/compare_dumps.py can set it beside the oracle or the reference forward.
//
//   eklow-engine-run --model GGUF --out DIR (--tokens-json FILE | --tokens 2,105,...)
//                    [--tensors PREFIX,...] [--logits-last N] [--chunk N] [--decode-last N]
//                    [--twin] [--max-ctx N] [--kv-int8]
//
// The prompt is fed in prefill chunks of --chunk tokens (default: one chunk), then its last
// --decode-last tokens one per step, as decode would. --twin runs the same prompt in a second slot
// in the same steps, offset by one chunk, writes its logits to DIR/twin, and fails when their mean
// KL from the first slot's exceeds --twin-tol (default 0.05). The two slots see different batch
// shapes, so they differ by the int8 noise floor; a slot reading another's cache is far worse.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, sep);)
    if (!p.empty()) out.push_back(p);
  return out;
}

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// KL(p || q) of two logit rows, in double.
double kl(const float* p, const float* q, int64_t n) {
  auto lse = [n](const float* x) {
    double m = x[0], s = 0;
    for (int64_t i = 1; i < n; ++i) m = std::max(m, double(x[i]));
    for (int64_t i = 0; i < n; ++i) s += std::exp(double(x[i]) - m);
    return m + std::log(s);
  };
  const double lp = lse(p), lq = lse(q);
  double d = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double a = double(p[i]) - lp;
    d += std::exp(a) * (a - (double(q[i]) - lq));
  }
  return d;
}

// The 26B's MoE tensor names (each ends in the layer number).
const std::vector<std::string> moe_prefixes = {"ffn_mlp-", "ffn_moe-", "ffn_moe_combined-", "ffn_moe_logits-"};

}  // namespace

int main(int argc, char** argv) {
  std::string model, out, tokens_json, tokens_arg;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  std::set<int64_t> moe_layers;
  int64_t logits_last = 1, chunk = 0, decode_last = 0, max_ctx = 0;
  bool twin = false, kv_int8 = false;
  std::string expert_profile;
  int64_t experts_on_card = -1;
  int cpu_threads = 0;
  int stream_slots = 32, stream_min_rows = 16;
  double stream_copy_us = 360, stream_cpu_us = 20;
  double twin_tol = 0.05;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--tokens-json") tokens_json = next();
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--tensors") prefixes = split(next(), ',');
    else if (a == "--logits-last") logits_last = std::atoll(next().c_str());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--decode-last") decode_last = std::atoll(next().c_str());
    else if (a == "--max-ctx") max_ctx = std::atoll(next().c_str());
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--expert-profile") expert_profile = next();
    else if (a == "--experts-on-card") experts_on_card = std::atoll(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
    else if (a == "--stream-slots") stream_slots = std::atoi(next().c_str());
    else if (a == "--stream-min-rows") stream_min_rows = std::atoi(next().c_str());
    else if (a == "--stream-copy-us") stream_copy_us = std::atof(next().c_str());
    else if (a == "--stream-cpu-us") stream_cpu_us = std::atof(next().c_str());
    else if (a == "--twin") twin = true;
    else if (a == "--twin-tol") twin_tol = std::atof(next().c_str());
    else if (a == "--moe-layers")
      for (auto s : split(next(), ',')) moe_layers.insert(std::atoll(s.c_str()));
    else {
      std::fprintf(stderr, "usage: eklow-engine-run --model GGUF --out DIR (--tokens-json FILE | --tokens IDS) "
                           "[--tensors PREFIX,...] [--logits-last N] [--chunk N] [--decode-last N] [--twin] "
                           "[--twin-tol KL] [--max-ctx N] [--kv-int8] [--moe-layers N,...] [--expert-profile JSON --experts-on-card N] [--cpu-threads N] [--stream-slots N] [--stream-min-rows N] [--stream-copy-us US] [--stream-cpu-us US]\n");
      return 2;
    }
  }
  if (!moe_layers.empty())
    for (const auto& mp : moe_prefixes) prefixes.push_back(mp);
  try {
    std::vector<int32_t> tokens;
    if (!tokens_json.empty()) {
      std::ifstream in(tokens_json);
      std::string s((std::istreambuf_iterator<char>(in)), {});
      const json::Value doc = json::parse(s);
      const json::Value* ids = doc.find("ids");
      if (!ids || !ids->is_array()) throw std::runtime_error(tokens_json + ": no \"ids\" array");
      for (const auto& v : ids->items()) tokens.push_back(int32_t(std::atoll(v.int_digits().c_str())));
    } else {
      for (const auto& t : split(tokens_arg, ',')) tokens.push_back(int32_t(std::atoi(t.c_str())));
    }
    if (tokens.empty() || out.empty() || model.empty()) throw std::runtime_error("need --model, --out and tokens");
    const int64_t n = int64_t(tokens.size());
    logits_last = std::min(logits_last, n);
    decode_last = std::min(decode_last, n - 1);
    const int64_t prefill = n - decode_last;
    if (chunk <= 0) chunk = prefill;

    // The steps, as [start, end) token ranges.
    std::vector<std::pair<int64_t, int64_t>> steps;
    for (int64_t s = 0; s < prefill; s += chunk) steps.push_back({s, std::min(prefill, s + chunk)});
    for (int64_t s = prefill; s < n; ++s) steps.push_back({s, s + 1});

    engine::Config ec;
    ec.n_slots = twin ? 2 : 1;
    ec.max_ctx = max_ctx > 0 ? max_ctx : (n + 255) / 256 * 256;
    int64_t widest = 0;
    for (auto [a, b] : steps) widest = std::max(widest, b - a);
    ec.max_batch = twin ? 2 * widest : widest;
    ec.max_chunk = widest;
    ec.int8_global_kv = kv_int8;
    ec.expert_profile = expert_profile;
    ec.experts_on_card = experts_on_card;
    ec.cpu_threads = cpu_threads;
    ec.stream_slots = stream_slots;
    ec.stream_min_rows = stream_min_rows;
    ec.stream_copy_us = stream_copy_us;
    ec.stream_cpu_us = stream_cpu_us;
    ec.max_logits = twin ? 2 * std::min(widest, logits_last) : std::min(widest, logits_last);

    auto t0 = std::chrono::steady_clock::now();
    auto f = gguf::File::open(model);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::Engine eng(q, f, ec);
    const int64_t V = eng.model().n_vocab;
    std::printf("loaded %.2f GiB of weights, %.2f GiB of KV cache onto %s in %.1f s\n", eng.weight_bytes() / 1073741824.0,
                eng.cache_bytes() / 1073741824.0, q.get_device().get_info<sycl::info::device::name>().c_str(), since(t0));

    // Each step's tensors are that step's tokens; the dumps concatenate them along the token axis.
    std::map<std::string, std::pair<std::vector<float>, std::vector<int64_t>>> dumps;
    std::vector<std::string> order;
    model::DumpSink sink;
    sink.wants = [&](const std::string& name) {
      for (const auto& p : prefixes) {
        if (name.compare(0, p.size(), p) != 0) continue;
        if (moe_layers.empty()) return true;
        // MoE tensors are only dumped for the requested layers (the long prompt's would be huge);
        // the layer is the number after the last '-'.
        bool is_moe = false;
        for (const auto& mp : moe_prefixes)
          if (name.compare(0, mp.size(), mp) == 0) is_moe = true;
        if (!is_moe) return true;
        return moe_layers.count(std::atoll(name.substr(name.rfind('-') + 1).c_str())) > 0;
      }
      return false;
    };
    sink.put = [&](const std::string& name, const std::vector<float>& data, const std::vector<int64_t>& ne) {
      auto it = dumps.find(name);
      if (it == dumps.end()) {
        order.push_back(name);
        dumps[name] = {data, ne};
        return;
      }
      it->second.first.insert(it->second.first.end(), data.begin(), data.end());
      it->second.second.back() += ne.back();
    };

    auto rows_in = [&](int64_t a, int64_t b) { return std::max<int64_t>(0, b - std::max(a, n - logits_last)); };
    std::vector<float> logits, twin_logits;
    double first_step = 0;
    t0 = std::chrono::steady_clock::now();
    // With --twin, slot 1 runs step i - 1's range alongside slot 0's step i (and one more step at the end).
    for (size_t i = 0; i < steps.size() + (twin ? 1 : 0); ++i) {
      std::vector<engine::Chunk> chunks;
      bool main_in = i < steps.size();
      if (main_in) {
        auto [a, b] = steps[i];
        chunks.push_back({0, a, std::vector<int32_t>(tokens.begin() + a, tokens.begin() + b), rows_in(a, b)});
      }
      if (twin && i > 0) {
        auto [a, b] = steps[i - 1];
        chunks.push_back({1, a, std::vector<int32_t>(tokens.begin() + a, tokens.begin() + b), rows_in(a, b)});
      }
      auto t1 = std::chrono::steady_clock::now();
      auto l = eng.step(chunks, twin ? nullptr : &sink);
      if (i == 0) first_step = since(t1);
      size_t off = 0;
      if (main_in) {
        const size_t r = size_t(chunks[0].logits_rows) * size_t(V);
        logits.insert(logits.end(), l.begin(), l.begin() + std::ptrdiff_t(r));
        off = r;
      }
      if (twin && i > 0) twin_logits.insert(twin_logits.end(), l.begin() + std::ptrdiff_t(off), l.end());
    }
    std::printf("%zu steps over %lld tokens: %.2f s (first step %.2f s)\n", steps.size(), (long long)n, since(t0), first_step);
    if (eng.experts_on_card() < eng.model().n_expert)
      std::printf("experts on the card: %lld of %lld per layer; %.2f%% of %llu routed pairs chose cold experts\n", (long long)eng.experts_on_card(),
                  (long long)eng.model().n_expert, eng.pairs() ? 100.0 * double(eng.cold_pairs()) / double(eng.pairs()) : 0.0,
                  (unsigned long long)eng.pairs());
    if (eng.model().n_expert > eng.experts_on_card()) {
      const auto& cs = eng.cold_stats();
      std::printf("cold experts: %.2f s waiting on the card's routing, %.2f s computing on the CPU over %llu layers (%llu with the card finished first)\n",
                  cs.wait_s, cs.cpu_s, (unsigned long long)cs.layers, (unsigned long long)cs.late_layers);
      const auto& ss = eng.stream_stats();
      std::printf("streamed: %d slots; %llu experts in %llu layers carried %llu of %llu cold pairs (%.1f%%; the CPU took the rest); copies %.2f s (beside the CPU), waited %.2f s for them after the CPU\n",
                  eng.stream_slots(), (unsigned long long)ss.experts, (unsigned long long)ss.layers, (unsigned long long)ss.pairs,
                  (unsigned long long)eng.cold_pairs(), eng.cold_pairs() ? 100.0 * double(ss.pairs) / double(eng.cold_pairs()) : 0.0, ss.copy_s, ss.wait_s);
    }

    if (twin) {
      if (twin_logits.size() != logits.size()) throw std::runtime_error("twin produced a different number of logits");
      const size_t rows = logits.size() / size_t(V);
      double sum = 0, worst = 0;
      size_t same = 0;
      for (size_t r = 0; r < rows; ++r) {
        const float *a = logits.data() + r * V, *b = twin_logits.data() + r * V;
        const double d = kl(a, b, V);
        sum += d;
        worst = std::max(worst, d);
        same += std::max_element(a, a + V) == a + (std::max_element(b, b + V) - b);
      }
      std::printf("twin slot: top-1 agreement %zu/%zu, mean KL %.3g, worst %.3g\n", same, rows, sum / double(rows), worst);
      if (sum / double(rows) > twin_tol) throw std::runtime_error("twin slot disagrees with the first");
    }

    auto write_dump = [&](const std::string& dir, const std::vector<float>& lg, bool tensors) {
      json::Writer index;
      index.begin_object().kv("n_tokens", n).kv("n_vocab", V);
      index.key("tensors").begin_array();
      for (const auto& name : tensors ? order : std::vector<std::string>{}) {
        const auto& [data, ne] = dumps[name];
        std::ofstream(dir + "/" + name + ".f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
        index.begin_object().kv("name", name).kv("type", "f32").key("ne").begin_array();
        for (size_t k = 0; k < 4; ++k) index.value(k < ne.size() ? ne[k] : int64_t(1));
        index.end_array().kv("file", name + ".f32").end_object();
      }
      std::ofstream(dir + "/logits.f32", std::ios::binary)
          .write(reinterpret_cast<const char*>(lg.data()), std::streamsize(lg.size() * sizeof(float)));
      index.end_array().kv("logits_rows", int64_t(lg.size() / size_t(V))).end_object();
      std::ofstream(dir + "/index.json") << index.str() << "\n";
    };
    write_dump(out, logits, true);
    if (twin) {
      std::filesystem::create_directories(out + "/twin");
      write_dump(out + "/twin", twin_logits, false);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-engine-run: %s\n", e.what());
    return 1;
  }
  return 0;
}
