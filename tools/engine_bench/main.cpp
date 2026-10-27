// eklow-engine-bench: the engine's speed on synthetic prompts (random token ids; the work does not
// depend on them).
//
//   eklow-engine-bench --model GGUF [--prefill 1024,4096,16384] [--chunk 2048]
//                      [--depth 0,4096,16384] [--decode-ctx 0,4096,16384] [--decode-steps 64]
//                      [--slots N] [--profile] [--kv-int8] [--json FILE]
//
// Prefill: a fresh sequence of N tokens in chunks of --chunk; tokens per second over the whole
// prompt. Depth (llama-bench's -d): the sequence first filled to D tokens untimed, then two chunks
// timed; the rate of prefill at that depth. Decode: every slot prefilled to CTX tokens, then --decode-steps steps of one token per
// slot; tokens per second per sequence. --profile adds the card's time per section of the forward
// pass, from timestamps of marker kernels between sections.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "engine/engine.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

std::vector<int64_t> list(const std::string& s) {
  std::vector<int64_t> out;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, ',');)
    if (!p.empty()) out.push_back(std::atoll(p.c_str()));
  return out;
}

double since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void print_profile(const std::map<std::string, double>& p, json::Writer& w) {
  double total = 0;
  for (const auto& [k, v] : p) total += v;
  std::vector<std::pair<double, std::string>> rows;
  for (const auto& [k, v] : p) rows.push_back({v, k});
  std::sort(rows.rbegin(), rows.rend());
  w.key("profile_s").begin_object();
  for (const auto& [v, k] : rows) {
    std::printf("    %-16s %9.1f ms %5.1f%%\n", k.c_str(), v * 1e3, 100 * v / total);
    w.kv(k, v);
  }
  w.end_object();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, json_path;
  std::vector<int64_t> prefill = {1024, 4096, 16384}, decode_ctx = {0, 4096, 16384}, depth;
  int64_t chunk = 2048, decode_steps = 64;
  int slots = 1;
  bool profile = false, kv_int8 = false;
  std::string expert_profile;
  int64_t experts_on_card = -1;
  int cpu_threads = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--prefill") prefill = list(next());
    else if (a == "--depth") depth = list(next());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--decode-ctx") decode_ctx = list(next());
    else if (a == "--decode-steps") decode_steps = std::atoll(next().c_str());
    else if (a == "--slots") slots = std::atoi(next().c_str());
    else if (a == "--profile") profile = true;
    else if (a == "--json") json_path = next();
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--expert-profile") expert_profile = next();
    else if (a == "--experts-on-card") experts_on_card = std::atoll(next().c_str());
    else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-engine-bench --model GGUF [--prefill N,...] [--chunk N] [--depth N,...] [--decode-ctx N,...] "
                           "[--decode-steps N] [--slots N] [--profile] [--kv-int8] [--json FILE] [--expert-profile JSON --experts-on-card N] [--cpu-threads N]\n");
      return 2;
    }
  }
  try {
    int64_t longest = 1;
    for (auto n : prefill) longest = std::max(longest, n);
    for (auto n : depth) longest = std::max(longest, n + 2 * chunk);
    for (auto n : decode_ctx) longest = std::max(longest, n + decode_steps);
    engine::Config ec;
    ec.n_slots = slots;
    ec.max_ctx = (longest + 255) / 256 * 256;
    ec.max_batch = std::max<int64_t>(chunk, slots);
    ec.max_chunk = chunk;
    ec.int8_global_kv = kv_int8;
    ec.expert_profile = expert_profile;
    ec.experts_on_card = experts_on_card;
    ec.cpu_threads = cpu_threads;
    ec.max_logits = slots;

    auto f = gguf::File::open(model);
    sycl::queue q = profile ? sycl::queue(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()})
                            : sycl::queue(rt::pick_device(), sycl::property::queue::in_order());
    engine::Engine eng(q, f, ec);
    std::printf("%s: %.2f GiB of weights, %.2f GiB of KV cache (%d slots of %lld)\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), eng.weight_bytes() / 1073741824.0,
                eng.cache_bytes() / 1073741824.0, slots, (long long)ec.max_ctx);
    const bool cold = eng.experts_on_card() < eng.model().n_expert;
    if (cold) std::printf("experts on the card: %lld of %lld per layer, the rest on the CPU\n", (long long)eng.experts_on_card(), (long long)eng.model().n_expert);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> id(256, 200000);
    auto tokens = [&](int64_t n) {
      std::vector<int32_t> t(static_cast<size_t>(n));
      for (auto& x : t) x = id(rng);
      return t;
    };
    std::map<std::string, double> prof;
    auto fill = [&](int slot, int64_t n) {
      const int64_t start = eng.length(slot);
      for (int64_t s = start; s < start + n; s += chunk) {
        const int64_t c = std::min(chunk, start + n - s);
        eng.step({engine::Chunk{slot, s, tokens(c), s + c == start + n ? 1 : 0}});
      }
    };
    fill(0, std::min<int64_t>(chunk, 512));  // warm-up: first launches, scratch allocations
    eng.release(0);

    json::Writer w;
    w.begin_object().kv("device", q.get_device().get_info<sycl::info::device::name>()).kv("chunk", chunk).kv("slots", int64_t(slots));
    w.key("prefill").begin_array();
    for (int64_t n : prefill) {
      prof.clear();
      eng.set_profile(profile ? &prof : nullptr);
      auto t0 = std::chrono::steady_clock::now();
      fill(0, n);
      const double s = since(t0);
      eng.set_profile(nullptr);
      eng.release(0);
      std::printf("prefill %6lld tokens: %7.2f s, %8.1f tok/s\n", (long long)n, s, double(n) / s);
      w.begin_object().kv("tokens", n).kv("seconds", s).kv("tok_per_s", double(n) / s);
      if (profile) print_profile(prof, w);
      w.end_object();
    }
    w.end_array().key("depth").begin_array();
    for (int64_t d : depth) {
      fill(0, d);
      prof.clear();
      eng.set_profile(profile ? &prof : nullptr);
      auto t0 = std::chrono::steady_clock::now();
      fill(0, 2 * chunk);
      const double s = since(t0);
      eng.set_profile(nullptr);
      eng.release(0);
      std::printf("prefill at depth %6lld: %7.2f s for %lld tokens, %8.1f tok/s\n", (long long)d, s, (long long)(2 * chunk),
                  double(2 * chunk) / s);
      w.begin_object().kv("depth", d).kv("tokens", 2 * chunk).kv("seconds", s).kv("tok_per_s", double(2 * chunk) / s);
      if (profile) print_profile(prof, w);
      w.end_object();
    }
    w.end_array().key("decode").begin_array();
    for (int64_t ctx : decode_ctx) {
      for (int sl = 0; sl < slots; ++sl)
        if (ctx > 0) fill(sl, ctx);
      prof.clear();
      eng.set_profile(profile ? &prof : nullptr);
      double enqueue = 0;
      const auto cs0 = eng.cold_stats();
      const uint64_t pr0 = eng.pairs(), cp0 = eng.cold_pairs();
      auto t0 = std::chrono::steady_clock::now();
      for (int64_t i = 0; i < decode_steps; ++i) {
        std::vector<engine::Chunk> step;
        for (int sl = 0; sl < slots; ++sl) step.push_back(engine::Chunk{sl, eng.length(sl), tokens(1), 1});
        eng.step(step);
        enqueue += eng.enqueue_seconds();
      }
      const double s = since(t0);
      eng.set_profile(nullptr);
      for (int sl = 0; sl < slots; ++sl) eng.release(sl);
      std::printf("decode at %6lld: %7.2f ms/step (%.2f ms enqueueing), %6.1f tok/s per sequence\n", (long long)ctx,
                  1e3 * s / double(decode_steps), 1e3 * enqueue / double(decode_steps), double(decode_steps) / s);
      if (cold) {
        const auto cs = eng.cold_stats();
        const double n = double(decode_steps);
        std::printf("    cold experts per step: %.2f ms waiting on the card's routing, %.2f ms computing on the CPU, %.1f%% of pairs on the CPU; "
                    "%.1f of %.1f cold layers had the card finished before the CPU\n",
                    1e3 * (cs.wait_s - cs0.wait_s) / n, 1e3 * (cs.cpu_s - cs0.cpu_s) / n,
                    100.0 * double(eng.cold_pairs() - cp0) / double(std::max<uint64_t>(1, eng.pairs() - pr0)),
                    double(cs.late_layers - cs0.late_layers) / n, double(cs.layers - cs0.layers) / n);
      }
      w.begin_object().kv("ctx", ctx).kv("ms_per_step", 1e3 * s / double(decode_steps)).kv("enqueue_ms_per_step", 1e3 * enqueue / double(decode_steps));
      w.kv("tok_per_s", double(decode_steps) / s);
      if (profile) print_profile(prof, w);
      w.end_object();
    }
    w.end_array().end_object();
    if (!json_path.empty()) std::ofstream(json_path) << w.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-engine-bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
