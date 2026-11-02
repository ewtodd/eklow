// eklow-engine-bench: the Qwen3.6 engine's speed on synthetic prompts (random token ids, or the ids of a real text: the work
// depends on them only through the routing).
//
//   eklow-engine-bench --model GGUF [--prefill 1024,4096,16384] [--chunk 2048] [--depth 0,4096,16384]
//                      [--decode-ctx 0,4096,16384] [--decode-slots 1,2] [--decode-steps 64] [--verify 8,16] [--verify-ctx 8192]
//                      [--verify-steps 24] [--slots N] [--max-ctx N] [--profile] [--kv-int8] [--no-capture] [--json FILE]
//                      [--tokens-json FILE] [--expert-profile JSON --experts-on-card N] [--cpu-threads N] [--stream-slots N]
//                      [--stream-min-rows N] [--stream-copy-us US] [--stream-cpu-us US] [--head-fp32] [--embed-on-card]
//
// The engine has --slots slots (default 2) of --max-ctx positions (default: what the runs need; the production shape is
// --slots 2 --max-ctx 262144 --kv-int8 --experts-on-card -2, which is what the placement then counts).
// Prefill: a fresh sequence of N tokens in chunks of --chunk; tokens per second over the whole prompt. Depth (llama-bench's -d):
// the sequence first filled to D tokens untimed, then two chunks timed. Decode: --decode-slots slots (each) prefilled to CTX
// tokens, then --decode-steps steps of one token per slot; tokens per second per sequence. Verify: one slot filled to
// --verify-ctx, then --verify-steps blocks of B tokens (verify chunks with all B logits rows and the capture, each accepting
// about half of the previous block): ms a step and the accepted tokens a second. --profile adds the card's time per section.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
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
    std::printf("    %-18s %9.1f ms %5.1f%%\n", k.c_str(), v * 1e3, 100 * v / total);
    w.kv(k, v);
  }
  w.end_object();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, json_path, tokens_json;
  std::vector<int64_t> prefill = {1024, 4096, 16384}, decode_ctx = {0, 4096, 16384}, depth, decode_slots = {1}, verify, verify_ctx = {8192};
  int64_t chunk = 2048, decode_steps = 64, verify_steps = 24, max_ctx = 0;
  int64_t group = 1;  // sub-chunks of --chunk tokens run layer-major as one step (Config::max_group)
  int slots = 2;
  bool profile = false;
  engine::Config ec;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--prefill") prefill = list(next());
    else if (a == "--depth") depth = list(next());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--group") group = std::max<int64_t>(1, std::atoll(next().c_str()));
    else if (a == "--decode-ctx") decode_ctx = list(next());
    else if (a == "--decode-slots") decode_slots = list(next());
    else if (a == "--decode-steps") decode_steps = std::atoll(next().c_str());
    else if (a == "--verify") verify = list(next());
    else if (a == "--verify-ctx") verify_ctx = list(next());
    else if (a == "--verify-steps") verify_steps = std::atoll(next().c_str());
    else if (a == "--slots") slots = std::atoi(next().c_str());
    else if (a == "--max-ctx") max_ctx = std::atoll(next().c_str());
    else if (a == "--profile") profile = true;
    else if (a == "--json") json_path = next();
    else if (a == "--tokens-json") tokens_json = next();
    else if (a == "--kv-int8") ec.int8_kv = true;
    else if (a == "--no-capture") ec.capture_layers.clear();
    else if (a == "--head-fp32") ec.head_dpas = false;
    else if (a == "--embed-on-card") ec.embed_on_card = true;
    else if (a == "--expert-profile") ec.expert_profile = next();
    else if (a == "--experts-on-card") ec.experts_on_card = std::atoll(next().c_str());
    else if (a == "--cpu-threads") ec.cpu_threads = std::atoi(next().c_str());
    else if (a == "--stream-slots") ec.stream_slots = std::atoi(next().c_str());
    else if (a == "--stream-min-rows") ec.stream_min_rows = std::atoi(next().c_str());
    else if (a == "--stream-copy-us") ec.stream_copy_us = std::atof(next().c_str());
    else if (a == "--stream-cpu-us") ec.stream_cpu_us = std::atof(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-engine-bench --model GGUF [--prefill N,...] [--chunk N] [--group N] [--depth N,...] [--decode-ctx N,...] "
                           "[--decode-slots N,...] [--decode-steps N] [--verify B,...] [--verify-ctx N,...] [--verify-steps N] [--slots N] "
                           "[--max-ctx N] [--profile] [--kv-int8] [--no-capture] [--json FILE] [--tokens-json FILE] "
                           "[--expert-profile JSON --experts-on-card N] [--cpu-threads N] [--stream-slots N] [--stream-min-rows N] "
                           "[--stream-copy-us US] [--stream-cpu-us US] [--head-fp32] [--embed-on-card]\n");
      return 2;
    }
  }
  try {
    int64_t longest = 1;
    for (auto n : prefill) longest = std::max(longest, n);
    for (auto n : depth) longest = std::max(longest, n + std::max<int64_t>(2, group) * chunk);
    for (auto n : decode_ctx) longest = std::max(longest, n + decode_steps);
    for (auto n : verify_ctx) longest = std::max(longest, n + verify_steps * 16 + 16);
    ec.n_slots = slots;
    ec.max_ctx = max_ctx > 0 ? max_ctx : (longest + 255) / 256 * 256;
    ec.max_batch = std::max<int64_t>(chunk, 16 * slots);
    ec.max_chunk = std::max<int64_t>(chunk, 16);
    ec.max_group = ec.max_chunk * group;
    ec.max_logits = 16 * slots;
    ec.max_capture_rows = std::min<int64_t>(ec.max_batch, 2048);

    auto f = gguf::File::open(model);
    sycl::queue q = profile ? sycl::queue(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()})
                            : sycl::queue(rt::pick_device(), sycl::property::queue::in_order());
    engine::Engine eng(q, f, ec);
    std::printf("%s: %d slots of %lld; %s\n", q.get_device().get_info<sycl::info::device::name>().c_str(), slots, (long long)ec.max_ctx,
                eng.placement_line().c_str());
    const bool cold = eng.experts_on_card() < eng.model().n_expert;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int32_t> id(256, 200000);
    std::vector<int32_t> text;
    if (!tokens_json.empty()) {
      std::ifstream in(tokens_json);
      const json::Value doc = json::parse(std::string((std::istreambuf_iterator<char>(in)), {}));
      const json::Value* ids = doc.find("ids");
      if (!ids || !ids->is_array() || ids->items().empty()) throw std::runtime_error(tokens_json + ": no ids");
      for (const auto& v : ids->items()) text.push_back(int32_t(std::atoll(v.int_digits().c_str())));
    }
    auto tokens = [&](int64_t n) {
      std::vector<int32_t> t(static_cast<size_t>(n));
      size_t at = text.empty() ? 0 : size_t(rng()) % text.size();
      for (auto& x : t) x = text.empty() ? id(rng) : text[at++ % text.size()];
      return t;
    };
    std::map<std::string, double> prof;
    auto fill = [&](int slot, int64_t n) {
      const int64_t start = eng.length(slot);
      for (int64_t s = start; s < start + n; s += chunk * group) {
        const int64_t c = std::min(chunk * group, start + n - s);
        eng.step({engine::Chunk{slot, s, tokens(c), s + c == start + n ? 1 : 0}});
      }
    };
    auto cold_report = [&](const engine::Engine::ColdStats& cs0, uint64_t pr0, uint64_t cp0, const engine::Engine::StreamStats& ss0, double steps,
                           bool per_step) {
      if (!cold) return;
      const auto cs = eng.cold_stats();
      const auto ss = eng.stream_stats();
      if (per_step)
        std::printf("    cold experts per step: %.2f ms waiting on the card's routing, %.2f ms computing on the CPU, %.1f%% of pairs on the CPU; "
                    "%.1f of %.1f cold layers had the card finished before the CPU; CPU phases gate|up %.2f, glu %.2f, down %.2f ms, %.1f distinct experts a step\n",
                    1e3 * (cs.wait_s - cs0.wait_s) / steps, 1e3 * (cs.cpu_s - cs0.cpu_s) / steps,
                    100.0 * double(eng.cold_pairs() - cp0) / double(std::max<uint64_t>(1, eng.pairs() - pr0)),
                    double(cs.late_layers - cs0.late_layers) / steps, double(cs.layers - cs0.layers) / steps,
                    1e3 * (cs.gu_s - cs0.gu_s) / steps, 1e3 * (cs.glu_s - cs0.glu_s) / steps, 1e3 * (cs.dn_s - cs0.dn_s) / steps, double(cs.experts - cs0.experts) / steps);
      else
        std::printf("    cold experts: %.2f s waiting on the card's routing, %.2f s computing on the CPU, %.1f%% of pairs cold, %llu of %llu layers with the card finished first; "
                    "streamed %llu experts in %llu layers (%.1f%% of the cold pairs)\n",
                    cs.wait_s - cs0.wait_s, cs.cpu_s - cs0.cpu_s, 100.0 * double(eng.cold_pairs() - cp0) / double(std::max<uint64_t>(1, eng.pairs() - pr0)),
                    (unsigned long long)(cs.late_layers - cs0.late_layers), (unsigned long long)(cs.layers - cs0.layers),
                    (unsigned long long)(ss.experts - ss0.experts), (unsigned long long)(ss.layers - ss0.layers),
                    100.0 * double(ss.pairs - ss0.pairs) / double(std::max<uint64_t>(1, eng.cold_pairs() - cp0)));
    };
    fill(0, std::min<int64_t>(chunk, 512));  // warm-up: first launches, scratch allocations
    if (group > 1) {
      fill(0, 2 * chunk);  // and the layer-major path's
      eng.release(0);
    }
    auto group_report = [&](const engine::Engine::GroupStats& g0) {
      const auto g = eng.group_stats();
      if (group <= 1 || g.layers == g0.layers) return;
      const double nl = double(g.layers - g0.layers);
      std::printf("    layer-major: %llu group layers, per layer %.1f streamed experts in %.2f waves, %.0f streamed pairs, %.0f pairs on the CPU; host waits: routing %.1f ms, "
                  "repack before the next copy %.1f ms; CPU task %.1f ms (of which %.1f ms waited for after the waves)\n",
                  (unsigned long long)(g.layers - g0.layers), double(g.experts - g0.experts) / nl, double(g.waves - g0.waves) / nl,
                  double(g.streamed_pairs - g0.streamed_pairs) / nl, double(g.cpu_pairs - g0.cpu_pairs) / nl, 1e3 * (g.ids_wait_s - g0.ids_wait_s) / nl,
                  1e3 * (g.copy_wait_s - g0.copy_wait_s) / nl, 1e3 * (g.cpu_s - g0.cpu_s) / nl, 1e3 * (g.join_s - g0.join_s) / nl);
    };
    eng.release(0);

    json::Writer w;
    w.begin_object().kv("device", q.get_device().get_info<sycl::info::device::name>()).kv("chunk", chunk).kv("slots", int64_t(slots));
    w.kv("placement", eng.placement_line());
    w.key("prefill").begin_array();
    for (int64_t n : prefill) {
      prof.clear();
      eng.set_profile(profile ? &prof : nullptr);
      const auto ss0 = eng.stream_stats();
      const auto gs0 = eng.group_stats();
      const auto cs0 = eng.cold_stats();
      const uint64_t pr0 = eng.pairs(), cp0 = eng.cold_pairs();
      auto t0 = std::chrono::steady_clock::now();
      fill(0, n);
      const double s = since(t0);
      eng.set_profile(nullptr);
      eng.release(0);
      std::printf("prefill %6lld tokens: %7.2f s, %8.1f tok/s\n", (long long)n, s, double(n) / s);
      cold_report(cs0, pr0, cp0, ss0, 1, false);
      group_report(gs0);
      w.begin_object().kv("tokens", n).kv("seconds", s).kv("tok_per_s", double(n) / s);
      if (profile) print_profile(prof, w);
      w.end_object();
    }
    w.end_array().key("depth").begin_array();
    for (int64_t d : depth) {
      const int64_t timed = group > 1 ? group * chunk : 2 * chunk;  // plain: two chunks; grouped: one group
      fill(0, d);
      prof.clear();
      eng.set_profile(profile ? &prof : nullptr);
      const auto ss0 = eng.stream_stats();
      const auto gs0 = eng.group_stats();
      const auto cs0 = eng.cold_stats();
      const uint64_t pr0 = eng.pairs(), cp0 = eng.cold_pairs();
      auto t0 = std::chrono::steady_clock::now();
      fill(0, timed);
      const double s = since(t0);
      eng.set_profile(nullptr);
      eng.release(0);
      std::printf("prefill at depth %6lld: %7.2f s for %lld tokens, %8.1f tok/s\n", (long long)d, s, (long long)timed, double(timed) / s);
      cold_report(cs0, pr0, cp0, ss0, 1, false);
      group_report(gs0);
      w.begin_object().kv("depth", d).kv("tokens", timed).kv("seconds", s).kv("tok_per_s", double(timed) / s);
      if (profile) print_profile(prof, w);
      w.end_object();
    }
    w.end_array().key("decode").begin_array();
    for (int64_t ns : decode_slots) {
      for (int64_t ctx : decode_ctx) {
        for (int sl = 0; sl < ns; ++sl)
          if (ctx > 0) fill(sl, ctx);
        prof.clear();
        eng.set_profile(profile ? &prof : nullptr);
        double enqueue = 0;
        const auto ss0 = eng.stream_stats();
        const auto cs0 = eng.cold_stats();
        const uint64_t pr0 = eng.pairs(), cp0 = eng.cold_pairs();
        auto t0 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < decode_steps; ++i) {
          std::vector<engine::Chunk> step;
          for (int sl = 0; sl < ns; ++sl) step.push_back(engine::Chunk{sl, eng.length(sl), tokens(1), 1});
          eng.step(step);
          enqueue += eng.enqueue_seconds();
        }
        const double s = since(t0);
        eng.set_profile(nullptr);
        for (int sl = 0; sl < ns; ++sl) eng.release(sl);
        std::printf("decode, %lld slot%s, at %6lld: %7.2f ms/step (%.2f ms enqueueing), %6.1f tok/s per sequence\n", (long long)ns, ns == 1 ? "" : "s",
                    (long long)ctx, 1e3 * s / double(decode_steps), 1e3 * enqueue / double(decode_steps), double(decode_steps) / s);
        cold_report(cs0, pr0, cp0, ss0, double(decode_steps), true);
        w.begin_object().kv("slots", ns).kv("ctx", ctx).kv("ms_per_step", 1e3 * s / double(decode_steps)).kv("enqueue_ms_per_step", 1e3 * enqueue / double(decode_steps));
        w.kv("tok_per_s", double(decode_steps) / s);
        if (profile) print_profile(prof, w);
        w.end_object();
      }
    }
    w.end_array().key("verify").begin_array();
    for (int64_t B : verify) {
      for (int64_t ctx : verify_ctx) {
        fill(0, ctx);
        // Each block accepts about half of its tokens (3, 5 or 8 of 8; 6 to 11 of 16), the next block starting after them.
        const std::vector<int64_t> pattern = B == 16 ? std::vector<int64_t>{6, 9, 11, 8} : std::vector<int64_t>{3, 5, 4, 5};
        auto block = [&](int& acc_prev, int64_t& at, int64_t i) {
          engine::Chunk c{0, at, tokens(B), B};
          c.verify = true;
          c.accept = acc_prev;
          if (!ec.capture_layers.empty()) c.capture_rows = B;
          eng.step({c});
          acc_prev = int(std::min(B, pattern[size_t(i) % pattern.size()]));
          at += acc_prev;
        };
        int acc_prev = -1;
        int64_t at = eng.length(0), accepted = 0;
        for (int64_t i = 0; i < 3; ++i) block(acc_prev, at, i);  // warm-up
        prof.clear();
        eng.set_profile(profile ? &prof : nullptr);
        const auto ss0 = eng.stream_stats();
        const auto cs0 = eng.cold_stats();
        const uint64_t pr0 = eng.pairs(), cp0 = eng.cold_pairs();
        auto t0 = std::chrono::steady_clock::now();
        for (int64_t i = 0; i < verify_steps; ++i) {
          block(acc_prev, at, i);
          accepted += acc_prev;
        }
        const double s = since(t0);
        eng.set_profile(nullptr);
        eng.release(0);
        std::printf("verify blocks of %2lld at %6lld: %7.2f ms/step, %6.1f accepted tokens/s (%.1f tokens accepted a block)\n", (long long)B, (long long)ctx,
                    1e3 * s / double(verify_steps), double(accepted) / s, double(accepted) / double(verify_steps));
        cold_report(cs0, pr0, cp0, ss0, double(verify_steps), true);
        w.begin_object().kv("block", B).kv("ctx", ctx).kv("ms_per_step", 1e3 * s / double(verify_steps)).kv("accepted_per_s", double(accepted) / s);
        if (profile) print_profile(prof, w);
        w.end_object();
      }
    }
    w.end_array().end_object();
    if (!json_path.empty()) std::ofstream(json_path) << w.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-engine-bench: %s\n", e.what());
    return 1;
  }
  return 0;
}
