// eklow-engine-run: the serving engine over one prompt, written in eklow-oracle's dump format so
// tools/ref/compare_dumps.py can set it beside the oracle or the reference forward.
//
//   eklow-engine-run --model GGUF --out DIR (--tokens-json FILE | --tokens 2,105,...)
//                    [--tensors PREFIX,...] [--logits-last N] [--chunk N] [--decode-last N]
//                    [--verify-last N --verify-block B --accept A,...] [--max-ctx N] [--kv-int8]
//                    [--expert-profile JSON --experts-on-card N] [--cpu-threads N] [--stream-slots N]
//                    [--stream-min-rows N] [--stream-copy-us US] [--stream-cpu-us US] [--head-fp32] [--embed-on-card]
//                    [--record-routing FILE] [--capture-check]
//
// The prompt is fed in prefill chunks of --chunk tokens (default: one chunk), then --verify-last tokens as verify blocks of
// --verify-block tokens whose acceptance follows the pattern --accept (a count per block, cycled; the unaccepted tokens are fed
// again in the next block; the last block accepts all), then the last --decode-last tokens one per step, as decode would.
// Logits come from the step in which a position was accepted. With verify blocks only the logits are written (the tensors of a
// re-fed token would appear twice).
// --record-routing writes the (layer, expert) selection counts of the whole run as a routing profile (DECISIONS: data/*-experts.json).
// --capture-check asks every chunk for its hidden-state capture and compares layer 1's with l_out-1 in the dumps.
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

}  // namespace

int main(int argc, char** argv) {
  std::string model, out, tokens_json, tokens_arg, record_routing;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  std::vector<int64_t> accept_pattern = {3};
  int64_t logits_last = 1, chunk = 0, decode_last = 0, verify_last = 0, verify_block = 8, max_ctx = 0;
  bool kv_int8 = false, head_fp32 = false, embed_on_card = false, capture_check = false;
  engine::Config ec;
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
    else if (a == "--verify-last") verify_last = std::atoll(next().c_str());
    else if (a == "--verify-block") verify_block = std::atoll(next().c_str());
    else if (a == "--accept") {
      accept_pattern.clear();
      for (auto& s : split(next(), ',')) accept_pattern.push_back(std::atoll(s.c_str()));
    }
    else if (a == "--max-ctx") max_ctx = std::atoll(next().c_str());
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--head-fp32") head_fp32 = true;
    else if (a == "--embed-on-card") embed_on_card = true;
    else if (a == "--capture-check") capture_check = true;
    else if (a == "--record-routing") record_routing = next();
    else if (a == "--expert-profile") ec.expert_profile = next();
    else if (a == "--experts-on-card") ec.experts_on_card = std::atoll(next().c_str());
    else if (a == "--cpu-threads") ec.cpu_threads = std::atoi(next().c_str());
    else if (a == "--stream-slots") ec.stream_slots = std::atoi(next().c_str());
    else if (a == "--stream-min-rows") ec.stream_min_rows = std::atoi(next().c_str());
    else if (a == "--stream-copy-us") ec.stream_copy_us = std::atof(next().c_str());
    else if (a == "--stream-cpu-us") ec.stream_cpu_us = std::atof(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-engine-run --model GGUF --out DIR (--tokens-json FILE | --tokens IDS) [--tensors PREFIX,...] "
                           "[--logits-last N] [--chunk N] [--decode-last N] [--verify-last N --verify-block B --accept A,...] [--max-ctx N] "
                           "[--kv-int8] [--expert-profile JSON --experts-on-card N] [--cpu-threads N] [--stream-slots N] [--stream-min-rows N] "
                           "[--stream-copy-us US] [--stream-cpu-us US] [--head-fp32] [--embed-on-card] [--record-routing FILE] [--capture-check]\n");
      return 2;
    }
  }
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
    for (int64_t a : accept_pattern)
      if (a < 1) throw std::runtime_error("--accept counts must be at least 1");
    const int64_t n = int64_t(tokens.size());
    logits_last = std::min(logits_last, n);
    decode_last = std::min(decode_last, n - 1);
    verify_last = std::min(verify_last, n - 1 - decode_last);
    const int64_t prefill = n - decode_last - verify_last;
    if (chunk <= 0) chunk = std::max<int64_t>(prefill, 1);
    if (verify_last) prefixes.clear();

    ec.n_slots = 1;
    ec.max_ctx = max_ctx > 0 ? max_ctx : (n + 255) / 256 * 256;
    ec.max_batch = std::max<int64_t>(chunk, 16);
    ec.max_chunk = ec.max_batch;
    ec.max_logits = std::max<int64_t>(std::min(ec.max_batch, logits_last), verify_block);
    ec.int8_kv = kv_int8;
    ec.head_dpas = !head_fp32;
    ec.embed_on_card = embed_on_card;
    ec.record_routing = !record_routing.empty();
    ec.max_capture_rows = ec.max_batch;
    // Placement is the caller's: all on the card when nothing is given and everything fits is not the case for this model.
    auto t0 = std::chrono::steady_clock::now();
    auto f = gguf::File::open(model);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::Engine eng(q, f, ec);
    const int64_t V = eng.model().n_vocab, E = eng.model().n_embd;
    std::printf("loaded in %.1f s on %s: %s\n", since(t0), q.get_device().get_info<sycl::info::device::name>().c_str(), eng.placement_line().c_str());

    // Each step's tensors are that step's tokens; the dumps concatenate them along the token axis.
    std::map<std::string, std::pair<std::vector<float>, std::vector<int64_t>>> dumps;
    std::vector<std::string> order;
    model::DumpSink sink;
    sink.wants = [&](const std::string& name) {
      for (const auto& p : prefixes)
        if (name.compare(0, p.size(), p) == 0) return true;
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

    // Logits of the last logits_last positions, each from the step in which the position was accepted.
    std::vector<float> logits(size_t(logits_last * V), 0.0f);
    const int64_t first_pos = n - logits_last;
    auto keep = [&](int64_t pos, const float* row) {
      if (pos >= first_pos) std::copy(row, row + V, logits.begin() + (pos - first_pos) * V);
    };
    double max_cap_diff = 0;
    auto check_capture = [&](int64_t rows_before) {
      // The capture of layer capture_layers[0] against the dump of its l_out, over the rows this step captured.
      const auto& cap = eng.capture();
      if (!capture_check || cap.rows == 0) return;
      const int l0 = eng.config().capture_layers[0];
      const auto& d = dumps["l_out-" + std::to_string(l0)];
      std::vector<float> host(size_t(cap.rows * cap.width));
      q.memcpy(host.data(), cap.data, host.size() * 4).wait();
      for (int64_t r = 0; r < cap.rows; ++r)
        for (int64_t j = 0; j < E; ++j) {
          const double a = host[size_t(r * cap.width + j)], b = d.first[size_t((rows_before + cap.ids[size_t(r)].pos - cap.ids[0].pos) * E + j)];
          max_cap_diff = std::max(max_cap_diff, std::fabs(a - b));
        }
    };

    t0 = std::chrono::steady_clock::now();
    int64_t steps = 0, pos = 0, dumped_rows = 0;
    double first_step = 0;
    auto run = [&](engine::Chunk c, int64_t first_logit_pos) {
      const auto t1 = std::chrono::steady_clock::now();
      if (capture_check) c.capture_rows = std::min<int64_t>(int64_t(c.tokens.size()), ec.max_capture_rows);
      const std::vector<float> l = eng.step({c}, prefixes.empty() ? nullptr : &sink);
      if (steps++ == 0) first_step = since(t1);
      if (capture_check) check_capture(dumped_rows);
      dumped_rows += int64_t(c.tokens.size());
      return l;
    };
    // Prefill chunks.
    for (int64_t s = 0; s < prefill; s += chunk) {
      const int64_t e = std::min(prefill, s + chunk);
      engine::Chunk c{0, s, std::vector<int32_t>(tokens.begin() + s, tokens.begin() + e), std::max<int64_t>(0, e - std::max(s, first_pos))};
      const std::vector<float> l = run(c, s);
      for (int64_t r = 0; r < c.logits_rows; ++r) keep(e - c.logits_rows + r, l.data() + r * V);
      pos = e;
    }
    // Verify blocks: B tokens, the first `a` accepted, the rest fed again.
    size_t block = 0;
    const int64_t verify_end = prefill + verify_last;
    int accept_prev = -1;
    while (pos < verify_end) {
      const int64_t b = std::min(verify_block, verify_end - pos);
      engine::Chunk c{0, pos, std::vector<int32_t>(tokens.begin() + pos, tokens.begin() + pos + b), b};
      c.verify = true;
      c.accept = accept_prev;
      const std::vector<float> l = run(c, pos);
      const int64_t a = pos + b == verify_end ? b : std::min(b, accept_pattern[block++ % accept_pattern.size()]);
      for (int64_t r = 0; r < a; ++r) keep(pos + r, l.data() + r * V);
      accept_prev = int(a);
      pos += a;
    }
    // Decode, one token a step (the first one accepts the last verify block's tokens).
    for (int64_t s = verify_end; s < n; ++s) {
      engine::Chunk c{0, s, {tokens[size_t(s)]}, 1};
      c.accept = accept_prev;
      accept_prev = -1;
      const std::vector<float> l = run(c, s);
      keep(s, l.data());
    }
    std::printf("%lld steps over %lld tokens: %.2f s (first step %.2f s)\n", (long long)steps, (long long)n, since(t0), first_step);
    if (capture_check) std::printf("capture of layer %d against its l_out dump: max abs difference %.3g\n", eng.config().capture_layers[0], max_cap_diff);
    if (eng.experts_on_card() < eng.model().n_expert) {
      std::printf("experts on the card: %lld of %lld per layer; %.2f%% of %llu routed pairs chose cold experts\n", (long long)eng.experts_on_card(),
                  (long long)eng.model().n_expert, eng.pairs() ? 100.0 * double(eng.cold_pairs()) / double(eng.pairs()) : 0.0,
                  (unsigned long long)eng.pairs());
      const auto& cs = eng.cold_stats();
      std::printf("cold experts: %.2f s waiting on the card's routing, %.2f s computing on the CPU over %llu layers (%llu with the card finished first)\n",
                  cs.wait_s, cs.cpu_s, (unsigned long long)cs.layers, (unsigned long long)cs.late_layers);
      const auto& ss = eng.stream_stats();
      std::printf("streamed: %d slots, %.2f GiB of cold experts in pinned memory; %llu experts in %llu layers carried %llu of %llu cold pairs (%.1f%%; the CPU took the rest)\n",
                  eng.stream_slots(), double(eng.pinned_bytes()) / 1073741824.0, (unsigned long long)ss.experts, (unsigned long long)ss.layers, (unsigned long long)ss.pairs,
                  (unsigned long long)eng.cold_pairs(), eng.cold_pairs() ? 100.0 * double(ss.pairs) / double(eng.cold_pairs()) : 0.0);
    }
    if (!record_routing.empty()) {
      json::Writer w;
      w.begin_object().kv("model", model).kv("tokens", n);
      w.key("counts").begin_array();
      for (const auto& row : eng.routing_counts()) {
        w.begin_array();
        for (int64_t c : row) w.value(c);
        w.end_array();
      }
      w.end_array().end_object();
      std::ofstream(record_routing) << w.str() << "\n";
    }

    std::filesystem::create_directories(out);
    json::Writer index;
    index.begin_object().kv("n_tokens", n).kv("n_vocab", V);
    index.key("tensors").begin_array();
    for (const auto& name : order) {
      const auto& [data, ne] = dumps[name];
      std::ofstream(out + "/" + name + ".f32", std::ios::binary)
          .write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
      index.begin_object().kv("name", name).kv("type", "f32").key("ne").begin_array();
      for (size_t k = 0; k < 4; ++k) index.value(k < ne.size() ? ne[k] : int64_t(1));
      index.end_array().kv("file", name + ".f32").end_object();
    }
    std::ofstream(out + "/logits.f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
    index.end_array().kv("logits_rows", int64_t(logits.size() / size_t(V))).end_object();
    std::ofstream(out + "/index.json") << index.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-engine-run: %s\n", e.what());
    return 1;
  }
  return 0;
}
