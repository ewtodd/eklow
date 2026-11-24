// eklow-engine-run-inkling: Inkling 2 Mini's serving engine (engine/engine_inkling) over one prompt, in eklow-oracle's dump
// format, for tools/ref/inkling/compare.py against M0 (llama.cpp) or eklow-ref-run's reference.
//
//   eklow-engine-run-inkling --model GGUF --out DIR --tokens-json FILE [--tensors PREFIX,...] [--dump-last N]
//                            [--chunk N] [--max-ctx N] [--slots N] [--experts PROFILE.json] [--reserve-gib G]
//                            [--teacher DIR] [--gen-json FILE] [--profile]
//
// The prompt is prefilled in chunks of --chunk tokens (default 1024); the last chunk is its final --dump-last positions
// (default all of a prompt that fits one chunk), whose tensors and logits are written (as the oracle's --dump-last). With
// --gen-json (an oracle's gen.json) its tokens but the last are then decoded one a step and logits_gen.f32 gets the logits
// of every step, row 0 the prompt's last position's (as the oracle's --generate writes them). --teacher DIR continues each
// layer application from DIR's l_out-<slot> (all positions of the prompt, [n_tok][n_embd]) during the prefill. --experts
// places the experts by that routing profile (the busiest on the card while free memory less 1 GiB and --reserve-gib
// holds them, the rest on the CPU); --slots and --max-ctx size the K/V as a server would (the prompt runs in slot 0).
// --profile prints where the prefill's and the decode's time goes. --rewind (with --gen-json) checkpoints the slot after the
// prefill, decodes, rewinds to the checkpoint and decodes the same tokens again: the logits must not change.
// --cpu-max-tokens N: steps up to N tokens run the cold experts on the CPU, longer ones stage them on the card.
// --twin (with --gen-json, two slots): the prompt and its decode run in slots 0 and 1 in the same steps, and the two
// slots' logits are compared (they differ only by the batch shape's rounding). --twin-tokens FILE puts another prompt
// (an oracle's tokens.json) in slot 1 instead, and slot 0's decode logits go to DIR/logits_gen_twin.f32, to be set beside
// a one-slot run's logits_gen.f32.
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine_inkling.hpp"
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

std::vector<int32_t> ids_of(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path);
  const json::Value doc = json::parse(std::string((std::istreambuf_iterator<char>(in)), {}));
  const json::Value* ids = doc.find("ids");
  if (!ids || !ids->is_array()) throw std::runtime_error(path + ": no \"ids\" array");
  std::vector<int32_t> out;
  for (const auto& v : ids->items()) out.push_back(int32_t(std::atoll(v.int_digits().c_str())));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, out, tokens_json, teacher_dir, gen_json, twin_tokens;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  int64_t chunk = 1024, max_ctx = 0, dump_last = 0, slots = 1;
  std::string experts;
  double reserve_gib = 0;
  int64_t cpu_max_tokens = -1;
  bool profile = false, rewind = false, twin = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--tokens-json") tokens_json = next();
    else if (a == "--tensors") prefixes = split(next(), ',');
    else if (a == "--dump-last") dump_last = std::atoll(next().c_str());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--max-ctx") max_ctx = std::atoll(next().c_str());
    else if (a == "--experts") experts = next();
    else if (a == "--reserve-gib") reserve_gib = std::atof(next().c_str());
    else if (a == "--slots") slots = std::atoll(next().c_str());
    else if (a == "--teacher") teacher_dir = next();
    else if (a == "--gen-json") gen_json = next();
    else if (a == "--profile") profile = true;
    else if (a == "--rewind") rewind = true;
    else if (a == "--twin") twin = true;
    else if (a == "--twin-tokens") twin_tokens = next(), twin = true;
    else if (a == "--cpu-max-tokens") cpu_max_tokens = std::atoll(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-engine-run-inkling --model GGUF --out DIR --tokens-json FILE [--tensors P,...] [--dump-last N]\n"
                           "       [--chunk N] [--max-ctx N] [--slots N] [--experts PROFILE] [--reserve-gib G] [--teacher DIR]\n"
                           "       [--gen-json FILE] [--profile]\n");
      return 2;
    }
  }
  try {
    const std::vector<int32_t> tokens = ids_of(tokens_json);
    const std::vector<int32_t> gen = gen_json.empty() ? std::vector<int32_t>{} : ids_of(gen_json);
    const int64_t n = int64_t(tokens.size());
    const int64_t last = dump_last > 0 && dump_last < n ? dump_last : std::min(n, chunk);
    if (last > chunk) throw std::runtime_error("--dump-last above --chunk");
    if (max_ctx <= 0) max_ctx = (n + int64_t(gen.size()) + attn::kPage) / attn::kPage * attn::kPage;

    auto t0 = std::chrono::steady_clock::now();
    auto f = gguf::File::open(model);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::inkling::Config cfg;
    cfg.n_slots = int(slots);
    cfg.max_ctx = max_ctx;
    cfg.max_batch = cfg.max_chunk = chunk;
    cfg.max_logits = int(last);
    cfg.expert_profile = experts;
    cfg.card_reserve_bytes = uint64_t(reserve_gib * 1073741824.0);
    if (cpu_max_tokens >= 0) cfg.cpu_max_tokens = cpu_max_tokens;
    engine::inkling::Engine eng(q, f, cfg);
    std::printf("engine: %.2f GiB of weights on %s, %lld cold experts (%.2f GiB) on the CPU, %.2f GiB of caches, %.2f GiB free "
                "before the experts, %.1f s\n",
                eng.weight_bytes() / 1073741824.0, q.get_device().get_info<sycl::info::device::name>().c_str(),
                (long long)eng.cold_experts(), eng.host_expert_bytes() / 1073741824.0, eng.cache_bytes() / 1073741824.0,
                eng.free_at_placement() / 1073741824.0, since(t0));

    // The dumps: the last chunk's tensors only.
    std::filesystem::create_directories(out);
    json::Writer index;
    index.begin_object().kv("n_tokens", n).kv("n_vocab", int64_t(eng.model().n_vocab_out)).kv("first_position", n - last);
    index.key("tensors").begin_array();
    bool dumping = false;
    model::DumpSink sink;
    sink.wants = [&](const std::string& name) {
      if (!dumping) return false;
      for (const auto& p : prefixes) {
        if (!p.empty() && p.back() == '$') {
          if (name == p.substr(0, p.size() - 1)) return true;
        } else if (name.compare(0, p.size(), p) == 0) {
          return true;
        }
      }
      return false;
    };
    sink.put = [&](const std::string& name, const std::vector<float>& data, const std::vector<int64_t>& ne) {
      std::ofstream(out + "/" + name + ".f32", std::ios::binary)
          .write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
      index.begin_object().kv("name", name).kv("type", "f32").key("ne").begin_array();
      for (size_t k = 0; k < 4; ++k) index.value(k < ne.size() ? ne[k] : int64_t(1));
      index.end_array().kv("file", name + ".f32").end_object();
    };
    // The teacher's rows for the current step: positions [from, from + T).
    int64_t from = 0, rows = 0;
    model::Teacher teacher = [&](const std::string& name, std::vector<float>& data) {
      std::ifstream in(teacher_dir + "/" + name + ".f32", std::ios::binary);
      if (!in) return false;
      const int64_t E = eng.model().n_embd;
      data.resize(size_t(rows * E));
      in.seekg(std::streamoff(from * E * int64_t(sizeof(float))));
      return bool(in.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float))));
    };

    std::map<std::string, double> prof;
    if (profile) eng.set_profile(&prof);
    t0 = std::chrono::steady_clock::now();
    std::vector<float> logits;
    for (int64_t p = 0; p < n;) {
      const int64_t end = p < n - last ? std::min(p + chunk, n - last) : n;
      engine::inkling::Chunk c{0, p, std::vector<int32_t>(tokens.begin() + p, tokens.begin() + end), end == n ? last : 0};
      dumping = end == n;
      from = p, rows = end - p;
      logits = eng.step({c}, &sink, teacher_dir.empty() ? nullptr : &teacher);
      p = end;
    }
    dumping = false;
    const double t_prefill = since(t0);
    std::printf("prefill: %lld tokens in %.2f s (%.0f tok/s)\n", (long long)n, t_prefill, double(n) / t_prefill);
    std::ofstream(out + "/logits.f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
    index.end_array().kv("logits_rows", int64_t(logits.size() / size_t(eng.model().n_vocab_out))).end_object();
    std::ofstream(out + "/index.json") << index.str() << "\n";
    if (profile) {
      for (const auto& [k, v] : prof) std::printf("  prefill %-14s %8.3f s\n", k.c_str(), v);
      prof.clear();
    }

    if (twin) {
      if (slots < 2 || gen.empty()) throw std::runtime_error("--twin needs --slots 2 and --gen-json");
      // slot 0 again from the start, and slot 1 beside it in every step
      eng.release(0);
      const int64_t VO = eng.model().n_vocab_out;
      auto kl_top = [&](const float* a, const float* b, double& kl, bool& same) {
        double ma = -1e30, mb = -1e30;
        int64_t ia = 0, ib = 0;
        for (int64_t v = 0; v < VO; ++v) {
          if (a[v] > ma) ma = a[v], ia = v;
          if (b[v] > mb) mb = b[v], ib = v;
        }
        double za = 0, zb = 0;
        for (int64_t v = 0; v < VO; ++v) za += std::exp(a[v] - ma), zb += std::exp(b[v] - mb);
        kl = 0;
        for (int64_t v = 0; v < VO; ++v) {
          const double pa = std::exp(a[v] - ma) / za;
          if (pa > 0) kl += pa * ((a[v] - ma - std::log(za)) - (b[v] - mb - std::log(zb)));
        }
        same = ia == ib;
      };
      double worst = 0, sum = 0;
      int agree = 0, rows = 0;
      std::vector<float> l;
      // slot 1's prompt: the same, or another one (then it decodes its own prompt's first tokens again as filler)
      const std::vector<int32_t> other = twin_tokens.empty() ? tokens : ids_of(twin_tokens);
      const int64_t n1 = int64_t(other.size());
      std::ofstream tg;
      if (!twin_tokens.empty()) tg.open(out + "/logits_gen_twin.f32", std::ios::binary);
      int64_t p0 = 0, p1 = 0;
      while (p0 < n || p1 < n1) {  // both slots' chunks share a step of at most `chunk` tokens
        std::vector<engine::inkling::Chunk> cs;
        if (p0 < n) {
          const int64_t e = std::min(p0 + chunk / 2, n);
          cs.push_back({0, p0, std::vector<int32_t>(tokens.begin() + p0, tokens.begin() + e), 1});
          p0 = e;
        }
        if (p1 < n1) {
          const int64_t e = std::min(p1 + chunk / 2, n1);
          cs.push_back({1, p1, std::vector<int32_t>(other.begin() + p1, other.begin() + e), 1});
          p1 = e;
        }
        const std::vector<float> r = eng.step(cs);
        if (cs[0].slot == 0 && p0 == n && (cs.size() == 1 || true)) l = r;
      }
      for (size_t i = 0;; ++i) {
        if (twin_tokens.empty()) {
          double kl;
          bool same;
          kl_top(l.data(), l.data() + VO, kl, same);
          worst = std::max(worst, kl), sum += kl, agree += same, ++rows;
        } else {
          tg.write(reinterpret_cast<const char*>(l.data()), std::streamsize(VO * sizeof(float)));
        }
        if (i + 1 >= gen.size()) break;
        l = eng.step({{0, n + int64_t(i), {gen[i]}, 1}, {1, n1 + int64_t(i), {other[i % size_t(n1)]}, 1}});
      }
      if (twin_tokens.empty())
        std::printf("twin: slots 0 and 1 in the same steps: top-1 %d/%d, KL mean %.2e, max %.2e\n", agree, rows, sum / rows, worst);
      else
        std::printf("twin: slot 0's %zu decode logits written beside slot 1's other prompt\n", gen.size());
      return 0;
    }

    if (!gen.empty()) {
      const int64_t VO = eng.model().n_vocab_out;
      std::ofstream gl(out + "/logits_gen.f32", std::ios::binary);
      gl.write(reinterpret_cast<const char*>(logits.data() + logits.size() - size_t(VO)), std::streamsize(VO * sizeof(float)));
      if (rewind) eng.checkpoint(0);
      std::vector<std::vector<float>> first;
      t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i + 1 < gen.size(); ++i) {
        engine::inkling::Chunk c{0, n + int64_t(i), {gen[i]}, 1};
        const std::vector<float> l = eng.step({c});
        gl.write(reinterpret_cast<const char*>(l.data()), std::streamsize(VO * sizeof(float)));
        if (rewind) first.push_back(l);
      }
      const double t_dec = since(t0);
      if (rewind) {
        if (!eng.rewind(0, n)) throw std::runtime_error("rewind to the prompt's end failed");
        double worst = 0;
        for (size_t i = 0; i + 1 < gen.size(); ++i) {
          engine::inkling::Chunk c{0, n + int64_t(i), {gen[i]}, 1};
          const std::vector<float> l = eng.step({c});
          for (size_t v = 0; v < l.size(); ++v) worst = std::max(worst, double(std::fabs(l[v] - first[i][v])));
        }
        std::printf("rewind: %zu steps decoded again after a rewind to %lld, largest logit difference %g\n", gen.size() - 1,
                    (long long)n, worst);
      }
      std::printf("decode: %zu steps in %.2f s (%.1f tok/s)\n", gen.size() - 1, t_dec, double(gen.size() - 1) / t_dec);
      if (profile)
        for (const auto& [k, v] : prof) std::printf("  decode  %-14s %8.3f s\n", k.c_str(), v);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-engine-run-inkling: %s\n", e.what());
    return 1;
  }
  return 0;
}
