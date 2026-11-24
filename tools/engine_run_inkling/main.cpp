// eklow-engine-run-inkling: Inkling 2 Mini's serving engine (engine/engine_inkling) over one prompt, in eklow-oracle's dump
// format, for tools/ref/inkling/compare.py against M0 (llama.cpp) or eklow-ref-run's reference.
//
//   eklow-engine-run-inkling --model GGUF --out DIR --tokens-json FILE [--tensors PREFIX,...] [--dump-last N]
//                            [--chunk N] [--max-ctx N] [--stage L,L,...] [--teacher DIR] [--gen-json FILE] [--profile]
//
// The prompt is prefilled in chunks of --chunk tokens (default 1024); the last chunk is its final --dump-last positions
// (default all of a prompt that fits one chunk), whose tensors and logits are written (as the oracle's --dump-last). With
// --gen-json (an oracle's gen.json) its tokens but the last are then decoded one a step and logits_gen.f32 gets the logits
// of every step, row 0 the prompt's last position's (as the oracle's --generate writes them). --teacher DIR continues each
// layer application from DIR's l_out-<slot> (all positions of the prompt, [n_tok][n_embd]) during the prefill. --stage
// keeps those layers' experts in host memory. --profile prints where the prefill's and the decode's time goes.
#include <chrono>
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
  std::string model, out, tokens_json, teacher_dir, gen_json;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  int64_t chunk = 1024, max_ctx = 0, dump_last = 0;
  std::vector<int> stage;
  bool profile = false;
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
    else if (a == "--stage") for (const auto& s : split(next(), ',')) stage.push_back(std::atoi(s.c_str()));
    else if (a == "--teacher") teacher_dir = next();
    else if (a == "--gen-json") gen_json = next();
    else if (a == "--profile") profile = true;
    else {
      std::fprintf(stderr, "usage: eklow-engine-run-inkling --model GGUF --out DIR --tokens-json FILE [--tensors P,...] [--dump-last N]\n"
                           "       [--chunk N] [--max-ctx N] [--stage L,...] [--teacher DIR] [--gen-json FILE] [--profile]\n");
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
    cfg.n_slots = 1;
    cfg.max_ctx = max_ctx;
    cfg.max_batch = cfg.max_chunk = chunk;
    cfg.max_logits = int(last);
    cfg.staged_expert_layers = stage;
    engine::inkling::Engine eng(q, f, cfg);
    std::printf("engine: %.2f GiB of weights on %s, %.2f GiB of experts in host memory, %.2f GiB of caches, %.1f s\n",
                eng.weight_bytes() / 1073741824.0, q.get_device().get_info<sycl::info::device::name>().c_str(),
                eng.host_expert_bytes() / 1073741824.0, eng.cache_bytes() / 1073741824.0, since(t0));

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

    if (!gen.empty()) {
      const int64_t VO = eng.model().n_vocab_out;
      std::ofstream gl(out + "/logits_gen.f32", std::ios::binary);
      gl.write(reinterpret_cast<const char*>(logits.data() + logits.size() - size_t(VO)), std::streamsize(VO * sizeof(float)));
      t0 = std::chrono::steady_clock::now();
      for (size_t i = 0; i + 1 < gen.size(); ++i) {
        engine::inkling::Chunk c{0, n + int64_t(i), {gen[i]}, 1};
        const std::vector<float> l = eng.step({c});
        gl.write(reinterpret_cast<const char*>(l.data()), std::streamsize(VO * sizeof(float)));
      }
      const double t_dec = since(t0);
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
