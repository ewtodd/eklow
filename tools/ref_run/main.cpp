// eklow-ref-run: eklow's reference forward pass over one prompt, written in eklow-oracle's dump
// format so tools/ref/compare_dumps.py can set the two side by side.
//
//   eklow-ref-run --model GGUF --out DIR (--tokens-json FILE | --tokens 2,105,...)
//                 [--tensors PREFIX,...] [--logits-last N]
//
// --tokens-json reads the ids from an oracle's tokens.json, so both runs see the same tokens.
// --tensors: graph names by prefix (default l_out-,result_norm); --logits-last: only the last N
// positions' logits (default all).
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gemma4.hpp"
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
  std::string model, out, tokens_json, tokens_arg;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  int64_t logits_last = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--tokens-json") tokens_json = next();
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--tensors") prefixes = split(next(), ',');
    else if (a == "--logits-last") logits_last = std::atoll(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-ref-run --model GGUF --out DIR (--tokens-json FILE | --tokens IDS) "
                           "[--tensors PREFIX,...] [--logits-last N]\n");
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

    auto t0 = std::chrono::steady_clock::now();
    auto f = gguf::File::open(model);
    auto cfg = model::Gemma4Config::from_gguf(f);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    auto w = model::Gemma4Weights::upload(f, cfg, q);
    std::printf("loaded %.2f GiB onto %s in %.1f s\n", w.bytes / 1073741824.0,
                q.get_device().get_info<sycl::info::device::name>().c_str(), since(t0));

    json::Writer index;
    index.begin_object().kv("n_tokens", int64_t(tokens.size())).kv("n_vocab", int64_t(cfg.n_vocab));
    index.key("tensors").begin_array();
    model::DumpSink sink;
    sink.wants = [&](const std::string& name) {
      for (const auto& p : prefixes)
        if (name.compare(0, p.size(), p) == 0) return true;
      return false;
    };
    sink.put = [&](const std::string& name, const std::vector<float>& data, const std::vector<int64_t>& ne) {
      std::ofstream(out + "/" + name + ".f32", std::ios::binary)
          .write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
      index.begin_object().kv("name", name).kv("type", "f32").key("ne").begin_array();
      for (size_t k = 0; k < 4; ++k) index.value(k < ne.size() ? ne[k] : int64_t(1));
      index.end_array().kv("file", name + ".f32").end_object();
    };

    t0 = std::chrono::steady_clock::now();
    auto logits = model::reference_forward(q, cfg, w, tokens, logits_last, &sink);
    std::printf("reference forward over %zu tokens: %.1f s\n", tokens.size(), since(t0));
    std::ofstream(out + "/logits.f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
    index.end_array().kv("logits_rows", int64_t(logits.size() / size_t(cfg.n_vocab))).end_object();
    std::ofstream(out + "/index.json") << index.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-ref-run: %s\n", e.what());
    return 1;
  }
  return 0;
}
