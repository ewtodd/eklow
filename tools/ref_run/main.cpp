// eklow-ref-run: eklow's reference forward pass (Gemma 4 E4B or Inkling 2 Mini, fp32) over one prompt, written in
// eklow-oracle's dump format so tools/ref/compare_dumps.py can set the two side by side.
//
//   eklow-ref-run --model GGUF --out DIR (--tokens-json FILE | --tokens 2,105,...)
//                 [--tensors PREFIX,...] [--logits-last N] [--dump-from N] [--teacher DIR] [--emulate-ggml-cpu]
//
// --tokens-json reads the ids from an oracle's tokens.json, so both runs see the same tokens.
// --tensors: graph names by prefix, a trailing $ for an exact name (default l_out-,result_norm); --logits-last: only the last N
// positions' logits (default all); --dump-from: tensors of positions N.. only (Inkling; an oracle's first_position).
// --teacher DIR (Inkling): after each layer application the stream continues from DIR's l_out-<slot> (an oracle dump of
// every position), so each layer is checked on the reference's own inputs. --emulate-ggml-cpu (Inkling) rounds every
// matmul's activations as ggml's CPU backend does, for a tight check against llama.cpp's CPU oracle.
// A host build (tools/ref/inkling/host-build.sh, EKLOW_HOST_NO_FP16) runs on SYCL's CPU device instead of the card.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "model/gemma4.hpp"
#include "model/inkling2.hpp"
#include "model/gguf.hpp"
#ifndef EKLOW_HOST_NO_FP16
#include "rt/device.hpp"
#endif

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
  std::string model, out, tokens_json, tokens_arg, teacher_dir;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  int64_t logits_last = 0, dump_from = 0;
  bool emulate = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--tokens-json") tokens_json = next();
    else if (a == "--tokens") tokens_arg = next();
    else if (a == "--tensors") prefixes = split(next(), ',');
    else if (a == "--logits-last") logits_last = std::atoll(next().c_str());
    else if (a == "--dump-from") dump_from = std::atoll(next().c_str());
    else if (a == "--teacher") teacher_dir = next();
    else if (a == "--emulate-ggml-cpu") emulate = true;
    else {
      std::fprintf(stderr, "usage: eklow-ref-run --model GGUF --out DIR (--tokens-json FILE | --tokens IDS) "
                           "[--tensors PREFIX,...] [--logits-last N] [--dump-from N]\n");
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
#ifdef EKLOW_HOST_NO_FP16
    sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
#endif
    const std::string arch = f.at("general.architecture").as_string();
    if (arch != "gemma4" && arch != "inkling2_loop") throw std::runtime_error(model + ": not a gemma4 or inkling2_loop GGUF");
    std::optional<model::Gemma4Config> gcfg;
    std::optional<model::Gemma4Weights> gw;
    std::optional<model::Inkling2Config> icfg;
    std::optional<model::Inkling2Weights> iw;
    int n_vocab = 0;
    uint64_t bytes = 0;
    if (arch == "gemma4") {
      gcfg = model::Gemma4Config::from_gguf(f);
      gw.emplace(model::Gemma4Weights::upload(f, *gcfg, q));
      n_vocab = gcfg->n_vocab;
      bytes = gw->bytes;
    } else {
      icfg = model::Inkling2Config::from_gguf(f);
      iw.emplace(model::Inkling2Weights::upload(f, *icfg, q));
      n_vocab = icfg->n_vocab_out;
      bytes = iw->bytes;
    }
    std::printf("loaded %.2f GiB onto %s in %.1f s\n", bytes / 1073741824.0,
                q.get_device().get_info<sycl::info::device::name>().c_str(), since(t0));

    json::Writer index;
    index.begin_object().kv("n_tokens", int64_t(tokens.size())).kv("n_vocab", int64_t(n_vocab));
    index.key("tensors").begin_array();
    model::DumpSink sink;
    sink.wants = [&](const std::string& name) {
      for (const auto& p : prefixes) {
        if (!p.empty() && p.back() == '$') {  // exact name, as in eklow-oracle
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

    model::Teacher teacher = [&](const std::string& name, std::vector<float>& data) {
      std::ifstream in(teacher_dir + "/" + name + ".f32", std::ios::binary | std::ios::ate);
      if (!in) return false;
      const std::streamsize n = in.tellg();
      data.resize(size_t(n) / sizeof(float));
      in.seekg(0);
      return bool(in.read(reinterpret_cast<char*>(data.data()), n));
    };
    t0 = std::chrono::steady_clock::now();
    auto logits = gcfg ? model::reference_forward(q, *gcfg, *gw, tokens, logits_last, &sink)
                       : model::inkling2_reference_forward(q, *icfg, *iw, tokens, logits_last, &sink, dump_from,
                                                           teacher_dir.empty() ? nullptr : &teacher, emulate);
    std::printf("reference forward over %zu tokens: %.1f s\n", tokens.size(), since(t0));
    std::ofstream(out + "/logits.f32", std::ios::binary)
        .write(reinterpret_cast<const char*>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
    index.end_array().kv("logits_rows", int64_t(logits.size() / size_t(n_vocab))).kv("first_position", dump_from).end_object();
    std::ofstream(out + "/index.json") << index.str() << "\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-ref-run: %s\n", e.what());
    return 1;
  }
  return 0;
}
