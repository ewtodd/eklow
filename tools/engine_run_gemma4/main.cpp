// eklow-engine-run: the serving engine over one prompt, written in eklow-oracle's dump format so
// tools/ref/compare_dumps.py can set it beside the oracle or the reference forward.
//
//   eklow-engine-run --model GGUF --out DIR (--tokens-json FILE | --tokens 2,105,...)
//                    [--tensors PREFIX,...] [--logits-last N] [--chunk N] [--decode-last N]
//                    [--twin] [--max-ctx N] [--kv-int8] [--image-rows FILE ...]
//
// --image-rows injects an image's rows in place of its tokens: tokens.json's "images" (the oracle's, in image mode:
// each image's first position and row count; its tokens are -1) in order, FILE the rows as fp32 [n][n_embd] (the
// oracle's image_K.f32, or eklow-vision --out). A chunk never splits an image: a step's end moves to the image's end.
//
// The prompt is fed in prefill chunks of --chunk tokens (default: one chunk), then its last
// --decode-last tokens one per step, as decode would. --twin runs the same prompt in a second slot
// in the same steps, offset by one chunk, writes its logits to DIR/twin, and fails when their mean
// KL from the first slot's exceeds --twin-tol (default 0.05). The two slots see different batch
// shapes, so they differ by the int8 noise floor; a slot reading another's cache is far worse.
//
// The speculative checks (--verify-last N --verify-block B [--accept A,...] [--capture-layers L,...]): the prompt's
// first n - N tokens are prefilled, and the last N are run twice in one slot, once decoded a token at a time (the plain run:
// logits and captured hidden states of each position) and once as verify blocks of B tokens whose acceptance follows the pattern
// --accept (a count per block, cycled, at least 1; the unaccepted tokens are fed again in the next block after the engine's
// rewind with truncate; the last block accepts all). It reports each verify row's logits against the plain run's (top-1
// agreement, mean and worst KL) and the captured rows of the accepted tokens against the plain run's (relative RMS error), and
// fails above --verify-tol (mean KL, default 0.05) or --capture-tol (relative RMS, default 0.05). The prefill's captures are
// compared too (the same chunks in both runs: identical). --capture-layers defaults to 1,9,17,25,33 (Bolt's DFlash draft's).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include "engine/engine_gemma4.hpp"
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

}  // namespace

int main(int argc, char** argv) {
  std::string model, out, tokens_json, tokens_arg;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  std::vector<std::string> image_files;
  int64_t logits_last = 1, chunk = 0, decode_last = 0, max_ctx = 0;
  bool twin = false, kv_int8 = false;
  double twin_tol = 0.05;
  int64_t verify_last = 0, verify_block = 4;
  std::vector<int64_t> accept_pattern = {1};
  std::vector<int> capture_layers = {1, 9, 17, 25, 33};
  double verify_tol = 0.05, capture_tol = 0.05;
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
    else if (a == "--image-rows") image_files.push_back(next());
    else if (a == "--verify-last") verify_last = std::atoll(next().c_str());
    else if (a == "--verify-block") verify_block = std::atoll(next().c_str());
    else if (a == "--accept") {
      accept_pattern.clear();
      for (auto s : split(next(), ',')) accept_pattern.push_back(std::atoll(s.c_str()));
    } else if (a == "--capture-layers") {
      capture_layers.clear();
      for (auto s : split(next(), ',')) capture_layers.push_back(std::atoi(s.c_str()));
    } else if (a == "--verify-tol") verify_tol = std::atof(next().c_str());
    else if (a == "--capture-tol") capture_tol = std::atof(next().c_str());
    else if (a == "--twin") twin = true;
    else if (a == "--twin-tol") twin_tol = std::atof(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-engine-run --model GGUF --out DIR (--tokens-json FILE | --tokens IDS) "
                           "[--tensors PREFIX,...] [--logits-last N] [--chunk N] [--decode-last N] [--twin] "
                           "[--twin-tol KL] [--max-ctx N] [--kv-int8] [--image-rows FILE ...] "
                           "[--verify-last N --verify-block B --accept A,... [--capture-layers L,...] [--verify-tol KL] [--capture-tol REL]]\n");
      return 2;
    }
  }
  try {
    std::vector<int32_t> tokens;
    std::vector<std::pair<int64_t, int64_t>> image_spans;  // first position and rows of each image
    if (!tokens_json.empty()) {
      std::ifstream in(tokens_json);
      std::string s((std::istreambuf_iterator<char>(in)), {});
      const json::Value doc = json::parse(s);
      const json::Value* ids = doc.find("ids");
      if (!ids || !ids->is_array()) throw std::runtime_error(tokens_json + ": no \"ids\" array");
      for (const auto& v : ids->items()) tokens.push_back(int32_t(std::atoll(v.int_digits().c_str())));
      if (const json::Value* imgs = doc.find("images"); imgs && imgs->is_array())
        for (const auto& im : imgs->items())
          image_spans.push_back({std::atoll(im.find("pos")->int_digits().c_str()), std::atoll(im.find("n")->int_digits().c_str())});
    } else {
      for (const auto& t : split(tokens_arg, ',')) tokens.push_back(int32_t(std::atoi(t.c_str())));
    }
    if (tokens.empty() || out.empty() || model.empty()) throw std::runtime_error("need --model, --out and tokens");
    const int64_t n = int64_t(tokens.size());
    logits_last = std::min(logits_last, n);
    decode_last = std::min(decode_last, n - 1);
    if (verify_last) {
      verify_last = std::min(verify_last, n - 1);
      decode_last = verify_last;  // the plain run decodes them
      for (int64_t a : accept_pattern)
        if (a < 1) throw std::runtime_error("--accept counts must be at least 1");
      if (verify_block < 2 || verify_block > engine::gemma4::kMaxVerify) throw std::runtime_error("--verify-block must be 2 to 16");
    }
    const int64_t prefill = n - decode_last;
    if (chunk <= 0) chunk = prefill;

    if (image_files.size() != image_spans.size()) throw std::runtime_error("need one --image-rows file for each of tokens.json's images");
    // The steps, as [start, end) token ranges; an image is never cut (the step ends after it).
    std::vector<std::pair<int64_t, int64_t>> steps;
    for (int64_t s = 0; s < prefill;) {
      int64_t e = std::min(prefill, s + chunk);
      for (auto [p, c] : image_spans)
        if (p < e && e < p + c) e = std::min(prefill, p + c);
      steps.push_back({s, e});
      s = e;
    }
    for (int64_t s = prefill; s < n; ++s) steps.push_back({s, s + 1});

    engine::gemma4::Config ec;
    ec.n_slots = twin ? 2 : 1;
    ec.max_ctx = max_ctx > 0 ? max_ctx : (n + 255) / 256 * 256;
    int64_t widest = 0;
    for (auto [a, b] : steps) widest = std::max(widest, b - a);
    ec.max_batch = twin ? 2 * widest : widest;
    ec.max_chunk = widest;
    ec.int8_global_kv = kv_int8;
    ec.max_logits = twin ? 2 * std::min(widest, logits_last) : std::min(widest, logits_last);
    if (verify_last) {
      if (twin) throw std::runtime_error("--verify-last and --twin do not combine");
      ec.max_batch = ec.max_chunk = std::max(widest, verify_block);
      ec.capture_layers = capture_layers;
      ec.max_capture_rows = ec.max_chunk;
      ec.max_logits = std::max<int64_t>(ec.max_logits, verify_block);
    }

    auto t0 = std::chrono::steady_clock::now();
    auto f = gguf::File::open(model);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    engine::gemma4::Engine eng(q, f, ec);
    const int64_t V = eng.model().n_vocab;
    std::vector<std::vector<float>> image_data;
    for (size_t k = 0; k < image_files.size(); ++k) {
      std::ifstream in(image_files[k], std::ios::binary);
      std::vector<float> rows(size_t(image_spans[k].second * eng.model().n_embd));
      in.read(reinterpret_cast<char*>(rows.data()), std::streamsize(rows.size() * sizeof(float)));
      if (!in) throw std::runtime_error(image_files[k] + ": fewer than " + std::to_string(image_spans[k].second) + " rows of " + std::to_string(eng.model().n_embd));
      image_data.push_back(std::move(rows));
    }
    // A chunk of tokens [a, b): the ids (image positions are 0), and the images inside it.
    auto make_chunk = [&](int slot, int64_t a, int64_t b, int64_t rows) {
      engine::gemma4::Chunk c{slot, a, std::vector<int32_t>(tokens.begin() + a, tokens.begin() + b), rows};
      for (int32_t& t : c.tokens) t = std::max<int32_t>(t, 0);
      for (size_t k = 0; k < image_spans.size(); ++k)
        if (image_spans[k].first >= a && image_spans[k].first + image_spans[k].second <= b)
          c.images.push_back({image_spans[k].first - a, image_spans[k].second, image_data[k].data()});
      return c;
    };
    std::printf("loaded %.2f GiB of weights, %.2f GiB of KV cache onto %s in %.1f s\n", eng.weight_bytes() / 1073741824.0,
                eng.cache_bytes() / 1073741824.0, q.get_device().get_info<sycl::info::device::name>().c_str(), since(t0));

    if (verify_last) {
      // The speculative checks (see the header): the plain run, then the same tokens as verify blocks with rewinds.
      const int64_t E = eng.model().n_embd, W = eng.capture().width, nl = int64_t(capture_layers.size());
      const int64_t prefill_n = n - verify_last, keep_from = std::max<int64_t>(0, n - 1024);
      std::vector<float> ref_cap(size_t((n - keep_from) * W)), ref_logits(size_t(verify_last * V)), host;
      auto fetch = [&]() -> const engine::gemma4::Engine::Capture& {
        const auto& cap = eng.capture();
        host.resize(size_t(cap.rows * cap.width));
        if (cap.rows) q.memcpy(host.data(), cap.data, host.size() * sizeof(float)).wait();
        return cap;
      };
      // The prefill's chunks (those of `steps` that end within the prompt), capturing the rows from keep_from on.
      auto prefill_run = [&](const std::function<void(int64_t pos, const float* row)>& on_row) {
        eng.release(0);
        for (auto [a, b] : steps) {
          if (b > prefill_n) break;
          engine::gemma4::Chunk c = make_chunk(0, a, b, 0);
          c.capture_rows = std::max<int64_t>(0, std::min(b - a, b - keep_from));
          eng.step({c});
          const auto& cap = fetch();
          for (int64_t r = 0; r < cap.rows; ++r) on_row(cap.ids[size_t(r)].pos, host.data() + r * cap.width);
        }
      };
      // Relative RMS error of captured rows against the plain run's, per captured layer.
      std::vector<double> num(size_t(nl), 0.0), den(size_t(nl), 0.0), pre_num(size_t(nl), 0.0), pre_den(size_t(nl), 0.0);
      auto compare_row = [&](int64_t pos, const float* row, std::vector<double>& nu, std::vector<double>& de) {
        if (pos < keep_from) return;
        const float* ref = ref_cap.data() + (pos - keep_from) * W;
        for (int64_t l = 0; l < nl; ++l)
          for (int64_t j = 0; j < E; ++j) {
            const double a = row[l * E + j], b = ref[l * E + j];
            nu[size_t(l)] += (a - b) * (a - b);
            de[size_t(l)] += b * b;
          }
      };
      // The plain run: the prefill, then the last tokens a step each.
      auto t_plain = std::chrono::steady_clock::now();
      prefill_run([&](int64_t pos, const float* row) { std::copy(row, row + W, ref_cap.begin() + (pos - keep_from) * W); });
      for (int64_t p = prefill_n; p < n; ++p) {
        engine::gemma4::Chunk c = make_chunk(0, p, p + 1, 1);
        c.capture_rows = 1;
        const std::vector<float> l = eng.step({c});
        std::copy(l.begin(), l.end(), ref_logits.begin() + (p - prefill_n) * V);
        const auto& cap = fetch();
        if (p >= keep_from) std::copy(host.begin(), host.begin() + cap.width, ref_cap.begin() + (p - keep_from) * W);
      }
      const double plain_s = since(t_plain);
      // The verify run.
      auto t_spec = std::chrono::steady_clock::now();
      prefill_run([&](int64_t pos, const float* row) { compare_row(pos, row, pre_num, pre_den); });
      double kl_sum = 0, kl_worst = 0, maxdiff = 0;
      int64_t rows = 0, same = 0, blocks = 0, rewinds = 0, fed = 0;
      size_t bi = 0;
      for (int64_t pos = prefill_n; pos < n;) {
        const int64_t b = std::min(verify_block, n - pos);
        engine::gemma4::Chunk c = make_chunk(0, pos, pos + b, b);
        c.verify = b >= 2;
        c.capture_rows = b;
        const std::vector<float> l = eng.step({c});
        const int64_t a = pos + b == n ? b : std::min(b, accept_pattern[bi++ % accept_pattern.size()]);
        const auto& cap = fetch();
        for (int64_t r = 0; r < b; ++r) {
          const float *x = l.data() + r * V, *y = ref_logits.data() + (pos + r - prefill_n) * V;
          const double d = kl(y, x, V);
          kl_sum += d, kl_worst = std::max(kl_worst, d), ++rows;
          same += std::max_element(x, x + V) - x == std::max_element(y, y + V) - y;
          for (int64_t v = 0; v < V; ++v) maxdiff = std::max(maxdiff, double(std::fabs(x[v] - y[v])));
        }
        for (int64_t r = 0; r < a; ++r) compare_row(cap.ids[size_t(r)].pos, host.data() + r * cap.width, num, den);
        ++blocks, fed += b;
        if (a < b) {
          eng.truncate(0, pos + a);  // the rewind
          ++rewinds;
        }
        pos += a;
      }
      const double spec_s = since(t_spec);
      std::printf("verify: %lld blocks of up to %lld tokens (%lld rows fed, %lld rewinds) over the last %lld tokens; plain run %.2f s (with the prefill), verify run %.2f s\n", (long long)blocks,
                  (long long)verify_block, (long long)fed, (long long)rewinds, (long long)verify_last, plain_s, spec_s);
      const double mean_kl = kl_sum / double(std::max<int64_t>(rows, 1));
      std::printf("verify rows against the same tokens decoded one at a time: top-1 agreement %lld/%lld, mean KL %.3g, worst %.3g, max logit difference %.3g\n", (long long)same,
                  (long long)rows, mean_kl, kl_worst, maxdiff);
      double worst_rel = 0, pre_worst = 0;
      for (int64_t l = 0; l < nl; ++l) {
        const double rel = std::sqrt(num[size_t(l)] / std::max(den[size_t(l)], 1e-30)), prel = std::sqrt(pre_num[size_t(l)] / std::max(pre_den[size_t(l)], 1e-30));
        std::printf("capture of layer %d: accepted verify rows rel RMS %.3g against the plain run's, prefill rows %.3g\n", capture_layers[size_t(l)], rel, prel);
        worst_rel = std::max(worst_rel, rel), pre_worst = std::max(pre_worst, prel);
      }
      if (mean_kl > verify_tol) throw std::runtime_error("verify rows disagree with decoding one at a time (mean KL above --verify-tol)");
      if (worst_rel > capture_tol || pre_worst > capture_tol) throw std::runtime_error("captured rows differ from the plain run's (above --capture-tol)");
      std::printf("verify and capture checks passed\n");
      return 0;
    }

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

    auto rows_in = [&](int64_t a, int64_t b) { return std::max<int64_t>(0, b - std::max(a, n - logits_last)); };
    std::vector<float> logits, twin_logits;
    double first_step = 0;
    t0 = std::chrono::steady_clock::now();
    // With --twin, slot 1 runs step i - 1's range alongside slot 0's step i (and one more step at the end).
    for (size_t i = 0; i < steps.size() + (twin ? 1 : 0); ++i) {
      std::vector<engine::gemma4::Chunk> chunks;
      bool main_in = i < steps.size();
      if (main_in) {
        auto [a, b] = steps[i];
        chunks.push_back(make_chunk(0, a, b, rows_in(a, b)));
      }
      if (twin && i > 0) {
        auto [a, b] = steps[i - 1];
        chunks.push_back(make_chunk(1, a, b, rows_in(a, b)));
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
