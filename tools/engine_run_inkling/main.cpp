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
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/json_value.hpp"
#include <level_zero/ze_api.h>
#include <level_zero/layers/zel_tracing_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <map>
#include "engine/engine_inkling.hpp"
#include "model/dflash2.hpp"
#include "serve/spec.hpp"
#include "model/vision_inkling2v.hpp"
#include "image/inkling_image.hpp"
#include "kernels/attention.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

// EKLOW_TRACE=1: a step's kernels on the card, by the Level Zero loader's tracing layer: each kernel (and copy) appended
// to a command list gets a global timestamp written after it, so that the time from one to the next is that kernel and
// the gap before it. Printed per kernel name for the 10th decode step (speed work).
struct Tracer {
  zel_tracer_handle_t h = nullptr;
  uint64_t* ts = nullptr;  // host memory the card writes
  static constexpr size_t kMax = 16384;
  std::vector<std::string> names;
  bool on = false;
  double ns_per_tick = 1;
  void stamp(ze_command_list_handle_t cl, std::string name) {
    if (!on || names.size() >= kMax) return;
    on = false;  // our own append below is not traced
    zeCommandListAppendWriteGlobalTimestamp(cl, ts + names.size(), nullptr, 0, nullptr);
    on = true;
    names.push_back(std::move(name));
  }
};
Tracer g_tr;

void tracer_start(sycl::queue& q) {
  ze_context_handle_t zc = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
  ze_device_handle_t zd = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
  ze_host_mem_alloc_desc_t hd{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC};
  void* p = nullptr;
  if (zeMemAllocHost(zc, &hd, Tracer::kMax * 8, 64, &p) != ZE_RESULT_SUCCESS) throw std::runtime_error("trace: host memory");
  g_tr.ts = static_cast<uint64_t*>(p);
  ze_device_properties_t dp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
  zeDeviceGetProperties(zd, &dp);
  g_tr.ns_per_tick = double(dp.timerResolution);  // with ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES: nanoseconds a tick
  zel_tracer_desc_t td{ZEL_STRUCTURE_TYPE_TRACER_EXP_DESC, nullptr, &g_tr};
  if (zelTracerCreate(&td, &g_tr.h) != ZE_RESULT_SUCCESS) throw std::runtime_error("trace: zelTracerCreate (ZE_ENABLE_TRACING_LAYER=1?)");
  zel_core_callbacks_t ep{};
  ep.CommandList.pfnAppendLaunchKernelCb = [](ze_command_list_append_launch_kernel_params_t* pr, ze_result_t, void* ud, void**) {
    auto* t = static_cast<Tracer*>(ud);
    if (!t->on) return;
    size_t n = 0;
    zeKernelGetName(*pr->phKernel, &n, nullptr);
    std::string name(n, '\0');
    zeKernelGetName(*pr->phKernel, &n, name.data());
    t->stamp(*pr->phCommandList, name.c_str());
  };
  ep.CommandList.pfnAppendMemoryCopyCb = [](ze_command_list_append_memory_copy_params_t* pr, ze_result_t, void* ud, void**) {
    static_cast<Tracer*>(ud)->stamp(*pr->phCommandList, "(copy)");
  };
  ep.CommandList.pfnAppendBarrierCb = [](ze_command_list_append_barrier_params_t* pr, ze_result_t, void* ud, void**) {
    static_cast<Tracer*>(ud)->stamp(*pr->phCommandList, "(barrier)");
  };
  zelTracerSetEpilogues(g_tr.h, &ep);
  zelTracerSetEnabled(g_tr.h, true);
}

void tracer_print() {
  // a kernel's time: from the previous stamp to its own (its run and the gap before it); the first has none
  struct Agg {
    int n = 0;
    double us = 0;
  };
  std::map<std::string, Agg> agg;
  double total = 0;
  for (size_t i = 1; i < g_tr.names.size(); ++i) {
    std::string nm = g_tr.names[i];
    // shorten the mangled names: drop the template arguments past the first 60 characters
    if (nm.size() > 70) nm = nm.substr(0, 70);
    const double us = double(g_tr.ts[i] - g_tr.ts[i - 1]) * g_tr.ns_per_tick * 1e-3;
    agg[nm].n++, agg[nm].us += us, total += us;
  }
  std::vector<std::pair<std::string, Agg>> v(agg.begin(), agg.end());
  std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
  std::printf("trace: %zu commands, %.1f us from the first to the last\n", g_tr.names.size(), total);
  for (const auto& [nm, a] : v) std::printf("  %8.1f us %5d x %6.1f us  %s\n", a.us, a.n, a.us / a.n, nm.c_str());
}

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
  if (std::getenv("EKLOW_TRACE")) setenv("ZE_ENABLE_TRACING_LAYER", "1", 1);
  // the server's runtime for Inkling (tools/server/main.cpp): Level Zero's v2 adapter unless SYCL_UR_USE_LEVEL_ZERO_V2 is set
  if (!std::getenv("SYCL_UR_USE_LEVEL_ZERO_V2")) setenv("SYCL_UR_USE_LEVEL_ZERO_V2", "1", 1);
  std::string model, out, tokens_json, teacher_dir, gen_json, twin_tokens;
  std::vector<std::string> prefixes = {"l_out-", "result_norm"};
  int64_t chunk = 1024, max_ctx = 0, dump_last = 0, slots = 1, step_tokens = 0, kv_pool = 0;
  std::string experts;
  double reserve_gib = 0;
  int64_t cpu_max_tokens = -1;
  int cpu_threads = 0;  // 0: the engine's default (8)
  double loopcd = 0;  // LoopCD's contrast strength (Config::loopcd)
  bool profile = false, rewind = false, twin = false, kv_int8 = false;
  // --mmproj FILE --image FILE [--image-tokens N] (Inkling M5): the prompt's image (an oracle's tokens.json lists its
  // positions) encoded by the vision tower before the engine loads, its rows put in at those positions
  std::string mmproj, image_path;
  int64_t image_tokens = 4096;
  int64_t generate = 0;  // --generate N: N tokens decoded greedily after the prompt, their ids into DIR/generated.json
  // --block-time K,...: a speculative verify's cost: after --generate's tokens, rewind to the prompt's end and feed them
  // again in steps of K tokens (one slot), for each K, timing the steps
  std::vector<int64_t> block_time;
  // --draft FILE [--spec-block B] (with --generate N): the DFlash 2 draft (M6). After the plain greedy tokens, rewind to the
  // prompt's end and decode them again speculatively (blocks of B: the anchor and B - 1 drafted tokens, greedy acceptance);
  // the two runs' tokens are compared and the tokens a verify step reported.
  std::string draft_path;
  int spec_block = 8;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--tokens-json") tokens_json = next();
    else if (a == "--mmproj") mmproj = next();
    else if (a == "--image") image_path = next();
    else if (a == "--image-tokens") image_tokens = std::atoll(next().c_str());
    else if (a == "--tensors") prefixes = split(next(), ',');
    else if (a == "--dump-last") dump_last = std::atoll(next().c_str());
    else if (a == "--chunk") chunk = std::atoll(next().c_str());
    else if (a == "--step") step_tokens = std::atoll(next().c_str());
    else if (a == "--kv-pool") kv_pool = std::atoll(next().c_str());
    else if (a == "--max-ctx") max_ctx = std::atoll(next().c_str());
    else if (a == "--experts") experts = next();
    else if (a == "--reserve-gib") reserve_gib = std::atof(next().c_str());
    else if (a == "--slots") slots = std::atoll(next().c_str());
    else if (a == "--teacher") teacher_dir = next();
    else if (a == "--gen-json") gen_json = next();
    else if (a == "--generate") generate = std::atoll(next().c_str());
    else if (a == "--draft") draft_path = next();
    else if (a == "--spec-block") spec_block = std::atoi(next().c_str());
    else if (a == "--block-time")
      for (const std::string& v : split(next(), ',')) block_time.push_back(std::atoll(v.c_str()));
    else if (a == "--profile") profile = true;
    else if (a == "--rewind") rewind = true;
    else if (a == "--twin") twin = true;
    else if (a == "--twin-tokens") twin_tokens = next(), twin = true;
    else if (a == "--cpu-max-tokens") cpu_max_tokens = std::atoll(next().c_str());
    else if (a == "--loopcd") loopcd = std::atof(next().c_str());
    else if (a == "--kv-int8") kv_int8 = true;
    else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-engine-run-inkling --model GGUF --out DIR --tokens-json FILE [--tensors P,...] [--dump-last N]\n"
                           "       [--chunk N] [--step N] [--max-ctx N] [--slots N] [--experts PROFILE] [--reserve-gib G] [--teacher DIR]\n"
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
    // EKLOW_PROFILE_DEVICE (with --profile): the queue keeps kernel timestamps, and the profile is the device's time between
    // marks instead of the host's.
    sycl::queue q = std::getenv("EKLOW_PROFILE_DEVICE")
                        ? sycl::queue(rt::pick_device(), {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()})
                        : sycl::queue(rt::pick_device(), sycl::property::queue::in_order());
    if (std::getenv("EKLOW_TRACE")) tracer_start(q);
    engine::inkling::Config cfg;
    cfg.n_slots = int(slots);
    cfg.max_ctx = max_ctx;
    cfg.max_batch = cfg.max_chunk = chunk;
    // --step: the prompt's chunks before the dumped last one go in steps of up to this many tokens, layer by layer
    // (Config::max_step)
    if (step_tokens < chunk) step_tokens = chunk;
    cfg.max_step = step_tokens;
    cfg.pool_positions = kv_pool;  // --kv-pool: one K/V pool of this many positions for the slots (0: max-ctx each)
    cfg.max_logits = int(last);
    cfg.expert_profile = experts;
    cfg.card_reserve_bytes = uint64_t(reserve_gib * 1073741824.0);
    if (cpu_max_tokens >= 0) cfg.cpu_max_tokens = cpu_max_tokens;
    cfg.loopcd = float(loopcd);
    cfg.kv_int8 = kv_int8;
    if (cpu_threads > 0) cfg.cpu_threads = cpu_threads;
    std::vector<float> image_rows;
    int64_t image_pos = -1, image_n = 0;
    if (!mmproj.empty()) {
      std::ifstream tin(tokens_json);
      const json::Value doc = json::parse(std::string((std::istreambuf_iterator<char>(tin)), {}));
      const json::Value* ims = doc.find("images");
      if (!ims || !ims->is_array() || ims->items().size() != 1) throw std::runtime_error(tokens_json + ": one image expected");
      image_pos = std::atoll(ims->items()[0].find("pos")->int_digits().c_str());
      image_n = std::atoll(ims->items()[0].find("n")->int_digits().c_str());
      std::ifstream iin(image_path, std::ios::binary);
      const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(iin)), {});
      const gguf::File mf = gguf::File::open(mmproj);
      model::VisionInkling2V vt(q, mf);
      image_rows = vt.encode(image::prepare_inkling(image::decode(bytes), image_tokens));
      if (int64_t(image_rows.size()) != image_n * vt.n_out()) throw std::runtime_error("the image's tokens differ from the oracle's");
      std::printf("image: %lld tokens at %lld\n", (long long)image_n, (long long)image_pos);
    }
    auto attach = [&](engine::inkling::Chunk& c) {
      const int64_t end = c.start_pos + int64_t(c.tokens.size());
      if (image_pos < 0 || image_pos + image_n <= c.start_pos || image_pos >= end) return;
      if (image_pos < c.start_pos || image_pos + image_n > end) throw std::runtime_error("an image across chunks");
      c.images.push_back({image_pos - c.start_pos, image_n, image_rows.data()});
    };
    std::unique_ptr<gguf::File> dfile;
    if (!draft_path.empty()) {
      dfile = std::make_unique<gguf::File>(gguf::File::open(draft_path));
      cfg.capture_ids = model::Dflash2Config::from_gguf(*dfile).target_ids;
    }
    engine::inkling::Engine eng(q, f, cfg);
    std::unique_ptr<model::Dflash2> dr;
    if (dfile) {
      model::Dflash2::Options o;
      o.slots = 2, o.max_block = 16;
      dr = std::make_unique<model::Dflash2>(q, *dfile, model::Dflash2Target{eng.embedding_table(), eng.head_weight(), true}, o);
      std::printf("draft: %s, %.0f MB on the card, block %d, reach %lld\n", draft_path.c_str(), double(dr->bytes()) / 1e6, spec_block,
                  (long long)dr->reach());
    }
    // the draft's context: the chunk's rows within its reach of the prompt's end, injected after each step
    auto capture = [&](engine::inkling::Chunk& c, int64_t prompt_end) {
      if (!dr) return;
      const int64_t end = c.start_pos + int64_t(c.tokens.size()), from = std::max(c.start_pos, prompt_end - dr->reach());
      c.capture_rows = std::max<int64_t>(0, end - from);
    };
    auto inject = [&]() {
      if (!dr || eng.capture().rows == 0) return;
      const auto& cp = eng.capture();
      std::vector<int32_t> sl;
      std::vector<int64_t> ps;
      for (const auto& id : cp.ids) sl.push_back(id.slot), ps.push_back(id.pos);
      dr->inject_rows(cp.data, sl.data(), ps.data(), cp.rows);
    };
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
      std::vector<engine::inkling::Chunk> cs;
      int64_t q = p;
      while (teacher_dir.empty() && q < n - last && q + std::min(chunk, n - last - q) - p <= step_tokens) {
        const int64_t e = std::min(q + chunk, n - last);
        cs.push_back({0, q, std::vector<int32_t>(tokens.begin() + q, tokens.begin() + e), 0});
        attach(cs.back());
        capture(cs.back(), n);
        q = e;
      }
      if (cs.size() > 1) {
        dumping = false;
        logits = eng.step(cs);
        inject();
        p = q;
        continue;
      }
      const int64_t end = p < n - last ? std::min(p + chunk, n - last) : n;
      engine::inkling::Chunk c{0, p, std::vector<int32_t>(tokens.begin() + p, tokens.begin() + end), end == n ? last : 0};
      attach(c);
      capture(c, n);
      dumping = end == n;
      from = p, rows = end - p;
      logits = eng.step({c}, &sink, teacher_dir.empty() ? nullptr : &teacher);
      inject();
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
        // the same prompt: slot 1 decodes the same tokens; another: its prompt's first tokens again as filler
        l = eng.step({{0, n + int64_t(i), {gen[i]}, 1}, {1, n1 + int64_t(i), {twin_tokens.empty() ? gen[i] : other[i % size_t(n1)]}, 1}});
      }
      if (twin_tokens.empty())
        std::printf("twin: slots 0 and 1 in the same steps: top-1 %d/%d, KL mean %.2e, max %.2e\n", agree, rows, sum / rows, worst);
      else
        std::printf("twin: slot 0's %zu decode logits written beside slot 1's other prompt\n", gen.size());
      return 0;
    }

    if (generate > 0) {
      const int64_t VO = eng.model().n_vocab_out;
      std::vector<int32_t> ids;
      std::vector<float> l(logits.end() - VO, logits.end());
      if (!block_time.empty() || dr) eng.checkpoint(0);
      std::vector<std::vector<float>> plain_rows;  // with --draft: plain decode's logits a position, for the comparison
      // EKLOW_PLAIN_CAPTURE (with --draft): plain decode captures each token's features and injects them, as the server
      // does for a request that speculates, to measure that cost apart
      static const bool plain_capture = dr && std::getenv("EKLOW_PLAIN_CAPTURE");
      const auto tg = std::chrono::steady_clock::now();
      for (int64_t i = 0; i < generate; ++i) {
        const int32_t t = int32_t(std::max_element(l.begin(), l.end()) - l.begin());
        ids.push_back(t);
        if (dr) plain_rows.push_back(l);
        engine::inkling::Chunk c{0, n + i, {t}, 1};
        if (plain_capture) c.capture_rows = 1;
        l = eng.step({c});
        if (plain_capture) {
          const auto& cap = eng.capture();
          dr->inject(0, cap.data, n + i, 1);
        }
      }
      const double dt = since(tg);
      std::printf("decode: %lld greedy steps in %.2f s (%.2f tok/s)\n", (long long)generate, dt, double(generate) / dt);
      if (dr) {
        // the speculative run from the prompt's end: greedy drafts, accepted while the target's argmax agrees
        if (!eng.rewind(0, n)) throw std::runtime_error("--draft: no checkpoint at the prompt's end");
        dr->truncate(0, n);
        std::mt19937_64 rng(1);
        auto argmax = [&](const float* r) { return int32_t(std::max_element(r, r + VO) - r); };
        std::vector<int32_t> sids;
        int32_t anchor = argmax(logits.data() + logits.size() - size_t(VO));
        int64_t pos = n, steps = 0;
        const int B = spec_block;
        // each kept row's logits against plain decode's at its position while the two sequences agree
        double kl_sum = 0, kl_max = 0;
        int64_t kl_n = 0, top_same = 0;
        int64_t first_far = -1;
        auto compare = [&](const float* a, const std::vector<float>& b, int64_t at) {
          double ma = -1e30, mb = -1e30;
          for (int64_t v = 0; v < VO; ++v) ma = std::max(ma, double(a[v])), mb = std::max(mb, double(b[size_t(v)]));
          double za = 0, zb = 0;
          for (int64_t v = 0; v < VO; ++v) za += std::exp(double(a[v]) - ma), zb += std::exp(double(b[size_t(v)]) - mb);
          double kl = 0;
          for (int64_t v = 0; v < VO; ++v) {
            const double pb = std::exp(double(b[size_t(v)]) - mb) / zb;
            if (pb > 0) kl += pb * ((double(b[size_t(v)]) - mb - std::log(zb)) - (double(a[v]) - ma - std::log(za)));
          }
          kl_sum += kl, kl_max = std::max(kl_max, kl), ++kl_n;
          top_same += int64_t(std::max_element(a, a + VO) - a) == int64_t(std::max_element(b.begin(), b.begin() + VO) - b.begin());
          if (kl > 1e-2 && first_far < 0) first_far = at;
        };
        const auto ts = std::chrono::steady_clock::now();
        double t_draft = 0, t_step = 0, t_trunc = 0, t_inject = 0;
        while (int64_t(sids.size()) < generate) {
          auto tp = std::chrono::steady_clock::now();
          serve::DraftBlock blk = dr->draft(0, anchor, pos, B, 0.0f, rng);
          t_draft += since(tp), tp = std::chrono::steady_clock::now();
          // EKLOW_SPEC_ORACLE=1 (checks): the draft is plain decode's own next tokens (all accepted: only the batch shape
          // differs); =2: the same, cut at a random point (a rollback every step, on tokens known to be right)
          static const int oracle = std::getenv("EKLOW_SPEC_ORACLE") ? std::atoi(std::getenv("EKLOW_SPEC_ORACLE")) : 0;
          int cut = B - 1;
          if (oracle) {
            for (int j = 0; j < B - 1; ++j) {
              const int64_t at = pos - n + 1 + j;
              blk.tokens[size_t(j)] = at < int64_t(ids.size()) ? ids[size_t(at)] : 0;
            }
            if (oracle >= 2) cut = int(rng() % uint64_t(B));
            // =3: past the cut, other tokens (rejected, as wrong drafts are: what they wrote must not stay)
            if (oracle == 3)
              for (int j = cut; j < B - 1; ++j) blk.tokens[size_t(j)] = int32_t(1000 + rng() % 50000);
          }
          engine::inkling::Chunk c{0, pos, {anchor}, B};
          c.tokens.insert(c.tokens.end(), blk.tokens.begin(), blk.tokens.end());
          c.verify = true, c.capture_rows = B;
          const std::vector<float> lv = eng.step({c});
          t_step += since(tp);
          int a = 0;
          while (a < cut && argmax(lv.data() + size_t(a) * VO) == blk.tokens[size_t(a)]) ++a;
          const int32_t next = argmax(lv.data() + size_t(a) * VO);
          for (int j = 0; j <= a; ++j) {
            // row j has fed the tokens up to index `at` of the sequence and predicts token at + 1: plain_rows[at + 1]
            const int64_t at = pos - n + j;
            bool agree = at + 1 < int64_t(plain_rows.size());
            for (int64_t k = 0; agree && k <= at && k < int64_t(sids.size()) + j; ++k) {
              const int32_t sk = k < int64_t(sids.size()) ? sids[size_t(k)] : (k == int64_t(sids.size()) ? anchor : blk.tokens[size_t(k - int64_t(sids.size()) - 1)]);
              agree = sk == ids[size_t(k)];
            }
            if (agree) compare(lv.data() + size_t(j) * VO, plain_rows[size_t(at + 1)], at);
          }
          sids.push_back(anchor);
          for (int j = 0; j < a; ++j) sids.push_back(blk.tokens[size_t(j)]);
          tp = std::chrono::steady_clock::now();
          eng.truncate(0, pos + a + 1);
          t_trunc += since(tp), tp = std::chrono::steady_clock::now();
          const auto& cp = eng.capture();
          dr->inject(0, cp.data, pos, a + 1);
          q.wait();
          t_inject += since(tp);
          pos += a + 1, anchor = next, ++steps;
        }
        const double st = since(ts);
        sids.resize(size_t(generate));
        int64_t same = 0;
        while (same < generate && sids[size_t(same)] == ids[size_t(same)]) ++same;
        std::printf("speculative: %lld tokens in %lld verify steps of %d (%.2f a step), %.2f s (%.2f tok/s); the first %lld of %lld equal to plain greedy's\n",
                    (long long)generate, (long long)steps, B, double(generate) / double(steps), st, double(generate) / st, (long long)same,
                    (long long)generate);
        std::printf("speculative steps (ms each): draft %.2f, verify %.2f, truncate %.2f, inject %.2f\n", t_draft * 1e3 / double(steps),
                    t_step * 1e3 / double(steps), t_trunc * 1e3 / double(steps), t_inject * 1e3 / double(steps));
        std::printf("speculative rows against plain decode's at the same positions (%lld rows): KL mean %.2e, max %.2e, top-1 %lld of %lld; first row past KL 1e-2: %lld\n",
                    (long long)kl_n, kl_n ? kl_sum / double(kl_n) : 0.0, kl_max, (long long)top_same, (long long)kl_n, (long long)first_far);
      }
      for (int rep = 0; rep < 2; ++rep)
        for (const int64_t K : block_time) {
          if (!eng.rewind(0, n)) throw std::runtime_error("--block-time: no checkpoint at the prompt's end");
          const int64_t steps = generate / K;
          prof.clear();
          const auto tb = std::chrono::steady_clock::now();
          for (int64_t s = 0; s < steps; ++s) {
            engine::inkling::Chunk c{0, n + s * K, std::vector<int32_t>(ids.begin() + s * K, ids.begin() + (s + 1) * K), K};
            (void)eng.step({c});
          }
          const double ms = since(tb) * 1e3 / double(steps);
          std::printf("block %2lld tokens: %6.2f ms a step (%.2f x one token's), %lld steps\n", (long long)K, ms,
                      ms / (dt * 1e3 / double(generate)), (long long)steps);
          if (profile && rep == 1)  // where a step's time goes, ms a step
            for (const auto& [k, v] : prof) std::printf("  block %2lld %-14s %8.3f ms\n", (long long)K, k.c_str(), v * 1e3 / double(steps));
        }
      std::ofstream gj(out + "/generated.json");
      gj << "[";
      for (size_t i = 0; i < ids.size(); ++i) gj << (i ? "," : "") << ids[i];
      gj << "]\n";
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
        if (std::getenv("EKLOW_TRACE") && i == 10) {
          q.wait();
          g_tr.names.clear();
          g_tr.on = true;
        }
        // EKLOW_ATTN_KERNELS (with EKLOW_PROFILE_DEVICE): the attention kernels' device times in the 10th decode step
        std::vector<sycl::event> alog;
        const bool alog_on = std::getenv("EKLOW_ATTN_KERNELS") && std::getenv("EKLOW_PROFILE_DEVICE") && i == 10;
        if (alog_on) attn::g_event_log = &alog;
        // EKLOW_DUMP_DECODE_STEP=k: decode step k's tensors (--tensors) into DIR/decode-k
        static const long dump_step = std::getenv("EKLOW_DUMP_DECODE_STEP") ? std::atol(std::getenv("EKLOW_DUMP_DECODE_STEP")) : -1;
        const std::string saved_out = out;
        if (long(i) == dump_step) {
          out += "/decode-" + std::to_string(i);
          std::filesystem::create_directories(out);
          dumping = true;
        }
        const std::vector<float> l = long(i) == dump_step ? eng.step({c}, &sink) : eng.step({c});
        dumping = false;
        out = saved_out;
        if (std::getenv("EKLOW_TRACE") && i == 10) {
          q.wait();
          g_tr.on = false;
          tracer_print();
        }
        if (alog_on) {
          attn::g_event_log = nullptr;
          using namespace sycl::info;
          double sum = 0;
          std::printf("attention kernels (us):");
          for (auto& e : alog) {
            const double d = double(e.get_profiling_info<event_profiling::command_end>() - e.get_profiling_info<event_profiling::command_start>()) * 1e-3;
            sum += d;
            std::printf(" %.0f", d);
          }
          std::printf("\nattention kernels: %zu, %.1f us in all\n", alog.size(), sum);
        }
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
