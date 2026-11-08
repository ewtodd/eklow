// eklow-dflash-host: the DFlash draft's reference forward (Dflash2Ref) on SYCL's host device, no card needed, on fixed
// random features and tokens: the lattice and the final hidden rows of a few blocks, written to a file and hashed. The
// same inputs before and after a change to src/model/dflash2.* must give the same bytes for a DFlash 2 draft (Qwen's path
// stays bit-identical); with a plain DFlash draft (z-lab's kind, as Gemma 4 26B A4B's was) it is the run whose output tools/dflash_test/ref_torch_plain.py
// is compared against (host_compare.py).
//
//   eklow-dflash-host DRAFT.gguf TARGET.gguf OUT.bin [--ctx N] [--B B] [--window W] [--seed S] [--blocks K] [--feat-scale F] [--feats FILE] [--anchors a,b,...]
// Built by tools/dflash_test/host-build.sh (clang++ -fsycl-targets=native_cpu over the plain-SYCL sources only).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "model/dflash2.hpp"

using namespace eklow;

namespace {
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc < 4) throw std::runtime_error("usage: eklow-dflash-host DRAFT.gguf TARGET.gguf OUT.bin [--ctx N] [--B B] [--window W] [--seed S] [--blocks K] [--feat-scale F] [--feats FILE] [--anchors a,b,...]");
  const std::string draft_path = argv[1], target_path = argv[2], out_path = argv[3];
  int ctx = 48, B = 8, window = 0, blocks = 2;
  uint64_t seed = 1;
  float feat_scale = 1.0f;
  std::string feats_path;
  std::vector<int32_t> anchors;
  for (int i = 4; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
      return argv[++i];
    };
    if (k == "--ctx") ctx = std::stoi(val());
    else if (k == "--B") B = std::stoi(val());
    else if (k == "--window") window = std::stoi(val());
    else if (k == "--seed") seed = std::stoull(val());
    else if (k == "--blocks") blocks = std::stoi(val());
    else if (k == "--feat-scale") feat_scale = std::stof(val());
    else if (k == "--feats") feats_path = val();  // fp32 [ctx + 3 * blocks][n_feat], instead of the random ones
    else if (k == "--anchors") {  // comma separated token ids, one per block
      std::stringstream ss(val());
      for (std::string t; std::getline(ss, t, ',');) anchors.push_back(std::stoi(t));
    }
    else throw std::runtime_error("unknown argument " + k);
  }
  sycl::queue q(sycl::default_selector_v, sycl::property::queue::in_order{});
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  const auto draft = gguf::File::open(draft_path), target = gguf::File::open(target_path);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  const auto w = model::Dflash2Weights::upload(draft, target, cfg, q);
  const int64_t extra = int64_t(blocks) * 3;  // the contexts of later blocks
  model::Dflash2Ref ref(q, cfg, w, ctx + extra + B + 8, window);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> nd(0.0f, feat_scale);
  std::vector<float> feats(size_t(ctx + extra) * size_t(cfg.n_feat()));
  for (auto& f : feats) f = nd(rng);
  if (!feats_path.empty()) {
    std::ifstream in(feats_path, std::ios::binary);
    in.read(reinterpret_cast<char*>(feats.data()), std::streamsize(feats.size() * 4));
    if (!in) throw std::runtime_error("short read of " + feats_path);
  }
  ref.inject(std::span<const float>(feats.data(), size_t(ctx) * size_t(cfg.n_feat())), 0, ctx);
  std::ofstream out(out_path, std::ios::binary);
  uint64_t all = 1469598103934665603ull;
  int64_t next = ctx;
  for (int b = 0; b < blocks; ++b) {
    int32_t anchor = int32_t(rng() % uint64_t(cfg.n_vocab));
    if (b < int(anchors.size())) anchor = anchors[size_t(b)];
    model::BlockDebug dbg;
    const serve::Lattice lat = ref.block(anchor, next, B, &dbg);
    const uint64_t hc = fnv(lat.cand.data(), lat.cand.size() * 4), hu = fnv(lat.unary.data(), lat.unary.size() * 4),
                   hs = fnv(lat.score.data(), lat.score.size() * 4), hh = fnv(dbg.hidden.data(), dbg.hidden.size() * 4);
    std::printf("block %d: anchor %d pos %lld  cand %016llx unary %016llx score %016llx hidden %016llx\n", b, anchor, (long long)next,
                (unsigned long long)hc, (unsigned long long)hu, (unsigned long long)hs, (unsigned long long)hh);
    std::printf("  top1 per position:");
    for (int i = 0; i < lat.positions(); ++i) std::printf(" %d(%.3f)", lat.cand[size_t(i) * size_t(lat.top_k)], lat.unary[size_t(i) * size_t(lat.top_k)]);
    std::printf("\n");
    for (auto h : {hc, hu, hs, hh}) all = (all ^ h) * 1099511628211ull;
    const int32_t hdr[4] = {anchor, int32_t(next), B, lat.top_k};
    out.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
    out.write(reinterpret_cast<const char*>(lat.cand.data()), std::streamsize(lat.cand.size() * 4));
    out.write(reinterpret_cast<const char*>(lat.unary.data()), std::streamsize(lat.unary.size() * 4));
    out.write(reinterpret_cast<const char*>(lat.score.data()), std::streamsize(lat.score.size() * 4));
    out.write(reinterpret_cast<const char*>(dbg.hidden.data()), std::streamsize(dbg.hidden.size() * 4));
    // the next block's context: three more committed rows
    ref.inject(std::span<const float>(feats.data() + size_t(next) * size_t(cfg.n_feat()), size_t(3) * size_t(cfg.n_feat())), next, 3);
    next += 3;
  }
  std::printf("total %016llx\n", (unsigned long long)all);
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
