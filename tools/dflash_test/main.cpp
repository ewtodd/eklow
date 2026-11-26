// eklow-dflash-test: the DFlash 2 draft's checks (TASK qwen/dflash).
//
//   eklow-dflash-test torch   --draft GGUF --target GGUF --oracle DIR --tokens FILE --ref FILE.bin [--window W]
//       the reference forward against model.py run in PyTorch (ref_torch.py's output) on the target's real hidden states
//   eklow-dflash-test fast    ... same inputs: the serving forward against the reference forward
//   eklow-dflash-test time    --draft GGUF --target GGUF: timings on the card
//   eklow-dflash-test plain   --draft Q8_0.gguf --target GGUF --feats F.f32 --anchors a,b,.. --ctx N [--block B] [--window W] [--full-context N] [--no-int8]
//       a plain DFlash draft (z-lab's kind; the one built for was Gemma 4 26B A4B's, since removed) on fixed features: the serving forward against the reference (tools/dflash_test/ref_torch_plain.py makes the inputs)
//   eklow-dflash-test dump    --draft GGUF --target GGUF [--ctx N] [--block B] [--window W]: hashes of the serving forward's lattices on fixed random features
// The inputs are what ref_torch.py read: ORACLE_DIR holds eklow-oracle's l_out-N.f32 for the draft's target layers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kernels/ref_dflash.hpp"
#include "model/dflash2.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

struct Args {
  std::string cmd, draft, target, oracle, tokens, ref, feats;
  int window = 0, block = 0, ctx = 0, sliding_causal = -1;
  int64_t full_context = 0;
  bool verbose = false, emulate = false, no_int8 = false;
  std::vector<int32_t> anchors;
  int stage_layer = 0;
  int from = 20, stride = 3;
  float jitter = 0;
  std::vector<int64_t> positions;
};

Args parse(int argc, char** argv) {
  Args a;
  if (argc < 2) throw std::runtime_error("usage: eklow-dflash-test torch|fast|time ...");
  a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(k + " needs a value");
      return argv[++i];
    };
    if (k == "--draft") a.draft = val();
    else if (k == "--target") a.target = val();
    else if (k == "--oracle") a.oracle = val();
    else if (k == "--tokens") a.tokens = val();
    else if (k == "--ref") a.ref = val();
    else if (k == "--window") a.window = std::stoi(val());
    else if (k == "--verbose") a.verbose = true;
    else if (k == "--emulate") a.emulate = true;
    else if (k == "--from") a.from = std::stoi(val());
    else if (k == "--stride") a.stride = std::stoi(val());
    else if (k == "--jitter") a.jitter = std::stof(val());
    else if (k == "--stage-layer") a.stage_layer = std::stoi(val());
    else if (k == "--block") a.block = std::stoi(val());
    else if (k == "--feats") a.feats = val();
    else if (k == "--ctx") a.ctx = std::stoi(val());
    else if (k == "--full-context") a.full_context = std::stoll(val());
    else if (k == "--no-int8") a.no_int8 = true;
    else if (k == "--sliding-causal") a.sliding_causal = std::stoi(val());
    else if (k == "--anchors") {
      std::stringstream ss(val());
      for (std::string t; std::getline(ss, t, ',');) a.anchors.push_back(std::stoi(t));
    }
    else throw std::runtime_error("unknown argument " + k);
  }
  return a;
}

// ref_torch.py's records.
struct Record {
  bool is_float;
  std::vector<uint32_t> shape;
  std::vector<float> f;
  std::vector<int32_t> i;
};
std::map<std::string, Record> read_records(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::map<std::string, Record> out;
  uint32_t nl;
  while (in.read(reinterpret_cast<char*>(&nl), 4)) {
    std::string name(nl, 0);
    in.read(name.data(), nl);
    uint8_t dt;
    uint32_t nd;
    in.read(reinterpret_cast<char*>(&dt), 1);
    in.read(reinterpret_cast<char*>(&nd), 4);
    Record r;
    r.is_float = dt == 0;
    r.shape.resize(nd);
    in.read(reinterpret_cast<char*>(r.shape.data()), 4 * nd);
    size_t n = 1;
    for (auto s : r.shape) n *= s;
    if (r.is_float) r.f.resize(n), in.read(reinterpret_cast<char*>(r.f.data()), 4 * n);
    else r.i.resize(n), in.read(reinterpret_cast<char*>(r.i.data()), 4 * n);
    out[name] = std::move(r);
  }
  return out;
}

std::vector<float> read_f32(const std::string& path, size_t n) {
  std::ifstream in(path, std::ios::binary);
  std::vector<float> v(n);
  in.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 4));
  if (!in) throw std::runtime_error("short read of " + path);
  return v;
}

// The target features [n][n_feat] of the oracle's layers.
std::vector<float> load_features(const model::Dflash2Config& c, const std::string& dir, int64_t n) {
  const int E = c.n_embd, F = c.n_feat();
  std::vector<float> feats(size_t(n) * F);
  for (size_t j = 0; j < c.target_layers.size(); ++j) {
    const auto layer = read_f32(dir + "/l_out-" + std::to_string(c.target_layers[j]) + ".f32", size_t(n) * E);
    for (int64_t t = 0; t < n; ++t) std::memcpy(&feats[size_t(t) * F + j * E], &layer[size_t(t) * E], size_t(E) * 4);
  }
  return feats;
}

std::vector<int32_t> load_tokens(const std::string& path) {
  std::ifstream in(path);
  std::vector<int32_t> t;
  int x;
  while (in >> x) t.push_back(x);
  return t;
}

// A lattice with its candidates sorted by id, which is how ref_torch.py writes its own.
serve::Lattice by_id(const serve::Lattice& l) {
  const int K = l.top_k, P = l.positions();
  serve::Lattice o = l;
  std::vector<std::vector<int>> perm(static_cast<size_t>(P), std::vector<int>(static_cast<size_t>(K)));
  for (int i = 0; i < P; ++i) {
    std::iota(perm[size_t(i)].begin(), perm[size_t(i)].end(), 0);
    std::sort(perm[size_t(i)].begin(), perm[size_t(i)].end(), [&](int a, int b) { return l.cand[size_t(i) * K + a] < l.cand[size_t(i) * K + b]; });
  }
  for (int i = 0; i < P; ++i)
    for (int b = 0; b < K; ++b) {
      const int sb = perm[size_t(i)][size_t(b)];
      o.cand[size_t(i) * K + b] = l.cand[size_t(i) * K + sb];
      o.unary[size_t(i) * K + b] = l.unary[size_t(i) * K + sb];
      for (int a = 0; a < K; ++a) {
        const int sa = i == 0 ? a : perm[size_t(i - 1)][size_t(a)];
        o.score[(size_t(i) * K + a) * K + b] = l.score[(size_t(i) * K + sa) * K + sb];
      }
    }
  return o;
}

struct Err {
  double max_abs = 0, sq_err = 0, sq_ref = 0;
  void add(double x, double r) {
    max_abs = std::max(max_abs, std::fabs(x - r));
    sq_err += (x - r) * (x - r);
    sq_ref += r * r;
  }
  double rel() const { return sq_ref > 0 ? std::sqrt(sq_err / sq_ref) : 0; }
};

int cmd_torch(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft), target = gguf::File::open(a.target);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  const auto recs = read_records(a.ref);
  const auto& meta = recs.at("meta").i;
  const int B = meta[0], window = meta[1], n_tok = meta[2], n_blocks = meta[3];
  std::printf("draft %s: block %d window %d, %d tokens, %d blocks\n", a.draft.c_str(), B, window, n_tok, n_blocks);
  const auto weights = model::Dflash2Weights::upload(draft, target, cfg, q);
  const auto feats = load_features(cfg, a.oracle, n_tok);
  const auto tokens = load_tokens(a.tokens);
  std::vector<int64_t> ps;
  for (const auto& [name, r] : recs)
    if (name.rfind("path", 0) == 0) ps.push_back(std::stoll(name.substr(4)));
  std::sort(ps.begin(), ps.end());
  model::Dflash2Ref ref(q, cfg, weights, n_tok + 16, window);
  Err he, ue, le;
  int path_ok = 0, cand_diff = 0;
  int64_t next = 0;
  for (int64_t p : ps) {
    ref.inject(std::span<const float>(feats.data() + size_t(next) * cfg.n_feat(), size_t(p - next) * cfg.n_feat()), next, p - next);
    next = p;
    model::BlockDebug dbg;
    const serve::Lattice lat = ref.block(tokens[size_t(p)], p, B, &dbg);
    const serve::Lattice s = by_id(lat);
    const auto& hid = recs.at("hid" + std::to_string(p)).f;
    const auto& cand = recs.at("cand" + std::to_string(p)).i;
    const auto& un = recs.at("unary" + std::to_string(p)).f;
    const auto& lt = recs.at("lat" + std::to_string(p)).f;
    const auto& path = recs.at("path" + std::to_string(p)).i;
    Err h, u, l;
    for (size_t i = 0; i < hid.size(); ++i) h.add(dbg.hidden[i], hid[i]);
    int cd = 0;
    for (size_t i = 0; i < cand.size(); ++i) cd += cand[i] != s.cand[i];
    for (size_t i = 0; i < un.size(); ++i) u.add(s.unary[i], un[i]);
    for (size_t i = 0; i < lt.size(); ++i) l.add(s.score[i], lt[i]);
    std::mt19937_64 rng(1);
    const serve::DraftBlock blk = serve::walk(lat, 0.0f, rng);
    bool ok = true;
    for (int i = 0; i < B - 1; ++i) ok &= blk.tokens[size_t(i)] == path[size_t(i)];
    std::printf("p=%4lld hidden rel %.2e max %.2e | candidates differing %d | unary max %.2e | lattice max %.2e rel %.2e | greedy path %s\n",
                (long long)p, h.rel(), h.max_abs, cd, u.max_abs, l.max_abs, l.rel(), ok ? "equal" : "DIFFERENT");
    he.sq_err += h.sq_err, he.sq_ref += h.sq_ref, he.max_abs = std::max(he.max_abs, h.max_abs);
    ue.max_abs = std::max(ue.max_abs, u.max_abs);
    le.sq_err += l.sq_err, le.sq_ref += l.sq_ref, le.max_abs = std::max(le.max_abs, l.max_abs);
    path_ok += ok, cand_diff += cd;
  }
  std::printf("total: hidden rel %.2e (max %.2e), unary max %.2e, lattice rel %.2e (max %.2e), %d of %zu greedy paths equal, %d candidates differ\n",
              he.rel(), he.max_abs, ue.max_abs, le.rel(), le.max_abs, path_ok, ps.size(), cand_diff);
  return 0;
}

// The serving forward against the reference forward on the same weights (the draft's Q8_0 file), block by block: three
// comparisons, each of unary scores (the head's top-16 logits), lattice scores and the greedy walk: serving vs the exact
// fp32 reference, the reference with the serving activations' rounding emulated vs exact (what that rounding costs by
// itself), and serving vs the emulating reference (what is left).
int cmd_fast(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft), target = gguf::File::open(a.target);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  const auto recs = read_records(a.ref);
  const auto& meta = recs.at("meta").i;
  const int B = meta[0], window = meta[1], n_tok = meta[2];
  const auto weights = model::Dflash2Weights::upload(draft, target, cfg, q);
  const model::OwnedTarget owned(q, target);
  model::Dflash2::Options opt;
  opt.max_block = B;
  opt.window = window;
  model::Dflash2 fast(q, draft, owned.view(), opt);
  std::printf("draft %s: block %d window %d, %d tokens; the draft holds %.0f MiB on the card\n", a.draft.c_str(), B, window, n_tok,
              double(fast.bytes()) / 1048576.0);
  const auto feats = load_features(cfg, a.oracle, n_tok);
  float* dfeats = sycl::malloc_device<float>(feats.size(), q);
  q.memcpy(dfeats, feats.data(), feats.size() * sizeof(float)).wait();
  const auto tokens = load_tokens(a.tokens);
  std::vector<int64_t> ps;
  for (const auto& [name, r] : recs)
    if (name.rfind("path", 0) == 0) ps.push_back(std::stoll(name.substr(4)));
  std::sort(ps.begin(), ps.end());
  model::Dflash2Ref exact(q, cfg, weights, n_tok + 16, window, a.jitter != 0 ? true : false, a.jitter), emu(q, cfg, weights, n_tok + 16, window, true);
  int64_t next = 0;
  struct Pair {
    const char* name;
    Err u, l;
    int tok_ok = 0, tok_all = 0, set_diff = 0;
  } pairs[3] = {{"serving vs exact"}, {"emulated vs exact"}, {"serving vs emulated"}};
  for (int64_t p : ps) {
    const std::span<const float> rows(feats.data() + size_t(next) * cfg.n_feat(), size_t(p - next) * cfg.n_feat());
    exact.inject(rows, next, p - next);
    emu.inject(rows, next, p - next);
    fast.inject(0, dfeats + next * cfg.n_feat(), next, p - next);
    next = p;
    model::BlockDebug rd, ed, fd;
    rd.stage_layer = ed.stage_layer = fd.stage_layer = a.stage_layer;
    const serve::Lattice lat[3] = {by_id(fast.lattices({{0, tokens[size_t(p)], p}}, B, nullptr, &fd)[0]), by_id(emu.block(tokens[size_t(p)], p, B, &ed)),
                                   by_id(exact.block(tokens[size_t(p)], p, B, &rd))};
    const int pa[3][2] = {{0, 2}, {1, 2}, {0, 1}};
    std::printf("p=%4lld", (long long)p);
    for (int k = 0; k < 3; ++k) {
      const serve::Lattice &x = lat[pa[k][0]], &y = lat[pa[k][1]];
      Err u, l;
      int cd = 0, same = 0;
      for (size_t i = 0; i < y.cand.size(); ++i) cd += x.cand[i] != y.cand[i];
      for (size_t i = 0; i < y.unary.size(); ++i) u.add(x.unary[i], y.unary[i]);
      for (size_t i = 0; i < y.score.size(); ++i) l.add(x.score[i], y.score[i]);
      std::mt19937_64 rng(1);
      const auto xb = serve::walk(x, 0.0f, rng), yb = serve::walk(y, 0.0f, rng);
      for (int i = 0; i < B - 1; ++i) same += xb.tokens[size_t(i)] == yb.tokens[size_t(i)];
      std::printf(" | %s: cand diff %2d unary rel %.1e tokens %d/%d", pairs[k].name, cd, u.rel(), same, B - 1);
      Pair& pr = pairs[k];
      pr.u.sq_err += u.sq_err, pr.u.sq_ref += u.sq_ref, pr.u.max_abs = std::max(pr.u.max_abs, u.max_abs);
      pr.l.sq_err += l.sq_err, pr.l.sq_ref += l.sq_ref, pr.l.max_abs = std::max(pr.l.max_abs, l.max_abs);
      pr.tok_ok += same, pr.tok_all += B - 1, pr.set_diff += cd;
    }
    std::printf("\n");
    if (a.verbose) {
      for (size_t i = 0; i < ed.stages.size(); ++i) {
        Err e;
        for (size_t j = 0; j < ed.stages[i].second.size(); ++j) e.add(fd.stages[i].second[j], ed.stages[i].second[j]);
        std::printf("   layer %d %s: serving vs emulated rel %.2e max %.2e (signal rms %.2e)\n", a.stage_layer, ed.stages[i].first.c_str(), e.rel(), e.max_abs,
                    std::sqrt(e.sq_ref / double(ed.stages[i].second.size())));
      }
      for (size_t l = 0; l < ed.layers.size(); ++l) {
        Err e, at;
        for (size_t i = 0; i < ed.layers[l].size(); ++i) e.add(fd.layers[l][i], ed.layers[l][i]);
        for (size_t i = 0; i < ed.parts[l].size(); ++i) at.add(fd.parts[l][i], ed.parts[l][i]);
        std::printf("   layer %zu: attention output rel %.2e, residual stream rel %.2e\n", l, at.rel(), e.rel());
      }
    }
  }
  for (const Pair& pr : pairs)
    std::printf("total %-20s: unary rel %.2e (max %.2e), lattice rel %.2e (max %.2e), greedy tokens equal %d of %d, %d candidates differ\n", pr.name, pr.u.rel(),
                pr.u.max_abs, pr.l.rel(), pr.l.max_abs, pr.tok_ok, pr.tok_all, pr.set_diff);
  sycl::free(dfeats, q);
  return 0;
}

// A plain DFlash draft (z-lab's kind; the one built for was Gemma 4 26B A4B's, since removed) on fixed features and anchors (ref_torch_plain.py writes the feature file and compares
// the reference forward with model.py; eklow-dflash-host runs that reference without a card): blocks at positions
// ctx, ctx + 3, ... after injecting the features of the rows before them, drafted by the serving forward (the draft's Q8_0
// file) and by the exact fp32 reference (--emulate-free) and the reference with the serving activations' rounding. Prints
// the three pairs as `fast` does. With --full-context N the serving forward's full layer keeps N positions (a ring); the
// run is repeated with the whole context kept, and the two servings are compared: what a capped window changes.
int cmd_plain(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft), target = gguf::File::open(a.target);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  if (!cfg.plain) throw std::runtime_error("plain: not a plain DFlash draft");
  const int B = a.block ? a.block : cfg.block_size, window = a.window, nb = int(a.anchors.size());
  if (nb == 0 || a.ctx <= 0 || a.feats.empty()) throw std::runtime_error("plain: needs --feats, --anchors and --ctx");
  const int64_t n_rows = a.ctx + 3 * nb;
  const auto feats = read_f32(a.feats, size_t(n_rows) * size_t(cfg.n_feat()));
  const auto weights = model::Dflash2Weights::upload(draft, target, cfg, q);
  const model::OwnedTarget owned(q, target);
  auto make = [&](int64_t full_context) {
    model::Dflash2::Options opt;
    opt.max_block = B;
    opt.window = window;
    opt.full_context = full_context;
    opt.full_int8 = !a.no_int8;
    opt.sliding_causal = a.sliding_causal;
    return std::make_unique<model::Dflash2>(q, draft, owned.view(), opt);
  };
  auto fast = make(a.full_context ? a.full_context : 262144);
  std::printf("draft %s: block %d, window %d, full layer %lld positions (%s), the draft holds %.0f MiB on the card\n", a.draft.c_str(), B,
              window ? window : cfg.window, (long long)(a.full_context ? a.full_context : 262144), a.no_int8 ? "fp16" : "int8", double(fast->bytes()) / 1048576.0);
  std::unique_ptr<model::Dflash2> whole = a.full_context ? make(262144) : nullptr;
  float* dfeats = sycl::malloc_device<float>(feats.size(), q);
  q.memcpy(dfeats, feats.data(), feats.size() * sizeof(float)).wait();
  model::Dflash2Ref exact(q, cfg, weights, n_rows + 16, window, false), emu(q, cfg, weights, n_rows + 16, window, true);
  struct Pair {
    const char* name;
    Err u;
    int tok_ok = 0, tok_all = 0, set_diff = 0;
  } pairs[4] = {{"serving vs exact"}, {"emulated vs exact"}, {"serving vs emulated"}, {"capped vs whole serving"}};
  int64_t next = 0;
  for (int b = 0; b < nb; ++b) {
    const int64_t p = a.ctx + 3 * b;
    const std::span<const float> rows(feats.data() + size_t(next) * cfg.n_feat(), size_t(p - next) * cfg.n_feat());
    exact.inject(rows, next, p - next);
    emu.inject(rows, next, p - next);
    fast->inject(0, dfeats + next * cfg.n_feat(), next, p - next);
    if (whole) whole->inject(0, dfeats + next * cfg.n_feat(), next, p - next);
    next = p;
    const int32_t anchor = a.anchors[size_t(b)];
    const serve::Lattice lat[4] = {by_id(fast->lattices({{0, anchor, p}}, B)[0]), by_id(emu.block(anchor, p, B)), by_id(exact.block(anchor, p, B)),
                                   whole ? by_id(whole->lattices({{0, anchor, p}}, B)[0]) : serve::Lattice{}};
    const int pa[4][2] = {{0, 2}, {1, 2}, {0, 1}, {0, 3}};
    std::printf("p=%4lld", (long long)p);
    for (int k = 0; k < (whole ? 4 : 3); ++k) {
      const serve::Lattice &x = lat[pa[k][0]], &y = lat[pa[k][1]];
      Err u;
      int cd = 0, same = 0;
      for (size_t i = 0; i < y.cand.size(); ++i) cd += x.cand[i] != y.cand[i];
      for (size_t i = 0; i < y.unary.size(); ++i) u.add(x.unary[i], y.unary[i]);
      std::mt19937_64 rng(1);
      const auto xb = serve::walk(x, 0.0f, rng), yb = serve::walk(y, 0.0f, rng);
      for (int i = 0; i < B - 1; ++i) same += xb.tokens[size_t(i)] == yb.tokens[size_t(i)];
      std::printf(" | %s: cand diff %2d unary rel %.1e tokens %d/%d", pairs[k].name, cd, u.rel(), same, B - 1);
      Pair& pr = pairs[k];
      pr.u.sq_err += u.sq_err, pr.u.sq_ref += u.sq_ref, pr.u.max_abs = std::max(pr.u.max_abs, u.max_abs);
      pr.tok_ok += same, pr.tok_all += B - 1, pr.set_diff += cd;
    }
    std::printf("\n");
  }
  for (int k = 0; k < (whole ? 4 : 3); ++k)
    std::printf("total %-24s: unary rel %.2e (max %.2e), greedy tokens equal %d of %d, %d candidates differ\n", pairs[k].name, pairs[k].u.rel(), pairs[k].u.max_abs,
                pairs[k].tok_ok, pairs[k].tok_all, pairs[k].set_diff);
  sycl::free(dfeats, q);
  return 0;
}

// The serving forward's lattices on random features, as hashes: the same line from two builds means the same bytes. For
// checking that a change to the draft code leaves a draft's output as it was (run it with both builds on the card, diff).
int cmd_dump(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft), target = gguf::File::open(a.target);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  const int B = a.block ? a.block : cfg.block_size, nb = a.anchors.empty() ? 3 : int(a.anchors.size());
  const int64_t ctx = a.ctx ? a.ctx : 200, n_rows = ctx + 3 * nb;
  const model::OwnedTarget owned(q, target);
  model::Dflash2::Options opt;
  opt.max_block = B;
  opt.window = a.window;
  model::Dflash2 fast(q, draft, owned.view(), opt);
  std::mt19937_64 rng(7);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> feats(size_t(n_rows) * size_t(cfg.n_feat()));
  for (auto& f : feats) f = nd(rng);
  float* dfeats = sycl::malloc_device<float>(feats.size(), q);
  q.memcpy(dfeats, feats.data(), feats.size() * sizeof(float)).wait();
  auto fnv = [](const void* p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ static_cast<const uint8_t*>(p)[i]) * 1099511628211ull;
    return h;
  };
  fast.inject(0, dfeats, 0, ctx);
  int64_t next = ctx;
  for (int b = 0; b < nb; ++b) {
    const int32_t anchor = b < int(a.anchors.size()) ? a.anchors[size_t(b)] : int32_t(rng() % uint64_t(cfg.n_vocab));
    const serve::Lattice l = fast.lattices({{0, anchor, next}}, B)[0];
    std::printf("block %d anchor %d pos %lld: cand %016llx unary %016llx score %016llx\n", b, anchor, (long long)next, (unsigned long long)fnv(l.cand.data(), l.cand.size() * 4),
                (unsigned long long)fnv(l.unary.data(), l.unary.size() * 4), (unsigned long long)fnv(l.score.data(), l.score.size() * 4));
    fast.inject(0, dfeats + next * cfg.n_feat(), next, 3);
    next += 3;
  }
  sycl::free(dfeats, q);
  return 0;
}

// How well the draft predicts a real text, teacher forced: at every `stride`-th position of the text a block is drafted
// from the real features of the positions before it, and the accepted length is how many of its greedy tokens agree with
// the text that follows (the draft's acceptance if the target said the text). For the exact fp32 reference, the
// reference with the serving activations' rounding, and the serving forward: what the int8 activations cost the draft.
int cmd_accept(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft), target = gguf::File::open(a.target);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  const int B = a.block ? a.block : cfg.block_size, window = a.window;
  const auto tokens = load_tokens(a.tokens);
  const int64_t n_tok = int64_t(tokens.size());
  const auto weights = model::Dflash2Weights::upload(draft, target, cfg, q);
  const model::OwnedTarget owned(q, target);
  model::Dflash2::Options opt;
  opt.max_block = B;
  opt.window = window;
  model::Dflash2 fast(q, draft, owned.view(), opt);
  const auto feats = load_features(cfg, a.oracle, n_tok);
  float* dfeats = sycl::malloc_device<float>(feats.size(), q);
  q.memcpy(dfeats, feats.data(), feats.size() * sizeof(float)).wait();
  model::Dflash2Ref exact(q, cfg, weights, n_tok + 16, window, false), emu(q, cfg, weights, n_tok + 16, window, true);
  struct Tally {
    const char* name;
    double sum = 0;
    int n = 0;
    std::vector<int> hist;
  } t[3] = {{"exact fp32 reference"}, {"reference with int8 activations"}, {"serving forward"}};
  for (auto& x : t) x.hist.assign(size_t(B), 0);
  int64_t next = 0;
  int same_exact_serving = 0, blocks = 0;
  for (int64_t p = a.from; p + B <= n_tok; p += a.stride) {
    const std::span<const float> rows(feats.data() + size_t(next) * cfg.n_feat(), size_t(p - next) * cfg.n_feat());
    exact.inject(rows, next, p - next);
    emu.inject(rows, next, p - next);
    fast.inject(0, dfeats + next * cfg.n_feat(), next, p - next);
    next = p;
    const serve::Lattice lat[3] = {exact.block(tokens[size_t(p)], p, B), emu.block(tokens[size_t(p)], p, B), fast.lattices({{0, tokens[size_t(p)], p}}, B)[0]};
    std::mt19937_64 rng(1);
    std::vector<int32_t> w[3];
    for (int k = 0; k < 3; ++k) {
      w[k] = serve::walk(lat[k], 0.0f, rng).tokens;
      int agree = 0;
      while (agree < B - 1 && w[k][size_t(agree)] == tokens[size_t(p + 1 + agree)]) ++agree;
      t[k].sum += agree, ++t[k].n, ++t[k].hist[size_t(agree)];
    }
    same_exact_serving += w[0] == w[2];
    ++blocks;
  }
  std::printf("%d blocks of %d tokens (window %d), drafts compared with the text that follows; mean tokens agreeing (the accepted length, without the bonus token):\n", blocks, B,
              window ? window : cfg.window);
  for (auto& x : t) {
    std::printf("  %-34s %.3f   histogram of 0..%d agreeing:", x.name, x.sum / x.n, B - 1);
    for (int h : x.hist) std::printf(" %d", h);
    std::printf("\n");
  }
  std::printf("  the exact reference's and the serving forward's whole drafts are equal in %d of %d blocks\n", same_exact_serving, blocks);
  sycl::free(dfeats, q);
  return 0;
}

// Timings on the card: context injection (per row, at several row counts), and a block at B = 8 and 16 with its phases
// (layers, the head, the selector: top 16, codebook rows, lattice, copy back), one slot and two.
int cmd_time(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft), target = gguf::File::open(a.target);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  const model::OwnedTarget owned(q, target);
  model::Dflash2::Options opt;
  opt.slots = 2;
  opt.max_block = 16;
  model::Dflash2 d(q, draft, owned.view(), opt);
  const int F = cfg.n_feat();
  const int64_t rows = 2048;
  std::vector<float> h(size_t(rows) * F);
  std::mt19937 g(3);
  for (auto& x : h) x = std::normal_distribution<float>(0, 1)(g);
  float* feats = sycl::malloc_device<float>(h.size(), q);
  q.memcpy(feats, h.data(), h.size() * 4).wait();
  auto ms = [&](auto&& f, int reps) {
    f();
    q.wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) f();
    q.wait();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
  };
  std::printf("draft %s on the card: %.0f MiB, window %d\n", a.draft.c_str(), double(d.bytes()) / 1048576.0, d.window());
  for (int64_t n : {1, 4, 8, 16, 32, 256, 1024, 2048}) {
    d.reset(0);
    const double t = ms([&] { d.reset(0), d.inject(0, feats, 0, n); }, n >= 256 ? 5 : 20);
    std::printf("inject %4lld rows: %8.3f ms, %7.1f us a row\n", (long long)n, t, 1000 * t / double(n));
  }
  for (int B : {8, 16}) {
    for (int nb : {1, 2}) {
      for (int s = 0; s < nb; ++s) d.reset(s), d.inject(s, feats, 0, 700);
      std::vector<model::Dflash2::Request> reqs;
      for (int s = 0; s < nb; ++s) reqs.push_back({s, 1000 + s, 700});
      model::Profile pr;
      const int reps = 20;
      d.lattices(reqs, B);
      for (int i = 0; i < reps; ++i) d.lattices(reqs, B, &pr);
      const double total = ms([&] { d.lattices(reqs, B); }, reps);
      std::printf("block B=%2d, %d slot%s: %.3f ms (profiled phases: embed %.3f, layers %.3f, head %.3f, selector %.3f ms)\n", B, nb, nb > 1 ? "s" : "", total,
                  pr.embed_ms / reps, pr.layers_ms / reps, pr.head_ms / reps, pr.select_ms / reps);
    }
  }
  sycl::free(feats, q);
  return 0;
}

// Matmuls of the draft's weights on the matrix units against the reference's, on real activations: with int8 activations
// the result should equal the reference on the same rounded activations to fp32 rounding.
int cmd_matmul(const Args& a) {
  sycl::queue q(rt::pick_device(), sycl::property::queue::in_order{});
  const auto draft = gguf::File::open(a.draft);
  const auto cfg = model::Dflash2Config::from_gguf(draft);
  for (const char* name : {"blk.0.attn_conv_proj.weight", "blk.0.attn_q.weight", "blk.0.ffn_down.weight", "fc.weight"}) {
    const gguf::Tensor& t = *draft.tensor(name);
    auto* raw = sycl::malloc_device<uint8_t>(t.nbytes, q);
    q.memcpy(raw, t.data, t.nbytes).wait();
    const ref::Weight rw{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), raw, t.nbytes / t.ne[1]};
    xmx::WeightFormat f;
    xmx::format_of(t.type, f);
    const xmx::Weight xw = xmx::repack_format(q, rw, f);
    for (int rows : {1, 8, 16, 32}) {
      std::mt19937 g(5);
      std::vector<float> h(size_t(rows) * size_t(rw.k));
      for (size_t i = 0; i < h.size(); ++i) h[i] = std::normal_distribution<float>(0, 1)(g) * (i % 97 == 0 ? 8.0f : 1.0f);
      float *x = sycl::malloc_device<float>(h.size(), q), *xr = sycl::malloc_device<float>(h.size(), q);
      float *o1 = sycl::malloc_device<float>(size_t(rows) * size_t(rw.n), q), *o2 = sycl::malloc_device<float>(size_t(rows) * size_t(rw.n), q),
            *o3 = sycl::malloc_device<float>(size_t(rows) * size_t(rw.n), q);
      q.memcpy(x, h.data(), h.size() * 4).wait();
      xmx::Activations act = xmx::alloc_activations(q, 32, rw.k);
      xmx::quantize(q, x, rows, act);
      xmx::matmul(q, act, xw, o1);
      ref::round_acts(q, x, rows, rw.k, false, xr);
      ref::matmul(q, rw, xr, rows, o2);
      ref::matmul(q, rw, x, rows, o3);
      q.wait();
      std::vector<float> a1(size_t(rows) * size_t(rw.n)), a2(a1.size()), a3(a1.size());
      q.memcpy(a1.data(), o1, a1.size() * 4);
      q.memcpy(a2.data(), o2, a1.size() * 4);
      q.memcpy(a3.data(), o3, a1.size() * 4).wait();
      Err e_emul, e_raw;
      for (size_t i = 0; i < a1.size(); ++i) e_emul.add(a1[i], a2[i]), e_raw.add(a1[i], a3[i]);
      std::printf("%-28s rows %2d: fast vs reference on rounded activations rel %.2e, vs exact %.2e\n", name, rows, e_emul.rel(), e_raw.rel());
      xmx::free_activations(q, act);
      for (float* p : {x, xr, o1, o2, o3}) sycl::free(p, q);
    }
    xmx::free_weight(q, const_cast<xmx::Weight&>(xw));
    sycl::free(raw, q);
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    if (a.cmd == "torch") return cmd_torch(a);
    if (a.cmd == "fast") return cmd_fast(a);
    if (a.cmd == "plain") return cmd_plain(a);
    if (a.cmd == "dump") return cmd_dump(a);
    if (a.cmd == "matmul") return cmd_matmul(a);
    if (a.cmd == "accept") return cmd_accept(a);
    if (a.cmd == "time") return cmd_time(a);
    throw std::runtime_error("unknown command " + a.cmd);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-dflash-test: %s\n", e.what());
    return 1;
  }
}
