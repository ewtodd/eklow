// eklow-dflash-test: the DFlash 2 draft's checks (TASK qwen/dflash).
//
//   eklow-dflash-test torch   --draft GGUF --target GGUF --oracle DIR --tokens FILE --ref FILE.bin [--window W]
//       the reference forward against model.py run in PyTorch (ref_torch.py's output) on the target's real hidden states
//   eklow-dflash-test fast    ... same inputs: the serving forward against the reference forward
//   eklow-dflash-test time    --draft GGUF --target GGUF: timings on the card
// The inputs are what ref_torch.py read: ORACLE_DIR holds eklow-oracle's l_out-N.f32 for the draft's target layers.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include "model/dflash2.hpp"
#include "rt/device.hpp"

using namespace eklow;

namespace {

struct Args {
  std::string cmd, draft, target, oracle, tokens, ref;
  int window = 0, block = 0;
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
    else if (k == "--block") a.block = std::stoi(val());
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

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args a = parse(argc, argv);
    if (a.cmd == "torch") return cmd_torch(a);
    throw std::runtime_error("unknown command " + a.cmd);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-dflash-test: %s\n", e.what());
    return 1;
  }
}
