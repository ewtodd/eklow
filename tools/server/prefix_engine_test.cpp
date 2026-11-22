// eklow-prefix-engine-test: on the card, what the disk prefix cache and the rings' rewind do to Bolt's logits, and what a
// restore costs (DECISIONS 2026-11-22). Three slots of --ctx positions, chunks of 2,048, the last 64 positions' logits.
//
//   eklow-prefix-engine-test --model GGUF --doc-a TEXT --doc-b TEXT --store DIR [--ctx N]
//
// 1. The rewind. A slot holds A (8,192 tokens of doc-a); B shares A's first 3,000 and goes on with doc-b. main's admission
//    kept 3,000 - window = 2,488 positions and prefilled B from there although the rings no longer held that window
//    ("main"); the fix starts over ("fix"). Against B prefilled in an empty slot ("fresh"), and against B prefilled from
//    2,488 in a slot whose rings do hold the window ("valid": the same chunk shapes as main's, so main against valid is the
//    stale K/V alone, and valid against fresh the noise of other chunk shapes).
// 2. A restore. S (20,000 tokens of doc-b) is prefilled; its pages go through the store (sealed, written, the files dropped
//    from the page cache); slot 1 restores the 74 pages (18,944 positions) and prefills the rest. Against slot 0 rewound to
//    the same point within the rings' slack and prefilled alike: equal shapes, so they should agree bit for bit. Times: the
//    prefill, the export to host memory, the store's writes, the restore cold and warm.
#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "engine/engine_gemma4.hpp"
#include "model/gguf.hpp"
#include "serve/prefix_store.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;
using engine::gemma4::Chunk;
using engine::gemma4::Engine;
namespace fs = std::filesystem;

namespace {

constexpr int64_t kChunk = 2048, kRows = 64;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Prefills tokens[from..] into the slot (which holds `from` positions) in chunks of kChunk; the last kRows positions' logits.
std::vector<float> prefill(Engine& e, int slot, const std::vector<int32_t>& t, int64_t from) {
  std::vector<float> out;
  const int64_t n = int64_t(t.size());
  for (int64_t p = from; p < n; p += kChunk) {
    const int64_t m = std::min(kChunk, n - p);
    const bool last = p + m == n;
    Chunk c{slot, p, std::vector<int32_t>(t.begin() + p, t.begin() + p + m), last ? std::min(kRows, m) : 0};
    std::vector<float> l = e.step({c});
    if (last) out = std::move(l);
  }
  return out;
}

struct Diff {
  double mean_kl = 0, max_kl = 0, max_abs = 0;
  int top1 = 0, rows = 0;
};
// p against q, row by row: KL(p || q) of the softmax, top-1 agreement, the largest logit difference.
Diff diff(const std::vector<float>& p, const std::vector<float>& q, int64_t V) {
  Diff d;
  d.rows = int(p.size() / size_t(V));
  for (int r = 0; r < d.rows; ++r) {
    const float* a = p.data() + size_t(r) * size_t(V);
    const float* b = q.data() + size_t(r) * size_t(V);
    double ma = -1e30, mb = -1e30;
    int64_t ia = 0, ib = 0;
    for (int64_t i = 0; i < V; ++i) {
      if (a[i] > ma) ma = a[i], ia = i;
      if (b[i] > mb) mb = b[i], ib = i;
      d.max_abs = std::max(d.max_abs, double(std::fabs(a[i] - b[i])));
    }
    double za = 0, zb = 0;
    for (int64_t i = 0; i < V; ++i) za += std::exp(a[i] - ma), zb += std::exp(b[i] - mb);
    const double la = std::log(za) + ma, lb = std::log(zb) + mb;
    double kl = 0;
    for (int64_t i = 0; i < V; ++i) {
      const double lpa = a[i] - la, lpb = b[i] - lb;
      kl += std::exp(lpa) * (lpa - lpb);
    }
    d.mean_kl += kl / d.rows;
    d.max_kl = std::max(d.max_kl, kl);
    d.top1 += ia == ib;
  }
  return d;
}

void show(const char* what, const Diff& d) {
  std::printf("  %-44s top-1 %2d/%d, mean KL %.3e, worst %.3e, largest |logit diff| %.4f\n", what, d.top1, d.rows, d.mean_kl, d.max_kl, d.max_abs);
}

std::vector<int32_t> tokens_of(const text::Tokenizer& tok, const std::string& path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  return tok.encode(ss.str(), false);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model, doc_a, doc_b, store_dir;
  int64_t ctx = 32768;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
    if (a == "--model") model = next();
    else if (a == "--doc-a") doc_a = next();
    else if (a == "--doc-b") doc_b = next();
    else if (a == "--store") store_dir = next();
    else if (a == "--ctx") ctx = std::atoll(next().c_str());
  }
  if (model.empty() || doc_a.empty() || doc_b.empty() || store_dir.empty()) {
    std::fprintf(stderr, "usage: eklow-prefix-engine-test --model GGUF --doc-a TEXT --doc-b TEXT --store DIR [--ctx N]\n");
    return 2;
  }
  auto file = gguf::File::open(model);
  const text::Tokenizer tok = text::Tokenizer::from_gguf(file);
  const std::vector<int32_t> da = tokens_of(tok, doc_a), db = tokens_of(tok, doc_b);
  if (da.size() < 8192 || db.size() < 20000) {
    std::fprintf(stderr, "the documents are too short (%zu, %zu tokens; 8,192 and 20,000 needed)\n", da.size(), db.size());
    return 2;
  }
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
  engine::gemma4::Config ec;
  ec.n_slots = 3;
  ec.max_ctx = ctx;
  ec.max_batch = kChunk;
  ec.max_chunk = kChunk;
  ec.max_logits = kRows;
  Engine e(q, file, ec);
  const int64_t V = e.model().n_vocab;
  std::printf("engine: %d slots of %lld, rings of %lld pages, slack %lld, window %d\n", e.n_slots(), (long long)ctx, (long long)e.ring_pages(),
              (long long)e.ring_slack(), e.model().window);

  std::printf("1. the rewind (A: 8,192 tokens; B: A's first 3,000, then 3,000 others)\n");
  const std::vector<int32_t> A(da.begin(), da.begin() + 8192);
  std::vector<int32_t> B(da.begin(), da.begin() + 3000);
  B.insert(B.end(), db.begin(), db.begin() + 3000);
  const int64_t keep_main = 3000 - e.model().window;
  e.release(2);
  const auto fresh = prefill(e, 2, B, 0);
  e.release(0);
  prefill(e, 0, A, 0);
  e.truncate(0, keep_main);  // main: 8,192 - 3,000 > slack, so 3,000 - window, with the rings past it
  const auto main_l = prefill(e, 0, B, keep_main);
  e.release(1);
  prefill(e, 1, std::vector<int32_t>(A.begin(), A.begin() + 3500), 0);
  e.truncate(1, keep_main);  // 3,500 - 2,488 <= slack: the rings hold the window
  const auto valid = prefill(e, 1, B, keep_main);
  e.release(0);
  prefill(e, 0, A, 0);
  e.release(0);  // the fix: the sequence starts over
  const auto fix = prefill(e, 0, B, 0);
  show("valid reuse against fresh (chunk shapes)", diff(fresh, valid, V));
  show("main's rewind against valid (stale K/V)", diff(valid, main_l, V));
  show("main's rewind against fresh", diff(fresh, main_l, V));
  show("the fix against fresh", diff(fresh, fix, V));

  std::printf("2. a restore (S: 20,000 tokens)\n");
  const std::vector<int32_t> S(db.begin(), db.begin() + 20000);
  const int64_t reach_pages = 4, t = (int64_t(S.size()) - 1) / attn::kPage - reach_pages, w = e.window_pages();
  const auto pb = e.page_bytes();
  std::printf("  restore point %lld pages (%lld positions); a page is %.1f MiB global, %.1f MiB sliding\n", (long long)t, (long long)(t * attn::kPage),
              pb.global / 1048576.0, pb.sliding / 1048576.0);
  e.release(0);
  double t0 = now_s();
  const auto full = prefill(e, 0, S, 0);
  const double prefill_s = now_s() - t0;
  std::printf("  prefill of 20,000 from empty: %.2f s, %.0f tok/s\n", prefill_s, 20000 / prefill_s);
  std::vector<uint8_t*> gbuf(static_cast<size_t>(t)), sbuf(static_cast<size_t>(w));
  for (auto& b : gbuf) b = sycl::malloc_host<uint8_t>(pb.global, q);
  for (auto& b : sbuf) b = sycl::malloc_host<uint8_t>(pb.sliding, q);
  t0 = now_s();
  sycl::event ev;
  for (int64_t j = 0; j < t; ++j) ev = e.export_page(0, j, gbuf[size_t(j)], nullptr);
  ev.wait();
  const double export_s = now_s() - t0;
  for (int64_t j = t - w; j < t; ++j) e.export_page(0, j, nullptr, sbuf[size_t(j - (t - w))]).wait();
  std::printf("  export of %lld global parts to host memory: %.3f s, %.2f GB/s, %.2f ms a page\n", (long long)t, export_s, double(t) * pb.global / export_s / 1e9,
              1e3 * export_s / double(t));

  fs::remove_all(store_dir);
  serve::PrefixStore::Options so;
  so.dir = store_dir;
  std::random_device rd;
  for (auto& b : so.key) b = uint8_t(rd());
  so.max_queued = size_t(t + w + 8);
  serve::PrefixStore st(so);
  std::vector<serve::PageKey> keys;
  serve::extend_chain(serve::chain_root("prefix-engine-test"), keys, S, int64_t(S.size()) / attn::kPage);
  // The store seals in place: copies go to it, the exported buffers stay for the comparison.
  std::vector<std::vector<uint8_t>> gcopy(static_cast<size_t>(t)), scopy(static_cast<size_t>(w));
  for (int64_t j = 0; j < t; ++j) gcopy[size_t(j)].assign(gbuf[size_t(j)], gbuf[size_t(j)] + pb.global);
  for (int64_t j = 0; j < w; ++j) scopy[size_t(j)].assign(sbuf[size_t(j)], sbuf[size_t(j)] + pb.sliding);
  st.flush();
  t0 = now_s();
  for (int64_t j = 0; j < t; ++j) st.put(keys[size_t(j)], serve::Part::Global, j, gcopy[size_t(j)].data(), pb.global, {}, {});
  for (int64_t j = t - w; j < t; ++j) st.put(keys[size_t(j)], serve::Part::Sliding, j, scopy[size_t(j - (t - w))].data(), pb.sliding, {}, {});
  st.flush();
  const double write_s = now_s() - t0;
  const auto ss0 = st.stats();
  std::printf("  the store: %llu files, %.1f MiB, sealed and written in %.3f s (%.2f GB/s)\n", (unsigned long long)ss0.files, ss0.bytes / 1048576.0, write_s,
              ss0.bytes / write_s / 1e9);
  // Cold: each file written back and dropped from the page cache.
  for (const auto& d : fs::recursive_directory_iterator(store_dir)) {
    if (!d.is_regular_file()) continue;
    const int fd = ::open(d.path().c_str(), O_RDONLY);
    if (fd < 0) continue;
    ::fdatasync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
  }

  // The reference: slot 0 rewound to the point (within the slack) and prefilled from it.
  e.truncate(0, t * attn::kPage);
  const auto rewound = prefill(e, 0, S, t * attn::kPage);

  auto restore = [&](int slot, double& secs) {
    std::vector<uint8_t*> rb(8), rs(static_cast<size_t>(w));
    for (auto& b : rb) b = sycl::malloc_host<uint8_t>(pb.global, q);
    for (auto& b : rs) b = sycl::malloc_host<uint8_t>(pb.sliding, q);
    const double t1 = now_s();
    bool ok = true;
    for (int64_t j = t - w; j < t; ++j) ok = st.read(keys[size_t(j)], serve::Part::Sliding, rs[size_t(j - (t - w))], pb.sliding) && ok;
    e.adopt(slot, t);
    for (int64_t j0 = 0; j0 < t && ok; j0 += 8) {
      const int64_t n = std::min<int64_t>(8, t - j0);
      std::vector<std::future<bool>> f;
      for (int64_t i = 0; i < n; ++i)
        f.push_back(std::async(std::launch::async, [&, i] { return st.read(keys[size_t(j0 + i)], serve::Part::Global, rb[size_t(i)], pb.global); }));
      for (auto& x : f) ok = x.get() && ok;
      sycl::event last;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t j = j0 + i;
        last = e.import_page(slot, j, rb[size_t(i)], j >= t - w ? rs[size_t(j - (t - w))] : nullptr);
      }
      last.wait();
    }
    secs = now_s() - t1;
    for (auto* b : rb) sycl::free(b, q);
    for (auto* b : rs) sycl::free(b, q);
    return ok;
  };
  double cold_s = 0, warm_s = 0;
  const bool ok1 = restore(1, cold_s);
  const auto restored = prefill(e, 1, S, t * attn::kPage);
  const bool ok2 = restore(2, warm_s);
  const auto restored2 = prefill(e, 2, S, t * attn::kPage);
  std::printf("  restore of %lld positions, cold: %s in %.3f s, %.0f positions/s\n", (long long)(t * attn::kPage), ok1 ? "read" : "FAILED", cold_s,
              double(t * attn::kPage) / cold_s);
  std::printf("  restore again, warm page cache: %s in %.3f s, %.0f positions/s\n", ok2 ? "read" : "FAILED", warm_s, double(t * attn::kPage) / warm_s);
  bool same_bytes = true;
  {
    // The pages read back are what was exported.
    std::vector<uint8_t> b(pb.global);
    for (int64_t j = 0; j < t && same_bytes; j += 17) same_bytes = st.read(keys[size_t(j)], serve::Part::Global, b.data(), pb.global) && std::memcmp(b.data(), gbuf[size_t(j)], pb.global) == 0;
  }
  std::printf("  pages read back equal the exported bytes: %s\n", same_bytes ? "yes" : "NO");
  show("restored (cold) against rewound in place", diff(rewound, restored, V));
  show("restored (warm) against rewound in place", diff(rewound, restored2, V));
  show("rewound in place against the whole prefill", diff(full, rewound, V));
  for (auto* b : gbuf) sycl::free(b, q);
  for (auto* b : sbuf) sycl::free(b, q);
  fs::remove_all(store_dir);
  return 0;
}
