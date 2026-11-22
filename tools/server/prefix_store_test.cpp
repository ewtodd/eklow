// eklow-prefix-store-test: the disk prefix cache (serve/prefix_store.hpp) and the scheduler's decisions about it
// (serve/prefix_policy.hpp), on the host, no card.
//
//   eklow-prefix-store-test [DIR]      DIR: where to put its files (default: a new directory under $TMPDIR, removed after)
//
// 1. The key chain: deterministic, rooted in the fingerprint, a changed token changes its page's key and every later one.
// 2. AES-256-GCM: a round trip, and a flipped bit of the data, the header, the tag or the key fails it. The key file's forms.
// 3. The store: parts written and read back whole, each buffer's ready before its use and release once, dedup, a full queue
//    drops, restore points, a tampered or truncated file is a miss and goes, a restart finds what is there.
// 4. Eviction: the least recently used first, and within a minute deepest first, sliding before global: a conversation
//    shrinks from its end.
// 5. The policy: the rings' rewind limit, restore limits and points that meet, and the restore-or-prefill choice.
// 6. Throughput, printed (not checked): sealing and opening a Bolt page, and reading pages back with 8 threads.
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "serve/prefix_policy.hpp"
#include "serve/prefix_store.hpp"

using namespace eklow::serve;
namespace fs = std::filesystem;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
  std::printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAIL");
  if (!ok) ++g_fail;
}

std::vector<int32_t> tokens(size_t n, uint32_t seed) {
  std::mt19937 g(seed);
  std::vector<int32_t> t(n);
  for (auto& x : t) x = int32_t(g() % 262144);
  return t;
}

std::vector<uint8_t> bytes(size_t n, uint32_t seed) {
  std::mt19937 g(seed);
  std::vector<uint8_t> b(n);
  for (auto& x : b) x = uint8_t(g());
  return b;
}

AeadKey key_of(uint8_t v) {
  AeadKey k;
  for (size_t i = 0; i < k.size(); ++i) k[i] = uint8_t(v + i);
  return k;
}

// A part for put(): its own copy of the payload, ready and release counted.
struct Payload {
  std::vector<uint8_t> data;
  std::atomic<int> ready{0}, released{0};
};

bool put(PrefixStore& s, const PageKey& k, Part p, int64_t depth, Payload& pl) {
  return s.put(k, p, depth, pl.data.data(), pl.data.size(), [&pl] { ++pl.ready; }, [&pl] { ++pl.released; });
}

void test_chain() {
  std::printf("chain\n");
  const auto t = tokens(4 * kPrefixPage, 1);
  const PageKey r = chain_root("model a"), r2 = chain_root("model b");
  std::vector<PageKey> a, b, c;
  extend_chain(r, a, t, 4);
  extend_chain(r, b, std::span<const int32_t>(t).first(2 * kPrefixPage), 2);
  extend_chain(r, b, t, 4);
  check(a.size() == 4 && a == b, "extending page by page gives the keys of all at once");
  check(!(r == r2), "the root depends on the fingerprint");
  extend_chain(r2, c, t, 4);
  check(!(a[0] == c[0]) && !(a[3] == c[3]), "another root, other keys for the same tokens");
  auto t2 = t;
  t2[kPrefixPage + 7] ^= 1;
  std::vector<PageKey> d;
  extend_chain(r, d, t2, 4);
  check(a[0] == d[0] && !(a[1] == d[1]) && !(a[2] == d[2]) && !(a[3] == d[3]), "a changed token changes its page's key and every later one");
  auto t3 = t;
  t3[5] = -1234567;  // an image row's id
  std::vector<PageKey> e;
  extend_chain(r, e, t3, 1);
  check(!(e[0] == a[0]), "image row ids (below zero) hash like tokens");
  check(a[0].hex().size() == 64 && a[0].hex().find_first_not_of("0123456789abcdef") == std::string::npos, "hex names are 64 lower-case digits");
  bool threw = false;
  try {
    extend_chain(r, e, std::span<const int32_t>(t).first(100), 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "a page needs 256 ids");
}

void test_aead(const fs::path& dir) {
  std::printf("AES-256-GCM\n");
  const AeadKey k = key_of(1), k2 = key_of(2);
  const auto plain = bytes(1 << 20, 2);
  const std::vector<uint8_t> aad = bytes(56, 3);
  auto buf = plain;
  uint8_t nonce[12], tag[16];
  seal(k, aad, buf.data(), buf.size(), nonce, tag);
  check(buf != plain, "sealing changes the data");
  auto c = buf;
  check(open_sealed(k, aad, c.data(), c.size(), nonce, tag) && c == plain, "a round trip");
  c = buf;
  c[777] ^= 0x10;
  check(!open_sealed(k, aad, c.data(), c.size(), nonce, tag), "a flipped bit of the data fails");
  auto aad2 = aad;
  aad2[20] ^= 1;
  c = buf;
  check(!open_sealed(k, aad2, c.data(), c.size(), nonce, tag), "a flipped bit of the header (aad) fails");
  uint8_t tag2[16];
  std::memcpy(tag2, tag, 16);
  tag2[0] ^= 1;
  c = buf;
  check(!open_sealed(k, aad, c.data(), c.size(), nonce, tag2), "a flipped bit of the tag fails");
  c = buf;
  check(!open_sealed(k2, aad, c.data(), c.size(), nonce, tag), "another key fails");
  uint8_t nonce2[12], tag3[16];
  auto again = plain;
  seal(k, aad, again.data(), again.size(), nonce2, tag3);
  check(std::memcmp(nonce, nonce2, 12) != 0 && again != buf, "each seal takes a fresh nonce");

  std::printf("key file\n");
  const fs::path raw = dir / "key.raw", hex = dir / "key.hex", bad = dir / "key.bad";
  {
    std::ofstream(raw, std::ios::binary).write(reinterpret_cast<const char*>(k.data()), 32);
    std::ofstream h(hex);
    for (uint8_t x : k2) h << "0123456789abcdef"[x >> 4] << "0123456789ABCDEF"[x & 15];
    h << "\n";
    std::ofstream(bad) << "0123";
  }
  check(read_key_file(raw) == k, "32 raw bytes");
  check(read_key_file(hex) == k2, "64 hex digits and a newline (either case)");
  bool threw = false;
  try {
    read_key_file(bad);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "anything else is refused");
  threw = false;
  try {
    read_key_file(dir / "missing");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "a missing file is refused");
}

void test_store(const fs::path& dir) {
  std::printf("store\n");
  const fs::path d = dir / "store";
  const auto t = tokens(6 * kPrefixPage, 4);
  std::vector<PageKey> keys;
  extend_chain(chain_root("fp"), keys, t, 6);
  constexpr size_t G = 4096, S = 10000;
  std::vector<std::unique_ptr<Payload>> g, s;
  for (int j = 0; j < 6; ++j) {
    g.push_back(std::make_unique<Payload>());
    g.back()->data = bytes(G, 100 + j);
    s.push_back(std::make_unique<Payload>());
    s.back()->data = bytes(S, 200 + j);
  }
  const auto g0 = g[0]->data, s3 = s[3]->data;
  {
    PrefixStore::Options o;
    o.dir = d;
    o.key = key_of(9);
    PrefixStore st(o);
    for (int j = 0; j < 4; ++j) put(st, keys[size_t(j)], Part::Global, j, *g[size_t(j)]);
    put(st, keys[2], Part::Sliding, 2, *s[2]);
    put(st, keys[3], Part::Sliding, 3, *s[3]);
    st.flush();
    bool once = true;
    for (int j = 0; j < 4; ++j) once = once && g[size_t(j)]->ready == 1 && g[size_t(j)]->released == 1;
    check(once, "each part's ready and release are called once");
    check(st.matched(keys) == 4 && st.has(keys[3], Part::Sliding) && !st.has(keys[1], Part::Sliding), "the parts are there");
    std::vector<uint8_t> out(G);
    check(st.read(keys[0], Part::Global, out.data(), G) && out == g0, "a global part reads back whole");
    std::vector<uint8_t> outs(S);
    check(st.read(keys[3], Part::Sliding, outs.data(), S) && outs == s3, "a sliding part reads back whole");
    check(!st.read(keys[4], Part::Global, out.data(), G), "a part never stored is a miss");
    // Dedup: a part already stored, or already queued.
    Payload again;
    again.data = bytes(G, 5);
    check(!put(st, keys[0], Part::Global, 0, again) && again.released == 1 && again.ready == 1, "a stored part is not queued again (released at once)");
    check(st.stats().deduplicated == 1, "and counted");

    std::printf("restore points\n");
    check(st.restore_point(keys, {4, 2}) == 4, "the deepest point whose window of sliding parts is stored: 4");
    check(st.restore_point(keys, {3, 2}) == 0, "below it, none (page 1's sliding part is missing)");
    check(st.restore_point(keys, {6, 2}) == 4, "a limit past the stored pages stops at them");
    check(st.restore_point(keys, {3, 0}) == 3, "without sliding layers, any stored page");
    check(st.restore_point(keys, {4, 1}) == 4 && st.restore_point(keys, {3, 1}) == 3, "a window of one page needs only the page before");

    std::printf("broken files\n");
    const fs::path f1 = d / keys[1].hex().substr(0, 2) / (keys[1].hex() + ".g");
    {
      std::fstream f(f1, std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(PrefixStore::kHeader + 100);
      char c = 0;
      f.read(&c, 1);
      f.seekp(PrefixStore::kHeader + 100);
      c ^= 0x40;
      f.write(&c, 1);
    }
    check(!st.read(keys[1], Part::Global, out.data(), G), "a flipped bit in a page is a miss");
    check(!fs::exists(f1) && !st.has(keys[1], Part::Global) && st.stats().broken == 1, "and the file goes");
    check(st.matched(keys) == 1 && st.restore_point(keys, {4, 2}) == 0, "a gap ends every restore past it");
    const fs::path f2 = d / keys[2].hex().substr(0, 2) / (keys[2].hex() + ".g");
    fs::resize_file(f2, fs::file_size(f2) - 10);
    check(!st.read(keys[2], Part::Global, out.data(), G) && !fs::exists(f2), "a truncated file is a miss, and goes");
    check(!st.read(keys[3], Part::Global, out.data(), G - 1), "a read of another length is a miss");
  }
  std::printf("restart\n");
  fs::create_directories(d / "zz");
  std::ofstream(d / "zz" / "not-a-key.g") << "junk";
  std::ofstream(d / "tmp" / "left-over") << "junk";
  // A file renamed to another key's name: its header names the old key.
  const fs::path f0 = d / keys[0].hex().substr(0, 2) / (keys[0].hex() + ".g");
  const fs::path f5 = d / keys[5].hex().substr(0, 2) / (keys[5].hex() + ".g");
  fs::create_directories(f5.parent_path());
  fs::copy_file(f0, f5);
  {
    PrefixStore::Options o;
    o.dir = d;
    o.key = key_of(9);
    PrefixStore st(o);
    st.flush();
    std::vector<uint8_t> out(G);
    check(st.has(keys[0], Part::Global) && st.read(keys[0], Part::Global, out.data(), G) && out == g0, "a restart finds the stored parts");
    check(st.has(keys[3], Part::Sliding) && !st.has(keys[1], Part::Global), "all of them, and only them");
    check(!fs::exists(d / "zz" / "not-a-key.g") && !fs::exists(d / "tmp" / "left-over"), "junk and left-over temporary files go");
    check(!st.has(keys[5], Part::Global) && !fs::exists(f5), "a file under another key's name goes");
    check(st.stats().files == 3, "the count is the files kept (pages 0 and 3, page 3's sliding part)");
  }
  {
    PrefixStore::Options o;
    o.dir = d;
    o.key = key_of(10);
    PrefixStore st(o);
    st.flush();
    std::vector<uint8_t> out(G);
    check(!st.read(keys[0], Part::Global, out.data(), G), "under another key the files do not open");
  }

  std::printf("queue\n");
  {
    PrefixStore::Options o;
    o.dir = dir / "queue";
    o.key = key_of(9);
    o.max_queued = 1;
    PrefixStore st(o);
    st.flush();
    std::promise<void> go;
    std::shared_future<void> gate = go.get_future().share();
    Payload a, b, c;
    a.data = bytes(G, 1), b.data = bytes(G, 2), c.data = bytes(G, 3);
    std::atomic<bool> in_ready{false};
    st.put(keys[0], Part::Global, 0, a.data.data(), G, [&] { in_ready = true; gate.wait(); }, [&] { ++a.released; });
    while (!in_ready) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const bool qb = put(st, keys[1], Part::Global, 1, b);
    const bool qc = put(st, keys[2], Part::Global, 2, c);
    check(qb && !qc && c.released == 1 && c.ready == 1, "past max_queued a part is dropped (and released)");
    go.set_value();
    st.flush();
    check(a.released == 1 && b.released == 1 && st.stats().dropped == 1 && st.matched(keys) == 2, "the writer goes on with the rest");
  }
}

void test_eviction(const fs::path& dir) {
  std::printf("eviction\n");
  std::atomic<int64_t> clock{1000};
  PrefixStore::Options o;
  o.dir = dir / "evict";
  o.key = key_of(3);
  constexpr size_t N = 1000;
  const uint64_t file = PrefixStore::kHeader + N;
  o.max_bytes = 9 * file;  // nine files; eviction goes down to 95% of it
  o.clock = [&] { return clock.load(); };
  PrefixStore st(o);
  std::vector<PageKey> a, b;
  extend_chain(chain_root("a"), a, tokens(4 * kPrefixPage, 7), 4);
  extend_chain(chain_root("b"), b, tokens(4 * kPrefixPage, 8), 4);
  std::vector<std::unique_ptr<Payload>> pl;
  auto add = [&](const PageKey& k, Part p, int64_t depth) {
    pl.push_back(std::make_unique<Payload>());
    pl.back()->data = bytes(N, uint32_t(pl.size()));
    put(st, k, p, depth, *pl.back());
    st.flush();
  };
  // Conversation a at t=1000 (pages 0-3, page 3's sliding part), b at t=5000 (the same): ten files.
  for (int j = 0; j < 4; ++j) add(a[size_t(j)], Part::Global, j);
  add(a[3], Part::Sliding, 3);
  clock = 5000;
  for (int j = 0; j < 4; ++j) add(b[size_t(j)], Part::Global, j);
  check(st.stats().files == 9 && st.stats().evicted == 0, "nine files fit");
  add(b[3], Part::Sliding, 3);
  // Over the cap: down to 95% of nine files, so two go: a's newest page's sliding part, then its global part.
  const auto s = st.stats();
  check(s.files == 8 && s.evicted == 2, "over the cap, files go until it is 95% full (2 of 10)");
  check(!st.has(a[3], Part::Sliding) && !st.has(a[3], Part::Global) && st.has(a[2], Part::Global), "the oldest conversation loses its end first, sliding part first");
  check(st.matched(b) == 4 && st.has(b[3], Part::Sliding), "the newer one is whole");
  // A read marks a part used: a's pages are read (as a restore reads them, first to last) at t=9000, then c comes.
  clock = 9000;
  std::vector<uint8_t> out(N);
  for (int j = 0; j < 3; ++j) st.read(a[size_t(j)], Part::Global, out.data(), N);
  std::vector<PageKey> c;
  extend_chain(chain_root("c"), c, tokens(2 * kPrefixPage, 9), 2);
  add(c[0], Part::Global, 0);
  add(c[1], Part::Global, 1);
  check(st.matched(a) == 3 && st.matched(c) == 2, "parts read since are kept");
  check(!st.has(b[3], Part::Sliding) && st.matched(b) < 4 && st.matched(b) >= 2, "the least recently used conversation (b) shrinks from its end");
}

void test_policy() {
  std::printf("policy\n");
  // Bolt: window 512, chunk 2048 -> rings of 10 pages, slack 2048.
  check(ring_safe_keep(5000, 6000, 2048) == 5000, "a rewind within the slack keeps");
  check(ring_safe_keep(5000, 7049, 2048) == 0, "past it the sequence starts over (not keep - window: that window's own window is gone too)");
  check(restore_limit(1000, 256, 4) == 0 && restore_limit(2000, 256, 4) == 3 && restore_limit(2049, 256, 0) == 8, "restore limits");
  const SlidingPages sp = point_pages(5000, 256, 4, 2);
  check(sp.from == 13 && sp.to == 15, "a sequence of 5,000 positions stores pages 13-14's sliding parts (point 15 = 19 - 4)");
  check(point_pages(1000, 256, 4, 2).to == 0, "a sequence shorter than the reach has no point");
  // Every later prompt that extends a stored sequence reaches its point.
  bool meets = true;
  for (int64_t held = 256; held < 20000; held += 37)
    for (int64_t reach : {0, 4})
      for (int64_t extra : {1, 2, 255, 256, 3000}) {
        const SlidingPages p = point_pages(held, 256, reach, 2);
        if (p.to > 0) meets = meets && restore_limit(held + extra, 256, reach) >= p.to;
      }
  check(meets, "a prompt extending a stored sequence reaches the sequence's restore point");
  // The positions a restore leaves to prefill cover the draft's reach (1,024 for Bolt's).
  bool reach_ok = true;
  for (int64_t P = 300; P < 20000; P += 41) {
    const int64_t t = restore_limit(P, 256, 4);
    if (t > 0) reach_ok = reach_ok && P - t * 256 >= 1024 + 1;
  }
  check(reach_ok, "after a restore at the limit, at least the draft's reach is prefilled");
  const RestoreCost c;
  check(prefer_restore(0, 20000, c), "20k positions from the disk beat prefilling them");
  check(!prefer_restore(19800, 20000, c), "but not 200 of them when the slot holds the rest");
  check(!prefer_restore(5000, 5000, c), "never when the slot holds as much");
}

void bench(const fs::path& dir) {
  std::printf("throughput (Bolt's page: 4 MiB of global K/V, 256 positions)\n");
  constexpr size_t G = 4 << 20;
  const AeadKey k = key_of(5);
  auto buf = bytes(G, 11);
  const std::vector<uint8_t> aad(56, 7);
  uint8_t nonce[12], tag[16];
  const int reps = 64;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) seal(k, aad, buf.data(), G, nonce, tag);
  const double seal_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  seal(k, aad, buf.data(), G, nonce, tag);
  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) {
    auto c = buf;
    open_sealed(k, aad, c.data(), G, nonce, tag);
  }
  const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  seal %.2f GB/s, open (with a copy) %.2f GB/s, one thread\n", double(reps) * G / seal_s / 1e9, double(reps) * G / open_s / 1e9);

  PrefixStore::Options o;
  o.dir = dir / "bench";
  o.key = k;
  o.max_queued = 256;
  PrefixStore st(o);
  constexpr int pages = 80;  // a 20k-token conversation
  std::vector<PageKey> keys;
  extend_chain(chain_root("bench"), keys, tokens(pages * kPrefixPage, 12), pages);
  std::vector<std::vector<uint8_t>> data(pages);
  for (int j = 0; j < pages; ++j) data[size_t(j)] = bytes(G, uint32_t(j));
  t0 = std::chrono::steady_clock::now();
  for (int j = 0; j < pages; ++j) st.put(keys[size_t(j)], Part::Global, j, data[size_t(j)].data(), G, {}, {});
  st.flush();
  const double write_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %d pages written (sealed, written, renamed) in %.2f s: %.2f GB/s\n", pages, write_s, double(pages) * G / write_s / 1e9);
  std::vector<std::vector<uint8_t>> out(8, std::vector<uint8_t>(G));
  t0 = std::chrono::steady_clock::now();
  bool all = true;
  for (int j0 = 0; j0 < pages; j0 += 8) {
    std::vector<std::future<bool>> f;
    for (int i = 0; i < 8; ++i) f.push_back(std::async(std::launch::async, [&, i] { return st.read(keys[size_t(j0 + i)], Part::Global, out[size_t(i)].data(), G); }));
    for (auto& x : f) all = x.get() && all;
  }
  const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  %d pages read back by 8 threads (page cache warm) in %.3f s: %.2f GB/s, %.0f positions/s\n", pages, read_s, double(pages) * G / read_s / 1e9,
              double(pages) * kPrefixPage / read_s);
  check(all, "every page read back");
}

}  // namespace

int main(int argc, char** argv) {
  fs::path dir;
  bool own = argc < 2;
  if (own) {
    const char* tmp = std::getenv("TMPDIR");
    dir = fs::path(tmp ? tmp : "/tmp") / ("eklow-prefix-store-test." + std::to_string(::getpid()));
  } else {
    dir = argv[1];
  }
  fs::create_directories(dir);
  test_chain();
  test_aead(dir);
  test_store(dir);
  test_eviction(dir);
  test_policy();
  bench(dir);
  if (own) fs::remove_all(dir);
  std::printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
  return g_fail ? 1 : 0;
}
