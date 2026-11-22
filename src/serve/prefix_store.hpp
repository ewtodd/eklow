// The on-disk prefix cache: the K/V of prompts and replies, page by page, so that a conversation whose slot went to
// another one comes back by reading its pages instead of prefilling them again (DECISIONS 2026-11-22).
//
// A page is attn::kPage (256) positions of one sequence. Its key names the whole prefix up to its end: a SHA-256 chain
// from a root (what makes two caches' K/V interchangeable: the model file, the cache's format, eklow's K/V revision)
// through each page's token ids, so equal keys mean equal prefixes. A page has two parts, each its own file:
//   Global   the global layers' K/V of the page (every page of a stored sequence has one);
//   Sliding  the sliding layers' K/V of the page, kept only for the pages a restore point needs (the rings' window).
// Files are AES-256-GCM sealed under one key with a random nonce each; the tag is the checksum, and the header (the key,
// the part, the page's depth, the length) is authenticated with it, so a file cannot be swapped for another page's. A
// part that is missing, truncated or fails its tag is a miss (and is removed), never an error.
//
// Writes are asynchronous: put() queues a part and a writer thread seals it, writes it to a temporary file and renames it
// into place (no fsync: a crash loses at most what a later read finds broken). Over the byte cap the least recently used
// parts go first; among parts last used in the same minute the deepest go first, and a page's sliding part before its
// global one, so a conversation's cache shrinks from its end rather than losing its start (a gap ends every restore past
// it).
#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace eklow::serve {

constexpr int64_t kPrefixPage = 256;  // positions a page: attn::kPage (scheduler_gemma4.cpp checks they agree)

struct PageKey {
  std::array<uint8_t, 32> b{};
  bool operator==(const PageKey&) const = default;
  std::string hex() const;
};
struct PageKeyHash {
  size_t operator()(const PageKey& k) const;
};

// The chain. root(fingerprint) starts it; next(prev, page) is the key of a page of kPrefixPage token ids that follows the
// prefix `prev` names. Image rows carry ids below zero that name the image (image_row_id), so they hash like tokens.
PageKey chain_root(std::string_view fingerprint);
PageKey chain_next(const PageKey& prev, std::span<const int32_t> page);
// Appends to `keys` (the chain of `tokens` so far) the keys of its full pages up to `pages`.
void extend_chain(const PageKey& root, std::vector<PageKey>& keys, std::span<const int32_t> tokens, int64_t pages);

enum class Part : uint8_t { Global = 'g', Sliding = 's' };

// AES-256-GCM over a buffer, in place: seal writes a fresh random nonce and the tag; open returns false when the tag does not
// match (the buffer then holds garbage). The aad is authenticated, not encrypted.
using AeadKey = std::array<uint8_t, 32>;
void seal(const AeadKey& key, std::span<const uint8_t> aad, uint8_t* buf, size_t n, uint8_t nonce[12], uint8_t tag[16]);
bool open_sealed(const AeadKey& key, std::span<const uint8_t> aad, uint8_t* buf, size_t n, const uint8_t nonce[12], const uint8_t tag[16]);
// The key file: 32 raw bytes, or 64 hex digits (whitespace around them ignored). Throws std::runtime_error otherwise.
AeadKey read_key_file(const std::filesystem::path& path);

// Where a restore can start: the largest t (pages) with t <= max_pages, every page below t stored (Global), and the
// sliding parts of pages [t - window_pages, t) stored. 0 when none.
struct RestoreQuery {
  int64_t max_pages = 0;     // the prompt's usable full pages less the draft's reach (see Scheduler::admit)
  int64_t window_pages = 0;  // pages of sliding K/V a restore needs below its point
};

class PrefixStore {
 public:
  struct Options {
    std::filesystem::path dir;
    uint64_t max_bytes = uint64_t(200) << 30;
    AeadKey key{};
    size_t max_queued = 64;     // parts waiting for the writer; a put past them is dropped (and counted)
    int64_t lru_bucket_s = 60;  // parts last used within the same bucket are evicted deepest first
    std::function<int64_t()> clock = {};  // seconds since the epoch (tests set it; null: the wall clock)
  };
  explicit PrefixStore(Options o);
  ~PrefixStore();  // writes what is queued, then stops
  PrefixStore(const PrefixStore&) = delete;
  PrefixStore& operator=(const PrefixStore&) = delete;

  bool has(const PageKey& k, Part p) const;
  // How many of keys' leading pages have a global part.
  int64_t matched(std::span<const PageKey> keys) const;
  // The restore point for a prompt's keys (0: none).
  int64_t restore_point(std::span<const PageKey> keys, const RestoreQuery& q) const;

  // Reads a part into dst, whose size must be the stored part's. False when it is missing or broken (a broken file is removed).
  // Marks the part used. Safe to call from several threads at once.
  bool read(const PageKey& k, Part p, uint8_t* dst, size_t n);

  // Queues a part. The writer calls `ready` before it reads `data` (the copy into it may still run), and `release` once it
  // no longer needs it, also when the part is dropped (already stored, queue full, the store closing). The writer seals the
  // buffer in place, so its contents are gone after. False when it was not queued (release has been called).
  bool put(const PageKey& k, Part p, int64_t depth, uint8_t* data, size_t n, std::function<void()> ready, std::function<void()> release);

  struct Stats {
    uint64_t bytes = 0, files = 0;        // on disk now
    uint64_t written = 0, written_bytes = 0, dropped = 0, deduplicated = 0;
    uint64_t read = 0, read_bytes = 0, broken = 0, evicted = 0, evicted_bytes = 0;
    bool scanned = false;  // the startup scan of the directory is done
    uint64_t queued = 0;   // parts waiting for the writer now, or being written
  };
  Stats stats() const;
  // Waits until the startup scan is done and every queued part is written (for tests and shutdown).
  void flush();

  static constexpr size_t kHeader = 84;

 private:
  struct Entry {
    bool g = false, s = false;
    uint64_t g_bytes = 0, s_bytes = 0;  // file sizes
    int64_t g_used = 0, s_used = 0;     // last use, seconds since the epoch
    int64_t depth = 0;
  };
  struct Job {
    PageKey key;
    Part part;
    int64_t depth;
    uint8_t* data;
    size_t n;
    std::function<void()> ready, release;
  };
  int64_t now() const;
  std::filesystem::path path_of(const PageKey& k, Part p) const;
  void header(uint8_t h[kHeader], const PageKey& k, Part p, int64_t depth, size_t n) const;
  void scan();
  void writer();
  void write(Job& j);
  void evict_locked();
  void forget_locked(const PageKey& k, Part p, bool remove_file);

  Options o_;
  mutable std::mutex mu_;
  std::condition_variable cv_, idle_cv_;
  std::unordered_map<PageKey, Entry, PageKeyHash> index_;
  std::deque<Job> queue_;
  bool stop_ = false, busy_ = false;
  Stats stats_;
  std::thread thread_;
};

}  // namespace eklow::serve
