#include "serve/prefix_store.hpp"

#include <fcntl.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace eklow::serve {

namespace {

constexpr char kMagic[4] = {'E', 'K', 'P', 'C'};
constexpr uint32_t kVersion = 1;
constexpr size_t kAad = 56;  // the header up to the nonce: magic, version, part, depth, length, key



struct CipherCtx {
  EVP_CIPHER_CTX* c = EVP_CIPHER_CTX_new();
  ~CipherCtx() { EVP_CIPHER_CTX_free(c); }
};

void sha256(std::initializer_list<std::span<const uint8_t>> parts, uint8_t out[32]) {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> c(EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!c || EVP_DigestInit_ex(c.get(), EVP_sha256(), nullptr) != 1) throw std::runtime_error("prefix cache: SHA-256 is unavailable");
  for (const auto& p : parts) EVP_DigestUpdate(c.get(), p.data(), p.size());
  unsigned int n = 0;
  EVP_DigestFinal_ex(c.get(), out, &n);
}

void put_u32(uint8_t* p, uint32_t v) {
  for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i));
}
void put_u64(uint8_t* p, uint64_t v) {
  for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i));
}
uint32_t get_u32(const uint8_t* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i) v |= uint32_t(p[i]) << (8 * i);
  return v;
}
uint64_t get_u64(const uint8_t* p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; ++i) v |= uint64_t(p[i]) << (8 * i);
  return v;
}

int hex_digit(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

bool parse_hex(std::string_view s, uint8_t* out, size_t n) {
  if (s.size() != 2 * n) return false;
  for (size_t i = 0; i < n; ++i) {
    const int hi = hex_digit(s[2 * i]), lo = hex_digit(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = uint8_t(hi * 16 + lo);
  }
  return true;
}

bool write_all(int fd, const uint8_t* p, size_t n) {
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += w, n -= size_t(w);
  }
  return true;
}

bool pread_all(int fd, uint8_t* p, size_t n, off_t off) {
  while (n > 0) {
    const ssize_t r = ::pread(fd, p, n, off);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (r == 0) return false;
    p += r, n -= size_t(r), off += r;
  }
  return true;
}

// GCM's lengths are ints: a buffer goes through in pieces.
constexpr size_t kPiece = size_t(1) << 30;

}  // namespace

std::string PageKey::hex() const {
  static const char* d = "0123456789abcdef";
  std::string s(64, '0');
  for (size_t i = 0; i < 32; ++i) s[2 * i] = d[b[i] >> 4], s[2 * i + 1] = d[b[i] & 15];
  return s;
}

size_t PageKeyHash::operator()(const PageKey& k) const {
  size_t h;
  std::memcpy(&h, k.b.data(), sizeof h);
  return h;
}

PageKey chain_root(std::string_view fingerprint) {
  static constexpr std::string_view tag = "eklow prefix cache v1";
  PageKey k;
  sha256({{reinterpret_cast<const uint8_t*>(tag.data()), tag.size() + 1},  // with the terminating zero, so tag and fingerprint never run together
          {reinterpret_cast<const uint8_t*>(fingerprint.data()), fingerprint.size()}},
         k.b.data());
  return k;
}

PageKey chain_next(const PageKey& prev, std::span<const int32_t> page) {
  if (int64_t(page.size()) != kPrefixPage) throw std::invalid_argument("prefix cache: a page is 256 token ids");
  uint8_t ids[kPrefixPage * 4];
  for (size_t i = 0; i < page.size(); ++i) put_u32(ids + 4 * i, uint32_t(page[i]));
  PageKey k;
  sha256({{prev.b.data(), prev.b.size()}, {ids, sizeof ids}}, k.b.data());
  return k;
}

void extend_chain(const PageKey& root, std::vector<PageKey>& keys, std::span<const int32_t> tokens, int64_t pages) {
  if (pages * kPrefixPage > int64_t(tokens.size())) throw std::invalid_argument("prefix cache: fewer tokens than pages");
  while (int64_t(keys.size()) < pages) {
    const int64_t j = int64_t(keys.size());
    keys.push_back(chain_next(j == 0 ? root : keys.back(), tokens.subspan(size_t(j * kPrefixPage), size_t(kPrefixPage))));
  }
}

void seal(const AeadKey& key, std::span<const uint8_t> aad, uint8_t* buf, size_t n, uint8_t nonce[12], uint8_t tag[16]) {
  if (RAND_bytes(nonce, 12) != 1) throw std::runtime_error("prefix cache: no random bytes for a nonce");
  CipherCtx c;
  int len = 0;
  uint8_t none[16];
  if (!c.c || EVP_EncryptInit_ex(c.c, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
      EVP_CIPHER_CTX_ctrl(c.c, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) != 1 || EVP_EncryptInit_ex(c.c, nullptr, nullptr, key.data(), nonce) != 1 ||
      EVP_EncryptUpdate(c.c, nullptr, &len, aad.data(), int(aad.size())) != 1)
    throw std::runtime_error("prefix cache: AES-256-GCM is unavailable");
  for (size_t off = 0; off < n; off += kPiece) {
    const int m = int(std::min(kPiece, n - off));
    if (EVP_EncryptUpdate(c.c, buf + off, &len, buf + off, m) != 1 || len != m) throw std::runtime_error("prefix cache: encryption failed");
  }
  if (EVP_EncryptFinal_ex(c.c, none, &len) != 1 || EVP_CIPHER_CTX_ctrl(c.c, EVP_CTRL_GCM_GET_TAG, 16, tag) != 1)
    throw std::runtime_error("prefix cache: encryption failed");
}

bool open_sealed(const AeadKey& key, std::span<const uint8_t> aad, uint8_t* buf, size_t n, const uint8_t nonce[12], const uint8_t tag[16]) {
  CipherCtx c;
  int len = 0;
  uint8_t none[16];
  if (!c.c || EVP_DecryptInit_ex(c.c, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) != 1 ||
      EVP_CIPHER_CTX_ctrl(c.c, EVP_CTRL_GCM_SET_IVLEN, 12, nullptr) != 1 || EVP_DecryptInit_ex(c.c, nullptr, nullptr, key.data(), nonce) != 1 ||
      EVP_DecryptUpdate(c.c, nullptr, &len, aad.data(), int(aad.size())) != 1)
    return false;
  for (size_t off = 0; off < n; off += kPiece) {
    const int m = int(std::min(kPiece, n - off));
    if (EVP_DecryptUpdate(c.c, buf + off, &len, buf + off, m) != 1 || len != m) return false;
  }
  if (EVP_CIPHER_CTX_ctrl(c.c, EVP_CTRL_GCM_SET_TAG, 16, const_cast<uint8_t*>(tag)) != 1) return false;
  return EVP_DecryptFinal_ex(c.c, none, &len) > 0;
}

int64_t PrefixStore::now() const { return o_.clock ? o_.clock() : int64_t(std::time(nullptr)); }

AeadKey read_key_file(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("prefix cache: cannot read the key file " + path.string());
  const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  AeadKey k{};
  if (s.size() == k.size()) {
    std::memcpy(k.data(), s.data(), k.size());
    return k;
  }
  const size_t a = s.find_first_not_of(" \t\r\n"), z = s.find_last_not_of(" \t\r\n");
  if (a != std::string::npos && parse_hex(std::string_view(s).substr(a, z - a + 1), k.data(), k.size())) return k;
  throw std::runtime_error("prefix cache: " + path.string() + " holds neither 32 bytes nor 64 hex digits");
}

PrefixStore::PrefixStore(Options o) : o_(std::move(o)) {
  std::filesystem::create_directories(o_.dir / "tmp");
  thread_ = std::thread([this] { writer(); });
}

PrefixStore::~PrefixStore() {
  {
    std::lock_guard lk(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  thread_.join();
}

std::filesystem::path PrefixStore::path_of(const PageKey& k, Part p) const {
  const std::string h = k.hex();
  return o_.dir / h.substr(0, 2) / (h + (p == Part::Global ? ".g" : ".s"));
}

void PrefixStore::header(uint8_t h[kHeader], const PageKey& k, Part p, int64_t depth, size_t n) const {
  std::memset(h, 0, kHeader);
  std::memcpy(h, kMagic, 4);
  put_u32(h + 4, kVersion);
  h[8] = uint8_t(p);
  put_u32(h + 12, uint32_t(depth));
  put_u64(h + 16, uint64_t(n));
  std::memcpy(h + 24, k.b.data(), 32);
}

bool PrefixStore::has(const PageKey& k, Part p) const {
  std::lock_guard lk(mu_);
  const auto it = index_.find(k);
  return it != index_.end() && (p == Part::Global ? it->second.g : it->second.s);
}

int64_t PrefixStore::matched(std::span<const PageKey> keys) const {
  std::lock_guard lk(mu_);
  int64_t m = 0;
  for (const auto& k : keys) {
    const auto it = index_.find(k);
    if (it == index_.end() || !it->second.g) break;
    ++m;
  }
  return m;
}

int64_t PrefixStore::restore_point(std::span<const PageKey> keys, const RestoreQuery& q) const {
  std::lock_guard lk(mu_);
  const int64_t limit = std::min<int64_t>(q.max_pages, int64_t(keys.size()));
  std::vector<uint8_t> sliding;
  int64_t m = 0;
  for (; m < limit; ++m) {
    const auto it = index_.find(keys[size_t(m)]);
    if (it == index_.end() || !it->second.g) break;
    sliding.push_back(it->second.s ? 1 : 0);
  }
  for (int64_t t = m; t > 0; --t) {
    bool ok = true;
    for (int64_t j = std::max<int64_t>(0, t - q.window_pages); j < t && ok; ++j) ok = sliding[size_t(j)] != 0;
    if (ok) return t;
  }
  return 0;
}

void PrefixStore::forget_locked(const PageKey& k, Part p, bool remove_file) {
  auto it = index_.find(k);
  if (it == index_.end()) return;
  Entry& e = it->second;
  bool& present = p == Part::Global ? e.g : e.s;
  uint64_t& bytes = p == Part::Global ? e.g_bytes : e.s_bytes;
  if (!present) return;
  if (remove_file) ::unlink(path_of(k, p).c_str());
  present = false;
  stats_.bytes -= bytes, --stats_.files;
  bytes = 0;
  if (!e.g && !e.s) index_.erase(it);
}

bool PrefixStore::read(const PageKey& k, Part p, uint8_t* dst, size_t n) {
  {
    std::lock_guard lk(mu_);
    const auto it = index_.find(k);
    if (it == index_.end() || !(p == Part::Global ? it->second.g : it->second.s)) return false;
  }
  const std::filesystem::path path = path_of(k, p);
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  bool ok = fd >= 0;
  uint8_t h[kHeader], want[kHeader];
  if (ok) {
    // The header must name this key, part and length; the depth is the file's own (authenticated with the rest).
    struct stat st{};
    ok = ::fstat(fd, &st) == 0 && uint64_t(st.st_size) == kHeader + n && pread_all(fd, h, kHeader, 0);
    if (ok) header(want, k, p, int64_t(get_u32(h + 12)), n);
    ok = ok && std::memcmp(h, want, kAad) == 0 && pread_all(fd, dst, n, off_t(kHeader)) && open_sealed(o_.key, {h, kAad}, dst, n, h + 56, h + 68);
    ::close(fd);
  }
  std::lock_guard lk(mu_);
  if (!ok) {
    // A file that is gone was evicted under us; one that is there and does not open is broken.
    if (fd >= 0) ++stats_.broken;
    forget_locked(k, p, fd >= 0);
    return false;
  }
  const int64_t t = now();
  if (auto it = index_.find(k); it != index_.end()) (p == Part::Global ? it->second.g_used : it->second.s_used) = t;
  const struct timespec times[2] = {{t, 0}, {t, 0}};
  ::utimensat(AT_FDCWD, path.c_str(), times, 0);  // the startup scan reads last use from the mtime
  ++stats_.read;
  stats_.read_bytes += n;
  return true;
}

bool PrefixStore::put(const PageKey& k, Part p, int64_t depth, uint8_t* data, size_t n, std::function<void()> ready, std::function<void()> release) {
  bool queued = false;
  {
    std::lock_guard lk(mu_);
    const auto it = index_.find(k);
    const bool stored = it != index_.end() && (p == Part::Global ? it->second.g : it->second.s);
    const bool waiting = std::any_of(queue_.begin(), queue_.end(), [&](const Job& j) { return j.key == k && j.part == p; });
    if (stored || waiting) ++stats_.deduplicated;
    else if (stop_ || queue_.size() >= o_.max_queued) ++stats_.dropped;
    else {
      queue_.push_back({k, p, depth, data, n, std::move(ready), release});
      queued = true;
    }
  }
  if (!queued) {
    // The copy into the buffer may still run: the caller reuses it only once it has finished.
    try {
      if (ready) ready();
    } catch (const std::exception&) {
    }
    if (release) release();
    return false;
  }
  cv_.notify_all();
  return true;
}

PrefixStore::Stats PrefixStore::stats() const {
  std::lock_guard lk(mu_);
  return stats_;
}

void PrefixStore::flush() {
  std::unique_lock lk(mu_);
  idle_cv_.wait(lk, [&] { return stats_.scanned && queue_.empty() && !busy_; });
}

void PrefixStore::scan() {
  std::error_code ec;
  for (const auto& f : std::filesystem::directory_iterator(o_.dir / "tmp", ec)) std::filesystem::remove(f.path(), ec);
  for (const auto& d : std::filesystem::directory_iterator(o_.dir, ec)) {
    if (!d.is_directory() || d.path().filename() == "tmp") continue;
    std::error_code ec2;
    for (const auto& f : std::filesystem::directory_iterator(d.path(), ec2)) {
      const std::string name = f.path().filename().string();
      PageKey k;
      const bool named = name.size() == 66 && name[64] == '.' && (name[65] == 'g' || name[65] == 's') && parse_hex(std::string_view(name).substr(0, 64), k.b.data(), 32);
      const Part p = named && name[65] == 'g' ? Part::Global : Part::Sliding;
      struct stat st{};
      uint8_t h[kHeader];
      bool ok = named && ::stat(f.path().c_str(), &st) == 0 && S_ISREG(st.st_mode) && uint64_t(st.st_size) > kHeader;
      if (ok) {
        const int fd = ::open(f.path().c_str(), O_RDONLY | O_CLOEXEC);
        ok = fd >= 0 && pread_all(fd, h, kHeader, 0);
        if (fd >= 0) ::close(fd);
      }
      ok = ok && std::memcmp(h, kMagic, 4) == 0 && get_u32(h + 4) == kVersion && h[8] == uint8_t(p) && std::memcmp(h + 24, k.b.data(), 32) == 0 &&
           get_u64(h + 16) + kHeader == uint64_t(st.st_size);
      if (!ok) {
        std::filesystem::remove(f.path(), ec2);
        continue;
      }
      std::lock_guard lk(mu_);
      Entry& e = index_[k];
      bool& present = p == Part::Global ? e.g : e.s;
      if (present) continue;  // put() raced the scan to this part and wrote it
      present = true;
      (p == Part::Global ? e.g_bytes : e.s_bytes) = uint64_t(st.st_size);
      (p == Part::Global ? e.g_used : e.s_used) = int64_t(st.st_mtim.tv_sec);
      e.depth = int64_t(get_u32(h + 12));
      stats_.bytes += uint64_t(st.st_size), ++stats_.files;
    }
  }
  std::lock_guard lk(mu_);
  evict_locked();
  stats_.scanned = true;
}

void PrefixStore::write(Job& j) {
  bool ok = true;
  try {
    if (j.ready) j.ready();
  } catch (const std::exception& e) {  // the copy failed (the engine's step did): nothing to store
    std::fprintf(stderr, "eklow-server: prefix cache: a page's copy failed: %s\n", e.what());
    ok = false;
  }
  if (!ok) {
    if (j.release) j.release();
    std::lock_guard lk(mu_);
    ++stats_.dropped;
    return;
  }
  const std::filesystem::path final_path = path_of(j.key, j.part);
  static std::atomic<uint64_t> serial{0};
  const std::filesystem::path tmp = o_.dir / "tmp" / (j.key.hex() + "." + std::to_string(serial++));
  uint8_t h[kHeader];
  header(h, j.key, j.part, j.depth, j.n);
  try {
    seal(o_.key, {h, kAad}, j.data, j.n, h + 56, h + 68);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-server: prefix cache: %s\n", e.what());
    ok = false;
  }
  if (ok) {
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    ok = fd >= 0 && write_all(fd, h, kHeader) && write_all(fd, j.data, j.n);
    if (fd >= 0) ok = ::close(fd) == 0 && ok;
    std::error_code ec;
    std::filesystem::create_directories(final_path.parent_path(), ec);
    ok = ok && ::rename(tmp.c_str(), final_path.c_str()) == 0;
    if (!ok) {
      static std::atomic<bool> told{false};
      if (!told.exchange(true)) std::fprintf(stderr, "eklow-server: prefix cache: cannot write %s: %s\n", final_path.c_str(), std::strerror(errno));
      ::unlink(tmp.c_str());
    }
  }
  if (j.release) j.release();
  std::lock_guard lk(mu_);
  if (!ok) {
    ++stats_.dropped;
    return;
  }
  Entry& e = index_[j.key];
  bool& present = j.part == Part::Global ? e.g : e.s;
  const uint64_t bytes = kHeader + j.n;
  if (present) stats_.bytes -= (j.part == Part::Global ? e.g_bytes : e.s_bytes), --stats_.files;
  present = true;
  (j.part == Part::Global ? e.g_bytes : e.s_bytes) = bytes;
  (j.part == Part::Global ? e.g_used : e.s_used) = now();
  e.depth = j.depth;
  stats_.bytes += bytes, ++stats_.files;
  ++stats_.written;
  stats_.written_bytes += bytes;
  evict_locked();
}

void PrefixStore::evict_locked() {
  if (stats_.bytes <= o_.max_bytes) return;
  struct Candidate {
    int64_t bucket, depth;
    Part part;
    PageKey key;
  };
  std::vector<Candidate> c;
  c.reserve(index_.size() * 2);
  const int64_t b = std::max<int64_t>(1, o_.lru_bucket_s);
  for (const auto& [k, e] : index_) {
    if (e.s) c.push_back({e.s_used / b, e.depth, Part::Sliding, k});
    if (e.g) c.push_back({e.g_used / b, e.depth, Part::Global, k});
  }
  std::sort(c.begin(), c.end(), [](const Candidate& x, const Candidate& y) {
    if (x.bucket != y.bucket) return x.bucket < y.bucket;
    if (x.depth != y.depth) return x.depth > y.depth;
    return x.part == Part::Sliding && y.part == Part::Global;
  });
  const uint64_t target = o_.max_bytes - o_.max_bytes / 20;
  for (const Candidate& x : c) {
    if (stats_.bytes <= target) break;
    const auto it = index_.find(x.key);
    if (it == index_.end()) continue;
    const bool present = x.part == Part::Global ? it->second.g : it->second.s;
    if (!present) continue;
    const uint64_t before = stats_.bytes;
    // A global part's sliding part is no use without it.
    if (x.part == Part::Global) forget_locked(x.key, Part::Sliding, true);
    forget_locked(x.key, x.part, true);
    ++stats_.evicted;
    stats_.evicted_bytes += before - stats_.bytes;
  }
}

void PrefixStore::writer() {
  scan();
  idle_cv_.notify_all();
  std::unique_lock lk(mu_);
  for (;;) {
    cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
    if (queue_.empty()) break;  // stop_ with nothing left
    Job j = std::move(queue_.front());
    queue_.pop_front();
    busy_ = true;
    lk.unlock();
    write(j);
    lk.lock();
    busy_ = false;
    idle_cv_.notify_all();
  }
}

}  // namespace eklow::serve
