// VisionGemma4V's fast path (vision_gemma4v.hpp): xmx fp16 GEMMs, the dense attention of kernels/vision_attn.hpp,
// fused elementwise kernels, the layers' weights streamed from pinned host memory.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/vision_attn.hpp"
#include "kernels/xmx.hpp"
#include "model/vision_gemma4v_fast.hpp"
#include "rt/copy_engine.hpp"

namespace eklow::model {

namespace {

constexpr int kSg = 32;
constexpr int64_t kFfnRows = 2048;  // the FFN runs in chunks of this many rows: its intermediates stay small

int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

// A kernel with one sub-group per row: f(row, sub-group).
template <class F>
void rows_sg(sycl::queue& q, int64_t rows, F f) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(rows) * kSg), sycl::range<1>(kSg)),
                 [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] { f(int64_t(it.get_group(0)), it.get_sub_group()); });
}

uint64_t tile_bytes(xmx::WeightFormat f) { return f == xmx::WeightFormat::Q8_0 ? 256 : 512; }
uint64_t mat_tiles(int64_t k, int64_t n, xmx::WeightFormat f) { return uint64_t(n / 8) * uint64_t(k / 32) * tile_bytes(f); }
uint64_t mat_scales(int64_t k, int64_t n, xmx::WeightFormat f) { return f == xmx::WeightFormat::Q8_0 ? uint64_t(n / 8) * uint64_t(k / 32) * 16 : 0; }
uint64_t align256(uint64_t x) { return (x + 255) / 256 * 256; }

}  // namespace

VisionGemma4V::~VisionGemma4V() {
  q_.wait();
  for (void* p : owned_) sycl::free(p, q_);
  if (fast_) {
    free_activations();
    for (auto* s : fast_->stage)
      if (s) sycl::free(s, q_);
    for (auto& L : fast_->layers)
      if (L.blob) sycl::free(L.blob, q_);
    xmx::free_weight(q_, fast_->patch_w);
    xmx::free_weight(q_, fast_->proj_w);
  }
}

bool VisionGemma4V::load_fast(const gguf::File& f) {
  const int64_t E = n_embd_, n_layer = f.at("clip.vision.block_count").as_int();
  if (n_head_kv_ != n_head_ || (d_head_ != 64 && d_head_ != 72) || E != n_head_ * d_head_ || E % 128 || (3 * E) % 128) return false;
  auto is = [&](const std::string& n) { return f.tensor(n) != nullptr; };
  for (const auto& t : f.tensors()) {
    const std::string& n = t.name;
    if (n.rfind("v.", 0) != 0 && n.rfind("mm.input", 0) != 0) continue;
    if (n.find("input_max") != std::string::npos || n.find("output_max") != std::string::npos || n.find("input_min") != std::string::npos ||
        n.find("output_min") != std::string::npos || n.find(".bias") != std::string::npos || n.find(".ls1") != std::string::npos ||
        n.find(".ls2") != std::string::npos || n.find("out_scale") != std::string::npos)
      return false;
  }
  const gguf::Tensor* first = f.tensor("v.blk.0.attn_q.weight");
  if (!first) return false;
  const bool q8 = first->type == gguf::Type::Q8_0;
  auto type_ok = [&](const gguf::Tensor* t) {
    if (!t) return false;
    return q8 ? t->type == gguf::Type::Q8_0 : (t->type == gguf::Type::BF16 || t->type == gguf::Type::F16 || t->type == gguf::Type::F32);
  };
  const std::vector<std::string> mats{"attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down"};
  for (int64_t l = 0; l < n_layer; ++l) {
    const std::string b = "v.blk." + std::to_string(l) + ".";
    for (const auto& m : mats)
      if (!type_ok(f.tensor(b + m + ".weight"))) return false;
    for (const char* v : {"ln1", "ln2", "attn_q_norm", "attn_k_norm", "attn_post_norm", "ffn_post_norm"})
      if (!is(b + v + ".weight")) return false;
  }
  const gguf::Tensor *pe = f.tensor("v.patch_embd.weight"), *pt = f.tensor("v.position_embd.weight"), *pj = f.tensor("mm.input_projection.weight");
  if (!pe || !pt || !pj || !(q8 ? pj->type == gguf::Type::Q8_0 : type_ok(pj))) return false;
  if (pe->type != gguf::Type::F32 || pt->type != gguf::Type::F32 || pj->ne[1] % 128) return false;

  auto fast = std::make_unique<Fast>();
  Fast& F = *fast;
  F.fmt = q8 ? xmx::WeightFormat::Q8_0 : xmx::WeightFormat::F16;
  F.Dp = round_up(d_head_, 32);
  F.Fp = round_up(n_ff_, 128);
  F.Kp = round_up(n_ff_, 32);
  if (q8 && (n_ff_ % 32)) return false;
  // The blob: tiles then scales of qkv, o, gate|up, down, each part 256-byte aligned.
  const int64_t K[4] = {E, E, E, F.Kp}, N[4] = {3 * E, E, 2 * F.Fp, E};
  uint64_t off = 0;
  for (int i = 0; i < 4; ++i) {
    F.t_off[i] = off;
    off += align256(mat_tiles(K[i], N[i], F.fmt));
    F.s_off[i] = off;
    off += align256(mat_scales(K[i], N[i], F.fmt));
  }
  F.blob_bytes = off;

  auto upload_vec = [&](const std::string& name, int64_t len) {
    const gguf::Tensor* t = f.tensor(name);
    if (t->type != gguf::Type::F32 || int64_t(t->elements()) != len) throw std::runtime_error(f.path() + ": unexpected shape or type of " + name);
    auto* d = sycl::malloc_device<uint8_t>(t->nbytes, q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory loading " + name);
    owned_.push_back(d);
    q_.memcpy(d, t->data, t->nbytes).wait();
    bytes_ += t->nbytes;
    return reinterpret_cast<const float*>(d);
  };
  // A matrix of the parts' rows stacked (each part padded to rows_padded rows of zeros), k_total columns (zeros past
  // the file's), through the device into the xmx tiles, copied to the blob.
  struct Part {
    const gguf::Tensor* t;
    int64_t rows;
  };
  auto pack = [&](const std::vector<Part>& parts, int64_t k_total, uint8_t* blob, uint64_t t_off, uint64_t s_off) {
    const gguf::Type type = parts[0].t->type;
    int64_t n_total = 0;
    for (const auto& p : parts) n_total += p.rows;
    const uint64_t rb = type == gguf::Type::Q8_0 ? uint64_t(k_total / 32) * 34 : uint64_t(k_total) * (type == gguf::Type::F32 ? 4 : 2);
    std::vector<uint8_t> host(size_t(uint64_t(n_total) * rb), 0);
    uint64_t row0 = 0;
    for (const auto& p : parts) {
      const uint64_t src_rb = p.t->nbytes / p.t->ne[1];
      if (src_rb > rb) throw std::runtime_error(f.path() + ": " + p.t->name + " is wider than its padded shape");
      for (uint64_t r = 0; r < p.t->ne[1]; ++r) std::memcpy(host.data() + (row0 + r) * rb, p.t->data + r * src_rb, src_rb);
      row0 += uint64_t(p.rows);
    }
    auto* d = sycl::malloc_device<uint8_t>(host.size(), q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory repacking");
    q_.memcpy(d, host.data(), host.size()).wait();
    xmx::Weight w = xmx::repack(q_, ref::Weight{type, k_total, n_total, d, rb});
    q_.memcpy(blob + t_off, w.tiles, mat_tiles(k_total, n_total, F.fmt));
    if (w.scales) q_.memcpy(blob + s_off, w.scales, mat_scales(k_total, n_total, F.fmt));
    q_.wait();
    xmx::free_weight(q_, w);
    sycl::free(d, q_);
  };

  F.layers.resize(size_t(n_layer));
  try {
    for (int64_t l = 0; l < n_layer; ++l) {
      const std::string b = "v.blk." + std::to_string(l) + ".";
      auto T = [&](const char* m) { return f.tensor(b + m + ".weight"); };
      Fast::Layer& L = F.layers[size_t(l)];
      L.blob = static_cast<uint8_t*>(sycl::malloc_host(F.blob_bytes, q_));
      if (!L.blob) throw std::runtime_error("vision encoder: out of pinned host memory");
      std::memset(L.blob, 0, F.blob_bytes);
      pack({{T("attn_q"), E}, {T("attn_k"), E}, {T("attn_v"), E}}, E, L.blob, F.t_off[0], F.s_off[0]);
      pack({{T("attn_out"), E}}, E, L.blob, F.t_off[1], F.s_off[1]);
      pack({{T("ffn_gate"), F.Fp}, {T("ffn_up"), F.Fp}}, E, L.blob, F.t_off[2], F.s_off[2]);
      pack({{T("ffn_down"), E}}, F.Kp, L.blob, F.t_off[3], F.s_off[3]);
      L.ln1 = upload_vec(b + "ln1.weight", E);
      L.ln2 = upload_vec(b + "ln2.weight", E);
      L.q_norm = upload_vec(b + "attn_q_norm.weight", d_head_);
      L.k_norm = upload_vec(b + "attn_k_norm.weight", d_head_);
      L.attn_post = upload_vec(b + "attn_post_norm.weight", E);
      L.ffn_post = upload_vec(b + "ffn_post_norm.weight", E);
    }
    auto resident = [&](const gguf::Tensor* t) {
      auto* d = sycl::malloc_device<uint8_t>(t->nbytes, q_);
      if (!d) throw std::runtime_error("vision encoder: out of device memory loading " + t->name);
      q_.memcpy(d, t->data, t->nbytes).wait();
      xmx::Weight w = xmx::repack(q_, ref::Weight{t->type, int64_t(t->ne[0] * (t->ne.size() > 2 ? t->ne[1] * t->ne[2] : 1)),
                                                  int64_t(t->ne.size() > 2 ? t->ne[3] : t->ne[1]), d, t->nbytes / (t->ne.size() > 2 ? t->ne[3] : t->ne[1])});
      sycl::free(d, q_);
      bytes_ += mat_tiles(w.k, w.n, w.format) + mat_scales(w.k, w.n, w.format);
      return w;
    };
    F.patch_w = resident(pe);
    F.proj_w = resident(pj);
  } catch (...) {
    for (auto& L : F.layers)
      if (L.blob) sycl::free(L.blob, q_);
    throw;
  }
  n_out_ = F.proj_w.n;
  n_in_ = 3 * int64_t(patch_) * patch_;
  if (F.patch_w.k != n_in_ || F.patch_w.n != E || F.proj_w.k != E) throw std::runtime_error(f.path() + ": unexpected vision tensor shapes");
  if (f.tensor("v.std_bias") && f.tensor("v.std_scale")) {
    std_bias_ = upload_vec("v.std_bias", E);
    std_scale_ = upload_vec("v.std_scale", E);
  }
  n_pos_ = int64_t(pt->ne[1]);
  F.pos = reinterpret_cast<const float*>(pt->data);
  try {
    F.copy = std::make_unique<rt::CopyEngine>(q_);
  } catch (const std::exception&) {
    F.copy_q = std::make_unique<sycl::queue>(q_.get_context(), q_.get_device(), sycl::property::queue::in_order());
  }
  fast_ = std::move(fast);
  return true;
}

uint64_t VisionGemma4V::activation_bytes(int64_t n) const {
  const Fast& F = *fast_;
  const int64_t R = vattn::rows_padded(n), E = n_embd_, H = n_head_;
  const int64_t wide = std::max({R * 3 * E, kFfnRows * (2 * F.Fp + E), R * n_in_});
  return uint64_t(R * E + wide) * 4 + uint64_t(R * E + kFfnRows * F.Kp) * 2 + uint64_t(3 * H * R * F.Dp) * 2 + vattn::scratch_bytes(n, F.Dp, H);
}

uint64_t VisionGemma4V::card_bytes(int tokens) const {
  if (!fast_) return bytes_;
  return bytes_ + uint64_t(fast_->n_stage) * fast_->blob_bytes + activation_bytes(int64_t(tokens) * merge_ * merge_);
}

void VisionGemma4V::free_activations() {
  Fast& F = *fast_;
  q_.wait();
  for (void* p : {static_cast<void*>(F.x), static_cast<void*>(F.wide), static_cast<void*>(F.act_main), static_cast<void*>(F.act_ffn),
                  static_cast<void*>(F.qt), static_cast<void*>(F.kt), static_cast<void*>(F.vt)})
    if (p) sycl::free(p, q_);
  F.x = F.wide = nullptr;
  F.act_main = F.act_ffn = F.qt = F.kt = F.vt = nullptr;
  vattn::release(q_, F.att);
  F.n_cap = 0;
  held_ = 0;
}

void VisionGemma4V::release() {
  std::lock_guard lock(mu_);
  if (!fast_) return;
  free_activations();
  for (auto*& s : fast_->stage)
    if (s) sycl::free(s, q_), s = nullptr;
}

void VisionGemma4V::ensure(int64_t n) {
  Fast& F = *fast_;
  if (n <= F.n_cap && F.stage[0]) return;
  free_activations();
  if (const char* e = std::getenv("EKLOW_VISION_STAGES")) F.n_stage = std::clamp(std::atoi(e), 2, Fast::kStages);
  if (std::getenv("EKLOW_VISION_NOCE") && F.copy) {
    F.copy.reset();
    F.copy_q = std::make_unique<sycl::queue>(q_.get_context(), q_.get_device(), sycl::property::queue::in_order());
  }
  for (int i = 0; i < F.n_stage; ++i) {
    auto*& s = F.stage[i];
    if (s) continue;
    s = sycl::malloc_device<uint8_t>(F.blob_bytes, q_);
    if (!s) throw std::runtime_error("vision encoder: out of device memory for the weight staging");
  }
  const int64_t R = vattn::rows_padded(n), E = n_embd_, H = n_head_;
  F.wide_floats = std::max({R * 3 * E, kFfnRows * (2 * F.Fp + E), R * n_in_});
  auto dev = [&](auto*& p, int64_t count) {
    using T = std::remove_reference_t<decltype(*p)>;
    p = sycl::malloc_device<T>(size_t(count), q_);
    if (!p) {
      free_activations();
      throw std::runtime_error("vision encoder: out of device memory for activations");
    }
  };
  dev(F.x, R * E);
  dev(F.wide, F.wide_floats);
  dev(F.act_main, R * E);
  dev(F.act_ffn, kFfnRows * F.Kp);
  dev(F.qt, H * R * F.Dp);
  dev(F.kt, H * R * F.Dp);
  dev(F.vt, H * R * F.Dp);
  // The padding dimensions of the attention operands are never written: zero once.
  q_.memset(F.qt, 0, size_t(H * R * F.Dp) * 2);
  q_.memset(F.kt, 0, size_t(H * R * F.Dp) * 2);
  q_.memset(F.vt, 0, size_t(H * R * F.Dp) * 2);
  vattn::reserve(q_, F.att, n, F.Dp, H);
  q_.wait();
  F.n_cap = n;
  held_ = activation_bytes(n) + uint64_t(F.n_stage) * F.blob_bytes;
}

void VisionGemma4V::reserve(int tokens) {
  if (reference_) return;
  std::lock_guard lock(mu_);
  ensure(int64_t(tokens) * merge_ * merge_);
}

DeviceRows VisionGemma4V::encode_fast(const image::Patches& p) {
  std::lock_guard lock(mu_);
  Fast& F = *fast_;
  const int64_t cols = p.cols, rows = p.rows, n = cols * rows, E = n_embd_, H = n_head_, D = d_head_, Dp = F.Dp, Fp = F.Fp, Kp = F.Kp;
  const int64_t ox = cols / merge_, oy = rows / merge_, n_out = ox * oy, R = vattn::rows_padded(n);
  const int64_t n_layer = int64_t(F.layers.size());
  ensure(n);
  const sycl::context ctx = q_.get_context();
  // EKLOW_VISION_PROFILE=1: wait after each phase and print the sums (the phases then do not overlap).
  static const char* pe = std::getenv("EKLOW_VISION_PROFILE");
  static const bool profile = pe && pe[0] != 'h', host_prof = pe && pe[0] == 'h';
  const auto t_begin = std::chrono::steady_clock::now();
  std::map<std::string, double> prof;
  auto t_last = std::chrono::steady_clock::now();
  auto mark = [&](const char* name) {
    if (!profile) return;
    q_.wait();
    const auto t = std::chrono::steady_clock::now();
    prof[name] += std::chrono::duration<double, std::milli>(t - t_last).count();
    t_last = t;
  };
  auto tap = [&](const char* name, int il, const float* dev, int64_t r, int64_t d) {
    if (!tap_) return;
    q_.wait();
    tap_(il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name), dev, r, d);
  };

  // Streaming: layer l's blob goes to staging l % 2; the copy for l + 2 starts once layer l's kernels are done with it.
  std::vector<sycl::event> copied(static_cast<size_t>(n_layer)), done(static_cast<size_t>(n_layer));
  auto start_copy = [&](int64_t l) {
    static const bool nocopy = std::getenv("EKLOW_VISION_NOCOPY") != nullptr;  // diagnostic: weights stay as they were
    if (nocopy) {
      copied[size_t(l)] = q_.submit([&](sycl::handler& h) { h.single_task([] {}); });
      return;
    }
    uint8_t* dst = F.stage[l % F.n_stage];
    if (F.copy) {
      F.copy->copy(dst, F.layers[size_t(l)].blob, F.blob_bytes);
      copied[size_t(l)] = F.copy->event(F.copy->fence());
    } else {
      copied[size_t(l)] = F.copy_q->memcpy(dst, F.layers[size_t(l)].blob, F.blob_bytes);
    }
  };
  for (int64_t l = 0; l < std::min<int64_t>(F.n_stage, n_layer); ++l) start_copy(l);

  // Small per-image device data: the position rows (columns' table rows, then rows'), the RoPE angles.
  const int64_t rot = D / 2, half = rot / 2, maxpos = std::max(cols, rows);
  const ref::RopeTables rt = ref::rope_tables(maxpos, rot, double(rope_theta_), nullptr);
  float* small = sycl::malloc_device<float>(size_t((cols + rows) * E + 2 * maxpos * half), q_);
  if (!small) throw std::runtime_error("vision encoder: out of device memory");
  std::shared_ptr<float> small_guard(small, [ctx](float* x) { sycl::free(x, ctx); });
  float *posx = small, *posy = small + cols * E, *cos_t = posy + rows * E, *sin_t = cos_t + maxpos * half;
  q_.memcpy(posx, F.pos, size_t(cols * E) * 4);
  q_.memcpy(posy, F.pos + n_pos_ * E, size_t(rows * E) * 4);
  q_.memcpy(cos_t, rt.cos.data(), size_t(maxpos * half) * 4);
  q_.memcpy(sin_t, rt.sin.data(), size_t(maxpos * half) * 4);

  auto* out_raw = sycl::malloc_device<float>(size_t(n_out * n_out_), q_);
  if (!out_raw) throw std::runtime_error("vision encoder: out of device memory for the rows");
  std::shared_ptr<float> out(out_raw, [ctx](float* x) { sycl::free(x, ctx); });

  auto gemm = [&](sycl::half* a_ptr, int64_t m, int64_t k, const xmx::Weight& w, float* dst) {
    xmx::Activations a;
    a.h = a_ptr;
    a.capacity = int64_t(1) << 40;
    xmx::set_rows(a, m, k, true);
    xmx::matmul(q_, a, w, dst);
  };
  const float eps = eps_;
  float* x = F.x;
  sycl::half* am = F.act_main;

  // Patch embedding: 2 * pixels - 1 as fp16 GEMM input (the oracle's im2col rounds to fp16 too).
  {
    float* px = F.wide;
    q_.memcpy(px, p.data.data(), p.data.size() * 4);
    q_.parallel_for(sycl::range<2>(size_t(n), size_t(n_in_)), [=](sycl::id<2> id) {
      const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
      am[xmx::act16_index(r, c, 48)] = sycl::half(px[r * 768 + c] * 2.0f - 1.0f);
    });
  }
  gemm(am, n, n_in_, F.patch_w, x);
  mark("patch");
  tap("inp", -1, x, n, E);
  // x += pos_x[column] + pos_y[row]; the first layer's norm.
  {
    const float* w1 = F.layers[0].ln1;
    rows_sg(q_, n, [=](int64_t r, sycl::sub_group sg) {
      const int lane = int(sg.get_local_id()[0]);
      const int64_t col = r % cols, row = r / cols;
      float ss = 0;
      for (int64_t c = lane; c < E; c += kSg) {
        const float v = (x[r * E + c] + posx[col * E + c]) + posy[row * E + c];
        x[r * E + c] = v;
        ss += v * v;
      }
      ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
      const float inv = 1.0f / sycl::sqrt(ss / float(E) + eps);
      for (int64_t c = lane; c < E; c += kSg) am[xmx::act16_index(r, c, E / 16)] = sycl::half(x[r * E + c] * inv * w1[c]);
    });
  }
  tap("pos_embd", -1, x, n, E);

  for (int64_t l = 0; l < n_layer; ++l) {
    const Fast::Layer& L = F.layers[size_t(l)];
    q_.submit([&](sycl::handler& h) {
      h.depends_on(copied[size_t(l)]);
      h.single_task([] {});
    });
    const uint8_t* st = F.stage[l % F.n_stage];
    auto W = [&](int i, int64_t k, int64_t nn) {
      xmx::Weight w;
      w.k = k, w.n = nn, w.format = F.fmt;
      w.tiles = const_cast<uint8_t*>(st) + F.t_off[i];
      w.scales = F.fmt == xmx::WeightFormat::Q8_0 ? reinterpret_cast<sycl::half*>(const_cast<uint8_t*>(st) + F.s_off[i]) : nullptr;
      return w;
    };
    float* wide = F.wide;
    // Attention. The prep kernel: per (patch, head) the q, k, v rows normed, q and k turned by the 2D RoPE, written as
    // dpas tiles (zero padding past the head size is already there).
    mark("wait_copy");
    gemm(am, n, E, W(0, E, 3 * E), wide);
    mark("gemm_qkv");
    {
      sycl::half *qt = F.qt, *kt = F.kt, *vt = F.vt;
      const float *qn = L.q_norm, *kn = L.k_norm;
      const int64_t E3 = 3 * E;
      q_.parallel_for(sycl::range<3>(3, size_t(H), size_t(n)), [=](sycl::id<3> id) {
        const int which = int(id[0]);  // 0 q, 1 k, 2 v: three work-items a (patch, head), the patches fastest
        const int64_t h = int64_t(id[1]), t = int64_t(id[2]);
        const float* src = wide + t * E3 + which * E + h * D;
        float x[80];
        float ss = 0;
        for (int64_t d = 0; d < D; ++d) x[d] = src[d], ss += x[d] * x[d];
        const float inv = 1.0f / sycl::sqrt(ss / float(D) + eps);
        const float* w = which == 0 ? qn : kn;
        for (int64_t d = 0; d < D; ++d) x[d] = which == 2 ? x[d] * inv : x[d] * inv * w[d];
        // dims [0, D/2) turn by the column, [D/2, D) by the row; NeoX pairs (i, i + D/4) within each half
        if (which < 2) {
          const int64_t col = t % cols, row = t / cols;
          for (int part = 0; part < 2; ++part) {
            const int64_t pos = part ? row : col, o = part * rot;
            for (int64_t i = 0; i < half; ++i) {
              const float c = cos_t[pos * half + i], s = sin_t[pos * half + i];
              const float a = x[o + i], b = x[o + i + half];
              x[o + i] = a * c - b * s, x[o + i + half] = a * s + b * c;
            }
          }
        }
        sycl::half* dst = (which == 0 ? qt : which == 1 ? kt : vt) + h * R * Dp;
        for (int64_t d = 0; d < D; ++d)
          dst[which == 0 ? vattn::q_index(t, d, Dp) : which == 1 ? vattn::k_index(t, d, Dp) : vattn::v_index(t, d, Dp)] = sycl::half(x[d]);
      });
    }
    mark("prep");
    vattn::attend(q_, F.qt, F.kt, F.vt, n, H, Dp, D, F.att, am, E);
    mark("attention");
    gemm(am, n, E, W(1, E, E), wide);
    mark("gemm_o");
    tap("attn_out", int(l), wide, n, E);
    // x += post_norm(o); the FFN's norm.
    {
      const float *wp = L.attn_post, *w2 = L.ln2;
      rows_sg(q_, n, [=](int64_t r, sycl::sub_group sg) {
        const int lane = int(sg.get_local_id()[0]);
        float s1 = 0;
        for (int64_t c = lane; c < E; c += kSg) s1 += wide[r * E + c] * wide[r * E + c];
        s1 = sycl::reduce_over_group(sg, s1, sycl::plus<float>());
        const float i1 = 1.0f / sycl::sqrt(s1 / float(E) + eps);
        float s2 = 0;
        for (int64_t c = lane; c < E; c += kSg) {
          const float v = x[r * E + c] + wide[r * E + c] * i1 * wp[c];
          x[r * E + c] = v;
          s2 += v * v;
        }
        s2 = sycl::reduce_over_group(sg, s2, sycl::plus<float>());
        const float i2 = 1.0f / sycl::sqrt(s2 / float(E) + eps);
        for (int64_t c = lane; c < E; c += kSg) am[xmx::act16_index(r, c, E / 16)] = sycl::half(x[r * E + c] * i2 * w2[c]);
      });
    }
    mark("post1");
    // FFN in chunks of rows.
    for (int64_t c0 = 0; c0 < n; c0 += kFfnRows) {
      const int64_t m = std::min(kFfnRows, n - c0);
      sycl::half* a_chunk = am + (c0 / 8) * (E / 16) * 128;
      float* gu = wide;
      float* dn = wide + kFfnRows * 2 * Fp;
      gemm(a_chunk, m, E, W(2, E, 2 * Fp), gu);
      mark("gemm_gu");
      {
        sycl::half* af = F.act_ffn;
        q_.parallel_for(sycl::range<2>(size_t(m), size_t(Kp)), [=](sycl::id<2> id) {
          const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
          const float g = gu[r * 2 * Fp + c], u = gu[r * 2 * Fp + Fp + c];
          af[xmx::act16_index(r, c, Kp / 16)] = sycl::half(g * (1.0f / (1.0f + sycl::exp(-1.702f * g))) * u);
        });
      }
      mark("gelu");
      gemm(F.act_ffn, m, Kp, W(3, Kp, E), dn);
      mark("gemm_down");
      const float* wp = L.ffn_post;
      const float* wn = l + 1 < n_layer ? F.layers[size_t(l) + 1].ln1 : nullptr;
      float* xc = x + c0 * E;
      rows_sg(q_, m, [=](int64_t r, sycl::sub_group sg) {
        const int lane = int(sg.get_local_id()[0]);
        float s1 = 0;
        for (int64_t c = lane; c < E; c += kSg) s1 += dn[r * E + c] * dn[r * E + c];
        s1 = sycl::reduce_over_group(sg, s1, sycl::plus<float>());
        const float i1 = 1.0f / sycl::sqrt(s1 / float(E) + eps);
        float s2 = 0;
        for (int64_t c = lane; c < E; c += kSg) {
          const float v = xc[r * E + c] + dn[r * E + c] * i1 * wp[c];
          xc[r * E + c] = v;
          s2 += v * v;
        }
        if (!wn) return;
        s2 = sycl::reduce_over_group(sg, s2, sycl::plus<float>());
        const float i2 = 1.0f / sycl::sqrt(s2 / float(E) + eps);
        sycl::half* ac = am + (c0 / 8) * (E / 16) * 128;
        for (int64_t c = lane; c < E; c += kSg) ac[xmx::act16_index(r, c, E / 16)] = sycl::half(xc[r * E + c] * i2 * wn[c]);
      });
    }
    mark("post2");
    tap("layer_out", int(l), x, n, E);
    // Stream: layer l + 1 goes into the staging layer l - 1 used, once the card is done with that layer; layer l is queued,
    // so the card does not idle while the host waits.
    done[size_t(l)] = q_.submit([&](sycl::handler& h) { h.single_task([] {}); });
    if (l >= 1 && l + F.n_stage - 1 < n_layer) {
      done[size_t(l) - 1].wait();
      start_copy(l + F.n_stage - 1);
    }
  }

  // Gemma4VisionPooler, std scaling, the embedder's norm: one sub-group per output row.
  {
    const int k = merge_;
    const float s = std::sqrt(float(E)), *sb = std_bias_, *ss = std_scale_;
    rows_sg(q_, n_out, [=](int64_t o, sycl::sub_group sg) {
      const int lane = int(sg.get_local_id()[0]);
      const int64_t gx = o % ox, gy = o / ox;
      float tot = 0;
      for (int64_t c = lane; c < E; c += kSg) {
        float acc = 0;
        for (int iy = 0; iy < k; ++iy)
          for (int ix = 0; ix < k; ++ix) acc += x[((gy * k + iy) * cols + gx * k + ix) * E + c];
        float v = acc / float(k * k) * s;
        if (sb) v = (v - sb[c]) * ss[c];
        tot += v * v;
      }
      tot = sycl::reduce_over_group(sg, tot, sycl::plus<float>());
      const float inv = 1.0f / sycl::sqrt(tot / float(E) + eps);
      for (int64_t c = lane; c < E; c += kSg) {
        float acc = 0;
        for (int iy = 0; iy < k; ++iy)
          for (int ix = 0; ix < k; ++ix) acc += x[((gy * k + iy) * cols + gx * k + ix) * E + c];
        float v = acc / float(k * k) * s;
        if (sb) v = (v - sb[c]) * ss[c];
        am[xmx::act16_index(o, c, E / 16)] = sycl::half(v * inv);
      }
    });
  }
  mark("pool");
  gemm(am, n_out, E, F.proj_w, out.get());
  mark("proj");
  if (profile) {
    for (const auto& [k, v] : prof) std::fprintf(stderr, "  %-10s %8.2f ms\n", k.c_str(), v);
  }
  tap("projected", -1, out.get(), n_out, n_out_);
  const auto t_enq = std::chrono::steady_clock::now();
  q_.wait();
  if (host_prof) {
    const auto t_end = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  host: enqueued in %.1f ms, done at %.1f ms\n", std::chrono::duration<double, std::milli>(t_enq - t_begin).count(), std::chrono::duration<double, std::milli>(t_end - t_begin).count());
  }
  return {out, n_out, n_out_};
}

}  // namespace eklow::model
