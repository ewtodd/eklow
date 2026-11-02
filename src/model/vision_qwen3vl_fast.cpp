// VisionQwen3VL's fast path (vision_qwen3vl.hpp): xmx fp16 GEMMs, the dense attention of kernels/vision_attn.hpp, fused
// elementwise kernels, the weights streamed from pinned host memory one unit (a layer, or a third of the merger) at a time
// or, as an option, all resident on the card. Structure as vision_gemma4v_fast.cpp (DECISIONS 2026-10-27).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

#include "model/vision_qwen3vl_fast.hpp"

namespace eklow::model {

namespace {

constexpr int kSg = 32;
constexpr int kPerLane = 40;     // a row of up to 1280 held in a sub-group's registers
constexpr int64_t kRows = 2048;  // GEMMs run in chunks of at most this many rows: their fp32 outputs stay small

int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }
uint64_t align256(uint64_t x) { return (x + 255) / 256 * 256; }
uint64_t mat_tiles(int64_t k, int64_t n) { return uint64_t(n / 8) * uint64_t(k / 32) * 512; }

template <class F>
void rows_sg(sycl::queue& q, int64_t rows, F f) {
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(rows) * kSg), sycl::range<1>(kSg)),
                 [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] { f(int64_t(it.get_group(0)), it.get_sub_group()); });
}

// The element types a vision weight may be stored in, to float on the host.
float load_elem(gguf::Type t, const uint8_t* p, uint64_t i) {
  if (t == gguf::Type::F32) return reinterpret_cast<const float*>(p)[i];
  if (t == gguf::Type::BF16) {
    uint32_t u = uint32_t(reinterpret_cast<const uint16_t*>(p)[i]) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  }
  uint16_t h = reinterpret_cast<const uint16_t*>(p)[i];
  return float(sycl::bit_cast<sycl::half>(h));
}

}  // namespace

VisionQwen3VL::~VisionQwen3VL() {
  q_.wait();
  for (void* p : owned_) sycl::free(p, q_);
  if (fast_) {
    free_activations();
    for (auto* s : fast_->stage)
      if (s) sycl::free(s, q_);
    for (auto& u : fast_->units)
      if (u.blob) sycl::free(u.blob, q_);
    xmx::free_weight(q_, fast_->patch_w);
  }
}

bool VisionQwen3VL::load_fast(const gguf::File& f) {
  const int64_t E = n_embd_, n_layer = f.at("clip.vision.block_count").as_int();
  if ((d_head_ != 64 && d_head_ != 72) || E % 128 || (3 * E) % 128 || E > kPerLane * kSg || n_hidden_ != 4 * E || n_hidden_ % 256 || n_out_ % 128) return false;
  auto ok_type = [&](const gguf::Tensor* t) {
    return t && (t->type == gguf::Type::BF16 || t->type == gguf::Type::F16 || t->type == gguf::Type::F32);
  };
  const std::vector<std::string> mats{"attn_qkv", "attn_out", "ffn_up", "ffn_down"};
  for (int64_t l = 0; l < n_layer; ++l) {
    const std::string b = "v.blk." + std::to_string(l) + ".";
    for (const auto& m : mats)
      if (!ok_type(f.tensor(b + m + ".weight"))) return false;
  }
  if (!ok_type(f.tensor("mm.0.weight")) || !ok_type(f.tensor("mm.2.weight")) || !ok_type(f.tensor("v.patch_embd.weight")) ||
      !ok_type(f.tensor("v.patch_embd.weight.1")))
    return false;
  const gguf::Tensor* pt = f.tensor("v.position_embd.weight");

  auto fast = std::make_unique<Fast>();
  Fast& F = *fast;
  F.Dp = round_up(d_head_, 32);
  F.Fp = round_up(n_ff_, 128);
  F.Kp = round_up(n_ff_, 32);
  const int64_t K[4] = {E, E, E, F.Kp}, N[4] = {3 * E, E, F.Fp, E};
  uint64_t off = 0;
  for (int i = 0; i < 4; ++i) {
    F.t_off[i] = off;
    off += align256(mat_tiles(K[i], N[i]));
  }
  F.blob_bytes = off;
  // The merger's pieces, mm0's halves of n_hidden / 2 outputs and mm2, each in part 0 of a unit.
  const int64_t half_hidden = n_hidden_ / 2;
  F.blob_bytes = std::max({F.blob_bytes, align256(mat_tiles(n_hidden_, half_hidden)), align256(mat_tiles(n_hidden_, n_out_))});

  auto upload_vec = [&](const std::string& name, int64_t len, int64_t padded) {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    if (t->type != gguf::Type::F32 || int64_t(t->elements()) != len) throw std::runtime_error(f.path() + ": unexpected shape or type of " + name);
    std::vector<float> h(size_t(padded), 0.0f);
    std::memcpy(h.data(), t->data, t->nbytes);
    auto* d = sycl::malloc_device<float>(size_t(padded), q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory loading " + name);
    owned_.push_back(d);
    q_.memcpy(d, h.data(), size_t(padded) * 4).wait();
    bytes_ += uint64_t(padded) * 4;
    return static_cast<const float*>(d);
  };
  struct Part {
    const gguf::Tensor* t;
    int64_t row0, rows, rows_padded;  // rows [row0, row0 + rows) of the file's matrix, padded with zero rows
  };
  // A matrix of the parts' rows stacked, k_total columns (zeros past the file's), through the device into xmx tiles.
  auto pack = [&](const std::vector<Part>& parts, int64_t k_total, uint8_t* blob, uint64_t t_off) {
    int64_t n_total = 0;
    for (const auto& p : parts) n_total += p.rows_padded;
    std::vector<uint16_t> host(size_t(n_total) * size_t(k_total), 0);
    int64_t row0 = 0;
    for (const auto& p : parts) {
      const int64_t src_k = int64_t(p.t->ne[0]);
      if (src_k > k_total) throw std::runtime_error(f.path() + ": " + p.t->name + " is wider than its padded shape");
      for (int64_t r = 0; r < p.rows; ++r)
        for (int64_t c = 0; c < src_k; ++c) {
          const float v = load_elem(p.t->type, p.t->data, uint64_t(p.row0 + r) * uint64_t(src_k) + uint64_t(c));
          host[size_t(row0 + r) * size_t(k_total) + size_t(c)] = sycl::bit_cast<uint16_t>(sycl::half(v));
        }
      row0 += p.rows_padded;
    }
    auto* d = sycl::malloc_device<uint8_t>(host.size() * 2, q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory repacking");
    q_.memcpy(d, host.data(), host.size() * 2).wait();
    xmx::Weight w = xmx::repack(q_, ref::Weight{gguf::Type::F16, k_total, n_total, d, uint64_t(k_total) * 2});
    q_.memcpy(blob + t_off, w.tiles, mat_tiles(k_total, n_total));
    q_.wait();
    xmx::free_weight(q_, w);
    sycl::free(d, q_);
  };

  const int64_t n_units = n_layer + 3;
  F.units.resize(size_t(n_units));
  F.vecs.resize(size_t(n_layer));
  try {
    for (int64_t u = 0; u < n_units; ++u) {
      auto& U = F.units[size_t(u)];
      if (resident_) {
        U.blob = sycl::malloc_device<uint8_t>(F.blob_bytes, q_);
        if (U.blob) q_.memset(U.blob, 0, F.blob_bytes).wait();
      } else {
        U.blob = static_cast<uint8_t*>(sycl::malloc_host(F.blob_bytes, q_));
        if (U.blob) std::memset(U.blob, 0, F.blob_bytes);
      }
      if (!U.blob) throw std::runtime_error(resident_ ? "vision encoder: out of device memory for the weights" : "vision encoder: out of pinned host memory");
      if (resident_) bytes_ += F.blob_bytes;
      if (u < n_layer) {
        const std::string b = "v.blk." + std::to_string(u) + ".";
        auto T = [&](const char* m) { return f.tensor(b + m + ".weight"); };
        pack({{T("attn_qkv"), 0, 3 * E, 3 * E}}, E, U.blob, F.t_off[0]);
        pack({{T("attn_out"), 0, E, E}}, E, U.blob, F.t_off[1]);
        pack({{T("ffn_up"), 0, n_ff_, F.Fp}}, E, U.blob, F.t_off[2]);
        pack({{T("ffn_down"), 0, E, E}}, F.Kp, U.blob, F.t_off[3]);
        Fast::LayerVecs& V = F.vecs[size_t(u)];
        V.ln1_w = upload_vec(b + "ln1.weight", E, E);
        V.ln1_b = upload_vec(b + "ln1.bias", E, E);
        V.ln2_w = upload_vec(b + "ln2.weight", E, E);
        V.ln2_b = upload_vec(b + "ln2.bias", E, E);
        V.qkv_b = upload_vec(b + "attn_qkv.bias", 3 * E, 3 * E);
        V.o_b = upload_vec(b + "attn_out.bias", E, E);
        V.up_b = upload_vec(b + "ffn_up.bias", n_ff_, F.Fp);
        V.down_b = upload_vec(b + "ffn_down.bias", E, E);
      } else if (u == n_layer) {
        pack({{f.tensor("mm.0.weight"), 0, half_hidden, half_hidden}}, 4 * E, U.blob, 0);
      } else if (u == n_layer + 1) {
        pack({{f.tensor("mm.0.weight"), half_hidden, half_hidden, half_hidden}}, 4 * E, U.blob, 0);
      } else {
        pack({{f.tensor("mm.2.weight"), 0, n_out_, n_out_}}, n_hidden_, U.blob, 0);
      }
    }
    // The patch embedding: both conv kernels see the same picture, so one weight, their sum (in fp32).
    {
      const gguf::Tensor *p0 = f.tensor("v.patch_embd.weight"), *p1 = f.tensor("v.patch_embd.weight.1");
      n_in_ = int64_t(3) * patch_ * patch_;
      if (int64_t(p0->elements()) != n_in_ * E || int64_t(p1->elements()) != n_in_ * E || n_in_ % 32) throw std::runtime_error(f.path() + ": unexpected patch embedding shape");
      std::vector<float> sum(size_t(n_in_ * E));
      for (size_t i = 0; i < sum.size(); ++i) sum[i] = load_elem(p0->type, p0->data, i) + load_elem(p1->type, p1->data, i);
      auto* d = sycl::malloc_device<uint8_t>(sum.size() * 4, q_);
      if (!d) throw std::runtime_error("vision encoder: out of device memory loading the patch embedding");
      q_.memcpy(d, sum.data(), sum.size() * 4).wait();
      F.patch_w = xmx::repack(q_, ref::Weight{gguf::Type::F32, n_in_, E, d, uint64_t(n_in_) * 4});
      sycl::free(d, q_);
      bytes_ += mat_tiles(n_in_, E);
    }
    F.patch_b = upload_vec("v.patch_embd.bias", E, E);
    F.post_w = upload_vec("v.post_ln.weight", E, E);
    F.post_b = upload_vec("v.post_ln.bias", E, E);
    F.mm0_b = upload_vec("mm.0.bias", n_hidden_, n_hidden_);
    F.mm2_b = upload_vec("mm.2.bias", n_out_, n_out_);
  } catch (...) {
    for (auto& u : F.units)
      if (u.blob) sycl::free(u.blob, q_);
    throw;
  }
  F.pos = reinterpret_cast<const float*>(pt->data);
  if (!resident_) {
    try {
      F.copy = std::make_unique<rt::CopyEngine>(q_);
    } catch (const std::exception&) {
      F.copy_q = std::make_unique<sycl::queue>(q_.get_context(), q_.get_device(), sycl::property::queue::in_order());
    }
  }
  fast_ = std::move(fast);
  return true;
}

uint64_t VisionQwen3VL::activation_bytes(int64_t n) const {
  const Fast& F = *fast_;
  const int64_t R = vattn::rows_padded(n), E = n_embd_, H = n_head_;
  const int64_t wide = std::max(kRows * 3 * E, kRows * (F.Fp + E));
  return uint64_t(R * E + wide) * 4 + uint64_t(R * E + kRows * F.Kp) * 2 + uint64_t(3 * H * R * F.Dp) * 2 + vattn::scratch_bytes(n, F.Dp, H);
}

uint64_t VisionQwen3VL::card_bytes(int tokens) const {
  if (!fast_) return bytes_;
  return bytes_ + (resident_ ? 0 : uint64_t(fast_->kStages) * fast_->blob_bytes) + activation_bytes(int64_t(tokens) * merge_ * merge_);
}

void VisionQwen3VL::free_activations() {
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

void VisionQwen3VL::release() {
  std::lock_guard lock(mu_);
  if (!fast_) return;
  free_activations();
  for (auto*& s : fast_->stage)
    if (s) sycl::free(s, q_), s = nullptr;
}

void VisionQwen3VL::ensure(int64_t n) {
  Fast& F = *fast_;
  if (n <= F.n_cap && (resident_ || F.stage[0])) return;
  free_activations();
  if (!resident_)
    for (int i = 0; i < F.kStages; ++i) {
      auto*& s = F.stage[i];
      if (s) continue;
      s = sycl::malloc_device<uint8_t>(F.blob_bytes, q_);
      if (!s) throw std::runtime_error("vision encoder: out of device memory for the weight staging");
    }
  const int64_t R = vattn::rows_padded(n), E = n_embd_, H = n_head_;
  F.wide_floats = std::max(kRows * 3 * E, kRows * (F.Fp + E));
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
  dev(F.act_ffn, kRows * F.Kp);
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
  held_ = activation_bytes(n) + F.held();
}

void VisionQwen3VL::reserve(int tokens) {
  if (reference_) return;
  std::lock_guard lock(mu_);
  ensure(int64_t(tokens) * merge_ * merge_);
}

QwenImageRows VisionQwen3VL::encode_fast(const image::Rgb& pic) {
  std::lock_guard lock(mu_);
  Fast& F = *fast_;
  const int64_t cols = pic.width / patch_, rows = pic.height / patch_, n = cols * rows, E = n_embd_, H = n_head_, D = d_head_, Dp = F.Dp, Fp = F.Fp,
                Kp = F.Kp;
  const int64_t oxm = cols / 2, oym = rows / 2, n_out = oxm * oym, R = vattn::rows_padded(n), n_units = int64_t(F.units.size()),
                n_layer = n_units - 3, KM = n_hidden_;
  const int side = int(std::lround(std::sqrt(double(n_pos_))));
  const int64_t chunk = round_up((n + (n + kRows - 1) / kRows - 1) / ((n + kRows - 1) / kRows), 256);
  const int64_t chunk_m = round_up((n_out + (n_out + kRows - 1) / kRows - 1) / ((n_out + kRows - 1) / kRows), 256);
  ensure(n);
  const sycl::context ctx = q_.get_context();
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

  // Streaming: unit u's blob goes to staging u % 2; the copy for u + 2 starts once u's kernels are done with it.
  std::vector<sycl::event> copied(static_cast<size_t>(n_units)), done(static_cast<size_t>(n_units));
  auto start_copy = [&](int64_t u) {
    uint8_t* dst = F.stage[u % F.kStages];
    if (F.copy) {
      F.copy->copy(dst, F.units[size_t(u)].blob, F.blob_bytes);
      copied[size_t(u)] = F.copy->event(F.copy->fence());
    } else {
      copied[size_t(u)] = F.copy_q->memcpy(dst, F.units[size_t(u)].blob, F.blob_bytes);
    }
  };
  if (!resident_)
    for (int64_t u = 0; u < std::min<int64_t>(F.kStages, n_units); ++u) start_copy(u);

  // Small per-image device data: the position table, the RoPE angles (rows' then columns' use the same table), the picture.
  const int64_t rot = D / 2, half = rot / 2, maxpos = std::max(cols, rows);
  const ref::RopeTables rt = ref::rope_tables(maxpos, rot, double(rope_theta_), nullptr);
  const int64_t pos_floats = n_pos_ * E, tab_floats = 2 * maxpos * half;
  float* small = sycl::malloc_device<float>(size_t(pos_floats + tab_floats), q_);
  uint8_t* rgb = sycl::malloc_device<uint8_t>(pic.pixels.size(), q_);
  if (!small || !rgb) {
    if (small) sycl::free(small, q_);
    if (rgb) sycl::free(rgb, q_);
    throw std::runtime_error("vision encoder: out of device memory");
  }
  std::shared_ptr<void> small_guard(small, [ctx](void* x) { sycl::free(x, ctx); });
  std::shared_ptr<void> rgb_guard(rgb, [ctx](void* x) { sycl::free(x, ctx); });
  float *posd = small, *cos_t = small + pos_floats, *sin_t = cos_t + maxpos * half;
  q_.memcpy(posd, F.pos, size_t(pos_floats) * 4);
  q_.memcpy(cos_t, rt.cos.data(), size_t(maxpos * half) * 4);
  q_.memcpy(sin_t, rt.sin.data(), size_t(maxpos * half) * 4);
  q_.memcpy(rgb, pic.pixels.data(), pic.pixels.size());

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
  const float eps = eps_, qscale = 1.0f / std::sqrt(float(D));
  float* x = F.x;
  sycl::half* am = F.act_main;
  const float mean0 = mean_[0], mean1 = mean_[1], mean2 = mean_[2], std0 = std_[0], std1 = std_[1], std2 = std_[2];

  // The picture as the patch GEMM's activations: patch r of the merge order is the (r % 4 / 2, r % 2) member of block r / 4;
  // element c is (channel c / 256, y, x) of its 16 x 16 pixels, scaled as clip_image_f32::normalize.
  {
    const int64_t W = pic.width;
    q_.parallel_for(sycl::range<2>(size_t(n), size_t(n_in_)), [=](sycl::id<2> id) {
      const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
      const int64_t b = r / 4, w = r % 4, py = 2 * (b / oxm) + w / 2, px = 2 * (b % oxm) + w % 2;
      const int64_t ch = c / 256, yy = (c % 256) / 16, xx = c % 16;
      const float mean = ch == 0 ? mean0 : ch == 1 ? mean1 : mean2, sd = ch == 0 ? std0 : ch == 1 ? std1 : std2;
      const float v = (float(rgb[((py * 16 + yy) * W + px * 16 + xx) * 3 + ch]) / 255.0f - mean) / sd;
      am[xmx::act16_index(r, c, 48)] = sycl::half(v);
    });
  }
  gemm(am, n, n_in_, F.patch_w, x);
  mark("patch");
  // x += patch bias + position (the table interpolated to this grid); the first layer's norm.
  {
    const float *pb = F.patch_b, *w1 = F.vecs[0].ln1_w, *b1 = F.vecs[0].ln1_b;
    const int icols = int(cols), irows = int(rows);
    rows_sg(q_, n, [=](int64_t r, sycl::sub_group sg) {
      const int lane = int(sg.get_local_id()[0]);
      const int64_t b = r / 4, w = r % 4;
      const PosLerp l = pos_lerp(int(2 * (b / oxm) + w / 2), int(2 * (b % oxm) + w % 2), irows, icols, side);
      float v[kPerLane];
      float s = 0;
      int j = 0;
      for (int64_t c = lane; c < E; c += kSg, ++j) {
        const float p = posd[int64_t(l.i00) * E + c] * l.w00 + posd[int64_t(l.i01) * E + c] * l.w01 + posd[int64_t(l.i10) * E + c] * l.w10 +
                        posd[int64_t(l.i11) * E + c] * l.w11;
        v[j] = (x[r * E + c] + pb[c]) + p;
        x[r * E + c] = v[j];
        s += v[j];
      }
      const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / float(E);
      float ss = 0;
      j = 0;
      for (int64_t c = lane; c < E; c += kSg, ++j) ss += (v[j] - mean) * (v[j] - mean);
      const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / float(E) + eps);
      j = 0;
      for (int64_t c = lane; c < E; c += kSg, ++j) am[xmx::act16_index(r, c, E / 16)] = sycl::half((v[j] - mean) * inv * w1[c] + b1[c]);
    });
  }
  mark("pos");
  tap("inp_pos_emb", -1, x, n, E);

  for (int64_t u = 0; u < n_units; ++u) {
    if (!resident_) {
      q_.submit([&](sycl::handler& h) {
        h.depends_on(copied[size_t(u)]);
        h.single_task([] {});
      });
    }
    const uint8_t* st = resident_ ? F.units[size_t(u)].blob : F.stage[u % F.kStages];
    auto W = [&](int i, int64_t k, int64_t nn) {
      xmx::Weight w;
      w.k = k, w.n = nn, w.format = F.fmt;
      w.tiles = const_cast<uint8_t*>(st) + F.t_off[i];
      w.scales = nullptr;
      return w;
    };
    float* wide = F.wide;
    mark("wait_copy");
    if (u < n_layer) {
      const Fast::LayerVecs& V = F.vecs[size_t(u)];
      for (int64_t c0 = 0; c0 < n; c0 += chunk) {
        const int64_t m = std::min(chunk, n - c0);
        gemm(am + (c0 / 8) * (E / 16) * 128, m, E, W(0, E, 3 * E), wide);
        mark("gemm_qkv");
        const int64_t t0 = c0;
        sycl::half *qt = F.qt, *kt = F.kt, *vt = F.vt;
        const float* qkv_b = V.qkv_b;
        const int64_t E3 = 3 * E;
        // q, k: the row plus its bias, turned by the vision M-RoPE: head dims (i, i + D/2) pair up, pairs [0, D/4) turn by the
        // patch's row, [D/4, D/2) by its column; q also scaled by 1/sqrt(D) (the attention kernel's softmax scale is 1).
        auto rope = [=](const float* src, const float* bias, int64_t t, float* y, float scale) {
          const int64_t b = t / 4, w = t % 4, py = 2 * (b / oxm) + w / 2, px = 2 * (b % oxm) + w % 2;
          for (int64_t d = 0; d < D; ++d) y[d] = src[d] + bias[d];
          for (int64_t i = 0; i < rot; ++i) {
            const int64_t pos = i < half ? py : px, ti = pos * half + (i % half);
            const float c = cos_t[ti], s = sin_t[ti];
            const float a = y[i], bb = y[i + rot];
            y[i] = (a * c - bb * s) * scale, y[i + rot] = (a * s + bb * c) * scale;
          }
        };
        q_.parallel_for(sycl::range<2>(size_t(H), size_t(m)), [=](sycl::id<2> id) {
          const int64_t h = int64_t(id[0]), t = t0 + int64_t(id[1]);
          float y[80];
          rope(wide + (t - t0) * E3 + h * D, qkv_b + h * D, t, y, qscale);
          sycl::half* dst = qt + h * R * Dp;
          for (int64_t d = 0; d < D; d += 8) {
            sycl::vec<sycl::half, 8> v;
            for (int j = 0; j < 8; ++j) v[j] = sycl::half(y[d + j]);
            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(dst + vattn::q_index(t, d, Dp)) = v;
          }
        });
        q_.parallel_for(sycl::range<2>(size_t(H), size_t(m)), [=](sycl::id<2> id) {
          const int64_t h = int64_t(id[0]), t = t0 + int64_t(id[1]);
          float y[80];
          rope(wide + (t - t0) * E3 + E + h * D, qkv_b + E + h * D, t, y, 1.0f);
          sycl::half* dst = kt + h * R * Dp;
          for (int64_t d = 0; d < D; d += 2)
            *reinterpret_cast<sycl::vec<sycl::half, 2>*>(dst + vattn::k_index(t, d, Dp)) = sycl::vec<sycl::half, 2>(sycl::half(y[d]), sycl::half(y[d + 1]));
        });
        // v: plus its bias, two patches a work-item (positions 2j, 2j + 1 are adjacent in the tile).
        q_.parallel_for(sycl::range<2>(size_t(H), size_t((m + 1) / 2)), [=](sycl::id<2> id) {
          const int64_t h = int64_t(id[0]), tv = t0 + 2 * int64_t(id[1]);  // c0 is even
          sycl::half* dst = vt + h * R * Dp;
          const float* vb = qkv_b + 2 * E + h * D;
          const int64_t ta = std::min(tv, t0 + m - 1), tb = std::min(tv + 1, t0 + m - 1);
          const float *sa = wide + (ta - t0) * E3 + 2 * E + h * D, *sb = wide + (tb - t0) * E3 + 2 * E + h * D;
          for (int64_t d = 0; d < D; ++d)
            *reinterpret_cast<sycl::vec<sycl::half, 2>*>(dst + vattn::v_index(tv, d, Dp)) =
                sycl::vec<sycl::half, 2>(sycl::half(sa[d] + vb[d]), sycl::half(sb[d] + vb[d]));
        });
      }
      mark("prep");
      vattn::attend(q_, F.qt, F.kt, F.vt, n, H, Dp, D, F.att, am, E);
      mark("attention");
      for (int64_t c0 = 0; c0 < n; c0 += chunk) {
        const int64_t m = std::min(chunk, n - c0);
        sycl::half* a_chunk = am + (c0 / 8) * (E / 16) * 128;
        gemm(a_chunk, m, E, W(1, E, E), wide);
        mark("gemm_o");
        // x += o + bias; the FFN's norm.
        {
          const float *ob = V.o_b, *w2 = V.ln2_w, *b2 = V.ln2_b;
          float* xc = x + c0 * E;
          sycl::half* ac = a_chunk;
          rows_sg(q_, m, [=](int64_t r, sycl::sub_group sg) {
            const int lane = int(sg.get_local_id()[0]);
            float v[kPerLane];
            float s = 0;
            int j = 0;
            for (int64_t c = lane; c < E; c += kSg, ++j) {
              v[j] = xc[r * E + c] + (wide[r * E + c] + ob[c]);
              xc[r * E + c] = v[j];
              s += v[j];
            }
            const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / float(E);
            float ss = 0;
            j = 0;
            for (int64_t c = lane; c < E; c += kSg, ++j) ss += (v[j] - mean) * (v[j] - mean);
            const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / float(E) + eps);
            j = 0;
            for (int64_t c = lane; c < E; c += kSg, ++j) ac[xmx::act16_index(r, c, E / 16)] = sycl::half((v[j] - mean) * inv * w2[c] + b2[c]);
          });
        }
        mark("post1");
        float* gu = wide;
        float* dn = wide + kRows * Fp;
        gemm(a_chunk, m, E, W(2, E, Fp), gu);
        mark("gemm_up");
        {
          sycl::half* af = F.act_ffn;
          const float* ub = V.up_b;
          q_.parallel_for(sycl::range<2>(size_t(m), size_t(Kp)), [=](sycl::id<2> id) {
            const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
            const float g = gu[r * Fp + c] + ub[c];
            af[xmx::act16_index(r, c, Kp / 16)] = sycl::half(0.5f * g * (1.0f + sycl::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g))));
          });
        }
        mark("gelu");
        gemm(F.act_ffn, m, Kp, W(3, Kp, E), dn);
        mark("gemm_down");
        // x += down + bias; then the next layer's first norm, or (after the last) the post norm written as the merger's
        // activations: row r (patch) is columns [(r % 4) * E, +E) of merged row r / 4.
        const bool last = u + 1 == n_layer;
        const float *db = V.down_b, *wn = last ? F.post_w : F.vecs[size_t(u) + 1].ln1_w, *bn = last ? F.post_b : F.vecs[size_t(u) + 1].ln1_b;
        float* xc = x + c0 * E;
        const int64_t base = c0;
        rows_sg(q_, m, [=](int64_t r, sycl::sub_group sg) {
          const int lane = int(sg.get_local_id()[0]);
          float v[kPerLane];
          float s = 0;
          int j = 0;
          for (int64_t c = lane; c < E; c += kSg, ++j) {
            v[j] = xc[r * E + c] + (dn[r * E + c] + db[c]);
            xc[r * E + c] = v[j];
            s += v[j];
          }
          const float mean = sycl::reduce_over_group(sg, s, sycl::plus<float>()) / float(E);
          float ss = 0;
          j = 0;
          for (int64_t c = lane; c < E; c += kSg, ++j) ss += (v[j] - mean) * (v[j] - mean);
          const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(sg, ss, sycl::plus<float>()) / float(E) + eps);
          j = 0;
          if (last) {
            const int64_t g = base + r;
            for (int64_t c = lane; c < E; c += kSg, ++j)
              am[xmx::act16_index(g / 4, (g % 4) * E + c, 4 * E / 16)] = sycl::half((v[j] - mean) * inv * wn[c] + bn[c]);
          } else {
            sycl::half* ac = am + (base / 8) * (E / 16) * 128;
            for (int64_t c = lane; c < E; c += kSg, ++j) ac[xmx::act16_index(r, c, E / 16)] = sycl::half((v[j] - mean) * inv * wn[c] + bn[c]);
          }
        });
      }
      mark("post2");
      tap("layer_out", int(u), x, n, E);
    } else if (u < n_layer + 2) {
      // mm0, half the hidden units at a time: gelu(mm0 . merged + bias) as fp16 activations (in x's memory, free now).
      sycl::half* g = reinterpret_cast<sycl::half*>(x);
      const int64_t hh = KM / 2, c_off = (u - n_layer) * hh;
      const float* b0 = F.mm0_b;
      for (int64_t c0 = 0; c0 < n_out; c0 += chunk_m) {
        const int64_t m = std::min(chunk_m, n_out - c0);
        gemm(am + (c0 / 8) * (KM / 16) * 128, m, KM, W(0, KM, hh), wide);
        mark("gemm_mm0");
        const int64_t base = c0;
        q_.parallel_for(sycl::range<2>(size_t(m), size_t(hh)), [=](sycl::id<2> id) {
          const int64_t r = int64_t(id[0]), c = int64_t(id[1]);
          const float v = wide[r * hh + c] + b0[c_off + c];
          g[xmx::act16_index(base + r, c_off + c, KM / 16)] = sycl::half(0.5f * v * (1.0f + sycl::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))));
        });
        mark("gelu_mm0");
      }
    } else {
      const sycl::half* g = reinterpret_cast<sycl::half*>(x);
      const float* b2 = F.mm2_b;
      const int64_t NO = n_out_;
      for (int64_t c0 = 0; c0 < n_out; c0 += chunk_m) {
        const int64_t m = std::min(chunk_m, n_out - c0);
        float* oc = out.get() + c0 * NO;
        gemm(const_cast<sycl::half*>(g) + (c0 / 8) * (KM / 16) * 128, m, KM, W(0, KM, NO), oc);
        q_.parallel_for(sycl::range<2>(size_t(m), size_t(NO)), [=](sycl::id<2> id) { oc[int64_t(id[0]) * NO + int64_t(id[1])] += b2[int64_t(id[1])]; });
      }
      mark("mm2");
    }
    // Stream: unit u + 1 goes into the staging unit u - 1 used, once the card is done with that unit; unit u is queued,
    // so the card does not idle while the host waits.
    if (!resident_) {
      done[size_t(u)] = q_.submit([&](sycl::handler& h) { h.single_task([] {}); });
      if (u >= 1 && u + F.kStages - 1 < n_units) {
        done[size_t(u) - 1].wait();
        start_copy(u + F.kStages - 1);
      }
    }
  }
  if (profile)
    for (const auto& [k, v] : prof) std::fprintf(stderr, "  %-10s %8.2f ms\n", k.c_str(), v);
  tap("embeddings", -1, out.get(), n_out, n_out_);
  const auto t_enq = std::chrono::steady_clock::now();
  q_.wait();
  if (host_prof) {
    const auto t_end = std::chrono::steady_clock::now();
    std::fprintf(stderr, "  host: enqueued in %.1f ms, done at %.1f ms\n", std::chrono::duration<double, std::milli>(t_enq - t_begin).count(),
                 std::chrono::duration<double, std::milli>(t_end - t_begin).count());
  }
  return {{out, n_out, n_out_}, int(oxm), int(oym)};
}

}  // namespace eklow::model
