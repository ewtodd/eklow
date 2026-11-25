#include "model/vision_inkling2v.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>

#include "image/inkling_image.hpp"
#include "kernels/ref.hpp"

namespace eklow::model {

namespace {

// LayerNorm of each row (eps, weight, bias), a work-group a row.
void layernorm(sycl::queue& q, const float* x, const float* w, const float* b, int64_t rows, int64_t cols, float eps, float* out) {
  constexpr int kWg = 256;
  q.parallel_for(sycl::nd_range<1>(size_t(rows * kWg), kWg), [=](sycl::nd_item<1> it) {
    const int64_t r = int64_t(it.get_group(0));
    const int lid = int(it.get_local_id(0));
    const float* xr = x + r * cols;
    float s = 0;
    for (int64_t i = lid; i < cols; i += kWg) s += xr[i];
    const float mean = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>()) / float(cols);
    float v = 0;
    for (int64_t i = lid; i < cols; i += kWg) v += (xr[i] - mean) * (xr[i] - mean);
    const float inv = 1.0f / sycl::sqrt(sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>()) / float(cols) + eps);
    for (int64_t i = lid; i < cols; i += kWg) out[r * cols + i] = (xr[i] - mean) * inv * w[i] + b[i];
  });
}

inline float gelu_erf(float x) { return 0.5f * x * (1.0f + sycl::erf(x * 0.70710678118654752f)); }

// y += b (each row), then GELU-erf when gelu.
void bias_act(sycl::queue& q, float* y, const float* b, int64_t rows, int64_t cols, bool gelu) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(cols)), [=](sycl::id<2> id) {
    const int64_t i = int64_t(id[0]) * cols + int64_t(id[1]);
    const float v = y[i] + (b ? b[id[1]] : 0.0f);
    y[i] = gelu ? gelu_erf(v) : v;
  });
}

// x += ls * (y + b)
void residual(sycl::queue& q, float* x, const float* y, const float* b, const float* ls, int64_t rows, int64_t cols) {
  q.parallel_for(sycl::range<2>(size_t(rows), size_t(cols)), [=](sycl::id<2> id) {
    const int64_t i = int64_t(id[0]) * cols + int64_t(id[1]);
    x[i] += ls[id[1]] * (y[i] + b[id[1]]);
  });
}

}  // namespace

float* VisionInkling2V::upload_f32(const gguf::Tensor& t) {
  if (t.type != gguf::Type::F32) throw std::runtime_error("inkling2v: " + t.name + " is not F32");
  float* p = sycl::malloc_device<float>(t.elements(), q_);
  if (!p) throw std::runtime_error("inkling2v: out of device memory");
  q_.memcpy(p, t.data, t.nbytes).wait();
  owned_.push_back(p);
  bytes_ += t.nbytes;
  return p;
}

// F16 matrices of K = k inputs stacked by their outputs (parts), each row padded to k_pad with zeros, as the fp16 GEMM's
// weight.
xmx::Weight VisionInkling2V::upload_f16(const std::vector<const gguf::Tensor*>& parts, int64_t k, int64_t k_pad) {
  int64_t n = 0;
  for (const gguf::Tensor* t : parts) {
    if (t->type != gguf::Type::F16) throw std::runtime_error("inkling2v: " + t->name + " is not F16");
    if (int64_t(t->elements()) % k) throw std::runtime_error("inkling2v: " + t->name + " has another K");
    n += int64_t(t->elements()) / k;
  }
  std::vector<uint16_t> host(size_t(n * k_pad), 0);
  int64_t row = 0;
  for (const gguf::Tensor* t : parts) {
    const auto* src = reinterpret_cast<const uint16_t*>(t->data);
    const int64_t rows = int64_t(t->elements()) / k;
    for (int64_t r = 0; r < rows; ++r) std::memcpy(host.data() + (row + r) * k_pad, src + r * k, size_t(k) * 2);
    row += rows;
  }
  auto* dev = sycl::malloc_device<uint16_t>(host.size(), q_);
  if (!dev) throw std::runtime_error("inkling2v: out of device memory");
  q_.memcpy(dev, host.data(), host.size() * 2).wait();
  xmx::Weight w = xmx::repack(q_, ref::Weight{gguf::Type::F16, k_pad, n, reinterpret_cast<const uint8_t*>(dev), uint64_t(k_pad) * 2});
  q_.wait();
  sycl::free(dev, q_);
  const uint64_t b = wbytes(w);
  if (host_weights_) {
    // repacked on the card, kept in pinned host memory
    auto* h = sycl::malloc_host<uint8_t>(b, q_);
    if (!h) throw std::runtime_error("inkling2v: out of pinned host memory");
    q_.memcpy(h, w.tiles, b).wait();
    sycl::free(w.tiles, q_);
    w.tiles = h;
    host_bytes_ += b;
  } else {
    bytes_ += b;
  }
  return w;
}

VisionInkling2V::VisionInkling2V(sycl::queue& q, const gguf::File& f, bool host_weights) : q_(q), host_weights_(host_weights) {
  if (f.at("general.architecture").as_string() != "clip" || !f.find("clip.vision.projector_type") ||
      f.at("clip.vision.projector_type").as_string() != "inkling2v")
    throw std::runtime_error(f.path() + ": not an mmproj with the inkling2v projector");
  n_embd_ = f.at("clip.vision.embedding_length").as_int();
  n_head_ = f.at("clip.vision.attention.head_count").as_int();
  d_head_ = n_embd_ / n_head_;
  n_ff_ = f.at("clip.vision.feed_forward_length").as_int();
  n_out_ = f.at("clip.vision.projection_dim").as_int();
  eps_ = float(f.at("clip.vision.attention.layer_norm_epsilon").as_float());
  if (f.at("clip.vision.patch_size").as_int() != image::kInklingPatch || f.at("clip.vision.spatial_merge_size").as_int() != image::kInklingMerge)
    throw std::runtime_error(f.path() + ": patch 20 merged 2 x 2 is what the preprocessing makes");
  if (d_head_ != 64) throw std::runtime_error(f.path() + ": vision_attn takes heads of 64 here");
  for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"})
    for (const gguf::Value& v : f.at(key).as_array())
      if (v.as_float() != 0.5) throw std::runtime_error(f.path() + ": image_mean/image_std other than 0.5 are not supported");
  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing " + name);
    return *t;
  };
  const int64_t n_layer = f.at("clip.vision.block_count").as_int();
  for (int64_t l = 0; l < n_layer; ++l) {
    const std::string b = "v.blk." + std::to_string(l) + ".";
    Layer L;
    L.qkv = upload_f16({&need(b + "attn_q.weight"), &need(b + "attn_k.weight"), &need(b + "attn_v.weight")}, n_embd_, n_embd_);
    L.o = upload_f16({&need(b + "attn_out.weight")}, n_embd_, n_embd_);
    L.up = upload_f16({&need(b + "ffn_up.weight")}, n_embd_, n_embd_);
    L.down = upload_f16({&need(b + "ffn_down.weight")}, n_ff_, n_ff_);
    {
      // the three biases side by side, as the stacked matrix's outputs
      std::vector<float> host;
      for (const char* p : {"attn_q.bias", "attn_k.bias", "attn_v.bias"}) {
        const gguf::Tensor& t = need(b + p);
        const auto* v = reinterpret_cast<const float*>(t.data);
        host.insert(host.end(), v, v + t.elements());
      }
      L.b_qkv = sycl::malloc_device<float>(host.size(), q_);
      q_.memcpy(L.b_qkv, host.data(), host.size() * 4).wait();
      owned_.push_back(L.b_qkv);
    }
    L.b_o = upload_f32(need(b + "attn_out.bias"));
    L.b_up = upload_f32(need(b + "ffn_up.bias"));
    L.b_down = upload_f32(need(b + "ffn_down.bias"));
    L.ln1_w = upload_f32(need(b + "ln1.weight"));
    L.ln1_b = upload_f32(need(b + "ln1.bias"));
    L.ln2_w = upload_f32(need(b + "ln2.weight"));
    L.ln2_b = upload_f32(need(b + "ln2.bias"));
    L.ls1 = upload_f32(need(b + "ls1.weight"));
    L.ls2 = upload_f32(need(b + "ls2.weight"));
    layers_.push_back(L);
  }
  // the patch embedding [n_embd][3 x 20 x 20] ([channel][y][x], as image::Patches), K padded to a multiple of 32
  const int64_t kp = 3 * image::kInklingPatch * image::kInklingPatch;
  patch_w_ = upload_f16({&need("v.patch_embd.weight")}, kp, (kp + 31) / 32 * 32);
  n_proj_ = int64_t(need("mm.1.weight").elements()) / (4 * n_embd_);
  mm1_ = upload_f16({&need("mm.1.weight")}, 4 * n_embd_, 4 * n_embd_);
  mm2_ = upload_f16({&need("mm.2.weight")}, n_proj_, n_proj_);
  pre_ln_w_ = upload_f32(need("v.pre_ln.weight"));
  pre_ln_b_ = upload_f32(need("v.pre_ln.bias"));
  {
    // the CLS token's row: its embedding plus its position embedding
    const gguf::Tensor &c = need("v.class_embd"), &cp = need("v.class_position_embd.weight");
    std::vector<float> host(static_cast<size_t>(n_embd_));
    for (int64_t e = 0; e < n_embd_; ++e) host[size_t(e)] = reinterpret_cast<const float*>(c.data)[e] + reinterpret_cast<const float*>(cp.data)[e];
    cls_ = sycl::malloc_device<float>(size_t(n_embd_), q_);
    q_.memcpy(cls_, host.data(), host.size() * 4).wait();
    owned_.push_back(cls_);
  }
  {
    const gguf::Tensor& p = need("v.position_embd.weight");
    if (p.type != gguf::Type::F32) throw std::runtime_error(f.path() + ": the position embeddings are not F32");
    grid_ = int64_t(std::lround(std::sqrt(double(p.elements() / uint64_t(n_embd_)))));
    pos_.assign(reinterpret_cast<const float*>(p.data), reinterpret_cast<const float*>(p.data) + p.elements());
  }
}

VisionInkling2V::~VisionInkling2V() {
  q_.wait();
  if (copy_q_) copy_q_->wait();
  for (auto& L : layers_)
    for (xmx::Weight* w : {&L.qkv, &L.o, &L.up, &L.down}) xmx::free_weight(q_, *w);
  for (xmx::Weight* w : {&patch_w_, &mm1_, &mm2_}) xmx::free_weight(q_, *w);
  for (void* p : owned_) sycl::free(p, q_);
  if (cap_) {  // (the workspace's are someone else's)
    for (void* p : {static_cast<void*>(x_), static_cast<void*>(h_), static_cast<void*>(qkv_), static_cast<void*>(wide_), static_cast<void*>(pin_),
                    static_cast<void*>(rp_), static_cast<void*>(qt_), static_cast<void*>(kt_), static_cast<void*>(vt_)})
      if (p) sycl::free(p, q_);
    xmx::free_activations(q_, act_);
    vattn::release(q_, att_);
  }
}

void VisionInkling2V::use_workspace(std::vector<std::pair<void*, size_t>> regions, std::function<void()> drain) {
  if (cap_) throw std::runtime_error("inkling2v: the encoder has activations of its own already");
  ws_ = std::move(regions);
  drain_ = std::move(drain);
  if (host_weights_ && !copy_q_) copy_q_ = std::make_unique<sycl::queue>(q_.get_context(), q_.get_device(), sycl::property::queue::in_order());
}

uint64_t VisionInkling2V::layer_wbytes() const {
  const Layer& L = layers_.front();
  return wbytes(L.qkv) + wbytes(L.o) + wbytes(L.up) + wbytes(L.down);
}

// The buffers an encode of n_pos positions uses (as ensure allocates them, and with host weights the weight slots),
// largest first, each into the first workspace region with room (4 KiB aligned).
bool VisionInkling2V::place(int64_t n_pos, bool apply) {
  const int64_t R = vattn::rows_padded(n_pos), kp = (3 * image::kInklingPatch * image::kInklingPatch + 31) / 32 * 32, E = n_embd_;
  const int64_t cap = R * std::max({kp, n_ff_, 4 * E});
  const vattn::ScratchSizes ss = vattn::scratch_sizes(R, d_head_, n_head_);
  struct Item {
    uint64_t bytes;
    std::function<void(uint8_t*)> set;
  };
  auto as = [](auto*& p) { return [&p](uint8_t* a) { p = reinterpret_cast<std::remove_reference_t<decltype(p)>>(a); }; };
  std::vector<Item> items = {
      {uint64_t(R * E) * 4, as(x_)},
      {uint64_t(R * E) * 4, as(h_)},
      {uint64_t(R * 3 * E) * 4, as(qkv_)},
      {uint64_t(R * std::max(n_ff_, 4 * E)) * 4, as(wide_)},
      {uint64_t(R * kp) * 4, as(pin_)},
      {uint64_t(2 * R) * 4, as(rp_)},
      {uint64_t(R * E) * 2, as(qt_)},
      {uint64_t(R * E) * 2, as(kt_)},
      {uint64_t(R * E) * 2, as(vt_)},
      {uint64_t(cap), as(act_.q)},
      {uint64_t(cap / 32) * 4, as(act_.s)},
      {uint64_t(cap) * 2, as(act_.h)},
      {uint64_t(ss.p_halves) * 2, as(att_.p)},
      {uint64_t(ss.stat_floats) * 4, as(att_.stat)},
  };
  if (host_weights_) {
    for (int i = 0; i < 2; ++i) items.push_back({layer_wbytes(), as(slot_[i])});
    items.push_back({wbytes(patch_w_), as(dpatch_.tiles)});
    items.push_back({wbytes(mm1_), as(dmm1_.tiles)});
    items.push_back({wbytes(mm2_), as(dmm2_.tiles)});
  }
  std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.bytes > b.bytes; });
  std::vector<uint64_t> used(ws_.size(), 0);
  std::vector<uint8_t*> at;
  for (const Item& it : items) {
    size_t r = 0;
    for (; r < ws_.size() && used[r] + it.bytes > ws_[r].second; ++r) {
    }
    if (r == ws_.size()) return false;
    at.push_back(static_cast<uint8_t*>(ws_[r].first) + used[r]);
    used[r] = (used[r] + it.bytes + 4095) / 4096 * 4096;
  }
  if (apply) {
    for (size_t i = 0; i < items.size(); ++i) items[i].set(at[i]);
    act_.capacity = cap;
    att_.p_halves = ss.p_halves, att_.stat_floats = ss.stat_floats;
    if (host_weights_) {
      dpatch_ = xmx::Weight{patch_w_.k, patch_w_.n, dpatch_.tiles, nullptr, patch_w_.format};
      dmm1_ = xmx::Weight{mm1_.k, mm1_.n, dmm1_.tiles, nullptr, mm1_.format};
      dmm2_ = xmx::Weight{mm2_.k, mm2_.n, dmm2_.tiles, nullptr, mm2_.format};
    }
  }
  return true;
}

VisionInkling2V::Layer VisionInkling2V::slot_layer(size_t l, int slot) const {
  Layer L = layers_[l];
  uint8_t* p = slot_[slot];
  for (xmx::Weight* w : {&L.qkv, &L.o, &L.up, &L.down}) w->tiles = p, p += wbytes(*w);
  return L;
}

sycl::event VisionInkling2V::put_layer(size_t l, int slot, const std::vector<sycl::event>& after) {
  const Layer& L = layers_[l];
  uint8_t* p = slot_[slot];
  sycl::event e;
  for (const xmx::Weight* w : {&L.qkv, &L.o, &L.up, &L.down}) {
    e = copy_q_->submit([&](sycl::handler& h) {
      h.depends_on(after);
      h.memcpy(p, w->tiles, wbytes(*w));
    });
    p += wbytes(*w);
  }
  return e;
}

void VisionInkling2V::ensure(int64_t n_pos) {
  if (n_pos <= cap_) return;
  q_.wait();
  for (void* p : {static_cast<void*>(x_), static_cast<void*>(h_), static_cast<void*>(qkv_), static_cast<void*>(wide_), static_cast<void*>(pin_),
                  static_cast<void*>(rp_), static_cast<void*>(qt_), static_cast<void*>(kt_), static_cast<void*>(vt_)})
    if (p) sycl::free(p, q_);
  if (cap_) xmx::free_activations(q_, act_);
  vattn::release(q_, att_);
  cap_ = n_pos;
  const int64_t R = vattn::rows_padded(cap_), kp = (3 * image::kInklingPatch * image::kInklingPatch + 31) / 32 * 32;
  x_ = sycl::malloc_device<float>(size_t(R * n_embd_), q_);
  h_ = sycl::malloc_device<float>(size_t(R * n_embd_), q_);
  qkv_ = sycl::malloc_device<float>(size_t(R * 3 * n_embd_), q_);
  wide_ = sycl::malloc_device<float>(size_t(R * std::max(n_ff_, 4 * n_embd_)), q_);
  pin_ = sycl::malloc_device<float>(size_t(R * kp), q_);
  rp_ = sycl::malloc_device<int32_t>(size_t(2 * R), q_);
  qt_ = sycl::malloc_device<sycl::half>(size_t(R * n_embd_), q_);
  kt_ = sycl::malloc_device<sycl::half>(size_t(R * n_embd_), q_);
  vt_ = sycl::malloc_device<sycl::half>(size_t(R * n_embd_), q_);
  if (!x_ || !h_ || !qkv_ || !wide_ || !pin_ || !rp_ || !qt_ || !kt_ || !vt_) throw std::runtime_error("inkling2v: out of device memory for the activations");
  act_ = xmx::alloc_activations(q_, R, std::max({kp, n_ff_, 4 * n_embd_}));
  vattn::reserve(q_, att_, R, d_head_, n_head_);
  act_bytes_ = uint64_t(R) * uint64_t(n_embd_ + n_embd_ + 3 * n_embd_ + std::max(n_ff_, 4 * n_embd_) + kp + 2) * 4 +
               uint64_t(R) * uint64_t(3 * n_embd_) * 2 + uint64_t(act_.capacity) * 3 + vattn::scratch_bytes(R, d_head_, n_head_);
}

std::vector<float> VisionInkling2V::encode(const image::Patches& p, const Dump& dump) {
  const int64_t gw = p.cols, gh = p.rows, n_patch = gw * gh, n = n_patch + 1, E = n_embd_, H = n_head_, D = d_head_;
  if (gw % 2 || gh % 2) throw std::runtime_error("inkling2v: the patch grid must be even (2 x 2 merges)");
  const int64_t kp = (p.dim + 31) / 32 * 32, R = vattn::rows_padded(n), tokens = n_patch / 4;
  if (!ws_.empty()) {
    if (!place(n, true)) throw std::invalid_argument("inkling2v: the image does not fit the encoder's workspace");
    if (drain_) drain_();
    q_.memset(att_.p, 0, att_.p_halves * sizeof(sycl::half));  // as vattn::reserve leaves it
    q_.memset(att_.stat, 0, att_.stat_floats * sizeof(float));
  } else {
    if (host_weights_) throw std::runtime_error("inkling2v: host weights need a workspace");
    ensure(n);
  }
  // With host weights: the patch embedding and the first two layers on their way (the rest follow as slots free up,
  // the projector after the last layer).
  std::vector<sycl::event> lay(layers_.size());
  sycl::event e_patch, e_mm;
  if (host_weights_) {
    e_patch = copy_q_->memcpy(dpatch_.tiles, patch_w_.tiles, wbytes(patch_w_));
    for (size_t l = 0; l < std::min<size_t>(2, layers_.size()); ++l) lay[l] = put_layer(l, int(l), {});
  }
  const xmx::Weight& patch_w = host_weights_ ? dpatch_ : patch_w_;
  const xmx::Weight& mm1 = host_weights_ ? dmm1_ : mm1_;
  const xmx::Weight& mm2 = host_weights_ ? dmm2_ : mm2_;
  // the patches, K padded
  {
    std::vector<float> host(size_t(n_patch * kp), 0.0f);
    for (int64_t i = 0; i < n_patch; ++i) std::memcpy(host.data() + i * kp, p.data.data() + i * p.dim, size_t(p.dim) * 4);
    q_.memcpy(pin_, host.data(), host.size() * 4).wait();
  }
  act_.k = kp;
  xmx::quantize(q_, pin_, n_patch, act_, true);
  if (host_weights_) q_.ext_oneapi_submit_barrier({e_patch});
  xmx::matmul(q_, act_, patch_w, x_ + E);  // rows 1..n: the patches (row 0, the CLS token, follows)
  // the position grid resized to gh x gw (ggml_interpolate, bilinear, half-pixel centers), added on the host's copy
  {
    std::vector<float> pos(static_cast<size_t>(n_patch * E));
    if (gw == grid_ && gh == grid_) {
      std::copy(pos_.begin(), pos_.begin() + n_patch * E, pos.begin());
    } else {
      const float sf0 = float(gw) / float(grid_), sf1 = float(gh) / float(grid_);
      for (int64_t i1 = 0; i1 < gh; ++i1) {
        const float y = (float(i1) + 0.5f) / sf1 - 0.5f;
        int64_t y0 = int64_t(std::floor(y)), y1 = y0 + 1;
        y0 = std::clamp<int64_t>(y0, 0, grid_ - 1), y1 = std::clamp<int64_t>(y1, 0, grid_ - 1);
        const float dy = std::clamp(y - float(y0), 0.0f, 1.0f);
        for (int64_t i0 = 0; i0 < gw; ++i0) {
          const float x = (float(i0) + 0.5f) / sf0 - 0.5f;
          int64_t x0 = int64_t(std::floor(x)), x1 = x0 + 1;
          x0 = std::clamp<int64_t>(x0, 0, grid_ - 1), x1 = std::clamp<int64_t>(x1, 0, grid_ - 1);
          const float dx = std::clamp(x - float(x0), 0.0f, 1.0f);
          const float* a = pos_.data() + (y0 * grid_ + x0) * E;
          const float* b = pos_.data() + (y0 * grid_ + x1) * E;
          const float* c = pos_.data() + (y1 * grid_ + x0) * E;
          const float* d = pos_.data() + (y1 * grid_ + x1) * E;
          float* o = pos.data() + (i1 * gw + i0) * E;
          for (int64_t e = 0; e < E; ++e)
            o[e] = a[e] * (1 - dx) * (1 - dy) + b[e] * dx * (1 - dy) + c[e] * (1 - dx) * dy + d[e] * dx * dy;
        }
      }
    }
    q_.memcpy(h_, pos.data(), pos.size() * 4).wait();
    float *x = x_, *h = h_;
    const float* cls = cls_;
    q_.parallel_for(sycl::range<2>(size_t(n), size_t(E)), [=](sycl::id<2> id) {
      const int64_t r = int64_t(id[0]), e = int64_t(id[1]);
      x[r * E + e] = r == 0 ? cls[e] : x[r * E + e] + h[(r - 1) * E + e];
    });
  }
  // RoPE positions: the CLS token at (0, 0), patch (x, y) at (x + 1, y + 1)
  {
    std::vector<int32_t> rp(size_t(2 * R), 0);
    for (int64_t i = 0; i < n_patch; ++i) rp[size_t(2 * (i + 1))] = int32_t(i % gw + 1), rp[size_t(2 * (i + 1) + 1)] = int32_t(i / gw + 1);
    q_.memcpy(rp_, rp.data(), rp.size() * 4).wait();
  }
  if (dump) {
    q_.wait();
    dump("embeddings", x_, n, E);
  }
  layernorm(q_, x_, pre_ln_w_, pre_ln_b_, n, E, eps_, x_);
  const float scale = 1.0f / std::sqrt(float(D)), theta = theta_;
  for (size_t l = 0; l < layers_.size(); ++l) {
    const Layer L = host_weights_ ? slot_layer(l, int(l % 2)) : layers_[l];
    if (host_weights_) q_.ext_oneapi_submit_barrier({lay[l]});
    layernorm(q_, x_, L.ln1_w, L.ln1_b, n, E, eps_, h_);
    act_.k = E;
    xmx::quantize(q_, h_, n, act_, true);
    xmx::matmul(q_, act_, L.qkv, qkv_);
    {
      // biases, RoPE on Q and K (the first half of a head by the column, the second by the row: adjacent pairs, theta
      // over 32 dims), Q scaled, into vision_attn's tiles; rows past n zero
      const float *qkv = qkv_, *bq = L.b_qkv;
      const int32_t* rp = rp_;
      sycl::half *qt = qt_, *kt = kt_, *vt = vt_;
      q_.parallel_for(sycl::range<3>(size_t(R), size_t(H), size_t(D / 2)), [=](sycl::id<3> id) {
        const int64_t r = int64_t(id[0]), hh = int64_t(id[1]), i = int64_t(id[2]);
        const int64_t d0 = 2 * i, half = d0 / (D / 2), j = (d0 % (D / 2)) / 2;
        float q0 = 0, q1 = 0, k0 = 0, k1 = 0, v0 = 0, v1 = 0;
        if (r < n) {
          const float* row = qkv + r * 3 * E;
          const int64_t c = hh * D + d0;
          q0 = row[c] + bq[c], q1 = row[c + 1] + bq[c + 1];
          k0 = row[E + c] + bq[E + c], k1 = row[E + c + 1] + bq[E + c + 1];
          v0 = row[2 * E + c] + bq[2 * E + c], v1 = row[2 * E + c + 1] + bq[2 * E + c + 1];
          const float pos = float(rp[2 * r + half]);
          const float ang = pos * sycl::pow(theta, -2.0f * float(j) / float(D / 2));
          const float cs = sycl::cos(ang), sn = sycl::sin(ang);
          const float a0 = q0 * cs - q1 * sn, a1 = q0 * sn + q1 * cs, b0 = k0 * cs - k1 * sn, b1 = k0 * sn + k1 * cs;
          q0 = a0 * scale, q1 = a1 * scale, k0 = b0, k1 = b1;
        }
        const int64_t hq = hh * R * D;
        qt[hq + vattn::q_index(r, d0, D)] = sycl::half(q0), qt[hq + vattn::q_index(r, d0 + 1, D)] = sycl::half(q1);
        kt[hq + vattn::k_index(r, d0, D)] = sycl::half(k0), kt[hq + vattn::k_index(r, d0 + 1, D)] = sycl::half(k1);
        vt[hq + vattn::v_index(r, d0, D)] = sycl::half(v0), vt[hq + vattn::v_index(r, d0 + 1, D)] = sycl::half(v1);
      });
    }
    // attention into the activation tiles the output projection reads
    vattn::attend(q_, qt_, kt_, vt_, n, H, D, D, att_, act_.h, E);
    xmx::set_rows(act_, n, E, true);
    xmx::matmul(q_, act_, L.o, h_);
    residual(q_, x_, h_, L.b_o, L.ls1, n, E);
    layernorm(q_, x_, L.ln2_w, L.ln2_b, n, E, eps_, h_);
    act_.k = E;
    xmx::quantize(q_, h_, n, act_, true);
    xmx::matmul(q_, act_, L.up, wide_);
    bias_act(q_, wide_, L.b_up, n, n_ff_, true);
    act_.k = n_ff_;
    xmx::quantize(q_, wide_, n, act_, true);
    xmx::matmul(q_, act_, L.down, h_);
    residual(q_, x_, h_, L.b_down, L.ls2, n, E);
    if (host_weights_ && l + 2 < layers_.size()) lay[l + 2] = put_layer(l + 2, int(l % 2), {q_.ext_oneapi_submit_barrier()});
    if (dump) {
      q_.wait();
      dump("layer_out-" + std::to_string(l), x_, n, E);
    }
  }
  if (host_weights_) {
    copy_q_->memcpy(dmm1_.tiles, mm1_.tiles, wbytes(mm1_));
    e_mm = copy_q_->memcpy(dmm2_.tiles, mm2_.tiles, wbytes(mm2_));
  }
  // the CLS token dropped; each 2 x 2 block of patches one token: (top left, top right, bottom left, bottom right)
  {
    const float* x = x_;
    float* w = wide_;
    const int64_t tw = gw / 2;
    q_.parallel_for(sycl::range<2>(size_t(tokens), size_t(4 * E)), [=](sycl::id<2> id) {
      const int64_t t = int64_t(id[0]), k = int64_t(id[1]);
      const int64_t xm = t % tw, ym = t / tw, part = k / E, e = k % E;
      const int64_t px = 2 * xm + part % 2, py = 2 * ym + part / 2;
      w[t * 4 * E + k] = x[(1 + py * gw + px) * E + e];
    });
  }
  if (dump) {
    q_.wait();
    dump("pixel_shuffle", wide_, tokens, 4 * E);
  }
  act_.k = 4 * E;
  xmx::quantize(q_, wide_, tokens, act_, true);
  if (host_weights_) q_.ext_oneapi_submit_barrier({e_mm});
  xmx::matmul(q_, act_, mm1, qkv_);
  bias_act(q_, qkv_, nullptr, tokens, n_proj_, true);
  act_.k = n_proj_;
  xmx::quantize(q_, qkv_, tokens, act_, true);
  xmx::matmul(q_, act_, mm2, h_);
  std::vector<float> out(static_cast<size_t>(tokens * n_out_));
  q_.memcpy(out.data(), h_, out.size() * 4).wait();
  if (dump) dump("projected", h_, tokens, n_out_);
  return out;
}

}  // namespace eklow::model
