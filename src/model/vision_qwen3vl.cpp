#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "model/vision_qwen3vl_fast.hpp"

namespace eklow::model {

VisionQwen3VL::VisionQwen3VL(sycl::queue& q, const gguf::File& f, Options o) : q_(q), reference_(o.reference), resident_(o.resident) {
  if (f.at("general.architecture").as_string() != "clip" || !f.find("clip.projector_type") || f.at("clip.projector_type").as_string() != "qwen3vl_merger")
    throw std::runtime_error(f.path() + ": not a Qwen3-VL mmproj with the qwen3vl_merger vision projector");
  const gguf::Value* merge = f.find("clip.vision.spatial_merge_size");
  merge_ = merge ? int(merge->as_int()) : 2;
  if (merge_ != 2) throw std::runtime_error(f.path() + ": only a spatial merge of 2 is supported");
  patch_ = int(f.at("clip.vision.patch_size").as_int());
  eps_ = float(f.at("clip.vision.attention.layer_norm_epsilon").as_float());
  n_embd_ = f.at("clip.vision.embedding_length").as_int();
  n_ff_ = f.at("clip.vision.feed_forward_length").as_int();
  n_head_ = f.at("clip.vision.attention.head_count").as_int();
  d_head_ = n_embd_ / n_head_;
  const int64_t n_layer = f.at("clip.vision.block_count").as_int();
  if (const gguf::Value* ds = f.find("clip.vision.is_deepstack_layers"))
    for (const gguf::Value& v : ds->as_array())
      if (v.kind == gguf::Value::Kind::Bool ? std::get<bool>(v.v) : v.as_int() != 0)
        throw std::runtime_error(f.path() + ": deepstack layers are not supported");
  const gguf::Value *mean = f.find("clip.vision.image_mean"), *sd = f.find("clip.vision.image_std");
  if (mean && sd && mean->as_array().size() == 3 && sd->as_array().size() == 3)
    for (int c = 0; c < 3; ++c) mean_[c] = float(mean->as_array()[size_t(c)].as_float()), std_[c] = float(sd->as_array()[size_t(c)].as_float());
  if (d_head_ * n_head_ != n_embd_ || d_head_ % 4) throw std::runtime_error(f.path() + ": unexpected head layout");

  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  const gguf::Tensor& pos = need("v.position_embd.weight");
  if (pos.type != gguf::Type::F32 || int64_t(pos.ne[0]) != n_embd_) throw std::runtime_error(f.path() + ": unexpected position table shape");
  n_pos_ = int64_t(pos.ne[1]);
  const int side = int(std::lround(std::sqrt(double(n_pos_))));
  if (int64_t(side) * side != n_pos_) throw std::runtime_error(f.path() + ": the position table is not square");
  n_hidden_ = need("mm.0.weight").ne[1];
  n_out_ = need("mm.2.weight").ne[1];

  if (!reference_ && load_fast(f)) return;
  reference_ = true;
  resident_ = false;

  auto upload = [&](const gguf::Tensor& t) {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory loading " + t.name);
    owned_.push_back(d);
    q_.memcpy(d, t.data, t.nbytes).wait();
    bytes_ += t.nbytes;
    return d;
  };
  auto matrix = [&](const gguf::Tensor& t, int64_t k, int64_t n) {
    if (int64_t(t.elements()) != k * n) throw std::runtime_error(f.path() + ": unexpected shape of " + t.name);
    return ref::Weight{t.type, k, n, upload(t), t.nbytes / uint64_t(n)};
  };
  auto vec = [&](const std::string& name, int64_t len) -> const float* {
    const gguf::Tensor& t = need(name);
    if (t.type != gguf::Type::F32 || int64_t(t.elements()) != len) throw std::runtime_error(f.path() + ": unexpected shape or type of " + name);
    return reinterpret_cast<const float*>(upload(t));
  };
  auto linear = [&](const std::string& base, int64_t k, int64_t n) {
    Linear l;
    l.w = matrix(need(base + ".weight"), k, n);
    l.bias = vec(base + ".bias", n);
    return l;
  };
  n_in_ = int64_t(3) * patch_ * patch_;
  patch0_ = matrix(need("v.patch_embd.weight"), n_in_, n_embd_);
  patch1_ = matrix(need("v.patch_embd.weight.1"), n_in_, n_embd_);
  patch_bias_ = vec("v.patch_embd.bias", n_embd_);
  pos_ = reinterpret_cast<const float*>(upload(pos));
  post_w_ = vec("v.post_ln.weight", n_embd_);
  post_b_ = vec("v.post_ln.bias", n_embd_);
  {
    Linear l0 = linear("mm.0", n_embd_ * 4, n_hidden_), l2 = linear("mm.2", n_hidden_, n_out_);
    mm0_ = l0.w, mm0_b_ = l0.bias, mm2_ = l2.w, mm2_b_ = l2.bias;
  }
  layers_.resize(size_t(n_layer));
  for (int64_t il = 0; il < n_layer; ++il) {
    const std::string b = "v.blk." + std::to_string(il) + ".";
    Layer& L = layers_[size_t(il)];
    L.qkv = linear(b + "attn_qkv", n_embd_, 3 * n_embd_);
    L.o = linear(b + "attn_out", n_embd_, n_embd_);
    L.up = linear(b + "ffn_up", n_embd_, n_ff_);
    L.down = linear(b + "ffn_down", n_ff_, n_embd_);
    L.ln1_w = vec(b + "ln1.weight", n_embd_);
    L.ln1_b = vec(b + "ln1.bias", n_embd_);
    L.ln2_w = vec(b + "ln2.weight", n_embd_);
    L.ln2_b = vec(b + "ln2.bias", n_embd_);
  }
}

image::QwenSpec VisionQwen3VL::spec(int max_tokens, int min_tokens) const {
  image::QwenSpec s;
  s.patch = patch_;
  s.merge = merge_;
  s.min_tokens = std::max(1, std::min(min_tokens, max_tokens));
  s.max_tokens = std::max(1, max_tokens);
  for (int c = 0; c < 3; ++c) s.mean[c] = mean_[c], s.stddev[c] = std_[c];
  return s;
}

QwenImageRows VisionQwen3VL::encode(std::span<const uint8_t> bytes, const image::QwenSpec& spec) {
  return encode(image::prepare_qwen3vl(image::decode(bytes), spec));
}

QwenImageRows VisionQwen3VL::encode(const image::Rgb& pic) {
  const int unit = patch_ * merge_;
  if (pic.width <= 0 || pic.height <= 0 || pic.width % unit || pic.height % unit)
    throw std::invalid_argument("vision encoder: the picture's sides must be multiples of 32 pixels");
  if (int64_t(pic.width / patch_) * (pic.height / patch_) > 65536) throw std::invalid_argument("vision encoder: the image is too large (more than 16,384 tokens)");
  if (pic.pixels.size() != size_t(pic.width) * size_t(pic.height) * 3) throw std::invalid_argument("vision encoder: bad picture");
  return reference_ ? encode_ref(pic) : encode_fast(pic);
}

QwenImageRows VisionQwen3VL::encode_ref(const image::Rgb& pic) {
  image::QwenSpec s = spec();
  const image::Patches p = image::patchify_merged(pic, s);
  const int64_t n = int64_t(p.cols) * p.rows, E = n_embd_, F = n_ff_, H = n_head_, D = d_head_;
  const int cols = p.cols, rows = p.rows, side = int(std::lround(std::sqrt(double(n_pos_))));
  const int64_t n_out = n / 4, oxm = cols / 2;
  std::lock_guard lock(mu_);
  const sycl::context ctx = q_.get_context();
  auto alloc = [&](int64_t floats) {
    float* d = sycl::malloc_device<float>(size_t(std::max<int64_t>(floats, 1)), q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory");
    return std::shared_ptr<float>(d, [ctx](float* x) { sycl::free(x, ctx); });
  };
  auto tap = [&](const char* name, int il, const float* dev, int64_t r, int64_t d) {
    if (!tap_) return;
    q_.wait();
    tap_(il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name), dev, r, d);
  };

  // M-RoPE angles (vision): pairs 0..17 by the patch's row, 18..35 by its column, pair i = angle pos * 10000^(-2 i' / 36)
  const int64_t rot = D / 2, half = rot / 2;  // 36, 18
  const ref::RopeTables rt = ref::rope_tables(std::max(cols, rows), rot, double(rope_theta_), nullptr);
  auto tables = alloc(2 * int64_t(std::max(cols, rows)) * half);
  float *cos_t = tables.get(), *sin_t = cos_t + int64_t(std::max(cols, rows)) * half;
  q_.memcpy(cos_t, rt.cos.data(), rt.cos.size() * 4);
  q_.memcpy(sin_t, rt.sin.data(), rt.sin.size() * 4);

  auto px = alloc(n * n_in_), x = alloc(n * E), h = alloc(n * E), qkv = alloc(n * 3 * E), qb = alloc(n * E), kb = alloc(n * E),
       vb = alloc(n * E), ab = alloc(n * E), up = alloc(n * F), t2 = alloc(n * E), hid = alloc(n_out * n_hidden_), out = alloc(n_out * n_out_);
  q_.memcpy(px.get(), p.data.data(), p.data.size() * 4);
  ref::matmul(q_, patch0_, px.get(), n, x.get());
  ref::matmul(q_, patch1_, px.get(), n, h.get());
  ref::add(q_, x.get(), h.get(), n * E, x.get());
  ref::add_rows(q_, x.get(), patch_bias_, n, E);
  tap("patch_bias", -1, x.get(), n, E);
  {
    float* xp = x.get();
    const float* tbl = pos_;
    q_.parallel_for(sycl::range<2>(size_t(n), size_t(E)), [=](sycl::id<2> id) {
       const int64_t r = int64_t(id[0]), e = int64_t(id[1]);
       const int64_t b = r / 4, w = r % 4;
       const int iy = int(2 * (b / oxm) + w / 2), ix = int(2 * (b % oxm) + w % 2);
       const PosLerp l = pos_lerp(iy, ix, rows, cols, side);
       const float v = tbl[int64_t(l.i00) * E + e] * l.w00 + tbl[int64_t(l.i01) * E + e] * l.w01 + tbl[int64_t(l.i10) * E + e] * l.w10 +
                       tbl[int64_t(l.i11) * E + e] * l.w11;
       xp[r * E + e] += v;
     });
  }
  tap("inp_pos_emb", -1, x.get(), n, E);

  const float eps = eps_, scale = 1.0f / std::sqrt(float(D));
  auto layernorm = [&](const float* in, const float* w, const float* b, float* o) {
    q_.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
       const int64_t r = int64_t(id[0]);
       float s = 0;
       for (int64_t c = 0; c < E; ++c) s += in[r * E + c];
       const float mean = s / float(E);
       float v = 0;
       for (int64_t c = 0; c < E; ++c) v += (in[r * E + c] - mean) * (in[r * E + c] - mean);
       const float inv = 1.0f / sycl::sqrt(v / float(E) + eps);
       for (int64_t c = 0; c < E; ++c) o[r * E + c] = (in[r * E + c] - mean) * inv * w[c] + b[c];
     });
  };
  for (size_t il = 0; il < layers_.size(); ++il) {
    const Layer& L = layers_[il];
    const int li = int(il);
    layernorm(x.get(), L.ln1_w, L.ln1_b, h.get());
    tap("ln1", li, h.get(), n, E);
    ref::matmul(q_, L.qkv.w, h.get(), n, qkv.get());
    ref::add_rows(q_, qkv.get(), L.qkv.bias, n, 3 * E);
    {
      // split, and turn q and k: pair (i, i + 36) of a head by the row (i < 18) or the column (i >= 18)
      const float *src = qkv.get(), *ct = cos_t, *st = sin_t;
      float *qo = qb.get(), *ko = kb.get(), *vo = vb.get();
      q_.parallel_for(sycl::range<2>(size_t(n), size_t(H)), [=](sycl::id<2> id) {
         const int64_t r = int64_t(id[0]), hd = int64_t(id[1]);
         const int64_t b = r / 4, w = r % 4;
         const int64_t py = 2 * (b / oxm) + w / 2, px_ = 2 * (b % oxm) + w % 2;
         for (int part = 0; part < 2; ++part) {
           const float* s = src + r * 3 * E + part * E + hd * D;
           float* d = (part ? ko : qo) + r * E + hd * D;
           for (int64_t i = 0; i < rot; ++i) {
             const int64_t pos = i < half ? py : px_, ti = (pos * half + (i % half));
             const float c = ct[ti], sn = st[ti];
             const float a = s[i], bb = s[i + rot];
             d[i] = a * c - bb * sn;
             d[i + rot] = a * sn + bb * c;
           }
         }
         const float* s = src + r * 3 * E + 2 * E + hd * D;
         for (int64_t i = 0; i < D; ++i) vo[r * E + hd * D + i] = s[i];
       });
    }
    tap("Qcur_rope", li, qb.get(), n, E);
    tap("Kcur_rope", li, kb.get(), n, E);
    ref::attention_dense(q_, qb.get(), kb.get(), vb.get(), n, H, H, D, scale, ab.get());
    tap("kqv_out", li, ab.get(), n, E);
    ref::matmul(q_, L.o.w, ab.get(), n, h.get());
    ref::add_rows(q_, h.get(), L.o.bias, n, E);
    tap("attn_out", li, h.get(), n, E);
    ref::add(q_, h.get(), x.get(), n * E, x.get());
    tap("ffn_inp", li, x.get(), n, E);
    layernorm(x.get(), L.ln2_w, L.ln2_b, h.get());
    ref::matmul(q_, L.up.w, h.get(), n, up.get());
    ref::add_rows(q_, up.get(), L.up.bias, n, F);
    ref::gelu(q_, up.get(), n * F, up.get());
    ref::matmul(q_, L.down.w, up.get(), n, t2.get());
    ref::add_rows(q_, t2.get(), L.down.bias, n, E);
    tap("ffn_out", li, t2.get(), n, E);
    ref::add(q_, t2.get(), x.get(), n * E, x.get());
    tap("layer_out", li, x.get(), n, E);
  }
  layernorm(x.get(), post_w_, post_b_, h.get());
  tap("post_ln", -1, h.get(), n, E);
  // [n][1152] read as [n / 4][4608]
  ref::matmul(q_, mm0_, h.get(), n_out, hid.get());
  ref::add_rows(q_, hid.get(), mm0_b_, n_out, n_hidden_);
  ref::gelu(q_, hid.get(), n_out * n_hidden_, hid.get());
  ref::matmul(q_, mm2_, hid.get(), n_out, out.get());
  ref::add_rows(q_, out.get(), mm2_b_, n_out, n_out_);
  tap("embeddings", -1, out.get(), n_out, n_out_);
  q_.wait();
  return {{out, n_out, n_out_}, int(oxm), rows / 2};
}

}  // namespace eklow::model
