#include "model/vision_gemma4v_fast.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <string>

namespace eklow::model {

VisionGemma4V::VisionGemma4V(sycl::queue& q, const gguf::File& f, bool reference) : q_(q), reference_(reference) {
  if (f.at("general.architecture").as_string() != "clip" || !f.find("clip.vision.projector_type") ||
      f.at("clip.vision.projector_type").as_string() != "gemma4v")
    throw std::runtime_error(f.path() + ": not a Gemma 4 mmproj with the gemma4v vision projector");
  // clip.cpp: rope_theta 100, pooling kernel 3 unless the file says otherwise.
  rope_theta_ = 100.0f;
  const gguf::Value* merge = f.find("clip.vision.projector.scale_factor");
  merge_ = merge ? int(merge->as_int()) : 3;
  patch_ = int(f.at("clip.vision.patch_size").as_int());
  eps_ = float(f.at("clip.vision.attention.layer_norm_epsilon").as_float());
  n_embd_ = f.at("clip.vision.embedding_length").as_int();
  n_ff_ = f.at("clip.vision.feed_forward_length").as_int();
  n_head_ = f.at("clip.vision.attention.head_count").as_int();
  const gguf::Value* kv = f.find("clip.vision.attention.head_count_kv");
  n_head_kv_ = kv ? kv->as_int() : n_head_;
  const gguf::Value* dh = f.find("clip.vision.attention.key_length");
  d_head_ = dh ? dh->as_int() : n_embd_ / n_head_;
  const int64_t n_layer = f.at("clip.vision.block_count").as_int();
  for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"})
    for (const gguf::Value& v : f.at(key).as_array())
      if (v.as_float() != (key[18] == 'm' ? 0.0 : 1.0))
        throw std::runtime_error(f.path() + ": image_mean/image_std other than 0/1 are not supported");
  if (f.tensor("v.pre_ln.weight") || f.tensor("v.post_ln.weight") || f.tensor("v.blk.0.attn_qkv.weight"))
    throw std::runtime_error(f.path() + ": pre/post layer norm and fused qkv are not in the gemma4v graph");

  // The fast path takes the files it can (no biases, clamps or layer scales; 64- or 72-wide heads); others run the
  // reference kernels.
  if (!reference_ && load_fast(f)) return;
  reference_ = true;
  auto find = [&](const std::string& name) { return f.tensor(name); };
  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = find(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto upload = [&](const gguf::Tensor& t) {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory loading " + t.name);
    owned_.push_back(d);
    q_.memcpy(d, t.data, t.nbytes).wait();
    bytes_ += t.nbytes;
    return d;
  };
  auto matrix = [&](const gguf::Tensor& t) {
    return ref::Weight{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), upload(t), t.nbytes / t.ne[1]};
  };
  auto vec = [&](const std::string& name, bool required, int64_t len) -> const float* {
    const gguf::Tensor* t = find(name);
    if (!t && !required) return nullptr;
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    if (t->type != gguf::Type::F32 || int64_t(t->elements()) != len)
      throw std::runtime_error(f.path() + ": unexpected shape or type of " + name);
    return reinterpret_cast<const float*>(upload(*t));
  };
  // Gemma4ClippableLinear: clip.cpp reads NAME.input_max/input_min/output_max/output_min beside each
  // weight, an absent one standing for no limit.
  auto scalar = [&](const std::string& name, float dflt, bool& found) {
    const gguf::Tensor* t = find(name);
    if (!t) return dflt;
    found = true;
    if (t->type != gguf::Type::F32 || t->elements() != 1) throw std::runtime_error(f.path() + ": " + name + " is not an f32 scalar");
    return *reinterpret_cast<const float*>(t->data);
  };
  auto linear = [&](const std::string& weight, bool required) {
    Linear l;
    const gguf::Tensor* t = find(weight);
    if (!t && !required) {
      l.w.data = nullptr;
      return l;
    }
    l.w = matrix(need(weight));
    const std::string base = weight.substr(0, weight.size() - 6);  // ".weight"
    bool found = false;
    l.in_max = scalar(base + ".input_max", FLT_MAX, found);
    l.in_min = scalar(base + ".input_min", -FLT_MAX, found);
    l.out_max = scalar(base + ".output_max", FLT_MAX, found);
    l.out_min = scalar(base + ".output_min", -FLT_MAX, found);
    l.clamp = found;
    if (find(base + ".bias")) l.bias = vec(base + ".bias", true, l.w.n);
    return l;
  };

  const gguf::Tensor& pe = need("v.patch_embd.weight");
  if (find("v.patch_embd.bias")) throw std::runtime_error(f.path() + ": gemma4v has no patch bias");
  // [16][16][3][n_embd] in the file: n_embd rows of 768, the layout of a patch
  patch_embd_ = {pe.type, int64_t(pe.ne[0] * pe.ne[1] * pe.ne[2]), int64_t(pe.ne[3]), upload(pe), pe.nbytes / pe.ne[3]};
  n_in_ = patch_embd_.k;
  if (n_in_ != int64_t(3) * patch_ * patch_ || patch_embd_.n != n_embd_) throw std::runtime_error(f.path() + ": unexpected patch embedding shape");
  const gguf::Tensor& pos = need("v.position_embd.weight");
  if (pos.type != gguf::Type::F32 || int64_t(pos.ne[0]) != n_embd_ || pos.ne.size() < 3 || pos.ne[2] != 2)
    throw std::runtime_error(f.path() + ": unexpected position table shape");
  n_pos_ = int64_t(pos.ne[1]);
  pos_ = reinterpret_cast<const float*>(upload(pos));
  std_bias_ = vec("v.std_bias", false, n_embd_);
  std_scale_ = vec("v.std_scale", false, n_embd_);
  {
    Linear p = linear("mm.input_projection.weight", true);
    if (p.bias || p.w.k != n_embd_) throw std::runtime_error(f.path() + ": unexpected input projection");
    proj_ = p.w;
    proj_lin_ = p;
  }
  n_out_ = proj_.n;
  if (n_head_kv_ <= 0 || n_head_ % n_head_kv_ || d_head_ * n_head_ != n_embd_) throw std::runtime_error(f.path() + ": unexpected head layout");

  layers_.resize(size_t(n_layer));
  for (int64_t il = 0; il < n_layer; ++il) {
    const std::string b = "v.blk." + std::to_string(il) + ".";
    Layer& L = layers_[size_t(il)];
    L.q = linear(b + "attn_q.weight", true);
    L.k = linear(b + "attn_k.weight", true);
    L.v = linear(b + "attn_v.weight", true);
    L.o = linear(b + "attn_out.weight", true);
    L.up = linear(b + "ffn_up.weight", true);
    L.gate = linear(b + "ffn_gate.weight", true);
    L.down = linear(b + "ffn_down.weight", true);
    L.ln1 = vec(b + "ln1.weight", true, n_embd_);
    L.ln2 = vec(b + "ln2.weight", true, n_embd_);
    L.q_norm = vec(b + "attn_q_norm.weight", false, d_head_);
    L.k_norm = vec(b + "attn_k_norm.weight", false, d_head_);
    L.attn_post = vec(b + "attn_post_norm.weight", false, n_embd_);
    L.ffn_post = vec(b + "ffn_post_norm.weight", false, n_embd_);
    L.ls1 = vec(b + "ls1.weight", false, n_embd_);
    L.ls2 = vec(b + "ls2.weight", false, n_embd_);
    L.ls_out = vec(b + "out_scale.weight", false, n_embd_);
    if (L.q.w.n != n_head_ * d_head_ || L.k.w.n != n_head_kv_ * d_head_ || L.v.w.n != L.k.w.n || L.o.w.k != n_head_ * d_head_ ||
        L.up.w.n != n_ff_ || L.gate.w.n != n_ff_ || L.down.w.k != n_ff_ || L.down.w.n != n_embd_ || L.up.w.k != n_embd_)
      throw std::runtime_error(f.path() + ": unexpected shapes in layer " + std::to_string(il));
  }
  if (d_head_ % 4) throw std::runtime_error(f.path() + ": head size not divisible by 4 for 2D RoPE");
}

image::Spec VisionGemma4V::spec(int tokens) const {
  image::Spec s;
  s.patch = patch_ * merge_;
  s.cut = patch_;
  s.min_patches = 70;
  s.max_patches = tokens > 0 ? tokens : 1120;
  if (s.max_patches < s.min_patches) s.min_patches = s.max_patches;
  return s;
}

void VisionGemma4V::mm(const Linear& l, const float* x, int64_t rows, float* out, float* scratch) {
  const float* in = x;
  if (l.clamp) {
    ref::clamp(q_, x, l.in_min, l.in_max, rows * l.w.k, scratch);
    in = scratch;
  }
  if (bf16_act_ && l.w.type == gguf::Type::BF16) {
    ref::round_bf16(q_, in, rows * l.w.k, scratch);
    in = scratch;
  }
  ref::matmul(q_, l.w, in, rows, out);
  if (l.bias) ref::add_rows(q_, out, l.bias, rows, l.w.n);
  if (l.clamp) ref::clamp(q_, out, l.out_min, l.out_max, rows * l.w.n, out);
}

DeviceRows VisionGemma4V::encode(const image::Patches& p) {
  if (p.dim != n_in_) throw std::invalid_argument("vision encoder: patches of the wrong size");
  if (p.cols <= 0 || p.rows <= 0 || p.cols % merge_ || p.rows % merge_)
    throw std::invalid_argument("vision encoder: the patch grid must be a multiple of the pooling kernel");
  if (p.cols > n_pos_ || p.rows > n_pos_) throw std::invalid_argument("vision encoder: the image is too wide or too tall");
  if (!reference_) return encode_fast(p);
  const int64_t n = int64_t(p.cols) * p.rows, E = n_embd_, F = n_ff_, cols = p.cols, P = n_pos_;
  const int64_t H = n_head_, HK = n_head_kv_, D = d_head_;
  const int64_t ox = p.cols / merge_, oy = p.rows / merge_, n_out = ox * oy;
  std::lock_guard lock(mu_);
  const sycl::context ctx = q_.get_context();
  auto alloc = [&](int64_t floats) {
    float* d = sycl::malloc_device<float>(size_t(std::max<int64_t>(floats, 1)), q_);
    if (!d) throw std::runtime_error("vision encoder: out of device memory");
    return std::shared_ptr<float>(d, [ctx](float* x) { sycl::free(x, ctx); });
  };
  auto tap = [&](std::string_view name, int il, const float* dev, int64_t rows, int64_t dim) {
    if (!tap_) return;
    q_.wait();
    tap_(il >= 0 ? std::string(name) + "-" + std::to_string(il) : std::string(name), dev, rows, dim);
  };

  // RoPE angles per patch: x by the column for the first half of a head's rotated dims, y by the row.
  const int64_t rot = D / 2, half = rot / 2;
  const ref::RopeTables rt = ref::rope_tables(std::max(p.cols, p.rows), rot, double(rope_theta_), nullptr);
  std::vector<float> hc(size_t(n * half)), hs(size_t(n * half)), vc(size_t(n * half)), vs(size_t(n * half));
  for (int64_t i = 0; i < n; ++i) {
    const int64_t col = i % cols, row = i / cols;
    std::copy_n(&rt.cos[size_t(col * half)], half, &hc[size_t(i * half)]);
    std::copy_n(&rt.sin[size_t(col * half)], half, &hs[size_t(i * half)]);
    std::copy_n(&rt.cos[size_t(row * half)], half, &vc[size_t(i * half)]);
    std::copy_n(&rt.sin[size_t(row * half)], half, &vs[size_t(i * half)]);
  }
  auto tables = alloc(4 * n * half);
  float *cos_x = tables.get(), *sin_x = cos_x + n * half, *cos_y = sin_x + n * half, *sin_y = cos_y + n * half;
  q_.memcpy(cos_x, hc.data(), hc.size() * sizeof(float));
  q_.memcpy(sin_x, hs.data(), hs.size() * sizeof(float));
  q_.memcpy(cos_y, vc.data(), vc.size() * sizeof(float));
  q_.memcpy(sin_y, vs.data(), vs.size() * sizeof(float));

  auto patches = alloc(n * n_in_), x = alloc(n * E), h = alloc(n * E), qb = alloc(n * H * D), kb = alloc(n * HK * D),
       vb = alloc(n * HK * D), ab = alloc(n * H * D), up = alloc(n * F), gate = alloc(n * F), scratch = alloc(n * std::max(E, F)),
       out = alloc(n_out * n_out_);
  q_.memcpy(patches.get(), p.data.data(), p.data.size() * sizeof(float));
  {
    // patches = 2 * (patches - 0.5)
    float* d = patches.get();
    q_.parallel_for(sycl::range<1>(size_t(n * n_in_)), [=](sycl::id<1> id) { d[id[0]] = d[id[0]] * 2.0f - 1.0f; });
  }
  ref::matmul(q_, patch_embd_, patches.get(), n, x.get());
  tap("inp", -1, x.get(), n, E);
  {
    float* xp = x.get();
    const float* pos = pos_;
    q_.parallel_for(sycl::range<2>(size_t(n), size_t(E)), [=](sycl::id<2> id) {
       const int64_t r = int64_t(id[0]), e = int64_t(id[1]);
       xp[r * E + e] = (xp[r * E + e] + pos[(r % cols) * E + e]) + pos[(P + r / cols) * E + e];
     });
  }
  tap("pos_embd", -1, x.get(), n, E);

  for (size_t il = 0; il < layers_.size(); ++il) {
    const Layer& L = layers_[il];
    const int li = int(il);
    // attention
    ref::rmsnorm(q_, x.get(), L.ln1, n, E, eps_, h.get());
    tap("layer_inp_normed", li, h.get(), n, E);
    mm(L.q, h.get(), n, qb.get(), scratch.get());
    mm(L.k, h.get(), n, kb.get(), scratch.get());
    mm(L.v, h.get(), n, vb.get(), scratch.get());
    if (L.q_norm) ref::rmsnorm(q_, qb.get(), L.q_norm, n * H, D, eps_, qb.get());
    if (L.k_norm) ref::rmsnorm(q_, kb.get(), L.k_norm, n * HK, D, eps_, kb.get());
    tap("Qcur_norm", li, qb.get(), n, H * D);
    // 2D RoPE: dims [0, D/2) turn by the column, [D/2, D) by the row (both NeoX within their half)
    ref::rope_neox(q_, qb.get(), n, H, D, rot, cos_x, sin_x);
    ref::rope_neox(q_, qb.get() + rot, n, H, D, rot, cos_y, sin_y);
    ref::rope_neox(q_, kb.get(), n, HK, D, rot, cos_x, sin_x);
    ref::rope_neox(q_, kb.get() + rot, n, HK, D, rot, cos_y, sin_y);
    tap("Qcur_pos", li, qb.get(), n, H * D);
    tap("Kcur_pos", li, kb.get(), n, HK * D);
    ref::rmsnorm(q_, vb.get(), nullptr, n * HK, D, eps_, vb.get());
    tap("Vcur_normed", li, vb.get(), n, HK * D);
    ref::attention_dense(q_, qb.get(), kb.get(), vb.get(), n, H, HK, D, 1.0f, ab.get());
    tap("kqv_out", li, ab.get(), n, H * D);
    mm(L.o, ab.get(), n, h.get(), scratch.get());
    tap("attn_out", li, h.get(), n, E);
    if (L.ls1) ref::mul_rows(q_, h.get(), L.ls1, n, E);
    if (L.attn_post) ref::rmsnorm(q_, h.get(), L.attn_post, n, E, eps_, h.get());
    ref::add(q_, h.get(), x.get(), n * E, x.get());
    // ffn
    ref::rmsnorm(q_, x.get(), L.ln2, n, E, eps_, h.get());
    mm(L.up, h.get(), n, up.get(), scratch.get());
    mm(L.gate, h.get(), n, gate.get(), scratch.get());
    ref::gelu_quick_mul(q_, gate.get(), up.get(), n * F, up.get());
    mm(L.down, up.get(), n, h.get(), scratch.get());
    tap("ffn_out", li, h.get(), n, E);
    if (L.ffn_post) ref::rmsnorm(q_, h.get(), L.ffn_post, n, E, eps_, h.get());
    if (L.ls2) ref::mul_rows(q_, h.get(), L.ls2, n, E);
    ref::add(q_, h.get(), x.get(), n * E, x.get());
    if (L.ls_out) ref::mul_rows(q_, x.get(), L.ls_out, n, E);
    tap("layer_out", li, x.get(), n, E);
  }

  // Gemma4VisionPooler: ksize x ksize average over the patch grid, then * sqrt(n_embd)
  {
    const float* xp = x.get();
    float* pooled = h.get();
    const int k = merge_;
    const float s = std::sqrt(float(E));
    q_.parallel_for(sycl::range<2>(size_t(n_out), size_t(E)), [=](sycl::id<2> id) {
       const int64_t o = int64_t(id[0]), e = int64_t(id[1]);
       const int64_t gx = o % ox, gy = o / ox;
       float acc = 0;
       for (int iy = 0; iy < k; ++iy)
         for (int ix = 0; ix < k; ++ix) acc += xp[((gy * k + iy) * cols + gx * k + ix) * E + e];
       pooled[o * E + e] = acc / float(k * k) * s;
     });
  }
  tap("pooled", -1, h.get(), n_out, E);
  if (std_bias_ && std_scale_) {
    float* hp = h.get();
    const float *sb = std_bias_, *ss = std_scale_;
    q_.parallel_for(sycl::range<2>(size_t(n_out), size_t(E)), [=](sycl::id<2> id) {
       const int64_t e = int64_t(id[1]);
       hp[int64_t(id[0]) * E + e] = (hp[int64_t(id[0]) * E + e] - sb[e]) * ss[e];
     });
    tap("std_scaled", -1, h.get(), n_out, E);
  }
  // Gemma4MultimodalEmbedder: embedding_pre_projection_norm, then the projection
  ref::rmsnorm(q_, h.get(), nullptr, n_out, E, eps_, h.get());
  mm(proj_lin_, h.get(), n_out, out.get(), scratch.get());
  tap("projected", -1, out.get(), n_out, n_out_);
  q_.wait();  // the buffers are freed on return
  return {out, n_out, n_out_};
}

}  // namespace eklow::model
