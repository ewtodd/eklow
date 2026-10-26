#include "model/vision_gemma4.hpp"

#include <stdexcept>
#include <string>

namespace eklow::model {

VisionGemma4::VisionGemma4(sycl::queue& q, const gguf::File& f) : q_(q) {
  if (f.at("general.architecture").as_string() != "clip" || !f.find("clip.vision.projector_type") ||
      f.at("clip.vision.projector_type").as_string() != "gemma4uv")
    throw std::runtime_error(f.path() + ": not a Gemma 4 mmproj with the gemma4uv vision projector");
  // clip.cpp: the unified projector's patches are patch_size times the pooling kernel on a side,
  // 3 unless the file says otherwise.
  const gguf::Value* merge = f.find("clip.vision.projector.scale_factor");
  spec_.patch = int(f.at("clip.vision.patch_size").as_int() * (merge ? merge->as_int() : 3));
  eps_ = float(f.at("clip.vision.attention.layer_norm_epsilon").as_float());

  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto upload = [&](const gguf::Tensor& t) {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q_);
    if (!d) throw std::runtime_error("vision embedder: out of device memory loading " + t.name);
    owned_.push_back(d);
    q_.memcpy(d, t.data, t.nbytes).wait();
    bytes_ += t.nbytes;
    return d;
  };
  auto matrix = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    return ref::Weight{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), upload(t), t.nbytes / t.ne[1]};
  };
  auto vec = [&](const std::string& name) { return reinterpret_cast<const float*>(upload(need(name))); };
  patch_embd_ = matrix("v.patch_embd.weight");
  proj_ = matrix("mm.input_projection.weight");
  patch_bias_ = vec("v.patch_embd.bias");
  norm1_w_ = vec("v.patch_norm.1.weight");
  norm1_b_ = vec("v.patch_norm.1.bias");
  norm2_w_ = vec("v.patch_norm.2.weight");
  norm2_b_ = vec("v.patch_norm.2.bias");
  norm3_w_ = vec("v.patch_norm.3.weight");
  norm3_b_ = vec("v.patch_norm.3.bias");
  const gguf::Tensor& pos = need("v.position_embd.weight");
  pos_ = vec("v.position_embd.weight");
  n_in_ = patch_embd_.k;
  n_embd_ = patch_embd_.n;
  n_out_ = proj_.n;
  n_pos_ = int64_t(pos.ne[1]);
  if (n_in_ != int64_t(3) * spec_.patch * spec_.patch || proj_.k != n_embd_ || int64_t(pos.ne[0]) != n_embd_ || pos.ne[2] != 2 ||
      pos.type != gguf::Type::F32)
    throw std::runtime_error(f.path() + ": unexpected vision tensor shapes");
}

VisionGemma4::~VisionGemma4() {
  for (void* p : owned_) sycl::free(p, q_);
}

DeviceRows VisionGemma4::encode(const image::Patches& p) {
  if (p.dim != n_in_) throw std::invalid_argument("vision embedder: patches of the wrong size");
  if (p.cols > n_pos_ || p.rows > n_pos_) throw std::invalid_argument("vision embedder: the image is too wide or too tall");
  const int64_t n = int64_t(p.cols) * p.rows, E = n_embd_, cols = p.cols, P = n_pos_;
  std::lock_guard lock(mu_);
  const sycl::context ctx = q_.get_context();
  auto alloc = [&](int64_t floats) {
    float* d = sycl::malloc_device<float>(size_t(floats), q_);
    if (!d) throw std::runtime_error("vision embedder: out of device memory");
    return std::shared_ptr<float>(d, [ctx](float* x) { sycl::free(x, ctx); });
  };
  auto patches = alloc(n * n_in_), x = alloc(n * E), out = alloc(n * n_out_);
  q_.memcpy(patches.get(), p.data.data(), p.data.size() * sizeof(float)).wait();
  ref::layernorm(q_, patches.get(), norm1_w_, norm1_b_, n, n_in_, 1e-5f, patches.get());
  ref::matmul(q_, patch_embd_, patches.get(), n, x.get());
  ref::add_rows(q_, x.get(), patch_bias_, n, E);
  ref::layernorm(q_, x.get(), norm2_w_, norm2_b_, n, E, 1e-5f, x.get());
  float* xp = x.get();
  const float* pos = pos_;
  q_.parallel_for(sycl::range<2>(size_t(n), size_t(E)), [=](sycl::id<2> id) {
     const int64_t r = int64_t(id[0]), e = int64_t(id[1]);
     xp[r * E + e] = (xp[r * E + e] + pos[(r % cols) * E + e]) + pos[(P + r / cols) * E + e];
   });
  ref::layernorm(q_, xp, norm3_w_, norm3_b_, n, E, 1e-5f, xp);
  ref::rmsnorm(q_, xp, nullptr, n, E, eps_, xp);
  ref::matmul(q_, proj_, xp, n, out.get());
  q_.wait();  // patches and x are freed on return
  return {out, n, n_out_};
}

}  // namespace eklow::model
