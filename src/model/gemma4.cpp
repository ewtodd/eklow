#include "model/gemma4.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace eklow::model {

namespace {

int64_t get_int(const gguf::File& f, const std::string& key) { return f.at(key).as_int(); }
float get_float(const gguf::File& f, const std::string& key) { return float(f.at(key).as_float()); }

[[noreturn]] void unsupported(const gguf::File& f, const std::string& what) {
  throw std::runtime_error(f.path() + ": not a Gemma 4 this engine runs: " + what);
}

}  // namespace

Gemma4Config Gemma4Config::from_gguf(const gguf::File& f) {
  if (f.at("general.architecture").as_string() != "gemma4") unsupported(f, "architecture is not gemma4");
  Gemma4Config c{};
  const std::string p = "gemma4.";
  c.n_layer = int(get_int(f, p + "block_count"));
  c.n_embd = int(get_int(f, p + "embedding_length"));
  c.n_head = int(get_int(f, p + "attention.head_count"));
  c.window = int(get_int(f, p + "attention.sliding_window"));
  c.head_dim_global = int(get_int(f, p + "attention.key_length"));
  c.head_dim_sliding = int(get_int(f, p + "attention.key_length_swa"));
  c.rope_base_global = get_float(f, p + "rope.freq_base");
  c.rope_base_sliding = get_float(f, p + "rope.freq_base_swa");
  c.eps = get_float(f, p + "attention.layer_norm_rms_epsilon");
  c.softcap = f.find(p + "final_logit_softcapping") ? get_float(f, p + "final_logit_softcapping") : 0.0f;

  const gguf::Value& ff = f.at(p + "feed_forward_length");
  if (ff.kind == gguf::Value::Kind::Array) unsupported(f, "per-layer feed-forward lengths");
  c.n_ff = int(ff.as_int());
  for (const auto& v : f.at(p + "attention.sliding_window_pattern").as_array()) c.sliding.push_back(std::get<bool>(v.v));
  for (const auto& v : f.at(p + "attention.head_count_kv").as_array()) c.n_head_kv.push_back(int(v.as_int()));
  if (int(c.sliding.size()) != c.n_layer || int(c.n_head_kv.size()) != c.n_layer) unsupported(f, "per-layer arrays of the wrong length");

  if (auto* v = f.find(p + "embedding_length_per_layer_input"); v && v->as_int() != 0) unsupported(f, "per-layer embeddings (E2B/E4B)");
  if (auto* v = f.find(p + "attention.shared_kv_layers"); v && v->as_int() != 0) unsupported(f, "shared KV layers");
  if (f.find(p + "expert_count")) unsupported(f, "mixture of experts (26B A4B)");
  if (get_int(f, p + "attention.value_length") != c.head_dim_global ||
      get_int(f, p + "attention.value_length_swa") != c.head_dim_sliding)
    unsupported(f, "value heads differ from key heads");
  if (get_int(f, p + "rope.dimension_count") != c.head_dim_global ||
      get_int(f, p + "rope.dimension_count_swa") != c.head_dim_sliding)
    unsupported(f, "partial rotary dimensions");
  const gguf::Tensor* emb = f.tensor("token_embd.weight");
  if (!emb || emb->ne.size() != 2) unsupported(f, "no token_embd.weight");
  c.n_vocab = int(emb->ne[1]);
  if (f.tensor("output.weight")) unsupported(f, "an untied output.weight");
  return c;
}

Gemma4Weights::Gemma4Weights(Gemma4Weights&& o) noexcept
    : tok_embd(o.tok_embd), output_norm(o.output_norm), rope_freqs(std::move(o.rope_freqs)), layers(std::move(o.layers)),
      bytes(o.bytes), q_(o.q_), allocations_(std::move(o.allocations_)) {
  o.allocations_.clear();
}

Gemma4Weights::~Gemma4Weights() {
  for (void* p : allocations_) sycl::free(p, *q_);
}

Gemma4Weights Gemma4Weights::upload(const gguf::File& f, const Gemma4Config& c, sycl::queue& q) {
  Gemma4Weights w(q);
  auto put = [&](const gguf::Tensor& t) -> const uint8_t* {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q);
    if (!d) throw std::runtime_error("out of device memory uploading " + t.name);
    w.allocations_.push_back(d);
    q.memcpy(d, t.data, t.nbytes).wait();
    w.bytes += t.nbytes;
    return d;
  };
  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto matrix = [&](const std::string& name, int64_t k, int64_t n) {
    const gguf::Tensor& t = need(name);
    if (t.ne.size() != 2 || int64_t(t.ne[0]) != k || int64_t(t.ne[1]) != n)
      throw std::runtime_error(f.path() + ": " + name + " has an unexpected shape");
    return ref::Weight{t.type, k, n, put(t), t.nbytes / uint64_t(n)};
  };
  auto vec = [&](const std::string& name, int64_t n) {
    const gguf::Tensor& t = need(name);
    if (t.type != gguf::Type::F32 || int64_t(t.elements()) != n)
      throw std::runtime_error(f.path() + ": " + name + " is not an F32 vector of " + std::to_string(n));
    return reinterpret_cast<const float*>(put(t));
  };

  w.tok_embd = matrix("token_embd.weight", c.n_embd, c.n_vocab);
  w.output_norm = vec("output_norm.weight", c.n_embd);
  const gguf::Tensor& rf = need("rope_freqs.weight");
  if (rf.type != gguf::Type::F32 || int64_t(rf.elements()) != c.head_dim_global / 2)
    throw std::runtime_error(f.path() + ": rope_freqs.weight has an unexpected shape");
  w.rope_freqs.assign(reinterpret_cast<const float*>(rf.data), reinterpret_cast<const float*>(rf.data) + rf.elements());

  for (int l = 0; l < c.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    const int64_t d = c.head_dim(l), q_dim = d * c.n_head, kv_dim = d * c.n_head_kv[size_t(l)];
    Layer L{};
    L.attn_norm = vec(b + "attn_norm.weight", c.n_embd);
    L.wq = matrix(b + "attn_q.weight", c.n_embd, q_dim);
    L.wk = matrix(b + "attn_k.weight", c.n_embd, kv_dim);
    if (f.tensor(b + "attn_v.weight")) L.wv = matrix(b + "attn_v.weight", c.n_embd, kv_dim);
    L.wo = matrix(b + "attn_output.weight", q_dim, c.n_embd);
    L.q_norm = vec(b + "attn_q_norm.weight", d);
    L.k_norm = vec(b + "attn_k_norm.weight", d);
    L.attn_post_norm = vec(b + "post_attention_norm.weight", c.n_embd);
    L.ffn_norm = vec(b + "ffn_norm.weight", c.n_embd);
    L.gate = matrix(b + "ffn_gate.weight", c.n_embd, c.n_ff);
    L.up = matrix(b + "ffn_up.weight", c.n_embd, c.n_ff);
    L.down = matrix(b + "ffn_down.weight", c.n_ff, c.n_embd);
    L.ffn_post_norm = vec(b + "post_ffw_norm.weight", c.n_embd);
    const gguf::Tensor* os = f.tensor(b + "layer_output_scale.weight");
    L.out_scale = os ? *reinterpret_cast<const float*>(os->data) : 1.0f;
    w.layers.push_back(L);
  }
  return w;
}

std::vector<float> reference_forward(sycl::queue& q, const Gemma4Config& c, const Gemma4Weights& w,
                                     const std::vector<int32_t>& tokens, int64_t logits_rows, const DumpSink* dump) {
  const int64_t T = int64_t(tokens.size()), E = c.n_embd, F = c.n_ff, H = c.n_head;
  const int64_t D_max = std::max(c.head_dim_global, c.head_dim_sliding);
  int64_t kv_max = 0;
  for (int l = 0; l < c.n_layer; ++l) kv_max = std::max<int64_t>(kv_max, c.n_head_kv[size_t(l)] * c.head_dim(l));
  if (logits_rows < 1 || logits_rows > T) logits_rows = T;

  std::vector<void*> owned;
  auto alloc = [&](int64_t n) {
    auto* p = sycl::malloc_device<float>(size_t(n), q);
    if (!p) throw std::runtime_error("out of device memory for activations");
    owned.push_back(p);
    return p;
  };
  struct Free {
    sycl::queue& q;
    std::vector<void*>& v;
    ~Free() {
      for (void* p : v) sycl::free(p, q);
    }
  } free_all{q, owned};

  auto* tok = sycl::malloc_device<int32_t>(size_t(T), q);
  owned.push_back(tok);
  q.memcpy(tok, tokens.data(), size_t(T) * sizeof(int32_t)).wait();
  float *x = alloc(T * E), *x2 = alloc(T * E), *h = alloc(T * E), *o = alloc(T * E);
  float *qb = alloc(T * H * D_max), *kb = alloc(T * kv_max), *vb = alloc(T * kv_max), *att = alloc(T * H * D_max);
  float *gate = alloc(T * F), *up = alloc(T * F);

  auto upload_tables = [&](const ref::RopeTables& t) {
    float* cs = alloc(int64_t(t.cos.size()));
    float* sn = alloc(int64_t(t.sin.size()));
    q.memcpy(cs, t.cos.data(), t.cos.size() * sizeof(float));
    q.memcpy(sn, t.sin.data(), t.sin.size() * sizeof(float)).wait();
    return std::pair{cs, sn};
  };
  auto [cos_s, sin_s] = upload_tables(ref::rope_tables(T, c.head_dim_sliding, c.rope_base_sliding, nullptr));
  auto [cos_g, sin_g] = upload_tables(ref::rope_tables(T, c.head_dim_global, c.rope_base_global, w.rope_freqs.data()));

  auto emit = [&](const std::string& name, const float* p, std::vector<int64_t> ne) {
    if (!dump || !dump->wants(name)) return;
    int64_t n = 1;
    for (auto e : ne) n *= e;
    std::vector<float> host(static_cast<size_t>(n));
    q.memcpy(host.data(), p, size_t(n) * sizeof(float)).wait();
    dump->put(name, host, ne);
  };

  ref::embed(q, w.tok_embd, tok, T, std::sqrt(float(E)), x);
  emit("inp_scaled", x, {E, T});

  for (int l = 0; l < c.n_layer; ++l) {
    const auto& L = w.layers[size_t(l)];
    const std::string s = "-" + std::to_string(l);
    const bool sliding = c.sliding[size_t(l)];
    const int64_t D = c.head_dim(l), Hkv = c.n_head_kv[size_t(l)];

    ref::rmsnorm(q, x, L.attn_norm, T, E, c.eps, h);
    emit("attn_norm" + s, h, {E, T});

    ref::matmul(q, L.wq, h, T, qb);
    emit("Qcur" + s, qb, {H * D, T});
    ref::rmsnorm(q, qb, L.q_norm, T * H, D, c.eps, qb);
    emit("Qcur_normed" + s, qb, {D, H, T});
    ref::rope_neox(q, qb, T, H, D, D, sliding ? cos_s : cos_g, sliding ? sin_s : sin_g);
    emit("Qcur_pos" + s, qb, {D, H, T});

    ref::matmul(q, L.wk, h, T, kb);
    // Global layers have no V projection: V is K's projection, taken before K's norm.
    if (L.wv.data) ref::matmul(q, L.wv, h, T, vb);
    else q.memcpy(vb, kb, size_t(T * Hkv * D) * sizeof(float)).wait();
    ref::rmsnorm(q, kb, L.k_norm, T * Hkv, D, c.eps, kb);
    emit("Kcur_normed" + s, kb, {D, Hkv, T});
    ref::rope_neox(q, kb, T, Hkv, D, D, sliding ? cos_s : cos_g, sliding ? sin_s : sin_g);
    emit("Kcur_pos" + s, kb, {D, Hkv, T});
    ref::rmsnorm(q, vb, nullptr, T * Hkv, D, c.eps, vb);
    emit("Vcur_normed" + s, vb, {D, Hkv, T});

    ref::attention(q, qb, kb, vb, T, H, Hkv, D, sliding ? c.window : 0, 1.0f, att);
    emit("kqv_out" + s, att, {H * D, T});
    ref::matmul(q, L.wo, att, T, o);
    ref::rmsnorm(q, o, L.attn_post_norm, T, E, c.eps, o);
    emit("attn_post_norm" + s, o, {E, T});
    ref::add(q, o, x, T * E, x2);
    emit("attn_out" + s, x2, {E, T});

    ref::rmsnorm(q, x2, L.ffn_norm, T, E, c.eps, h);
    emit("ffn_norm" + s, h, {E, T});
    ref::matmul(q, L.gate, h, T, gate);
    ref::matmul(q, L.up, h, T, up);
    ref::gelu_mul(q, gate, up, T * F, gate);
    emit("ffn_geglu" + s, gate, {F, T});
    ref::matmul(q, L.down, gate, T, o);
    emit("ffn_out" + s, o, {E, T});
    ref::rmsnorm(q, o, L.ffn_post_norm, T, E, c.eps, o);
    emit("ffn_post_norm" + s, o, {E, T});
    ref::add(q, o, x2, T * E, x);
    ref::scale(q, x, L.out_scale, T * E);
    emit("l_out" + s, x, {E, T});
  }

  ref::rmsnorm(q, x, w.output_norm, T, E, c.eps, h);
  emit("result_norm", h, {E, T});
  float* logits = alloc(logits_rows * c.n_vocab);
  ref::matmul(q, w.tok_embd, h + (T - logits_rows) * E, logits_rows, logits);
  if (c.softcap > 0) ref::softcap(q, logits, c.softcap, logits_rows * c.n_vocab);
  std::vector<float> out(size_t(logits_rows * c.n_vocab));
  q.memcpy(out.data(), logits, out.size() * sizeof(float)).wait();
  return out;
}

}  // namespace eklow::model
