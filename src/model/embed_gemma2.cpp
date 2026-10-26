#include "model/embed_gemma2.hpp"

#include <cmath>
#include <stdexcept>

namespace eklow::model {

namespace {

int64_t get_int(const gguf::File& f, const std::string& k) { return f.at(k).as_int(); }
float get_float(const gguf::File& f, const std::string& k) { return float(f.at(k).as_float()); }

}  // namespace

float* EmbedGemma2::buf(size_t floats) {
  auto* p = sycl::malloc_device<float>(floats, q_);
  if (!p) throw std::runtime_error("embedding model: out of device memory");
  owned_.push_back(p);
  return p;
}

EmbedGemma2::EmbedGemma2(sycl::queue& q, const gguf::File& f) : q_(q) {
  if (f.at("general.architecture").as_string() != "gemma-embedding2") throw std::runtime_error(f.path() + ": not gemma-embedding2");
  const std::string p = "gemma-embedding2.";
  n_embd_ = get_int(f, p + "embedding_length");
  n_ff_ = get_int(f, p + "feed_forward_length");
  n_head_ = get_int(f, p + "attention.head_count");
  n_layer_ = get_int(f, p + "block_count");
  n_pl_ = get_int(f, p + "embedding_length_per_layer_input");
  n_out_ = get_int(f, p + "embedding_length_out");
  window_ = get_int(f, p + "attention.sliding_window");
  eps_ = get_float(f, p + "attention.layer_norm_rms_epsilon");
  rope_base_g_ = get_float(f, p + "rope.freq_base");
  rope_base_s_ = get_float(f, p + "rope.freq_base_swa");
  if (get_int(f, p + "pooling_type") != 1) throw std::runtime_error(f.path() + ": only mean pooling is supported");
  const int64_t d_s = get_int(f, p + "attention.key_length_swa"), d_g = get_int(f, p + "attention.key_length");
  std::vector<bool> sliding;
  for (const auto& v : f.at(p + "attention.sliding_window_pattern").as_array()) sliding.push_back(std::get<bool>(v.v));
  std::vector<int64_t> hkv;
  for (const auto& v : f.at(p + "attention.head_count_kv").as_array()) hkv.push_back(v.as_int());

  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto upload = [&](const gguf::Tensor& t) {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q_);
    if (!d) throw std::runtime_error("embedding model: out of device memory loading " + t.name);
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

  tok_embd_ = matrix("token_embd.weight");
  pl_proj_ = matrix("per_layer_model_proj.weight");
  pl_norm_ = vec("per_layer_proj_norm.weight");
  output_ = matrix("output.weight");
  output_norm_ = vec("output_norm.weight");
  for (int64_t l = 0; l < n_layer_; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L{};
    L.sliding = sliding.at(size_t(l));
    L.head_dim = L.sliding ? d_s : d_g;
    L.n_head_kv = hkv.at(size_t(l));
    L.wq = matrix(b + "attn_q.weight");
    L.wk = matrix(b + "attn_k.weight");
    L.wv = matrix(b + "attn_v.weight");
    L.wo = matrix(b + "attn_output.weight");
    L.gate = matrix(b + "ffn_gate.weight");
    L.up = matrix(b + "ffn_up.weight");
    L.down = matrix(b + "ffn_down.weight");
    L.inp_gate = matrix(b + "inp_gate.weight");
    L.proj = matrix(b + "proj.weight");
    L.attn_norm = vec(b + "attn_norm.weight");
    L.q_norm = vec(b + "attn_q_norm.weight");
    L.k_norm = vec(b + "attn_k_norm.weight");
    L.attn_post_norm = vec(b + "post_attention_norm.weight");
    L.ffn_norm = vec(b + "ffn_norm.weight");
    L.ffn_post_norm = vec(b + "post_ffw_norm.weight");
    L.post_norm = vec(b + "post_norm.weight");
    L.out_scale = *reinterpret_cast<const float*>(need(b + "layer_output_scale.weight").data);
    layers_.push_back(L);
  }
}

EmbedGemma2::~EmbedGemma2() {
  q_.wait();
  for (void* p : owned_) sycl::free(p, q_);
  for (float* p : act_) sycl::free(p, q_);
}

// Activation buffers for n_tok tokens: x, x2, h, o, q, k, v, att, gate, up, per-layer inputs,
// per-layer gate, embeddings, and the token ids; then the RoPE tables for positions 0..n_tok-1
// (cos and sin of the sliding layers, then of the global ones), which serve any shorter input.
void EmbedGemma2::ensure(int64_t n_tok) {
  if (n_tok <= max_tok_) return;
  q_.wait();
  for (float* p : act_) sycl::free(p, q_);
  act_.clear();
  int64_t dmax = 0, d_s = 0, d_g = 0;
  for (const auto& L : layers_) {
    dmax = std::max(dmax, L.head_dim);
    (L.sliding ? d_s : d_g) = L.head_dim;
  }
  const int64_t T = std::max<int64_t>(n_tok, 2 * max_tok_), E = n_embd_;
  for (int64_t n : {T * E, T * E, T * E, T * E, T * n_head_ * dmax, T * n_head_ * dmax, T * n_head_ * dmax, T * n_head_ * dmax,
                    T * n_ff_, T * n_ff_, T * n_pl_ * n_layer_, T * n_pl_, T * n_out_, T, T * d_s / 2, T * d_s / 2, T * d_g / 2,
                    T * d_g / 2}) {
    auto* p = sycl::malloc_device<float>(size_t(n), q_);
    if (!p) throw std::runtime_error("embedding model: out of device memory for activations");
    act_.push_back(p);
  }
  const ref::RopeTables s = ref::rope_tables(T, d_s, rope_base_s_, nullptr), g = ref::rope_tables(T, d_g, rope_base_g_, nullptr);
  q_.memcpy(act_[14], s.cos.data(), s.cos.size() * sizeof(float));
  q_.memcpy(act_[15], s.sin.data(), s.sin.size() * sizeof(float));
  q_.memcpy(act_[16], g.cos.data(), g.cos.size() * sizeof(float));
  q_.memcpy(act_[17], g.sin.data(), g.sin.size() * sizeof(float)).wait();
  max_tok_ = T;
}

std::vector<float> EmbedGemma2::embed(const std::vector<int32_t>& tokens) {
  return std::move(embed(std::vector<std::vector<int32_t>>{tokens})[0]);
}

// The inputs one after another on the queue, each one's last layer copied to the host, and a single
// wait at the end: with a long prefill on the card, every wait costs a turn on it (~90 ms at 50k
// tokens of depth), and a wait per kernel made an input take a minute.
std::vector<std::vector<float>> EmbedGemma2::embed(const std::vector<std::vector<int32_t>>& seqs) {
  int64_t longest = 0;
  for (const auto& s : seqs) {
    if (s.empty()) throw std::runtime_error("embedding model: empty input");
    longest = std::max(longest, int64_t(s.size()));
  }
  ensure(longest);
  std::vector<std::vector<float>> rows(seqs.size());
  try {
    for (size_t i = 0; i < seqs.size(); ++i) {
      rows[i].resize(seqs[i].size() * size_t(n_out_));
      forward(seqs[i], rows[i].data());
    }
  } catch (...) {
    q_.wait();  // copies already queued write into rows
    throw;
  }
  q_.wait();
  std::vector<std::vector<float>> out;
  for (size_t i = 0; i < seqs.size(); ++i) {
    const int64_t T = int64_t(seqs[i].size());
    std::vector<double> mean(size_t(n_out_), 0.0);
    for (int64_t t = 0; t < T; ++t)
      for (int64_t j = 0; j < n_out_; ++j) mean[size_t(j)] += rows[i][size_t(t * n_out_ + j)];
    double norm = 0;
    for (auto& v : mean) {
      v /= double(T);
      norm += v * v;
    }
    norm = std::sqrt(norm);
    std::vector<float> e(static_cast<size_t>(n_out_));
    for (int64_t j = 0; j < n_out_; ++j) e[size_t(j)] = float(norm > 0 ? mean[size_t(j)] / norm : 0.0);
    out.push_back(std::move(e));
  }
  return out;
}

void EmbedGemma2::forward(const std::vector<int32_t>& tokens, float* rows) {
  const int64_t T = int64_t(tokens.size()), E = n_embd_, H = n_head_, L = n_layer_;
  float *x = act_[0], *x2 = act_[1], *h = act_[2], *o = act_[3], *qb = act_[4], *kb = act_[5], *vb = act_[6], *att = act_[7];
  float *gate = act_[8], *up = act_[9], *pl = act_[10], *pg = act_[11], *emb = act_[12];
  auto* tok = reinterpret_cast<int32_t*>(act_[13]);
  const float *cos_s = act_[14], *sin_s = act_[15], *cos_g = act_[16], *sin_g = act_[17];
  q_.memcpy(tok, tokens.data(), tokens.size() * sizeof(int32_t));

  ref::embed(q_, tok_embd_, tok, T, std::sqrt(float(E)), x);
  ref::matmul(q_, pl_proj_, x, T, pl);
  ref::scale(q_, pl, 1.0f / std::sqrt(float(E)), T * n_pl_ * L);
  ref::rmsnorm(q_, pl, pl_norm_, T * L, n_pl_, eps_, pl);

  for (int64_t l = 0; l < L; ++l) {
    const Layer& Ly = layers_[size_t(l)];
    const int64_t D = Ly.head_dim, Hkv = Ly.n_head_kv;
    ref::rmsnorm(q_, x, Ly.attn_norm, T, E, eps_, h);
    ref::matmul(q_, Ly.wq, h, T, qb);
    ref::matmul(q_, Ly.wk, h, T, kb);
    ref::matmul(q_, Ly.wv, h, T, vb);
    ref::rmsnorm(q_, qb, Ly.q_norm, T * H, D, eps_, qb);
    ref::rmsnorm(q_, kb, Ly.k_norm, T * Hkv, D, eps_, kb);
    ref::rmsnorm(q_, vb, nullptr, T * Hkv, D, eps_, vb);
    ref::rope_neox(q_, qb, T, H, D, D, Ly.sliding ? cos_s : cos_g, Ly.sliding ? sin_s : sin_g);
    ref::rope_neox(q_, kb, T, Hkv, D, D, Ly.sliding ? cos_s : cos_g, Ly.sliding ? sin_s : sin_g);
    ref::attention_bidir(q_, qb, kb, vb, T, H, Hkv, D, Ly.sliding ? window_ / 2 : 0, 1.0f, att);
    ref::matmul(q_, Ly.wo, att, T, o);
    ref::rmsnorm(q_, o, Ly.attn_post_norm, T, E, eps_, o);
    ref::add(q_, o, x, T * E, x2);

    ref::rmsnorm(q_, x2, Ly.ffn_norm, T, E, eps_, h);
    ref::matmul(q_, Ly.gate, h, T, gate);
    ref::matmul(q_, Ly.up, h, T, up);
    ref::gelu_mul(q_, gate, up, T * n_ff_, gate);
    ref::matmul(q_, Ly.down, gate, T, o);
    ref::rmsnorm(q_, o, Ly.ffn_post_norm, T, E, eps_, o);
    ref::add(q_, o, x2, T * E, x2);

    // The per-layer input: this layer's slice of P, [t][l][:], gathered row by row.
    ref::matmul(q_, Ly.inp_gate, x2, T, pg);
    ref::gelu(q_, pg, T * n_pl_, pg);
    const int64_t npl = n_pl_, nl = L, li = l;
    q_.parallel_for(sycl::range<2>(size_t(T), size_t(npl)), [=](sycl::id<2> id) {
       const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
       pg[t * npl + i] *= pl[(t * nl + li) * npl + i];
     });
    ref::matmul(q_, Ly.proj, pg, T, o);
    ref::rmsnorm(q_, o, Ly.post_norm, T, E, eps_, o);
    ref::add(q_, o, x2, T * E, x);
    ref::scale(q_, x, Ly.out_scale, T * E);
  }

  ref::rmsnorm(q_, x, output_norm_, T, E, eps_, h);
  ref::matmul(q_, output_, h, T, emb);
  q_.memcpy(rows, emb, size_t(T * n_out_) * sizeof(float));
}

}  // namespace eklow::model
