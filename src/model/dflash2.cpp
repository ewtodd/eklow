#include "model/dflash2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels/ref_dflash.hpp"

namespace eklow::model {

namespace {

int64_t get_int(const gguf::File& f, const std::string& key) { return f.at(key).as_int(); }

[[noreturn]] void unsupported(const gguf::File& f, const std::string& what) {
  throw std::runtime_error(f.path() + ": not a DFlash 2 draft this engine runs: " + what);
}

}  // namespace

void plain_lattice(serve::Lattice& lat, float softcap) {
  const size_t K = size_t(lat.top_k), P = K ? lat.cand.size() / K : 0;
  if (softcap > 0)
    for (float& u : lat.unary) u = softcap * std::tanh(u / softcap);
  lat.score.resize(P * K * K);
  for (size_t i = 0; i < P; ++i)
    for (size_t a = 0; a < K; ++a)
      std::copy(lat.unary.begin() + long(i * K), lat.unary.begin() + long((i + 1) * K), lat.score.begin() + long((i * K + a) * K));
}

Dflash2Config Dflash2Config::from_gguf(const gguf::File& f) {
  if (f.at("general.architecture").as_string() != "dflash") unsupported(f, "architecture is not dflash");
  const std::string p = "dflash.";
  Dflash2Config c{};
  c.n_layer = int(get_int(f, p + "block_count"));
  c.n_embd = int(get_int(f, p + "embedding_length"));
  c.n_ff = int(get_int(f, p + "feed_forward_length"));
  c.n_head = int(get_int(f, p + "attention.head_count"));
  c.n_head_kv = int(get_int(f, p + "attention.head_count_kv"));
  c.head_dim = int(get_int(f, p + "attention.key_length"));
  c.window = int(get_int(f, p + "attention.sliding_window"));
  c.block_size = int(get_int(f, p + "block_size"));
  const bool has_conv = f.find(p + "conv_kernel_size") != nullptr, has_sel = f.find(p + "selector_top_k") != nullptr;
  if (has_conv != has_sel) unsupported(f, "convolutions without a selector or the reverse");
  c.plain = !has_conv;
  if (!c.plain) {
    c.conv_kernel = int(get_int(f, p + "conv_kernel_size"));
    c.conv_group = int(get_int(f, p + "conv_group_size"));
    c.rank = int(get_int(f, p + "selector_rank"));
    c.top_k = int(get_int(f, p + "selector_top_k"));
  } else {
    c.top_k = 16;
  }
  c.mask_token = int(get_int(f, "tokenizer.ggml.mask_token_id"));
  c.eps = float(f.at(p + "attention.layer_norm_rms_epsilon").as_float());
  c.rope_base = float(f.at(p + "rope.freq_base").as_float());
  for (const auto& v : f.at(p + "target_layers").as_array()) c.target_layers.push_back(int(v.as_int()) - 1);
  const auto& pattern = f.at(p + "attention.sliding_window_pattern").as_array();
  if (int(pattern.size()) != c.n_layer) unsupported(f, "a sliding window pattern of another length than the layers");
  for (const auto& v : pattern) {
    const bool sliding = std::get<bool>(v.v);
    if (!sliding && !c.plain) unsupported(f, "a full-attention layer");
    if (c.plain) c.full.push_back(sliding ? 0 : 1);
  }
  if (const auto* v = f.find(p + "attention.causal"); v && std::get<bool>(v->v)) unsupported(f, "causal attention");
  for (const char* k : {"logit_scale", "attention.value_scale", "attention.scale"})
    if (f.find(p + k)) unsupported(f, std::string("the key ") + k);
  if (!c.plain) {
    for (const char* k : {"embedding_scale", "final_logit_softcapping"})
      if (f.find(p + k)) unsupported(f, std::string("the key ") + k);
  } else {
    if (const auto* v = f.find(p + "embedding_scale")) c.embed_scale = float(v->as_float());
    if (const auto* v = f.find(p + "final_logit_softcapping")) c.softcap = float(v->as_float());
    // z-lab's drafts carry no attention.causal and their model.py runs the sliding layers causally inside the block;
    // NVIDIA's Bolt draft says causal false and runs the whole block non-causally (its card: "a single non-causal forward
    // pass").
    c.block_causal = f.find(p + "attention.causal") == nullptr;
  }
  if (get_int(f, p + "attention.value_length") != c.head_dim) unsupported(f, "value heads differ from key heads");
  if (c.plain) {
    if (c.head_dim != 128 || c.n_head_kv <= 0 || c.n_head != 4 * c.n_head_kv) unsupported(f, "head geometry (128-wide heads, four query heads per K/V head)");
    const gguf::Value* toks = f.find("tokenizer.ggml.tokens");
    if (!toks) unsupported(f, "no tokenizer.ggml.tokens to take the vocabulary size from");
    c.n_vocab = int(toks->as_array().size());
    return c;
  }
  if (c.head_dim != 128 || c.head_dim * c.n_head != 4096 || c.n_head % c.n_head_kv != 0) unsupported(f, "head geometry");
  if (c.n_embd % c.conv_group != 0 || c.conv_kernel != 2 || c.top_k != 16 || c.rank != 256) unsupported(f, "convolution or selector geometry");
  const gguf::Tensor* pred = f.tensor("selector_predecessor.weight");
  if (!pred || pred->ne.size() != 2 || int(pred->ne[0]) != c.rank) unsupported(f, "no selector_predecessor.weight");
  c.n_vocab = int(pred->ne[1]);
  return c;
}

Dflash2Weights::Dflash2Weights(Dflash2Weights&& o) noexcept
    : fc(o.fc), sel_hidden(o.sel_hidden), sel_pred(o.sel_pred), sel_succ(o.sel_succ), tok_embd(o.tok_embd), head(o.head),
      hidden_norm(o.hidden_norm), out_norm(o.out_norm), layers(std::move(o.layers)), bytes(o.bytes), q_(o.q_),
      allocations_(std::move(o.allocations_)) {
  o.allocations_.clear();
}

Dflash2Weights::~Dflash2Weights() {
  for (void* p : allocations_) sycl::free(p, *q_);
}

Dflash2Weights Dflash2Weights::upload(const gguf::File& f, const gguf::File& target, const Dflash2Config& c, sycl::queue& q) {
  Dflash2Weights w(q);
  auto put = [&](const gguf::Tensor& t) -> const uint8_t* {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q);
    if (!d) throw std::runtime_error("out of device memory uploading " + t.name);
    w.allocations_.push_back(d);
    q.memcpy(d, t.data, t.nbytes).wait();
    w.bytes += t.nbytes;
    return d;
  };
  auto need = [](const gguf::File& g, const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = g.tensor(name);
    if (!t) throw std::runtime_error(g.path() + ": missing tensor " + name);
    return *t;
  };
  auto matrix = [&](const gguf::File& g, const std::string& name, int64_t k, int64_t n) {
    const gguf::Tensor& t = need(g, name);
    if (t.ne.size() != 2 || int64_t(t.ne[0]) != k || int64_t(t.ne[1]) != n)
      throw std::runtime_error(g.path() + ": " + name + " has an unexpected shape");
    return ref::Weight{t.type, k, n, put(t), t.nbytes / uint64_t(n)};
  };
  auto vec = [&](const std::string& name, int64_t n) {
    const gguf::Tensor& t = need(f, name);
    if (t.type != gguf::Type::F32 || int64_t(t.elements()) != n)
      throw std::runtime_error(f.path() + ": " + name + " is not an F32 array of " + std::to_string(n));
    return reinterpret_cast<const float*>(put(t));
  };
  const int E = c.n_embd, Hd = c.head_dim, nc = c.plain ? 0 : 2 * c.conv_kernel * (E / c.conv_group);
  w.fc = matrix(f, "fc.weight", c.n_feat(), E);
  if (!c.plain) {
    w.sel_hidden = matrix(f, "selector_hidden.weight", E, c.rank);
    w.sel_pred = matrix(f, "selector_predecessor.weight", c.rank, c.n_vocab);
    w.sel_succ = matrix(f, "selector_successor.weight", c.rank, c.n_vocab);
  }
  w.hidden_norm = vec("enc.output_norm.weight", E);
  w.out_norm = vec("output_norm.weight", E);
  w.tok_embd = matrix(target, "token_embd.weight", E, c.n_vocab);
  // a target that ties its head to the embedding has no output.weight
  w.head = matrix(target, target.tensor("output.weight") ? "output.weight" : "token_embd.weight", E, c.n_vocab);
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L{};
    L.attn_norm = vec(b + "attn_norm.weight", E);
    L.wq = matrix(f, b + "attn_q.weight", E, int64_t(c.n_head) * Hd);
    L.wk = matrix(f, b + "attn_k.weight", E, int64_t(c.n_head_kv) * Hd);
    L.wv = matrix(f, b + "attn_v.weight", E, int64_t(c.n_head_kv) * Hd);
    L.wo = matrix(f, b + "attn_output.weight", int64_t(c.n_head) * Hd, E);
    L.q_norm = vec(b + "attn_q_norm.weight", Hd);
    L.k_norm = vec(b + "attn_k_norm.weight", Hd);
    L.ffn_norm = vec(b + "ffn_norm.weight", E);
    L.gate = matrix(f, b + "ffn_gate.weight", E, c.n_ff);
    L.up = matrix(f, b + "ffn_up.weight", E, c.n_ff);
    L.down = matrix(f, b + "ffn_down.weight", c.n_ff, E);
    if (!c.plain) {
      L.conv_attn = matrix(f, b + "attn_conv_proj.weight", E, nc);
      L.conv_ffn = matrix(f, b + "ffn_conv_proj.weight", E, nc);
      L.base_attn = vec(b + "attn_conv_base", int64_t(2) * c.conv_kernel * E);
      L.base_ffn = vec(b + "ffn_conv_base", int64_t(2) * c.conv_kernel * E);
    }
    w.layers.push_back(L);
  }
  return w;
}

// ---------------------------------------------------------------------------------------------------------------
// The reference forward.

Dflash2Ref::Dflash2Ref(sycl::queue& q, const Dflash2Config& c, const Dflash2Weights& w, int64_t max_pos, int window, bool emulate_acts, float jitter)
    : q_(q), c_(c), w_(w), window_(window ? window : c.window), emulate_(emulate_acts), jitter_(jitter), max_pos_(max_pos) {
  const size_t kv = size_t(max_pos) * size_t(c.n_head_kv * c.head_dim);
  for (int l = 0; l < c.n_layer; ++l) {
    k_.push_back(sycl::malloc_device<float>(kv, q));
    v_.push_back(sycl::malloc_device<float>(kv, q));
    if (!k_.back() || !v_.back()) throw std::runtime_error("dflash2 reference: out of device memory");
  }
  const ref::RopeTables t = ref::rope_tables(max_pos, c.head_dim, c.rope_base, nullptr);
  cos_ = sycl::malloc_device<float>(t.cos.size(), q);
  sin_ = sycl::malloc_device<float>(t.sin.size(), q);
  q.memcpy(cos_, t.cos.data(), t.cos.size() * sizeof(float));
  q.memcpy(sin_, t.sin.data(), t.sin.size() * sizeof(float)).wait();
}

Dflash2Ref::~Dflash2Ref() {
  q_.wait();
  for (float* p : k_) sycl::free(p, q_);
  for (float* p : v_) sycl::free(p, q_);
  sycl::free(cos_, q_);
  sycl::free(sin_, q_);
}

void Dflash2Ref::inject(std::span<const float> feats, int64_t pos0, int64_t n) {
  const int E = c_.n_embd, Hkv = c_.n_head_kv, D = c_.head_dim, F = c_.n_feat();
  if (int64_t(feats.size()) != n * F) throw std::runtime_error("dflash2 reference: inject: wrong feature size");
  if (pos0 + n > max_pos_) throw std::runtime_error("dflash2 reference: positions beyond max_pos");
  auto dev = [&](size_t floats) {
    float* p = sycl::malloc_device<float>(floats, q_);
    if (!p) throw std::runtime_error("dflash2 reference: out of device memory");
    return p;
  };
  float *x = dev(size_t(n * F)), *h = dev(size_t(n * E)), *kb = dev(size_t(n * Hkv * D)), *vb = dev(size_t(n * Hkv * D));
  q_.memcpy(x, feats.data(), feats.size() * sizeof(float)).wait();
  float* tmp = emulate_ ? dev(size_t(n) * F) : nullptr;
  // matmul on x [n][k], rounded like the serving forward's activations when emulating
  auto mm = [&](const ref::Weight& w, const float* in, float* out) {
    if (emulate_) {
      ref::round_acts(q_, in, n, w.k, n > xmx::kFp16Rows, tmp);
      in = tmp;
    }
    ref::matmul(q_, w, in, n, out);
  };
  mm(w_.fc, x, h);
  ref::rmsnorm(q_, h, w_.hidden_norm, n, E, c_.eps, h);
  for (int l = 0; l < c_.n_layer; ++l) {
    const auto& L = w_.layers[size_t(l)];
    mm(L.wk, h, kb);
    mm(L.wv, h, vb);
    ref::rmsnorm(q_, kb, L.k_norm, n * Hkv, D, c_.eps, kb);
    ref::rope_neox(q_, kb, n, Hkv, D, D, cos_ + pos0 * (D / 2), sin_ + pos0 * (D / 2));
    if (emulate_) {  // the serving cache holds fp16
      ref::round_acts(q_, kb, n * Hkv, D, true, kb);
      ref::round_acts(q_, vb, n * Hkv, D, true, vb);
    }
    q_.memcpy(k_[size_t(l)] + pos0 * Hkv * D, kb, size_t(n * Hkv * D) * sizeof(float));
    q_.memcpy(v_[size_t(l)] + pos0 * Hkv * D, vb, size_t(n * Hkv * D) * sizeof(float));
  }
  q_.wait();
  for (float* p : {x, h, kb, vb}) sycl::free(p, q_);
  if (tmp) sycl::free(tmp, q_);
}

serve::Lattice Dflash2Ref::block(int32_t anchor, int64_t pos, int B, BlockDebug* debug) {
  const int E = c_.n_embd, H = c_.n_head, Hkv = c_.n_head_kv, D = c_.head_dim, K = c_.top_k, R = c_.rank, P = B - 1;
  const int nc = c_.plain ? 0 : 2 * c_.conv_kernel * (E / c_.conv_group);
  if (pos + B > max_pos_) throw std::runtime_error("dflash2 reference: positions beyond max_pos");
  auto dev = [&](size_t n) {
    float* p = sycl::malloc_device<float>(std::max<size_t>(n, 1), q_);
    if (!p) throw std::runtime_error("dflash2 reference: out of device memory");
    return p;
  };
  std::vector<void*> frees;
  auto fdev = [&](size_t n) {
    float* p = dev(n);
    frees.push_back(p);
    return p;
  };
  int32_t* tok = sycl::malloc_device<int32_t>(size_t(B), q_);
  int32_t *lo = sycl::malloc_device<int32_t>(size_t(B), q_), *hi = sycl::malloc_device<int32_t>(size_t(B), q_);
  int32_t *lo_full = sycl::malloc_device<int32_t>(size_t(B), q_), *hi_full = sycl::malloc_device<int32_t>(size_t(B), q_);  // a plain draft's full layers
  frees.insert(frees.end(), {tok, lo, hi, lo_full, hi_full});
  std::vector<int32_t> htok(size_t(B), c_.mask_token), hlo(static_cast<size_t>(B)), hhi(static_cast<size_t>(B), int32_t(pos + B - 1));
  std::vector<int32_t> hlo_full(static_cast<size_t>(B), 0), hhi_full(static_cast<size_t>(B), int32_t(pos + B - 1));
  htok[0] = anchor;
  for (int t = 0; t < B; ++t) {
    hlo[size_t(t)] = int32_t(std::max<int64_t>(0, pos + t - window_ + 1));
    if (c_.block_causal) hhi[size_t(t)] = int32_t(pos + t);  // sliding layers of a plain draft: causal inside the block
  }
  q_.memcpy(tok, htok.data(), size_t(B) * sizeof(int32_t));
  q_.memcpy(lo, hlo.data(), size_t(B) * sizeof(int32_t));
  q_.memcpy(hi, hhi.data(), size_t(B) * sizeof(int32_t));
  q_.memcpy(lo_full, hlo_full.data(), size_t(B) * sizeof(int32_t));
  q_.memcpy(hi_full, hhi_full.data(), size_t(B) * sizeof(int32_t));

  float *x = fdev(size_t(B) * E), *n1 = fdev(size_t(B) * E), *dyn = fdev(size_t(B) * nc), *cv = fdev(size_t(B) * E);
  float *qb = fdev(size_t(B) * H * D), *kb = fdev(size_t(B) * Hkv * D), *vb = fdev(size_t(B) * Hkv * D), *ab = fdev(size_t(B) * H * D);
  float *ob = fdev(size_t(B) * E), *ob2 = fdev(size_t(B) * E), *g = fdev(size_t(B) * c_.n_ff), *u = fdev(size_t(B) * c_.n_ff);
  ref::embed(q_, w_.tok_embd, tok, B, c_.embed_scale, x);
  if (jitter_ != 0) ref::jitter(q_, x, int64_t(B) * E, jitter_);
  float* rnd = fdev(size_t(B) * c_.n_ff);
  auto mm = [&](const ref::Weight& w, const float* in, int64_t rows, float* out) {
    if (emulate_) {
      ref::round_acts(q_, in, rows, w.k, rows > xmx::kFp16Rows, rnd);
      in = rnd;
    }
    ref::matmul(q_, w, in, rows, out);
  };
  const float scale = 1.0f / std::sqrt(float(D));
  const int64_t rope_off = pos * (D / 2);
  for (int l = 0; l < c_.n_layer; ++l) {
    const auto& L = w_.layers[size_t(l)];
    auto stage = [&](const char* name, const float* p, size_t n) {
      if (!debug || l != debug->stage_layer) return;
      std::vector<float> h(n);
      q_.memcpy(h.data(), p, n * sizeof(float)).wait();
      debug->stages.emplace_back(name, std::move(h));
    };
    ref::rmsnorm(q_, x, L.attn_norm, B, E, c_.eps, n1);
    stage("n1", n1, size_t(B) * E);
    if (!c_.plain) {
      mm(L.conv_attn, n1, B, dyn);
      stage("dyn", dyn, size_t(B) * nc);
      ref::grouped_conv(q_, n1, dyn, L.base_attn, 0, B, B, E, c_.conv_kernel, c_.conv_group, cv);
    } else {
      q_.memcpy(cv, n1, size_t(B) * E * sizeof(float));
    }
    stage("conv", cv, size_t(B) * E);
    mm(L.wq, cv, B, qb);
    mm(L.wk, cv, B, kb);
    mm(L.wv, cv, B, vb);
    ref::rmsnorm(q_, qb, L.q_norm, int64_t(B) * H, D, c_.eps, qb);
    ref::rmsnorm(q_, kb, L.k_norm, int64_t(B) * Hkv, D, c_.eps, kb);
    ref::rope_neox(q_, qb, B, H, D, D, cos_ + rope_off, sin_ + rope_off);
    ref::rope_neox(q_, kb, B, Hkv, D, D, cos_ + rope_off, sin_ + rope_off);
    stage("q", qb, size_t(B) * H * D);
    stage("k", kb, size_t(B) * Hkv * D);
    stage("v", vb, size_t(B) * Hkv * D);
    float attn_scale = scale;
    if (emulate_) {  // the cache is fp16, and attention takes q * scale as fp16
      ref::round_acts(q_, kb, int64_t(B) * Hkv, D, true, kb);
      ref::round_acts(q_, vb, int64_t(B) * Hkv, D, true, vb);
      ref::scale(q_, qb, scale, int64_t(B) * H * D);
      ref::round_acts(q_, qb, int64_t(B) * H, D, true, qb);
      attn_scale = 1.0f;
    }
    q_.memcpy(k_[size_t(l)] + pos * Hkv * D, kb, size_t(B) * Hkv * D * sizeof(float));
    q_.memcpy(v_[size_t(l)] + pos * Hkv * D, vb, size_t(B) * Hkv * D * sizeof(float));
    const bool full_layer = c_.full_layer(l);
    ref::attention_ranges(q_, qb, k_[size_t(l)], v_[size_t(l)], 0, B, H, Hkv, D, full_layer ? lo_full : lo, full_layer ? hi_full : hi, attn_scale, ab);
    if (debug) {
      debug->parts.emplace_back(size_t(B) * H * D);
      q_.memcpy(debug->parts.back().data(), ab, size_t(B) * H * D * sizeof(float)).wait();
    }
    mm(L.wo, ab, B, ob);
    if (!c_.plain) {
      ref::grouped_conv(q_, ob, dyn, L.base_attn, 1, B, B, E, c_.conv_kernel, c_.conv_group, ob2);
      ref::add(q_, x, ob2, int64_t(B) * E, x);
    } else {
      ref::add(q_, x, ob, int64_t(B) * E, x);
    }
    stage("x_mid", x, size_t(B) * E);
    ref::rmsnorm(q_, x, L.ffn_norm, B, E, c_.eps, n1);
    if (!c_.plain) {
      mm(L.conv_ffn, n1, B, dyn);
      ref::grouped_conv(q_, n1, dyn, L.base_ffn, 0, B, B, E, c_.conv_kernel, c_.conv_group, cv);
    } else {
      q_.memcpy(cv, n1, size_t(B) * E * sizeof(float));
    }
    stage("conv2", cv, size_t(B) * E);
    mm(L.gate, cv, B, g);
    mm(L.up, cv, B, u);
    ref::silu_mul(q_, g, u, int64_t(B) * c_.n_ff, g);
    stage("glu", g, size_t(B) * c_.n_ff);
    mm(L.down, g, B, ob);
    stage("down", ob, size_t(B) * E);
    if (!c_.plain) {
      ref::grouped_conv(q_, ob, dyn, L.base_ffn, 1, B, B, E, c_.conv_kernel, c_.conv_group, ob2);
      ref::add(q_, x, ob2, int64_t(B) * E, x);
    } else {
      ref::add(q_, x, ob, int64_t(B) * E, x);
    }
    if (debug) {
      debug->layers.emplace_back(size_t(B) * E);
      q_.memcpy(debug->layers.back().data(), x, size_t(B) * E * sizeof(float)).wait();
    }
  }
  ref::rmsnorm(q_, x, w_.out_norm, B, E, c_.eps, x);
  // Positions 1..B-1: logits through the target's head, the top K, the hidden projection, the codebooks' rows.
  const float* hid = x + E;
  float* logits = fdev(size_t(P) * c_.n_vocab);
  mm(w_.head, hid, P, logits);
  int32_t* cand = sycl::malloc_device<int32_t>(size_t(P) * K, q_);
  int32_t* pred_ids = sycl::malloc_device<int32_t>(size_t(P) * K, q_);
  frees.insert(frees.end(), {cand, pred_ids});
  float* unary = fdev(size_t(P) * K);
  ref::topk_rows(q_, logits, P, c_.n_vocab, K, cand, unary);
  if (c_.plain) {  // no selector: each position's own top candidates, softcapped
    serve::Lattice out;
    out.top_k = K;
    out.cand.resize(size_t(P) * K);
    out.unary.resize(size_t(P) * K);
    q_.memcpy(out.cand.data(), cand, out.cand.size() * sizeof(int32_t));
    q_.memcpy(out.unary.data(), unary, out.unary.size() * sizeof(float));
    if (debug) {
      debug->hidden.resize(size_t(P) * E);
      q_.memcpy(debug->hidden.data(), hid, debug->hidden.size() * sizeof(float));
    }
    q_.wait();
    plain_lattice(out, c_.softcap);
    for (void* p : frees) sycl::free(p, q_);
    return out;
  }
  float* gate = fdev(size_t(P) * R);
  mm(w_.sel_hidden, hid, P, gate);
  // Predecessor ids: position 0's anchor (the first of its K slots), position i's the candidates of position i - 1.
  std::vector<int32_t> hcand(size_t(P) * K), hpred(size_t(P) * K, anchor);
  q_.memcpy(hcand.data(), cand, hcand.size() * sizeof(int32_t)).wait();
  for (int i = 1; i < P; ++i)
    for (int a = 0; a < K; ++a) hpred[size_t(i) * K + a] = hcand[size_t(i - 1) * K + a];
  q_.memcpy(pred_ids, hpred.data(), hpred.size() * sizeof(int32_t));
  float *pred = fdev(size_t(P) * K * R), *succ = fdev(size_t(P) * K * R), *lat = fdev(size_t(P) * K * K);
  ref::embed(q_, w_.sel_pred, pred_ids, int64_t(P) * K, 1.0f, pred);
  ref::embed(q_, w_.sel_succ, cand, int64_t(P) * K, 1.0f, succ);
  ref::selector_lattice(q_, unary, gate, pred, succ, P, K, R, lat);
  serve::Lattice out;
  out.top_k = K;
  out.cand = hcand;
  out.unary.resize(size_t(P) * K);
  out.score.resize(size_t(P) * K * K);
  q_.memcpy(out.unary.data(), unary, out.unary.size() * sizeof(float));
  q_.memcpy(out.score.data(), lat, out.score.size() * sizeof(float));
  if (debug) {
    debug->hidden.resize(size_t(P) * E);
    q_.memcpy(debug->hidden.data(), hid, debug->hidden.size() * sizeof(float));
  }
  q_.wait();
  for (void* p : frees) sycl::free(p, q_);
  return out;
}

}  // namespace eklow::model
