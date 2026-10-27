#include "model/embed_gemma2.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace eklow::model {

namespace {

int64_t get_int(const gguf::File& f, const std::string& k) { return f.at(k).as_int(); }
float get_float(const gguf::File& f, const std::string& k) { return float(f.at(k).as_float()); }

// Attention takes an input in chunks of up to kAttnChunk tokens, and a pass at most kPassInputs inputs: they bound
// its scratch (the global layers' segment buffers grow with a chunk's rows, the query tiles with the chunks).
constexpr int64_t kAttnChunk = 2048, kPassInputs = 256;

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
    q_.memcpy(d, t.data, t.nbytes).wait();
    return d;
  };
  auto table = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    uint8_t* d = upload(t);
    owned_.push_back(d);
    bytes_ += t.nbytes;
    return ref::Weight{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), d, t.nbytes / t.ne[1]};
  };
  // A matrix for the GEMMs: uploaded as stored, repacked into the xmx tiles, the stored copy freed.
  auto matrix = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    uint8_t* d = upload(t);
    const xmx::Weight w = xmx::repack(q_, ref::Weight{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), d, t.nbytes / t.ne[1]});
    sycl::free(d, q_);
    repacked_.push_back(w);
    bytes_ += xmx::weight_tile_bytes(w.k, w.n) * (w.format == xmx::WeightFormat::Q4_0 ? 1 : w.format == xmx::WeightFormat::Q8_0 ? 2 : 4) +
              (w.scales ? xmx::weight_scale_bytes(w.k, w.n) : 0);
    return w;
  };
  auto vec = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    uint8_t* d = upload(t);
    owned_.push_back(d);
    bytes_ += t.nbytes;
    return reinterpret_cast<const float*>(d);
  };

  tok_embd_ = table("token_embd.weight");
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
  for (auto& w : repacked_) xmx::free_weight(q_, w);
  for (float* p : act_) sycl::free(p, q_);
  xmx::free_activations(q_, a_);
  xmx::free_activations(q_, a0_);
  for (void* p : {kc_, vc_, static_cast<void*>(table_), static_cast<void*>(scratch_.q), static_cast<void*>(scratch_.part),
                  static_cast<void*>(scratch_.part2), static_cast<void*>(scratch_.p), static_cast<void*>(scratch_.stat)})
    if (p) sycl::free(p, q_);
}

// Activation buffers for passes of n_tok tokens: x, x2, h, o, q, k, v, att, gate, up, the per-layer gate and input,
// the last layer, the token ids and positions, the inputs' means; the RoPE tables for positions 0..n_tok-1 (cos and
// sin of the sliding layers, then of the global ones); the attention's slots, cache positions and key ranges (sliding,
// global) and tile lists; the GEMMs' fp16 inputs; the attention's cache, page table and scratch.
void EmbedGemma2::ensure(int64_t n_tok) {
  if (n_tok <= max_tok_) return;
  q_.wait();
  for (float* p : act_) sycl::free(p, q_);
  act_.clear();
  xmx::free_activations(q_, a_);
  xmx::free_activations(q_, a0_);
  for (void* p : {kc_, vc_, static_cast<void*>(table_)})
    if (p) sycl::free(p, q_);
  int64_t dmax = 0, d_s = 0, d_g = 0, hkv = 0, hkv_s = 1, hkv_g = 1;
  for (const auto& L : layers_) {
    dmax = std::max(dmax, L.head_dim);
    hkv = std::max(hkv, L.n_head_kv);
    (L.sliding ? d_s : d_g) = L.head_dim, (L.sliding ? hkv_s : hkv_g) = L.n_head_kv;
  }
  const int64_t T = n_tok, E = n_embd_;
  // A pass's query tiles on either kind of layer: per chunk and K/V head ceil(rows / 8), rows = tokens x the query
  // heads per K/V head; each tile takes 4 ints of the list and at most 2 of its groups.
  const int64_t max_tiles = (T * n_head_ + 7) / 8 + hkv * (kPassInputs + T / kAttnChunk);
  for (int64_t n : {T * E, T * E, T * E, T * E, T * n_head_ * dmax, T * hkv * dmax, T * hkv * dmax, T * n_head_ * dmax, T * n_ff_,
                    T * n_ff_, T * n_pl_, T * n_pl_, T * n_out_, T, T, T * n_out_, T * d_s / 2, T * d_s / 2, T * d_g / 2,
                    T * d_g / 2, T, T, T, T, T, T, 6 * max_tiles, 6 * max_tiles}) {
    auto* p = sycl::malloc_device<float>(size_t(n), q_);
    if (!p) throw std::runtime_error("embedding model: out of device memory for activations");
    act_.push_back(p);
  }
  const ref::RopeTables s = ref::rope_tables(T, d_s, rope_base_s_, nullptr), g = ref::rope_tables(T, d_g, rope_base_g_, nullptr);
  q_.memcpy(act_[16], s.cos.data(), s.cos.size() * sizeof(float));
  q_.memcpy(act_[17], s.sin.data(), s.sin.size() * sizeof(float));
  q_.memcpy(act_[18], g.cos.data(), g.cos.size() * sizeof(float));
  q_.memcpy(act_[19], g.sin.data(), g.sin.size() * sizeof(float)).wait();
  a_ = xmx::alloc_activations(q_, T, std::max({E, n_ff_, n_head_ * dmax, n_pl_}));
  a0_ = xmx::alloc_activations(q_, T, E);
  // One slot of pages_ pages in order (the page table is the identity), [page][K/V head][256][dmax] fp16 each.
  pages_ = (T + attn::kPage - 1) / attn::kPage;
  const size_t cache = size_t(pages_ * hkv * attn::kPage * dmax);
  kc_ = sycl::malloc_device<sycl::half>(cache, q_);
  vc_ = sycl::malloc_device<sycl::half>(cache, q_);
  table_ = sycl::malloc_device<int32_t>(size_t(pages_), q_);
  if (!kc_ || !vc_ || !table_) throw std::runtime_error("embedding model: out of device memory for the attention cache");
  std::vector<int32_t> identity(static_cast<size_t>(pages_));
  for (int64_t i = 0; i < pages_; ++i) identity[size_t(i)] = int32_t(i);
  q_.memcpy(table_, identity.data(), identity.size() * sizeof(int32_t)).wait();
  const int64_t chunk = std::min(T, kAttnChunk);
  attn::reserve(q_, scratch_, max_tiles, chunk * (n_head_ / hkv_s), hkv_s, d_s);
  attn::reserve(q_, scratch_, max_tiles, chunk * (n_head_ / hkv_g), hkv_g, d_g);
  max_tok_ = T;
}

std::vector<float> EmbedGemma2::embed(const std::vector<int32_t>& tokens) {
  return std::move(embed(std::vector<std::vector<int32_t>>{tokens})[0]);
}

// Passes of consecutive inputs up to max_tok_ tokens each, their means copied to the host, and a single wait at the
// end: with a long prefill on the card, every wait costs a turn on it (DECISIONS 2026-10-26, "embeddings beside a
// long prefill").
std::vector<std::vector<float>> EmbedGemma2::embed(const std::vector<std::vector<int32_t>>& seqs) {
  int64_t longest = 0;
  for (const auto& s : seqs) {
    if (s.empty()) throw std::runtime_error("embedding model: empty input");
    longest = std::max(longest, int64_t(s.size()));
  }
  ensure(longest);
  std::vector<float> means(seqs.size() * size_t(n_out_));
  std::vector<std::vector<int32_t>> keep;
  try {
    for (size_t i0 = 0; i0 < seqs.size();) {
      size_t i1 = i0;
      int64_t tokens = 0;
      while (i1 < seqs.size() && i1 - i0 < size_t(kPassInputs) && tokens + int64_t(seqs[i1].size()) <= max_tok_)
        tokens += int64_t(seqs[i1++].size());
      forward(seqs, i0, i1, means.data() + i0 * size_t(n_out_), keep);
      i0 = i1;
    }
  } catch (...) {
    q_.wait();  // copies already queued write into means
    throw;
  }
  q_.wait();
  std::vector<std::vector<float>> out;
  for (size_t i = 0; i < seqs.size(); ++i) {
    const float* m = means.data() + i * size_t(n_out_);
    double norm = 0;
    for (int64_t j = 0; j < n_out_; ++j) norm += double(m[j]) * double(m[j]);
    norm = std::sqrt(norm);
    std::vector<float> e(static_cast<size_t>(n_out_));
    for (int64_t j = 0; j < n_out_; ++j) e[size_t(j)] = float(norm > 0 ? double(m[j]) / norm : 0.0);
    out.push_back(std::move(e));
  }
  return out;
}

void EmbedGemma2::forward(const std::vector<std::vector<int32_t>>& seqs, size_t i0, size_t i1, float* means,
                          std::vector<std::vector<int32_t>>& keep) {
  const int64_t E = n_embd_, H = n_head_, L = n_layer_, n_seq = int64_t(i1 - i0);
  std::vector<int32_t> tokens, pos;
  std::vector<int64_t> start{0};
  for (size_t i = i0; i < i1; ++i) {
    for (size_t j = 0; j < seqs[i].size(); ++j) tokens.push_back(seqs[i][j]), pos.push_back(int32_t(j));
    start.push_back(int64_t(tokens.size()));
  }
  const int64_t T = int64_t(tokens.size());
  float *x = act_[0], *x2 = act_[1], *h = act_[2], *o = act_[3], *qb = act_[4], *kb = act_[5], *vb = act_[6], *att = act_[7];
  float *gate = act_[8], *up = act_[9], *pg = act_[10], *pl = act_[11], *emb = act_[12], *mean = act_[15];
  auto* tok = reinterpret_cast<int32_t*>(act_[13]);
  auto* tpos = reinterpret_cast<int32_t*>(act_[14]);
  const float *cos_s = act_[16], *sin_s = act_[17], *cos_g = act_[18], *sin_g = act_[19];
  q_.memcpy(tok, tokens.data(), tokens.size() * sizeof(int32_t));
  q_.memcpy(tpos, pos.data(), pos.size() * sizeof(int32_t));
  keep.push_back(std::move(tokens));
  keep.push_back(std::move(pos));

  // Attention: the pass's inputs back to back in one slot (cache position = row), each input in chunks whose tokens
  // see their own input, within window/2 on the sliding layers. Every Hkv and head size is the same within a kind.
  int64_t hkv_s = 0, hkv_g = 0, d_s = 0, d_g = 0;
  for (const auto& Ly : layers_) (Ly.sliding ? hkv_s : hkv_g) = Ly.n_head_kv, (Ly.sliding ? d_s : d_g) = Ly.head_dim;
  const int64_t hw = window_ / 2;
  const size_t nt = static_cast<size_t>(T);
  std::vector<int32_t> slot(nt, 0), cpos(nt), lo_s(nt), hi_s(nt), lo_g(nt), hi_g(nt);
  attn::TileList tl_s(d_s), tl_g(d_g);
  for (int64_t s = 0; s < n_seq; ++s) {
    const int64_t a = start[size_t(s)], b = start[size_t(s) + 1];
    for (int64_t t = a; t < b; ++t) {
      cpos[size_t(t)] = int32_t(t);
      lo_s[size_t(t)] = int32_t(std::max(a, t - hw)), hi_s[size_t(t)] = int32_t(std::min(b - 1, t + hw));
      lo_g[size_t(t)] = int32_t(a), hi_g[size_t(t)] = int32_t(b - 1);
    }
    for (int64_t c0 = a; c0 < b; c0 += kAttnChunk) {
      const int64_t n = std::min(kAttnChunk, b - c0);
      if (hkv_s) tl_s.add_chunk(int32_t(c0), n, c0 + n - 1, H / hkv_s, hkv_s, 0, std::max(a, c0 - hw), std::min(b - 1, c0 + n - 1 + hw));
      if (hkv_g) tl_g.add_chunk(int32_t(c0), n, c0 + n - 1, H / hkv_g, hkv_g, 0, a, b - 1);
    }
  }
  auto* d_slot = reinterpret_cast<int32_t*>(act_[20]);
  auto* d_cpos = reinterpret_cast<int32_t*>(act_[21]);
  auto* d_lo_s = reinterpret_cast<int32_t*>(act_[22]);
  auto* d_hi_s = reinterpret_cast<int32_t*>(act_[23]);
  auto* d_lo_g = reinterpret_cast<int32_t*>(act_[24]);
  auto* d_hi_g = reinterpret_cast<int32_t*>(act_[25]);
  auto* d_tl_s = reinterpret_cast<int32_t*>(act_[26]);
  auto* d_tl_g = reinterpret_cast<int32_t*>(act_[27]);
  for (auto [dst, v] : {std::pair{d_slot, &slot}, {d_cpos, &cpos}, {d_lo_s, &lo_s}, {d_hi_s, &hi_s}, {d_lo_g, &lo_g}, {d_hi_g, &hi_g}}) {
    q_.memcpy(dst, v->data(), v->size() * sizeof(int32_t));
    keep.push_back(std::move(*v));
  }
  keep.push_back(tl_s.packed());
  q_.memcpy(d_tl_s, keep.back().data(), keep.back().size() * sizeof(int32_t));
  keep.push_back(tl_g.packed());
  q_.memcpy(d_tl_g, keep.back().data(), keep.back().size() * sizeof(int32_t));
  const attn::Tokens tk_s{d_slot, d_cpos, T, d_lo_s, d_hi_s}, tk_g{d_slot, d_cpos, T, d_lo_g, d_hi_g};
  // Rows of the GEMMs: xmx::quantize writes them in the fp16 form at width k.
  auto gemm_in = [&](xmx::Activations& a, const float* v, int64_t k) {
    a.k = k;
    xmx::quantize(q_, v, T, a, true);
  };
  // NeoX RoPE over all of each head, at each row's position within its input.
  auto rope = [&](float* v, int64_t n_head, int64_t D, const float* cos_tab, const float* sin_tab) {
    const int64_t half = D / 2;
    q_.parallel_for(sycl::range<3>(size_t(T), size_t(n_head), size_t(half)), [=](sycl::id<3> id) {
      const int64_t t = int64_t(id[0]), hh = int64_t(id[1]), i = int64_t(id[2]), p = tpos[t];
      const float c = cos_tab[p * half + i], s = sin_tab[p * half + i];
      float* r = v + (t * n_head + hh) * D;
      const float a = r[i], b = r[i + half];
      r[i] = a * c - b * s;
      r[i + half] = a * s + b * c;
    });
  };

  ref::embed(q_, tok_embd_, tok, T, std::sqrt(float(E)), x);
  gemm_in(a0_, x, E);
  for (int64_t l = 0; l < L; ++l) {
    const Layer& Ly = layers_[size_t(l)];
    const int64_t D = Ly.head_dim, Hkv = Ly.n_head_kv;
    ref::rmsnorm(q_, x, Ly.attn_norm, T, E, eps_, h);
    gemm_in(a_, h, E);
    xmx::matmul(q_, a_, Ly.wq, qb);
    xmx::matmul(q_, a_, Ly.wk, kb);
    xmx::matmul(q_, a_, Ly.wv, vb);
    ref::rmsnorm(q_, qb, Ly.q_norm, T * H, D, eps_, qb);
    ref::rmsnorm(q_, kb, Ly.k_norm, T * Hkv, D, eps_, kb);
    ref::rmsnorm(q_, vb, nullptr, T * Hkv, D, eps_, vb);
    rope(qb, H, D, Ly.sliding ? cos_s : cos_g, Ly.sliding ? sin_s : sin_g);
    rope(kb, Hkv, D, Ly.sliding ? cos_s : cos_g, Ly.sliding ? sin_s : sin_g);
    const attn::Cache cache{kc_, vc_, table_, pages_, 0, Hkv, D};
    attn::write(q_, kb, vb, Ly.sliding ? tk_s : tk_g, cache);
    attn::attend(q_, qb, Ly.sliding ? tk_s : tk_g, Ly.sliding ? tl_s : tl_g, Ly.sliding ? d_tl_s : d_tl_g, cache, H, 1.0f, att, scratch_);
    gemm_in(a_, att, H * D);
    xmx::matmul(q_, a_, Ly.wo, o);
    ref::rmsnorm(q_, o, Ly.attn_post_norm, T, E, eps_, o);
    ref::add(q_, o, x, T * E, x2);

    ref::rmsnorm(q_, x2, Ly.ffn_norm, T, E, eps_, h);
    gemm_in(a_, h, E);
    xmx::matmul(q_, a_, Ly.gate, gate);
    xmx::matmul(q_, a_, Ly.up, up);
    ref::gelu_mul(q_, gate, up, T * n_ff_, gate);
    gemm_in(a_, gate, n_ff_);
    xmx::matmul(q_, a_, Ly.down, o);
    ref::rmsnorm(q_, o, Ly.ffn_post_norm, T, E, eps_, o);
    ref::add(q_, o, x2, T * E, x2);

    // The per-layer input: this layer's rows of per_layer_model_proj over the scaled embeddings, made here rather
    // than for every layer up front (all 24 layers' took 400 MB at 8,192 tokens).
    xmx::matmul(q_, a0_, xmx::rows_of(pl_proj_, l * n_pl_, n_pl_), pl);
    ref::scale(q_, pl, 1.0f / std::sqrt(float(E)), T * n_pl_);
    ref::rmsnorm(q_, pl, pl_norm_, T, n_pl_, eps_, pl);
    gemm_in(a_, x2, E);
    xmx::matmul(q_, a_, Ly.inp_gate, pg);
    ref::gelu(q_, pg, T * n_pl_, pg);
    ref::mul(q_, pg, pl, T * n_pl_, pg);
    gemm_in(a_, pg, n_pl_);
    xmx::matmul(q_, a_, Ly.proj, o);
    ref::rmsnorm(q_, o, Ly.post_norm, T, E, eps_, o);
    ref::add(q_, o, x2, T * E, x);
    ref::scale(q_, x, Ly.out_scale, T * E);
  }

  ref::rmsnorm(q_, x, output_norm_, T, E, eps_, h);
  gemm_in(a_, h, E);
  xmx::matmul(q_, a_, output_, emb);
  // Each input's mean over its rows.
  auto* st = reinterpret_cast<int32_t*>(tpos);  // the positions are spent: reuse them for the inputs' starts
  keep.emplace_back(start.begin(), start.end());
  q_.memcpy(st, keep.back().data(), keep.back().size() * sizeof(int32_t));
  const int64_t D = n_out_;
  q_.parallel_for(sycl::range<2>(size_t(n_seq), size_t(D)), [=](sycl::id<2> id) {
    const int64_t s = int64_t(id[0]), j = int64_t(id[1]), t0 = st[s], t1 = st[s + 1];
    float acc = 0;
    for (int64_t t = t0; t < t1; ++t) acc += emb[t * D + j];
    mean[s * D + j] = acc / float(t1 - t0);
  });
  q_.memcpy(means, mean, size_t(n_seq * D) * sizeof(float));
}

}  // namespace eklow::model
