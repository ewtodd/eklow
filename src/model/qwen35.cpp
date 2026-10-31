#include "model/qwen35.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace eklow::model {

namespace {

int64_t get_int(const gguf::File& f, const std::string& key) { return f.at(key).as_int(); }

[[noreturn]] void unsupported(const gguf::File& f, const std::string& what) {
  throw std::runtime_error(f.path() + ": not a qwen35moe this engine runs: " + what);
}

uint64_t row_bytes_of(gguf::Type t, int64_t k) {
  auto tr = gguf::traits(t);
  if (!tr || k % tr->block) throw std::runtime_error("bad row length for a quantized type");
  return uint64_t(k / tr->block) * tr->bytes;
}

}  // namespace

Qwen35Config Qwen35Config::from_gguf(const gguf::File& f) {
  if (f.at("general.architecture").as_string() != "qwen35moe") unsupported(f, "architecture is not qwen35moe");
  Qwen35Config c{};
  const std::string p = "qwen35moe.";
  const int all = int(get_int(f, p + "block_count"));
  const int nextn = f.find(p + "nextn_predict_layers") ? int(get_int(f, p + "nextn_predict_layers")) : 0;
  c.n_layer = all - nextn;
  c.n_embd = int(get_int(f, p + "embedding_length"));
  c.n_head = int(get_int(f, p + "attention.head_count"));
  const gguf::Value& kv = f.at(p + "attention.head_count_kv");
  if (kv.kind == gguf::Value::Kind::Array) {
    for (const auto& v : kv.as_array()) c.n_head_kv = std::max(c.n_head_kv, int(v.as_int()));
  } else {
    c.n_head_kv = int(kv.as_int());
  }
  c.head_dim = int(get_int(f, p + "attention.key_length"));
  if (get_int(f, p + "attention.value_length") != c.head_dim) unsupported(f, "value heads differ from key heads");
  c.n_rot = int(get_int(f, p + "rope.dimension_count"));
  c.rope_base = float(f.at(p + "rope.freq_base").as_float());
  c.eps = float(f.at(p + "attention.layer_norm_rms_epsilon").as_float());
  if (auto* v = f.find(p + "expert_weights_scale")) c.expert_weights_scale = float(v->as_float());
  c.n_expert = int(get_int(f, p + "expert_count"));
  c.n_expert_used = int(get_int(f, p + "expert_used_count"));
  c.n_ff_exp = int(get_int(f, p + "expert_feed_forward_length"));
  c.n_ff_shexp = int(get_int(f, p + "expert_shared_feed_forward_length"));
  c.ssm_conv = int(get_int(f, p + "ssm.conv_kernel"));
  c.ssm_inner = int(get_int(f, p + "ssm.inner_size"));
  c.ssm_head_dim = int(get_int(f, p + "ssm.state_size"));
  c.ssm_k_heads = int(get_int(f, p + "ssm.group_count"));
  c.ssm_v_heads = int(get_int(f, p + "ssm.time_step_rank"));
  if (c.n_layer <= 0 || c.n_embd <= 0 || c.n_head <= 0 || c.n_head_kv <= 0 || c.head_dim <= 0 || c.n_expert <= 0 ||
      c.n_expert_used <= 0 || c.n_ff_exp <= 0 || c.n_ff_shexp <= 0)
    unsupported(f, "metadata");
  if (c.ssm_inner != c.ssm_v_heads * c.ssm_head_dim || c.ssm_v_heads % c.ssm_k_heads != 0)
    unsupported(f, "DeltaNet head layout (value heads must be a multiple of key heads, all of one size)");
  if (c.n_head % c.n_head_kv != 0 || c.n_rot > c.head_dim || c.n_rot % 2) unsupported(f, "attention layout");

  // IMRoPE for text: every position component equals the token's position, so the rotation is plain NeoX
  // over n_rot dims as long as the sections cover all n_rot/2 frequencies and none is left to the 4th component.
  const auto& sec = f.at(p + "rope.dimension_sections").as_array();
  int sum = 0;
  for (size_t i = 0; i < sec.size(); ++i) {
    if (i == 3 && sec[i].as_int() != 0) unsupported(f, "a fourth rope section");
    sum += int(sec[i].as_int());
  }
  if (sum != c.n_rot / 2) unsupported(f, "rope sections that do not cover the rotated dims");

  if (const gguf::Value* rl = f.find(p + "attention.recurrent_layers")) {
    for (const auto& v : rl->as_array()) c.recurrent.push_back(v.kind == gguf::Value::Kind::Bool ? std::get<bool>(v.v) : v.as_int() != 0);
    c.recurrent.resize(size_t(c.n_layer));
  } else {
    const int interval = f.find(p + "full_attention_interval") ? int(get_int(f, p + "full_attention_interval")) : 4;
    for (int i = 0; i < c.n_layer; ++i) c.recurrent.push_back((i + 1) % interval != 0);
  }
  const gguf::Tensor* emb = f.tensor("token_embd.weight");
  if (!emb || emb->ne.size() != 2) unsupported(f, "no token_embd.weight");
  c.n_vocab = int(emb->ne[1]);
  return c;
}

Qwen35Layer& Qwen35Layer::operator=(Qwen35Layer&& o) noexcept {
  if (this != &o) {
    for (void* p : allocations) sycl::free(p, *q);
    wqkv = o.wqkv, wz = o.wz, w_beta = o.w_beta, w_alpha = o.w_alpha, ssm_out = o.ssm_out;
    conv1d = o.conv1d, dt_bias = o.dt_bias, ssm_a = o.ssm_a, ssm_norm = o.ssm_norm;
    wq = o.wq, wk = o.wk, wv = o.wv, wo = o.wo, q_norm = o.q_norm, k_norm = o.k_norm;
    attn_norm = o.attn_norm, post_norm = o.post_norm;
    router = o.router, shexp_gate_inp = o.shexp_gate_inp, shexp_gate = o.shexp_gate, shexp_up = o.shexp_up;
    shexp_down = o.shexp_down;
    exp_gate_up = o.exp_gate_up, exp_gate = o.exp_gate, exp_up = o.exp_up, exp_down = o.exp_down;
    allocations = std::move(o.allocations);
    o.allocations.clear();
    q = o.q;
  }
  return *this;
}

Qwen35Layer::~Qwen35Layer() {
  for (void* p : allocations) sycl::free(p, *q);
}

Qwen35Weights::Qwen35Weights(Qwen35Weights&& o) noexcept
    : tok_embd(o.tok_embd), output(o.output), output_norm(o.output_norm), bytes(o.bytes), f_(o.f_), c_(o.c_), q_(o.q_),
      allocations_(std::move(o.allocations_)) {
  o.allocations_.clear();
}

Qwen35Weights::~Qwen35Weights() {
  for (void* p : allocations_) sycl::free(p, *q_);
}

namespace {

const gguf::Tensor& need(const gguf::File& f, const std::string& name) {
  const gguf::Tensor* t = f.tensor(name);
  if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
  return *t;
}

// Copies a tensor to the card and returns it as a matrix of `k` columns and `n` rows (n = rows of one
// expert for a 3-d tensor, whose ne[2] must be `experts`).
ref::Weight upload_matrix(const gguf::File& f, sycl::queue& q, std::vector<void*>& owned, const std::string& name,
                          int64_t k, int64_t n, int64_t experts = 0, uint64_t* bytes = nullptr) {
  const gguf::Tensor& t = need(f, name);
  const bool ok = experts ? t.ne.size() == 3 && int64_t(t.ne[2]) == experts : (t.ne.size() == 2 || (t.ne.size() == 1 && n == 1));
  if (!ok || int64_t(t.ne[0]) != k || (t.ne.size() > 1 && int64_t(t.ne[1]) != n))
    throw std::runtime_error(f.path() + ": " + name + " has an unexpected shape");
  auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q);
  if (!d) throw std::runtime_error("out of device memory uploading " + name);
  owned.push_back(d);
  q.memcpy(d, t.data, t.nbytes).wait();
  if (bytes) *bytes += t.nbytes;
  return ref::Weight{t.type, k, n, d, row_bytes_of(t.type, k)};
}

const float* upload_vec(const gguf::File& f, sycl::queue& q, std::vector<void*>& owned, const std::string& name,
                        int64_t n, uint64_t* bytes = nullptr) {
  const gguf::Tensor& t = need(f, name);
  if (t.type != gguf::Type::F32 || int64_t(t.elements()) != n)
    throw std::runtime_error(f.path() + ": " + name + " is not an F32 vector of " + std::to_string(n));
  auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q);
  if (!d) throw std::runtime_error("out of device memory uploading " + name);
  owned.push_back(d);
  q.memcpy(d, t.data, t.nbytes).wait();
  if (bytes) *bytes += t.nbytes;
  return reinterpret_cast<const float*>(d);
}

}  // namespace

Qwen35Weights Qwen35Weights::upload(const gguf::File& f, const Qwen35Config& c, sycl::queue& q) {
  Qwen35Weights w(f, c, q);
  w.tok_embd = upload_matrix(f, q, w.allocations_, "token_embd.weight", c.n_embd, c.n_vocab, 0, &w.bytes);
  w.output_norm = upload_vec(f, q, w.allocations_, "output_norm.weight", c.n_embd, &w.bytes);
  w.output = upload_matrix(f, q, w.allocations_, "output.weight", c.n_embd, c.n_vocab, 0, &w.bytes);
  for (int l = 0; l < c.n_layer; ++l)  // fail now, not at layer 30, on a file with a tensor missing
    need(f, "blk." + std::to_string(l) + (c.recurrent[size_t(l)] ? ".ssm_out.weight" : ".attn_output.weight"));
  return w;
}

Qwen35Layer Qwen35Weights::load_layer(int l) const {
  const Qwen35Config& c = *c_;
  const gguf::File& f = *f_;
  sycl::queue& q = *q_;
  const std::string b = "blk." + std::to_string(l) + ".";
  Qwen35Layer L{};
  L.q = &q;
  auto& own = L.allocations;
  auto mat = [&](const std::string& n, int64_t k, int64_t rows, int64_t experts = 0) {
    return upload_matrix(f, q, own, b + n, k, rows, experts);
  };
  const int64_t E = c.n_embd;
  L.attn_norm = upload_vec(f, q, own, b + "attn_norm.weight", E);
  L.post_norm = upload_vec(f, q, own, b + "post_attention_norm.weight", E);
  if (c.recurrent[size_t(l)]) {
    const int64_t conv = c.conv_dim();
    L.wqkv = mat("attn_qkv.weight", E, conv);
    L.wz = mat("attn_gate.weight", E, c.ssm_inner);
    L.w_beta = mat("ssm_beta.weight", E, c.ssm_v_heads);
    L.w_alpha = mat("ssm_alpha.weight", E, c.ssm_v_heads);
    L.ssm_out = mat("ssm_out.weight", c.ssm_inner, E);
    L.conv1d = upload_vec(f, q, own, b + "ssm_conv1d.weight", conv * c.ssm_conv);
    L.dt_bias = upload_vec(f, q, own, b + "ssm_dt.bias", c.ssm_v_heads);
    L.ssm_a = upload_vec(f, q, own, b + "ssm_a", c.ssm_v_heads);
    L.ssm_norm = upload_vec(f, q, own, b + "ssm_norm.weight", c.ssm_head_dim);
  } else {
    const int64_t D = c.head_dim;
    L.wq = mat("attn_q.weight", E, 2 * D * c.n_head);
    L.wk = mat("attn_k.weight", E, D * c.n_head_kv);
    L.wv = mat("attn_v.weight", E, D * c.n_head_kv);
    L.wo = mat("attn_output.weight", D * c.n_head, E);
    L.q_norm = upload_vec(f, q, own, b + "attn_q_norm.weight", D);
    L.k_norm = upload_vec(f, q, own, b + "attn_k_norm.weight", D);
  }
  L.router = mat("ffn_gate_inp.weight", E, c.n_expert);
  L.shexp_gate_inp = mat("ffn_gate_inp_shexp.weight", E, 1);
  L.shexp_gate = mat("ffn_gate_shexp.weight", E, c.n_ff_shexp);
  L.shexp_up = mat("ffn_up_shexp.weight", E, c.n_ff_shexp);
  L.shexp_down = mat("ffn_down_shexp.weight", c.n_ff_shexp, E);
  if (f.tensor(b + "ffn_gate_up_exps.weight")) {
    L.exp_gate_up = mat("ffn_gate_up_exps.weight", E, 2 * c.n_ff_exp, c.n_expert);
  } else {
    L.exp_gate = mat("ffn_gate_exps.weight", E, c.n_ff_exp, c.n_expert);
    L.exp_up = mat("ffn_up_exps.weight", E, c.n_ff_exp, c.n_expert);
  }
  L.exp_down = mat("ffn_down_exps.weight", c.n_ff_exp, E, c.n_expert);
  return L;
}

std::vector<float> reference_forward(sycl::queue& q, const Qwen35Config& c, const Qwen35Weights& w,
                                     const std::vector<int32_t>& tokens, int64_t logits_rows,
                                     const DumpSink* dump, std::vector<int32_t>* routing) {
  const int64_t T = int64_t(tokens.size()), E = c.n_embd, H = c.n_head, Hkv = c.n_head_kv, D = c.head_dim;
  const int64_t Fe = c.n_ff_exp, Fs = c.n_ff_shexp, NE = c.n_expert, K = c.n_expert_used;
  const int64_t CD = c.conv_dim(), HK = c.ssm_k_heads, HV = c.ssm_v_heads, S = c.ssm_head_dim;
  const int64_t KEY = HK * S, VAL = c.ssm_inner;
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
      q.wait();
      for (void* p : v) sycl::free(p, q);
    }
  } free_all{q, owned};

  auto* tok = sycl::malloc_device<int32_t>(size_t(T), q);
  owned.push_back(tok);
  q.memcpy(tok, tokens.data(), size_t(T) * sizeof(int32_t)).wait();
  float *x = alloc(T * E), *x2 = alloc(T * E), *h = alloc(T * E), *o = alloc(T * E), *o2 = alloc(T * E);
  float *qkv = alloc(T * CD), *conv = alloc(T * CD), *z = alloc(T * VAL), *beta = alloc(T * HV), *alpha = alloc(T * HV);
  float *qn = alloc(T * KEY), *kn = alloc(T * KEY), *dout = alloc(T * VAL), *state = alloc(HV * S * S);
  float *qfull = alloc(T * H * 2 * D), *qb = alloc(T * H * D), *kb = alloc(T * Hkv * D), *vb = alloc(T * Hkv * D);
  float *gate = alloc(T * H * D), *att = alloc(T * H * D);
  float *rlog = alloc(T * NE), *sg = alloc(T * Fs), *su = alloc(T * Fs), *sgate = alloc(T);
  float *moe = alloc(T * E), *gx = alloc(T * E), *gu = alloc(T * 2 * Fe), *gb = alloc(T * Fe), *ub = alloc(T * Fe),
        *y = alloc(T * E);
  auto* dids = reinterpret_cast<int32_t*>(alloc(T));
  float* dws = alloc(T);

  const ref::RopeTables rt = ref::rope_tables(T, c.n_rot, c.rope_base, nullptr);
  float* cos_t = alloc(int64_t(rt.cos.size()));
  float* sin_t = alloc(int64_t(rt.sin.size()));
  q.memcpy(cos_t, rt.cos.data(), rt.cos.size() * sizeof(float));
  q.memcpy(sin_t, rt.sin.data(), rt.sin.size() * sizeof(float)).wait();

  auto emit = [&](const std::string& name, const float* p, std::vector<int64_t> ne) {
    if (!dump || !dump->wants(name)) return;
    int64_t n = 1;
    for (auto e : ne) n *= e;
    std::vector<float> host(static_cast<size_t>(n));
    q.memcpy(host.data(), p, size_t(n) * sizeof(float)).wait();
    dump->put(name, host, ne);
  };
  auto emit_host = [&](const std::string& name, const std::vector<float>& host, std::vector<int64_t> ne) {
    if (dump && dump->wants(name)) dump->put(name, host, ne);
  };

  ref::embed(q, w.tok_embd, tok, T, 1.0f, x);
  emit("model.input_embed", x, {E, T});

  for (int l = 0; l < c.n_layer; ++l) {
    const Qwen35Layer L = w.load_layer(l);
    const std::string s = "-" + std::to_string(l);

    ref::rmsnorm(q, x, L.attn_norm, T, E, c.eps, h);
    emit("attn_norm" + s, h, {E, T});

    if (c.recurrent[size_t(l)]) {
      ref::matmul(q, L.wqkv, h, T, qkv);
      emit("linear_attn_qkv_mixed" + s, qkv, {CD, T});
      ref::matmul(q, L.wz, h, T, z);
      emit("z" + s, z, {VAL, T});
      ref::matmul(q, L.w_beta, h, T, beta);
      emit("beta" + s, beta, {1, HV, T});
      ref::sigmoid(q, beta, T * HV, beta);
      emit("beta_sigmoid" + s, beta, {1, HV, T});
      ref::matmul(q, L.w_alpha, h, T, alpha);
      emit("alpha" + s, alpha, {HV, T});
      ref::add_rows(q, alpha, L.dt_bias, T, HV);
      ref::softplus(q, alpha, T * HV, alpha);
      emit("a_softplus" + s, alpha, {HV, T});
      ref::mul_rows(q, alpha, L.ssm_a, T, HV);  // ssm_a = -exp(A_log)
      emit("gate" + s, alpha, {HV, T});

      ref::causal_conv(q, qkv, L.conv1d, T, CD, c.ssm_conv, conv);
      emit("conv_output_raw" + s, conv, {CD, T, 1});
      ref::silu(q, conv, T * CD, conv);
      emit("conv_output_silu" + s, conv, {CD, T, 1});
      // q | k | v along the channels; q and k per key head, l2-normalized.
      ref::l2_norm_heads(q, conv, CD, T, HK, S, c.eps, qn);
      ref::l2_norm_heads(q, conv + KEY, CD, T, HK, S, c.eps, kn);
      if (dump && (dump->wants("q_conv_predelta" + s) || dump->wants("k_conv_predelta" + s))) {
        // llama.cpp repeats the key heads to the value heads (tiled) before the recurrence.
        for (int which = 0; which < 2; ++which) {
          const std::string name = std::string(which ? "k" : "q") + "_conv_predelta" + s;
          if (!dump->wants(name)) continue;
          std::vector<float> src(size_t(T * KEY)), rep(size_t(T * HV * S));
          q.memcpy(src.data(), which ? kn : qn, src.size() * sizeof(float)).wait();
          for (int64_t t = 0; t < T; ++t)
            for (int64_t hv = 0; hv < HV; ++hv)
              std::copy_n(src.data() + (t * HK + hv % HK) * S, S, rep.data() + (t * HV + hv) * S);
          dump->put(name, rep, {S, HV, T, 1});
        }
      }
      ref::gated_delta_rule(q, qn, kn, conv + 2 * KEY, CD, alpha, beta, T, HK, HV, S, state, dout);
      emit("attn_output" + s, dout, {S, HV, T, 1});
      ref::gated_rmsnorm(q, dout, z, L.ssm_norm, T * HV, S, c.eps, dout);
      emit("final_output" + s, dout, {VAL, T, 1});
      ref::matmul(q, L.ssm_out, dout, T, o);
      emit("linear_attn_out" + s, o, {E, T, 1});
    } else {
      ref::matmul(q, L.wq, h, T, qfull);
      emit("Qcur_full" + s, qfull, {2 * H * D, T});
      ref::matmul(q, L.wk, h, T, kb);
      ref::matmul(q, L.wv, h, T, vb);
      emit("Vcur" + s, vb, {D, Hkv, T});
      ref::split_q_gate(q, qfull, T, H, D, qb, gate);
      emit("gate_reshaped" + s, gate, {H * D, T});
      ref::rmsnorm(q, qb, L.q_norm, T * H, D, c.eps, qb);
      emit("Qcur_normed" + s, qb, {D, H, T});
      ref::rmsnorm(q, kb, L.k_norm, T * Hkv, D, c.eps, kb);
      emit("Kcur_normed" + s, kb, {D, Hkv, T});
      ref::rope_neox(q, qb, T, H, D, c.n_rot, cos_t, sin_t);
      ref::rope_neox(q, kb, T, Hkv, D, c.n_rot, cos_t, sin_t);
      emit("Qcur" + s, qb, {D, H, T});
      emit("Kcur" + s, kb, {D, Hkv, T});
      ref::attention(q, qb, kb, vb, T, H, Hkv, D, 0, 1.0f / std::sqrt(float(D)), att);
      emit("attn_pregate" + s, att, {H * D, T});
      ref::sigmoid(q, gate, T * H * D, gate);
      emit("gate_sigmoid" + s, gate, {H * D, T});
      ref::mul(q, att, gate, T * H * D, att);
      emit("attn_gated" + s, att, {H * D, T});
      ref::matmul(q, L.wo, att, T, o);
      emit("attn_output" + s, o, {E, T});
    }
    ref::add(q, o, x, T * E, x2);
    emit("attn_residual" + s, x2, {E, T});
    ref::rmsnorm(q, x2, L.post_norm, T, E, c.eps, h);
    emit("attn_post_norm" + s, h, {E, T});

    // Routed experts: the routing is on the host; per expert the tokens are gathered, computed, and
    // scattered back with their weights.
    ref::matmul(q, L.router, h, T, rlog);
    emit("ffn_moe_logits" + s, rlog, {NE, T});
    std::vector<std::vector<int>> etok(static_cast<size_t>(NE));
    std::vector<std::vector<float>> ewt(static_cast<size_t>(NE));
    {
      std::vector<float> host(size_t(T * NE)), probs(size_t(T * NE)), wnorm(size_t(T * K));
      q.memcpy(host.data(), rlog, host.size() * sizeof(float)).wait();
      for (int64_t t = 0; t < T; ++t) {
        const float* lg = host.data() + t * NE;
        float* p = probs.data() + t * NE;
        const float mx = *std::max_element(lg, lg + NE);
        float sum = 0.0f;
        for (int64_t e = 0; e < NE; ++e) {
          p[e] = std::exp(lg[e] - mx);
          sum += p[e];
        }
        for (int64_t e = 0; e < NE; ++e) p[e] /= sum;
        std::vector<int> idx(static_cast<size_t>(NE));
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b2) { return p[a] > p[b2]; });
        float wsum = 0.0f;
        for (int64_t i = 0; i < K; ++i) wsum += p[idx[size_t(i)]];
        const float denom = std::max(wsum, 6.103515625e-5f);
        for (int64_t i = 0; i < K; ++i) {
          const int e = idx[size_t(i)];
          if (routing) routing->push_back(int32_t(e));
          float wt = p[e] / denom;
          wnorm[size_t(t * K + i)] = wt;
          if (c.expert_weights_scale != 0.0f && c.expert_weights_scale != 1.0f) wt *= c.expert_weights_scale;
          etok[size_t(e)].push_back(int(t));
          ewt[size_t(e)].push_back(wt);
        }
      }
      emit_host("ffn_moe_probs" + s, probs, {NE, T});
      emit_host("ffn_moe_weights_norm" + s, wnorm, {1, K, T});
    }
    q.fill(moe, 0.0f, size_t(T * E)).wait();
    for (int64_t e = 0; e < NE; ++e) {
      const int64_t n_e = int64_t(etok[size_t(e)].size());
      if (!n_e) continue;
      q.memcpy(dids, etok[size_t(e)].data(), size_t(n_e) * sizeof(int32_t));
      q.memcpy(dws, ewt[size_t(e)].data(), size_t(n_e) * sizeof(float)).wait();
      ref::gather_rows(q, h, dids, n_e, E, gx);
      auto expert = [&](const ref::Weight& m) {
        ref::Weight r = m;
        r.data += uint64_t(e) * uint64_t(m.n) * m.row_bytes;
        return r;
      };
      if (L.exp_gate_up.data) {
        ref::matmul(q, expert(L.exp_gate_up), gx, n_e, gu);
        ref::split_gate_up(q, gu, n_e, Fe, gb, ub);
      } else {
        ref::matmul(q, expert(L.exp_gate), gx, n_e, gb);
        ref::matmul(q, expert(L.exp_up), gx, n_e, ub);
      }
      ref::swiglu(q, gb, ub, n_e * Fe, gb);
      ref::matmul(q, expert(L.exp_down), gb, n_e, y);
      ref::scatter_add_rows(q, y, dids, dws, n_e, E, moe);
    }
    emit("ffn_moe_out" + s, moe, {E, T});

    // Shared expert, scaled by sigmoid of its one-row gate.
    ref::matmul(q, L.shexp_gate, h, T, sg);
    ref::matmul(q, L.shexp_up, h, T, su);
    ref::swiglu(q, sg, su, T * Fs, sg);
    ref::matmul(q, L.shexp_down, sg, T, o2);
    emit("ffn_shexp" + s, o2, {E, T});
    ref::matmul(q, L.shexp_gate_inp, h, T, sgate);
    emit("shared_expert_gate" + s, sgate, {1, T});
    ref::sigmoid(q, sgate, T, sgate);
    emit("shared_expert_gate_sigmoid" + s, sgate, {1, T});
    ref::mul_by_row_scalar(q, o2, sgate, T, E, o2);
    emit("ffn_shexp_gated" + s, o2, {E, T});
    ref::add(q, moe, o2, T * E, o);
    emit("ffn_out" + s, o, {E, T});
    ref::add(q, o, x2, T * E, x);
    emit("l_out" + s, x, {E, T});
    q.wait();  // the reference kernels do not wait, and this layer's weights are freed at the end of the scope
  }

  ref::rmsnorm(q, x, w.output_norm, T, E, c.eps, h);
  emit("result_norm", h, {E, T});
  float* logits = alloc(logits_rows * c.n_vocab);
  ref::matmul(q, w.output, h + (T - logits_rows) * E, logits_rows, logits);
  std::vector<float> out(size_t(logits_rows * c.n_vocab));
  q.memcpy(out.data(), logits, out.size() * sizeof(float)).wait();
  return out;
}

}  // namespace eklow::model
