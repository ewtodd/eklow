#include "model/inkling2.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "kernels/ref_inkling.hpp"

namespace eklow::model {

namespace {

[[noreturn]] void unsupported(const gguf::File& f, const std::string& what) {
  throw std::runtime_error(f.path() + ": not an Inkling 2 this engine runs: " + what);
}

}  // namespace

Inkling2Config Inkling2Config::from_gguf(const gguf::File& f) {
  if (f.at("general.architecture").as_string() != "inkling2_loop") unsupported(f, "architecture is not inkling2_loop");
  const std::string p = "inkling2_loop.";
  auto i = [&](const std::string& k) { return int(f.at(p + k).as_int()); };
  auto fl = [&](const std::string& k) { return float(f.at(p + k).as_float()); };
  // head counts are per-layer arrays in the file; this code needs them uniform
  auto uniform = [&](const std::string& k) {
    const gguf::Value& v = f.at(p + k);
    if (v.kind != gguf::Value::Kind::Array) return int(v.as_int());
    const auto& a = v.as_array();
    for (const auto& e : a)
      if (e.as_int() != a.at(0).as_int()) unsupported(f, k + " differs between layers");
    return int(a.at(0).as_int());
  };
  Inkling2Config c;
  c.n_layer = i("block_count");
  c.n_embd = i("embedding_length");
  c.n_head = uniform("attention.head_count");
  c.n_head_kv = uniform("attention.head_count_kv");
  c.head_dim = i("attention.key_length");
  c.n_vocab = i("vocab_size");
  c.n_vocab_out = i("vocab_size_unpadded");
  c.n_expert = i("expert_count");
  c.n_expert_used = i("expert_used_count");
  c.n_ff_exp = uniform("expert_feed_forward_length");
  c.n_rel = i("attention.relative_length");
  c.rel_extent = i("attention.relative_extent");
  c.window = i("attention.sliding_window");
  c.n_conv = i("shortconv.l_cache");
  c.block = i("attn_res.block_size");
  c.log_floor = i("attention.log_scale_floor");
  c.log_alpha = fl("attention.log_scale_alpha");
  c.qk_clip = fl("attention.qk_clip");
  c.rel_clip = fl("attention.relative_clip");
  c.eps = fl("attention.layer_norm_rms_epsilon");
  c.situ_beta = fl("activation.situ_beta");
  c.situ_linear_beta = fl("activation.situ_linear_beta");
  c.w_scale = f.find(p + "expert_weights_scale") ? fl("expert_weights_scale") : 1.0f;
  c.softcap = f.find(p + "final_logit_softcapping") ? fl("final_logit_softcapping") : 0.0f;
  c.core_start = i("loop.start");
  c.core_end = i("loop.end");
  c.n_passes = i("loop.count");
  c.gate_rank = i("loop.gate_rank");
  if (const gguf::Value* v = f.find(p + "loop.varying"); v && std::get<bool>(v->v)) unsupported(f, "a variable loop");
  for (const auto& v : f.at(p + "attention.sliding_window_pattern").as_array()) c.sliding.push_back(std::get<bool>(v.v));
  if (int(c.sliding.size()) != c.n_layer) unsupported(f, "a sliding-window pattern of the wrong length");
  if (i("attention.value_length") != c.head_dim) unsupported(f, "value heads differ from key heads");
  if (c.n_conv != 4) unsupported(f, "a short convolution kernel other than 4");
  if (c.n_expert_used > 16 || c.n_rel > 64) unsupported(f, "more than 16 experts a token or 64 relative channels");
  if (c.core_start <= 0 || c.core_start >= c.core_end || c.core_end > c.n_layer || c.core_start % c.block ||
      c.core_end % c.block || c.n_passes < 2)
    unsupported(f, "a loop core this code does not run");
  // the depth mixes hold at most 8 states: the embeddings, one a block, and the stream
  if (c.n_layer / c.block + 2 > 8) unsupported(f, "more than 6 blocks");
  if (c.n_vocab_out > c.n_vocab) unsupported(f, "an unpadded vocabulary larger than the table");
  return c;
}

Inkling2Weights::Inkling2Weights(Inkling2Weights&& o) noexcept
    : tok_embd(o.tok_embd), output(o.output), tok_norm(o.tok_norm), output_norm(o.output_norm),
      output_res_query(o.output_res_query), layers(std::move(o.layers)), loops(std::move(o.loops)), bytes(o.bytes),
      q_(o.q_), allocations_(std::move(o.allocations_)) {
  o.allocations_.clear();
}

Inkling2Weights::~Inkling2Weights() {
  for (void* p : allocations_) sycl::free(p, *q_);
}

Inkling2Weights Inkling2Weights::upload(const gguf::File& f, const Inkling2Config& c, sycl::queue& q) {
  Inkling2Weights w(q);
  const bool in_place = q.get_device().is_cpu();
  auto put = [&](const gguf::Tensor& t) -> const uint8_t* {
    w.bytes += t.nbytes;
    if (in_place) return t.data;
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q);
    if (!d) throw std::runtime_error("out of device memory uploading " + t.name);
    w.allocations_.push_back(d);
    q.memcpy(d, t.data, t.nbytes).wait();
    return d;
  };
  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto shape = [&](const gguf::Tensor& t, std::vector<uint64_t> ne) {
    if (t.ne != ne) throw std::runtime_error(f.path() + ": " + t.name + " has an unexpected shape");
  };
  auto matrix = [&](const std::string& name, int64_t k, int64_t n) {
    const gguf::Tensor& t = need(name);
    shape(t, {uint64_t(k), uint64_t(n)});
    return ref::Weight{t.type, k, n, put(t), t.nbytes / uint64_t(n)};
  };
  auto vec = [&](const std::string& name, std::vector<uint64_t> ne) {
    const gguf::Tensor& t = need(name);
    if (t.type != gguf::Type::F32) throw std::runtime_error(f.path() + ": " + name + " is not F32");
    shape(t, ne);
    return reinterpret_cast<const float*>(put(t));
  };
  auto experts = [&](const std::string& name, int64_t k, int64_t n) {
    const gguf::Tensor& t = need(name);
    shape(t, {uint64_t(k), uint64_t(n), uint64_t(c.n_expert)});
    Experts e;
    e.stride = t.nbytes / uint64_t(c.n_expert);
    e.w = ref::Weight{t.type, k, n, put(t), e.stride / uint64_t(n)};
    const std::string sname = name.substr(0, name.size() - std::string("weight").size()) + "scale";
    if (f.tensor(sname)) e.scale = vec(sname, {uint64_t(c.n_expert)});
    return e;
  };
  const int64_t E = c.n_embd, HD = int64_t(c.n_head) * c.head_dim, KD = int64_t(c.n_head_kv) * c.head_dim;
  const uint64_t uE = uint64_t(E);
  w.tok_embd = matrix("token_embd.weight", E, c.n_vocab);
  w.output = matrix("output.weight", E, c.n_vocab);
  w.tok_norm = vec("token_embd_norm.weight", {uE});
  w.output_norm = vec("output_norm.weight", {uE});
  w.output_res_query = vec("output_res_query.weight", {uE});
  for (int l = 0; l < c.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L;
    const bool swa = c.sliding[size_t(l)];
    L.wq = matrix(b + "attn_q.weight", E, HD);
    L.wk = matrix(b + "attn_k.weight", E, KD);
    L.wv = matrix(b + "attn_v.weight", E, KD);
    L.wo = matrix(b + "attn_output.weight", HD, E);
    L.gate = matrix(b + "attn_gate.weight", E, HD);
    L.rel = matrix(b + "attn_rel.weight", E, int64_t(c.n_rel) * c.n_head);
    L.sink = matrix(b + "attn_sink_proj.weight", E, c.n_head);
    L.q_clip = vec(b + "attn_q_clip.weight", {uint64_t(c.head_dim), uint64_t(c.n_head)});
    L.rel_bias = vec(b + "attn_rel_bias.weight", {uint64_t(c.n_rel), uint64_t(swa ? c.window : c.rel_extent)});
    L.rel_clip = vec(b + "attn_rel_clip.weight", {uint64_t(c.n_rel), uint64_t(c.n_head)});
    L.k_conv = vec(b + "attn_k_conv.weight", {4, uint64_t(KD)});
    L.v_conv = vec(b + "attn_v_conv.weight", {4, uint64_t(KD)});
    L.attn_post_conv = vec(b + "attn_post_conv.weight", {4, uE});
    L.ffn_post_conv = vec(b + "ffn_post_conv.weight", {4, uE});
    if (swa) {
      L.attn_norm = vec(b + "attn_norm.weight", {uE});
    } else {
      L.q_inp_norm = vec(b + "attn_q_inp_norm.weight", {uE});
      L.k_inp_norm = vec(b + "attn_k_inp_norm.weight", {uE});
      L.v_inp_norm = vec(b + "attn_v_inp_norm.weight", {uE});
      L.k_conv_proj = matrix(b + "attn_k_conv_proj.weight", E, 4 * c.n_head_kv);
      L.v_conv_proj = matrix(b + "attn_v_conv_proj.weight", E, 4 * c.n_head_kv);
    }
    if (l > 0) {
      if (swa) L.attn_res_query = vec(b + "attn_res_query.weight", {uE});
      else L.attn_res_query = vec(b + "attn_res_query.weight", {uE, 3});
    }
    L.ffn_res_query = vec(b + "ffn_res_query.weight", {uE});
    L.ffn_norm = vec(b + "ffn_norm.weight", {uE});
    L.router = matrix(b + "ffn_gate_inp.weight", E, c.n_expert);
    if (f.tensor(b + "exp_probs_b.bias")) L.probs_bias = vec(b + "exp_probs_b.bias", {uint64_t(c.n_expert)});
    L.gate_up = experts(b + "ffn_gate_up_exps.weight", E, 2 * int64_t(c.n_ff_exp));
    L.down = experts(b + "ffn_down_exps.weight", c.n_ff_exp, E);
    w.layers.push_back(L);
  }
  static const char* kCtrl[4] = {"q", "k", "v", "ffn"};
  for (int l = c.core_start; l < c.core_end; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Loop lp;
    lp.msg_query = vec(b + "loop_msg_query.weight", {uE, 4});
    for (int k = 0; k < 4; ++k) {
      const std::string n = b + "loop_" + kCtrl[k] + "_ctrl_";
      lp.down[k] = matrix(n + "down.weight", 2 * E, c.gate_rank);
      lp.gain[k] = matrix(n + "gain.weight", c.gate_rank, E);
      lp.forget[k] = matrix(n + "forget.weight", c.gate_rank, E);
    }
    w.loops.push_back(lp);
  }
  return w;
}

namespace {

// Device buffers of floats that wait for the queue before they are freed (kernels on an in-order queue may still read
// them).
struct Buf {
  std::shared_ptr<float> p;
  float* get() const { return p.get(); }
};

Buf alloc(sycl::queue& q, int64_t n) {
  float* d = sycl::malloc_device<float>(size_t(std::max<int64_t>(n, 1)), q);
  if (!d) throw std::runtime_error("inkling2 reference: out of device memory");
  sycl::queue* qq = &q;
  return Buf{std::shared_ptr<float>(d, [qq](float* x) {
    qq->wait();
    sycl::free(x, *qq);
  })};
}

struct Forward {
  sycl::queue& q;
  const Inkling2Config& c;
  const Inkling2Weights& w;
  const DumpSink* dump;
  int64_t T, E, dump_from;
  float* hist;       // [slot][3][conv_channels]
  float* tau;        // [T], the full layers' log-position scale
  const float** ptrs;  // device array of state pointers for depth_mix
  const Teacher* teacher;
  bool emulate;  // round each matmul's activations as ggml's CPU backend does (ref::inkling::ggml_cpu_acts)

  // out = x W^T, with x first rounded as ggml's CPU backend rounds it when emulating
  void mm(const ref::Weight& W, const float* x, int64_t rows, float* out) {
    if (!emulate || W.type == gguf::Type::F32) return ref::matmul(q, W, x, rows, out);
    Buf xq = alloc(q, rows * W.k);
    ref::inkling::ggml_cpu_acts(q, W.type, x, rows, W.k, xq.get());
    ref::matmul(q, W, xq.get(), rows, out);
  }
  void mm_experts(const Inkling2Weights::Experts& X, const float* x, bool per_slot, const int32_t* ids, int64_t tc,
                  int64_t nu, float* out) {
    Buf xq;
    if (emulate) {
      const int64_t rows = per_slot ? tc * nu : tc;
      xq = alloc(q, rows * X.w.k);
      ref::inkling::ggml_cpu_acts(q, X.w.type, x, rows, X.w.k, xq.get());
      x = xq.get();
    }
    ref::matmul_experts(q, X.w, X.stride, x, per_slot, ids, tc, nu, X.scale, out);
  }

  bool wants(const std::string& name) const { return dump && dump_from < T && dump->wants(name); }

  // rows dump_from.. of a [T][...] tensor, with ggml's shape (inner dims, then the positions)
  void put(const std::string& name, const float* d, std::vector<int64_t> inner) {
    if (!wants(name)) return;
    int64_t per = 1;
    for (int64_t v : inner) per *= v;
    std::vector<float> h(size_t((T - dump_from) * per));
    q.memcpy(h.data(), d + dump_from * per, h.size() * sizeof(float)).wait();
    inner.push_back(T - dump_from);
    dump->put(name, h, inner);
  }
  void put_ids(const std::string& name, const int32_t* d, int64_t per) {
    if (!wants(name)) return;
    std::vector<int32_t> h(size_t((T - dump_from) * per));
    q.memcpy(h.data(), d + dump_from * per, h.size() * sizeof(int32_t)).wait();
    dump->put(name, std::vector<float>(h.begin(), h.end()), {per, T - dump_from});
  }

  // depth_mix over states (the blocks, then cur) with n_query queries: [n_query][T][E]
  Buf mix(const std::vector<Buf>& blocks, const Buf& cur, const float* query, int n_query) {
    std::vector<const float*> s;
    for (const auto& b : blocks) s.push_back(b.get());
    s.push_back(cur.get());
    q.memcpy(ptrs, s.data(), s.size() * sizeof(float*)).wait();
    Buf out = alloc(q, int64_t(n_query) * T * E);
    ref::inkling::depth_mix(q, ptrs, int(s.size()), query, n_query, T, E, 1e-6f, out.get());
    q.wait();  // ptrs is reused by the next mix
    return out;
  }

  // graph::control: (1 - tanh(forget)) cur + tanh(gain) msg, gains of rank gate_rank from the normed pair
  Buf control(const float* cur, const float* msg, const Inkling2Weights::Loop& lp, int k) {
    Buf in = alloc(q, T * 2 * E), d = alloc(q, T * c.gate_rank), g = alloc(q, T * E), f = alloc(q, T * E), out = alloc(q, T * E);
    ref::inkling::control_input(q, cur, msg, T, E, 1e-6f, in.get());
    mm(lp.down[k], in.get(), T, d.get());
    mm(lp.gain[k], d.get(), T, g.get());
    mm(lp.forget[k], d.get(), T, f.get());
    ref::inkling::control_mix(q, cur, msg, g.get(), f.get(), T * E, out.get());
    return out;
  }

  Buf norm(const float* x, const float* weight) {
    Buf o = alloc(q, T * E);
    ref::rmsnorm(q, x, weight, T, E, c.eps, o.get());
    return o;
  }

  // One layer application (slot), physical layer ip, as llama_model_inkling2::graph::build_layer.
  Buf layer(int slot, int ip, const Buf& x, std::vector<Buf>& blocks, const Buf* msgs, const Inkling2Weights::Loop* lp) {
    const auto& L = w.layers[size_t(ip)];
    const bool swa = c.sliding[size_t(ip)];
    const int64_t H = c.n_head, HK = c.n_head_kv, D = c.head_dim, HD = H * D, KD = HK * D, C = c.conv_channels();
    const std::string sl = "-" + std::to_string(slot);
    float* h = hist + int64_t(slot) * 3 * C;

    // the attention's inputs
    Buf in[3] = {x, x, x};
    if (L.attn_res_query) {
      Buf m = mix(blocks, x, L.attn_res_query, swa ? 1 : 3);
      for (int j = 0; j < 3; ++j) {
        const int64_t off = (swa ? 0 : j) * T * E;
        in[j] = Buf{std::shared_ptr<float>(m.p, m.get() + off)};
      }
    }
    if (msgs)
      for (int j = 0; j < 3; ++j) in[j] = control(in[j].get(), msgs->get() + int64_t(j) * T * E, *lp, j);
    if (swa && msgs) {
      for (int j = 0; j < 3; ++j) in[j] = norm(in[j].get(), L.attn_norm);
    } else if (swa) {
      in[0] = norm(in[0].get(), L.attn_norm);
      in[1] = in[2] = in[0];
    } else {
      in[0] = norm(in[0].get(), L.q_inp_norm);
      in[1] = norm(in[1].get(), L.k_inp_norm);
      in[2] = norm(in[2].get(), L.v_inp_norm);
    }

    put("attn_norm_out" + sl, in[0].get(), {E});  // eklow's names: llama.cpp calls every norm "norm"

    // projections, convolutions, clips
    Buf Q = alloc(q, T * HD), K = alloc(q, T * KD), V = alloc(q, T * KD), R = alloc(q, T * H * c.n_rel), S = alloc(q, T * H);
    {
      Buf kr = alloc(q, T * KD), vr = alloc(q, T * KD);
      mm(L.wq, in[0].get(), T, Q.get());
      mm(L.wk, in[1].get(), T, kr.get());
      mm(L.wv, in[2].get(), T, vr.get());
      Buf kd, vd;
      if (!swa) {
        kd = alloc(q, T * 4 * HK);
        vd = alloc(q, T * 4 * HK);
        mm(L.k_conv_proj, in[1].get(), T, kd.get());
        mm(L.v_conv_proj, in[2].get(), T, vd.get());
      }
      ref::inkling::short_conv(q, kr.get(), T, KD, h, C, L.k_conv, kd.get(), HK, K.get());
      ref::inkling::short_conv(q, vr.get(), T, KD, h + KD, C, L.v_conv, vd.get(), HK, V.get());
    }
    mm(L.rel, in[0].get(), T, R.get());
    ref::inkling::l2_clip(q, Q.get(), T * H, H, D, L.q_clip, c.qk_clip);
    ref::inkling::l2_clip(q, K.get(), T * HK, HK, D, nullptr, c.qk_clip);
    ref::rmsnorm(q, V.get(), nullptr, T * HK, D, c.eps, V.get());
    ref::inkling::l2_clip(q, R.get(), T * H, H, c.n_rel, L.rel_clip, c.rel_clip);
    if (!swa) {
      ref::inkling::scale_tokens(q, Q.get(), tau, T, HD);
      ref::inkling::scale_tokens(q, R.get(), tau, T, H * c.n_rel);
    }
    put("Qcur" + sl, Q.get(), {D, H});
    put("Kcur" + sl, K.get(), {D, HK});
    put("Vcur" + sl, V.get(), {D, HK});
    put("Rcur" + sl, R.get(), {c.n_rel, H});
    mm(L.sink, in[0].get(), T, S.get());

    // attention, gate, output projection and its convolution
    Buf A = alloc(q, T * HD), G = alloc(q, T * HD), O = alloc(q, T * E), Oc = alloc(q, T * E);
    ref::inkling::attention(q, Q.get(), K.get(), V.get(), R.get(), L.rel_bias, S.get(), T, H, HK, D, c.n_rel,
                            swa ? c.window : c.rel_extent, swa ? c.window : 0, 1.0f / std::sqrt(float(D)), A.get());
    put("kqv_out" + sl, A.get(), {HD});
    mm(L.gate, in[0].get(), T, G.get());
    ref::inkling::sigmoid_mul(q, A.get(), G.get(), T * HD);
    mm(L.wo, A.get(), T, O.get());
    ref::inkling::short_conv(q, O.get(), T, E, h + 2 * KD, C, L.attn_post_conv, nullptr, 0, Oc.get());

    // a block's first layer starts a new stream; the others add to theirs
    Buf cur;
    if (ip % c.block == 0) {
      blocks.push_back(x);
      cur = Oc;
    } else {
      put("attn_out" + sl, Oc.get(), {E});
      cur = alloc(q, T * E);
      ref::add(q, x.get(), Oc.get(), T * E, cur.get());
    }
    put("ffn_inp" + sl, cur.get(), {E});

    // the MoE
    Buf m = mix(blocks, cur, L.ffn_res_query, 1);
    if (msgs) m = control(m.get(), msgs->get() + 3 * T * E, *lp, 3);
    Buf hn = norm(m.get(), L.ffn_norm);
    put("ffn_norm_out" + sl, hn.get(), {E});
    const int64_t NU = c.n_expert_used, F = c.n_ff_exp;
    Buf lg = alloc(q, T * c.n_expert), wts = alloc(q, T * NU);
    std::shared_ptr<int32_t> ids(sycl::malloc_device<int32_t>(size_t(T * NU), q), [this](int32_t* p) {
      q.wait();
      sycl::free(p, q);
    });
    mm(L.router, hn.get(), T, lg.get());
    ref::inkling::route(q, lg.get(), L.probs_bias, T, c.n_expert, NU, c.w_scale, ids.get(), wts.get());
    put("ffn_moe_logits" + sl, lg.get(), {c.n_expert});
    put_ids("ffn_moe_topk" + sl, ids.get(), NU);
    put("ffn_moe_weights_scaled" + sl, wts.get(), {1, NU});
    Buf moe = alloc(q, T * E);
    const bool keep_situ = wants("ffn_moe_situ" + sl), keep_gu = wants("ffn_moe_gate_up" + sl);
    Buf situ_all = keep_situ ? alloc(q, T * NU * F) : Buf{};
    Buf gu_all = keep_gu ? alloc(q, T * NU * 2 * F) : Buf{};
    const int64_t chunk = 512;
    Buf gu = alloc(q, std::min(chunk, T) * NU * 2 * F), act = alloc(q, std::min(chunk, T) * NU * F),
        dn = alloc(q, std::min(chunk, T) * NU * E);
    for (int64_t t0 = 0; t0 < T; t0 += chunk) {
      const int64_t tc = std::min(chunk, T - t0);
      mm_experts(L.gate_up, hn.get() + t0 * E, false, ids.get() + t0 * NU, tc, NU, gu.get());
      if (keep_gu) q.memcpy(gu_all.get() + t0 * NU * 2 * F, gu.get(), size_t(tc * NU * 2 * F) * sizeof(float));
      float* a = keep_situ ? situ_all.get() + t0 * NU * F : act.get();
      ref::inkling::situ(q, gu.get(), tc * NU, F, c.situ_beta, c.situ_linear_beta, a);
      mm_experts(L.down, a, true, ids.get() + t0 * NU, tc, NU, dn.get());
      ref::inkling::moe_combine(q, dn.get(), wts.get() + t0 * NU, tc, NU, E, moe.get() + t0 * E);
    }
    if (keep_situ) put("ffn_moe_situ" + sl, situ_all.get(), {F, NU});
    if (keep_gu) put("ffn_moe_gate_up" + sl, gu_all.get(), {2 * F, NU});
    put("ffn_moe_out" + sl, moe.get(), {E});
    Buf fo = alloc(q, T * E);
    ref::inkling::short_conv(q, moe.get(), T, E, h + 2 * KD + E, C, L.ffn_post_conv, nullptr, 0, fo.get());
    put("ffn_out" + sl, fo.get(), {E});
    Buf out = alloc(q, T * E);
    ref::add(q, cur.get(), fo.get(), T * E, out.get());
    put("l_out" + sl, out.get(), {E});
    if (teacher) {
      std::vector<float> forced;
      if ((*teacher)("l_out" + sl, forced) && int64_t(forced.size()) == T * E)
        q.memcpy(out.get(), forced.data(), forced.size() * sizeof(float)).wait();
    }
    return out;
  }
};

}  // namespace

std::vector<float> inkling2_reference_forward(sycl::queue& q, const Inkling2Config& c, const Inkling2Weights& w,
                                              const std::vector<int32_t>& tokens, int64_t logits_rows,
                                              const DumpSink* dump, int64_t dump_from, const Teacher* teacher,
                                              bool emulate_ggml_cpu) {
  const int64_t T = int64_t(tokens.size()), E = c.n_embd, C = c.conv_channels();
  if (T == 0) return {};
  logits_rows = logits_rows <= 0 || logits_rows > T ? T : logits_rows;

  std::shared_ptr<int32_t> tok(sycl::malloc_device<int32_t>(size_t(T), q), [&q](int32_t* p) {
    q.wait();
    sycl::free(p, q);
  });
  q.memcpy(tok.get(), tokens.data(), size_t(T) * sizeof(int32_t)).wait();
  Buf hist = alloc(q, int64_t(c.n_slot()) * 3 * C);
  q.memset(hist.get(), 0, size_t(c.n_slot()) * 3 * C * sizeof(float));
  // the full layers' scale, as ggml computes it: 1 + alpha * log(clamp((pos + 1) * (1 / floor), 1, inf))
  std::vector<float> th(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    th[size_t(t)] = c.log_alpha * std::log(std::max(1.0f, (float(t) + 1.0f) * (1.0f / float(c.log_floor)))) + 1.0f;
  Buf tau = alloc(q, T);
  q.memcpy(tau.get(), th.data(), size_t(T) * sizeof(float)).wait();
  std::shared_ptr<const float*> ptrs(sycl::malloc_device<const float*>(16, q), [&q](const float** p) {
    q.wait();
    sycl::free(p, q);
  });

  Forward fw{q, c, w, dump, T, E, std::min(dump_from, T), hist.get(), tau.get(), ptrs.get(), teacher, emulate_ggml_cpu};

  Buf x = alloc(q, T * E);
  ref::embed(q, w.tok_embd, tok.get(), T, 1.0f, x.get());
  ref::rmsnorm(q, x.get(), w.tok_norm, T, E, c.eps, x.get());

  std::vector<Buf> blocks;
  for (int l = 0; l < c.core_start; ++l) x = fw.layer(l, l, x, blocks, nullptr, nullptr);
  const Buf prelude = x;
  const std::vector<Buf> prelude_blocks = blocks;
  for (int pass = 0; pass < c.n_passes; ++pass) {
    std::vector<Buf> bank;
    Buf bank_out;
    if (pass > 0) {
      // the previous pass's blocks after its first core block, and its output; every pass starts from the prelude
      bank.assign(blocks.begin() + std::ptrdiff_t(prelude_blocks.size()) + 1, blocks.end());
      bank_out = x;
      x = prelude;
      blocks = prelude_blocks;
    }
    for (int l = c.core_start; l < c.core_end; ++l) {
      const int s = c.slot(l, pass);
      if (pass == 0) {
        x = fw.layer(s, l, x, blocks, nullptr, nullptr);
        continue;
      }
      const auto& lp = w.loops[size_t(l - c.core_start)];
      Buf msgs = fw.mix(bank, bank_out, lp.msg_query, 4);
      x = fw.layer(s, l, x, blocks, &msgs, &lp);
    }
  }
  for (int l = c.core_end; l < c.n_layer; ++l) x = fw.layer(l, l, x, blocks, nullptr, nullptr);

  Buf m = fw.mix(blocks, x, w.output_res_query, 1);
  Buf hn = fw.norm(m.get(), w.output_norm);
  fw.put("result_norm", hn.get(), {E});
  const int64_t first = T - logits_rows, V = c.n_vocab;
  Buf lg = alloc(q, logits_rows * V);
  fw.mm(w.output, hn.get() + first * E, logits_rows, lg.get());
  if (c.softcap > 0) ref::softcap(q, lg.get(), c.softcap, logits_rows * V);
  std::vector<float> full(size_t(logits_rows * V));
  q.memcpy(full.data(), lg.get(), full.size() * sizeof(float)).wait();
  std::vector<float> out(size_t(logits_rows) * size_t(c.n_vocab_out));
  for (int64_t r = 0; r < logits_rows; ++r)
    std::memcpy(out.data() + r * c.n_vocab_out, full.data() + r * V, size_t(c.n_vocab_out) * sizeof(float));
  if (fw.wants("result_output")) {
    const int64_t from = std::max(fw.dump_from, first);
    std::vector<float> d(out.begin() + (from - first) * c.n_vocab_out, out.end());
    dump->put("result_output", d, {c.n_vocab_out, T - from});
  }
  return out;
}

}  // namespace eklow::model
