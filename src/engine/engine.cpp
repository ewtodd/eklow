#include "engine/engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <iterator>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/json_value.hpp"
#include "kernels/fast.hpp"
#include "rt/device.hpp"

namespace eklow::engine {

namespace {

uint64_t align256(uint64_t x) { return (x + 255) & ~uint64_t(255); }

bool qtype_of(gguf::Type t, cpu::QType& q) {
  switch (t) {
    case gguf::Type::Q4_0: q = cpu::QType::Q4_0; return true;
    case gguf::Type::Q8_0: q = cpu::QType::Q8_0; return true;
    case gguf::Type::Q4_K: q = cpu::QType::Q4_K; return true;
    case gguf::Type::Q5_K: q = cpu::QType::Q5_K; return true;
    case gguf::Type::Q6_K: q = cpu::QType::Q6_K; return true;
    case gguf::Type::IQ4_XS: q = cpu::QType::IQ4_XS; return true;
    default: return false;
  }
}

float half_to_float(uint16_t h) {
  const uint32_t s = uint32_t(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
  uint32_t u;
  if (e == 0) {
    if (m == 0) {
      u = s;
    } else {  // subnormal
      int sh = 0;
      uint32_t mm = m;
      while (!(mm & 1024)) mm <<= 1, ++sh;
      u = s | (uint32_t(127 - 15 - sh + 1) << 23) | ((mm & 1023) << 13);
    }
  } else if (e == 31) {
    u = s | 0x7f800000u | (m << 13);
  } else {
    u = s | ((e + 112) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// A small matrix (F32, F16, BF16 or Q8_0) as fp32 rows on the host.
std::vector<float> dequant_host(const gguf::Tensor& t) {
  const size_t n = size_t(t.elements());
  std::vector<float> out(n);
  switch (t.type) {
    case gguf::Type::F32: std::memcpy(out.data(), t.data, n * 4); break;
    case gguf::Type::F16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, t.data + 2 * i, 2);
        out[i] = half_to_float(h);
      }
      break;
    case gguf::Type::BF16:
      for (size_t i = 0; i < n; ++i) {
        uint16_t h;
        std::memcpy(&h, t.data + 2 * i, 2);
        const uint32_t u = uint32_t(h) << 16;
        std::memcpy(&out[i], &u, 4);
      }
      break;
    case gguf::Type::Q8_0:
      for (size_t b = 0; b < n / 32; ++b) {
        const uint8_t* blk = t.data + b * 34;
        uint16_t h;
        std::memcpy(&h, blk, 2);
        const float d = half_to_float(h);
        for (int i = 0; i < 32; ++i) out[b * 32 + size_t(i)] = d * float(int8_t(blk[2 + i]));
      }
      break;
    default: throw std::runtime_error(t.name + ": a small matrix in " + gguf::type_name(t.type) + " is not supported");
  }
  return out;
}

}  // namespace

float* Engine::dev(int64_t n) {
  auto* p = sycl::malloc_device<float>(size_t(n), q_);
  if (!p) throw std::runtime_error("engine: out of device memory");
  owned_.push_back(p);
  return p;
}

Engine::Engine(sycl::queue& q, const gguf::File& f, const Config& cfg)
    : q_(q), c_(cfg), cfg_(model::Qwen35Config::from_gguf(f)) {
  if (c_.max_ctx % attn::kPage) throw std::runtime_error("engine: max_ctx must be a multiple of the page size");
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, Hkv = cfg_.n_head_kv, D = cfg_.head_dim, NE = cfg_.n_expert, NU = cfg_.n_expert_used;
  const int64_t Fe = cfg_.n_ff_exp, Fs = cfg_.n_ff_shexp, B = c_.max_batch;
  if (cfg_.ssm_k_heads != gdn::kKeyHeads || cfg_.ssm_v_heads != gdn::kValueHeads || cfg_.ssm_head_dim != gdn::kHead ||
      cfg_.ssm_conv != gdn::kConvTaps)
    throw std::runtime_error("engine: the DeltaNet kernels take 16 key heads and 32 value heads of 128 with a 4-tap conv");
  if (NE != 256 || NU > 16 || Fe % 64 || Fs % 64) throw std::runtime_error("engine: unsupported expert shape");
  if (c_.max_pending < 0 || c_.max_pending > gdn::kMaxPendingLimit) throw std::runtime_error("engine: max_pending out of range");
  group_ = H / Hkv;
  if (c_.max_group <= 0) c_.max_group = c_.max_chunk;
  if (c_.max_group < c_.max_chunk || c_.max_group % c_.max_chunk) throw std::runtime_error("engine: max_group must be a multiple of max_chunk");
  gmax_ = c_.max_group / c_.max_chunk;
  cap_tokens_ = std::max(B, c_.max_group);
  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto upload = [&](const gguf::Tensor& t) {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q_);
    if (!d) throw std::runtime_error("engine: out of device memory loading " + t.name);
    owned_.push_back(d);
    q_.memcpy(d, t.data, t.nbytes).wait();
    weight_bytes_ += t.nbytes;
    return d;
  };
  auto vec = [&](const std::string& name) { return reinterpret_cast<const float*>(upload(need(name))); };
  auto vec_host = [&](const std::vector<float>& v) {
    float* d = dev(int64_t(v.size()));
    q_.memcpy(d, v.data(), v.size() * sizeof(float)).wait();
    return d;
  };

  // Matrices go through one staging buffer and are repacked for the matrix units. Matrices that read the same input are
  // stacked into one (their outputs side by side in each row), so a step launches one matmul for them.
  uint64_t staging_bytes = 0;
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    for (const char* group : {"attn_qkv.weight|attn_gate.weight", "attn_q.weight|attn_k.weight|attn_v.weight", "ssm_out.weight",
                              "attn_output.weight", "ffn_gate_shexp.weight|ffn_up_shexp.weight", "ffn_down_shexp.weight"}) {
      uint64_t sum = 0;
      std::stringstream names(group);
      for (std::string n; std::getline(names, n, '|');)
        if (const gguf::Tensor* t = f.tensor(b + n)) sum += t->nbytes;
      staging_bytes = std::max(staging_bytes, sum);
    }
  }
  if (const gguf::Tensor* t = f.tensor("output.weight")) staging_bytes = std::max(staging_bytes, t->nbytes);
  auto* staging = sycl::malloc_device<uint8_t>(staging_bytes, q_);
  if (!staging) throw std::runtime_error("engine: out of device memory for weight staging");
  // A matrix from tensors that stack (same type and width), on the matrix units in the file's format.
  auto make_w = [&](const std::vector<const gguf::Tensor*>& parts, bool interleave) {
    xmx::WeightFormat fmt;
    if (!xmx::format_of(parts[0]->type, fmt))
      throw std::runtime_error(f.path() + ": " + parts[0]->name + " is " + gguf::type_name(parts[0]->type) + ", which the matrix units do not run");
    uint64_t at = 0;
    int64_t n = 0;
    for (const gguf::Tensor* t : parts) {
      if (t->type != parts[0]->type || t->ne[0] != parts[0]->ne[0]) throw std::runtime_error(f.path() + ": stacked matrices of different types or widths");
      q_.memcpy(staging + at, t->data, t->nbytes);
      at += t->nbytes;
      n += int64_t(t->ne[1]);
      weight_bytes_ += t->nbytes;
    }
    q_.wait();
    const ref::Weight raw{parts[0]->type, int64_t(parts[0]->ne[0]), n, staging, parts[0]->nbytes / parts[0]->ne[1]};
    xmx::Weight w = fmt == xmx::WeightFormat::Q4_0 ? xmx::repack_q4_0(q_, raw, interleave) : xmx::repack_format(q_, raw, fmt, interleave);
    owned_.push_back(w.tiles);
    owned_.push_back(w.scales);
    return w;
  };
  auto dense_of = [&](const std::vector<const gguf::Tensor*>& parts) {
    Dense d;
    bool same = true;
    for (const gguf::Tensor* t : parts) same &= t->type == parts[0]->type && t->ne[0] == parts[0]->ne[0];
    if (same) {
      d.parts.push_back(make_w(parts, false));
    } else {
      for (const gguf::Tensor* t : parts) d.parts.push_back(make_w({t}, false));
    }
    for (const gguf::Tensor* t : parts) d.n += int64_t(t->ne[1]);
    return d;
  };

  // Head, embedding.
  const gguf::Tensor& out_t = need("output.weight");
  if (c_.head_dpas) {
    head_w_ = make_w({&out_t}, false);
    act_head_ = xmx::alloc_activations(q_, c_.max_logits, E);
  } else {
    if (out_t.type != gguf::Type::Q6_K) throw std::runtime_error(f.path() + ": the fp32-activation head is Q6_K only");
    auto* raw = sycl::malloc_device<uint8_t>(out_t.nbytes, q_);
    if (!raw) throw std::runtime_error("engine: out of device memory loading output.weight");
    q_.memcpy(raw, out_t.data, out_t.nbytes).wait();
    head_q6_ = fast::repack_q6k(q_, ref::Weight{out_t.type, int64_t(out_t.ne[0]), int64_t(out_t.ne[1]), raw, out_t.nbytes / out_t.ne[1]});
    sycl::free(raw, q_);
    owned_.push_back(head_q6_.lo);
    weight_bytes_ += out_t.nbytes;
  }
  output_norm_ = vec("output_norm.weight");
  embd_ = &need("token_embd.weight");
  embd_row_bytes_ = embd_->nbytes / embd_->ne[1];
  if (c_.embed_on_card) {
    embd_card_ = ref::Weight{embd_->type, int64_t(embd_->ne[0]), int64_t(embd_->ne[1]), upload(*embd_), embd_row_bytes_};
  } else {
    emb_host_ = sycl::malloc_host<uint8_t>(size_t(B) * embd_row_bytes_, q_);
    emb_dev_ = sycl::malloc_device<uint8_t>(size_t(B) * embd_row_bytes_, q_);
    if (!emb_host_ || !emb_dev_) throw std::runtime_error("engine: out of memory for the embedding rows");
    owned_.push_back(emb_host_);
    owned_.push_back(emb_dev_);
    std::vector<int32_t> iota(static_cast<size_t>(B));
    for (int64_t i = 0; i < B; ++i) iota[size_t(i)] = int32_t(i);
    iota_ = sycl::malloc_device<int32_t>(size_t(B), q_);
    owned_.push_back(iota_);
    q_.memcpy(iota_, iota.data(), iota.size() * 4).wait();
  }

  // RoPE inverse frequencies, in double, stored as fp32; the router's down scales (none: ones).
  {
    std::vector<float> inv(size_t(cfg_.n_rot / 2));
    for (size_t i = 0; i < inv.size(); ++i) inv[i] = float(std::pow(double(cfg_.rope_base), -2.0 * double(i) / cfg_.n_rot));
    inv_freq_ = vec_host(inv);
    ones_ = vec_host(std::vector<float>(size_t(NE), 1.0f));
  }

  int n_dn = 0, n_attn = 0;
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L;
    L.recurrent = cfg_.recurrent[size_t(l)];
    L.attn_norm = vec(b + "attn_norm.weight");
    L.post_norm = vec(b + "post_attention_norm.weight");
    if (L.recurrent) {
      L.gdn_idx = n_dn++;
      L.in = dense_of({&need(b + "attn_qkv.weight"), &need(b + "attn_gate.weight")});
      L.out = dense_of({&need(b + "ssm_out.weight")});
      if (L.in.n != gdn::kChannels + gdn::kInner) throw std::runtime_error("engine: DeltaNet input projections have an unexpected width");
      const gguf::Tensor& cv = need(b + "ssm_conv1d.weight");
      auto* raw = sycl::malloc_device<float>(size_t(cv.nbytes / 4), q_);
      q_.memcpy(raw, cv.data, cv.nbytes).wait();
      float* cw = gdn::repack_conv_w(q_, raw);
      sycl::free(raw, q_);
      owned_.push_back(cw);
      L.gw = gdn::Weights{cw, vec(b + "ssm_a"), vec(b + "ssm_dt.bias"), vec(b + "ssm_norm.weight"), cfg_.eps};
      std::vector<float> ab = dequant_host(need(b + "ssm_alpha.weight"));
      const std::vector<float> beta = dequant_host(need(b + "ssm_beta.weight"));
      ab.insert(ab.end(), beta.begin(), beta.end());
      if (int64_t(ab.size()) != 2 * gdn::kValueHeads * E) throw std::runtime_error("engine: ssm_alpha / ssm_beta have an unexpected shape");
      L.w_ab = vec_host(ab);
    } else {
      L.attn_idx = n_attn++;
      L.in = dense_of({&need(b + "attn_q.weight"), &need(b + "attn_k.weight"), &need(b + "attn_v.weight")});
      L.out = dense_of({&need(b + "attn_output.weight")});
      L.q_norm = vec(b + "attn_q_norm.weight");
      L.k_norm = vec(b + "attn_k_norm.weight");
    }
    L.shexp_gu = make_w({&need(b + "ffn_gate_shexp.weight"), &need(b + "ffn_up_shexp.weight")}, true);
    L.shexp_dn = make_w({&need(b + "ffn_down_shexp.weight")}, false);
    {
      // The router's rows, the shared expert's gate row and zero rows up to 32: one fp32 matrix.
      std::vector<float> rg = dequant_host(need(b + "ffn_gate_inp.weight"));
      if (int64_t(rg.size()) != NE * E) throw std::runtime_error("engine: router shape");
      const std::vector<float> sg = dequant_host(need(b + "ffn_gate_inp_shexp.weight"));
      if (int64_t(sg.size()) != E) throw std::runtime_error("engine: shared expert gate shape");
      rg.insert(rg.end(), sg.begin(), sg.end());
      rg.resize(size_t((NE + 32) * E), 0.0f);
      L.w_rg = vec_host(rg);
    }
    if (f.tensor(b + "ffn_gate_up_exps.weight")) throw std::runtime_error(f.path() + ": merged gate_up expert tensors are not supported");
    const gguf::Tensor &g = need(b + "ffn_gate_exps.weight"), &u = need(b + "ffn_up_exps.weight"), &d = need(b + "ffn_down_exps.weight");
    if (g.type != u.type || g.nbytes != u.nbytes) throw std::runtime_error(f.path() + ": gate and up experts differ in type");
    if (l == 0) {
      t_gu_ = g.type;
      t_dn_ = d.type;
    } else if (g.type != t_gu_ || d.type != t_dn_) {
      throw std::runtime_error(f.path() + ": experts of different types in different layers are not supported");
    }
    L.gate_host = g.data;
    L.up_host = u.data;
    L.down_host = d.data;
    layers_.push_back(std::move(L));
  }
  sycl::free(staging, q_);
  if (!xmx::format_of(t_gu_, f_gu_) || !xmx::format_of(t_dn_, f_dn_) || !qtype_of(t_gu_, q_gu_) || !qtype_of(t_dn_, q_dn_))
    throw std::runtime_error(f.path() + ": expert type " + gguf::type_name(t_gu_) + " / " + gguf::type_name(t_dn_) + " is not supported");
  {
    const gguf::Tensor& g = need("blk.0.ffn_gate_exps.weight");
    const gguf::Tensor& d = need("blk.0.ffn_down_exps.weight");
    gu_row_ = g.nbytes / (g.ne[1] * g.ne[2]);
    dn_row_ = d.nbytes / (d.ne[1] * d.ne[2]);
    raw_gu_ = 2 * uint64_t(Fe) * gu_row_;
    raw_dn_ = uint64_t(E) * dn_row_;
    raw_expert_ = raw_gu_ + raw_dn_;
    gu_tb_ = align256(xmx::weight_tile_bytes(f_gu_, E, 2 * Fe));
    gu_mb_ = align256(xmx::weight_meta_bytes(f_gu_, E, 2 * Fe));
    dn_tb_ = align256(xmx::weight_tile_bytes(f_dn_, Fe, E));
    dn_mb_ = align256(xmx::weight_meta_bytes(f_dn_, Fe, E));
    packed_expert_ = gu_tb_ + gu_mb_ + dn_tb_ + dn_mb_;
    expert_set_bytes_ = uint64_t(cfg_.n_layer) * packed_expert_;
  }

  // KV cache: a page pool for the attention layers.
  max_pages_ = c_.max_ctx / attn::kPage;
  n_pages_ = max_pages_ * c_.n_slots;
  const size_t page_elems = size_t(n_pages_ * Hkv * attn::kPage * D);
  const size_t elem = c_.int8_kv ? 1 : 2;
  page_table_ = sycl::malloc_device<int32_t>(size_t(c_.n_slots * max_pages_), q_);
  if (!page_table_) throw std::runtime_error("engine: out of device memory for the KV cache");
  owned_.push_back(page_table_);
  for (int l = 0; l < n_attn; ++l) {
    KvLayer g;
    g.k = sycl::malloc_device(page_elems * elem, q_);
    g.v = sycl::malloc_device(page_elems * elem, q_);
    if (c_.int8_kv) {
      g.ks = sycl::malloc_device<sycl::half>(page_elems / 16, q_);
      g.vs = sycl::malloc_device<sycl::half>(page_elems / 16, q_);
    }
    if (!g.k || !g.v || (c_.int8_kv && (!g.ks || !g.vs))) throw std::runtime_error("engine: out of device memory for the KV cache");
    for (void* p : {g.k, g.v, static_cast<void*>(g.ks), static_cast<void*>(g.vs)})
      if (p) owned_.push_back(p);
    cache_bytes_ += 2 * page_elems * elem + (c_.int8_kv ? 2 * (page_elems / 16) * sizeof(sycl::half) : 0);
    kv_.push_back(g);
  }
  page_table_host_.assign(size_t(c_.n_slots * max_pages_), 0);
  for (int64_t p = n_pages_ - 1; p >= 0; --p) free_pages_.push_back(int32_t(p));
  slots_.resize(size_t(c_.n_slots));

  // The DeltaNet's recurrent state, the pending buffers and the step's scratch.
  gcache_ = gdn::alloc_cache(q_, n_dn, c_.n_slots, c_.max_pending);
  gscratch_ = gdn::alloc_scratch(q_, B + int64_t(c_.n_slots) * c_.max_pending);
  cache_bytes_ += uint64_t(gcache_.snapshot_bytes()) * uint64_t(c_.n_slots) +
                  uint64_t(c_.n_slots) * uint64_t(n_dn) * uint64_t(c_.max_pending) * (2 * gdn::kValueHeads * gdn::kHeadChannels + 64) * 4;
  books_.assign(size_t(c_.n_slots), gdn::SlotBook{});
  plan_dev_ = sycl::malloc_device<gdn::SlotStep>(size_t(std::max<int64_t>(c_.n_slots, gmax_)), q_);
  plan_host_ = sycl::malloc_host<gdn::SlotStep>(size_t(std::max<int64_t>(c_.n_slots, gmax_)), q_);
  if (!plan_dev_ || !plan_host_) throw std::runtime_error("engine: out of memory for the step plan");
  owned_.push_back(plan_dev_);
  owned_.push_back(plan_host_);
  for (int s = 0; s < c_.n_slots; ++s) gdn::reset_slot(q_, gcache_, s);

  // Workspace.
  x_ = dev(cap_tokens_ * E);  // a group's residual stream is all its sub-chunks' rows
  x2_ = dev(B * E);
  h_ = dev(B * E);
  o_ = dev(B * E);
  qz_ = dev(B * std::max<int64_t>(gdn::kChannels + gdn::kInner, (2 * H + 2 * Hkv) * D));
  qb_ = dev(B * H * D);
  gt_ = dev(B * H * D);
  att_ = dev(B * std::max<int64_t>(H * D, 4 * E));  // also the streamed pairs' fp16 rows (up to B x n_used x E halves)
  kb_ = dev(B * Hkv * D);
  vb_ = dev(B * Hkv * D);
  ab_ = dev(B * 2 * gdn::kValueHeads);
  rlog_ = dev(B * NE);
  sg_ = dev(B * 32);
  rg_ = dev(std::min<int64_t>(B, 16) * (NE + 32));
  moe_ = dev(B * E);
  gx_ = dev(xmx::kFp16Rows * E);
  moe_gu_ = dev(xmx::kFp16Rows * 2 * Fe);
  sh_gu_ = dev(xmx::kFp16Rows * 2 * Fs);
  rows_ = dev(c_.max_logits * E);
  logits_ = dev(c_.max_logits * cfg_.n_vocab);
  tmp_ = dev(B * 8192);
  for (auto [p, n] : {std::pair{&tok_, cap_tokens_}, {&slot_, cap_tokens_}, {&pos_, cap_tokens_}, {&rpos_, 3 * cap_tokens_}}) {
    *p = sycl::malloc_device<int32_t>(size_t(n), q_);
    if (!*p) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(*p);
  }
  rows_idx_ = sycl::malloc_device<int32_t>(size_t(c_.max_logits), q_);
  owned_.push_back(rows_idx_);
  // At most one chunk per slot in a step, so at most n_slots partial tiles per K/V head.
  max_tiles_ = Hkv * ((B * group_ + 7) / 8 + c_.n_slots);
  tiles_ = sycl::malloc_device<int32_t>(size_t(6 * max_tiles_ * gmax_), q_);  // one list per sub-chunk of a group
  if (!tiles_) throw std::runtime_error("engine: out of device memory");
  owned_.push_back(tiles_);
  if (gmax_ > 1) {
    // What outlives a sub-chunk in a layer-major group (allocated before the experts are placed, so that the placement counts it):
    // the MoE inputs for the streamed waves, in fp16, and the routing. The residual stream is x_.
    hg_ = sycl::malloc_device<sycl::half>(size_t(cap_tokens_ * E), q_);
    gids_ = sycl::malloc_device<int32_t>(size_t(cap_tokens_ * NU), q_);
    if (!hg_ || !gids_) throw std::runtime_error("engine: out of device memory for layer-major groups");
    owned_.push_back(hg_);
    owned_.push_back(gids_);
    gw_ = dev(cap_tokens_ * NU);
  }
  attn::reserve(q_, attn_scratch_, max_tiles_, c_.max_chunk * group_, Hkv, D, group_);
  act_ = xmx::alloc_activations(q_, B, std::max<int64_t>(E, H * D));
  act_sh_ = xmx::alloc_activations(q_, B, Fs);
  act_exp_ffn_ = xmx::alloc_activations(q_, xmx::kFp16Rows, Fe);
  {
    const int64_t P = B * NU;
    moe_ids_ = sycl::malloc_device<int32_t>(size_t(P), q_);
    moe_row_ = sycl::malloc_device<int32_t>(size_t(P), q_);
    owned_.push_back(moe_ids_);
    owned_.push_back(moe_row_);
    moe_w_ = dev(P);
    // Prefill: pairs sorted by expert in row tiles of moe_tile_ rows: at most P / tile + n_expert tiles. The fp16 expert inputs
    // share the fp32 down outputs' memory (dead once the gate_up is done).
    moe_max_tiles_ = (P + moe_tile_ - 1) / moe_tile_ + NE;
    const int64_t rows = moe_max_tiles_ * moe_tile_;
    moe_y_ = dev(rows * E);
    moe_a_ = reinterpret_cast<sycl::half*>(moe_y_);
    moe_h_ = sycl::malloc_device<sycl::half>(size_t(rows * Fe), q_);
    moe_tiles_ = sycl::malloc_device<int32_t>(size_t(4 + 4 * moe_max_tiles_), q_);
    moe_scratch_ = sycl::malloc_device<int32_t>(size_t((P / 256 + 2) * NE), q_);
    if (!moe_h_ || !moe_tiles_ || !moe_scratch_) throw std::runtime_error("engine: out of device memory");
    for (void* p : {static_cast<void*>(moe_h_), static_cast<void*>(moe_tiles_), static_cast<void*>(moe_scratch_)}) owned_.push_back(p);
  }
  hq_ = sycl::malloc_device<int8_t>(size_t(B * E), q_);
  hqs_ = dev(B * E / 32);
  owned_.push_back(hq_);
  // Capture of hidden states.
  cap_col_.assign(size_t(cfg_.n_layer), -1);
  for (size_t j = 0; j < c_.capture_layers.size(); ++j) {
    const int l = c_.capture_layers[j];
    if (l < 0 || l >= cfg_.n_layer) throw std::runtime_error("engine: capture layer out of range");
    cap_col_[size_t(l)] = int(j);
  }
  if (!c_.capture_layers.empty() && c_.max_capture_rows > 0) {
    cap_ = dev(c_.max_capture_rows * int64_t(c_.capture_layers.size()) * E);
    cap_idx_ = sycl::malloc_device<int32_t>(size_t(c_.max_capture_rows), q_);
    owned_.push_back(cap_idx_);
    capture_.width = int64_t(c_.capture_layers.size()) * E;
  }
  if (c_.record_routing) {
    counts_.assign(size_t(cfg_.n_layer), std::vector<int64_t>(size_t(NE), 0));
    rec_host_ = sycl::malloc_host<int32_t>(size_t(B * NU), q_);
    owned_.push_back(rec_host_);
  }
  q_.wait();
  load_experts();
}

void Engine::load_experts() {
  const int64_t E = cfg_.n_embd, Fe = cfg_.n_ff_exp, NE = cfg_.n_expert;
  const bool fit = c_.experts_on_card == -2;
  const int64_t want = fit || c_.experts_on_card < 0 || c_.experts_on_card >= NE ? NE : c_.experts_on_card;
  // Each layer's experts by the profile's counts, busiest first, ties to the lower id.
  std::vector<std::vector<int>> order(static_cast<size_t>(cfg_.n_layer), std::vector<int>(static_cast<size_t>(NE)));
  for (auto& o : order)
    for (int e = 0; e < NE; ++e) o[size_t(e)] = e;
  if (!c_.expert_profile.empty()) {
    std::ifstream in(c_.expert_profile);
    if (!in) throw std::runtime_error("engine: cannot read " + c_.expert_profile);
    const json::Value doc = json::parse(std::string((std::istreambuf_iterator<char>(in)), {}));
    const json::Value* cs = doc.find("counts");
    if (!cs || !cs->is_array() || int64_t(cs->items().size()) != cfg_.n_layer)
      throw std::runtime_error(c_.expert_profile + ": counts must hold one row per layer");
    for (int l = 0; l < cfg_.n_layer; ++l) {
      const json::Value& row = cs->items()[size_t(l)];
      if (!row.is_array() || int64_t(row.items().size()) != NE) throw std::runtime_error(c_.expert_profile + ": counts must hold one entry per expert");
      std::vector<int64_t> c;
      for (const auto& v : row.items()) c.push_back(std::atoll(v.int_digits().c_str()));
      std::stable_sort(order[size_t(l)].begin(), order[size_t(l)].end(), [&](int a, int b) { return c[size_t(a)] > c[size_t(b)]; });
    }
  }
  const int64_t slots_wanted = std::min<int64_t>(std::max(c_.stream_slots, 0), 256);
  int64_t n_hot = want;
  if (fit) {
    const std::optional<uint64_t> fm = rt::free_memory(q_.get_device());
    if (!fm) throw std::runtime_error("engine: the driver does not report free device memory; give experts_on_card");
    free_at_placement_ = *fm;
    const uint64_t keep = c_.card_margin_bytes + c_.draft_reserve_bytes + c_.vision_reserve_bytes;
    const uint64_t budget = *fm > keep ? *fm - keep : 0;
    if (budget >= uint64_t(want) * expert_set_bytes_) {
      n_hot = want;
    } else {
      const uint64_t slot_bytes = uint64_t(slots_wanted) * (raw_expert_ + packed_expert_);
      n_hot = budget > slot_bytes ? int64_t((budget - slot_bytes) / expert_set_bytes_) : 0;
    }
  }
  n_hot_ = n_hot;
  if (n_hot_ < NE && c_.expert_profile.empty()) throw std::runtime_error("engine: the experts do not all fit; an expert profile is needed to place some on the CPU");

  auto expert_at = [&](uint8_t* base, xmx::Weight& gu, xmx::Weight& dn) {
    gu = xmx::Weight{E, 2 * Fe, base, reinterpret_cast<sycl::half*>(base + gu_tb_), f_gu_};
    dn = xmx::Weight{Fe, E, base + gu_tb_ + gu_mb_, reinterpret_cast<sycl::half*>(base + gu_tb_ + gu_mb_ + dn_tb_), f_dn_};
  };
  auto repack_into = [&](const ref::Weight& raw, bool interleave, const xmx::Weight& w) {
    if (w.format == xmx::WeightFormat::Q4_0) xmx::repack_q4_0_into(q_, raw, interleave, w);
    else xmx::repack_format_into(q_, raw, interleave, w);
  };

  auto* staging = sycl::malloc_device<uint8_t>(raw_expert_, q_);
  if (!staging) throw std::runtime_error("engine: out of device memory for weight staging");
  for (Layer& L : layers_) {
    L.cold.assign(size_t(NE), 1);
    L.exp_gu.assign(size_t(NE), xmx::Weight{E, 2 * Fe, nullptr, nullptr, f_gu_});  // cold: a table entry with no tiles
    L.exp_dn.assign(size_t(NE), xmx::Weight{Fe, E, nullptr, nullptr, f_dn_});
    L.n_cold = int(NE - n_hot_);
  }
  // Experts go on the card one rank at a time, that rank of every layer, so that the layers stay alike.
  std::vector<uint8_t*> slab(layers_.size(), nullptr);
  if (n_hot_ > 0)
    for (size_t l = 0; l < layers_.size(); ++l) {
      slab[l] = sycl::malloc_device<uint8_t>(size_t(uint64_t(n_hot_) * packed_expert_), q_);
      if (!slab[l]) throw std::runtime_error("engine: out of device memory for the experts");
      owned_.push_back(slab[l]);
    }
  const uint64_t gate_b = uint64_t(Fe) * gu_row_;
  for (int64_t rank = 0; rank < n_hot_; ++rank) {
    for (int l = 0; l < cfg_.n_layer; ++l) {
      Layer& L = layers_[size_t(l)];
      const int e = order[size_t(l)][size_t(rank)];
      L.cold[size_t(e)] = 0;
      // The expert's gate, up and down rows go to the staging buffer and are repacked into its place in the slab.
      q_.memcpy(staging, L.gate_host + size_t(e) * gate_b, size_t(gate_b));
      q_.memcpy(staging + gate_b, L.up_host + size_t(e) * gate_b, size_t(gate_b));
      q_.memcpy(staging + raw_gu_, L.down_host + size_t(e) * raw_dn_, size_t(raw_dn_));
      expert_at(slab[size_t(l)] + uint64_t(rank) * packed_expert_, L.exp_gu[size_t(e)], L.exp_dn[size_t(e)]);
      repack_into(ref::Weight{t_gu_, E, 2 * Fe, staging, gu_row_}, true, L.exp_gu[size_t(e)]);
      repack_into(ref::Weight{t_dn_, Fe, E, staging + raw_gu_, dn_row_}, false, L.exp_dn[size_t(e)]);
      q_.wait();
      weight_bytes_ += raw_expert_;
    }
  }
  sycl::free(staging, q_);
  for (Layer& L : layers_) {
    L.tab_gu = xmx::make_expert_table(q_, L.exp_gu);
    L.tab_dn = xmx::make_expert_table(q_, L.exp_dn);
    for (void* p : {static_cast<void*>(L.tab_gu.tiles), static_cast<void*>(L.tab_gu.scales), static_cast<void*>(L.tab_dn.tiles),
                    static_cast<void*>(L.tab_dn.scales)})
      owned_.push_back(p);
    if (L.n_cold) {
      L.cold_dev = sycl::malloc_device<uint8_t>(size_t(NE), q_);
      if (!L.cold_dev) throw std::runtime_error("engine: out of device memory");
      owned_.push_back(L.cold_dev);
      q_.memcpy(L.cold_dev, L.cold.data(), size_t(NE)).wait();
    }
  }
  if (n_hot_ == NE) return;

  // Streaming slots on the card: a raw one (the bytes as the copy engine lands them) and a repacked one (what the grouped
  // GEMMs read) each.
  n_stream_ = int(std::min<int64_t>(slots_wanted, NE - n_hot_));
  if (n_stream_ > 0) {
    try {
      copy_ = std::make_unique<rt::CopyEngine>(q_);
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "engine: no streaming of cold experts (%s)\n", ex.what());
      n_stream_ = 0;
    }
  }
  if (n_stream_ > 0) {
    const int64_t S = n_stream_, B = c_.max_batch, P = B * cfg_.n_expert_used;
    stream_raw_ = sycl::malloc_device<uint8_t>(size_t(uint64_t(S) * raw_expert_), q_);
    stream_slab_ = sycl::malloc_device<uint8_t>(size_t(uint64_t(S) * packed_expert_), q_);
    if (!stream_raw_ || !stream_slab_) throw std::runtime_error("engine: out of device memory for the streaming slots");
    owned_.push_back(stream_raw_);
    owned_.push_back(stream_slab_);
    stream_w_gu_.resize(size_t(S));
    stream_w_dn_.resize(size_t(S));
    for (int64_t j = 0; j < S; ++j) expert_at(stream_slab_ + uint64_t(j) * packed_expert_, stream_w_gu_[size_t(j)], stream_w_dn_[size_t(j)]);
    stream_gu_ = xmx::make_expert_table(q_, stream_w_gu_);
    stream_dn_ = xmx::make_expert_table(q_, stream_w_dn_);
    for (void* p : {static_cast<void*>(stream_gu_.tiles), static_cast<void*>(stream_gu_.scales), static_cast<void*>(stream_dn_.tiles),
                    static_cast<void*>(stream_dn_.scales)})
      owned_.push_back(p);
    stream_map_ = sycl::malloc_device<int32_t>(size_t(NE), q_);  // a wave's map: expert to slot
    stream_tiles_ = sycl::malloc_device<int32_t>(size_t(4 + 4 * ((P + moe_tile_ - 1) / moe_tile_ + S)), q_);
    stream_scratch_ = sycl::malloc_device<int32_t>(size_t((P / 256 + 2) * S), q_);
    if (!stream_map_ || !stream_tiles_ || !stream_scratch_) throw std::runtime_error("engine: out of device memory for the streaming slots");
    for (void* p : {static_cast<void*>(stream_map_), static_cast<void*>(stream_tiles_), static_cast<void*>(stream_scratch_)}) owned_.push_back(p);
    // The streamed pairs' inputs (fp16, one E row a pair) go in att_, which holds nothing during the MoE.
    stream_rows_cap_ = std::min<int64_t>(B * std::max<int64_t>(cfg_.n_head * cfg_.head_dim, 4 * E) * 4 / (E * 2), (P / moe_tile_ + S) * moe_tile_) /
                       moe_tile_ * moe_tile_;
  }

  // The host side: a pool, buffers the card and the host share, and the cold experts in pinned memory (the CPU computes from
  // there and the copy engine streams from there; the copy also brings the pages in, which a request should not wait for).
  const int threads = c_.cpu_threads > 0 ? c_.cpu_threads : 16;
  pool_ = std::make_unique<cpu::Pool>(threads);
  const int64_t B = std::max<int64_t>(std::max<int64_t>(c_.max_batch, xmx::kFp16Rows), cap_tokens_), P = B * cfg_.n_expert_used;  // a group's tokens too
  auto host = [&](auto*& p, int64_t n) {
    p = sycl::malloc_host<std::remove_reference_t<decltype(*p)>>(size_t(n), q_);
    if (!p) throw std::runtime_error("engine: out of pinned host memory");
    owned_.push_back(p);
  };
  host(hp_ids_, P);
  host(hp_w_, P);
  host(hp_xq_, B * E);
  host(hp_xs_, B * E / 32);
  host(hp_map_, B);
  host(hp_y_, xmx::kFp16Rows * E);
  host(hp_sum_, B * E);  // as many rows as tokens in the worst case
  host(hp_aq_, E / 32 * 256);
  host(hp_as_, E / 32 * 8);
  if (n_stream_ > 0) host(hp_smap_, NE * NE);  // a map per wave of a layer-major group (a wave holds at least one expert)
  cold_map_ = reinterpret_cast<int32_t*>(dev(B));
  // cold_sum_ lives in workspace that is dead during the MoE (qz_: B x 12,288 fp32).
  cold_sum_ = qz_;
  for (Layer& L : layers_) {
    L.cold_gu.assign(size_t(NE), nullptr);
    L.cold_dn.assign(size_t(NE), nullptr);
    std::vector<int> cold;
    for (int e = 0; e < NE; ++e)
      if (L.cold[size_t(e)]) cold.push_back(e);
    auto* blk = sycl::malloc_host<uint8_t>(size_t(uint64_t(cold.size()) * raw_expert_), q_);
    if (!blk) throw std::runtime_error("engine: out of pinned host memory for the cold experts");
    owned_.push_back(blk);
    pinned_bytes_ += uint64_t(cold.size()) * raw_expert_;
    for (size_t j = 0; j < cold.size(); ++j) {
      L.cold_gu[size_t(cold[j])] = blk + j * raw_expert_;
      L.cold_dn[size_t(cold[j])] = blk + j * raw_expert_ + raw_gu_;
    }
    pool_->for_each(int64_t(cold.size()), 1, [&](int64_t lo, int64_t hi) {
      for (int64_t j = lo; j < hi; ++j) {
        const size_t e = size_t(cold[size_t(j)]);
        uint8_t* dst = blk + uint64_t(j) * raw_expert_;
        std::memcpy(dst, L.gate_host + e * gate_b, gate_b);
        std::memcpy(dst + gate_b, L.up_host + e * gate_b, gate_b);
        std::memcpy(dst + raw_gu_, L.down_host + e * raw_dn_, raw_dn_);
      }
    });
  }
}

std::string Engine::placement_line() const {
  char buf[640];
  const double gib = 1073741824.0;
  const int64_t NE = cfg_.n_expert;
  std::snprintf(buf, sizeof buf,
                "%.2f GiB of weights on the card, %.2f GiB of KV and DeltaNet state; experts %s/%s: %lld of %lld per layer on the card "
                "(free %.2f GiB before them, margin %.2f, draft %.2f, vision %.2f), %d streaming slots, %.2f GiB of cold experts in pinned host memory; head %s",
                double(weight_bytes_) / gib, double(cache_bytes_) / gib, gguf::type_name(t_gu_), gguf::type_name(t_dn_), (long long)n_hot_,
                (long long)NE, double(free_at_placement_) / gib, double(c_.card_margin_bytes) / gib, double(c_.draft_reserve_bytes) / gib,
                double(c_.vision_reserve_bytes) / gib, n_stream_, double(pinned_bytes_) / gib, c_.head_dpas ? "dpas GEMV" : "fp32 activations");
  return buf;
}

void Engine::dense(const xmx::Activations& a, const Dense& d, float* out) {
  if (d.parts.size() == 1) return xmx::matmul(q_, a, d.parts[0], out);
  int64_t c0 = 0;
  for (const xmx::Weight& w : d.parts) {
    xmx::matmul(q_, a, w, tmp_);
    qwenk::copy_cols(q_, tmp_, a.m, w.n, out, d.n, c0);
    c0 += w.n;
  }
}

std::vector<int32_t> Engine::cold_list(const Layer& L, const int32_t* ids, int64_t P) const {
  std::vector<int32_t> cp;
  for (int64_t p = 0; p < P; ++p)
    if (L.cold[size_t(ids[p])]) cp.push_back(int32_t(p));
  return cp;
}

void Engine::cold_compute(const Layer& L, const std::vector<int32_t>& cp, const int32_t* ids, const std::vector<cpu::Q8Blocks>& x, float* y) {
  const int64_t E = cfg_.n_embd, Fe = cfg_.n_ff_exp, NE = cfg_.n_expert, NU = cfg_.n_expert_used;
  const int64_t nc = int64_t(cp.size());
  if (!nc) return;
  // The pairs grouped by expert (pair order within an expert): each expert's weights are read once for all its rows.
  struct Seg {
    int e;
    std::vector<const cpu::Q8Blocks*> xs;
    std::vector<float*> ys;
  };
  std::vector<std::vector<int32_t>> by_expert(static_cast<size_t>(NE));
  for (int64_t k = 0; k < nc; ++k) by_expert[size_t(ids[cp[size_t(k)]])].push_back(int32_t(k));
  cold_gu_.resize(size_t(nc * 2 * Fe));
  if (int64_t(cold_g_.size()) < nc) cold_g_.resize(size_t(nc));
  std::vector<Seg> gu_segs, dn_segs;
  for (int e = 0; e < NE; ++e) {
    if (by_expert[size_t(e)].empty()) continue;
    Seg a{e, {}, {}};
    for (int32_t k : by_expert[size_t(e)]) {
      a.xs.push_back(&x[size_t(cp[size_t(k)] / NU)]);
      a.ys.push_back(cold_gu_.data() + int64_t(k) * 2 * Fe);
    }
    gu_segs.push_back(std::move(a));
    Seg b{e, {}, {}};
    for (int32_t k : by_expert[size_t(e)]) {
      b.xs.push_back(&cold_g_[size_t(k)]);
      b.ys.push_back(y + int64_t(k) * E);
    }
    dn_segs.push_back(std::move(b));
  }
  // Tasks are row ranges of one expert's matrix.
  auto run = [&](const std::vector<Seg>& segs, const std::vector<const uint8_t*>& expert, cpu::QType qt, int64_t k, int64_t n, int64_t step) {
    const int64_t per = (n + step - 1) / step;
    pool_->for_each(int64_t(segs.size()) * per, 1, [&](int64_t lo, int64_t hi) {
      for (int64_t t = lo; t < hi; ++t) {
        const Seg& s = segs[size_t(t / per)];
        const int64_t r0 = (t % per) * step, r1 = std::min(n, r0 + step);
        cpu::gemv_q_multi(qt, expert[size_t(s.e)], k, r0, r1, s.xs.data(), int(s.xs.size()), s.ys.data());
      }
    });
  };
  const auto c0 = std::chrono::steady_clock::now();
  run(gu_segs, L.cold_gu, q_gu_, E, 2 * Fe, 32);
  const auto c1 = std::chrono::steady_clock::now();
  // SwiGLU and the int8 quantization of its output, per pair.
  pool_->for_each(nc, 4, [&](int64_t lo, int64_t hi) {
    std::vector<float> g(static_cast<size_t>(Fe));
    for (int64_t k = lo; k < hi; ++k) {
      cpu::glu(cold_gu_.data() + k * 2 * Fe, Fe, cpu::CpuGlu::Silu, g.data());
      cold_g_[size_t(k)] = cpu::quantize(g.data(), Fe);
    }
  });
  const auto c2 = std::chrono::steady_clock::now();
  run(dn_segs, L.cold_dn, q_dn_, Fe, E, 64);
  const auto c3 = std::chrono::steady_clock::now();
  cold_.gu_s += std::chrono::duration<double>(c1 - c0).count();
  cold_.glu_s += std::chrono::duration<double>(c2 - c1).count();
  cold_.dn_s += std::chrono::duration<double>(c3 - c2).count();
  cold_.experts += gu_segs.size();
}

void Engine::cold_decode(const Layer& L, int64_t T, int64_t P, sycl::event routed, sycl::event hot_done) {
  using Clock = std::chrono::steady_clock;
  const int64_t E = cfg_.n_embd, kb = E / 32;
  const auto t0 = Clock::now();
  routed.wait();
  const auto t1 = Clock::now();
  const std::vector<int32_t> cp = cold_list(L, hp_ids_, P);
  pairs_ += uint64_t(P);
  cold_pairs_ += cp.size();
  if (!cp.empty()) {
    // The expert inputs are the card's own int8 rows (the first 8-row tile of act_).
    std::vector<cpu::Q8Blocks> x(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      cpu::Q8Blocks& r = x[size_t(t)];
      r.q.resize(size_t(E));
      r.s.resize(size_t(kb));
      r.sum.resize(size_t(kb));
      for (int64_t b = 0; b < kb; ++b) {
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) {
          const int8_t v = hp_aq_[b * 256 + t * 32 + i];
          r.q[size_t(b * 32 + i)] = v;
          sum += v;
        }
        r.s[size_t(b)] = hp_as_[b * 8 + t];
        r.sum[size_t(b)] = sum;
      }
    }
    cold_compute(L, cp, hp_ids_, x, hp_y_);
    fast::PairList pl{};
    for (size_t k = 0; k < cp.size(); ++k) pl.pair[k] = cp[k];
    fast::cold_scatter(q_, hp_y_, pl, int64_t(cp.size()), E, gx_);
  }
  const auto t2 = Clock::now();
  cold_.wait_s += std::chrono::duration<double>(t1 - t0).count();
  cold_.cpu_s += std::chrono::duration<double>(t2 - t1).count();
  ++cold_.layers;
  if (hot_done.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete) ++cold_.late_layers;
}

// group: counts over a layer-major group; the experts go in waves, so neither the slots nor the gathered rows bound the choice.
std::vector<int32_t> Engine::choose_stream(const Layer& L, const std::vector<int32_t>& cnt, bool group) const {
  const int64_t NE = cfg_.n_expert;
  std::vector<int32_t> cand;
  double rest = 0;
  for (int e = 0; e < NE; ++e)
    if (L.cold[size_t(e)] && cnt[size_t(e)] > 0) {
      rest += cnt[size_t(e)];
      if (cnt[size_t(e)] >= c_.stream_min_rows) cand.push_back(e);
    }
  std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return cnt[size_t(a)] != cnt[size_t(b)] ? cnt[size_t(a)] > cnt[size_t(b)] : a < b; });
  // The copies and the CPU work run side by side: stream the busiest candidates until the copies would take longer than
  // the CPU does on what is left, within the slots and the rows the gathered inputs can hold.
  size_t best_k = 0;
  double best = rest * c_.stream_cpu_us;
  int64_t rows = 0;
  for (size_t k = 1; k <= cand.size() && (group || k <= size_t(n_stream_)); ++k) {
    const int32_t n = cnt[size_t(cand[k - 1])];
    rest -= n;
    rows += (n + moe_tile_ - 1) / moe_tile_ * moe_tile_;
    if (!group && rows > stream_rows_cap_) break;
    const double cost = std::max(double(k) * c_.stream_copy_us, rest * c_.stream_cpu_us);
    if (cost < best) {
      best = cost;
      best_k = k;
    }
  }
  cand.resize(best_k);
  return cand;
}

void Engine::stream_experts(const Layer& L, const std::vector<int32_t>& list, int64_t P, int32_t hot_rows, int64_t s_tiles,
                            const std::function<void(const char*)>& mark) {
  const int64_t E = cfg_.n_embd, Fe = cfg_.n_ff_exp, NU = cfg_.n_expert_used, NE = cfg_.n_expert;
  // The copy engine moves each expert's bytes (gate | up rows, then down, as the GGUF has them) from pinned memory into its raw
  // slot; the host does not wait: the queue does, on the copies' fence, before the repacking, the grouping and the GEMMs, which
  // come after the hot experts'.
  for (size_t j = 0; j < list.size(); ++j) copy_->copy(stream_raw_ + uint64_t(j) * raw_expert_, L.cold_gu[size_t(list[j])], raw_expert_);
  const sycl::event copied = copy_->event(copy_->fence());
  q_.submit([&](sycl::handler& h) {
    h.depends_on(copied);
    h.single_task([] {});
  });
  auto repack_into = [&](const ref::Weight& raw, bool interleave, const xmx::Weight& w) {
    if (w.format == xmx::WeightFormat::Q4_0) xmx::repack_q4_0_into(q_, raw, interleave, w);
    else xmx::repack_format_into(q_, raw, interleave, w);
  };
  for (size_t j = 0; j < list.size(); ++j) {
    const uint8_t* raw = stream_raw_ + uint64_t(j) * raw_expert_;
    repack_into(ref::Weight{t_gu_, E, 2 * Fe, raw, gu_row_}, true, stream_w_gu_[j]);
    repack_into(ref::Weight{t_dn_, Fe, E, raw + raw_gu_, dn_row_}, false, stream_w_dn_[j]);
  }
  q_.memcpy(stream_map_, hp_smap_, size_t(NE) * 4);
  mark("stream_copy_repack");
  // The pairs that chose them get rows after the hot experts' (so that moe_combine takes both from one row table).
  fast::moe_group(q_, moe_ids_, P, n_stream_, moe_tile_, stream_scratch_, stream_tiles_, moe_row_, nullptr, stream_map_, hot_rows);
  sycl::half* a = reinterpret_cast<sycl::half*>(att_);
  fast::moe_gather(q_, h_, moe_row_, P, NU, E, a, hot_rows);
  xmx::matmul_geglu_grouped16(q_, a, stream_gu_, stream_tiles_, s_tiles, moe_tile_, moe_h_ + int64_t(hot_rows) * Fe, xmx::Glu::Silu);
  xmx::matmul_grouped16(q_, moe_h_ + int64_t(hot_rows) * Fe, stream_dn_, stream_tiles_, s_tiles, moe_tile_, moe_y_ + int64_t(hot_rows) * E);
  mark("stream_gemm");
}

void Engine::cold_prefill(const Layer& L, int64_t T, int64_t P, sycl::event ids_ready, sycl::event routed, sycl::event hot_done,
                          const std::function<void(const char*)>& mark) {
  using Clock = std::chrono::steady_clock;
  const int64_t E = cfg_.n_embd, NU = cfg_.n_expert_used, NE = cfg_.n_expert;
  const auto t0 = Clock::now();
  // The routing comes first: it says which cold experts are worth copying, and the copies should start while the
  // hot experts' GEMMs (already enqueued) run.
  ids_ready.wait();
  std::vector<uint8_t> streamed(size_t(NE), 0);
  bool streamed_any = false;
  uint64_t n_streamed_pairs = 0;
  std::vector<int32_t> list;
  if (n_stream_ > 0) {
    std::vector<int32_t> cnt(size_t(NE), 0);
    for (int64_t p = 0; p < P; ++p) ++cnt[size_t(hp_ids_[p])];
    list = choose_stream(L, cnt);
    if (!list.empty()) {
      for (int e = 0; e < NE; ++e) hp_smap_[e] = -1;
      int64_t hot_rows = 0, s_tiles = 0;
      for (int e = 0; e < NE; ++e)
        if (!L.cold[size_t(e)]) hot_rows += (cnt[size_t(e)] + moe_tile_ - 1) / moe_tile_ * moe_tile_;
      for (size_t j = 0; j < list.size(); ++j) {
        hp_smap_[list[j]] = int32_t(j);
        streamed[size_t(list[j])] = 1;
        s_tiles += (cnt[size_t(list[j])] + moe_tile_ - 1) / moe_tile_;
        n_streamed_pairs += uint64_t(cnt[size_t(list[j])]);
      }
      stream_experts(L, list, P, int32_t(hot_rows), s_tiles, mark);
      streamed_any = true;
    }
  }
  routed.wait();
  const auto t1 = Clock::now();
  std::vector<int32_t> cp;
  for (int64_t p = 0; p < P; ++p)
    if (L.cold[size_t(hp_ids_[p])] && !streamed[size_t(hp_ids_[p])]) cp.push_back(int32_t(p));
  pairs_ += uint64_t(P);
  cold_pairs_ += cp.size() + n_streamed_pairs;
  std::vector<int32_t> k_of(size_t(P), -1);
  std::vector<char> need(size_t(T), 0);
  for (size_t k = 0; k < cp.size(); ++k) {
    k_of[size_t(cp[k])] = int32_t(k);
    need[size_t(cp[k] / NU)] = 1;
  }
  const int64_t kb = E / 32;
  std::vector<cpu::Q8Blocks> x(static_cast<size_t>(T));
  int32_t n_need = 0;
  for (int64_t t = 0; t < T; ++t) hp_map_[t] = need[size_t(t)] ? n_need++ : -1;
  pool_->for_each(T, 8, [&](int64_t lo, int64_t hi) {
    for (int64_t t = lo; t < hi; ++t) {
      if (!need[size_t(t)]) continue;
      cpu::Q8Blocks& r = x[size_t(t)];
      r.q.assign(hp_xq_ + t * E, hp_xq_ + (t + 1) * E);
      r.s.assign(hp_xs_ + t * kb, hp_xs_ + (t + 1) * kb);
      r.sum.resize(size_t(kb));
      for (int64_t b = 0; b < kb; ++b) {
        int32_t sum = 0;
        for (int i = 0; i < 32; ++i) sum += r.q[size_t(b * 32 + i)];
        r.sum[size_t(b)] = sum;
      }
    }
  });
  cold_y_.resize(cp.size() * size_t(E));
  cold_compute(L, cp, hp_ids_, x, cold_y_.data());
  // Each token's cold share, weighted and summed in slot order (zero for the others).
  pool_->for_each(T, 16, [&](int64_t lo, int64_t hi) {
    for (int64_t t = lo; t < hi; ++t) {
      if (!need[size_t(t)]) continue;
      float* out = hp_sum_ + int64_t(hp_map_[t]) * E;
      bool first = true;
      for (int64_t i = 0; i < NU; ++i) {
        const int32_t k = k_of[size_t(t * NU + i)];
        if (k < 0) continue;
        const float w = hp_w_[t * NU + i];
        const float* yk = cold_y_.data() + int64_t(k) * E;
        if (first)
          for (int64_t j = 0; j < E; ++j) out[j] = w * yk[j];
        else
          for (int64_t j = 0; j < E; ++j) out[j] += w * yk[j];
        first = false;
      }
    }
  });
  const auto t2 = Clock::now();
  if (streamed_any) {
    ++stream_.layers;
    stream_.experts += list.size();
    stream_.pairs += n_streamed_pairs;
  }
  q_.memcpy(cold_sum_, hp_sum_, size_t(n_need) * size_t(E) * sizeof(float));
  q_.memcpy(cold_map_, hp_map_, size_t(T) * 4);
  cold_.wait_s += std::chrono::duration<double>(t1 - t0).count();
  cold_.cpu_s += std::chrono::duration<double>(t2 - t1).count();
  ++cold_.layers;
  if (hot_done.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete) ++cold_.late_layers;
}

Engine::~Engine() {
  q_.wait();
  xmx::free_activations(q_, act_);
  xmx::free_activations(q_, act_sh_);
  xmx::free_activations(q_, act_exp_ffn_);
  if (act_head_.q) xmx::free_activations(q_, act_head_);
  gdn::free_scratch(q_, gscratch_);
  gdn::free_cache(q_, gcache_);
  if (attn_scratch_.q) sycl::free(attn_scratch_.q, q_);
  if (attn_scratch_.part) sycl::free(attn_scratch_.part, q_);
  if (attn_scratch_.part2) sycl::free(attn_scratch_.part2, q_);
  if (attn_scratch_.p) sycl::free(attn_scratch_.p, q_);
  if (attn_scratch_.stat) sycl::free(attn_scratch_.stat, q_);
  for (void* p : owned_) sycl::free(p, q_);
}

void Engine::ensure_pages(int slot, int64_t positions) {
  Slot& s = slots_[size_t(slot)];
  const int64_t need = (positions + attn::kPage - 1) / attn::kPage;
  if (need > max_pages_) throw std::runtime_error("engine: sequence longer than max_ctx");
  while (int64_t(s.pages.size()) < need) {
    if (free_pages_.empty()) throw std::runtime_error("engine: out of KV pages");
    s.pages.push_back(free_pages_.back());
    free_pages_.pop_back();
    page_table_host_[size_t(slot * max_pages_ + int64_t(s.pages.size()) - 1)] = s.pages.back();
  }
}

void Engine::free_pages_beyond(Slot& s, int64_t length) {
  const size_t keep = size_t((length + attn::kPage - 1) / attn::kPage);
  while (s.pages.size() > keep) {
    free_pages_.push_back(s.pages.back());
    s.pages.pop_back();
  }
}

int Engine::accepted(const Slot& s, int accept) const {
  if (s.pending == 0) {
    if (accept > 0) throw std::runtime_error("engine: accept given for a slot with nothing pending");
    return 0;
  }
  const int a = accept < 0 ? s.pending : accept;
  if (a > s.pending) throw std::runtime_error("engine: accept " + std::to_string(a) + " of " + std::to_string(s.pending) + " pending tokens");
  return a;
}

void Engine::upload_plan(const std::vector<gdn::SlotStep>& steps, gdn::Plan& plan, size_t at) {
  plan = gdn::Plan{};
  plan.n_slots = int32_t(steps.size());
  int32_t shift = 0;
  for (size_t i = 0; i < steps.size(); ++i) {
    gdn::SlotStep st = steps[i];
    shift += st.accept;
    st.shift = shift;
    plan.n_tokens += st.n;
    plan.max_n = std::max(plan.max_n, st.n);
    plan_host_[at + i] = st;
  }
  plan.n_rows = plan.n_tokens + shift;
  q_.memcpy(plan_dev_ + at, plan_host_ + at, steps.size() * sizeof(gdn::SlotStep));
  plan.dev = plan_dev_ + at;
}

void Engine::release(int slot) {
  Slot& s = slots_.at(size_t(slot));
  for (int32_t p : s.pages) free_pages_.push_back(p);
  s.pages.clear();
  s.length = 0;
  s.pending = 0;
  books_[size_t(slot)].pending = 0;
  gdn::reset_slot(q_, gcache_, slot);
}

void Engine::truncate(int slot, int64_t length) {
  Slot& s = slots_.at(size_t(slot));
  if (s.pending) throw std::runtime_error("engine: truncate with tokens pending");
  if (length != s.length) throw std::runtime_error("engine: the K/V alone cannot be cut: the DeltaNet state goes with it (restore a snapshot)");
}

void Engine::settle(int slot, int accept) {
  Slot& s = slots_.at(size_t(slot));
  if (!s.pending) return;
  const int acc = accepted(s, accept);
  const gdn::SlotStep st = books_[size_t(slot)].next(slot, 0, 0, acc, 0, c_.max_pending);
  gdn::Plan plan;
  upload_plan({st}, plan);
  const gdn::Inputs in{qz_, qz_, ab_, ab_};
  for (const Layer& L : layers_)
    if (L.recurrent) gdn::step(q_, L.gw, gcache_, L.gdn_idx, plan, in, gscratch_, act_);
  q_.wait();
  s.length = s.verify_start + acc;
  free_pages_beyond(s, s.length);
  s.pending = 0;
}

Engine::Snapshot Engine::snapshot(int slot) {
  Slot& s = slots_.at(size_t(slot));
  if (s.pending) throw std::runtime_error("engine: snapshot with tokens pending (settle the slot first)");
  Snapshot snap;
  snap.bytes = uint64_t(gcache_.snapshot_bytes());
  sycl::queue* q = &q_;
  snap.host = std::shared_ptr<uint8_t>(sycl::malloc_host<uint8_t>(size_t(snap.bytes), q_), [q](uint8_t* p) { sycl::free(p, *q); });
  if (!snap.host) throw std::runtime_error("engine: out of pinned memory for a snapshot");
  gdn::snapshot_to_host(q_, gcache_, slot, snap.host.get()).wait();
  snap.position = s.length;
  return snap;
}

void Engine::restore(int slot, const Snapshot& snap, int64_t length) {
  Slot& s = slots_.at(size_t(slot));
  if (s.pending) throw std::runtime_error("engine: restore with tokens pending");
  if (!snap.host || snap.bytes != uint64_t(gcache_.snapshot_bytes())) throw std::runtime_error("engine: not a snapshot of this engine");
  if (length != snap.position) throw std::runtime_error("engine: a snapshot restores only to its own position");
  if (length > s.length) throw std::runtime_error("engine: the slot does not hold the snapshot's positions");
  gdn::restore_from_host(q_, gcache_, slot, snap.host.get()).wait();
  free_pages_beyond(s, length);
  s.length = length;
}

std::vector<float> Engine::step(const std::vector<Chunk>& chunks, const model::DumpSink* dump) {
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, Hkv = cfg_.n_head_kv, D = cfg_.head_dim, V = cfg_.n_vocab;
  const int64_t NE = cfg_.n_expert, NU = cfg_.n_expert_used;
  const int ncap = int(c_.capture_layers.size());

  // Validation, before anything changes.
  std::set<int> seen;
  std::vector<int> acc(chunks.size(), 0);
  int64_t n_tok = 0, n_logits = 0, n_cap = 0;
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Chunk& c = chunks[i];
    if (c.slot < 0 || c.slot >= c_.n_slots) throw std::runtime_error("engine: bad slot");
    if (!seen.insert(c.slot).second) throw std::runtime_error("engine: two chunks of one slot in a step");
    const int64_t n = int64_t(c.tokens.size());
    if (n == 0) throw std::runtime_error("engine: empty chunk (settle() commits pending tokens alone)");
    if (n > c_.max_chunk && !(chunks.size() == 1 && !c.verify && n <= c_.max_group)) throw std::runtime_error("engine: chunk longer than max_chunk");
    if (c.verify && n > c_.max_pending) throw std::runtime_error("engine: verify chunk longer than max_pending");
    const Slot& s = slots_[size_t(c.slot)];
    acc[i] = accepted(s, c.accept);
    const int64_t at = s.pending ? s.verify_start + acc[i] : s.length;
    if (c.start_pos != at) throw std::runtime_error("engine: chunk does not continue its slot");
    if (c.capture_rows < 0 || c.capture_rows > n || (c.capture_rows > 0 && !cap_)) throw std::runtime_error("engine: bad capture_rows");
    for (int32_t t : c.tokens)
      if (t < 0 || t >= V) throw std::runtime_error("engine: token id out of range");
    {
      int64_t at_row = 0;
      for (const model::ImageRows& im : c.images) {
        if (im.n <= 0 || im.offset < at_row || im.offset + im.n > n || !im.rows || im.nx <= 0 || im.ny <= 0 || im.nx * im.ny != im.n)
          throw std::runtime_error("engine: bad image span (rows, grid, order or place)");
        at_row = im.offset + im.n;
      }
      if (c.rope_delta < 0 || c.start_pos - c.rope_delta < 0) throw std::runtime_error("engine: bad rope_delta");
    }
    if (at + n > c_.max_ctx) throw std::runtime_error("engine: sequence longer than max_ctx");
    n_tok += n;
    n_logits += std::min(n, std::max<int64_t>(c.logits_rows, 0));
    n_cap += c.capture_rows;
  }
  const bool group_step = chunks.size() == 1 && n_tok > c_.max_chunk;
  if (n_tok > c_.max_batch && !group_step) throw std::runtime_error("engine: step larger than max_batch");
  if (n_logits > c_.max_logits) throw std::runtime_error("engine: too many logits rows");
  if (n_cap > c_.max_capture_rows) throw std::runtime_error("engine: too many capture rows");
  if (group_step) return step_group(chunks[0], acc[0]);

  std::vector<int32_t> tok, slot, pos, logit_rows, cap_rows, rpos;  // rpos: [3][T] once built
  std::vector<int32_t> rpos_t, rpos_h, rpos_w;
  std::vector<gdn::SlotStep> gsteps;
  attn::TileList tiles(D);
  capture_.ids.clear();
  for (size_t i = 0; i < chunks.size(); ++i) {
    const Chunk& c = chunks[i];
    Slot& s = slots_[size_t(c.slot)];
    if (s.pending) {
      s.length = s.verify_start + acc[i];
      free_pages_beyond(s, s.length);
    }
    const int64_t base = int64_t(tok.size()), n = int64_t(c.tokens.size());
    for (int64_t j = 0; j < n; ++j) {
      tok.push_back(c.tokens[size_t(j)]);
      slot.push_back(c.slot);
      pos.push_back(int32_t(c.start_pos + j));
    }
    // The rotary positions (time, row, column) of the chunk's tokens: text at its sequence index less the images' saving so far,
    // an image's rows on its grid.
    {
      int64_t delta = c.rope_delta;
      size_t k = 0;
      for (int64_t j = 0; j < n;) {
        if (k < c.images.size() && j == c.images[k].offset) {
          const model::ImageRows& im = c.images[k];
          const int64_t p = c.start_pos + j - delta;
          for (int64_t r = 0; r < im.n; ++r) {
            rpos_t.push_back(int32_t(p)), rpos_h.push_back(int32_t(p + r / im.nx)), rpos_w.push_back(int32_t(p + r % im.nx));
          }
          delta += im.n - im.n_pos();
          j += im.n;
          ++k;
        } else {
          const int32_t p = int32_t(c.start_pos + j - delta);
          rpos_t.push_back(p), rpos_h.push_back(p), rpos_w.push_back(p);
          ++j;
        }
      }
    }
    tiles.add_chunk(int32_t(base), n, c.start_pos + n - 1, group_, Hkv, 0);
    for (int64_t r = std::max<int64_t>(0, n - c.logits_rows); r < n; ++r) logit_rows.push_back(int32_t(base + r));
    for (int64_t r = n - c.capture_rows; r < n; ++r) {
      cap_rows.push_back(int32_t(base + r));
      capture_.ids.push_back({int32_t(c.slot), int32_t(c.start_pos + r)});
    }
    ensure_pages(c.slot, c.start_pos + n);
    gsteps.push_back(books_[size_t(c.slot)].next(c.slot, int32_t(base), int32_t(n), acc[i], c.verify ? 0 : int32_t(n), c_.max_pending));
    s.pending = c.verify ? int(n) : 0;
    s.verify_start = c.start_pos;
    s.length = c.start_pos + n;
  }
  const int64_t T = int64_t(tok.size()), R = int64_t(logit_rows.size());
  capture_.rows = int64_t(cap_rows.size());
  capture_.data = cap_;
  if (T == 0) return {};
  const int64_t P = T * NU;
  const bool small = P <= xmx::kFp16Rows;  // decode's grouped GEMVs: up to 4 tokens
  // Decode calls the pool every few hundred microseconds and wants its workers spinning; prefill's phases
  // last milliseconds and the spinning only takes the cores from the hot kernels' host thread.
  if (pool_) pool_->set_spin_us(T <= 16 ? 400 : 0);
  const auto t_step = std::chrono::steady_clock::now();

  // The embedding rows of the step's tokens, gathered on the host from the mapping.
  if (!c_.embed_on_card) {
    for (int64_t t = 0; t < T; ++t)
      std::memcpy(emb_host_ + uint64_t(t) * embd_row_bytes_, embd_->data + uint64_t(tok[size_t(t)]) * embd_row_bytes_, embd_row_bytes_);
    q_.memcpy(emb_dev_, emb_host_, size_t(uint64_t(T) * embd_row_bytes_));
  }
  q_.memcpy(tok_, tok.data(), size_t(T) * 4);
  q_.memcpy(slot_, slot.data(), size_t(T) * 4);
  q_.memcpy(pos_, pos.data(), size_t(T) * 4);
  rpos.insert(rpos.end(), rpos_t.begin(), rpos_t.end());
  rpos.insert(rpos.end(), rpos_h.begin(), rpos_h.end());
  rpos.insert(rpos.end(), rpos_w.begin(), rpos_w.end());
  q_.memcpy(rpos_, rpos.data(), size_t(3 * T) * 4);
  const std::vector<int32_t> packed = tiles.packed();
  q_.memcpy(tiles_, packed.data(), packed.size() * 4);
  q_.memcpy(page_table_, page_table_host_.data(), page_table_host_.size() * 4);
  if (R) q_.memcpy(rows_idx_, logit_rows.data(), size_t(R) * 4);
  if (!cap_rows.empty()) q_.memcpy(cap_idx_, cap_rows.data(), cap_rows.size() * 4);
  gdn::Plan plan;
  upload_plan(gsteps, plan);
  const attn::Tokens toks{slot_, pos_, T};

  // On a profiling queue each mark is an empty kernel whose end timestamp closes the section, so
  // the card's own time is measured; otherwise the host waits at each mark.
  const bool gpu_marks = profile_ && q_.has_property<sycl::property::queue::enable_profiling>();
  std::vector<std::pair<const char*, sycl::event>> marks;
  auto settle_marks = [&] {
    if (!gpu_marks || marks.empty()) return;
    namespace ep = sycl::info::event_profiling;
    uint64_t prev = marks[0].second.get_profiling_info<ep::command_end>();
    for (size_t i = 1; i < marks.size(); ++i) {
      const uint64_t end = marks[i].second.get_profiling_info<ep::command_end>();
      (*profile_)[marks[i].first] += double(end - prev) * 1e-9;
      prev = end;
    }
  };
  auto t_mark = std::chrono::steady_clock::now();
  auto mark = [&](const char* name) {
    if (!profile_) return;
    if (gpu_marks) {
      marks.push_back({name, q_.single_task([] {})});
      return;
    }
    q_.wait();
    const auto now = std::chrono::steady_clock::now();
    (*profile_)[name] += std::chrono::duration<double>(now - t_mark).count();
    t_mark = now;
  };
  mark("upload");

  auto emit = [&](const std::string& name, const float* p, std::vector<int64_t> ne) {
    if (!dump || !dump->wants(name)) return;
    int64_t n = 1;
    for (auto e : ne) n *= e;
    std::vector<float> host(static_cast<size_t>(n));
    q_.memcpy(host.data(), p, size_t(n) * sizeof(float)).wait();
    dump->put(name, host, ne);
  };

  // Each layer starts with its input already normed and quantized in act_ (and, for DeltaNet layers, in fp32 in h_ for the
  // alpha / beta matmul): by the fused kernel at the end of the layer before, or here for the first.
  if (c_.embed_on_card) ref::embed(q_, embd_card_, tok_, T, 1.0f, x_);
  else ref::embed(q_, ref::Weight{embd_->type, E, T, emb_dev_, embd_row_bytes_}, iota_, T, 1.0f, x_);
  // Image rows replace the lookups (unscaled: Qwen's embeddings are not scaled either).
  for (size_t i = 0, base = 0; i < chunks.size(); base += chunks[i].tokens.size(), ++i)
    for (const model::ImageRows& im : chunks[i].images)
      q_.memcpy(x_ + (int64_t(base) + im.offset) * E, im.rows, size_t(im.n * E) * sizeof(float));
  emit("model.input_embed", x_, {E, T});
  qwenk::add_norm_quant(q_, nullptr, x_, layers_[0].attn_norm, cfg_.eps, T, E, nullptr, &act_, h_);
  mark("embed");
  const float attn_scale = 1.0f / std::sqrt(float(D));
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const Layer& L = layers_[size_t(l)];
    const std::string s = dump ? "-" + std::to_string(l) : std::string();
    if (dump) emit("attn_norm" + s, h_, {E, T});

    if (L.recurrent) {
      dense(act_, L.in, qz_);
      mark("qkvz");
      if (T <= 16) fast::matmul_f32_small(q_, L.w_ab, 2 * gdn::kValueHeads, E, h_, T, ab_);
      else fast::matmul_f32(q_, L.w_ab, 2 * gdn::kValueHeads, E, h_, T, ab_);
      const gdn::Inputs in{qz_, qz_ + gdn::kChannels, ab_, ab_ + gdn::kValueHeads, L.in.n, L.in.n, 2 * gdn::kValueHeads, 2 * gdn::kValueHeads};
      mark("alpha_beta");
      gdn::step(q_, L.gw, gcache_, L.gdn_idx, plan, in, gscratch_, act_);
      mark("gdn");
      dense(act_, L.out, o_);
      mark("ssm_out");
      if (dump) emit("linear_attn_out" + s, o_, {E, T, 1});
    } else {
      dense(act_, L.in, qz_);
      mark("qkv");
      qwenk::attn_heads(q_, qz_, T, H, Hkv, D, cfg_.n_rot, L.q_norm, L.k_norm, cfg_.eps, rpos_, cfg_.rope_sections, inv_freq_, qb_, kb_, vb_, gt_);
      mark("attn_heads");
      const KvLayer& kv = kv_[size_t(L.attn_idx)];
      const attn::Cache cache{kv.k, kv.v, page_table_, max_pages_, 0, Hkv, D, kv.ks, kv.vs};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles, tiles_, cache, H, attn_scale, att_, attn_scratch_);
      mark("attn");
      qwenk::gate_quant(q_, att_, gt_, T, H * D, act_);
      dense(act_, L.out, o_);
      mark("attn_out");
      if (dump) emit("attn_output" + s, o_, {E, T});
    }
    // x2 = x + out; m = rmsnorm(x2) * post_attention_norm, fp32 in h_ and quantized in act_.
    qwenk::add_norm_quant(q_, o_, x_, L.post_norm, cfg_.eps, T, E, x2_, &act_, h_);
    if (dump) {
      emit("attn_residual" + s, x2_, {E, T});
      emit("attn_post_norm" + s, h_, {E, T});
    }
    mark("post_attn");

    // The routed experts and the shared expert (gate | up interleaved, SwiGLU, down, into o_).
    const float* next_norm = l + 1 < cfg_.n_layer ? layers_[size_t(l) + 1].attn_norm : nullptr;
    const bool next_h = next_norm && (layers_[size_t(l) + 1].recurrent || (dump && dump->wants("attn_norm-" + std::to_string(l + 1))));
    const float* sgate;
    int64_t ld_sg;
    if (T <= 16) {
      fast::matmul_f32_small(q_, L.w_rg, NE + 32, E, h_, T, rg_);
      mark("router");
      if (dump && dump->wants("ffn_moe_logits" + s)) {
        std::vector<float> host(size_t(T * (NE + 32))), logits(size_t(T * NE));
        q_.memcpy(host.data(), rg_, host.size() * 4).wait();
        for (int64_t t = 0; t < T; ++t) std::memcpy(&logits[size_t(t * NE)], &host[size_t(t * (NE + 32))], size_t(NE) * 4);
        dump->put("ffn_moe_logits" + s, logits, {NE, T});
      }
      fast::moe_route(q_, rg_, T, NE, NU, nullptr, moe_ids_, moe_w_, NE + 32);
      sgate = rg_ + NE;
      ld_sg = NE + 32;
    } else {
      fast::matmul_f32(q_, L.w_rg, NE, E, h_, T, rlog_);
      fast::matmul_f32(q_, L.w_rg + NE * E, 32, E, h_, T, sg_);
      mark("router");
      if (dump) emit("ffn_moe_logits" + s, rlog_, {NE, T});
      fast::moe_route(q_, rlog_, T, NE, NU, nullptr, moe_ids_, moe_w_);
      sgate = sg_;
      ld_sg = 32;
    }
    mark("moe_route");
    if (c_.record_routing) {
      q_.memcpy(rec_host_, moe_ids_, size_t(P) * 4).wait();
      for (int64_t p = 0; p < P; ++p) ++counts_[size_t(l)][size_t(rec_host_[p])];
    }
    if (small) {
      // Decode (up to 4 tokens): the experts' grouped GEMVs and the weighted sum on the card; the cold experts' rows come from
      // the host after the hot kernels are enqueued, so the card works meanwhile.
      sycl::event routed;
      if (L.n_cold) {
        const int64_t kb = E / 32;
        routed = fast::host_pack(q_, moe_ids_, P, act_.q, kb * 256, act_.s, kb * 8, hp_ids_, hp_aq_, hp_as_);
      }
      xmx::matmul_geglu_grouped(q_, act_, L.tab_gu, moe_ids_, P, NU, act_exp_ffn_, moe_gu_, xmx::Glu::Silu);
      mark("moe_gate_up");
      xmx::matmul_grouped(q_, act_exp_ffn_, L.tab_dn, moe_ids_, P, 1, gx_);
      mark("moe_down");
      xmx::matmul_geglu(q_, act_, L.shexp_gu, act_sh_, sh_gu_, xmx::Glu::Silu);
      xmx::matmul(q_, act_sh_, L.shexp_dn, o_);
      mark("shexp");
      if (dump) emit("ffn_shexp" + s, o_, {E, T});
      if (L.n_cold) cold_decode(L, T, P, routed, q_.single_task([] {}));
      qwenk::moe_tail(q_, x2_, gx_, moe_w_, NU, nullptr, sgate, ld_sg, o_, next_norm, cfg_.eps, T, E, x_, &act_, next_h ? h_ : nullptr);
    } else {
      // Prefill and verify: the pairs grouped by expert into row tiles, the experts' GEMMs over all their tokens at once.
      sycl::event routed, ids_ready;
      if (L.n_cold) {
        // As at decode, the host waits for the routing and the input rows after the hot work is enqueued (the ids first: they say
        // which cold experts to stream). The expert inputs go as int8 (the form the CPU computes with), a quarter of the bytes.
        ids_ready = q_.memcpy(hp_ids_, moe_ids_, size_t(P) * 4);
        q_.memcpy(hp_w_, moe_w_, size_t(P) * 4);
        fast::quantize_rows(q_, h_, T, E, hq_, hqs_);
        q_.memcpy(hp_xq_, hq_, size_t(T * E));
        routed = q_.memcpy(hp_xs_, hqs_, size_t(T) * size_t(E / 32) * sizeof(float));
      }
      fast::moe_group(q_, moe_ids_, P, NE, moe_tile_, moe_scratch_, moe_tiles_, moe_row_, L.cold_dev);
      mark("moe_group");
      fast::moe_gather(q_, h_, moe_row_, P, NU, E, moe_a_);
      mark("moe_gather");
      const int64_t max_tiles = (P + moe_tile_ - 1) / moe_tile_ + NE;
      xmx::matmul_geglu_grouped16(q_, moe_a_, L.tab_gu, moe_tiles_, max_tiles, moe_tile_, moe_h_, xmx::Glu::Silu);
      mark("moe_gate_up");
      xmx::matmul_grouped16(q_, moe_h_, L.tab_dn, moe_tiles_, max_tiles, moe_tile_, moe_y_);
      mark("moe_down");
      xmx::matmul_geglu(q_, act_, L.shexp_gu, act_sh_, sh_gu_, xmx::Glu::Silu);
      xmx::matmul(q_, act_sh_, L.shexp_dn, o_);
      mark("shexp");
      if (dump) emit("ffn_shexp" + s, o_, {E, T});
      if (L.n_cold) cold_prefill(L, T, P, ids_ready, routed, q_.single_task([] {}), mark);
      fast::moe_combine(q_, moe_y_, moe_w_, T, NU, E, moe_, moe_row_, L.n_cold ? cold_sum_ : nullptr, L.n_cold ? cold_map_ : nullptr);
      mark("moe_sum");
      if (dump) emit("ffn_moe_out" + s, moe_, {E, T});
      qwenk::moe_tail(q_, x2_, nullptr, nullptr, NU, moe_, sgate, ld_sg, o_, next_norm, cfg_.eps, T, E, x_, &act_, next_h ? h_ : nullptr);
    }
    if (dump) emit("l_out" + s, x_, {E, T});
    if (cap_col_[size_t(l)] >= 0 && !cap_rows.empty())
      qwenk::gather_rows(q_, x_, cap_idx_, int64_t(cap_rows.size()), E, cap_, int64_t(ncap) * E, int64_t(cap_col_[size_t(l)]) * E);
    mark("tail");
  }

  if (R == 0) {
    enqueue_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
    q_.wait();
    settle_marks();
    return {};
  }
  // Logits only for the rows asked for: gather them, final norm, head.
  qwenk::gather_rows(q_, x_, rows_idx_, R, E, rows_, E, 0);
  fast::rmsnorm(q_, rows_, output_norm_, R, E, cfg_.eps, rows_);
  emit("result_norm", rows_, {E, R});
  if (c_.head_dpas) {
    xmx::quantize(q_, rows_, R, act_head_);
    xmx::matmul(q_, act_head_, head_w_, logits_);
  } else {
    fast::matmul_q6(q_, head_q6_, rows_, R, logits_);
  }
  mark("head");
  std::vector<float> out(size_t(R * V));
  auto done = q_.memcpy(out.data(), logits_, out.size() * sizeof(float));
  enqueue_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
  done.wait();
  settle_marks();
  return out;
}

// A lone prefill chunk of more than max_chunk tokens, layer by layer. The chunk is cut into sub-chunks of near-equal length (at most
// max_chunk); for each layer every sub-chunk in turn runs the mixer, the router, the hot experts and the shared expert (what step()
// does for a chunk) and adds them to its rows of the residual stream x_ in place; then the layer's cold experts are brought in
// once for the whole group (cold_group). Causality holds because sub-chunk c's layer l runs after sub-chunk c-1's, which left the
// layer's recurrent state and K/V as a plain chunk would. Not supported: dumps.
std::vector<float> Engine::step_group(const Chunk& c, int acc) {
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, Hkv = cfg_.n_head_kv, D = cfg_.head_dim, V = cfg_.n_vocab;
  const int64_t NE = cfg_.n_expert, NU = cfg_.n_expert_used, kb = E / 32;
  const int ncap = int(c_.capture_layers.size());
  const int64_t n = int64_t(c.tokens.size());
  Slot& s = slots_[size_t(c.slot)];
  if (s.pending) {
    s.length = s.verify_start + acc;
    free_pages_beyond(s, s.length);
  }
  const int64_t G = (n + c_.max_chunk - 1) / c_.max_chunk;
  std::vector<GroupSub> subs(static_cast<size_t>(G));
  for (int64_t k = 0, at = 0; k < G; ++k) {
    subs[size_t(k)] = {at, n / G + (k < n % G ? 1 : 0)};
    at += subs[size_t(k)].len;
  }

  std::vector<int32_t> tok, slot(size_t(n), c.slot), pos, logit_rows, cap_rows;
  std::vector<int32_t> rpos_t, rpos_h, rpos_w;
  for (int64_t j = 0; j < n; ++j) {
    tok.push_back(c.tokens[size_t(j)]);
    pos.push_back(int32_t(c.start_pos + j));
  }
  {
    int64_t delta = c.rope_delta;
    size_t k = 0;
    for (int64_t j = 0; j < n;) {
      if (k < c.images.size() && j == c.images[k].offset) {
        const model::ImageRows& im = c.images[k];
        const int64_t p = c.start_pos + j - delta;
        for (int64_t r = 0; r < im.n; ++r) {
          rpos_t.push_back(int32_t(p)), rpos_h.push_back(int32_t(p + r / im.nx)), rpos_w.push_back(int32_t(p + r % im.nx));
        }
        delta += im.n - im.n_pos();
        j += im.n;
        ++k;
      } else {
        const int32_t p = int32_t(c.start_pos + j - delta);
        rpos_t.push_back(p), rpos_h.push_back(p), rpos_w.push_back(p);
        ++j;
      }
    }
  }
  capture_.ids.clear();
  for (int64_t r = std::max<int64_t>(0, n - c.logits_rows); r < n; ++r) logit_rows.push_back(int32_t(r));
  for (int64_t r = n - c.capture_rows; r < n; ++r) {
    cap_rows.push_back(int32_t(r));
    capture_.ids.push_back({int32_t(c.slot), int32_t(c.start_pos + r)});
  }
  ensure_pages(c.slot, c.start_pos + n);
  // Per sub-chunk: its rotary positions [3][len], its attention tiles, its DeltaNet plan (the first replays the pending tokens).
  std::vector<std::vector<int32_t>> rblocks(static_cast<size_t>(G));
  std::vector<attn::TileList> tiles(static_cast<size_t>(G), attn::TileList{D});
  std::vector<gdn::Plan> plans(static_cast<size_t>(G));
  for (int64_t k = 0; k < G; ++k) {
    const GroupSub& sb = subs[size_t(k)];
    auto& rb = rblocks[size_t(k)];
    for (const auto* v : {&rpos_t, &rpos_h, &rpos_w}) rb.insert(rb.end(), v->begin() + sb.off, v->begin() + sb.off + sb.len);
    tiles[size_t(k)].add_chunk(0, sb.len, c.start_pos + sb.off + sb.len - 1, group_, Hkv, 0);
    const gdn::SlotStep st = books_[size_t(c.slot)].next(c.slot, 0, int32_t(sb.len), k == 0 ? acc : 0, int32_t(sb.len), c_.max_pending);
    upload_plan({st}, plans[size_t(k)], size_t(k));
  }
  s.pending = 0;
  s.verify_start = c.start_pos;
  s.length = c.start_pos + n;
  const int64_t R = int64_t(logit_rows.size());
  capture_.rows = int64_t(cap_rows.size());
  capture_.data = cap_;
  if (pool_) pool_->set_spin_us(0);

  q_.memcpy(tok_, tok.data(), size_t(n) * 4);
  q_.memcpy(slot_, slot.data(), size_t(n) * 4);
  q_.memcpy(pos_, pos.data(), size_t(n) * 4);
  for (int64_t k = 0; k < G; ++k) {
    q_.memcpy(rpos_ + 3 * subs[size_t(k)].off, rblocks[size_t(k)].data(), rblocks[size_t(k)].size() * 4);
    const std::vector<int32_t> packed = tiles[size_t(k)].packed();
    if (int64_t(packed.size()) > 6 * max_tiles_) throw std::runtime_error("engine: tile list larger than its buffer");
    q_.memcpy(tiles_ + k * 6 * max_tiles_, packed.data(), packed.size() * 4).wait();
  }
  q_.memcpy(page_table_, page_table_host_.data(), page_table_host_.size() * 4);
  if (R) q_.memcpy(rows_idx_, logit_rows.data(), size_t(R) * 4);
  if (!cap_rows.empty()) q_.memcpy(cap_idx_, cap_rows.data(), cap_rows.size() * 4);

  const bool gpu_marks = profile_ && q_.has_property<sycl::property::queue::enable_profiling>();
  std::vector<std::pair<const char*, sycl::event>> marks;
  auto settle_marks = [&] {
    if (!gpu_marks || marks.empty()) return;
    namespace ep = sycl::info::event_profiling;
    uint64_t prev = marks[0].second.get_profiling_info<ep::command_end>();
    for (size_t i = 1; i < marks.size(); ++i) {
      const uint64_t end = marks[i].second.get_profiling_info<ep::command_end>();
      (*profile_)[marks[i].first] += double(end - prev) * 1e-9;
      prev = end;
    }
  };
  auto t_mark = std::chrono::steady_clock::now();
  auto mark = [&](const char* name) {
    if (!profile_) return;
    if (gpu_marks) {
      marks.push_back({name, q_.single_task([] {})});
      return;
    }
    q_.wait();
    const auto now = std::chrono::steady_clock::now();
    (*profile_)[name] += std::chrono::duration<double>(now - t_mark).count();
    t_mark = now;
  };
  mark("upload");

  // The embedding rows, a sub-chunk at a time (the staging buffers hold max_batch rows), into the group's residual stream.
  for (const GroupSub& sb : subs) {
    float* xk = x_ + sb.off * E;
    if (c_.embed_on_card) {
      ref::embed(q_, embd_card_, tok_ + sb.off, sb.len, 1.0f, xk);
    } else {
      for (int64_t t = 0; t < sb.len; ++t)
        std::memcpy(emb_host_ + uint64_t(t) * embd_row_bytes_, embd_->data + uint64_t(tok[size_t(sb.off + t)]) * embd_row_bytes_, embd_row_bytes_);
      q_.memcpy(emb_dev_, emb_host_, size_t(uint64_t(sb.len) * embd_row_bytes_));
      ref::embed(q_, ref::Weight{embd_->type, E, sb.len, emb_dev_, embd_row_bytes_}, iota_, sb.len, 1.0f, xk);
      q_.wait();
    }
  }
  for (const model::ImageRows& im : c.images) q_.memcpy(x_ + im.offset * E, im.rows, size_t(im.n * E) * sizeof(float));
  mark("embed");
  const float attn_scale = 1.0f / std::sqrt(float(D));
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const Layer& L = layers_[size_t(l)];
    sycl::event ids_ready, routed;
    // With early copies a layer runs in two phases: every sub-chunk's mixer and routing first, the copies of the streamed set started
    // when the whole group is routed, and the hot experts and the shared expert after, while the copies land.
    const bool split = c_.group_early_copy && L.n_cold && n_stream_ > 0;
    auto phase_a = [&](int64_t k) {
      const GroupSub& sb = subs[size_t(k)];
      const int64_t T = sb.len, P = T * NU;
      float* xk = x_ + sb.off * E;
      int32_t* ids = gids_ + sb.off * NU;
      float* wts = gw_ + sb.off * NU;
      const gdn::Plan& plan = plans[size_t(k)];
      // The layer's input norm (the plain step gets it from the previous layer's tail).
      qwenk::add_norm_quant(q_, nullptr, xk, L.attn_norm, cfg_.eps, T, E, nullptr, &act_, L.recurrent ? h_ : nullptr);
      if (L.recurrent) {
        dense(act_, L.in, qz_);
        mark("qkvz");
        fast::matmul_f32(q_, L.w_ab, 2 * gdn::kValueHeads, E, h_, T, ab_);
        const gdn::Inputs in{qz_, qz_ + gdn::kChannels, ab_, ab_ + gdn::kValueHeads, L.in.n, L.in.n, 2 * gdn::kValueHeads, 2 * gdn::kValueHeads};
        mark("alpha_beta");
        gdn::step(q_, L.gw, gcache_, L.gdn_idx, plan, in, gscratch_, act_);
        mark("gdn");
        dense(act_, L.out, o_);
        mark("ssm_out");
      } else {
        dense(act_, L.in, qz_);
        mark("qkv");
        qwenk::attn_heads(q_, qz_, T, H, Hkv, D, cfg_.n_rot, L.q_norm, L.k_norm, cfg_.eps, rpos_ + 3 * sb.off, cfg_.rope_sections, inv_freq_,
                          qb_, kb_, vb_, gt_);
        mark("attn_heads");
        const KvLayer& kv = kv_[size_t(L.attn_idx)];
        const attn::Cache cache{kv.k, kv.v, page_table_, max_pages_, 0, Hkv, D, kv.ks, kv.vs};
        const attn::Tokens toks{slot_ + sb.off, pos_ + sb.off, T};
        attn::write(q_, kb_, vb_, toks, cache);
        attn::attend(q_, qb_, toks, tiles[size_t(k)], tiles_ + k * 6 * max_tiles_, cache, H, attn_scale, att_, attn_scratch_);
        mark("attn");
        qwenk::gate_quant(q_, att_, gt_, T, H * D, act_);
        dense(act_, L.out, o_);
        mark("attn_out");
      }
      // x = x + out, in place; m = rmsnorm(x) * post_attention_norm, fp32 in h_ and quantized in act_.
      qwenk::add_norm_quant(q_, o_, xk, L.post_norm, cfg_.eps, T, E, xk, &act_, h_);
      mark("post_attn");
      fast::matmul_f32(q_, L.w_rg, NE, E, h_, T, rlog_);
      fast::matmul_f32(q_, L.w_rg + NE * E, 32, E, h_, T, sg_);
      mark("router");
      fast::moe_route(q_, rlog_, T, NE, NU, nullptr, ids, wts);
      mark("moe_route");
      if (c_.record_routing) {
        q_.memcpy(rec_host_, ids, size_t(P) * 4).wait();
        for (int64_t p = 0; p < P; ++p) ++counts_[size_t(l)][size_t(rec_host_[p])];
      }
      if (L.n_cold) {
        // What the cold phase needs after the sub-chunk's own work: the routing and the CPU's int8 inputs on the host, the fp16 inputs
        // of the streamed waves on the card.
        ids_ready = q_.memcpy(hp_ids_ + sb.off * NU, ids, size_t(P) * 4);
        q_.memcpy(hp_w_ + sb.off * NU, wts, size_t(P) * 4);
        fast::quantize_rows(q_, h_, T, E, hq_, hqs_);
        q_.memcpy(hp_xq_ + sb.off * E, hq_, size_t(T * E));
        routed = q_.memcpy(hp_xs_ + sb.off * kb, hqs_, size_t(T * kb) * sizeof(float));
        if (n_stream_ > 0) qwenk::to_half(q_, h_, T * E, hg_ + sb.off * E);
      }
    };
    auto phase_b = [&](int64_t k) {
      const GroupSub& sb = subs[size_t(k)];
      const int64_t T = sb.len, P = T * NU;
      float* xk = x_ + sb.off * E;
      int32_t* ids = gids_ + sb.off * NU;
      float* wts = gw_ + sb.off * NU;
      if (split) {
        // Phase A left x = the residual after the mixer; the MoE input, its quantization and the shared gate are cheap to make again.
        qwenk::add_norm_quant(q_, nullptr, xk, L.post_norm, cfg_.eps, T, E, nullptr, &act_, h_);
        fast::matmul_f32(q_, L.w_rg + NE * E, 32, E, h_, T, sg_);
      }
      fast::moe_group(q_, ids, P, NE, moe_tile_, moe_scratch_, moe_tiles_, moe_row_, L.cold_dev);
      mark("moe_group");
      fast::moe_gather(q_, h_, moe_row_, P, NU, E, moe_a_);
      mark("moe_gather");
      const int64_t max_tiles = (P + moe_tile_ - 1) / moe_tile_ + NE;
      xmx::matmul_geglu_grouped16(q_, moe_a_, L.tab_gu, moe_tiles_, max_tiles, moe_tile_, moe_h_, xmx::Glu::Silu);
      mark("moe_gate_up");
      xmx::matmul_grouped16(q_, moe_h_, L.tab_dn, moe_tiles_, max_tiles, moe_tile_, moe_y_);
      mark("moe_down");
      xmx::matmul_geglu(q_, act_, L.shexp_gu, act_sh_, sh_gu_, xmx::Glu::Silu);
      xmx::matmul(q_, act_sh_, L.shexp_dn, o_);
      mark("shexp");
      qwenk::moe_accumulate(q_, xk, moe_y_, wts, moe_row_, NU, nullptr, nullptr, o_, sg_, 32, T, E);
      mark("moe_sum");
    };
    for (int64_t k = 0; k < G; ++k) {
      phase_a(k);
      if (!split) phase_b(k);
    }
    if (L.n_cold) cold_group(L, subs, ids_ready, routed, mark, split ? [&] { for (int64_t k = 0; k < G; ++k) phase_b(k); } : std::function<void()>());
    if (cap_col_[size_t(l)] >= 0 && !cap_rows.empty())
      qwenk::gather_rows(q_, x_, cap_idx_, int64_t(cap_rows.size()), E, cap_, int64_t(ncap) * E, int64_t(cap_col_[size_t(l)]) * E);
    mark("tail");
  }
  ++grp_.groups;

  if (R == 0) {
    q_.wait();
    settle_marks();
    return {};
  }
  qwenk::gather_rows(q_, x_, rows_idx_, R, E, rows_, E, 0);
  fast::rmsnorm(q_, rows_, output_norm_, R, E, cfg_.eps, rows_);
  if (c_.head_dpas) {
    xmx::quantize(q_, rows_, R, act_head_);
    xmx::matmul(q_, act_head_, head_w_, logits_);
  } else {
    fast::matmul_q6(q_, head_q6_, rows_, R, logits_);
  }
  mark("head");
  std::vector<float> out(size_t(R * V));
  q_.memcpy(out.data(), logits_, out.size() * sizeof(float)).wait();
  settle_marks();
  return out;
}

// The layer's cold experts for a whole group, after every sub-chunk's hot work is enqueued. The streamed set is chosen from the
// group's counts and goes through the slots in waves (the card runs a wave's GEMMs for every sub-chunk while the copy engine
// brings the next one in); the CPU computes the pairs of the experts not streamed on a thread of its own, meanwhile.
void Engine::cold_group(const Layer& L, const std::vector<GroupSub>& subs, sycl::event ids_ready, sycl::event routed,
                        const std::function<void(const char*)>& mark, const std::function<void()>& after_first_copy) {
  using Clock = std::chrono::steady_clock;
  const int64_t E = cfg_.n_embd, Fe = cfg_.n_ff_exp, NU = cfg_.n_expert_used, NE = cfg_.n_expert, kb = E / 32;
  const int64_t G = int64_t(subs.size()), Tg = subs.back().off + subs.back().len, Pg = Tg * NU;
  auto since = [](Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); };
  const auto t0 = Clock::now();
  ids_ready.wait();  // the queue is in order: the earlier sub-chunks' routing has landed too
  grp_.ids_wait_s += since(t0);
  cold_.wait_s += since(t0);
  ++grp_.layers;
  ++cold_.layers;

  std::vector<std::vector<int32_t>> cnt(size_t(G), std::vector<int32_t>(size_t(NE), 0));
  std::vector<int32_t> tot(size_t(NE), 0);
  for (int64_t k = 0; k < G; ++k)
    for (int64_t p = subs[size_t(k)].off * NU; p < (subs[size_t(k)].off + subs[size_t(k)].len) * NU; ++p) ++cnt[size_t(k)][size_t(hp_ids_[p])];
  for (int64_t k = 0; k < G; ++k)
    for (int e = 0; e < NE; ++e) tot[size_t(e)] += cnt[size_t(k)][size_t(e)];
  std::vector<int32_t> list;
  if (n_stream_ > 0) list = choose_stream(L, tot, true);

  // Waves: up to n_stream_ experts, and no sub-chunk's gathered rows past what att_ holds.
  struct Wave {
    std::vector<int32_t> experts;
    std::vector<int64_t> tiles;  // per sub-chunk
  };
  std::vector<Wave> waves;
  {
    std::vector<int64_t> rows(size_t(G), 0);
    Wave cur;
    cur.tiles.assign(size_t(G), 0);
    for (int32_t e : list) {
      bool fit = int64_t(cur.experts.size()) < n_stream_;
      for (int64_t k = 0; k < G && fit; ++k) {
        const int64_t n = cnt[size_t(k)][size_t(e)];
        fit = rows[size_t(k)] + (n + moe_tile_ - 1) / moe_tile_ * moe_tile_ <= stream_rows_cap_;
      }
      if (!fit && !cur.experts.empty()) {
        waves.push_back(std::move(cur));
        cur = Wave{};
        cur.tiles.assign(size_t(G), 0);
        std::fill(rows.begin(), rows.end(), 0);
      }
      for (int64_t k = 0; k < G; ++k) {
        const int64_t n = cnt[size_t(k)][size_t(e)];
        cur.tiles[size_t(k)] += (n + moe_tile_ - 1) / moe_tile_;
        rows[size_t(k)] += (n + moe_tile_ - 1) / moe_tile_ * moe_tile_;
      }
      cur.experts.push_back(e);
    }
    if (!cur.experts.empty()) waves.push_back(std::move(cur));
  }
  std::vector<uint8_t> streamed(size_t(NE), 0);
  uint64_t n_streamed_pairs = 0;
  for (int32_t e : list) {
    streamed[size_t(e)] = 1;
    n_streamed_pairs += uint64_t(tot[size_t(e)]);
  }

  sycl::event copied;
  auto issue = [&](size_t w) {
    const auto& ex = waves[w].experts;
    for (size_t j = 0; j < ex.size(); ++j) copy_->copy(stream_raw_ + uint64_t(j) * raw_expert_, L.cold_gu[size_t(ex[j])], raw_expert_);
    copied = copy_->event(copy_->fence());
  };
  if (!waves.empty()) issue(0);
  if (after_first_copy) after_first_copy();  // the hot experts of every sub-chunk, enqueued while the first wave lands
  routed.wait();

  // The CPU's pairs: the cold experts not streamed.
  std::vector<int32_t> cp;
  for (int64_t p = 0; p < Pg; ++p)
    if (L.cold[size_t(hp_ids_[p])] && !streamed[size_t(hp_ids_[p])]) cp.push_back(int32_t(p));
  pairs_ += uint64_t(Pg);
  cold_pairs_ += cp.size() + n_streamed_pairs;
  grp_.cpu_pairs += cp.size();
  grp_.streamed_pairs += n_streamed_pairs;
  grp_.experts += list.size();
  grp_.waves += waves.size();
  if (!list.empty()) {
    ++stream_.layers;
    stream_.experts += list.size();
    stream_.pairs += n_streamed_pairs;
  }
  std::vector<int32_t> k_of(size_t(Pg), -1), dense_row(size_t(Tg), -1);
  std::vector<char> need(size_t(Tg), 0);
  for (size_t k = 0; k < cp.size(); ++k) {
    k_of[size_t(cp[k])] = int32_t(k);
    need[size_t(cp[k] / NU)] = 1;
  }
  std::vector<int64_t> base(size_t(G), 0), n_need(size_t(G), 0);
  {
    int32_t dense = 0;
    for (int64_t k = 0; k < G; ++k) {
      base[size_t(k)] = dense;
      for (int64_t t = subs[size_t(k)].off; t < subs[size_t(k)].off + subs[size_t(k)].len; ++t) {
        hp_map_[t] = need[size_t(t)] ? dense - int32_t(base[size_t(k)]) : -1;  // the row within the sub-chunk's share
        if (need[size_t(t)]) dense_row[size_t(t)] = dense++;
      }
      n_need[size_t(k)] = dense - base[size_t(k)];
    }
  }
  auto cpu_task = [&]() -> double {
    const auto c0 = Clock::now();
    std::vector<cpu::Q8Blocks> x(static_cast<size_t>(Tg));
    pool_->for_each(Tg, 8, [&](int64_t lo, int64_t hi) {
      for (int64_t t = lo; t < hi; ++t) {
        if (!need[size_t(t)]) continue;
        cpu::Q8Blocks& r = x[size_t(t)];
        r.q.assign(hp_xq_ + t * E, hp_xq_ + (t + 1) * E);
        r.s.assign(hp_xs_ + t * kb, hp_xs_ + (t + 1) * kb);
        r.sum.resize(size_t(kb));
        for (int64_t b = 0; b < kb; ++b) {
          int32_t sum = 0;
          for (int i = 0; i < 32; ++i) sum += r.q[size_t(b * 32 + i)];
          r.sum[size_t(b)] = sum;
        }
      }
    });
    cold_y_.resize(cp.size() * size_t(E));
    cold_compute(L, cp, hp_ids_, x, cold_y_.data());
    // Each token's cold share, weighted and summed in slot order.
    pool_->for_each(Tg, 16, [&](int64_t lo, int64_t hi) {
      for (int64_t t = lo; t < hi; ++t) {
        if (!need[size_t(t)]) continue;
        float* out = hp_sum_ + int64_t(dense_row[size_t(t)]) * E;
        bool first = true;
        for (int64_t i = 0; i < NU; ++i) {
          const int32_t k = k_of[size_t(t * NU + i)];
          if (k < 0) continue;
          const float w = hp_w_[t * NU + i];
          const float* yk = cold_y_.data() + int64_t(k) * E;
          if (first)
            for (int64_t j = 0; j < E; ++j) out[j] = w * yk[j];
          else
            for (int64_t j = 0; j < E; ++j) out[j] += w * yk[j];
          first = false;
        }
      }
    });
    return since(c0);
  };
  std::future<double> cpu_done;
  if (!cp.empty()) cpu_done = std::async(std::launch::async, cpu_task);

  auto repack_into = [&](const ref::Weight& raw, bool interleave, const xmx::Weight& w) {
    if (w.format == xmx::WeightFormat::Q4_0) xmx::repack_q4_0_into(q_, raw, interleave, w);
    else xmx::repack_format_into(q_, raw, interleave, w);
  };
  for (size_t w = 0; w < waves.size(); ++w) {
    const Wave& wv = waves[w];
    q_.submit([&](sycl::handler& h) {
      h.depends_on(copied);
      h.single_task([] {});
    });
    for (size_t j = 0; j < wv.experts.size(); ++j) {
      const uint8_t* raw = stream_raw_ + uint64_t(j) * raw_expert_;
      repack_into(ref::Weight{t_gu_, E, 2 * Fe, raw, gu_row_}, true, stream_w_gu_[j]);
      repack_into(ref::Weight{t_dn_, Fe, E, raw + raw_gu_, dn_row_}, false, stream_w_dn_[j]);
    }
    int32_t* smap = hp_smap_ + int64_t(w) * NE;
    for (int e = 0; e < NE; ++e) smap[e] = -1;
    for (size_t j = 0; j < wv.experts.size(); ++j) smap[wv.experts[j]] = int32_t(j);
    q_.memcpy(stream_map_, smap, size_t(NE) * 4);
    sycl::event repacked = q_.single_task([] {});
    mark("stream_copy_repack");
    for (int64_t k = 0; k < G; ++k) {
      if (!wv.tiles[size_t(k)]) continue;
      const GroupSub& sb = subs[size_t(k)];
      const int64_t P = sb.len * NU;
      int32_t* ids = gids_ + sb.off * NU;
      q_.fill(moe_row_, int32_t(-1), size_t(P));
      fast::moe_group(q_, ids, P, n_stream_, moe_tile_, stream_scratch_, stream_tiles_, moe_row_, nullptr, stream_map_, 0);
      sycl::half* a = reinterpret_cast<sycl::half*>(att_);
      fast::moe_gather(q_, hg_ + sb.off * E, moe_row_, P, NU, E, a);
      xmx::matmul_geglu_grouped16(q_, a, stream_gu_, stream_tiles_, wv.tiles[size_t(k)], moe_tile_, moe_h_, xmx::Glu::Silu);
      xmx::matmul_grouped16(q_, moe_h_, stream_dn_, stream_tiles_, wv.tiles[size_t(k)], moe_tile_, moe_y_);
      qwenk::moe_accumulate(q_, x_ + sb.off * E, moe_y_, gw_ + sb.off * NU, moe_row_, NU, nullptr, nullptr, nullptr, nullptr, 0, sb.len, E);
    }
    mark("stream_gemm");
    if (w + 1 < waves.size()) {
      // The next wave's copy may overwrite the raw slots once this wave's repack has read them; it then runs beside the GEMMs.
      const auto tw = Clock::now();
      repacked.wait();
      grp_.copy_wait_s += since(tw);
      issue(w + 1);
    }
  }
  if (cpu_done.valid()) {
    const auto tj = Clock::now();
    grp_.cpu_s += cpu_done.get();
    grp_.join_s += since(tj);
    cold_.cpu_s += since(tj);
    for (int64_t k = 0; k < G; ++k) {
      if (!n_need[size_t(k)]) continue;
      const GroupSub& sb = subs[size_t(k)];
      q_.memcpy(cold_sum_, hp_sum_ + base[size_t(k)] * E, size_t(n_need[size_t(k)] * E) * sizeof(float));
      q_.memcpy(cold_map_, hp_map_ + sb.off, size_t(sb.len) * 4);
      qwenk::moe_accumulate(q_, x_ + sb.off * E, nullptr, nullptr, nullptr, NU, cold_sum_, cold_map_, nullptr, nullptr, 0, sb.len, E);
    }
  }
  mark("cold_group");
}

}  // namespace eklow::engine
