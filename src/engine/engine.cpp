#include "engine/engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <iterator>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "core/json_value.hpp"
#include "kernels/fast.hpp"
#include "rt/device.hpp"

namespace eklow::engine {

float* Engine::dev(int64_t n) {
  auto* p = sycl::malloc_device<float>(size_t(n), q_);
  if (!p) throw std::runtime_error("engine: out of device memory");
  owned_.push_back(p);
  return p;
}

Engine::Engine(sycl::queue& q, const gguf::File& f, const Config& cfg) : q_(q), c_(cfg), cfg_(model::Gemma4Config::from_gguf(f)) {
  if (c_.max_ctx % attn::kPage) throw std::runtime_error("engine: max_ctx must be a multiple of the page size");
  const int64_t E = cfg_.n_embd, F = cfg_.n_ff, H = cfg_.n_head;
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
    auto* d = sycl::malloc_device<float>(v.size(), q_);
    if (!d) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(d);
    q_.memcpy(d, v.data(), v.size() * sizeof(float)).wait();
    return d;
  };

  // Q4_0 matrices go through one staging buffer and are repacked for the matrix units. Matrices
  // that read the same input are stacked into one (their outputs side by side in each row), so a
  // decode step launches one matmul for Q, K and V and one for gate and up.
  uint64_t staging_bytes = 0;
  for (int l = 0; l < cfg_.n_layer; ++l)
    for (const char* group : {"attn_q.weight|attn_k.weight|attn_v.weight", "ffn_gate.weight|ffn_up.weight"}) {
      uint64_t sum = 0;
      std::stringstream names(group);
      for (std::string n; std::getline(names, n, '|');)
        if (const gguf::Tensor* t = f.tensor("blk." + std::to_string(l) + "." + n)) sum += t->nbytes;
      staging_bytes = std::max(staging_bytes, sum);
    }
  for (const auto& t : f.tensors())
    if (t.type == gguf::Type::Q4_0 && t.name.find("_exps") == std::string::npos) staging_bytes = std::max(staging_bytes, t.nbytes);
  auto* staging = sycl::malloc_device<uint8_t>(staging_bytes, q_);
  if (!staging) throw std::runtime_error("engine: out of device memory for weight staging");
  auto matrix = [&](const std::vector<const gguf::Tensor*>& parts, bool interleave = false) {
    uint64_t at = 0;
    int64_t n = 0;
    for (const gguf::Tensor* t : parts) {
      if (t->type != gguf::Type::Q4_0) throw std::runtime_error(f.path() + ": " + t->name + " is not Q4_0");
      if (t->ne[0] != parts[0]->ne[0]) throw std::runtime_error(f.path() + ": stacked matrices of different widths");
      q_.memcpy(staging + at, t->data, t->nbytes);
      at += t->nbytes;
      n += int64_t(t->ne[1]);
      weight_bytes_ += t->nbytes;
    }
    q_.wait();
    const ref::Weight raw{gguf::Type::Q4_0, int64_t(parts[0]->ne[0]), n, staging, parts[0]->nbytes / parts[0]->ne[1]};
    xmx::Weight w = xmx::repack_q4_0(q_, raw, interleave);
    owned_.push_back(w.tiles);
    owned_.push_back(w.scales);
    return w;
  };

  {
    const gguf::Tensor& emb = need("token_embd.weight");
    if (emb.type != gguf::Type::Q6_K) throw std::runtime_error(f.path() + ": token_embd.weight is not Q6_K");
    auto* raw = sycl::malloc_device<uint8_t>(emb.nbytes, q_);
    if (!raw) throw std::runtime_error("engine: out of device memory loading token_embd.weight");
    q_.memcpy(raw, emb.data, emb.nbytes).wait();
    head_ = fast::repack_q6k(q_, ref::Weight{emb.type, int64_t(emb.ne[0]), int64_t(emb.ne[1]), raw, emb.nbytes / emb.ne[1]});
    sycl::free(raw, q_);
    owned_.push_back(head_.lo);
    weight_bytes_ += emb.nbytes;
  }
  output_norm_ = vec("output_norm.weight");

  // RoPE inverse frequencies, computed in double and stored as fp32.
  const gguf::Tensor& rf = need("rope_freqs.weight");
  const float* freqs = reinterpret_cast<const float*>(rf.data);
  std::vector<float> inv_s(size_t(cfg_.head_dim_sliding / 2)), inv_g(size_t(cfg_.head_dim_global / 2));
  for (size_t i = 0; i < inv_s.size(); ++i)
    inv_s[i] = float(std::pow(double(cfg_.rope_base_sliding), -2.0 * double(i) / cfg_.head_dim_sliding));
  for (size_t i = 0; i < inv_g.size(); ++i)
    inv_g[i] = float(std::pow(double(cfg_.rope_base_global), -2.0 * double(i) / cfg_.head_dim_global) / double(freqs[i]));
  float* is = dev(int64_t(inv_s.size()));
  float* ig = dev(int64_t(inv_g.size()));
  q_.memcpy(is, inv_s.data(), inv_s.size() * sizeof(float));
  q_.memcpy(ig, inv_g.data(), inv_g.size() * sizeof(float)).wait();
  inv_freq_sliding_ = is;
  inv_freq_global_ = ig;

  int n_ring = 0, n_page = 0;
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L{};
    L.attn_norm = vec(b + "attn_norm.weight");
    L.has_v = f.tensor(b + "attn_v.weight") != nullptr;
    L.wqkv = L.has_v ? matrix({&need(b + "attn_q.weight"), &need(b + "attn_k.weight"), &need(b + "attn_v.weight")})
                     : matrix({&need(b + "attn_q.weight"), &need(b + "attn_k.weight")});
    L.wo = matrix({&need(b + "attn_output.weight")});
    L.q_norm = vec(b + "attn_q_norm.weight");
    L.k_norm = vec(b + "attn_k_norm.weight");
    L.attn_post_norm = vec(b + "post_attention_norm.weight");
    L.ffn_norm = vec(b + "ffn_norm.weight");
    L.gate_up = matrix({&need(b + "ffn_gate.weight"), &need(b + "ffn_up.weight")}, true);
    L.down = matrix({&need(b + "ffn_down.weight")});
    L.ffn_post_norm = vec(b + "post_ffw_norm.weight");
    const gguf::Tensor* os = f.tensor(b + "layer_output_scale.weight");
    L.out_scale = os ? *reinterpret_cast<const float*>(os->data) : 1.0f;
    if (cfg_.n_expert > 0) {
      const int64_t n_expert = cfg_.n_expert;
      const gguf::Tensor& gu_t = need(b + "ffn_gate_up_exps.weight");
      const gguf::Tensor& dn_t = need(b + "ffn_down_exps.weight");
      const gguf::Tensor& dn_s = need(b + "ffn_down_exps.scale");
      L.w_router = ref::Weight{gguf::Type::F32, E, n_expert, upload(need(b + "ffn_gate_inp.weight")), uint64_t(E) * 4};
      {
        // The router input is rmsnorm(x) * ffn_gate_inp.scale / sqrt(E); fold the scalar in.
        const gguf::Tensor& rs = need(b + "ffn_gate_inp.scale");
        std::vector<float> host;
        host.resize(size_t(E));
        const float* src = reinterpret_cast<const float*>(rs.data);
        const float inv_se = 1.0f / std::sqrt(float(E));
        for (int64_t i = 0; i < E; ++i) host[size_t(i)] = src[i] * inv_se;
        L.router_in_scale = vec_host(host);
      }
      L.post_ffw_norm_1 = vec(b + "post_ffw_norm_1.weight");
      L.pre_ffw_norm_2 = vec(b + "pre_ffw_norm_2.weight");
      L.post_ffw_norm_2 = vec(b + "post_ffw_norm_2.weight");
      L.exp_down_scale.assign(reinterpret_cast<const float*>(dn_s.data),
                              reinterpret_cast<const float*>(dn_s.data) + n_expert);
      L.exp_down_scale_dev = vec_host(L.exp_down_scale);
      L.gu_host = gu_t.data;
      L.dn_host = dn_t.data;
    }
    if (cfg_.sliding[size_t(l)]) L.ring_index = n_ring++;
    else L.page_index = n_page++;
    layers_.push_back(L);
  }
  sycl::free(staging, q_);

  // KV cache: rings for the sliding layers, a page pool for the global ones.
  for (int l = 0; l < cfg_.n_layer; ++l) {
    int64_t& hkv = cfg_.sliding[size_t(l)] ? hkv_sliding_ : hkv_global_;
    if (hkv && hkv != cfg_.n_head_kv[size_t(l)]) throw std::runtime_error("engine: K/V heads differ between layers of a kind");
    hkv = cfg_.n_head_kv[size_t(l)];
  }
  group_sliding_ = H / hkv_sliding_;
  group_global_ = H / hkv_global_;
  ring_pages_ = (cfg_.window + c_.max_chunk + attn::kPage - 1) / attn::kPage;
  ring_layer_stride_ = int64_t(c_.n_slots) * ring_pages_ * hkv_sliding_ * attn::kPage * cfg_.head_dim_sliding;
  const size_t ring_elems = size_t(ring_layer_stride_ * n_ring);
  ring_k_ = sycl::malloc_device<sycl::half>(ring_elems, q_);
  ring_v_ = sycl::malloc_device<sycl::half>(ring_elems, q_);
  max_pages_ = c_.max_ctx / attn::kPage;
  n_pages_ = max_pages_ * c_.n_slots;
  // The global layers' pages, one allocation per layer and array: at 2 x 262k an fp16 layer is
  // 537 MB, and Level Zero refuses single allocations above 4 GB.
  const size_t page_elems = size_t(n_pages_ * hkv_global_ * attn::kPage * cfg_.head_dim_global);
  const size_t elem = c_.int8_global_kv ? 1 : 2;
  page_table_ = sycl::malloc_device<int32_t>(size_t(c_.n_slots * max_pages_), q_);
  if (!ring_k_ || !ring_v_ || !page_table_) throw std::runtime_error("engine: out of device memory for the KV cache");
  for (void* p : {static_cast<void*>(ring_k_), static_cast<void*>(ring_v_), static_cast<void*>(page_table_)}) owned_.push_back(p);
  cache_bytes_ = ring_elems * 2 * sizeof(sycl::half);
  for (int l = 0; l < n_page; ++l) {
    GlobalCache g;
    g.k = sycl::malloc_device(page_elems * elem, q_);
    g.v = sycl::malloc_device(page_elems * elem, q_);
    if (c_.int8_global_kv) {
      g.ks = sycl::malloc_device<sycl::half>(page_elems / 16, q_);
      g.vs = sycl::malloc_device<sycl::half>(page_elems / 16, q_);
    }
    if (!g.k || !g.v || (c_.int8_global_kv && (!g.ks || !g.vs))) throw std::runtime_error("engine: out of device memory for the KV cache");
    for (void* p : {g.k, g.v, static_cast<void*>(g.ks), static_cast<void*>(g.vs)})
      if (p) owned_.push_back(p);
    cache_bytes_ += 2 * page_elems * elem + (c_.int8_global_kv ? 2 * (page_elems / 16) * sizeof(sycl::half) : 0);
    global_.push_back(g);
  }
  page_table_host_.assign(size_t(c_.n_slots * max_pages_), 0);
  for (int64_t p = n_pages_ - 1; p >= 0; --p) free_pages_.push_back(int32_t(p));
  slots_.resize(size_t(c_.n_slots));

  // Workspace.
  const int64_t B = c_.max_batch, Dg = cfg_.head_dim_global;
  int64_t kv_max = 0;
  for (int l = 0; l < cfg_.n_layer; ++l) kv_max = std::max<int64_t>(kv_max, cfg_.n_head_kv[size_t(l)] * cfg_.head_dim(l));
  x_ = dev(B * E);
  x2_ = dev(B * E);
  h_ = dev(B * E);
  o_ = dev(B * E);
  qb_ = dev(B * H * Dg);
  att_ = dev(B * H * Dg);
  kb_ = dev(B * kv_max);
  vb_ = dev(B * kv_max);
  int64_t qkv_max = 0;
  for (const Layer& L : layers_) qkv_max = std::max(qkv_max, L.wqkv.n);
  qkv_ = dev(B * qkv_max);
  gu_ = dev(std::min(B, xmx::kFp16Rows) * 2 * F);  // the GeGLU's int8 path only
  rows_ = dev(c_.max_logits * E);
  logits_ = dev(c_.max_logits * cfg_.n_vocab);
  // At most one chunk per slot in a step, so at most n_slots partial tiles per K/V head.
  max_tiles_sliding_ = hkv_sliding_ * ((B * group_sliding_ + 7) / 8 + c_.n_slots);
  max_tiles_global_ = hkv_global_ * ((B * group_global_ + 7) / 8 + c_.n_slots);
  // Tile lists packed with their groups: 4 ints a tile, at most 2 a tile for its group.
  for (auto [p, n] : {std::pair{&tok_, B}, {&slot_, B}, {&pos_, B}, {&tiles_sliding_, 6 * max_tiles_sliding_},
                      {&tiles_global_, 6 * max_tiles_global_}}) {
    *p = sycl::malloc_device<int32_t>(size_t(n), q_);
    if (!*p) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(*p);
  }
  // The attention scratch at its largest, before the experts take what is left of the card.
  attn::reserve(q_, attn_scratch_, max_tiles_sliding_, c_.max_chunk * group_sliding_, hkv_sliding_, cfg_.head_dim_sliding);
  attn::reserve(q_, attn_scratch_, max_tiles_global_, c_.max_chunk * group_global_, hkv_global_, cfg_.head_dim_global);
  act_ = xmx::alloc_activations(q_, B, std::max<int64_t>(E, H * Dg));
  act_ffn_ = xmx::alloc_activations(q_, B, F);
  if (cfg_.n_expert > 0) {
    const int64_t NE = cfg_.n_expert, NU = cfg_.n_expert_used, P = B * NU, Fe = cfg_.n_ff_exp;
    moe_ = dev(B * E);
    rlog_ = dev(B * NE);
    gx_ = dev(xmx::kFp16Rows * E);  // the decode path's expert outputs (at most kFp16Rows pairs)
    // Per pair: the chosen expert and weight (decode keeps at most kFp16Rows pairs in them).
    moe_ids_ = sycl::malloc_device<int32_t>(size_t(P), q_);
    moe_row_ = sycl::malloc_device<int32_t>(size_t(P), q_);
    if (!moe_ids_ || !moe_row_) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(moe_ids_);
    owned_.push_back(moe_row_);
    moe_w_ = dev(P);
    moe_gu_ = dev((xmx::kFp16Rows + 12) * 2 * Fe);  // up to 32 expert pairs and 3 dense per row of 4
    act_exp_ = xmx::alloc_activations(q_, xmx::kFp16Rows, E);
    act_exp_ffn_ = xmx::alloc_activations(q_, xmx::kFp16Rows, Fe);
    // Prefill: segments padded to row tiles, so at most P / tile + n_expert tiles. The tile and
    // gate_up outputs are sized for that; the fp32 down outputs (the biggest) take the memory the
    // fp16 inputs used.
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
  if (cfg_.n_expert > 0) load_experts();
}

void Engine::load_experts() {
  const int64_t E = cfg_.n_embd, Fe = cfg_.n_ff_exp, NE = cfg_.n_expert;
  const uint64_t gu_row = uint64_t(E) / 32 * 18, dn_row = uint64_t(Fe) / 32 * 18;
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
  } else if (want < NE) {
    throw std::runtime_error("engine: experts_on_card below the expert count needs an expert profile");
  }
  // One expert of one layer is gate_up tiles, gate_up scales, down tiles, down scales, back to back (each a
  // multiple of 128 bytes, so every piece stays aligned): the same bytes as the GGUF's two Q4_0 matrices. A layer's
  // hot experts share one allocation (a slab), which also saves the rounding four allocations an expert paid
  // (5% of the experts' bytes, 3a).
  const uint64_t gu_tb = xmx::weight_tile_bytes(E, 2 * Fe), gu_sb = xmx::weight_scale_bytes(E, 2 * Fe),
                 dn_tb = xmx::weight_tile_bytes(Fe, E), dn_sb = xmx::weight_scale_bytes(Fe, E);
  expert_bytes_ = gu_tb + gu_sb + dn_tb + dn_sb;
  expert_raw_gu_ = uint64_t(2 * Fe) * gu_row;
  if (expert_bytes_ != expert_raw_gu_ + uint64_t(E) * dn_row) throw std::runtime_error("engine: repacked expert size differs from the GGUF's");
  expert_set_bytes_ = uint64_t(cfg_.n_layer) * expert_bytes_;
  auto expert_at = [&](uint8_t* base, xmx::Weight& gu, xmx::Weight& dn) {
    gu = xmx::Weight{E, 2 * Fe, base, reinterpret_cast<sycl::half*>(base + gu_tb)};
    dn = xmx::Weight{Fe, E, base + gu_tb + gu_sb, reinterpret_cast<sycl::half*>(base + gu_tb + gu_sb + dn_tb)};
  };

  // How many experts go on the card. When fitting, the driver's free memory less the margin is spent on the
  // streaming slots (when there are cold experts to stream) and then on whole ranks of experts.
  std::optional<uint64_t> fm;
  int64_t n_hot = want;
  const int64_t slots_wanted = std::min<int64_t>(std::max(c_.stream_slots, 0), 256);
  if (fit) {
    fm = rt::free_memory(q_.get_device());
    if (!fm) throw std::runtime_error("engine: the driver does not report free device memory; give experts_on_card");
    free_at_placement_ = *fm;
    const uint64_t budget = *fm > c_.card_margin_bytes ? *fm - c_.card_margin_bytes : 0;
    if (budget >= uint64_t(want) * expert_set_bytes_) {
      n_hot = want;
    } else {
      const uint64_t slot_bytes = 2 * uint64_t(slots_wanted) * expert_bytes_;
      n_hot = budget > slot_bytes ? int64_t((budget - slot_bytes) / expert_set_bytes_) : 0;
    }
  }
  n_hot_ = n_hot;
  if (n_hot_ < NE && c_.expert_profile.empty()) throw std::runtime_error("engine: the experts do not all fit; an expert profile is needed to place some on the CPU");

  const uint64_t stage_bytes = expert_bytes_;
  auto* staging = sycl::malloc_device<uint8_t>(stage_bytes, q_);
  if (!staging) throw std::runtime_error("engine: out of device memory for weight staging");
  for (Layer& L : layers_) {
    L.cold.assign(size_t(NE), 1);
    L.exp_gate_up.assign(size_t(NE), xmx::Weight{E, 2 * Fe, nullptr, nullptr});  // cold: a table entry with no tiles
    L.exp_down.assign(size_t(NE), xmx::Weight{Fe, E, nullptr, nullptr});
    L.n_cold = int(NE - n_hot_);
  }
  // Experts go on the card one rank at a time, that rank of every layer, so that the layers stay alike.
  std::vector<uint8_t*> slab(layers_.size(), nullptr);
  if (n_hot_ > 0)
    for (size_t l = 0; l < layers_.size(); ++l) {
      slab[l] = sycl::malloc_device<uint8_t>(size_t(uint64_t(n_hot_) * expert_bytes_), q_);
      if (!slab[l]) throw std::runtime_error("engine: out of device memory for the experts");
      owned_.push_back(slab[l]);
    }
  for (int64_t rank = 0; rank < n_hot_; ++rank) {
    for (int l = 0; l < cfg_.n_layer; ++l) {
      Layer& L = layers_[size_t(l)];
      const int e = order[size_t(l)][size_t(rank)];
      L.cold[size_t(e)] = 0;
      // The expert's two matrices go to the staging buffer and are repacked into its place in the slab, so the
      // originals are not kept on the card.
      q_.memcpy(staging, L.gu_host + size_t(e) * expert_raw_gu_, size_t(expert_raw_gu_));
      q_.memcpy(staging + expert_raw_gu_, L.dn_host + size_t(e) * E * dn_row, size_t(E * dn_row));
      expert_at(slab[size_t(l)] + uint64_t(rank) * expert_bytes_, L.exp_gate_up[size_t(e)], L.exp_down[size_t(e)]);
      xmx::repack_q4_0_into(q_, ref::Weight{gguf::Type::Q4_0, E, 2 * Fe, staging, gu_row}, true, L.exp_gate_up[size_t(e)]);
      xmx::repack_q4_0_into(q_, ref::Weight{gguf::Type::Q4_0, Fe, E, staging + expert_raw_gu_, dn_row}, false, L.exp_down[size_t(e)]);
      q_.wait();
      weight_bytes_ += expert_bytes_;
    }
  }
  for (Layer& L : layers_) {
    // Device tables of the experts' pointers for the grouped decode kernels.
    {
      // The dense FFN's gate|up as three more entries (its 3 x 1,408 interleaved columns are contiguous tile ranges).
      std::vector<xmx::Weight> with_dense = L.exp_gate_up;
      const int64_t third = L.gate_up.n / 3 / 8 * (L.gate_up.k / 32);  // tiles per third
      for (int j = 0; j < 3; ++j)
        with_dense.push_back(xmx::Weight{L.gate_up.k, L.gate_up.n / 3, L.gate_up.tiles + j * third * 128, L.gate_up.scales + j * third * 8});
      L.tab_gate_up = xmx::make_expert_table(q_, with_dense);
    }
    L.tab_down = xmx::make_expert_table(q_, L.exp_down);
    for (void* p : {static_cast<void*>(L.tab_gate_up.tiles), static_cast<void*>(L.tab_gate_up.scales), static_cast<void*>(L.tab_down.tiles),
                    static_cast<void*>(L.tab_down.scales)})
      owned_.push_back(p);
    if (L.n_cold) {
      L.cold_dev = sycl::malloc_device<uint8_t>(size_t(NE), q_);
      if (!L.cold_dev) throw std::runtime_error("engine: out of device memory");
      owned_.push_back(L.cold_dev);
      q_.memcpy(L.cold_dev, L.cold.data(), size_t(NE)).wait();
    }
  }
  // Streaming slots on the card: a raw one (the bytes as the copy engine lands them) and a repacked one (what the
  // grouped GEMMs read) each.
  n_stream_ = int(std::min<int64_t>(slots_wanted, NE - n_hot_));
  if (n_hot_ < NE && n_stream_ > 0) {
    try {
      copy_ = std::make_unique<rt::CopyEngine>(q_);
    } catch (const std::exception& ex) {
      std::fprintf(stderr, "engine: no streaming of cold experts (%s)\n", ex.what());
      n_stream_ = 0;
    }
  }
  sycl::free(staging, q_);
  if (n_hot_ == NE) return;

  if (n_stream_ > 0) {
    const int64_t S = n_stream_, B = c_.max_batch, P = B * cfg_.n_expert_used;
    stream_raw_ = sycl::malloc_device<uint8_t>(size_t(uint64_t(S) * expert_bytes_), q_);
    stream_slab_ = sycl::malloc_device<uint8_t>(size_t(uint64_t(S) * expert_bytes_), q_);
    if (!stream_raw_ || !stream_slab_) throw std::runtime_error("engine: out of device memory for the streaming slots");
    owned_.push_back(stream_raw_);
    owned_.push_back(stream_slab_);
    stream_w_gu_.resize(size_t(S));
    stream_w_dn_.resize(size_t(S));
    for (int64_t j = 0; j < S; ++j) expert_at(stream_slab_ + uint64_t(j) * expert_bytes_, stream_w_gu_[size_t(j)], stream_w_dn_[size_t(j)]);
    stream_gu_ = xmx::make_expert_table(q_, stream_w_gu_);
    stream_dn_ = xmx::make_expert_table(q_, stream_w_dn_);
    for (void* p : {static_cast<void*>(stream_gu_.tiles), static_cast<void*>(stream_gu_.scales), static_cast<void*>(stream_dn_.tiles),
                    static_cast<void*>(stream_dn_.scales)})
      owned_.push_back(p);
    stream_map_ = sycl::malloc_device<int32_t>(size_t(NE), q_);
    stream_tiles_ = sycl::malloc_device<int32_t>(size_t(4 + 4 * ((P + moe_tile_ - 1) / moe_tile_ + S)), q_);
    stream_scratch_ = sycl::malloc_device<int32_t>(size_t((P / 256 + 2) * S), q_);
    if (!stream_map_ || !stream_tiles_ || !stream_scratch_) throw std::runtime_error("engine: out of device memory for the streaming slots");
    for (void* p : {static_cast<void*>(stream_map_), static_cast<void*>(stream_tiles_), static_cast<void*>(stream_scratch_)}) owned_.push_back(p);
    // The streamed pairs' inputs (fp16, B x H x Dg fp32's worth) go in att_, which holds nothing during the MoE.
    stream_rows_cap_ = std::min<int64_t>(B * cfg_.n_head * cfg_.head_dim_global * 4 / (E * 2), (P / moe_tile_ + S) * moe_tile_) / moe_tile_ * moe_tile_;
  }

  // The host side: a pool, buffers the card and the host share, and the cold experts' pages
  // touched now (they come out of the page cache the first time, which a request should not wait for).
  const int threads = c_.cpu_threads > 0 ? c_.cpu_threads : 16;
  pool_ = std::make_unique<cpu::Pool>(threads);
  const int64_t B = std::max<int64_t>(c_.max_batch, xmx::kFp16Rows), P = B * cfg_.n_expert_used;
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
  if (n_stream_ > 0) host(hp_smap_, NE);
  cold_map_ = reinterpret_cast<int32_t*>(dev(B));
  // cold_sum_, the quantized inputs and their scales live in workspace that is dead during the MoE
  // (qkv_, qb_, kb_: B x 2,816 fp32, B x 2,816 fp32 and B x 88 fp32 fit them), see step().
  cold_sum_ = qkv_;
  // Where the CPU reads the cold experts from. With streaming they are copied (once, in parallel) into pinned host
  // memory, from which the copy engine moves them to the card without the CPU's help (from the GGUF's mapping the
  // driver stages the bytes through the CPU, 530 us an expert while the cold GEMVs run, against 300). The copy also
  // brings the pages in, which a request should not wait for.
  for (Layer& L : layers_) {
    L.cold_gu.assign(size_t(NE), nullptr);
    L.cold_dn.assign(size_t(NE), nullptr);
    std::vector<int> cold;
    for (int e = 0; e < NE; ++e)
      if (L.cold[size_t(e)]) cold.push_back(e);
    uint8_t* blk = nullptr;
    if (n_stream_ > 0) blk = sycl::malloc_host<uint8_t>(size_t(uint64_t(cold.size()) * expert_bytes_), q_);
    if (n_stream_ > 0 && !blk) throw std::runtime_error("engine: out of pinned host memory for the cold experts");
    if (blk) {
      owned_.push_back(blk);
      pinned_bytes_ += uint64_t(cold.size()) * expert_bytes_;
    }
    for (size_t j = 0; j < cold.size(); ++j) {
      const size_t e = size_t(cold[j]);
      L.cold_gu[e] = blk ? blk + j * expert_bytes_ : L.gu_host + e * expert_raw_gu_;
      L.cold_dn[e] = blk ? blk + j * expert_bytes_ + expert_raw_gu_ : L.dn_host + e * E * dn_row;
    }
    if (blk) {
      pool_->for_each(int64_t(cold.size()), 1, [&](int64_t lo, int64_t hi) {
        for (int64_t j = lo; j < hi; ++j) {
          std::memcpy(blk + j * expert_bytes_, L.gu_host + size_t(cold[size_t(j)]) * expert_raw_gu_, expert_raw_gu_);
          std::memcpy(blk + j * expert_bytes_ + expert_raw_gu_, L.dn_host + size_t(cold[size_t(j)]) * E * dn_row, size_t(E * dn_row));
        }
      });
    }
  }
  if (n_stream_ == 0) {
    std::vector<std::pair<const uint8_t*, size_t>> spans;
    for (const Layer& L : layers_)
      for (int e = 0; e < NE; ++e)
        if (L.cold[size_t(e)]) {
          spans.push_back({L.gu_host + size_t(e) * 2 * Fe * gu_row, size_t(2 * Fe * gu_row)});
          spans.push_back({L.dn_host + size_t(e) * E * dn_row, size_t(E * dn_row)});
        }
    std::atomic<uint64_t> sink{0};
    pool_->for_each(int64_t(spans.size()), 8, [&](int64_t lo, int64_t hi) {
      uint64_t s = 0;
      for (int64_t i = lo; i < hi; ++i)
        for (size_t o = 0; o < spans[size_t(i)].second; o += 4096) s += spans[size_t(i)].first[o];
      sink += s;
    });
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
  auto run = [&](const std::vector<Seg>& segs, const std::vector<const uint8_t*>& expert, int64_t k, int64_t n, int64_t step) {
    const int64_t per = (n + step - 1) / step;
    pool_->for_each(int64_t(segs.size()) * per, 1, [&](int64_t lo, int64_t hi) {
      for (int64_t t = lo; t < hi; ++t) {
        const Seg& s = segs[size_t(t / per)];
        const int64_t r0 = (t % per) * step, r1 = std::min(n, r0 + step);
        cpu::gemv_q4_multi(expert[size_t(s.e)], k, r0, r1, s.xs.data(), int(s.xs.size()), s.ys.data());
      }
    });
  };
  run(gu_segs, L.cold_gu, E, 2 * Fe, 32);
  // GeGLU (the tanh form, as ref::gelu_mul) and the int8 quantization of its output, per pair.
  pool_->for_each(nc, 4, [&](int64_t lo, int64_t hi) {
    std::vector<float> g(static_cast<size_t>(Fe));
    for (int64_t k = lo; k < hi; ++k) {
      const float* gu = cold_gu_.data() + k * 2 * Fe;
      for (int64_t j = 0; j < Fe; ++j) {
        const float a = gu[j];
        g[size_t(j)] = 0.5f * a * (1.0f + std::tanh(0.7978845608028654f * (a + 0.044715f * a * a * a))) * gu[Fe + j];
      }
      cold_g_[size_t(k)] = cpu::quantize(g.data(), Fe);
    }
  });
  run(dn_segs, L.cold_dn, Fe, E, 64);
}

void Engine::cold_decode(const Layer& L, int64_t T, int64_t P, sycl::event routed, sycl::event hot_done) {
  using Clock = std::chrono::steady_clock;
  const int64_t E = cfg_.n_embd, NU = cfg_.n_expert_used, kb = E / 32;
  const auto t0 = Clock::now();
  routed.wait();
  const auto t1 = Clock::now();
  const std::vector<int32_t> cp = cold_list(L, hp_ids_, P);
  pairs_ += uint64_t(P);
  cold_pairs_ += cp.size();
  if (!cp.empty()) {
    // The expert inputs are the card's own int8 rows (the first 8-row tile of act_exp_).
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
  (void)NU;
  const auto t2 = Clock::now();
  cold_.wait_s += std::chrono::duration<double>(t1 - t0).count();
  cold_.cpu_s += std::chrono::duration<double>(t2 - t1).count();
  ++cold_.layers;
  if (hot_done.get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete) ++cold_.late_layers;
}

std::vector<int32_t> Engine::choose_stream(const Layer& L, const std::vector<int32_t>& cnt) const {
  const int64_t NE = cfg_.n_expert;
  std::vector<int32_t> cand;
  double rest = 0;
  for (int e = 0; e < NE; ++e)
    if (L.cold[size_t(e)] && cnt[size_t(e)] > 0) {
      rest += cnt[size_t(e)];
      if (cnt[size_t(e)] >= c_.stream_min_rows) cand.push_back(e);
    }
  std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return cnt[size_t(a)] != cnt[size_t(b)] ? cnt[size_t(a)] > cnt[size_t(b)] : a < b; });
  // The copies and the CPU work run side by side: stream the busiest candidates until the copies would take
  // longer than the CPU does on what is left, within the slots and the rows the gathered inputs can hold.
  size_t best_k = 0;
  double best = rest * c_.stream_cpu_us;
  int64_t rows = 0;
  for (size_t k = 1; k <= cand.size() && k <= size_t(n_stream_); ++k) {
    const int32_t n = cnt[size_t(cand[k - 1])];
    rest -= n;
    rows += (n + moe_tile_ - 1) / moe_tile_ * moe_tile_;
    if (rows > stream_rows_cap_) break;
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
  const uint64_t gu_row = uint64_t(E) / 32 * 18, dn_row = uint64_t(Fe) / 32 * 18;
  // The copy engine moves each expert's bytes (gate_up then down, as the GGUF has them) from pinned memory into its
  // raw slot; the host does not wait: the queue does, on the copies' fence, before the repacking, the grouping and
  // the GEMMs, which come after the hot experts'.
  for (size_t j = 0; j < list.size(); ++j) copy_->copy(stream_raw_ + uint64_t(j) * expert_bytes_, L.cold_gu[size_t(list[j])], expert_bytes_);
  const sycl::event copied = copy_->event(copy_->fence());
  q_.submit([&](sycl::handler& h) {
    h.depends_on(copied);
    h.single_task([] {});
  });
  for (size_t j = 0; j < list.size(); ++j) {
    const uint8_t* raw = stream_raw_ + uint64_t(j) * expert_bytes_;
    xmx::repack_q4_0_into(q_, ref::Weight{gguf::Type::Q4_0, E, 2 * Fe, raw, gu_row}, true, stream_w_gu_[j]);
    xmx::repack_q4_0_into(q_, ref::Weight{gguf::Type::Q4_0, Fe, E, raw + expert_raw_gu_, dn_row}, false, stream_w_dn_[j]);
  }
  q_.memcpy(stream_map_, hp_smap_, size_t(NE) * 4);
  mark("stream_copy_repack");
  // The pairs that chose them get rows after the hot experts' (so that moe_combine takes both from one row table).
  fast::moe_group(q_, moe_ids_, P, n_stream_, moe_tile_, stream_scratch_, stream_tiles_, moe_row_, nullptr, stream_map_, hot_rows);
  sycl::half* a = reinterpret_cast<sycl::half*>(att_);
  fast::moe_gather(q_, h_, moe_row_, P, NU, E, a, hot_rows);
  xmx::matmul_geglu_grouped16(q_, a, stream_gu_, stream_tiles_, s_tiles, moe_tile_, moe_h_ + int64_t(hot_rows) * Fe);
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
  xmx::free_activations(q_, act_ffn_);
  if (act_exp_.q) xmx::free_activations(q_, act_exp_);
  if (act_exp_ffn_.q) xmx::free_activations(q_, act_exp_ffn_);
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

void Engine::truncate(int slot, int64_t length) {
  Slot& s = slots_.at(size_t(slot));
  if (length < 0 || length > s.length) throw std::runtime_error("engine: truncate beyond the sequence");
  const size_t keep = size_t((length + attn::kPage - 1) / attn::kPage);
  while (s.pages.size() > keep) {
    free_pages_.push_back(s.pages.back());
    s.pages.pop_back();
  }
  s.length = length;
}

void Engine::release(int slot) {
  Slot& s = slots_.at(size_t(slot));
  for (int32_t p : s.pages) free_pages_.push_back(p);
  s.pages.clear();
  s.length = 0;
}

std::vector<float> Engine::step(const std::vector<Chunk>& chunks, const model::DumpSink* dump) {
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, V = cfg_.n_vocab;
  std::vector<int32_t> tok, slot, pos, logit_rows;
  attn::TileList tiles_s(cfg_.head_dim_sliding), tiles_g(cfg_.head_dim_global);
  for (const auto& c : chunks) {
    if (c.slot < 0 || c.slot >= c_.n_slots) throw std::runtime_error("engine: bad slot");
    if (c.start_pos != slots_[size_t(c.slot)].length) throw std::runtime_error("engine: chunk does not continue its slot");
    const int64_t base = int64_t(tok.size()), n = int64_t(c.tokens.size());
    if (n > c_.max_chunk) throw std::runtime_error("engine: chunk longer than max_chunk");
    if (n == 0) continue;
    for (int64_t i = 0; i < n; ++i) {
      tok.push_back(c.tokens[size_t(i)]);
      slot.push_back(c.slot);
      pos.push_back(int32_t(c.start_pos + i));
    }
    tiles_s.add_chunk(int32_t(base), n, c.start_pos + n - 1, group_sliding_, hkv_sliding_, cfg_.window);
    tiles_g.add_chunk(int32_t(base), n, c.start_pos + n - 1, group_global_, hkv_global_, 0);
    for (int64_t r = std::max<int64_t>(0, n - c.logits_rows); r < n; ++r) logit_rows.push_back(int32_t(base + r));
    ensure_pages(c.slot, c.start_pos + n);
  }
  const int64_t T = int64_t(tok.size()), R = int64_t(logit_rows.size());
  // Decode calls the pool every few hundred microseconds and wants its workers spinning; prefill's phases
  // last milliseconds and the spinning only takes the cores from the hot kernels' host thread.
  if (pool_) pool_->set_spin_us(T * cfg_.n_expert_used <= xmx::kFp16Rows ? 400 : 0);
  if (T == 0) return {};
  const auto t_step = std::chrono::steady_clock::now();
  if (T > c_.max_batch) throw std::runtime_error("engine: step larger than max_batch");
  if (R > c_.max_logits) throw std::runtime_error("engine: too many logits rows");
  q_.memcpy(tok_, tok.data(), size_t(T) * 4);
  q_.memcpy(slot_, slot.data(), size_t(T) * 4);
  q_.memcpy(pos_, pos.data(), size_t(T) * 4);
  const std::vector<int32_t> packed_s = tiles_s.packed(), packed_g = tiles_g.packed();
  q_.memcpy(tiles_sliding_, packed_s.data(), packed_s.size() * 4);
  q_.memcpy(tiles_global_, packed_g.data(), packed_g.size() * 4);
  q_.memcpy(page_table_, page_table_host_.data(), page_table_host_.size() * 4);
  const attn::Tokens toks{slot_, pos_, T};
  // On a profiling queue each mark is an empty kernel whose end timestamp closes the section, so
  // the card's own time is measured; otherwise the host waits at each mark.
  const bool gpu_marks = profile_ && q_.has_property<sycl::property::queue::enable_profiling>();
  std::vector<std::pair<const char*, sycl::event>> marks;
  auto settle = [&] {
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

  // Each layer starts with its attention input already normed and quantized in act_: by the
  // fused kernel at the end of the layer before, or here for the first.
  fast::embed(q_, head_, tok_, T, std::sqrt(float(E)), x_);
  emit("inp_scaled", x_, {E, T});
  fast::residual_norm_quant(q_, nullptr, nullptr, x_, 1.0f, nullptr, layers_[0].attn_norm, cfg_.eps, T, E, &act_, h_);
  mark("embed");
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const Layer& L = layers_[size_t(l)];
    const std::string s = "-" + std::to_string(l);
    const bool sliding = cfg_.sliding[size_t(l)];
    // The 26B at decode: fused norms and tail kernels, everything on the card.
    const bool fused = cfg_.n_expert > 0 && T * cfg_.n_expert_used <= xmx::kFp16Rows;
    const int64_t D = cfg_.head_dim(l), Hkv = cfg_.n_head_kv[size_t(l)];
    const float* inv = sliding ? inv_freq_sliding_ : inv_freq_global_;
    emit("attn_norm" + s, h_, {E, T});

    xmx::matmul(q_, act_, L.wqkv, qkv_);
    mark("qkv");
    fast::qkv_heads(q_, qkv_, T, H, Hkv, D, L.has_v, L.q_norm, L.k_norm, cfg_.eps, pos_, inv, qb_, kb_, vb_);
    emit("Qcur_pos" + s, qb_, {D, H, T});
    emit("Kcur_pos" + s, kb_, {D, Hkv, T});
    emit("Vcur_normed" + s, vb_, {D, Hkv, T});
    mark("qkv_heads");

    if (sliding) {
      const attn::Cache cache{ring_k_ + L.ring_index * ring_layer_stride_, ring_v_ + L.ring_index * ring_layer_stride_, nullptr,
                              ring_pages_, cfg_.window, Hkv, D};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_s, tiles_sliding_, cache, H, 1.0f, att_, attn_scratch_);
      mark("attn_sliding");
    } else {
      const GlobalCache& g = global_[size_t(L.page_index)];
      const attn::Cache cache{g.k, g.v, page_table_, max_pages_, 0, Hkv, D, g.ks, g.vs};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_g, tiles_global_, cache, H, 1.0f, att_, attn_scratch_);
      mark("attn_global");
    }
    emit("kqv_out" + s, att_, {H * D, T});
    act_.k = H * D;
    xmx::quantize(q_, att_, T, act_);
    xmx::matmul(q_, act_, L.wo, o_);
    mark("wo");
    if (fused)
      fast::residual_norm_quant3(q_, o_, L.attn_post_norm, x_, x2_, cfg_.eps, T, E, act_, L.ffn_norm, act_exp_, L.pre_ffw_norm_2, h_,
                                 L.router_in_scale);
    else
      fast::residual_norm_quant(q_, o_, L.attn_post_norm, x_, 1.0f, x2_, L.ffn_norm, cfg_.eps, T, E, &act_, h_);
    emit("attn_out" + s, x2_, {E, T});
    mark("post_attn");

    if (!fused) {
      xmx::matmul_geglu(q_, act_, L.gate_up, act_ffn_, gu_);
      mark("gate_up");
      xmx::matmul(q_, act_ffn_, L.down, o_);
      mark("down");
    }
    if (cfg_.n_expert > 0) {
      // The 26B's MoE: the dense part is normed on its own (post_ffw_norm_1), then added to the
      // routed experts' sum (normed with post_ffw_norm_2).
      const int64_t n_expert = cfg_.n_expert, n_used = cfg_.n_expert_used;
      if (!fused) fast::rmsnorm(q_, o_, L.post_ffw_norm_1, T, E, cfg_.eps, o_);
      if (!fused) emit("ffn_mlp" + s, o_, {E, T});
      // Router: r = rmsnorm(x2_) * (ffn_gate_inp.scale / sqrt(E)), then r @ W_router^T.
      if (!fused) fast::rmsnorm(q_, x2_, L.router_in_scale, T, E, cfg_.eps, h_);
      const float* w_router = reinterpret_cast<const float*>(L.w_router.data);
      if (T * n_used <= xmx::kFp16Rows) fast::matmul_f32_small(q_, w_router, n_expert, E, h_, T, rlog_);
      else fast::matmul_f32(q_, w_router, n_expert, E, h_, T, rlog_);
      mark("router");
      emit("ffn_moe_logits" + s, rlog_, {n_expert, T});
      const int64_t P = T * n_used;
      if (P <= xmx::kFp16Rows) {
        // Decode (up to 4 tokens): routing, the grouped expert GEMVs and the weighted sum all on the
        // card, nothing copied to or waited on by the host.
        fast::moe_route(q_, rlog_, T, n_expert, n_used, L.exp_down_scale_dev, moe_ids_, moe_w_);
        mark("moe_route");
        // Cold experts: the routing and the experts' int8 input row come to pinned host memory, and the host
        // waits for them only after the hot experts' kernels are enqueued, so the card works meanwhile.
        sycl::event routed;
        if (L.n_cold) {
          const int64_t kb = E / 32;
          routed = fast::host_pack(q_, moe_ids_, P, act_exp_.q, kb * 256, act_exp_.s, kb * 8, hp_ids_, hp_aq_, hp_as_);
        }
        // The dense FFN's gate|up rides along as three more experts; its down follows.
        xmx::matmul_geglu_grouped_dense(q_, act_exp_, act_, L.tab_gate_up, moe_ids_, P, n_used, T, act_exp_ffn_, act_ffn_, moe_gu_);
        mark("moe_gate_up");
        xmx::matmul(q_, act_ffn_, L.down, o_);
        mark("down");
        xmx::matmul_grouped(q_, act_exp_ffn_, L.tab_down, moe_ids_, P, 1, gx_);
        mark("moe_down");
        if (L.n_cold) cold_decode(L, T, P, routed, q_.single_task([] {}));
        // The layer's tail in one kernel; with a dump sink its intermediates are kept for it.
        const bool d1 = dump && dump->wants("ffn_mlp" + s), d2 = dump && dump->wants("ffn_moe" + s),
                   d3 = dump && dump->wants("ffn_moe_combined" + s);
        const float* next_norm = l + 1 < cfg_.n_layer ? layers_[size_t(l) + 1].attn_norm : nullptr;
        fast::moe_tail(q_, o_, L.post_ffw_norm_1, gx_, moe_w_, n_used, L.post_ffw_norm_2, L.ffn_post_norm, x2_, L.out_scale, x_,
                       next_norm, cfg_.eps, T, E, &act_, d1 ? h_ : nullptr, d2 ? moe_ : nullptr, d3 ? o_ : nullptr);
        if (d1) emit("ffn_mlp" + s, h_, {E, T});
        // The next layer's dump of attn_norm reads h_, which the unfused kernel wrote.
        if (next_norm && dump && dump->wants("attn_norm-" + std::to_string(l + 1))) fast::rmsnorm(q_, x_, next_norm, T, E, cfg_.eps, h_);
        mark("moe_tail");
      } else {
        // Prefill: routing, grouping of the pairs by expert into row tiles, the experts' GEMMs over all
        // their tokens at once and the weighted sum, all on the card.
        fast::moe_route(q_, rlog_, T, n_expert, n_used, L.exp_down_scale_dev, moe_ids_, moe_w_);
        mark("moe_route");
        // Expert input rmsnorm(x2_) * pre_ffw_norm_2 (h_ is scratch).
        fast::rmsnorm(q_, x2_, L.pre_ffw_norm_2, T, E, cfg_.eps, h_);
        sycl::event routed, ids_ready;
        if (L.n_cold) {
          // As at decode, the host waits for the routing and the input rows after the hot work is enqueued (the ids
          // first: they say which cold experts to stream).
          ids_ready = q_.memcpy(hp_ids_, moe_ids_, size_t(P) * 4);
          q_.memcpy(hp_w_, moe_w_, size_t(P) * 4);
          // The expert inputs go as int8 (the form the CPU computes with), a quarter of the bytes.
          fast::quantize_rows(q_, h_, T, E, reinterpret_cast<int8_t*>(qb_), kb_);
          q_.memcpy(hp_xq_, qb_, size_t(T * E));
          routed = q_.memcpy(hp_xs_, kb_, size_t(T) * size_t(E / 32) * sizeof(float));
        }
        fast::moe_group(q_, moe_ids_, P, n_expert, moe_tile_, moe_scratch_, moe_tiles_, moe_row_, L.cold_dev);
        mark("moe_group");
        // Each pair's row in sorted order.
        fast::moe_gather(q_, h_, moe_row_, P, n_used, E, moe_a_);
        mark("moe_gather");
        const int64_t max_tiles = (P + moe_tile_ - 1) / moe_tile_ + n_expert;
        xmx::matmul_geglu_grouped16(q_, moe_a_, L.tab_gate_up, moe_tiles_, max_tiles, moe_tile_, moe_h_);
        mark("moe_gate_up");
        xmx::matmul_grouped16(q_, moe_h_, L.tab_down, moe_tiles_, max_tiles, moe_tile_, moe_y_);
        mark("moe_down");
        if (L.n_cold) cold_prefill(L, T, P, ids_ready, routed, q_.single_task([] {}), mark);
        fast::moe_combine(q_, moe_y_, moe_w_, T, n_used, E, moe_, moe_row_, L.n_cold ? cold_sum_ : nullptr, L.n_cold ? cold_map_ : nullptr);
        mark("moe_sum");
      }
      if (!fused) fast::rmsnorm(q_, moe_, L.post_ffw_norm_2, T, E, cfg_.eps, moe_);
      mark("moe_post_norm");
      emit("ffn_moe" + s, moe_, {E, T});
      // Combine: o_ = ffn_mlp + moe.
      if (!fused) {
        float* o = o_;
        const float* moe = moe_;
        q_.parallel_for(sycl::range<1>(size_t(T * E)), [=](sycl::id<1> id) { o[id[0]] += moe[id[0]]; });
      }
      emit("ffn_moe_combined" + s, o_, {E, T});
      mark("moe_add");
    }
    const float* next_norm = l + 1 < cfg_.n_layer ? layers_[size_t(l) + 1].attn_norm : nullptr;
    if (!fused) fast::residual_norm_quant(q_, o_, L.ffn_post_norm, x2_, L.out_scale, x_, next_norm, cfg_.eps, T, E, &act_, h_);
    emit("l_out" + s, x_, {E, T});
    mark("post_ffn");
  }

  for (const auto& c : chunks) slots_[size_t(c.slot)].length += int64_t(c.tokens.size());
  if (R == 0) {
    enqueue_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
    q_.wait();
    settle();
    return {};
  }
  // Logits only for the rows asked for: gather them, final norm, tied head, softcap.
  for (int64_t r = 0; r < R; ++r) q_.memcpy(rows_ + r * E, x_ + int64_t(logit_rows[size_t(r)]) * E, size_t(E) * sizeof(float));
  fast::rmsnorm(q_, rows_, output_norm_, R, E, cfg_.eps, rows_);
  emit("result_norm", rows_, {E, R});
  fast::matmul_q6(q_, head_, rows_, R, logits_);
  if (cfg_.softcap > 0) fast::softcap(q_, logits_, cfg_.softcap, R * V);
  mark("head");
  std::vector<float> out(size_t(R * V));
  auto done = q_.memcpy(out.data(), logits_, out.size() * sizeof(float));
  enqueue_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
  done.wait();
  settle();
  return out;
}

}  // namespace eklow::engine
