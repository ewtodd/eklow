#include "engine/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "kernels/fast.hpp"

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
    if (t.type == gguf::Type::Q4_0) staging_bytes = std::max(staging_bytes, t.nbytes);
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
      const int64_t Fe = cfg_.n_ff_exp, n_expert = cfg_.n_expert;
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
      // Repack each expert's gate_up (interleaved) and down, staging each into the shared buffer
      // so the originals are not kept on the card.
      const uint64_t gu_row = uint64_t(E) / 32 * 18, dn_row = uint64_t(Fe) / 32 * 18;
      L.exp_gate_up.resize(n_expert);
      L.exp_down.resize(n_expert);
      for (int e = 0; e < n_expert; ++e) {
        q_.memcpy(staging, gu_t.data + size_t(e) * 2 * Fe * gu_row, size_t(2 * Fe * gu_row)).wait();
        L.exp_gate_up[size_t(e)] = xmx::repack_q4_0(q_, ref::Weight{gguf::Type::Q4_0, E, 2 * Fe, staging, gu_row}, true);
        q_.memcpy(staging, dn_t.data + size_t(e) * E * dn_row, size_t(E * dn_row)).wait();
        L.exp_down[size_t(e)] = xmx::repack_q4_0(q_, ref::Weight{gguf::Type::Q4_0, Fe, E, staging, dn_row}, false);
        for (xmx::Weight* w : {&L.exp_gate_up[size_t(e)], &L.exp_down[size_t(e)]}) {
          owned_.push_back(w->tiles);
          owned_.push_back(w->scales);
        }
        weight_bytes_ += size_t(2 * Fe * gu_row + E * dn_row);
      }
      // Device tables of the experts' pointers for the grouped decode kernels, and the down scales.
      L.tab_gate_up = xmx::make_expert_table(q_, L.exp_gate_up);
      L.tab_down = xmx::make_expert_table(q_, L.exp_down);
      owned_.push_back(L.tab_gate_up.tiles);
      owned_.push_back(L.tab_gate_up.scales);
      owned_.push_back(L.tab_down.tiles);
      owned_.push_back(L.tab_down.scales);
      L.exp_down_scale_dev = vec_host(L.exp_down_scale);
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
  act_ = xmx::alloc_activations(q_, B, std::max<int64_t>(E, H * Dg));
  act_ffn_ = xmx::alloc_activations(q_, B, F);
  if (cfg_.n_expert > 0) {
    const int64_t NE = cfg_.n_expert, NU = cfg_.n_expert_used, P = B * NU, Fe = cfg_.n_ff_exp;
    moe_ = dev(B * E);
    rlog_ = dev(B * NE);
    gx_ = dev(std::max<int64_t>(P, xmx::kFp16Rows) * E);
    // Per pair: the chosen expert and weight (decode keeps at most kFp16Rows pairs in them).
    moe_ids_ = sycl::malloc_device<int32_t>(size_t(P), q_);
    moe_row_ = sycl::malloc_device<int32_t>(size_t(P), q_);
    if (!moe_ids_ || !moe_row_) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(moe_ids_);
    owned_.push_back(moe_row_);
    moe_w_ = dev(P);
    moe_gu_ = dev(xmx::kFp16Rows * 2 * Fe);
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
    fast::residual_norm_quant(q_, o_, L.attn_post_norm, x_, 1.0f, x2_, L.ffn_norm, cfg_.eps, T, E, &act_, h_);
    emit("attn_out" + s, x2_, {E, T});
    mark("post_attn");

    xmx::matmul_geglu(q_, act_, L.gate_up, act_ffn_, gu_);
    mark("gate_up");
    xmx::matmul(q_, act_ffn_, L.down, o_);
    mark("down");
    if (cfg_.n_expert > 0) {
      // The 26B's MoE: the dense part is normed on its own (post_ffw_norm_1), then added to the
      // routed experts' sum (normed with post_ffw_norm_2).
      const int64_t Fe = cfg_.n_ff_exp, n_expert = cfg_.n_expert, n_used = cfg_.n_expert_used;
      fast::rmsnorm(q_, o_, L.post_ffw_norm_1, T, E, cfg_.eps, o_);
      emit("ffn_mlp" + s, o_, {E, T});
      // Router: r = rmsnorm(x2_) * (ffn_gate_inp.scale / sqrt(E)), then r @ W_router^T.
      fast::rmsnorm(q_, x2_, L.router_in_scale, T, E, cfg_.eps, h_);
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
        // Expert input rmsnorm(x2_) * pre_ffw_norm_2, quantized to T int8 rows (h_ is scratch).
        fast::residual_norm_quant(q_, nullptr, nullptr, x2_, 1.0f, nullptr, L.pre_ffw_norm_2, cfg_.eps, T, E, &act_exp_, h_);
        mark("moe_norm");
        xmx::matmul_geglu_grouped(q_, act_exp_, L.tab_gate_up, moe_ids_, P, n_used, act_exp_ffn_, moe_gu_);
        mark("moe_gate_up");
        xmx::matmul_grouped(q_, act_exp_ffn_, L.tab_down, moe_ids_, P, 1, gx_);
        mark("moe_down");
        fast::moe_combine(q_, gx_, moe_w_, T, n_used, E, moe_);
        mark("moe_sum");
      } else {
        // Prefill: routing, grouping of the pairs by expert into row tiles, the experts' GEMMs over all
        // their tokens at once and the weighted sum, all on the card.
        fast::moe_route(q_, rlog_, T, n_expert, n_used, L.exp_down_scale_dev, moe_ids_, moe_w_);
        mark("moe_route");
        fast::moe_group(q_, moe_ids_, P, n_expert, moe_tile_, moe_scratch_, moe_tiles_, moe_row_);
        mark("moe_group");
        // Expert input rmsnorm(x2_) * pre_ffw_norm_2 (h_ is scratch), each pair's row in sorted order.
        fast::rmsnorm(q_, x2_, L.pre_ffw_norm_2, T, E, cfg_.eps, h_);
        fast::moe_gather(q_, h_, moe_row_, P, n_used, E, moe_a_);
        mark("moe_gather");
        const int64_t max_tiles = (P + moe_tile_ - 1) / moe_tile_ + n_expert;
        xmx::matmul_geglu_grouped16(q_, moe_a_, L.tab_gate_up, moe_tiles_, max_tiles, moe_tile_, moe_h_);
        mark("moe_gate_up");
        xmx::matmul_grouped16(q_, moe_h_, L.tab_down, moe_tiles_, max_tiles, moe_tile_, moe_y_);
        mark("moe_down");
        fast::moe_combine(q_, moe_y_, moe_w_, T, n_used, E, moe_, moe_row_);
        mark("moe_sum");
      }
      fast::rmsnorm(q_, moe_, L.post_ffw_norm_2, T, E, cfg_.eps, moe_);
      mark("moe_post_norm");
      emit("ffn_moe" + s, moe_, {E, T});
      // Combine: o_ = ffn_mlp + moe.
      {
        float* o = o_;
        const float* moe = moe_;
        q_.parallel_for(sycl::range<1>(size_t(T * E)), [=](sycl::id<1> id) { o[id[0]] += moe[id[0]]; });
      }
      emit("ffn_moe_combined" + s, o_, {E, T});
      mark("moe_add");
    }
    const float* next_norm = l + 1 < cfg_.n_layer ? layers_[size_t(l) + 1].attn_norm : nullptr;
    fast::residual_norm_quant(q_, o_, L.ffn_post_norm, x2_, L.out_scale, x_, next_norm, cfg_.eps, T, E, &act_, h_);
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
