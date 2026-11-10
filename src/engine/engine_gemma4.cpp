#include "engine/engine_gemma4.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "kernels/fast.hpp"
#include "kernels/gemma_fast.hpp"
#include "kernels/qwen_fast.hpp"

namespace eklow::engine::gemma4 {

float* Engine::dev(int64_t n) {
  auto* p = sycl::malloc_device<float>(size_t(n), q_);
  if (!p) throw std::runtime_error("engine: out of device memory");
  owned_.push_back(p);
  return p;
}

Engine::Engine(sycl::queue& q, const gguf::File& f, const Config& cfg) : q_(q), c_(cfg), cfg_(model::Gemma4Config::from_gguf(f)) {
  if (c_.max_ctx % attn::kPage) throw std::runtime_error("engine: max_ctx must be a multiple of the page size");
  if (c_.max_chunk > c_.max_batch) throw std::runtime_error("engine: max_chunk above max_batch");
  const int64_t E = cfg_.n_embd, F = cfg_.n_ff, H = cfg_.n_head, NL = cfg_.n_layer, P = cfg_.n_embd_per_layer;
  if (P <= 0) throw std::runtime_error(f.path() + ": no per-layer inputs: eklow runs Gemma 4 E4B (Nemotron-3.5-Bolt-E4B)");
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

  // Matrices go through one staging buffer and are repacked for the matrix units in the file's format; matrices that
  // read the same input and share a type are stacked (their outputs side by side in each row).
  uint64_t staging_bytes = 0;
  for (const auto& t : f.tensors())
    if (t.ne.size() == 2 && t.type != gguf::Type::F32 && t.name != "per_layer_token_embd.weight") staging_bytes = std::max(staging_bytes, t.nbytes);
  for (int l = 0; l < NL; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    uint64_t sum = 0;
    for (const char* n : {"attn_q.weight", "attn_k.weight", "attn_v.weight"})
      if (const gguf::Tensor* t = f.tensor(b + n)) sum += t->nbytes;
    staging_bytes = std::max(staging_bytes, sum + need(b + "ffn_gate.weight").nbytes + need(b + "ffn_up.weight").nbytes);
  }
  auto* staging = sycl::malloc_device<uint8_t>(staging_bytes, q_);
  if (!staging) throw std::runtime_error("engine: out of device memory for weight staging");
  auto matrix = [&](const std::vector<const gguf::Tensor*>& parts, bool interleave = false) {
    uint64_t at = 0;
    int64_t n = 0;
    for (const gguf::Tensor* t : parts) {
      if (t->type != parts[0]->type || t->ne[0] != parts[0]->ne[0])
        throw std::runtime_error(f.path() + ": stacked matrices of different types or widths (" + t->name + ")");
      q_.memcpy(staging + at, t->data, t->nbytes);
      at += t->nbytes;
      n += int64_t(t->ne[1]);
      weight_bytes_ += t->nbytes;
    }
    q_.wait();
    const ref::Weight raw{parts[0]->type, int64_t(parts[0]->ne[0]), n, staging, parts[0]->nbytes / parts[0]->ne[1]};
    xmx::WeightFormat fmt;
    xmx::Weight w;
    if (xmx::format_of(raw.type, fmt))
      w = fmt == xmx::WeightFormat::Q4_0 ? xmx::repack_q4_0(q_, raw, interleave) : xmx::repack_format(q_, raw, fmt, interleave);
    else if (!interleave && (raw.type == gguf::Type::BF16 || raw.type == gguf::Type::F16 || raw.type == gguf::Type::F32))
      w = xmx::repack(q_, raw);  // F16 tiles, the fp16 GEMM only
    else
      throw std::runtime_error(f.path() + ": " + parts[0]->name + " is " + gguf::type_name(raw.type) + ", which the matrix units do not run");
    q_.wait();
    owned_.push_back(w.tiles);
    if (w.scales) owned_.push_back(w.scales);
    return w;
  };

  {
    // The head: the tied embedding on the matrix units. The lookups read the GGUF's mapping (no second copy on the card).
    const gguf::Tensor& emb = need("token_embd.weight");
    head_w_ = matrix({&emb});
    act_head_ = xmx::alloc_activations(q_, c_.max_logits, E);
    embd_ = &emb;
    embd_row_bytes_ = emb.nbytes / emb.ne[1];
    const int64_t Bm = c_.max_batch;
    emb_host_ = sycl::malloc_host<uint8_t>(size_t(Bm) * embd_row_bytes_, q_);
    emb_dev_ = sycl::malloc_device<uint8_t>(size_t(Bm) * embd_row_bytes_, q_);
    iota_ = sycl::malloc_device<int32_t>(size_t(Bm), q_);
    if (!emb_host_ || !emb_dev_ || !iota_) throw std::runtime_error("engine: out of memory for the embedding rows");
    owned_.push_back(emb_host_);
    owned_.push_back(emb_dev_);
    owned_.push_back(iota_);
    std::vector<int32_t> iota(static_cast<size_t>(Bm));
    for (int64_t i = 0; i < Bm; ++i) iota[size_t(i)] = int32_t(i);
    q_.memcpy(iota_, iota.data(), iota.size() * 4).wait();
  }
  output_norm_ = vec("output_norm.weight");
  pl_proj_ = matrix({&need("per_layer_model_proj.weight")});
  {
    const gguf::Tensor& t = need("per_layer_token_embd.weight");
    if (t.type != gguf::Type::Q8_0 || t.ne.size() != 2 || int64_t(t.ne[0]) != NL * P)
      throw std::runtime_error(f.path() + ": per_layer_token_embd.weight is not Q8_0 rows of n_layer x 256");
    pl_table_ = ref::Weight{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), upload(t), t.nbytes / t.ne[1]};
  }
  pl_norm_ = vec("per_layer_proj_norm.weight");
  if (const gguf::Value* v = f.find("tokenizer.ggml.padding_token_id")) pad_id_ = int32_t(v->as_int());

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
  int64_t qkv_max = 0;
  for (int l = 0; l < NL; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L{};
    L.attn_norm = vec(b + "attn_norm.weight");
    L.writes_kv = cfg_.owns_kv(l);
    std::vector<const gguf::Tensor*> proj = {&need(b + "attn_q.weight")};
    if (L.writes_kv) {
      proj.push_back(&need(b + "attn_k.weight"));
      L.has_v = f.tensor(b + "attn_v.weight") != nullptr;
      if (L.has_v) proj.push_back(&need(b + "attn_v.weight"));
      L.k_norm = vec(b + "attn_k_norm.weight");
    }
    // Runs of one type stack; a Q6_K V beside Q4_K Q and K is its own part.
    int64_t total = 0;
    for (size_t i = 0; i < proj.size();) {
      size_t j = i + 1;
      while (j < proj.size() && proj[j]->type == proj[i]->type) ++j;
      L.qkv.push_back(matrix(std::vector<const gguf::Tensor*>(proj.begin() + int64_t(i), proj.begin() + int64_t(j))));
      total += L.qkv.back().n;
      i = j;
    }
    qkv_max = std::max(qkv_max, total);
    L.wo = matrix({&need(b + "attn_output.weight")});
    L.q_norm = vec(b + "attn_q_norm.weight");
    L.attn_post_norm = vec(b + "post_attention_norm.weight");
    L.ffn_norm = vec(b + "ffn_norm.weight");
    L.gate_up = matrix({&need(b + "ffn_gate.weight"), &need(b + "ffn_up.weight")}, true);
    L.down = matrix({&need(b + "ffn_down.weight")});
    L.ffn_post_norm = vec(b + "post_ffw_norm.weight");
    L.inp_gate = matrix({&need(b + "inp_gate.weight")});
    L.proj = matrix({&need(b + "proj.weight")});
    L.post_norm = vec(b + "post_norm.weight");
    const gguf::Tensor* os = f.tensor(b + "layer_output_scale.weight");
    L.out_scale = os ? *reinterpret_cast<const float*>(os->data) : 1.0f;
    if (L.writes_kv) {
      if (cfg_.sliding[size_t(l)]) L.ring_index = n_ring++;
      else L.page_index = n_page++;
    } else {
      const Layer& src = layers_[size_t(cfg_.kv_source(l))];
      L.ring_index = src.ring_index;
      L.page_index = src.page_index;
    }
    layers_.push_back(L);
  }
  sycl::free(staging, q_);

  // KV cache: rings for the sliding layers, a page pool for the global ones.
  for (int l = 0; l < NL; ++l) {
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
  // The global layers' pages, one allocation per layer and array (Level Zero refuses single allocations above 4 GB).
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
  const int64_t Dmax = std::max(cfg_.head_dim_global, cfg_.head_dim_sliding), kv_max = std::max(hkv_sliding_, hkv_global_) * Dmax;
  xa_ = dev(B * E);
  xb_ = dev(B * E);
  h_ = dev(B * E);
  o_ = dev(B * E);
  qb_ = dev(B * H * Dg);
  att_ = dev(B * H * Dg);
  kb_ = dev(B * kv_max);
  vb_ = dev(B * kv_max);
  qkv_ = dev(B * qkv_max);
  gu_ = dev(std::min(B, xmx::kFp16Rows) * 2 * F);  // the GeGLU's int8 path only
  pl_ = dev(B * NL * P);
  pg_ = dev(B * P);
  rows_ = dev(c_.max_logits * E);
  logits_ = dev(c_.max_logits * cfg_.n_vocab);
  cap_col_.assign(size_t(NL), -1);
  for (size_t j = 0; j < c_.capture_layers.size(); ++j) {
    const int l = c_.capture_layers[j];
    if (l < 0 || l >= NL) throw std::runtime_error("engine: capture layer out of range");
    cap_col_[size_t(l)] = int(j);
  }
  if (!c_.capture_layers.empty() && c_.max_capture_rows > 0) {
    cap_ = dev(c_.max_capture_rows * int64_t(c_.capture_layers.size()) * E);
    cap_idx_ = sycl::malloc_device<int32_t>(size_t(c_.max_capture_rows), q_);
    if (!cap_idx_) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(cap_idx_);
    capture_.width = int64_t(c_.capture_layers.size()) * E;
  }
  // At most one chunk per slot in a step, so at most n_slots partial tiles per K/V head.
  max_tiles_sliding_ = hkv_sliding_ * ((B * group_sliding_ + 7) / 8 + c_.n_slots);
  max_tiles_global_ = hkv_global_ * ((B * group_global_ + 7) / 8 + c_.n_slots);
  // Tile lists packed with their groups: 4 ints a tile, at most 2 a tile for its group.
  for (auto [p, n] : {std::pair{&tok_, B}, {&slot_, B}, {&pos_, B}, {&lo_, B}, {&hi_, B}, {&tiles_sliding_, 6 * max_tiles_sliding_},
                      {&tiles_global_, 6 * max_tiles_global_}}) {
    *p = sycl::malloc_device<int32_t>(size_t(n), q_);
    if (!*p) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(*p);
  }
  // The attention scratch at its largest, so that no request allocates on the card.
  attn::reserve(q_, attn_scratch_, max_tiles_sliding_, c_.max_chunk * group_sliding_, hkv_sliding_, cfg_.head_dim_sliding, group_sliding_,
                cfg_.window + c_.max_chunk);
  attn::reserve(q_, attn_scratch_, max_tiles_global_, c_.max_chunk * group_global_, hkv_global_, cfg_.head_dim_global, group_global_,
                c_.max_ctx + attn::kPage);
  act_ = xmx::alloc_activations(q_, B, std::max<int64_t>(E, H * Dg));
  act_ffn_ = xmx::alloc_activations(q_, B, F);
  act_pl_ = xmx::alloc_activations(q_, B, P);
}

Engine::~Engine() {
  q_.wait();
  xmx::free_activations(q_, act_);
  xmx::free_activations(q_, act_ffn_);
  xmx::free_activations(q_, act_pl_);
  xmx::free_activations(q_, act_head_);
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

ref::Weight Engine::embedding_table() const {
  return ref::Weight{embd_->type, int64_t(embd_->ne[0]), int64_t(embd_->ne[1]), embd_->data, embd_row_bytes_};
}

void Engine::release(int slot) {
  Slot& s = slots_.at(size_t(slot));
  for (int32_t p : s.pages) free_pages_.push_back(p);
  s.pages.clear();
  s.length = 0;
}

std::vector<float> Engine::step(const std::vector<Chunk>& chunks, const model::DumpSink* dump) {
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, V = cfg_.n_vocab, NL = cfg_.n_layer, P = cfg_.n_embd_per_layer;
  std::vector<int32_t> tok, slot, pos, logit_rows, cap_rows;
  capture_.ids.clear();
  const int ncap = int(c_.capture_layers.size());
  // Image spans make the sliding layers' key ranges per token (lo, hi); without any, tokens are causal.
  std::vector<int32_t> lo, hi;
  std::vector<std::pair<int64_t, const model::ImageRows*>> spans;  // batch row of the span's first token
  bool ranged = false;
  for (const auto& c : chunks) ranged |= !c.images.empty();
  attn::TileList tiles_s(cfg_.head_dim_sliding), tiles_g(cfg_.head_dim_global);
  for (const auto& c : chunks) {
    if (c.slot < 0 || c.slot >= c_.n_slots) throw std::runtime_error("engine: bad slot");
    if (c.start_pos != slots_[size_t(c.slot)].length) throw std::runtime_error("engine: chunk does not continue its slot");
    const int64_t base = int64_t(tok.size()), n = int64_t(c.tokens.size());
    if (n > c_.max_chunk) throw std::runtime_error("engine: chunk longer than max_chunk");
    if (n == 0) continue;
    if (c.verify) {
      // The rewind after a rejection must stay within what the sliding rings can restore (see ring_slack()).
      if (n < 2 || n > kMaxVerify) throw std::runtime_error("engine: a verify chunk holds 2 to " + std::to_string(kMaxVerify) + " tokens");
      if (n - 1 > ring_slack()) throw std::runtime_error("engine: verify chunk longer than the sliding rings' slack");
    }
    if (c.capture_rows < 0 || c.capture_rows > n || (c.capture_rows > 0 && !cap_)) throw std::runtime_error("engine: bad capture_rows");
    for (int64_t r = n - c.capture_rows; r < n; ++r) {
      cap_rows.push_back(int32_t(base + r));
      capture_.ids.push_back({int32_t(c.slot), int32_t(c.start_pos + r)});
    }
    for (int64_t i = 0; i < n; ++i) {
      // An image row's id (below zero) looks up the pad token: its embedding is replaced by the row below, and its per-layer
      // input's token part is the pad's, as transformers' Gemma4Model computes it.
      tok.push_back(c.tokens[size_t(i)] < 0 ? pad_id_ : c.tokens[size_t(i)]);
      slot.push_back(c.slot);
      pos.push_back(int32_t(c.start_pos + i));
    }
    if (ranged) {
      int64_t at = 0;
      for (const auto& im : c.images) {
        if (im.n <= 0 || im.offset < at || im.offset + im.n > n || !im.rows) throw std::runtime_error("engine: bad image span");
        at = im.offset + im.n;
        spans.push_back({base + im.offset, &im});
      }
      // Image tokens see up to the span's last position; every other token is causal. lo and hi do not decrease.
      size_t next = 0;
      for (int64_t i = 0; i < n; ++i) {
        const int64_t p = c.start_pos + i;
        int64_t h = p;
        while (next < c.images.size() && i >= c.images[next].offset + c.images[next].n) ++next;
        if (next < c.images.size() && i >= c.images[next].offset) h = c.start_pos + c.images[next].offset + c.images[next].n - 1;
        lo.push_back(int32_t(std::max<int64_t>(0, p - cfg_.window + 1)));
        hi.push_back(int32_t(h));
      }
      tiles_s.add_chunk(int32_t(base), n, c.start_pos + n - 1, group_sliding_, hkv_sliding_, 0, lo[size_t(base)], hi.back());
    } else {
      tiles_s.add_chunk(int32_t(base), n, c.start_pos + n - 1, group_sliding_, hkv_sliding_, cfg_.window);
    }
    tiles_g.add_chunk(int32_t(base), n, c.start_pos + n - 1, group_global_, hkv_global_, 0);
    for (int64_t r = std::max<int64_t>(0, n - (c.verify ? n : c.logits_rows)); r < n; ++r) logit_rows.push_back(int32_t(base + r));
    ensure_pages(c.slot, c.start_pos + n);
  }
  const int64_t T = int64_t(tok.size()), R = int64_t(logit_rows.size());
  if (int64_t(cap_rows.size()) > c_.max_capture_rows) throw std::runtime_error("engine: too many capture rows");
  capture_.rows = int64_t(cap_rows.size());
  capture_.data = cap_;
  if (T == 0) return {};
  if (!cap_rows.empty()) q_.memcpy(cap_idx_, cap_rows.data(), cap_rows.size() * 4);
  const auto t_step = std::chrono::steady_clock::now();
  if (T > c_.max_batch) throw std::runtime_error("engine: step larger than max_batch");
  if (R > c_.max_logits) throw std::runtime_error("engine: too many logits rows");
  q_.memcpy(tok_, tok.data(), size_t(T) * 4);
  q_.memcpy(slot_, slot.data(), size_t(T) * 4);
  q_.memcpy(pos_, pos.data(), size_t(T) * 4);
  if (ranged) {
    q_.memcpy(lo_, lo.data(), size_t(T) * 4);
    q_.memcpy(hi_, hi.data(), size_t(T) * 4);
  }
  const std::vector<int32_t> packed_s = tiles_s.packed(), packed_g = tiles_g.packed();
  q_.memcpy(tiles_sliding_, packed_s.data(), packed_s.size() * 4);
  q_.memcpy(tiles_global_, packed_g.data(), packed_g.size() * 4);
  q_.memcpy(page_table_, page_table_host_.data(), page_table_host_.size() * 4);
  const attn::Tokens toks{slot_, pos_, T}, toks_s{slot_, pos_, T, ranged ? lo_ : nullptr, ranged ? hi_ : nullptr};
  // On a profiling queue each mark is an empty kernel whose end timestamp closes the section, so the card's own time is
  // measured; otherwise the host waits at each mark.
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
  // The layer's input and output alternate between the two residual buffers.
  float *x = xa_, *x2 = xb_;
  // The rows come from the GGUF's mapping (no table on the card), through pinned memory.
  for (int64_t t = 0; t < T; ++t)
    std::memcpy(emb_host_ + uint64_t(t) * embd_row_bytes_, embd_->data + uint64_t(tok[size_t(t)]) * embd_row_bytes_, embd_row_bytes_);
  q_.memcpy(emb_dev_, emb_host_, size_t(uint64_t(T) * embd_row_bytes_));
  ref::embed(q_, ref::Weight{embd_->type, E, T, emb_dev_, embd_row_bytes_}, iota_, T, std::sqrt(float(E)), x);
  // Image rows replace the lookups, unscaled.
  for (const auto& [row, im] : spans) q_.memcpy(x + row * E, im->rows, size_t(im->n * E) * sizeof(float));
  emit("inp_scaled", x, {E, T});
  // The per-layer inputs from the embeddings (F16 weights: the fp16 GEMM whatever the rows) and the token table.
  act_.k = E;
  xmx::quantize(q_, x, T, act_, true);
  xmx::matmul(q_, act_, pl_proj_, pl_);
  fast::ple_inputs(q_, pl_, pl_table_, tok_, T, NL, P, pl_norm_, cfg_.eps, 1.0f / std::sqrt(float(E)), std::sqrt(float(P)),
                   float(1.0 / std::sqrt(2.0)));
  emit("inp_per_layer", pl_, {P, NL, T});
  // Each layer starts with its attention input already normed and quantized in act_: by the fused kernel at the end of
  // the layer before, or here for the first.
  fast::residual_norm_quant(q_, nullptr, nullptr, x, 1.0f, nullptr, layers_[0].attn_norm, cfg_.eps, T, E, &act_, h_);
  mark("embed");
  for (int l = 0; l < NL; ++l) {
    const Layer& L = layers_[size_t(l)];
    const std::string s = "-" + std::to_string(l);
    const bool sliding = cfg_.sliding[size_t(l)];
    const int64_t D = cfg_.head_dim(l), Hkv = cfg_.n_head_kv[size_t(l)];
    const float* inv = sliding ? inv_freq_sliding_ : inv_freq_global_;
    emit("attn_norm" + s, h_, {E, T});
    // Each part of the projection into its own [T][n] block of qkv_; Q, K and V are found by their columns.
    const float* src[3] = {nullptr, nullptr, nullptr};
    int64_t ld[3] = {0, 0, 0};
    {
      const int64_t widths[3] = {H * D, Hkv * D, Hkv * D};
      const int n_mat = L.writes_kv ? (L.has_v ? 3 : 2) : 1;
      int64_t off = 0;
      int which = 0;
      for (const xmx::Weight& w : L.qkv) {
        xmx::matmul(q_, act_, w, qkv_ + off);
        for (int64_t c = 0; c < w.n && which < n_mat; c += widths[which], ++which) {
          src[which] = qkv_ + off + c;
          ld[which] = w.n;
        }
        off += T * w.n;
      }
    }
    mark("qkv");
    fast::qkv_heads(q_, src[0], ld[0], src[1], ld[1], src[2], ld[2], T, H, Hkv, D, L.q_norm, L.k_norm, cfg_.eps, pos_, inv, qb_, kb_,
                    vb_);
    emit("Qcur_pos" + s, qb_, {D, H, T});
    if (L.writes_kv) {
      emit("Kcur_pos" + s, kb_, {D, Hkv, T});
      emit("Vcur_normed" + s, vb_, {D, Hkv, T});
    }
    mark("qkv_heads");
    if (sliding) {
      const attn::Cache cache{ring_k_ + L.ring_index * ring_layer_stride_, ring_v_ + L.ring_index * ring_layer_stride_, nullptr,
                              ring_pages_, cfg_.window, Hkv, D};
      if (L.writes_kv) attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks_s, tiles_s, tiles_sliding_, cache, H, 1.0f, att_, attn_scratch_);
      mark("attn_sliding");
    } else {
      const GlobalCache& g = global_[size_t(L.page_index)];
      const attn::Cache cache{g.k, g.v, page_table_, max_pages_, 0, Hkv, D, g.ks, g.vs};
      if (L.writes_kv) attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_g, tiles_global_, cache, H, 1.0f, att_, attn_scratch_);
      mark("attn_global");
    }
    emit("kqv_out" + s, att_, {H * D, T});
    act_.k = H * D;
    xmx::quantize(q_, att_, T, act_);
    xmx::matmul(q_, act_, L.wo, o_);
    mark("wo");
    fast::residual_norm_quant(q_, o_, L.attn_post_norm, x, 1.0f, x2, L.ffn_norm, cfg_.eps, T, E, &act_, h_);
    emit("attn_out" + s, x2, {E, T});
    mark("post_attn");
    xmx::matmul_geglu(q_, act_, L.gate_up, act_ffn_, gu_);
    mark("gate_up");
    xmx::matmul(q_, act_ffn_, L.down, o_);
    mark("down");
    fast::residual_norm_quant(q_, o_, L.ffn_post_norm, x2, 1.0f, x, nullptr, cfg_.eps, T, E, nullptr, nullptr);
    mark("post_ffn");
    // The per-layer input: gelu(inp_gate . x) * P[l], projected back, normed and added; then the layer's scale, and the
    // next layer's attention norm.
    act_.k = E;
    xmx::quantize(q_, x, T, act_);
    xmx::matmul(q_, act_, L.inp_gate, pg_);
    fast::ple_gate(q_, pg_, pl_, l, NL, T, P, act_pl_);
    xmx::matmul(q_, act_pl_, L.proj, o_);
    emit("per_layer_out" + s, o_, {E, T});
    const float* next_norm = l + 1 < NL ? layers_[size_t(l) + 1].attn_norm : nullptr;
    fast::residual_norm_quant(q_, o_, L.post_norm, x, L.out_scale, x2, next_norm, cfg_.eps, T, E, &act_, h_);
    std::swap(x, x2);
    emit("l_out" + s, x, {E, T});
    if (cap_col_[size_t(l)] >= 0 && !cap_rows.empty())
      qwenk::gather_rows(q_, x, cap_idx_, int64_t(cap_rows.size()), E, cap_, int64_t(ncap) * E, int64_t(cap_col_[size_t(l)]) * E);
    mark("per_layer");
  }
  for (const auto& c : chunks) slots_[size_t(c.slot)].length += int64_t(c.tokens.size());
  if (R == 0) {
    enqueue_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
    q_.wait();
    settle();
    return {};
  }
  // Logits only for the rows asked for: gather them, final norm, tied head, softcap.
  for (int64_t r = 0; r < R; ++r) q_.memcpy(rows_ + r * E, x + int64_t(logit_rows[size_t(r)]) * E, size_t(E) * sizeof(float));
  fast::rmsnorm(q_, rows_, output_norm_, R, E, cfg_.eps, rows_);
  emit("result_norm", rows_, {E, R});
  act_head_.k = E;
  xmx::quantize(q_, rows_, R, act_head_);
  xmx::matmul(q_, act_head_, head_w_, logits_);
  if (cfg_.softcap > 0) fast::softcap(q_, logits_, cfg_.softcap, R * V);
  mark("head");
  std::vector<float> out(size_t(R * V));
  auto done = q_.memcpy(out.data(), logits_, out.size() * sizeof(float));
  enqueue_s_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_step).count();
  done.wait();
  settle();
  return out;
}

}  // namespace eklow::engine::gemma4
