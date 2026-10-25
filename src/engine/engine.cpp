#include "engine/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
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

  // Q4_0 matrices go through one staging buffer and are repacked for the matrix units.
  uint64_t staging_bytes = 0;
  for (const auto& t : f.tensors())
    if (t.type == gguf::Type::Q4_0) staging_bytes = std::max(staging_bytes, t.nbytes);
  auto* staging = sycl::malloc_device<uint8_t>(staging_bytes, q_);
  if (!staging) throw std::runtime_error("engine: out of device memory for weight staging");
  auto matrix = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    if (t.type != gguf::Type::Q4_0) throw std::runtime_error(f.path() + ": " + name + " is not Q4_0");
    q_.memcpy(staging, t.data, t.nbytes).wait();
    ref::Weight raw{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), staging, t.nbytes / t.ne[1]};
    xmx::Weight w = xmx::repack_q4_0(q_, raw);
    owned_.push_back(w.tiles);
    owned_.push_back(w.scales);
    weight_bytes_ += t.nbytes;
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
    L.wq = matrix(b + "attn_q.weight");
    L.wk = matrix(b + "attn_k.weight");
    if (f.tensor(b + "attn_v.weight")) L.wv = matrix(b + "attn_v.weight");
    L.wo = matrix(b + "attn_output.weight");
    L.q_norm = vec(b + "attn_q_norm.weight");
    L.k_norm = vec(b + "attn_k_norm.weight");
    L.attn_post_norm = vec(b + "post_attention_norm.weight");
    L.ffn_norm = vec(b + "ffn_norm.weight");
    L.gate = matrix(b + "ffn_gate.weight");
    L.up = matrix(b + "ffn_up.weight");
    L.down = matrix(b + "ffn_down.weight");
    L.ffn_post_norm = vec(b + "post_ffw_norm.weight");
    const gguf::Tensor* os = f.tensor(b + "layer_output_scale.weight");
    L.out_scale = os ? *reinterpret_cast<const float*>(os->data) : 1.0f;
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
  page_layer_stride_ = n_pages_ * hkv_global_ * attn::kPage * cfg_.head_dim_global;
  const size_t page_elems = size_t(page_layer_stride_ * n_page);
  page_k_ = sycl::malloc_device<sycl::half>(page_elems, q_);
  page_v_ = sycl::malloc_device<sycl::half>(page_elems, q_);
  page_table_ = sycl::malloc_device<int32_t>(size_t(c_.n_slots * max_pages_), q_);
  if (!ring_k_ || !ring_v_ || !page_k_ || !page_v_ || !page_table_) throw std::runtime_error("engine: out of device memory for the KV cache");
  for (void* p : {static_cast<void*>(ring_k_), static_cast<void*>(ring_v_), static_cast<void*>(page_k_),
                  static_cast<void*>(page_v_), static_cast<void*>(page_table_)})
    owned_.push_back(p);
  cache_bytes_ = (ring_elems + page_elems) * 2 * sizeof(sycl::half);
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
  gate_ = dev(B * F);
  up_ = dev(B * F);
  rows_ = dev(c_.max_logits * E);
  logits_ = dev(c_.max_logits * cfg_.n_vocab);
  // At most one chunk per slot in a step, so at most n_slots partial tiles per K/V head.
  max_tiles_sliding_ = hkv_sliding_ * ((B * group_sliding_ + 7) / 8 + c_.n_slots);
  max_tiles_global_ = hkv_global_ * ((B * group_global_ + 7) / 8 + c_.n_slots);
  for (auto [p, n] : {std::pair{&tok_, B}, {&slot_, B}, {&pos_, B}, {&tiles_sliding_, 4 * max_tiles_sliding_},
                      {&tiles_global_, 4 * max_tiles_global_}}) {
    *p = sycl::malloc_device<int32_t>(size_t(n), q_);
    if (!*p) throw std::runtime_error("engine: out of device memory");
    owned_.push_back(*p);
  }
  act_ = xmx::alloc_activations(q_, B, std::max<int64_t>({E, F, H * Dg}));
}

Engine::~Engine() {
  q_.wait();
  xmx::free_activations(q_, act_);
  if (attn_scratch_.q) sycl::free(attn_scratch_.q, q_);
  if (attn_scratch_.part) sycl::free(attn_scratch_.part, q_);
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

void Engine::release(int slot) {
  Slot& s = slots_.at(size_t(slot));
  for (int32_t p : s.pages) free_pages_.push_back(p);
  s.pages.clear();
  s.length = 0;
}

std::vector<float> Engine::step(const std::vector<Chunk>& chunks, const model::DumpSink* dump) {
  const int64_t E = cfg_.n_embd, F = cfg_.n_ff, H = cfg_.n_head, V = cfg_.n_vocab;
  std::vector<int32_t> tok, slot, pos, logit_rows;
  attn::TileList tiles_s, tiles_g;
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
  q_.memcpy(tiles_sliding_, tiles_s.desc.data(), tiles_s.desc.size() * 4);
  q_.memcpy(tiles_global_, tiles_g.desc.data(), tiles_g.desc.size() * 4);
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
  auto mm = [&](const float* x, int64_t k, const xmx::Weight& w, float* out) {
    act_.k = k;
    xmx::quantize(q_, x, T, act_);
    xmx::matmul(q_, act_, w, out);
  };

  fast::embed(q_, head_, tok_, T, std::sqrt(float(E)), x_);
  emit("inp_scaled", x_, {E, T});
  mark("embed");
  for (int l = 0; l < cfg_.n_layer; ++l) {
    const Layer& L = layers_[size_t(l)];
    const std::string s = "-" + std::to_string(l);
    const bool sliding = cfg_.sliding[size_t(l)];
    const int64_t D = cfg_.head_dim(l), Hkv = cfg_.n_head_kv[size_t(l)];
    const float* inv = sliding ? inv_freq_sliding_ : inv_freq_global_;

    fast::rmsnorm(q_, x_, L.attn_norm, T, E, cfg_.eps, h_);
    emit("attn_norm" + s, h_, {E, T});
    mark("norms");
    act_.k = E;
    xmx::quantize(q_, h_, T, act_);
    xmx::matmul(q_, act_, L.wq, qb_);
    xmx::matmul(q_, act_, L.wk, kb_);
    if (L.wv.tiles) xmx::matmul(q_, act_, L.wv, vb_);
    else q_.memcpy(vb_, kb_, size_t(T * Hkv * D) * sizeof(float));
    mark("qkv");
    fast::rmsnorm(q_, qb_, L.q_norm, T * H, D, cfg_.eps, qb_);
    fast::rope_neox(q_, qb_, pos_, T, H, D, D, inv);
    emit("Qcur_pos" + s, qb_, {D, H, T});
    fast::rmsnorm(q_, kb_, L.k_norm, T * Hkv, D, cfg_.eps, kb_);
    fast::rope_neox(q_, kb_, pos_, T, Hkv, D, D, inv);
    emit("Kcur_pos" + s, kb_, {D, Hkv, T});
    fast::rmsnorm(q_, vb_, nullptr, T * Hkv, D, cfg_.eps, vb_);
    emit("Vcur_normed" + s, vb_, {D, Hkv, T});
    mark("qkv_norm_rope");

    if (sliding) {
      const attn::Cache cache{ring_k_ + L.ring_index * ring_layer_stride_, ring_v_ + L.ring_index * ring_layer_stride_, nullptr,
                              ring_pages_, cfg_.window, Hkv, D};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_sliding_, tiles_s.size(), tiles_s.max_keys, cache, H, 1.0f, att_, attn_scratch_);
      mark("attn_sliding");
    } else {
      const attn::Cache cache{page_k_ + L.page_index * page_layer_stride_, page_v_ + L.page_index * page_layer_stride_,
                              page_table_, max_pages_, 0, Hkv, D};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_global_, tiles_g.size(), tiles_g.max_keys, cache, H, 1.0f, att_, attn_scratch_);
      mark("attn_global");
    }
    emit("kqv_out" + s, att_, {H * D, T});
    mm(att_, H * D, L.wo, o_);
    mark("wo");
    fast::rmsnorm(q_, o_, L.attn_post_norm, T, E, cfg_.eps, o_);
    fast::add_scale(q_, o_, x_, 1.0f, T * E, x2_);
    emit("attn_out" + s, x2_, {E, T});
    mark("norms");

    fast::rmsnorm(q_, x2_, L.ffn_norm, T, E, cfg_.eps, h_);
    mark("norms");
    act_.k = E;
    xmx::quantize(q_, h_, T, act_);
    xmx::matmul(q_, act_, L.gate, gate_);
    xmx::matmul(q_, act_, L.up, up_);
    mark("gate_up");
    fast::gelu_mul(q_, gate_, up_, T * F, gate_);
    mark("gelu");
    mm(gate_, F, L.down, o_);
    mark("down");
    fast::rmsnorm(q_, o_, L.ffn_post_norm, T, E, cfg_.eps, o_);
    fast::add_scale(q_, o_, x2_, L.out_scale, T * E, x_);
    emit("l_out" + s, x_, {E, T});
    mark("norms");
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
