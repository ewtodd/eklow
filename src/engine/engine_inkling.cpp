#include "engine/engine_inkling.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>

#include "kernels/fast.hpp"
#include "kernels/gemma_fast.hpp"
#include "kernels/inkling_fast.hpp"
#include "kernels/ref_inkling.hpp"

namespace eklow::engine::inkling {

namespace ik = eklow::inkling;

float* Engine::dev(int64_t n) {
  auto* p = sycl::malloc_device<float>(size_t(std::max<int64_t>(n, 1)), q_);
  if (!p) throw std::runtime_error("inkling engine: out of device memory");
  owned_.push_back(p);
  return p;
}

namespace {

// A matrix of any float type (F32, BF16, F16) as fp32 rows, on the host.
std::vector<float> as_f32(const gguf::Tensor& t) {
  const size_t n = size_t(t.elements());
  std::vector<float> out(n);
  if (t.type == gguf::Type::F32) {
    std::memcpy(out.data(), t.data, n * sizeof(float));
  } else if (t.type == gguf::Type::BF16) {
    for (size_t i = 0; i < n; ++i) {
      const uint32_t b = uint32_t(t.data[2 * i] | t.data[2 * i + 1] << 8) << 16;
      std::memcpy(&out[i], &b, 4);
    }
  } else {
    throw std::runtime_error(t.name + ": expected F32 or BF16");
  }
  return out;
}

}  // namespace

Engine::Engine(sycl::queue& q, const gguf::File& f, const Config& cfg) : q_(q), c_(cfg), cfg_(model::Inkling2Config::from_gguf(f)) {
  if (c_.max_ctx % attn::kPage) throw std::runtime_error("inkling engine: max_ctx must be a multiple of the page size");
  if (c_.max_chunk > c_.max_batch) throw std::runtime_error("inkling engine: max_chunk above max_batch");
  if (c_.n_slots < 1 || c_.n_slots > 8) throw std::runtime_error("inkling engine: 1 to 8 slots");
  // step() keeps the core's block outputs in fixed buffers: a core of two blocks
  if (cfg_.n_core() != 2 * cfg_.block) throw std::runtime_error("inkling engine: a loop core other than two blocks");
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim, NR = cfg_.n_rel;
  const int64_t HD = H * D, KD = HK * D, NL = cfg_.n_layer, NA = cfg_.n_slot(), CC = cfg_.conv_channels();
  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto up_f32 = [&](const std::vector<float>& h) {
    float* d = dev(int64_t(h.size()));
    q_.memcpy(d, h.data(), h.size() * sizeof(float)).wait();
    weight_bytes_ += h.size() * sizeof(float);
    return static_cast<const float*>(d);
  };
  auto vec = [&](const std::string& name) { return up_f32(as_f32(need(name))); };

  // Matrices go through one staging buffer and are repacked for the matrix units (Q8_0 as Q8_0I); Q, R and the gate read
  // the same input and are stacked.
  uint64_t staging_bytes = need("output.weight").nbytes;
  for (int l = 0; l < NL; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    staging_bytes = std::max(staging_bytes, need(b + "attn_q.weight").nbytes + need(b + "attn_rel.weight").nbytes + need(b + "attn_gate.weight").nbytes);
  }
  auto* staging = sycl::malloc_device<uint8_t>(staging_bytes, q_);
  if (!staging) throw std::runtime_error("inkling engine: out of device memory for weight staging");
  auto matrix = [&](const std::vector<const gguf::Tensor*>& parts) {
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
    if (!xmx::format_of(raw.type, fmt)) throw std::runtime_error(f.path() + ": " + parts[0]->name + " is not a matrix-unit format");
    xmx::Weight w = xmx::repack_format(q_, raw, fmt);
    q_.wait();
    owned_.push_back(w.tiles);
    if (w.scales) owned_.push_back(w.scales);
    return w;
  };

  {
    const gguf::Tensor& out = need("output.weight");
    head_w_ = matrix({&out});
    act_head_ = xmx::alloc_activations(q_, xmx::kFp16Rows, E);
    const gguf::Tensor& emb = need("token_embd.weight");
    embd_ = &emb;
    embd_row_bytes_ = emb.nbytes / emb.ne[1];
    const int64_t Bm = c_.max_batch;
    emb_host_ = sycl::malloc_host<uint8_t>(size_t(Bm) * embd_row_bytes_, q_);
    emb_dev_ = sycl::malloc_device<uint8_t>(size_t(Bm) * embd_row_bytes_, q_);
    iota_ = sycl::malloc_device<int32_t>(size_t(Bm), q_);
    if (!emb_host_ || !emb_dev_ || !iota_) throw std::runtime_error("inkling engine: out of memory for the embedding rows");
    owned_.push_back(emb_host_);
    owned_.push_back(emb_dev_);
    owned_.push_back(iota_);
    std::vector<int32_t> iota(static_cast<size_t>(Bm));
    for (int64_t i = 0; i < Bm; ++i) iota[size_t(i)] = int32_t(i);
    q_.memcpy(iota_, iota.data(), iota.size() * 4).wait();
  }
  tok_norm_ = vec("token_embd_norm.weight");
  output_norm_ = vec("output_norm.weight");
  output_res_query_ = vec("output_res_query.weight");

  std::vector<bool> staged(size_t(NL), false);
  for (int l : c_.staged_expert_layers) {
    if (l < 0 || l >= NL) throw std::runtime_error("inkling engine: staged expert layer out of range");
    staged[size_t(l)] = true;
  }
  uint64_t staged_max = 0;
  for (int l = 0; l < NL; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    const bool swa = cfg_.sliding[size_t(l)];
    Layer L;
    L.qrg = matrix({&need(b + "attn_q.weight"), &need(b + "attn_rel.weight"), &need(b + "attn_gate.weight")});
    L.k = matrix({&need(b + "attn_k.weight")});
    L.v = matrix({&need(b + "attn_v.weight")});
    L.o = matrix({&need(b + "attn_output.weight")});
    L.sink = vec(b + "attn_sink_proj.weight");
    L.router = vec(b + "ffn_gate_inp.weight");
    L.q_clip = vec(b + "attn_q_clip.weight");
    L.rel_bias = vec(b + "attn_rel_bias.weight");
    {
      const int64_t extent = swa ? cfg_.window : cfg_.rel_extent;
      auto* t = sycl::malloc_device<sycl::half>(attn::bias_table_halves(extent), q_);
      if (!t) throw std::runtime_error("inkling engine: out of device memory for a bias table");
      owned_.push_back(t);
      attn::bias_table(q_, L.rel_bias, extent, t);
      L.rel_tiles = t;
    }
    L.rel_clip = vec(b + "attn_rel_clip.weight");
    L.k_conv = vec(b + "attn_k_conv.weight");
    L.v_conv = vec(b + "attn_v_conv.weight");
    L.attn_post_conv = vec(b + "attn_post_conv.weight");
    L.ffn_post_conv = vec(b + "ffn_post_conv.weight");
    if (swa) {
      L.attn_norm = vec(b + "attn_norm.weight");
    } else {
      L.q_norm = vec(b + "attn_q_inp_norm.weight");
      L.k_norm = vec(b + "attn_k_inp_norm.weight");
      L.v_norm = vec(b + "attn_v_inp_norm.weight");
      L.k_dyn = vec(b + "attn_k_conv_proj.weight");
      L.v_dyn = vec(b + "attn_v_conv_proj.weight");
    }
    if (l > 0) L.attn_res_query = vec(b + "attn_res_query.weight");
    L.ffn_res_query = vec(b + "ffn_res_query.weight");
    L.ffn_norm = vec(b + "ffn_norm.weight");
    if (f.tensor(b + "exp_probs_b.bias")) L.probs_bias = vec(b + "exp_probs_b.bias");
    // The experts, from the GGUF's mapping (host memory).
    const gguf::Tensor& gu = need(b + "ffn_gate_up_exps.weight");
    const gguf::Tensor& dn = need(b + "ffn_down_exps.weight");
    const int64_t NE = cfg_.n_expert, F = cfg_.n_ff_exp;
    const uint64_t gs = gu.nbytes / uint64_t(NE), ds = dn.nbytes / uint64_t(NE);
    const gguf::Tensor* gsc = f.tensor(b + "ffn_gate_up_exps.scale");
    const gguf::Tensor* dsc = f.tensor(b + "ffn_down_exps.scale");
    L.experts = moe::upload(q_, ref::Weight{gu.type, E, 2 * F, gu.data, gs / uint64_t(2 * F)}, gs,
                            gsc ? reinterpret_cast<const float*>(gsc->data) : nullptr,
                            ref::Weight{dn.type, F, E, dn.data, ds / uint64_t(E)}, ds,
                            dsc ? reinterpret_cast<const float*>(dsc->data) : nullptr, NE);
    q_.wait();
    if (staged[size_t(l)]) {
      L.host_experts = sycl::malloc_host<uint8_t>(L.experts.bytes, q_);
      if (!L.host_experts) throw std::runtime_error("inkling engine: out of pinned host memory for staged experts");
      owned_.push_back(L.host_experts);
      q_.memcpy(L.host_experts, L.experts.base, L.experts.bytes).wait();
      sycl::free(L.experts.base, q_);
      host_expert_bytes_ += L.experts.bytes;
      staged_max = std::max(staged_max, L.experts.bytes);
    } else {
      weight_bytes_ += L.experts.bytes;
    }
    layers_.push_back(L);
  }
  sycl::free(staging, q_);
  if (staged_max) {
    staging_ = sycl::malloc_device<uint8_t>(staged_max, q_);
    if (!staging_) throw std::runtime_error("inkling engine: out of device memory for the expert staging region");
    owned_.push_back(staging_);
    // a staged layer's tables point into the staging region for good; its bytes are copied there before it runs
    for (auto& L : layers_)
      if (L.host_experts) moe::rebase(q_, L.experts, staging_);
    q_.wait();
  }
  for (int l = cfg_.core_start; l < cfg_.core_end; ++l) {
    static const char* kCtrl[4] = {"q", "k", "v", "ffn"};
    const std::string b = "blk." + std::to_string(l) + ".";
    Loop lp;
    lp.msg_query = vec(b + "loop_msg_query.weight");
    for (int k = 0; k < 4; ++k) {
      const std::string n = b + "loop_" + kCtrl[k] + "_ctrl_";
      lp.down[k] = vec(n + "down.weight");
      // gain and forget (k = gate_rank 64) on the fp16 GEMM, from the fp32 rows
      for (auto [dst, d32, name] : {std::tuple{&lp.gain[k], &lp.gain32[k], "gain.weight"}, {&lp.forget[k], &lp.forget32[k], "forget.weight"}}) {
        const float* w32 = *d32 = vec(n + name);
        *dst = xmx::repack(q_, ref::Weight{gguf::Type::F32, cfg_.gate_rank, E, reinterpret_cast<const uint8_t*>(w32),
                                            uint64_t(cfg_.gate_rank) * sizeof(float)});
        q_.wait();
        owned_.push_back(dst->tiles);
        if (dst->scales) owned_.push_back(dst->scales);
      }
    }
    loops_.push_back(lp);
  }

  // K/V: a ring per sliding application, pages per full one.
  int n_ring = 0, n_full = 0;
  for (int s = 0; s < NA; ++s) {
    const bool swa = cfg_.sliding[size_t(cfg_.phys(s))];
    ring_of_.push_back(swa ? n_ring++ : -1);
    page_of_.push_back(swa ? -1 : n_full++);
  }
  ring_pages_ = (cfg_.window + c_.max_chunk + attn::kPage - 1) / attn::kPage;
  ring_layer_stride_ = int64_t(c_.n_slots) * ring_pages_ * HK * attn::kPage * D;
  const size_t ring_elems = size_t(ring_layer_stride_ * n_ring);
  ring_k_ = sycl::malloc_device<sycl::half>(ring_elems, q_);
  ring_v_ = sycl::malloc_device<sycl::half>(ring_elems, q_);
  max_pages_ = c_.max_ctx / attn::kPage;
  n_pages_ = max_pages_ * c_.n_slots;
  page_table_ = sycl::malloc_device<int32_t>(size_t(c_.n_slots * max_pages_), q_);
  if (!ring_k_ || !ring_v_ || !page_table_) throw std::runtime_error("inkling engine: out of device memory for the K/V cache");
  for (void* p : {static_cast<void*>(ring_k_), static_cast<void*>(ring_v_), static_cast<void*>(page_table_)}) owned_.push_back(p);
  cache_bytes_ = ring_elems * 2 * sizeof(sycl::half);
  const size_t page_elems = size_t(n_pages_ * HK * attn::kPage * D);
  for (int l = 0; l < n_full; ++l) {
    auto* k = sycl::malloc_device<sycl::half>(page_elems, q_);
    auto* v = sycl::malloc_device<sycl::half>(page_elems, q_);
    if (!k || !v) throw std::runtime_error("inkling engine: out of device memory for the K/V cache");
    owned_.push_back(k);
    owned_.push_back(v);
    pages_kv_.push_back({k, v});
    cache_bytes_ += 2 * page_elems * sizeof(sycl::half);
  }
  page_table_host_.assign(size_t(c_.n_slots * max_pages_), 0);
  for (int64_t p = n_pages_ - 1; p >= 0; --p) free_pages_.push_back(int32_t(p));
  slots_.resize(size_t(c_.n_slots));
  conv_ = dev(int64_t(c_.n_slots) * NA * 3 * CC);
  q_.memset(conv_, 0, size_t(c_.n_slots * NA * 3 * CC) * sizeof(float)).wait();
  cache_bytes_ += size_t(c_.n_slots * NA * 3 * CC) * sizeof(float);

  // Workspace.
  const int64_t B = c_.max_batch, QRG = HD + H * NR + HD;
  for (int i = 0; i < 8; ++i) blk_.push_back(dev(B * E));
  mixq_ = dev(3 * B * E);
  msg_ = dev(4 * B * E);
  for (auto& h : h_) h = dev(B * E);
  qrg_ = dev(B * QRG);
  kr_ = dev(B * KD);
  vr_ = dev(B * KD);
  kd_ = dev(B * 4 * HK);
  vd_ = dev(B * 4 * HK);
  qb_ = dev(B * HD);
  rb_ = dev(B * H * NR);
  kb_ = dev(B * KD);
  vb_ = dev(B * KD);
  sk_ = dev(B * H);
  att_ = dev(B * HD);
  o_ = dev(B * E);
  oc_ = dev(B * E);
  f_ = dev(B * E);
  ci_ = dev(B * 2 * E);
  cd_ = dev(B * cfg_.gate_rank);
  cg_ = dev(B * E);
  cf_ = dev(B * E);
  lg_ = dev(B * cfg_.n_expert);
  wts_ = dev(B * cfg_.n_expert_used);
  rows_ = dev(int64_t(c_.max_logits) * E * 6);
  logits_ = dev(int64_t(c_.max_logits) * cfg_.n_vocab);
  max_tiles_sliding_ = HK * ((B * (H / HK) + 7) / 8 + c_.n_slots);
  max_tiles_full_ = max_tiles_sliding_;
  for (auto [p, n] : {std::pair{&ids_, B * cfg_.n_expert_used}, {&tok_, B}, {&slot_, B}, {&pos_, B}, {&base_, B},
                      {&lrows_, int64_t(c_.max_logits)}, {&tiles_sliding_, 6 * max_tiles_sliding_}, {&tiles_full_, 6 * max_tiles_full_}}) {
    *p = sycl::malloc_device<int32_t>(size_t(n), q_);
    if (!*p) throw std::runtime_error("inkling engine: out of device memory");
    owned_.push_back(*p);
  }
  const int64_t bias_extent = std::max(cfg_.window, cfg_.rel_extent);
  attn::reserve(q_, attn_scratch_, max_tiles_sliding_, c_.max_chunk * (H / HK), HK, D, H / HK, cfg_.window + c_.max_chunk, bias_extent);
  attn::reserve(q_, attn_scratch_, max_tiles_full_, c_.max_chunk * (H / HK), HK, D, H / HK, c_.max_ctx + attn::kPage, bias_extent);
  act_ = xmx::alloc_activations(q_, B, std::max<int64_t>(E, HD));
  moe_scratch_ = moe::reserve(q_, B, cfg_.n_expert_used, E, cfg_.n_ff_exp);
}

Engine::~Engine() {
  q_.wait();
  xmx::free_activations(q_, act_);
  xmx::free_activations(q_, act_head_);
  for (void* p : {static_cast<void*>(attn_scratch_.q), static_cast<void*>(attn_scratch_.part), static_cast<void*>(attn_scratch_.part2),
                  static_cast<void*>(attn_scratch_.p), static_cast<void*>(attn_scratch_.stat), static_cast<void*>(attn_scratch_.rt),
                  static_cast<void*>(attn_scratch_.tv)})
    if (p) sycl::free(p, q_);
  if (moe_scratch_) moe::free_scratch(q_, moe_scratch_);
  for (auto& L : layers_) {
    if (L.host_experts) L.experts.base = nullptr;  // the staging region, freed with owned_
    moe::free_layer(q_, L.experts);
  }
  for (void* p : owned_) sycl::free(p, q_);
}

void Engine::ensure_pages(int slot, int64_t positions) {
  Slot& s = slots_[size_t(slot)];
  const int64_t need = (positions + attn::kPage - 1) / attn::kPage;
  if (need > max_pages_) throw std::runtime_error("inkling engine: sequence longer than max_ctx");
  while (int64_t(s.pages.size()) < need) {
    if (free_pages_.empty()) throw std::runtime_error("inkling engine: out of K/V pages");
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
  const int64_t per = int64_t(cfg_.n_slot()) * 3 * cfg_.conv_channels();
  q_.memset(conv_ + int64_t(slot) * per, 0, size_t(per) * sizeof(float)).wait();
}

std::vector<float> Engine::step(const std::vector<Chunk>& chunks, const model::DumpSink* dump, const model::Teacher* teacher) {
  const int64_t E = cfg_.n_embd, H = cfg_.n_head, HK = cfg_.n_head_kv, D = cfg_.head_dim, NR = cfg_.n_rel;
  const int64_t HD = H * D, KD = HK * D, QRG = HD + H * NR + HD, CC = cfg_.conv_channels(), NA = cfg_.n_slot();
  const int64_t NU = cfg_.n_expert_used, V = cfg_.n_vocab, VO = cfg_.n_vocab_out;
  std::vector<int32_t> tok, slot, pos, base, logit_rows;
  ik::Chunks ch;
  attn::TileList tiles_s(D), tiles_f(D);
  for (const auto& c : chunks) {
    if (c.slot < 0 || c.slot >= c_.n_slots) throw std::runtime_error("inkling engine: bad slot");
    if (c.start_pos != slots_[size_t(c.slot)].length) throw std::runtime_error("inkling engine: chunk does not continue its slot");
    const int64_t b = int64_t(tok.size()), n = int64_t(c.tokens.size());
    if (n > c_.max_chunk) throw std::runtime_error("inkling engine: chunk longer than max_chunk");
    if (n == 0) continue;
    for (int k = 0; k < ch.count; ++k)
      if (ch.slot[k] == c.slot) throw std::runtime_error("inkling engine: two chunks of one slot in a step");
    ch.base[ch.count] = int32_t(b), ch.n[ch.count] = int32_t(n), ch.slot[ch.count] = c.slot, ++ch.count;
    for (int64_t i = 0; i < n; ++i) {
      tok.push_back(c.tokens[size_t(i)]);
      slot.push_back(c.slot);
      pos.push_back(int32_t(c.start_pos + i));
      base.push_back(int32_t(b));
    }
    tiles_s.add_chunk(int32_t(b), n, c.start_pos + n - 1, H / HK, HK, cfg_.window);
    tiles_f.add_chunk(int32_t(b), n, c.start_pos + n - 1, H / HK, HK, 0);
    for (int64_t r = std::max<int64_t>(0, n - c.logits_rows); r < n; ++r) logit_rows.push_back(int32_t(b + r));
    ensure_pages(c.slot, c.start_pos + n);
  }
  const int64_t T = int64_t(tok.size()), R = int64_t(logit_rows.size());
  if (T == 0) return {};
  if (T > c_.max_batch) throw std::runtime_error("inkling engine: step larger than max_batch");
  if (R > c_.max_logits) throw std::runtime_error("inkling engine: too many logits rows");
  q_.memcpy(tok_, tok.data(), size_t(T) * 4);
  q_.memcpy(slot_, slot.data(), size_t(T) * 4);
  q_.memcpy(pos_, pos.data(), size_t(T) * 4);
  q_.memcpy(base_, base.data(), size_t(T) * 4);
  if (R) q_.memcpy(lrows_, logit_rows.data(), size_t(R) * 4);
  const std::vector<int32_t> packed_s = tiles_s.packed(), packed_f = tiles_f.packed();
  q_.memcpy(tiles_sliding_, packed_s.data(), packed_s.size() * 4);
  q_.memcpy(tiles_full_, packed_f.data(), packed_f.size() * 4);
  q_.memcpy(page_table_, page_table_host_.data(), page_table_host_.size() * 4);
  const attn::Tokens toks{slot_, pos_, T};

  auto t_mark = std::chrono::steady_clock::now();
  auto mark = [&](const char* name) {
    if (!profile_) return;
    q_.wait();
    const auto now = std::chrono::steady_clock::now();
    (*profile_)[name] += std::chrono::duration<double>(now - t_mark).count();
    t_mark = now;
  };
  auto emit = [&](const std::string& name, const float* p, std::vector<int64_t> ne) {
    if (!dump || !dump->wants(name)) return;
    int64_t n = 1;
    for (auto e : ne) n *= e;
    std::vector<float> host(static_cast<size_t>(n));
    q_.memcpy(host.data(), p, size_t(n) * sizeof(float)).wait();
    dump->put(name, host, ne);
  };
  // fp32 matrices of few rows (the router, the conv taps, the controllers): a sub-group an output for steps of few tokens,
  // the shared tiled kernel (n a multiple of 32, at most 256) for longer ones.
  constexpr int64_t kFewTokens = 16;
  auto mm32 = [&](const float* w, int64_t n, int64_t k, const float* x, float* out) {
    if (T <= kFewTokens || n % 32 || n > 256) ik::matmul_f32(q_, w, n, k, x, k, T, out, n);
    else fast::matmul_f32(q_, w, n, k, x, T, out);
  };
  auto mm = [&](const xmx::Weight& w, const float* x, int64_t k, float* out) {
    act_.k = k;
    xmx::quantize(q_, x, T, act_);
    xmx::matmul(q_, act_, w, out);
  };

  // The embeddings: rows from the GGUF's mapping through pinned memory, then the token norm.
  float* const e0 = blk_[0];
  for (int64_t t = 0; t < T; ++t)
    std::memcpy(emb_host_ + uint64_t(t) * embd_row_bytes_, embd_->data + uint64_t(tok[size_t(t)]) * embd_row_bytes_, embd_row_bytes_);
  q_.memcpy(emb_dev_, emb_host_, size_t(uint64_t(T) * embd_row_bytes_));
  ref::embed(q_, ref::Weight{embd_->type, E, T, emb_dev_, embd_row_bytes_}, iota_, T, 1.0f, e0);
  fast::rmsnorm(q_, e0, tok_norm_, T, E, cfg_.eps, e0);
  mark("embed");

  // Buffers: e0 (block 0), the prelude's output (block 1), the core's layer-11 and layer-17 outputs of the current and
  // the previous pass, and two for the other layers' streams.
  float* const prelude = blk_[1];
  float* const o11[2] = {blk_[2], blk_[3]};
  float* const o17[2] = {blk_[4], blk_[5]};
  float* const xs[2] = {blk_[6], blk_[7]};
  auto next_of = [&](const float* x) { return x == xs[0] ? xs[1] : xs[0]; };

  // One layer application, as the reference's Forward::layer.
  auto layer = [&](int app, int ip, const float* x, std::vector<const float*>& blocks, const float* msgs, const Loop* lp, float* out) {
    const Layer& L = layers_[size_t(ip)];
    const bool swa = cfg_.sliding[size_t(ip)];
    const std::string sl = "-" + std::to_string(app);
    ik::States st;
    auto states = [&](const float* cur) {
      st.n = 0;
      for (const float* b : blocks) st.p[st.n++] = b;
      st.p[st.n++] = cur;
    };
    // The attention's inputs, normed into h_[0..2] (h_[0] alone on a sliding layer without messages).
    const float* in[3] = {x, x, x};
    if (L.attn_res_query) {
      states(x);
      ik::depth_mix(q_, st, L.attn_res_query, swa ? 1 : 3, T, E, 1e-6f, mixq_);
      for (int j = 0; j < 3; ++j) in[j] = mixq_ + (swa ? 0 : j) * T * E;
    }
    auto control = [&](const float* cur, int k, float* dst) {
      ik::control_input(q_, cur, msgs + int64_t(k) * T * E, T, E, 1e-6f, ci_);
      mm32(lp->down[k], cfg_.gate_rank, 2 * E, ci_, cd_);
      if (T <= kFewTokens) {
        ik::matmul_f32(q_, lp->gain32[k], E, cfg_.gate_rank, cd_, cfg_.gate_rank, T, cg_, E);
        ik::matmul_f32(q_, lp->forget32[k], E, cfg_.gate_rank, cd_, cfg_.gate_rank, T, cf_, E);
      } else {
        act_.k = cfg_.gate_rank;
        xmx::quantize(q_, cd_, T, act_, true);
        xmx::matmul(q_, act_, lp->gain[k], cg_);
        xmx::matmul(q_, act_, lp->forget[k], cf_);
      }
      ref::inkling::control_mix(q_, cur, msgs + int64_t(k) * T * E, cg_, cf_, T * E, dst);
    };
    const float* hin[3];
    if (msgs) {
      for (int j = 0; j < 3; ++j) {
        control(in[j], j, h_[j]);
        fast::rmsnorm(q_, h_[j], swa ? L.attn_norm : (j == 0 ? L.q_norm : j == 1 ? L.k_norm : L.v_norm), T, E, cfg_.eps, h_[j]);
        hin[j] = h_[j];
      }
    } else if (swa) {
      fast::rmsnorm(q_, in[0], L.attn_norm, T, E, cfg_.eps, h_[0]);
      hin[0] = hin[1] = hin[2] = h_[0];
    } else {
      fast::rmsnorm(q_, in[0], L.q_norm, T, E, cfg_.eps, h_[0]);
      fast::rmsnorm(q_, in[1], L.k_norm, T, E, cfg_.eps, h_[1]);
      fast::rmsnorm(q_, in[2], L.v_norm, T, E, cfg_.eps, h_[2]);
      for (int j = 0; j < 3; ++j) hin[j] = h_[j];
    }
    mark("inputs");
    // Projections; K and V through their convolutions (full layers: with taps from their inputs), then the heads.
    mm(L.qrg, hin[0], E, qrg_);
    mm(L.k, hin[1], E, kr_);
    mm(L.v, hin[2], E, vr_);
    mm32(L.sink, H, E, hin[0], sk_);
    if (!swa) {
      mm32(L.k_dyn, 4 * HK, E, hin[1], kd_);
      mm32(L.v_dyn, 4 * HK, E, hin[2], vd_);
    }
    mark("proj");
    float* hist = conv_ + int64_t(app) * 3 * CC;
    const int64_t sstride = NA * 3 * CC;
    ik::short_conv(q_, kr_, KD, T, KD, ch, slot_, base_, hist, sstride, CC, L.k_conv, swa ? nullptr : kd_, HK, kb_);
    ik::short_conv(q_, vr_, KD, T, KD, ch, slot_, base_, hist + KD, sstride, CC, L.v_conv, swa ? nullptr : vd_, HK, vb_);
    ik::heads(q_, qrg_, QRG, kb_, vb_, pos_, T, H, HK, D, NR, L.q_clip, L.rel_clip, cfg_.qk_clip, cfg_.rel_clip, !swa, cfg_.log_alpha,
              float(cfg_.log_floor), cfg_.eps, qb_, rb_, kb_, vb_);
    emit("Qcur" + sl, qb_, {D, H, T});
    emit("Kcur" + sl, kb_, {D, HK, T});
    emit("Vcur" + sl, vb_, {D, HK, T});
    emit("Rcur" + sl, rb_, {NR, H, T});
    mark("heads");
    // Attention over the cache, the gate, the output projection and its convolution.
    const attn::Bias bias{rb_, L.rel_bias, NR, swa ? cfg_.window : cfg_.rel_extent, sk_, L.rel_tiles};
    const float scale = 1.0f / std::sqrt(float(D));
    if (swa) {
      const int r = ring_of_[size_t(app)];
      const attn::Cache cache{ring_k_ + r * ring_layer_stride_, ring_v_ + r * ring_layer_stride_, nullptr, ring_pages_, cfg_.window, HK, D};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_s, tiles_sliding_, cache, H, scale, bias, att_, attn_scratch_);
    } else {
      const auto& kv = pages_kv_[size_t(page_of_[size_t(app)])];
      const attn::Cache cache{kv.first, kv.second, page_table_, max_pages_, 0, HK, D};
      attn::write(q_, kb_, vb_, toks, cache);
      attn::attend(q_, qb_, toks, tiles_f, tiles_full_, cache, H, scale, bias, att_, attn_scratch_);
    }
    emit("kqv_out" + sl, att_, {HD, T});
    mark(swa ? "attn_sliding" : "attn_full");
    ik::gate(q_, att_, qrg_ + HD + H * NR, QRG, T, HD);
    mm(L.o, att_, HD, o_);
    ik::short_conv(q_, o_, E, T, E, ch, slot_, base_, hist + 2 * KD, sstride, CC, L.attn_post_conv, nullptr, 0, oc_);
    mark("o");
    // A block's first layer starts a new stream; the others add to theirs. The stream goes to `out`.
    const float* cur;
    if (ip % cfg_.block == 0) {
      blocks.push_back(x);
      cur = oc_;
    } else {
      emit("attn_out" + sl, oc_, {E, T});
      ref::add(q_, x, oc_, T * E, out);
      cur = out;
    }
    emit("ffn_inp" + sl, cur, {E, T});
    // The MoE.
    states(cur);
    ik::depth_mix(q_, st, L.ffn_res_query, 1, T, E, 1e-6f, mixq_);
    if (msgs) {
      control(mixq_, 3, h_[0]);
      fast::rmsnorm(q_, h_[0], L.ffn_norm, T, E, cfg_.eps, h_[0]);
    } else {
      fast::rmsnorm(q_, mixq_, L.ffn_norm, T, E, cfg_.eps, h_[0]);
    }
    mm32(L.router, cfg_.n_expert, E, h_[0], lg_);
    ref::inkling::route(q_, lg_, L.probs_bias, T, cfg_.n_expert, NU, cfg_.w_scale, ids_, wts_);
    emit("ffn_moe_logits" + sl, lg_, {cfg_.n_expert, T});
    if (dump && dump->wants("ffn_moe_topk" + sl)) {
      std::vector<int32_t> h(size_t(T * NU));
      q_.memcpy(h.data(), ids_, h.size() * 4).wait();
      dump->put("ffn_moe_topk" + sl, std::vector<float>(h.begin(), h.end()), {NU, T});
    }
    mark("router");
    if (L.host_experts && staged_layer_ != ip) {
      q_.memcpy(staging_, L.host_experts, L.experts.bytes);
      staged_layer_ = ip;
      mark("stage");
    }
    moe::ffn(q_, L.experts, h_[0], T, ids_, wts_, NU, cfg_.situ_beta, cfg_.situ_linear_beta, *moe_scratch_, f_);
    emit("ffn_moe_out" + sl, f_, {E, T});
    mark("moe");
    ik::short_conv(q_, f_, E, T, E, ch, slot_, base_, hist + 2 * KD + E, sstride, CC, L.ffn_post_conv, nullptr, 0, o_);
    emit("ffn_out" + sl, o_, {E, T});
    ref::add(q_, cur, o_, T * E, out);
    if (teacher) {
      std::vector<float> forced;
      if ((*teacher)("l_out" + sl, forced) && int64_t(forced.size()) == T * E)
        q_.memcpy(out, forced.data(), forced.size() * sizeof(float)).wait();
    }
    emit("l_out" + sl, out, {E, T});
    mark("ffn_out");
  };

  std::vector<const float*> blocks;
  const float* x = e0;
  for (int l = 0; l < cfg_.core_start; ++l) {
    float* out = l == cfg_.core_start - 1 ? prelude : next_of(x);
    layer(l, l, x, blocks, nullptr, nullptr, out);
    x = out;
  }
  const std::vector<const float*> prelude_blocks = blocks;
  for (int pass = 0; pass < cfg_.n_passes; ++pass) {
    std::vector<const float*> bank;
    const float* bank_out = nullptr;
    if (pass > 0) {
      bank.assign(blocks.begin() + std::ptrdiff_t(prelude_blocks.size()) + 1, blocks.end());
      bank_out = x;
      x = prelude;
      blocks = prelude_blocks;
    }
    for (int l = cfg_.core_start; l < cfg_.core_end; ++l) {
      const int app = cfg_.slot(l, pass);
      // the core's two blocks end at layers 11 and 17: their outputs are a later block's input and the next pass's bank
      float* out = l == cfg_.core_end - 1 ? o17[pass % 2] : l == cfg_.core_start + cfg_.block - 1 ? o11[pass % 2] : next_of(x);
      if (pass == 0) {
        layer(app, l, x, blocks, nullptr, nullptr, out);
      } else {
        const Loop& lp = loops_[size_t(l - cfg_.core_start)];
        ik::States st;
        for (const float* b : bank) st.p[st.n++] = b;
        st.p[st.n++] = bank_out;
        ik::depth_mix(q_, st, lp.msg_query, 4, T, E, 1e-6f, msg_);
        layer(app, l, x, blocks, msg_, &lp, out);
      }
      x = out;
    }
  }
  for (int l = cfg_.core_end; l < cfg_.n_layer; ++l) {
    float* out = next_of(x);
    layer(l, l, x, blocks, nullptr, nullptr, out);
    x = out;
  }
  for (const auto& c : chunks) slots_[size_t(c.slot)].length += int64_t(c.tokens.size());
  if (R == 0) {
    q_.wait();
    return {};
  }

  // The output: the logit rows of every state, mixed, normed, the head (in groups of the int8 GEMV's rows), the soft-cap.
  ik::States st;
  for (size_t i = 0; i <= blocks.size(); ++i) {
    const float* s = i < blocks.size() ? blocks[i] : x;
    float* d = rows_ + int64_t(i) * R * E;
    ik::gather_rows(q_, s, lrows_, R, E, d);
    st.p[st.n++] = d;
  }
  float* mixed = rows_ + int64_t(st.n) * R * E;
  ik::depth_mix(q_, st, output_res_query_, 1, R, E, 1e-6f, mixed);
  fast::rmsnorm(q_, mixed, output_norm_, R, E, cfg_.eps, mixed);
  emit("result_norm", mixed, {E, R});
  for (int64_t r0 = 0; r0 < R; r0 += xmx::kFp16Rows) {
    const int64_t rc = std::min<int64_t>(xmx::kFp16Rows, R - r0);
    act_head_.k = E;
    xmx::quantize(q_, mixed + r0 * E, rc, act_head_);
    xmx::matmul(q_, act_head_, head_w_, logits_ + r0 * V);
  }
  if (cfg_.softcap > 0) fast::softcap(q_, logits_, cfg_.softcap, R * V);
  mark("head");
  std::vector<float> full(size_t(R * V));
  q_.memcpy(full.data(), logits_, full.size() * sizeof(float)).wait();
  std::vector<float> out(size_t(R * VO));
  for (int64_t r = 0; r < R; ++r) std::memcpy(out.data() + r * VO, full.data() + r * V, size_t(VO) * sizeof(float));
  if (dump && dump->wants("result_output")) dump->put("result_output", out, {VO, R});
  return out;
}

}  // namespace eklow::engine::inkling
