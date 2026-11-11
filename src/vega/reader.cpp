#include "vega/reader.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

#include "kernels/dense.hpp"
#include "kernels/fast.hpp"
#include "kernels/qwen_fast.hpp"
#include "kernels/ref.hpp"

namespace eklow::vega {

namespace {

constexpr int64_t kMaxOptions = 64;
constexpr int64_t kMaxChunks = 64;

float bf16_to_float(const uint8_t* p) {
  const uint32_t u = uint32_t(p[0] | p[1] << 8) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

float* Reader::dev(size_t floats, const char* what) {
  auto* p = sycl::malloc_device<float>(std::max<size_t>(floats, 1), q_);
  if (!p) throw std::runtime_error(std::string("vega reader: out of device memory for ") + what);
  q_.memset(p, 0, std::max<size_t>(floats, 1) * 4);
  owned_.push_back(p);
  bytes_ += floats * 4;
  return p;
}

Reader::Reader(sycl::queue& q, const gguf::File& f, std::vector<int> feature_layers, std::vector<int> option_layers, int64_t max_tokens,
               int64_t chunk)
    : q_(q), feat_(std::move(feature_layers)), optl_(std::move(option_layers)) {
  if (f.at("general.architecture").as_string() != "qwen35") throw std::runtime_error(f.path() + ": not a qwen35 (dense Qwen3.5) GGUF");
  const std::string p = "qwen35.";
  auto get = [&](const std::string& k) { return f.at(p + k).as_int(); };
  E_ = get("embedding_length"), F_ = get("feed_forward_length"), H_ = get("attention.head_count");
  Hkv_ = get("attention.head_count_kv"), D_ = get("attention.key_length"), n_rot_ = get("rope.dimension_count");
  eps_ = float(f.at(p + "attention.layer_norm_rms_epsilon").as_float());
  const double rope_base = f.at(p + "rope.freq_base").as_float();
  Hd_ = get("ssm.time_step_rank");
  if (get("ssm.group_count") != Hd_ || get("ssm.state_size") != gdn_seq::kHead || get("ssm.conv_kernel") != gdn_seq::kTaps ||
      get("ssm.inner_size") != Hd_ * gdn_seq::kHead)
    throw std::runtime_error(f.path() + ": a DeltaNet layout the reader does not run (it needs as many value as key heads, of 128, and 4 taps)");
  if (get("attention.value_length") != D_ || H_ % Hkv_ || n_rot_ > D_) throw std::runtime_error(f.path() + ": unsupported attention layout");
  const auto& sec = f.at(p + "rope.dimension_sections").as_array();
  int64_t sum = 0;
  for (size_t i = 0; i < sec.size() && i < 4; ++i) {
    if (i < 3) sections_[i] = int(sec[i].as_int());
    else if (sec[i].as_int() != 0) throw std::runtime_error(f.path() + ": a fourth rope section");
    sum += sec[i].as_int();
  }
  if (sum != n_rot_ / 2) throw std::runtime_error(f.path() + ": rope sections do not cover the rotated dims");
  std::vector<bool> recurrent;
  for (const auto& v : f.at(p + "attention.recurrent_layers").as_array())
    recurrent.push_back(v.kind == gguf::Value::Kind::Bool ? std::get<bool>(v.v) : v.as_int() != 0);
  const int n_layer = int(get("block_count"));
  std::set<int> caps(feat_.begin(), feat_.end());
  caps.insert(optl_.begin(), optl_.end());
  if (feat_.empty() || optl_.empty() || *caps.begin() < 0 || *caps.rbegin() >= n_layer)
    throw std::runtime_error("vega reader: feature or option layers outside the model's " + std::to_string(n_layer));
  caps_.assign(caps.begin(), caps.end());
  n_run_ = caps_.back() + 1;
  embd_ = f.tensor("token_embd.weight");
  if (!embd_ || embd_->type != gguf::Type::BF16 || embd_->ne[0] != uint64_t(E_))
    throw std::runtime_error(f.path() + ": the reader wants token_embd.weight in BF16 (convert with --outtype bf16)");

  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = f.tensor(name);
    if (!t) throw std::runtime_error(f.path() + ": missing tensor " + name);
    return *t;
  };
  auto vec = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    if (t.type != gguf::Type::F32) throw std::runtime_error(f.path() + ": " + name + " is not F32");
    float* d = dev(t.nbytes / 4, "vectors");
    q_.memcpy(d, t.data, t.nbytes).wait();
    return static_cast<const float*>(d);
  };
  // Rows of several BF16 matrices of the same width stacked (or, interleaved, alternating in groups of 8: the layout
  // xmx's gate|up epilogue reads), uploaded and repacked into fp16 tiles.
  auto matrix = [&](const std::vector<std::string>& names, bool interleave = false) {
    std::vector<const gguf::Tensor*> ts;
    int64_t k = -1, n = 0;
    for (const auto& nm : names) {
      const gguf::Tensor& t = need(nm);
      if (t.type != gguf::Type::BF16) throw std::runtime_error(f.path() + ": " + nm + " is not BF16");
      if (k >= 0 && int64_t(t.ne[0]) != k) throw std::runtime_error(f.path() + ": stacked matrices of different widths");
      k = int64_t(t.ne[0]);
      n += int64_t(t.ne[1]);
      ts.push_back(&t);
    }
    const size_t row = size_t(k) * 2;
    std::vector<uint8_t> host(size_t(n) * row);
    if (interleave) {
      if (ts.size() != 2 || ts[0]->ne[1] != ts[1]->ne[1] || ts[0]->ne[1] % 8) throw std::runtime_error("vega reader: bad gate|up pair");
      const int64_t half = int64_t(ts[0]->ne[1]);
      for (int64_t g = 0; g < half / 8; ++g)
        for (int s = 0; s < 2; ++s) std::memcpy(host.data() + size_t(2 * g + s) * 8 * row, ts[size_t(s)]->data + size_t(g) * 8 * row, 8 * row);
    } else {
      size_t off = 0;
      for (const auto* t : ts) std::memcpy(host.data() + off, t->data, t->nbytes), off += t->nbytes;
    }
    auto* d = sycl::malloc_device<uint8_t>(host.size(), q_);
    if (!d) throw std::runtime_error("vega reader: out of device memory loading weights");
    q_.memcpy(d, host.data(), host.size()).wait();
    const xmx::Weight w = xmx::repack(q_, ref::Weight{gguf::Type::BF16, k, n, d, row});
    q_.wait();
    sycl::free(d, q_);
    repacked_.push_back(w);
    bytes_ += xmx::weight_tile_bytes(w.k, w.n) * 4;
    return w;
  };

  int n_gdn = 0, n_attn = 0;
  for (int l = 0; l < n_run_; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L{};
    L.recurrent = recurrent.at(static_cast<size_t>(l));
    L.attn_norm = vec(b + "attn_norm.weight");
    L.post_norm = vec(b + "post_attention_norm.weight");
    L.gate_up = matrix({b + "ffn_gate.weight", b + "ffn_up.weight"}, true);
    L.down = matrix({b + "ffn_down.weight"});
    if (L.recurrent) {
      L.in = matrix({b + "attn_qkv.weight"});
      L.z = matrix({b + "attn_gate.weight"});
      L.out = matrix({b + "ssm_out.weight"});
      std::vector<float> ab(size_t(2 * Hd_ * E_));
      int64_t r = 0;
      for (const char* nm : {"ssm_alpha.weight", "ssm_beta.weight"}) {
        const gguf::Tensor& t = need(b + nm);
        if (t.type != gguf::Type::BF16 || int64_t(t.ne[0]) != E_ || int64_t(t.ne[1]) != Hd_) throw std::runtime_error(f.path() + ": bad " + b + nm);
        for (int64_t i = 0; i < Hd_ * E_; ++i) ab[size_t(r * E_ + i)] = bf16_to_float(t.data + 2 * i);
        r += Hd_;
      }
      float* d = dev(ab.size(), "the DeltaNet gates");
      q_.memcpy(d, ab.data(), ab.size() * 4).wait();
      L.ab = d;
      L.gdn = gdn_seq::Weights{vec(b + "ssm_conv1d.weight"), vec(b + "ssm_a"), vec(b + "ssm_dt.bias"), vec(b + "ssm_norm.weight"), eps_};
      L.gdn_index = n_gdn++;
    } else {
      L.in = matrix({b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"});
      L.out = matrix({b + "attn_output.weight"});
      L.q_norm = vec(b + "attn_q_norm.weight");
      L.k_norm = vec(b + "attn_k_norm.weight");
      L.attn_index = n_attn++;
    }
    layers_.push_back(L);
  }

  // Activations for chunks of up to chunk_ rows (the fp16 GEMMs pad rows to 256, and the MLP runs at least 33 rows so that
  // xmx keeps its fp16 form: the buffers hold 256 at least).
  max_tok_ = max_tokens, chunk_ = std::min(chunk, max_tokens);
  const int64_t R = std::max<int64_t>((chunk_ + 255) / 256 * 256, 256), HW = Hd_ * gdn_seq::kHead;
  x_ = dev(R * E_, "x"), h_ = dev(R * E_, "h"), out_ = dev(R * E_, "out");
  big_ = dev(R * std::max(3 * HW, (2 * H_ + 2 * Hkv_) * D_), "projections");
  zb_ = dev(R * HW, "z"), ab_ = dev(R * 2 * Hd_, "a, b");
  Q_ = dev(R * H_ * D_, "Q"), G_ = dev(R * H_ * D_, "gates"), att_ = dev(R * H_ * D_, "attention");
  K_ = dev(R * Hkv_ * D_, "K"), V_ = dev(R * Hkv_ * D_, "V");
  const int64_t kmax = std::max({E_, F_, H_ * D_, HW});
  a_ = xmx::alloc_activations(q_, R, kmax), a2_ = xmx::alloc_activations(q_, R, kmax);
  for (auto* a : {&a_, &a2_}) q_.memset(a->h, 0, size_t(a->capacity) * 2), bytes_ += uint64_t(a->capacity) * 7 / 2;
  gs_ = gdn_seq::alloc_scratch(q_, R, int(Hd_));
  bytes_ += uint64_t(R) * (3 * HW + 2 * Hd_ + HW) * 4;
  state_ = gdn_seq::alloc_state(q_, n_gdn, int(Hd_), kSlots);
  saved_ = gdn_seq::alloc_state(q_, n_gdn, int(Hd_), 1);
  bytes_ += (kSlots + 1) * uint64_t(state_.ssm_floats() + state_.conv_floats()) * 4;
  {
    std::vector<float> inv(size_t(n_rot_ / 2));
    for (size_t i = 0; i < inv.size(); ++i) inv[i] = float(std::pow(rope_base, -2.0 * double(i) / double(n_rot_)));
    inv_freq_ = dev(inv.size(), "rope");
    q_.memcpy(inv_freq_, inv.data(), inv.size() * 4).wait();
  }
  auto ints = [&](int64_t n, const char* what) { return reinterpret_cast<int32_t*>(dev(static_cast<size_t>(n), what)); };
  pos_ = ints(3 * R, "positions"), slot_ = ints(R, "slots"), run_of_row_ = ints(R, "runs of rows");
  runs_ = ints(4 * kSlots, "runs"), pool_runs_ = ints(4 * kSlots, "runs");
  // Attention: a pool of pages, [page][K/V head][256][D] fp16 per layer, that the slots' page tables share out; query
  // tiles of a pass: ceil(rows / 8) per K/V head and sequence, rows = tokens x the query heads per K/V head.
  pages_ = (max_tok_ + attn::kPage - 1) / attn::kPage;
  max_tiles_ = Hkv_ * ((chunk_ * (H_ / Hkv_) + 7) / 8) + 2 * Hkv_ * kSlots;
  tiles_ = ints(6 * max_tiles_, "tile list");
  for (int i = 0; i < n_attn; ++i) {
    const size_t n = size_t(pages_ * Hkv_ * attn::kPage * D_);
    auto* k = sycl::malloc_device<sycl::half>(n, q_);
    auto* v = sycl::malloc_device<sycl::half>(n, q_);
    if (!k || !v) throw std::runtime_error("vega reader: out of device memory for the attention cache");
    q_.memset(k, 0, n * 2), q_.memset(v, 0, n * 2);
    kc_.push_back(k), vc_.push_back(v);
    bytes_ += n * 4;
  }
  table_ = ints(kSlots * pages_, "page tables");
  attn::reserve(q_, attn_scratch_, max_tiles_, chunk_ * (H_ / Hkv_), Hkv_, D_, H_ / Hkv_, max_tok_);
  max_segs_ = 3 * int64_t(feat_.size()) + kMaxOptions * int64_t(optl_.size()) + kMaxChunks;
  segs_ = ints(5 * max_segs_ * kSlots, "pooled spans");
  sums_ = dev(size_t((kSlots + 1) * max_segs_ * E_), "pooled sums");
  q_.wait();
}

Reader::~Reader() {
  q_.wait();
  for (void* p : owned_) sycl::free(p, q_);
  for (auto& w : repacked_) xmx::free_weight(q_, w);
  for (void* p : kc_) sycl::free(p, q_);
  for (void* p : vc_) sycl::free(p, q_);
  xmx::free_activations(q_, a_);
  xmx::free_activations(q_, a2_);
  gdn_seq::free_state(q_, state_);
  gdn_seq::free_state(q_, saved_);
  gdn_seq::free_scratch(q_, gs_);
  for (void* p : {static_cast<void*>(attn_scratch_.q), static_cast<void*>(attn_scratch_.part), static_cast<void*>(attn_scratch_.part2),
                  static_cast<void*>(attn_scratch_.p), static_cast<void*>(attn_scratch_.stat)})
    if (p) sycl::free(p, q_);
}

// The sequences' tokens through the layers, from their DeltaNet states and the cache as they stand: each pass fills its
// rows with the sequences' next tokens one sequence after another (a run each), and each capture layer's output adds the
// positions of every span in `segs` into sums (rows of E floats). One wait, at the end: the passes' host inputs
// (embedding rows, positions, runs, tile lists) are built as they are enqueued and kept until then.
void Reader::forward(const std::vector<Seq>& seqs, const std::vector<Seg>& segs, float* sums) {
  const int64_t E = E_, group = H_ / Hkv_;
  if (seqs.empty() || int64_t(seqs.size()) > kSlots || int64_t(segs.size()) > kSlots * max_segs_)
    throw std::runtime_error("vega reader: too many sequences or pooled spans for one pass");
  std::vector<int32_t> flat, table(size_t(kSlots * pages_), 0);
  for (const Seg& s : segs) flat.insert(flat.end(), {s.cap, s.seq, s.a, s.b, s.dst});
  for (const Seq& sq : seqs) {
    const int64_t need = (sq.p1 + attn::kPage - 1) / attn::kPage;
    if (sq.page0 + need > pages_) throw std::runtime_error("vega reader: the pass's prompts do not fit the cache");
    for (int64_t j = 0; j < need; ++j) table[size_t(sq.slot * pages_ + j)] = int32_t(sq.page0 + j);
  }
  if (!flat.empty()) q_.memcpy(segs_, flat.data(), flat.size() * 4);
  q_.memcpy(table_, table.data(), table.size() * 4).wait();
  const int32_t n_segs = int32_t(segs.size());
  // EKLOW_VEGA_PROFILE=1: wait after each stage and add its time to a total per stage, printed by read().
  static const bool profile = std::getenv("EKLOW_VEGA_PROFILE") != nullptr;
  auto t_mark = std::chrono::steady_clock::now();
  auto mark = [&](const char* stage) {
    if (!profile) return;
    q_.wait();
    const auto now = std::chrono::steady_clock::now();
    profile_[stage] += std::chrono::duration<double, std::milli>(now - t_mark).count();
    t_mark = now;
  };
  std::vector<std::vector<float>> rows_kept;
  std::vector<std::vector<int32_t>> ints_kept;
  std::vector<int64_t> cur;
  for (const Seq& sq : seqs) cur.push_back(sq.p0);
  for (;;) {
    // The pass: up to chunk_ rows, the sequences' next tokens in order.
    std::vector<gdn_seq::Run> runs;
    std::vector<int32_t> pool_runs;  // per run: sequence, first row, first position, rows
    int64_t rows = 0;
    for (size_t i = 0; i < seqs.size() && rows < chunk_; ++i) {
      const int64_t m = std::min(seqs[i].p1 - cur[i], chunk_ - rows);
      if (m <= 0) continue;
      runs.push_back({seqs[i].slot, int32_t(rows), int32_t(m), 0});
      pool_runs.insert(pool_runs.end(), {int32_t(i), int32_t(rows), int32_t(cur[i]), int32_t(m)});
      rows += m;
    }
    if (rows == 0) break;
    const int64_t mg = std::max<int64_t>(rows, xmx::kFp16Rows + 1);
    // The embedding rows (BF16 in the mapped file, as fp32), each row's positions (three equal components), slot and run.
    rows_kept.emplace_back(size_t(rows * E));
    float* hr = rows_kept.back().data();
    std::vector<int32_t> host(size_t(5 * rows));
    int32_t *pos = host.data(), *slot = pos + 3 * rows, *run_of = slot + rows;
    attn::TileList tl(D_);
    for (size_t r = 0; r < runs.size(); ++r) {
      const Seq& sq = seqs[size_t(pool_runs[4 * r])];
      const int64_t c0 = pool_runs[4 * r + 2];
      for (int32_t t = 0; t < runs[r].n; ++t) {
        const int64_t row = runs[r].row0 + t, at = c0 + t;
        const int32_t id = (*sq.ids)[size_t(at)];
        if (id < 0 || uint64_t(id) >= embd_->ne[1]) throw std::runtime_error("vega reader: token id out of range");
        const uint8_t* src = embd_->data + uint64_t(id) * uint64_t(E) * 2;
        for (int64_t i = 0; i < E; ++i) hr[row * E + i] = bf16_to_float(src + 2 * i);
        pos[row] = pos[rows + row] = pos[2 * rows + row] = int32_t(at);
        slot[row] = sq.slot, run_of[row] = int32_t(r);
      }
      tl.add_chunk(runs[r].row0, runs[r].n, c0 + runs[r].n - 1, group, Hkv_, 0);
    }
    q_.memcpy(x_, hr, size_t(rows * E) * 4);
    q_.memcpy(pos_, pos, size_t(3 * rows) * 4);
    q_.memcpy(slot_, slot, size_t(rows) * 4);
    q_.memcpy(run_of_row_, run_of, size_t(rows) * 4);
    ints_kept.push_back(std::move(host));
    ints_kept.emplace_back(reinterpret_cast<const int32_t*>(runs.data()), reinterpret_cast<const int32_t*>(runs.data() + runs.size()));
    q_.memcpy(runs_, ints_kept.back().data(), ints_kept.back().size() * 4);
    ints_kept.push_back(pool_runs);
    q_.memcpy(pool_runs_, ints_kept.back().data(), ints_kept.back().size() * 4);
    ints_kept.push_back(tl.packed());
    q_.memcpy(tiles_, ints_kept.back().data(), ints_kept.back().size() * 4);
    const attn::Tokens tok{slot_, pos_, rows};
    const auto* gruns = reinterpret_cast<const gdn_seq::Run*>(runs_);
    const int32_t n_runs = int32_t(runs.size());
    mark("embed");

    // The MLP's output joins the residual stream in the next layer's norm, or at once where the layer's output is pooled.
    bool pending = false;
    for (int64_t l = 0; l < n_run_; ++l) {
      const Layer& L = layers_[size_t(l)];
      dense::norm_quant(q_, x_, pending ? out_ : nullptr, L.attn_norm, rows, E, eps_, a_, rows, L.recurrent ? h_ : nullptr);
      pending = false;
      mark("norm");
      if (L.recurrent) {
        xmx::matmul(q_, a_, L.in, big_);
        xmx::matmul(q_, a_, L.z, zb_);
        fast::matmul_f32(q_, L.ab, 2 * Hd_, E, h_, rows, ab_);
        mark("gemm in");
        gdn_seq::layer(q_, L.gdn, state_, L.gdn_index, gruns, n_runs, run_of_row_, rows, big_, zb_, ab_, gs_, a2_);
        mark("deltanet");
      } else {
        xmx::matmul(q_, a_, L.in, big_);
        mark("gemm in");
        qwenk::attn_heads(q_, big_, rows, H_, Hkv_, D_, n_rot_, L.q_norm, L.k_norm, eps_, pos_, sections_, inv_freq_, Q_, K_, V_, G_);
        const attn::Cache cache{kc_[size_t(L.attn_index)], vc_[size_t(L.attn_index)], table_, pages_, 0, Hkv_, D_};
        attn::write(q_, K_, V_, tok, cache);
        attn::attend(q_, Q_, tok, tl, tiles_, cache, H_, 1.0f / std::sqrt(float(D_)), att_, attn_scratch_);
        const int64_t W = H_ * D_;
        xmx::set_rows(a2_, rows, W, true);
        sycl::half* ah = a2_.h;
        const float *att = att_, *gt = G_;
        q_.parallel_for(sycl::range<2>(size_t(rows), size_t(W)), [=](sycl::id<2> id) {
          const int64_t t = int64_t(id[0]), c = int64_t(id[1]);
          const float g = gt[t * W + c];
          ah[xmx::act16_index(t, c, W / 16)] = sycl::half(att[t * W + c] / (1.0f + sycl::exp(-g)));
        });
        mark("attention");
      }
      xmx::matmul(q_, a2_, L.out, out_);
      mark("gemm out");
      // At least 33 rows, so that xmx keeps the fp16 form through the gate|up epilogue (rows past the pass's are ignored).
      dense::norm_quant(q_, x_, out_, L.post_norm, rows, E, eps_, a_, mg, nullptr);
      mark("norm");
      xmx::matmul_geglu(q_, a_, L.gate_up, a2_, nullptr, xmx::Glu::Silu);
      xmx::matmul(q_, a2_, L.down, out_);
      mark("mlp");
      // Pooling: each span's rows within this pass, summed in fp32 (one work-item per span and column).
      const auto cap_it = std::find(caps_.begin(), caps_.end(), int(l));
      if (cap_it == caps_.end() || n_segs == 0) {
        pending = true;
        continue;
      }
      ref::add(q_, out_, x_, rows * E, x_);
      const int32_t cap = int32_t(cap_it - caps_.begin()), EE = int32_t(E);
      const int32_t *sg = segs_, *pr = pool_runs_;
      const float* x = x_;
      q_.parallel_for(sycl::range<2>(size_t(n_segs), size_t(E)), [=](sycl::id<2> id) {
        const int32_t s = int32_t(id[0]), d = int32_t(id[1]);
        if (sg[5 * s] != cap) return;
        float acc = 0;
        for (int32_t r = 0; r < n_runs; ++r) {
          if (pr[4 * r] != sg[5 * s + 1]) continue;
          const int32_t row0 = pr[4 * r + 1], c0 = pr[4 * r + 2], a = sycl::max(sg[5 * s + 2], c0), b = sycl::min(sg[5 * s + 3], c0 + pr[4 * r + 3]);
          for (int32_t p = a; p < b; ++p) acc += x[int64_t(row0 + p - c0) * EE + d];
        }
        if (acc != 0) sums[int64_t(sg[5 * s + 4]) * EE + d] += acc;
      });
      mark("pool");
    }
    for (size_t r = 0; r < runs.size(); ++r) cur[size_t(pool_runs[4 * r])] += runs[r].n;
  }
  q_.wait();
}

std::vector<Pooled> Reader::read(const std::vector<Prompt>& prompts, const PoolSpec& spec) {
  const int64_t E = E_, nF = int64_t(feat_.size()), nO = int64_t(optl_.size());
  if (spec.chunks > kMaxChunks) throw std::runtime_error("vega reader: more memory chunks than the reader holds");
  auto cap_of = [&](int layer) { return int32_t(std::find(caps_.begin(), caps_.end(), layer) - caps_.begin()); };
  auto pages_of = [&](int64_t n) { return (n + attn::kPage - 1) / attn::kPage; };
  for (const Prompt& pr : prompts) {
    if (int64_t(pr.ids.size()) > max_tok_)
      throw std::length_error("the prompt has " + std::to_string(pr.ids.size()) + " tokens; this reader holds " + std::to_string(max_tok_));
    if (pr.ids.empty() || int64_t(pr.options.size()) > kMaxOptions) throw std::runtime_error("vega reader: empty prompt or too many options");
  }
  // Rows of a prompt's sums: per feature layer state, question, last; the chunks; per option and option layer its span. The
  // state's rows do not depend on the question, so prompts sharing a prefix share them.
  auto row_chunk = [&](int64_t c) { return int32_t(3 * nF + c); };
  auto row_opt = [&](int64_t k, int64_t oi) { return int32_t(3 * nF + kMaxChunks + k * nO + oi); };
  // The state span's pieces for the memory: k pieces, token j of n in piece j k / n.
  auto pieces = [&](const Prompt& pr) {
    std::vector<std::pair<int32_t, int32_t>> out;
    const int64_t n = pr.s1 - pr.s0;
    if (spec.chunks <= 0 || n < std::max<int64_t>(spec.chunk_min_tokens, 2)) return out;
    const int64_t k = std::max<int64_t>(2, std::min<int64_t>(spec.chunks, n / std::max<int64_t>(1, spec.chunk_tokens)));
    int64_t j = 0;
    for (int64_t c = 0; c < k; ++c) {
      const int64_t a = j;
      while (j < n && j * k / n == c) ++j;
      out.emplace_back(int32_t(pr.s0 + a), int32_t(pr.s0 + j));
    }
    return out;
  };
  auto state_segs = [&](const Prompt& pr, int32_t seq, int32_t base, std::vector<Seg>& s) {
    for (int64_t fi = 0; fi < nF; ++fi) s.push_back({cap_of(feat_[size_t(fi)]), seq, pr.s0, pr.s1, base + int32_t(3 * fi)});
    const auto pc = pieces(pr);
    for (size_t c = 0; c < pc.size(); ++c) s.push_back({cap_of(feat_.back()), seq, pc[c].first, pc[c].second, base + row_chunk(int64_t(c))});
  };
  auto rest_segs = [&](const Prompt& pr, int32_t seq, int32_t base, std::vector<Seg>& s) {
    const int32_t n = int32_t(pr.ids.size());
    for (int64_t fi = 0; fi < nF; ++fi) {
      s.push_back({cap_of(feat_[size_t(fi)]), seq, pr.q0, pr.q1, base + int32_t(3 * fi + 1)});
      s.push_back({cap_of(feat_[size_t(fi)]), seq, n - 1, n, base + int32_t(3 * fi + 2)});
    }
    for (size_t k = 0; k < pr.options.size(); ++k)
      for (int64_t oi = 0; oi < nO; ++oi)
        s.push_back({cap_of(optl_[size_t(oi)]), seq, pr.options[k].first, pr.options[k].second, base + row_opt(int64_t(k), oi)});
  };
  const size_t region = size_t(max_segs_ * E);
  std::vector<std::vector<float>> sums(prompts.size());
  // Prompts whose state reaches prefix_min and that share their prefix: one pass over it, then each continuation from a copy
  // of its DeltaNet state (the cache past the prefix is overwritten by each).
  std::map<std::vector<int32_t>, std::vector<size_t>> groups;
  for (size_t i = 0; i < prompts.size(); ++i)
    if (prompts[i].s1 >= spec.prefix_min)
      groups[std::vector<int32_t>(prompts[i].ids.begin(), prompts[i].ids.begin() + prompts[i].s1)].push_back(i);
  for (auto& [pre, idx] : groups) {
    if (idx.size() < 2) continue;
    const Prompt& pr = prompts[idx[0]];
    std::vector<Seg> segs;
    state_segs(pr, 0, 0, segs);
    q_.memset(sums_, 0, region * 4);
    gdn_seq::reset(q_, state_, 0);
    forward({Seq{&pr.ids, 0, pr.s1, 0, 0}}, segs, sums_);
    gdn_seq::copy(q_, state_, 0, saved_, 0);
    std::vector<float> pre_sums(region);
    q_.memcpy(pre_sums.data(), sums_, region * 4).wait();
    for (size_t i : idx) {
      const Prompt& m = prompts[i];
      segs.clear();
      rest_segs(m, 0, 0, segs);
      gdn_seq::copy(q_, saved_, 0, state_, 0);
      q_.memset(sums_, 0, region * 4);
      forward({Seq{&m.ids, m.s1, int64_t(m.ids.size()), 0, 0}}, segs, sums_);
      sums[i].resize(region);
      q_.memcpy(sums[i].data(), sums_, region * 4).wait();
      for (size_t j = 0; j < region; ++j) sums[i][j] += pre_sums[j];
    }
  }
  // The rest side by side, as many to a batch as there are slots and their pages fit the cache.
  std::vector<size_t> rest;
  for (size_t i = 0; i < prompts.size(); ++i)
    if (sums[i].empty()) rest.push_back(i);
  for (size_t b0 = 0; b0 < rest.size();) {
    std::vector<Seq> seqs;
    std::vector<Seg> segs;
    int64_t page = 0;
    size_t b1 = b0;
    while (b1 < rest.size() && int64_t(seqs.size()) < kSlots && page + pages_of(int64_t(prompts[rest[b1]].ids.size())) <= pages_) {
      const Prompt& pr = prompts[rest[b1]];
      const int32_t k = int32_t(seqs.size());
      seqs.push_back({&pr.ids, 0, int64_t(pr.ids.size()), k, int32_t(page)});
      state_segs(pr, k, int32_t(k * max_segs_), segs);
      rest_segs(pr, k, int32_t(k * max_segs_), segs);
      gdn_seq::reset(q_, state_, k);
      page += pages_of(int64_t(pr.ids.size()));
      ++b1;
    }
    q_.memset(sums_, 0, seqs.size() * region * 4);
    forward(seqs, segs, sums_);
    for (size_t k = 0; k < seqs.size(); ++k) {
      sums[rest[b0 + k]].resize(region);
      q_.memcpy(sums[rest[b0 + k]].data(), sums_ + k * region, region * 4);
    }
    q_.wait();
    b0 = b1;
  }
  std::vector<Pooled> out;
  for (size_t i = 0; i < prompts.size(); ++i) {
    const Prompt& pr = prompts[i];
    const std::vector<float>& sm = sums[i];
    Pooled po;
    auto mean = [&](int32_t row, int64_t count, float* dst) {
      const float inv = 1.0f / float(std::max<int64_t>(count, 1));
      for (int64_t d = 0; d < E; ++d) dst[d] = sm[size_t(row * E + d)] * inv;
    };
    po.item.resize(size_t(nF * 3 * E));
    for (int64_t fi = 0; fi < nF; ++fi) {
      mean(int32_t(3 * fi), pr.s1 - pr.s0, po.item.data() + (fi * 3 + 0) * E);
      mean(int32_t(3 * fi + 1), pr.q1 - pr.q0, po.item.data() + (fi * 3 + 1) * E);
      mean(int32_t(3 * fi + 2), 1, po.item.data() + (fi * 3 + 2) * E);
    }
    po.opts.resize(pr.options.size() * size_t(nO * E));
    for (size_t k = 0; k < pr.options.size(); ++k)
      for (int64_t oi = 0; oi < nO; ++oi)
        mean(row_opt(int64_t(k), oi), pr.options[k].second - pr.options[k].first, po.opts.data() + (int64_t(k) * nO + oi) * E);
    const auto pc = pieces(pr);
    po.n_chunks = int32_t(pc.size());
    po.chunks.assign(size_t(std::max(spec.chunks, 0) * E), 0.0f);
    for (size_t c = 0; c < pc.size(); ++c) mean(row_chunk(int64_t(c)), pc[c].second - pc[c].first, po.chunks.data() + int64_t(c) * E);
    out.push_back(std::move(po));
  }
  if (!profile_.empty()) {
    std::string line = "vega reader profile (ms):";
    for (const auto& [k, v] : profile_) {
      char b[64];
      std::snprintf(b, sizeof b, " %s %.1f", k.c_str(), v);
      line += b;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    profile_.clear();
  }
  return out;
}

}  // namespace eklow::vega
