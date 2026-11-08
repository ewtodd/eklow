// The DFlash 2 draft's serving forward (see dflash2.hpp): one slot's context injected into its ring cache, blocks of
// several slots drafted in one pass. Everything is enqueued on one in-order queue and waited for once, where the host
// needs the lattice.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

#include "kernels/dflash.hpp"
#include "kernels/fast.hpp"
#include "kernels/ref_dflash.hpp"
#include "model/dflash2.hpp"

namespace eklow::model {

namespace {

constexpr int kMaxRows = xmx::kFp16Rows;  // a block pass runs on int8 activations

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

}  // namespace

OwnedTarget::OwnedTarget(sycl::queue& q, const gguf::File& target) : q_(q) {
  const gguf::Tensor* emb = target.tensor("token_embd.weight");
  const gguf::Tensor* out = target.tensor("output.weight");
  if (!out) out = emb;  // a tied head: the embedding table is the head
  if (!emb || !out) throw std::runtime_error(target.path() + ": needs token_embd.weight (and output.weight, unless the head is tied)");
  auto put = [&](const gguf::Tensor& t) {
    auto* d = sycl::malloc_device<uint8_t>(t.nbytes, q);
    if (!d) throw std::runtime_error("out of device memory uploading " + t.name);
    q.memcpy(d, t.data, t.nbytes).wait();
    return d;
  };
  tok_embd = ref::Weight{emb->type, int64_t(emb->ne[0]), int64_t(emb->ne[1]), put(*emb), emb->nbytes / emb->ne[1]};
  uint8_t* raw = put(*out);
  raw_ = raw;
  xmx::WeightFormat f;
  if (!xmx::format_of(out->type, f)) throw std::runtime_error(target.path() + ": output.weight is of a type without an XMX layout");
  head = xmx::repack_format(q, ref::Weight{out->type, int64_t(out->ne[0]), int64_t(out->ne[1]), raw, out->nbytes / out->ne[1]}, f);
  q.wait();
  sycl::free(raw, q);
  raw_ = nullptr;
}

OwnedTarget::~OwnedTarget() {
  q_.wait();
  sycl::free(const_cast<uint8_t*>(tok_embd.data), q_);
  xmx::free_weight(q_, head);
}

struct Dflash2::Impl {
  struct LayerW {
    xmx::Weight qkv, wo, gate_up, down, conv_attn, conv_ffn;
    const float *attn_norm, *q_norm, *k_norm, *ffn_norm, *base_attn, *base_ffn;
  };
  std::vector<LayerW> layers;
  xmx::Weight fc, kv_all, sel_hidden;
  const float *hidden_norm = nullptr, *out_norm = nullptr, *inv_freq = nullptr;
  ref::Weight sel_pred{}, sel_succ{};
  Dflash2Target target{};
  int pages = 0;
  std::vector<void*> ck, cv;
  std::vector<void*> cks, cvs;      // the scales of an int8 cache (the full layer of a plain draft)
  std::vector<attn::Cache> caches;  // per layer
  int64_t full_window = 0;          // the full layer's ring: positions it keeps
  bool causal = false;              // sliding layers attend causally inside the block
  attn::Scratch scratch;
  xmx::Activations act_e, act_o, act_f, act_g, inj_feat, inj_e;
  // block pass buffers
  float *x, *n1, *cvb, *dyn, *qkv, *Qh, *Kh, *Vh, *att, *ob, *gu, *logits, *hid, *gate, *unary, *pred, *succ, *lat;
  int32_t *cand, *pred_ids, *tscr_ids, *meta, *packed, *packed_full = nullptr;
  uint8_t* emb_rows = nullptr;  // host_table: a row of the token table per block (its anchor), then the mask's
  float* tscr_vals;
  // injection buffers
  float *ih, *ikv, *iK, *iV;
  int32_t *ipos, *islot;
  std::vector<void*> allocs;
  uint64_t bytes = 0;
  sycl::queue* q = nullptr;

  template <class T>
  T* dev(size_t n) {
    T* p = sycl::malloc_device<T>(std::max<size_t>(n, 1), *q);
    if (!p) throw std::runtime_error("dflash2: out of device memory");
    allocs.push_back(p);
    bytes += n * sizeof(T);
    return p;
  }
  ~Impl() {
    if (!q) return;
    q->wait();
    for (void* p : allocs) sycl::free(p, *q);
    for (auto& l : layers) {
      xmx::free_weight(*q, l.qkv), xmx::free_weight(*q, l.wo), xmx::free_weight(*q, l.gate_up), xmx::free_weight(*q, l.down);
      xmx::free_weight(*q, l.conv_attn), xmx::free_weight(*q, l.conv_ffn);
    }
    xmx::free_weight(*q, fc), xmx::free_weight(*q, kv_all), xmx::free_weight(*q, sel_hidden);
    for (xmx::Activations* a : {&act_e, &act_o, &act_f, &act_g, &inj_feat, &inj_e}) xmx::free_activations(*q, *a);
    for (void* p : ck) sycl::free(p, *q);
    for (void* p : cv) sycl::free(p, *q);
    for (void* p : cks) sycl::free(p, *q);
    for (void* p : cvs) sycl::free(p, *q);
  }
};

Dflash2::~Dflash2() = default;

Dflash2::Dflash2(sycl::queue& q, const gguf::File& draft, const Dflash2Target& target, const Options& opt)
    : p_(std::make_unique<Impl>()), q_(q), c_(Dflash2Config::from_gguf(draft)), window_(opt.window ? opt.window : c_.window), opt_(opt) {
  Impl& s = *p_;
  s.q = &q;
  s.target = target;
  if (opt.slots * 1 > kMaxRows || opt.max_block > kMaxRows) throw std::runtime_error("dflash2: too many slots or too large a block");
  const int E = c_.n_embd, Hkv = c_.n_head_kv, D = c_.head_dim, H = c_.n_head, F = c_.n_feat();
  const int nc = c_.plain ? 0 : 2 * c_.conv_kernel * (E / c_.conv_group);
  if (c_.plain && c_.n_vocab != target.tok_embd.n)
    throw std::runtime_error(draft.path() + ": its vocabulary (" + std::to_string(c_.n_vocab) + " tokens) is not the target's (" + std::to_string(target.tok_embd.n) + " rows)");
  end_.assign(size_t(opt.slots), 0);
  first_.assign(size_t(opt.slots), 0);
  first_full_.assign(size_t(opt.slots), 0);

  auto need = [&](const std::string& name) -> const gguf::Tensor& {
    const gguf::Tensor* t = draft.tensor(name);
    if (!t) throw std::runtime_error(draft.path() + ": missing tensor " + name);
    return *t;
  };
  // Rows of several tensors of one type stacked into one matrix and repacked; the staging copy is freed.
  auto stacked = [&](std::vector<const gguf::Tensor*> parts, bool interleave) {
    const gguf::Tensor& t0 = *parts[0];
    uint64_t total = 0;
    int64_t n = 0;
    for (const auto* t : parts) {
      if (t->type != t0.type || t->ne[0] != t0.ne[0]) throw std::runtime_error(draft.path() + ": tensors to stack differ in type or width");
      total += t->nbytes, n += int64_t(t->ne[1]);
    }
    xmx::WeightFormat f;
    if (!xmx::format_of(t0.type, f))
      throw std::runtime_error(draft.path() + ": " + t0.name + " is " + gguf::type_name(t0.type) +
                               ", which has no XMX layout: the serving forward takes the draft's Q8_0 file (BF16 only runs in the reference)");
    auto* stage = sycl::malloc_device<uint8_t>(total, q);
    if (!stage) throw std::runtime_error("dflash2: out of device memory");
    uint64_t off = 0;
    for (const auto* t : parts) q.memcpy(stage + off, t->data, t->nbytes), off += t->nbytes;
    xmx::Weight w = xmx::repack_format(q, ref::Weight{t0.type, int64_t(t0.ne[0]), n, stage, t0.nbytes / t0.ne[1]}, f, interleave);
    q.wait();
    sycl::free(stage, q);
    s.bytes += xmx::weight_tile_bytes(w.format, w.k, w.n) + xmx::weight_meta_bytes(w.format, w.k, w.n);
    return w;
  };
  auto vec = [&](const std::string& name, int64_t n) {
    const gguf::Tensor& t = need(name);
    if (t.type != gguf::Type::F32 || int64_t(t.elements()) != n) throw std::runtime_error(draft.path() + ": " + name + " is not an F32 array of " + std::to_string(n));
    float* d = s.dev<float>(size_t(n));
    q.memcpy(d, t.data, t.nbytes).wait();
    return static_cast<const float*>(d);
  };
  auto raw = [&](const std::string& name) {
    const gguf::Tensor& t = need(name);
    uint8_t* d = s.dev<uint8_t>(t.nbytes);
    q.memcpy(d, t.data, t.nbytes).wait();
    return ref::Weight{t.type, int64_t(t.ne[0]), int64_t(t.ne[1]), d, t.nbytes / t.ne[1]};
  };

  s.fc = stacked({&need("fc.weight")}, false);
  if (!c_.plain) {
    s.sel_hidden = stacked({&need("selector_hidden.weight")}, false);
    s.sel_pred = raw("selector_predecessor.weight");
    s.sel_succ = raw("selector_successor.weight");
  }
  s.hidden_norm = vec("enc.output_norm.weight", E);
  s.out_norm = vec("output_norm.weight", E);
  std::vector<const gguf::Tensor*> kv_parts;
  for (int l = 0; l < c_.n_layer; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Impl::LayerW L{};
    L.qkv = stacked({&need(b + "attn_q.weight"), &need(b + "attn_k.weight"), &need(b + "attn_v.weight")}, false);
    L.wo = stacked({&need(b + "attn_output.weight")}, false);
    L.gate_up = stacked({&need(b + "ffn_gate.weight"), &need(b + "ffn_up.weight")}, true);
    L.down = stacked({&need(b + "ffn_down.weight")}, false);
    if (!c_.plain) {
      L.conv_attn = stacked({&need(b + "attn_conv_proj.weight")}, false);
      L.conv_ffn = stacked({&need(b + "ffn_conv_proj.weight")}, false);
    }
    L.attn_norm = vec(b + "attn_norm.weight", E);
    L.q_norm = vec(b + "attn_q_norm.weight", D);
    L.k_norm = vec(b + "attn_k_norm.weight", D);
    L.ffn_norm = vec(b + "ffn_norm.weight", E);
    if (!c_.plain) {
      L.base_attn = vec(b + "attn_conv_base", int64_t(2) * c_.conv_kernel * E);
      L.base_ffn = vec(b + "ffn_conv_base", int64_t(2) * c_.conv_kernel * E);
    }
    kv_parts.push_back(&need(b + "attn_k.weight"));
    kv_parts.push_back(&need(b + "attn_v.weight"));
    s.layers.push_back(L);
  }
  s.kv_all = stacked(kv_parts, false);  // every layer's k then v rows: the context's one matmul

  std::vector<float> inv(size_t(D / 2));
  for (int i = 0; i < D / 2; ++i) inv[size_t(i)] = float(std::pow(double(c_.rope_base), -2.0 * i / D));
  float* invd = s.dev<float>(inv.size());
  q.memcpy(invd, inv.data(), inv.size() * sizeof(float)).wait();
  s.inv_freq = invd;

  // The ring caches: pages of 256 positions per slot, enough for the window, a block and the page it starts in. A plain
  // draft's full layer has the same ring over opt.full_context positions (int8 or fp16).
  s.pages = int((int64_t(window_) + opt.max_block + attn::kPage - 1) / attn::kPage) + 1;
  s.causal = c_.block_causal;
  if (opt.sliding_causal >= 0) s.causal = opt.sliding_causal > 0;
  s.full_window = c_.has_full() ? (opt.full_context > 0 ? opt.full_context : 32768) : 0;
  reach_ = std::max<int64_t>(window_, s.full_window);
  full_window_ = s.full_window;
  for (int l = 0; l < c_.n_layer; ++l) {
    const bool full = c_.full_layer(l);
    const int pages = full ? int((s.full_window + opt.max_block + attn::kPage - 1) / attn::kPage) + 1 : s.pages;
    const size_t elems = size_t(opt.slots) * size_t(pages) * size_t(Hkv) * size_t(attn::kPage) * size_t(D);
    if (elems >= (size_t(1) << 31)) throw std::runtime_error("dflash2: a layer's cache is too large for 32-bit offsets (fewer slots or a smaller full_context)");
    attn::Cache c{};
    if (full && opt.full_int8) {
      s.ck.push_back(sycl::malloc_device<int8_t>(elems, q));
      s.cv.push_back(sycl::malloc_device<int8_t>(elems, q));
      const size_t scales = elems / 16;
      s.cks.push_back(sycl::malloc_device<sycl::half>(scales, q));
      s.cvs.push_back(sycl::malloc_device<sycl::half>(scales, q));
      if (!s.ck.back() || !s.cv.back() || !s.cks.back() || !s.cvs.back()) throw std::runtime_error("dflash2: out of device memory for the cache");
      s.bytes += 2 * elems + 2 * scales * sizeof(sycl::half);
      c = attn::Cache{s.ck.back(), s.cv.back(), nullptr, pages, s.full_window, Hkv, D, static_cast<sycl::half*>(s.cks.back()), static_cast<sycl::half*>(s.cvs.back())};
    } else {
      s.ck.push_back(sycl::malloc_device<sycl::half>(elems, q));
      s.cv.push_back(sycl::malloc_device<sycl::half>(elems, q));
      if (!s.ck.back() || !s.cv.back()) throw std::runtime_error("dflash2: out of device memory for the cache");
      s.bytes += 2 * elems * sizeof(sycl::half);
      c = attn::Cache{s.ck.back(), s.cv.back(), nullptr, pages, full ? s.full_window : window_, Hkv, D};
    }
    s.caches.push_back(c);
  }
  s.act_e = xmx::alloc_activations(q, kMaxRows, E);
  s.act_o = xmx::alloc_activations(q, kMaxRows, H * D);
  s.act_f = xmx::alloc_activations(q, kMaxRows, c_.n_ff);
  s.inj_feat = xmx::alloc_activations(q, opt.max_inject, F);
  s.inj_e = xmx::alloc_activations(q, opt.max_inject, E);
  const size_t R = kMaxRows, V = size_t(c_.n_vocab), K = size_t(c_.top_k);
  s.x = s.dev<float>(R * E);
  s.n1 = s.dev<float>(R * E);
  s.cvb = s.dev<float>(R * E);
  s.dyn = s.dev<float>(R * nc);
  s.qkv = s.dev<float>(R * (H + 2 * Hkv) * D);
  s.Qh = s.dev<float>(R * H * D);
  s.Kh = s.dev<float>(R * Hkv * D);
  s.Vh = s.dev<float>(R * Hkv * D);
  s.att = s.dev<float>(R * H * D);
  s.ob = s.dev<float>(R * E);
  s.gu = s.dev<float>(R * 2 * c_.n_ff);
  s.logits = s.dev<float>(R * V);
  s.hid = s.dev<float>(R * E);
  s.gate = s.dev<float>(R * c_.rank);
  s.unary = s.dev<float>(R * K);
  s.pred = s.dev<float>(R * K * c_.rank);
  s.succ = s.dev<float>(R * K * c_.rank);
  s.lat = s.dev<float>(R * K * K);
  s.cand = s.dev<int32_t>(R * K);
  s.pred_ids = s.dev<int32_t>(R * K);
  const size_t chunks = size_t(dflash::topk_chunks(c_.n_vocab));
  s.tscr_ids = s.dev<int32_t>(R * chunks * 16);
  s.tscr_vals = s.dev<float>(R * chunks * 16);
  s.meta = s.dev<int32_t>((c_.has_full() ? 8 : 6) * R);  // a full layer has key ranges of its own (lo, hi)
  s.packed = s.dev<int32_t>(8192);
  if (c_.has_full()) s.packed_full = s.dev<int32_t>(8192);
  if (target.host_table) s.emb_rows = s.dev<uint8_t>(size_t(opt.slots + 1) * target.tok_embd.row_bytes);
  const size_t I = size_t(opt.max_inject);
  s.ih = s.dev<float>(I * E);
  s.ikv = s.dev<float>(I * size_t(c_.n_layer) * 2 * Hkv * D);
  s.iK = s.dev<float>(I * Hkv * D);
  s.iV = s.dev<float>(I * Hkv * D);
  s.ipos = s.dev<int32_t>(I);
  s.islot = s.dev<int32_t>(I);
  attn::reserve(q, s.scratch, 256, 64, Hkv, D);
  bytes_ = s.bytes;
}

void Dflash2::reset(int slot) { end_[size_t(slot)] = first_[size_t(slot)] = first_full_[size_t(slot)] = 0; }

void Dflash2::truncate(int slot, int64_t pos) {
  end_[size_t(slot)] = std::min(end_[size_t(slot)], pos);
  first_[size_t(slot)] = std::min(first_[size_t(slot)], end_[size_t(slot)]);
  first_full_[size_t(slot)] = std::min(first_full_[size_t(slot)], end_[size_t(slot)]);
}

void Dflash2::inject_rows(const float* feats, const int32_t* slots, const int64_t* positions, int64_t n) {
  const int64_t F = c_.n_feat();
  for (int64_t i = 0; i < n;) {
    int64_t j = i + 1;
    while (j < n && slots[j] == slots[i] && positions[j] == positions[j - 1] + 1) ++j;
    inject(slots[i], feats + i * F, positions[i], j - i);
    i = j;
  }
}

void Dflash2::inject(int slot, const float* feats, int64_t pos0, int64_t n) {
  Impl& s = *p_;
  const int E = c_.n_embd, Hkv = c_.n_head_kv, D = c_.head_dim, F = c_.n_feat();
  if (slot < 0 || slot >= opt_.slots) throw std::runtime_error("dflash2: no such slot");
  if (pos0 < 0) throw std::runtime_error("dflash2: negative position");
  if (n <= 0) return;
  if (pos0 > end_[size_t(slot)]) end_[size_t(slot)] = first_[size_t(slot)] = first_full_[size_t(slot)] = pos0;  // a gap: the context restarts here
  const int64_t next = pos0 + n;
  const int64_t from = std::max<int64_t>(pos0, next - window_ + 1);  // the oldest row a block at `next` or later can see
  // A plain draft's full layer sees further back (its ring's size): rows from there on go to its cache alone.
  const int64_t from_full = full_window_ > 0 ? std::max<int64_t>(pos0, next - full_window_ + 1) : from;
  if (from > pos0 && end_[size_t(slot)] <= pos0) first_[size_t(slot)] = from;
  if (from_full > pos0 && end_[size_t(slot)] <= pos0) first_full_[size_t(slot)] = from_full;
  for (int64_t c0 = std::min(from, from_full); c0 < next; c0 += opt_.max_inject) {
    const int64_t m = std::min<int64_t>(opt_.max_inject, next - c0);
    const float* src = feats + (c0 - pos0) * F;
    xmx::quantize(q_, src, m, s.inj_feat);
    xmx::matmul(q_, s.inj_feat, s.fc, s.ih);
    fast::rmsnorm(q_, s.ih, s.hidden_norm, m, E, c_.eps, s.ih);
    xmx::quantize(q_, s.ih, m, s.inj_e);
    xmx::matmul(q_, s.inj_e, s.kv_all, s.ikv);
    std::vector<int32_t> hp(static_cast<size_t>(m)), hs(static_cast<size_t>(m), slot);
    for (int64_t i = 0; i < m; ++i) hp[size_t(i)] = int32_t(c0 + i);
    q_.memcpy(s.ipos, hp.data(), hp.size() * sizeof(int32_t));
    q_.memcpy(s.islot, hs.data(), hs.size() * sizeof(int32_t));
    const int64_t ld = int64_t(c_.n_layer) * 2 * Hkv * D;
    for (int l = 0; l < c_.n_layer; ++l) {
      const auto& L = s.layers[size_t(l)];
      // the rows of this chunk the layer keeps (all of them, but for the sliding layers of a draft whose full layer reaches further)
      const int64_t r0 = std::max<int64_t>(0, (c_.full_layer(l) ? from_full : from) - c0), cnt = m - r0;
      if (cnt <= 0) continue;
      const attn::Tokens tok{s.islot + r0, s.ipos + r0, cnt};
      dflash::heads(q_, s.ikv + r0 * ld + l * 2 * Hkv * D, ld, cnt, 0, Hkv, 0, 0, Hkv * D, nullptr, L.k_norm, c_.eps, s.ipos + r0, s.inv_freq, nullptr, s.iK, s.iV);
      attn::write(q_, s.iK, s.iV, tok, s.caches[size_t(l)]);
    }
    q_.wait();  // hs, hp go out of scope
  }
  end_[size_t(slot)] = next;
}

std::vector<serve::Lattice> Dflash2::lattices(const std::vector<Request>& reqs, int B, Profile* profile, BlockDebug* debug) {
  Impl& s = *p_;
  const int E = c_.n_embd, H = c_.n_head, Hkv = c_.n_head_kv, D = c_.head_dim, K = c_.top_k, Rk = c_.rank;
  const int nb = int(reqs.size()), R = nb * B, P = nb * (B - 1), nc = c_.plain ? 0 : 2 * c_.conv_kernel * (E / c_.conv_group);
  const bool full_layers = c_.has_full();
  if (nb == 0) return {};
  if (B < 2 || B > opt_.max_block || R > kMaxRows) throw std::runtime_error("dflash2: block of " + std::to_string(B) + " tokens x " + std::to_string(nb) + " requests is out of range");
  double t0 = profile ? now_ms() : 0;
  auto lap = [&](double& field) {
    if (!profile) return;
    q_.wait();
    const double t = now_ms();
    field += t - t0;
    t0 = t;
  };
  // Per row: token, position, slot, first and last key; per block: the anchor. A plain draft's full layers have their own
  // first and last key per row (after the others).
  std::vector<int32_t> meta(size_t((full_layers ? 8 : 6) * kMaxRows), 0);
  attn::TileList list(D), list_full(D);
  for (int b = 0; b < nb; ++b) {
    const Request& r = reqs[size_t(b)];
    if (r.slot < 0 || r.slot >= opt_.slots || r.pos != end_[size_t(r.slot)]) throw std::runtime_error("dflash2: a block must start at its slot's context end");
    for (int t = 0; t < B; ++t) {
      const int i = b * B + t;
      meta[size_t(i)] = s.emb_rows ? (t == 0 ? b : nb) : (t == 0 ? r.anchor : c_.mask_token);  // host_table: a row of emb_rows
      meta[size_t(R + i)] = int32_t(r.pos + t);
      meta[size_t(2 * R + i)] = r.slot;
      meta[size_t(3 * R + i)] = int32_t(std::max<int64_t>({0, first_[size_t(r.slot)], r.pos + t - window_ + 1}));
      meta[size_t(4 * R + i)] = int32_t(s.causal ? r.pos + t : r.pos + B - 1);
      if (full_layers) {
        meta[size_t(6 * R + i)] = int32_t(std::max<int64_t>({0, first_full_[size_t(r.slot)], r.pos + t - full_window_ + 1}));
        meta[size_t(7 * R + i)] = int32_t(r.pos + B - 1);
      }
    }
    meta[size_t(5 * R + b)] = r.anchor;
    list.add_chunk(b * B, B, r.pos + B - 1, H / Hkv, Hkv, window_, meta[size_t(3 * R + b * B)], r.pos + B - 1);
    if (full_layers) list_full.add_chunk(b * B, B, r.pos + B - 1, H / Hkv, Hkv, full_window_, meta[size_t(6 * R + b * B)], r.pos + B - 1);
  }
  const std::vector<int32_t> packed = list.packed(), packed_full = full_layers ? list_full.packed() : std::vector<int32_t>();
  if (packed.size() > 8192 || packed_full.size() > 8192) throw std::runtime_error("dflash2: tile list too long");
  int32_t *tok = s.meta, *pos = s.meta + R, *slot = s.meta + 2 * R, *lo = s.meta + 3 * R, *hi = s.meta + 4 * R, *anchors = s.meta + 5 * R;
  q_.memcpy(s.meta, meta.data(), size_t((full_layers ? 8 : 6) * R) * sizeof(int32_t));
  q_.memcpy(s.packed, packed.data(), packed.size() * sizeof(int32_t));
  if (full_layers) q_.memcpy(s.packed_full, packed_full.data(), packed_full.size() * sizeof(int32_t));
  const attn::Tokens tokens{slot, pos, R, lo, hi};
  const attn::Tokens tokens_full{slot, pos, R, full_layers ? s.meta + 6 * R : nullptr, full_layers ? s.meta + 7 * R : nullptr};
  const float scale = 1.0f / std::sqrt(float(D));
  if (s.emb_rows) {
    const ref::Weight& te = s.target.tok_embd;
    if (nb > opt_.slots) throw std::runtime_error("dflash2: more blocks than slots");
    for (int b = 0; b < nb; ++b) q_.memcpy(s.emb_rows + size_t(b) * te.row_bytes, te.data + size_t(reqs[size_t(b)].anchor) * te.row_bytes, te.row_bytes);
    q_.memcpy(s.emb_rows + size_t(nb) * te.row_bytes, te.data + size_t(c_.mask_token) * te.row_bytes, te.row_bytes);
    ref::Weight small = te;
    small.n = nb + 1;
    small.data = s.emb_rows;
    ref::embed(q_, small, tok, R, c_.embed_scale, s.x);
  } else {
    ref::embed(q_, s.target.tok_embd, tok, R, c_.embed_scale, s.x);
  }
  lap(profile ? profile->embed_ms : t0);
  for (int l = 0; l < c_.n_layer; ++l) {
    const auto& L = s.layers[size_t(l)];
    const bool full = c_.full_layer(l);
    auto stage = [&](const char* name, const float* p, size_t n) {
      if (!debug || l != debug->stage_layer) return;
      std::vector<float> h(n);
      q_.memcpy(h.data(), p, n * sizeof(float)).wait();
      debug->stages.emplace_back(name, std::move(h));
    };
    fast::rmsnorm(q_, s.x, L.attn_norm, R, E, c_.eps, s.n1);
    stage("n1", s.n1, size_t(R) * E);
    if (!c_.plain) {
      xmx::quantize(q_, s.n1, R, s.act_e);
      xmx::matmul(q_, s.act_e, L.conv_attn, s.dyn);
      stage("dyn", s.dyn, size_t(R) * nc);
      dflash::conv(q_, s.n1, s.dyn, L.base_attn, 0, R, B, E, c_.conv_group, s.cvb, false);
      stage("conv", s.cvb, size_t(R) * E);
      xmx::quantize(q_, s.cvb, R, s.act_e);
    } else {
      xmx::quantize(q_, s.n1, R, s.act_e);
    }
    xmx::matmul(q_, s.act_e, L.qkv, s.qkv);
    dflash::heads(q_, s.qkv, int64_t(H + 2 * Hkv) * D, R, H, Hkv, 0, int64_t(H) * D, int64_t(H + Hkv) * D, L.q_norm, L.k_norm, c_.eps, pos, s.inv_freq, s.Qh,
                  s.Kh, s.Vh);
    stage("q", s.Qh, size_t(R) * H * D);
    stage("k", s.Kh, size_t(R) * Hkv * D);
    stage("v", s.Vh, size_t(R) * Hkv * D);
    const attn::Cache& cache = s.caches[size_t(l)];
    attn::write(q_, s.Kh, s.Vh, tokens, cache);
    attn::attend(q_, s.Qh, full ? tokens_full : tokens, full ? list_full : list, full ? s.packed_full : s.packed, cache, H, scale, s.att, s.scratch);
    if (debug) {
      debug->parts.emplace_back(size_t(R) * H * D);
      q_.memcpy(debug->parts.back().data(), s.att, size_t(R) * H * D * sizeof(float)).wait();
    }
    xmx::quantize(q_, s.att, R, s.act_o);
    xmx::matmul(q_, s.act_o, L.wo, s.ob);
    if (!c_.plain) {
      dflash::conv(q_, s.ob, s.dyn, L.base_attn, 1, R, B, E, c_.conv_group, s.x, true);
    } else {
      ref::add(q_, s.x, s.ob, int64_t(R) * E, s.x);
    }
    stage("x_mid", s.x, size_t(R) * E);
    fast::rmsnorm(q_, s.x, L.ffn_norm, R, E, c_.eps, s.n1);
    xmx::quantize(q_, s.n1, R, s.act_e);
    if (!c_.plain) {
      xmx::matmul(q_, s.act_e, L.conv_ffn, s.dyn);
      dflash::conv(q_, s.n1, s.dyn, L.base_ffn, 0, R, B, E, c_.conv_group, s.cvb, false);
      stage("conv2", s.cvb, size_t(R) * E);
      xmx::quantize(q_, s.cvb, R, s.act_e);
    }
    xmx::matmul_geglu(q_, s.act_e, L.gate_up, s.act_f, s.gu, xmx::Glu::Silu);
    if (debug && l == debug->stage_layer) {  // the GLU's output, from the int8 activations' dequantization
      std::vector<float> h(size_t(R) * c_.n_ff);
      std::vector<int8_t> hq(size_t(R) * c_.n_ff);
      std::vector<float> hs(hq.size() / 32 + 64);
      q_.memcpy(hq.data(), s.act_f.q, hq.size()).wait();
      q_.memcpy(hs.data(), s.act_f.s, std::min(hs.size(), size_t(s.act_f.capacity / 32)) * sizeof(float)).wait();
      const int64_t kb = c_.n_ff / 32;
      for (int r = 0; r < R; ++r)
        for (int64_t k = 0; k < c_.n_ff; ++k)
          h[size_t(r) * c_.n_ff + k] = float(hq[size_t(xmx::act_index(r, k, kb))]) * hs[size_t(xmx::act_scale_index(r, k, kb))];
      debug->stages.emplace_back("glu", std::move(h));
    }
    xmx::matmul(q_, s.act_f, L.down, s.ob);
    stage("down", s.ob, size_t(R) * E);
    if (!c_.plain) {
      dflash::conv(q_, s.ob, s.dyn, L.base_ffn, 1, R, B, E, c_.conv_group, s.x, true);
    } else {
      ref::add(q_, s.x, s.ob, int64_t(R) * E, s.x);
    }
    if (debug) {
      debug->layers.emplace_back(size_t(R) * E);
      q_.memcpy(debug->layers.back().data(), s.x, size_t(R) * E * sizeof(float)).wait();
    }
  }
  fast::rmsnorm(q_, s.x, s.out_norm, R, E, c_.eps, s.x);
  lap(profile ? profile->layers_ms : t0);
  // Rows 1..B-1 of each block: the head, the top 16, the hidden projection.
  for (int b = 0; b < nb; ++b) q_.memcpy(s.hid + size_t(b) * (B - 1) * E, s.x + size_t(b * B + 1) * E, size_t(B - 1) * E * sizeof(float));
  xmx::quantize(q_, s.hid, P, s.act_e);
  xmx::matmul(q_, s.act_e, s.target.head, s.logits);
  lap(profile ? profile->head_ms : t0);
  if (c_.plain) {  // no selector: each position's top 16 of its own logits, softcapped on the host
    dflash::topk16(q_, s.logits, P, c_.n_vocab, s.tscr_ids, s.tscr_vals, s.cand, s.unary);
    std::vector<int32_t> hcand(size_t(P) * K);
    std::vector<float> hun(size_t(P) * K);
    q_.memcpy(hcand.data(), s.cand, hcand.size() * sizeof(int32_t));
    q_.memcpy(hun.data(), s.unary, hun.size() * sizeof(float)).wait();
    lap(profile ? profile->select_ms : t0);
    std::vector<serve::Lattice> out(static_cast<size_t>(nb));
    for (int b = 0; b < nb; ++b) {
      serve::Lattice& l = out[size_t(b)];
      l.top_k = K;
      const size_t p0 = size_t(b) * (B - 1);
      l.cand.assign(hcand.begin() + long(p0 * K), hcand.begin() + long((p0 + B - 1) * K));
      l.unary.assign(hun.begin() + long(p0 * K), hun.begin() + long((p0 + B - 1) * K));
      plain_lattice(l, c_.softcap);
    }
    if (debug) {
      debug->hidden.resize(size_t(P) * E);
      q_.memcpy(debug->hidden.data(), s.hid, debug->hidden.size() * sizeof(float)).wait();
    }
    return out;
  }
  xmx::matmul(q_, s.act_e, s.sel_hidden, s.gate);
  dflash::topk16(q_, s.logits, P, c_.n_vocab, s.tscr_ids, s.tscr_vals, s.cand, s.unary);
  dflash::predecessor_ids(q_, s.cand, anchors, nb, B - 1, s.pred_ids);
  ref::embed(q_, s.sel_pred, s.pred_ids, int64_t(P) * K, 1.0f, s.pred);
  ref::embed(q_, s.sel_succ, s.cand, int64_t(P) * K, 1.0f, s.succ);
  ref::selector_lattice(q_, s.unary, s.gate, s.pred, s.succ, P, K, Rk, s.lat);
  std::vector<int32_t> hcand(size_t(P) * K);
  std::vector<float> hun(size_t(P) * K), hlat(size_t(P) * K * K);
  q_.memcpy(hcand.data(), s.cand, hcand.size() * sizeof(int32_t));
  q_.memcpy(hun.data(), s.unary, hun.size() * sizeof(float));
  q_.memcpy(hlat.data(), s.lat, hlat.size() * sizeof(float)).wait();
  lap(profile ? profile->select_ms : t0);
  std::vector<serve::Lattice> out(static_cast<size_t>(nb));
  for (int b = 0; b < nb; ++b) {
    serve::Lattice& l = out[size_t(b)];
    l.top_k = K;
    const size_t p0 = size_t(b) * (B - 1);
    l.cand.assign(hcand.begin() + p0 * K, hcand.begin() + (p0 + B - 1) * K);
    l.unary.assign(hun.begin() + p0 * K, hun.begin() + (p0 + B - 1) * K);
    l.score.assign(hlat.begin() + p0 * K * K, hlat.begin() + (p0 + B - 1) * K * K);
  }
  return out;
}

std::vector<serve::DraftBlock> Dflash2::draft(const std::vector<Request>& reqs, int B, float temperature, std::mt19937_64& rng, Profile* profile) {
  const std::vector<serve::Lattice> lats = lattices(reqs, B, profile);
  const double t0 = profile ? now_ms() : 0;
  std::vector<serve::DraftBlock> out;
  for (const auto& l : lats) out.push_back(serve::walk(l, temperature, rng));
  if (profile) profile->host_ms += now_ms() - t0;
  return out;
}

serve::DraftBlock Dflash2::draft(int slot, int32_t anchor, int64_t pos, int B, float temperature, std::mt19937_64& rng, Profile* profile) {
  return draft(std::vector<Request>{{slot, anchor, pos}}, B, temperature, rng, profile)[0];
}

}  // namespace eklow::model
