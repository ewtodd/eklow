#include "model/rerank_qwen3.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "kernels/dense.hpp"
#include "kernels/ref.hpp"

namespace eklow::model {

namespace {

constexpr int64_t kMaxPairs = 128;      // pairs in one pass (they bound the tile list)
constexpr int64_t kAttnChunk = 2048;   // tokens of one attention chunk, as EmbeddingGemma 2 takes its inputs

float half_bits(uint16_t b) {
  const uint32_t sign = uint32_t(b >> 15) << 31, exp = (b >> 10) & 31, man = b & 1023;
  uint32_t f;
  if (exp == 0) {
    if (man == 0) {
      f = sign;
    } else {
      int e = -1;
      uint32_t m = man;
      do ++e, m <<= 1;
      while (!(m & 1024));
      f = sign | uint32_t(127 - 15 - e) << 23 | (m & 1023) << 13;
    }
  } else if (exp == 31) {
    f = sign | 0x7f800000u | man << 13;
  } else {
    f = sign | (exp + 127 - 15) << 23 | man << 13;
  }
  float x;
  std::memcpy(&x, &f, 4);
  return x;
}

// Row r of a Q8_0 matrix (blocks of 32: an fp16 scale, then 32 int8), dequantized.
void q8_row(const uint8_t* data, int64_t k, int64_t r, float* out) {
  const uint8_t* row = data + uint64_t(r) * uint64_t(k / 32) * 34;
  for (int64_t b = 0; b < k / 32; ++b) {
    const uint8_t* blk = row + b * 34;
    const float d = half_bits(uint16_t(blk[0] | blk[1] << 8));
    for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * float(int8_t(blk[2 + j]));
  }
}

}  // namespace

float* RerankQwen3::dev(size_t floats, const char* what) {
  auto* p = sycl::malloc_device<float>(std::max<size_t>(floats, 1), q_);
  if (!p) throw std::runtime_error(std::string("reranker: out of device memory for ") + what);
  q_.memset(p, 0, std::max<size_t>(floats, 1) * 4);
  owned_.push_back(p);
  bytes_ += floats * 4;
  return p;
}

RerankQwen3::RerankQwen3(sycl::queue& q, const gguf::File& f, int64_t max_tokens) : q_(q), f_(f) {
  if (f.at("general.architecture").as_string() != "qwen3") throw std::runtime_error(f.path() + ": not a qwen3 GGUF");
  const std::string p = "qwen3.";
  if (!f.find(p + "pooling_type") || f.at(p + "pooling_type").as_int() != 4)
    throw std::runtime_error(f.path() + ": not a reranker (pooling type is not rank)");
  const gguf::Value* labels = f.find(p + "classifier.output_labels");
  if (!labels || labels->as_array().size() != 2 || labels->as_array()[0].as_string() != "yes")
    throw std::runtime_error(f.path() + ": the classifier's labels are not [yes, no]");
  const gguf::Value* tmpl = f.find("tokenizer.chat_template.rerank");
  if (!tmpl) throw std::runtime_error(f.path() + ": no rerank template (tokenizer.chat_template.rerank)");
  template_ = tmpl->as_string();
  tok_ = std::make_unique<text::QwenTokenizer>(text::QwenTokenizer::from_gguf(f));
  auto get = [&](const std::string& k) { return f.at(p + k).as_int(); };
  E_ = get("embedding_length"), F_ = get("feed_forward_length"), H_ = get("attention.head_count"), Hkv_ = get("attention.head_count_kv");
  D_ = get("attention.key_length"), n_layer_ = get("block_count");
  eps_ = float(f.at(p + "attention.layer_norm_rms_epsilon").as_float());
  const double rope_base = f.at(p + "rope.freq_base").as_float();
  if (get("attention.value_length") != D_ || H_ % Hkv_) throw std::runtime_error(f.path() + ": unsupported attention layout");
  embd_ = f.tensor("token_embd.weight");
  if (!embd_ || embd_->type != gguf::Type::Q8_0) throw std::runtime_error(f.path() + ": the reranker wants token_embd.weight in Q8_0");

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
  // Rows of several matrices of one type and width stacked (or, interleaved, alternating in groups of 8, the layout xmx's
  // gate|up epilogue reads), uploaded and repacked.
  auto matrix = [&](const std::vector<std::string>& names, bool interleave = false) {
    std::vector<const gguf::Tensor*> ts;
    for (const auto& nm : names) ts.push_back(&need(nm));
    const gguf::Type type = ts[0]->type;
    const int64_t k = int64_t(ts[0]->ne[0]);
    int64_t n = 0;
    for (const auto* t : ts) {
      if (t->type != type || int64_t(t->ne[0]) != k) throw std::runtime_error(f.path() + ": stacked matrices differ in type or width");
      n += int64_t(t->ne[1]);
    }
    const size_t row = ts[0]->nbytes / ts[0]->ne[1];
    std::vector<uint8_t> host(size_t(n) * row);
    if (interleave) {
      const int64_t half = int64_t(ts[0]->ne[1]);
      for (int64_t g = 0; g < half / 8; ++g)
        for (int s = 0; s < 2; ++s) std::memcpy(host.data() + size_t(2 * g + s) * 8 * row, ts[size_t(s)]->data + size_t(g) * 8 * row, 8 * row);
    } else {
      size_t off = 0;
      for (const auto* t : ts) std::memcpy(host.data() + off, t->data, t->nbytes), off += t->nbytes;
    }
    auto* d = sycl::malloc_device<uint8_t>(host.size(), q_);
    if (!d) throw std::runtime_error("reranker: out of device memory loading weights");
    q_.memcpy(d, host.data(), host.size()).wait();
    const xmx::Weight w = xmx::repack(q_, ref::Weight{type, k, n, d, row});
    q_.wait();
    sycl::free(d, q_);
    repacked_.push_back(w);
    bytes_ += xmx::weight_tile_bytes(w.k, w.n) * (w.format == xmx::WeightFormat::Q8_0 ? 2 : 4) + (w.scales ? xmx::weight_scale_bytes(w.k, w.n) : 0);
    return w;
  };
  for (int64_t l = 0; l < n_layer_; ++l) {
    const std::string b = "blk." + std::to_string(l) + ".";
    Layer L;
    L.qkv = matrix({b + "attn_q.weight", b + "attn_k.weight", b + "attn_v.weight"});
    L.o = matrix({b + "attn_output.weight"});
    L.gate_up = matrix({b + "ffn_gate.weight", b + "ffn_up.weight"}, true);
    L.down = matrix({b + "ffn_down.weight"});
    L.attn_norm = vec(b + "attn_norm.weight"), L.ffn_norm = vec(b + "ffn_norm.weight");
    L.q_norm = vec(b + "attn_q_norm.weight"), L.k_norm = vec(b + "attn_k_norm.weight");
    layers_.push_back(L);
  }
  // The head on the host: cls.output's two rows (yes, no) and the final norm.
  const gguf::Tensor& cls = need("cls.output.weight");
  if (int64_t(cls.ne[0]) != E_ || cls.ne[1] != 2 || (cls.type != gguf::Type::Q8_0 && cls.type != gguf::Type::F32))
    throw std::runtime_error(f.path() + ": cls.output is not 2 rows of the model's width in Q8_0 or F32");
  cls_.resize(size_t(2 * E_));
  for (int64_t r = 0; r < 2; ++r) {
    if (cls.type == gguf::Type::Q8_0) q8_row(cls.data, E_, r, cls_.data() + r * E_);
    else std::memcpy(cls_.data() + r * E_, cls.data + r * E_ * 4, size_t(E_) * 4);
  }
  const gguf::Tensor& on = need("output_norm.weight");
  out_norm_.assign(reinterpret_cast<const float*>(on.data), reinterpret_cast<const float*>(on.data) + E_);

  max_tok_ = max_tokens;
  const int64_t R = std::max<int64_t>((max_tok_ + 255) / 256 * 256, 256), group = H_ / Hkv_;
  x_ = dev(R * E_, "x"), out_ = dev(R * E_, "out"), qkv_ = dev(R * (H_ + 2 * Hkv_) * D_, "qkv");
  Q_ = dev(R * H_ * D_, "Q"), K_ = dev(R * Hkv_ * D_, "K"), V_ = dev(R * Hkv_ * D_, "V"), att_ = dev(R * H_ * D_, "attention");
  a_ = xmx::alloc_activations(q_, R, std::max({E_, H_ * D_, F_})), a2_ = xmx::alloc_activations(q_, R, F_);
  for (auto* a : {&a_, &a2_}) q_.memset(a->h, 0, size_t(a->capacity) * 2), bytes_ += uint64_t(a->capacity) * 7 / 2;
  auto ints = [&](int64_t n, const char* what) { return reinterpret_cast<int32_t*>(dev(size_t(n), what)); };
  pos_ = ints(R, "positions"), cpos_ = ints(R, "cache positions"), slot_ = ints(R, "slots"), lo_ = ints(R, "key ranges"), hi_ = ints(R, "key ranges");
  pages_ = R / attn::kPage;
  max_tiles_ = Hkv_ * ((R * group + 7) / 8 + kMaxPairs + R / kAttnChunk);
  tiles_ = ints(6 * max_tiles_, "tile list");
  {
    std::vector<int32_t> identity(static_cast<size_t>(pages_));
    for (int64_t i = 0; i < pages_; ++i) identity[size_t(i)] = int32_t(i);
    table_ = ints(pages_, "page table");
    q_.memcpy(table_, identity.data(), identity.size() * 4).wait();
    std::vector<float> inv(size_t(D_ / 2));
    for (size_t i = 0; i < inv.size(); ++i) inv[i] = float(std::pow(rope_base, -2.0 * double(i) / double(D_)));
    inv_freq_ = dev(inv.size(), "rope");
    q_.memcpy(inv_freq_, inv.data(), inv.size() * 4).wait();
  }
  // The K/V of one layer for a pass's rows (each layer's overwrite the last's: a pass holds its pairs whole).
  const size_t cache = size_t(pages_ * Hkv_ * attn::kPage * D_);
  kc_ = sycl::malloc_device<sycl::half>(cache, q_), vc_ = sycl::malloc_device<sycl::half>(cache, q_);
  if (!kc_ || !vc_) throw std::runtime_error("reranker: out of device memory for the attention cache");
  q_.memset(kc_, 0, cache * 2), q_.memset(vc_, 0, cache * 2);
  bytes_ += cache * 4;
  attn::reserve(q_, scratch_, max_tiles_, std::min(R, kAttnChunk) * group, Hkv_, D_, group, R);
  q_.wait();
}

RerankQwen3::~RerankQwen3() {
  q_.wait();
  for (void* p : owned_) sycl::free(p, q_);
  for (auto& w : repacked_) xmx::free_weight(q_, w);
  xmx::free_activations(q_, a_);
  xmx::free_activations(q_, a2_);
  for (void* p : {static_cast<void*>(kc_), static_cast<void*>(vc_), static_cast<void*>(scratch_.q), static_cast<void*>(scratch_.part),
                  static_cast<void*>(scratch_.part2), static_cast<void*>(scratch_.p), static_cast<void*>(scratch_.stat)})
    if (p) sycl::free(p, q_);
}

std::vector<int32_t> RerankQwen3::prompt(const std::string& query, const std::string& document) const {
  std::string s;
  for (size_t pos = 0; pos < template_.size();) {
    if (template_.compare(pos, 7, "{query}") == 0) s += query, pos += 7;
    else if (template_.compare(pos, 10, "{document}") == 0) s += document, pos += 10;
    else s += template_[pos++];
  }
  return tok_->encode(s, true);
}

void RerankQwen3::pass(const std::vector<std::vector<int32_t>>& pairs, size_t i0, size_t i1, std::vector<float>& last) {
  const int64_t E = E_, group = H_ / Hkv_;
  int64_t rows = 0;
  for (size_t i = i0; i < i1; ++i) rows += int64_t(pairs[i].size());
  // Rows: the pairs back to back; each token at its position in its pair, its keys from its pair's first row to its own.
  std::vector<float> emb(size_t(rows * E));
  std::vector<int32_t> pos(static_cast<size_t>(rows)), cpos(static_cast<size_t>(rows)), lo(static_cast<size_t>(rows)), hi(static_cast<size_t>(rows)), last_rows;
  attn::TileList tl(D_);
  int64_t r = 0;
  for (size_t i = i0; i < i1; ++i) {
    const int64_t a = r, n = int64_t(pairs[i].size());
    for (int64_t t = 0; t < n; ++t, ++r) {
      const int32_t id = pairs[i][size_t(t)];
      if (id < 0 || uint64_t(id) >= embd_->ne[1]) throw std::runtime_error("reranker: token id out of range");
      q8_row(embd_->data, E, id, emb.data() + r * E);
      pos[size_t(r)] = int32_t(t), cpos[size_t(r)] = int32_t(r), lo[size_t(r)] = int32_t(a), hi[size_t(r)] = int32_t(r);
    }
    for (int64_t c0 = a; c0 < a + n; c0 += kAttnChunk) {
      const int64_t m = std::min(kAttnChunk, a + n - c0);
      tl.add_chunk(int32_t(c0), m, c0 + m - 1, group, Hkv_, 0, a, c0 + m - 1);
    }
    last_rows.push_back(int32_t(a + n - 1));
  }
  const std::vector<int32_t> packed = tl.packed();
  std::vector<int32_t> zeros(size_t(rows), 0);
  q_.memcpy(x_, emb.data(), emb.size() * 4);
  q_.memcpy(pos_, pos.data(), pos.size() * 4);
  q_.memcpy(cpos_, cpos.data(), cpos.size() * 4);
  q_.memcpy(lo_, lo.data(), lo.size() * 4);
  q_.memcpy(hi_, hi.data(), hi.size() * 4);
  q_.memcpy(slot_, zeros.data(), zeros.size() * 4);
  q_.memcpy(tiles_, packed.data(), packed.size() * 4);
  const attn::Tokens tok{slot_, cpos_, rows, lo_, hi_};
  const attn::Cache cache{kc_, vc_, table_, pages_, 0, Hkv_, D_};
  const int64_t mg = std::max<int64_t>(rows, xmx::kFp16Rows + 1);
  for (int64_t l = 0; l < n_layer_; ++l) {
    const Layer& L = layers_[size_t(l)];
    dense::norm_quant(q_, x_, l ? out_ : nullptr, L.attn_norm, rows, E, eps_, a_, rows, nullptr);
    xmx::matmul(q_, a_, L.qkv, qkv_);
    dense::qk_heads(q_, qkv_, rows, H_, Hkv_, D_, L.q_norm, L.k_norm, eps_, pos_, inv_freq_, Q_, K_, V_);
    attn::write(q_, K_, V_, tok, cache);
    attn::attend(q_, Q_, tok, tl, tiles_, cache, H_, 1.0f / std::sqrt(float(D_)), att_, scratch_);
    a_.k = H_ * D_;
    xmx::quantize(q_, att_, rows, a_, true);
    xmx::matmul(q_, a_, L.o, out_);
    // At least 33 rows, so that xmx keeps the fp16 form through the gate|up epilogue (rows past the pass's are ignored).
    dense::norm_quant(q_, x_, out_, L.ffn_norm, rows, E, eps_, a_, mg, nullptr);
    xmx::matmul_geglu(q_, a_, L.gate_up, a2_, nullptr, xmx::Glu::Silu);
    xmx::matmul(q_, a2_, L.down, out_);
  }
  ref::add(q_, out_, x_, rows * E, x_);
  // Each pair's last row to the host (llama.cpp's rank pooling reads the last token of a causal model).
  const size_t base = last.size();
  last.resize(base + last_rows.size() * size_t(E));
  for (size_t i = 0; i < last_rows.size(); ++i) q_.memcpy(last.data() + base + i * size_t(E), x_ + int64_t(last_rows[i]) * E, size_t(E) * 4);
  q_.wait();
}

std::vector<double> RerankQwen3::score(const std::vector<std::vector<int32_t>>& pairs) {
  for (const auto& p : pairs)
    if (p.empty() || int64_t(p.size()) > max_tok_)
      throw std::length_error("a query and document of " + std::to_string(p.size()) + " tokens; the reranker takes at most " + std::to_string(max_tok_));
  std::vector<float> last;
  for (size_t i0 = 0; i0 < pairs.size();) {
    size_t i1 = i0;
    int64_t rows = 0;
    while (i1 < pairs.size() && int64_t(i1 - i0) < kMaxPairs && rows + int64_t(pairs[i1].size()) <= max_tok_) rows += int64_t(pairs[i1++].size());
    pass(pairs, i0, i1, last);
    i0 = i1;
  }
  // The head: the final norm, the two rows, their softmax; the probability of yes.
  std::vector<double> out;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const float* h = last.data() + i * size_t(E_);
    double ss = 0;
    for (int64_t d = 0; d < E_; ++d) ss += double(h[d]) * double(h[d]);
    const double inv = 1.0 / std::sqrt(ss / double(E_) + double(eps_));
    double yes = 0, no = 0;
    for (int64_t d = 0; d < E_; ++d) {
      const double v = double(h[d]) * inv * double(out_norm_[size_t(d)]);
      yes += v * cls_[size_t(d)], no += v * cls_[size_t(E_ + d)];
    }
    out.push_back(1.0 / (1.0 + std::exp(no - yes)));
  }
  return out;
}

}  // namespace eklow::model
