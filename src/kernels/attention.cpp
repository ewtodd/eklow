#include "kernels/attention.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace eklow::attn {

namespace {

constexpr int kSg = 16;

float* scratch_get(sycl::queue& q, Scratch& s, size_t need) {
  if (need > s.floats) {
    if (s.p) {
      q.wait();
      sycl::free(s.p, q);
    }
    s.floats = std::max(need, s.floats * 2);
    s.p = sycl::malloc_device<float>(s.floats, q);
    if (!s.p) throw std::runtime_error("attention: out of device memory for split partials");
  }
  return s.p;
}

// Where key j of query token t lives, as a lane's DL dimensions in fp32.
struct SlidingSrc {
  const float *k_new, *v_new;
  Ring ring;
  int64_t n_head_kv, head_dim;
  template <int DL>
  void load(const Meta& m, int64_t t, int64_t j, int64_t kh, int64_t d0, bool want_v, float (&out)[DL]) const {
    if (j >= m.chunk_pos[t]) {
      const float* row = (want_v ? v_new : k_new) + ((m.chunk_base[t] + (j - m.chunk_pos[t])) * n_head_kv + kh) * head_dim + d0;
#pragma unroll
      for (int i = 0; i < DL; ++i) out[i] = row[i];
    } else {
      const sycl::half* row = (want_v ? ring.v : ring.k) + m.slot[t] * ring.slot_stride +
                              ((j % ring.window) * n_head_kv + kh) * head_dim + d0;
#pragma unroll
      for (int i = 0; i < DL; ++i) out[i] = float(row[i]);
    }
  }
};

struct PageSrc {
  Pages pages;
  int64_t head_dim;
  template <int DL>
  void load(const Meta& m, int64_t t, int64_t j, int64_t, int64_t d0, bool want_v, float (&out)[DL]) const {
    const int64_t page = pages.table[m.slot[t] * pages.max_pages + j / pages.page_size];
    const sycl::half* row = (want_v ? pages.v : pages.k) + (page * pages.page_size + j % pages.page_size) * head_dim + d0;
#pragma unroll
    for (int i = 0; i < DL; ++i) out[i] = float(row[i]);
  }
};

template <int DL, class Src>
void run(sycl::queue& q, const float* Q, const Meta& meta, const Src& src, int64_t n_head, int64_t n_head_kv,
         int64_t head_dim, int64_t window, float scale, float* out, Scratch& scratch) {
  if (head_dim != DL * kSg) throw std::runtime_error("attention: head_dim does not match the kernel");
  const int64_t T = meta.n_tok, H = n_head, D = head_dim, group = n_head / n_head_kv;
  const int64_t max_keys = window > 0 ? std::min<int64_t>(window, meta.max_pos + 1) : meta.max_pos + 1;
  // Enough sub-groups to fill the card: split each query's keys when the batch is small (decode).
  const int64_t splits = std::clamp<int64_t>(65536 / (T * H), 1, (max_keys + 255) / 256);
  const int64_t split_len = (max_keys + splits - 1) / splits;
  float* part = splits > 1 ? scratch_get(q, scratch, size_t(T * H * splits * (D + 2))) : nullptr;
  const Meta m = meta;
  q.parallel_for(sycl::nd_range<3>(sycl::range<3>(size_t(T), size_t(H * splits), kSg), sycl::range<3>(1, 1, kSg)),
                 [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kSg)]] {
                   const int64_t t = int64_t(it.get_global_id(0)), hs = int64_t(it.get_global_id(1));
                   const int64_t h = hs / splits, s = hs % splits, kh = h / group;
                   const int lane = int(it.get_local_id(2));
                   const int64_t d0 = lane * DL;
                   auto sg = it.get_sub_group();
                   const int64_t pos = m.pos[t];
                   const int64_t lo = window > 0 ? std::max<int64_t>(0, pos - window + 1) : 0;
                   const int64_t j0 = lo + s * split_len, j1 = std::min<int64_t>(pos + 1, j0 + split_len);
                   float qv[DL];
#pragma unroll
                   for (int i = 0; i < DL; ++i) qv[i] = Q[(t * H + h) * D + d0 + i];
                   float mx = -std::numeric_limits<float>::infinity(), l = 0, o[DL] = {};
                   for (int64_t j = j0; j < j1; ++j) {
                     float kv[DL];
                     src.template load<DL>(m, t, j, kh, d0, false, kv);
                     float part_dot = 0;
#pragma unroll
                     for (int i = 0; i < DL; ++i) part_dot += qv[i] * kv[i];
                     const float sc = sycl::reduce_over_group(sg, part_dot, sycl::plus<float>()) * scale;
                     const float mn = sycl::fmax(mx, sc);
                     const float corr = sycl::exp(mx - mn), p = sycl::exp(sc - mn);
                     l = l * corr + p;
                     src.template load<DL>(m, t, j, kh, d0, true, kv);
#pragma unroll
                     for (int i = 0; i < DL; ++i) o[i] = o[i] * corr + p * kv[i];
                     mx = mn;
                   }
                   if (splits == 1) {
#pragma unroll
                     for (int i = 0; i < DL; ++i) out[(t * H + h) * D + d0 + i] = o[i] / l;
                   } else {
                     float* dst = part + ((t * H + h) * splits + s) * (D + 2);
#pragma unroll
                     for (int i = 0; i < DL; ++i) dst[d0 + i] = o[i];
                     if (lane == 0) {
                       dst[D] = mx;
                       dst[D + 1] = l;
                     }
                   }
                 });
  if (splits > 1)
    q.parallel_for(sycl::range<2>(size_t(T * H), size_t(D)), [=](sycl::id<2> id) {
      const int64_t th = int64_t(id[0]), d = int64_t(id[1]);
      const float* base = part + th * splits * (D + 2);
      float mx = -std::numeric_limits<float>::infinity();
      for (int64_t s = 0; s < splits; ++s) mx = sycl::fmax(mx, base[s * (D + 2) + D]);
      float l = 0, o = 0;
      for (int64_t s = 0; s < splits; ++s) {
        const float* p = base + s * (D + 2);
        if (p[D + 1] == 0) continue;  // a split with no keys for this query
        const float w = sycl::exp(p[D] - mx);
        l += p[D + 1] * w;
        o += p[d] * w;
      }
      out[th * D + d] = o / l;
    });
}

}  // namespace

void sliding(sycl::queue& q, const float* Q, const float* K, const float* V, const Meta& m, const Ring& ring,
             int64_t n_head, int64_t n_head_kv, int64_t head_dim, float scale, float* out, Scratch& scratch) {
  SlidingSrc src{K, V, ring, n_head_kv, head_dim};
  if (head_dim == 256) run<16>(q, Q, m, src, n_head, n_head_kv, head_dim, ring.window, scale, out, scratch);
  else throw std::runtime_error("attention: sliding head_dim must be 256");
}

void global(sycl::queue& q, const float* Q, const Meta& m, const Pages& pages, int64_t n_head, int64_t head_dim,
            float scale, float* out, Scratch& scratch) {
  PageSrc src{pages, head_dim};
  if (head_dim == 512) run<32>(q, Q, m, src, n_head, 1, head_dim, 0, scale, out, scratch);
  else throw std::runtime_error("attention: global head_dim must be 512");
}

void write_ring(sycl::queue& q, const float* K, const float* V, const Meta& meta, const Ring& ring, int64_t n_head_kv,
                int64_t head_dim) {
  const Meta m = meta;
  const Ring r = ring;
  const int64_t row = n_head_kv * head_dim;
  q.parallel_for(sycl::range<2>(size_t(m.n_tok), size_t(row)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), i = int64_t(id[1]);
    if (m.pos[t] <= m.chunk_last[t] - r.window) return;  // overwritten later in the same chunk
    const int64_t dst = m.slot[t] * r.slot_stride + (m.pos[t] % r.window) * row + i;
    r.k[dst] = sycl::half(K[t * row + i]);
    r.v[dst] = sycl::half(V[t * row + i]);
  });
}

void write_pages(sycl::queue& q, const float* K, const float* V, const Meta& meta, const Pages& pages, int64_t head_dim) {
  const Meta m = meta;
  const Pages p = pages;
  q.parallel_for(sycl::range<2>(size_t(m.n_tok), size_t(head_dim)), [=](sycl::id<2> id) {
    const int64_t t = int64_t(id[0]), d = int64_t(id[1]);
    const int64_t page = p.table[m.slot[t] * p.max_pages + m.pos[t] / p.page_size];
    const int64_t dst = (page * p.page_size + m.pos[t] % p.page_size) * head_dim + d;
    p.k[dst] = sycl::half(K[t * head_dim + d]);
    p.v[dst] = sycl::half(V[t * head_dim + d]);
  });
}

}  // namespace eklow::attn
