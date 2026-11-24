#include "cpu/expert.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "cpu/nvfp4.hpp"

namespace eklow::cpu {
namespace {

// An int8 activation row in the layout of the weight format it meets.
struct Act {
  NvBlocks nv;
  Q8Blocks q8;
};

void quantize_for(QType t, const float* x, int64_t k, Act& a) {
  if (t == QType::NVFP4) nv_quantize_into(x, k, a.nv);
  else quantize_into(x, k, a.q8);
}

// Rows [lo, hi) of the matrix at `rows` against the nx activations a[0..nx) (indexed through `idx`, a base pointer and
// stride, so that callers hold their rows in one array); y[j] is the output row of input j, at r - lo.
struct Ptrs {
  std::vector<const NvBlocks*> nv;
  std::vector<const Q8Blocks*> q8;
  std::vector<float*> y;
};
thread_local Ptrs t_ptrs;

void gemv(QType t, const uint8_t* rows, int64_t k, int64_t n, const Act* a, int nx, float* y, int64_t ystride) {
  Ptrs& p = t_ptrs;
  p.y.resize(size_t(nx));
  for (int j = 0; j < nx; ++j) p.y[size_t(j)] = y + j * ystride;
  if (t == QType::NVFP4) {
    p.nv.resize(size_t(nx));
    for (int j = 0; j < nx; ++j) p.nv[size_t(j)] = &a[j].nv;
    gemv_nvfp4_multi(rows, k, 0, n, p.nv.data(), nx, p.y.data());
  } else {
    p.q8.resize(size_t(nx));
    for (int j = 0; j < nx; ++j) p.q8[size_t(j)] = &a[j].q8;
    gemv_q_multi(t, rows, k, 0, n, p.q8.data(), nx, p.y.data());
  }
}

constexpr int64_t kGrain = 16;

struct Work {
  std::vector<Act> xa_nv, xa_q8;  // the inputs, once per layout in use
  std::vector<Act> ha;            // situ's outputs, [ne][nx]
  std::vector<float> h;           // the same in fp32
};
thread_local Work t_work;
thread_local std::vector<float> t_g, t_u;  // a task's gate and up rows

inline float situ(float g, float u, float beta, float lb) {
  const float a = beta * std::tanh(g / beta) * (1.0f / (1.0f + std::exp(-g)));
  return a * (lb > 0 ? lb * std::tanh(u / lb) : u);
}

}  // namespace

void experts_ffn(const ExpertRef* e, int ne, int64_t n_embd, int64_t n_ff, const float* const* x, int nx, float* const* y,
                 float beta, float lb, Pool& pool) {
  if (ne <= 0 || nx <= 0) return;
  bool need_nv = false, need_q8 = false;
  for (int i = 0; i < ne; ++i) {
    (e[i].gu_type == QType::NVFP4 ? need_nv : need_q8) = true;
    if (n_ff % qtype_block(e[i].dn_type) || n_embd % qtype_block(e[i].gu_type))
      throw std::invalid_argument("expert_ffn: n_embd or n_ff is not a multiple of a format's block");
  }
  Work& w = t_work;
  const QType kinds[2] = {QType::NVFP4, QType::Q8_0};
  std::vector<Act>* xa[2] = {&w.xa_nv, &w.xa_q8};
  for (int v = 0; v < 2; ++v) {
    if (!(v == 0 ? need_nv : need_q8)) continue;
    xa[v]->resize(size_t(nx));
    pool.for_each(nx, 8, [&](int64_t lo, int64_t hi) {
      for (int64_t j = lo; j < hi; ++j) quantize_for(kinds[v], x[j], n_embd, (*xa[v])[size_t(j)]);
    });
  }
  w.ha.resize(size_t(ne) * size_t(nx));
  w.h.resize(size_t(ne) * size_t(nx) * size_t(n_ff));
  float* const hbase = w.h.data();

  // gate_up and situ: a task is one expert's 16 gate rows and their 16 up rows against every input.
  const int64_t gchunks = (n_ff + kGrain - 1) / kGrain;
  pool.for_each(int64_t(ne) * gchunks, 1, [&](int64_t lo, int64_t hi) {
    for (int64_t t = lo; t < hi; ++t) {
      const int i = int(t / gchunks);
      const int64_t r0 = (t % gchunks) * kGrain, cnt = std::min(kGrain, n_ff - r0);
      const ExpertRef& ex = e[i];
      const uint64_t rb = qrow_bytes(ex.gu_type, n_embd);
      const Act* a = (ex.gu_type == QType::NVFP4 ? w.xa_nv : w.xa_q8).data();
      t_g.resize(size_t(nx) * kGrain);
      t_u.resize(size_t(nx) * kGrain);
      gemv(ex.gu_type, ex.gu + uint64_t(r0) * rb, n_embd, cnt, a, nx, t_g.data(), kGrain);
      gemv(ex.gu_type, ex.gu + uint64_t(n_ff + r0) * rb, n_embd, cnt, a, nx, t_u.data(), kGrain);
      for (int j = 0; j < nx; ++j) {
        float* h = hbase + (int64_t(i) * nx + j) * n_ff + r0;
        for (int64_t r = 0; r < cnt; ++r) h[r] = situ(ex.gu_scale * t_g[size_t(j) * kGrain + size_t(r)], ex.gu_scale * t_u[size_t(j) * kGrain + size_t(r)], beta, lb);
      }
    }
  });

  // situ's rows to int8 in the down matrix's layout
  const int64_t nh = int64_t(ne) * nx;
  auto quant_h = [&](int64_t lo, int64_t hi) {
    for (int64_t m = lo; m < hi; ++m) quantize_for(e[m / nx].dn_type, hbase + m * n_ff, n_ff, w.ha[size_t(m)]);
  };
  if (nh >= 16) pool.for_each(nh, 4, quant_h);
  else quant_h(0, nh);

  // down
  const int64_t dchunks = (n_embd + kGrain - 1) / kGrain;
  pool.for_each(int64_t(ne) * dchunks, 1, [&](int64_t lo, int64_t hi) {
    for (int64_t t = lo; t < hi; ++t) {
      const int i = int(t / dchunks);
      const int64_t r0 = (t % dchunks) * kGrain, cnt = std::min(kGrain, n_embd - r0);
      const ExpertRef& ex = e[i];
      const uint64_t rb = qrow_bytes(ex.dn_type, n_ff);
      t_g.resize(size_t(nx) * kGrain);
      gemv(ex.dn_type, ex.dn + uint64_t(r0) * rb, n_ff, cnt, &w.ha[size_t(i) * size_t(nx)], nx, t_g.data(), kGrain);
      for (int j = 0; j < nx; ++j) {
        float* o = y[size_t(i) * size_t(nx) + size_t(j)] + r0;
        for (int64_t r = 0; r < cnt; ++r) o[r] = ex.dn_scale * t_g[size_t(j) * kGrain + size_t(r)];
      }
    }
  });
}

void expert_ffn(const ExpertRef& e, int64_t n_embd, int64_t n_ff, const float* const* x, int nx, float* const* y, float beta,
                float lb, Pool& pool) {
  experts_ffn(&e, 1, n_embd, n_ff, x, nx, y, beta, lb, pool);
}

}  // namespace eklow::cpu
