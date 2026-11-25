// Quantize + GEMV of Q8_0I in one launch (xmx_fused.hpp), a file of its own so that it compiles beside the others.
#include <cstdlib>

#include "kernels/xmx_fused.hpp"

namespace eklow::xmx {

std::vector<sycl::event>* g_event_log = nullptr;

namespace {

int env_int(const char* name, int def) {
  const char* v = std::getenv(name);
  return v ? std::atoi(v) : def;
}

template <int R, int NG>
void dispatch_t(int T, sycl::queue& q, const fmt::fused::JobDev (&jobs)[fmt::fused::kMaxJobs], int nj, int64_t k, int m) {
  switch (T) {
    case 1: return fmt::fused::launch<R, NG, 2, 1>(q, jobs, nj, k, m);
    case 2: return fmt::fused::launch<R, NG, 2, 2>(q, jobs, nj, k, m);
    case 4: return fmt::fused::launch<R, NG, 2, 4>(q, jobs, nj, k, m);
    case 8: return fmt::fused::launch<R, NG, 2, 8>(q, jobs, nj, k, m);
    case 16: return fmt::fused::launch<R, NG, 2, 16>(q, jobs, nj, k, m);
    default: return fmt::fused::launch<R, NG, 2, 32>(q, jobs, nj, k, m);
  }
}

template <int R>
void dispatch_ng(int NG, int T, sycl::queue& q, const fmt::fused::JobDev (&jobs)[fmt::fused::kMaxJobs], int nj, int64_t k, int m) {
  if (NG == 1) dispatch_t<R, 1>(T, q, jobs, nj, k, m);
  else if (NG == 2) dispatch_t<R, 2>(T, q, jobs, nj, k, m);
  else dispatch_t<R, 4>(T, q, jobs, nj, k, m);
}

}  // namespace

void matmul_fused(sycl::queue& q, const FusedJob* jobs, int njobs, int64_t m) {
  using namespace fmt::fused;
  if (njobs < 1 || njobs > kMaxJobs) throw std::runtime_error("xmx: matmul_fused takes 1 to 4 matrices");
  if (m < 1 || m > 2) throw std::runtime_error("xmx: matmul_fused takes 1 or 2 rows (more: quantize + matmul)");
  const int64_t k = jobs[0].w->k;
  const int64_t kb = k / 32;
  if (k % 64) throw std::runtime_error("xmx: matmul_fused needs K a multiple of 64");
  int NG = env_int("EKLOW_FUSED_NG", 2);
  for (int i = 0; i < njobs; ++i) {
    if (jobs[i].w->format != WeightFormat::Q8_0I || jobs[i].w->k != k) throw std::runtime_error("xmx: matmul_fused needs Q8_0I matrices of one K");
    while ((jobs[i].w->n / 8) % NG) NG /= 2;
  }
  JobDev dev[kMaxJobs] = {};
  int64_t slot_end = 0;
  for (int i = 0; i < njobs; ++i) {
    const Weight& w = *jobs[i].w;
    slot_end += w.n / 8 / NG;
    dev[i] = {w.tiles, reinterpret_cast<const uint8_t*>(w.scales), jobs[i].x, jobs[i].out, int32_t(w.n), int32_t(slot_end)};
  }
  for (int i = njobs; i < kMaxJobs; ++i) dev[i] = dev[njobs - 1];
  // K split: about 512 threads a row in all (more were slower in every shape tried), so a small matrix splits K further
  int T = 1;
  while (T < 16 && int64_t(T) * 2 * slot_end <= 512 * m) T *= 2;
  while (T > 1 && int64_t(T) * 2 > kb) T /= 2;
  if (int t = env_int("EKLOW_FUSED_T", 0)) T = t;
  if (m == 1) dispatch_ng<1>(NG, T, q, dev, njobs, k, int(m));
  else dispatch_ng<2>(NG, T, q, dev, njobs, k, int(m));
}

}  // namespace eklow::xmx
