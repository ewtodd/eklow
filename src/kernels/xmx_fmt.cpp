// Repack and dispatch of the Qwen weight formats (layouts: xmx_fmt.hpp; the decode GEMVs: xmx_fmt_gemv.hpp).
#include "kernels/xmx_fmt.hpp"

#include <stdexcept>

namespace eklow::xmx {

bool format_of(gguf::Type t, WeightFormat& f) {
  switch (t) {
    case gguf::Type::Q4_0: f = WeightFormat::Q4_0; return true;
    case gguf::Type::Q8_0: f = WeightFormat::Q8_0I; return true;
    case gguf::Type::Q4_K: f = WeightFormat::Q4_K; return true;
    case gguf::Type::Q5_K: f = WeightFormat::Q5_K; return true;
    case gguf::Type::Q6_K: f = WeightFormat::Q6_K; return true;
    case gguf::Type::IQ4_XS: f = WeightFormat::IQ4_XS; return true;
    default: return false;
  }
}

const char* format_name(WeightFormat f) {
  switch (f) {
    case WeightFormat::Q4_0: return "Q4_0";
    case WeightFormat::Q8_0: return "Q8_0 (fp16 GEMM layout)";
    case WeightFormat::F16: return "F16";
    case WeightFormat::Q8_0I: return "Q8_0";
    case WeightFormat::Q4_K: return "Q4_K";
    case WeightFormat::Q5_K: return "Q5_K";
    case WeightFormat::Q6_K: return "Q6_K";
    case WeightFormat::IQ4_XS: return "IQ4_XS";
  }
  return "?";
}

uint64_t weight_tile_bytes(WeightFormat f, int64_t k, int64_t n) {
  const fmt::Geo g = fmt::geo(f);
  return uint64_t(n / 8) * uint64_t(k / 32) * uint64_t(g.tile_bytes);
}

uint64_t weight_meta_bytes(WeightFormat f, int64_t k, int64_t n) {
  const fmt::Geo g = fmt::geo(f);
  return uint64_t(n / 8) * uint64_t(k / 32 / g.rec_tiles) * uint64_t(g.rec_bytes);
}

namespace {

// Element e (0..255) of a ggml block's 4-bit-low part and the full quantized value q.
// Q4_K / Q5_K: qs[32 (e / 64) + e % 32], the low nibble for e % 64 < 32, the high otherwise; Q5_K's bit 4 is bit e / 32
// of qh[e % 32].
inline int q45_value(const uint8_t* blk, int e, bool five) {
  const uint8_t* qs = blk + (five ? 48 : 16);
  const uint8_t byte = qs[32 * (e / 64) + e % 32];
  int q = (e % 64) >= 32 ? byte >> 4 : byte & 15;
  if (five) q |= ((blk[16 + e % 32] >> (e / 32)) & 1) << 4;
  return q;
}
// Q6_K: ql[128], qh[64], the 6-bit value (0..63) as ggml's dequantize_row_q6_K reads it.
inline int q6_value(const uint8_t* blk, int e) {
  const int h = e / 128, r = e % 128, l = r % 32, s = r / 32;
  const uint8_t qlb = blk[64 * h + l + 32 * (s & 1)];
  const int nib = s >= 2 ? qlb >> 4 : qlb & 15;
  return nib | ((blk[128 + 32 * h + l] >> (2 * s)) & 3) << 4;
}
// IQ4_XS: qs[16 (e / 32) + e % 16], low nibble for e % 32 < 16.
inline int iq4_code(const uint8_t* blk, int e) {
  const uint8_t byte = blk[8 + 16 * (e / 32) + e % 16];
  return (e % 32) >= 16 ? byte >> 4 : byte & 15;
}

}  // namespace

void repack_format_into(sycl::queue& q, const ref::Weight& raw, bool interleave, const Weight& w) {
  const WeightFormat f = w.format;
  if (!fmt::is_new_format(f)) throw std::runtime_error("xmx: repack_format_into needs a Qwen weight format");
  WeightFormat want;
  if (!format_of(raw.type, want) || want != f) throw std::runtime_error("xmx: the matrix's ggml type is not the target format's");
  const fmt::Geo G = fmt::geo(f);
  const int64_t kmult = f == WeightFormat::Q8_0I ? 32 : 256;
  if (raw.k % kmult || raw.n % 8) throw std::runtime_error("xmx: matrix shape not a multiple of the format's K block x 8");
  if (interleave && raw.n % 16) throw std::runtime_error("xmx: interleaved halves not a multiple of 8 rows");
  if (w.k != raw.k || w.n != raw.n || !w.tiles || !w.scales) throw std::runtime_error("xmx: repack target does not match the matrix");
  const int64_t kb = raw.k / 32, ng = raw.n / 8, nsb = raw.k / 256, half_rows = raw.n / 2;
  const uint8_t* src = raw.data;
  const uint64_t row_bytes = raw.row_bytes;
  uint8_t* tiles = w.tiles;
  uint8_t* meta = reinterpret_cast<uint8_t*>(w.scales);
  if (f == WeightFormat::Q8_0I) {
    // One work-item per tile: eight columns' Q8_0 blocks (fp16 d, 32 int8) into the s8 operand and the scales.
    q.parallel_for(sycl::range<2>(size_t(ng), size_t(kb)), [=](sycl::id<2> id) {
      const int64_t g = int64_t(id[0]), b = int64_t(id[1]);
      uint8_t* t = tiles + (g * kb + b) * 256;
      uint8_t* sc = meta + (g * kb + b) * 16;
      for (int n = 0; n < 8; ++n) {
        const int64_t row = interleave ? (g % 2) * half_rows + (g / 2) * 8 + n : g * 8 + n;
        const uint8_t* blk = src + uint64_t(row) * row_bytes + uint64_t(b) * 34;
        sc[n * 2] = blk[0];
        sc[n * 2 + 1] = blk[1];
        for (int k = 0; k < 32; ++k) t[(k / 4) * 32 + n * 4 + k % 4] = blk[2 + k];
      }
    });
    return;
  }
  // One work-item per (column group, superblock): eight columns' blocks into 8 tiles and the record.
  q.parallel_for(sycl::range<2>(size_t(ng), size_t(nsb)), [=](sycl::id<2> id) {
    const int64_t g = int64_t(id[0]), sb = int64_t(id[1]);
    uint8_t* rec = meta + (g * nsb + sb) * G.rec_bytes;
    uint8_t* tile0 = tiles + (g * kb + sb * 8) * 128;
    for (int n = 0; n < 8; ++n) {
      const int64_t row = interleave ? (g % 2) * half_rows + (g / 2) * 8 + n : g * 8 + n;
      const uint8_t* blk = src + uint64_t(row) * row_bytes + uint64_t(sb) * G.block_bytes;
      if (f == WeightFormat::Q4_K || f == WeightFormat::Q5_K) {
        rec[n * 2] = blk[0];
        rec[n * 2 + 1] = blk[1];
        rec[fmt::kQ4KDmin + n * 2] = blk[2];
        rec[fmt::kQ4KDmin + n * 2 + 1] = blk[3];
        for (int b = 0; b < 12; ++b) rec[fmt::kQ4KScales + b * 8 + n] = blk[4 + b];
      } else if (f == WeightFormat::Q6_K) {
        for (int s = 0; s < 16; ++s) rec[fmt::kQ6KSc + s * 8 + n] = blk[192 + s];
        rec[fmt::kQ6KD + n * 2] = blk[208];
        rec[fmt::kQ6KD + n * 2 + 1] = blk[209];
      } else {
        rec[n * 2] = blk[0];
        rec[n * 2 + 1] = blk[1];
        rec[fmt::kIqScalesH + n * 2] = blk[2];
        rec[fmt::kIqScalesH + n * 2 + 1] = blk[3];
        for (int b = 0; b < 4; ++b) rec[fmt::kIqScalesL + b * 8 + n] = blk[4 + b];
      }
      for (int j = 0; j < 8; ++j) {
        int qv[32];
        for (int kk = 0; kk < 32; ++kk) {
          const int e = j * 32 + kk;
          qv[kk] = f == WeightFormat::Q6_K ? q6_value(blk, e) : f == WeightFormat::IQ4_XS ? iq4_code(blk, e)
                                                                                           : q45_value(blk, e, f == WeightFormat::Q5_K);
        }
        uint8_t* tile = tile0 + j * 128;
        if (f == WeightFormat::IQ4_XS) {
          // Byte (k4 * 8 + n) * 2 + h: the codes of k = 4 k4 + 2 h and the next.
          for (int k4 = 0; k4 < 8; ++k4)
            for (int h = 0; h < 2; ++h) tile[(k4 * 8 + n) * 2 + h] = uint8_t((qv[4 * k4 + 2 * h] & 15) | (qv[4 * k4 + 2 * h + 1] & 15) << 4);
        } else {
          for (int kg = 0; kg < 4; ++kg) {
            uint32_t dw = 0;
            for (int i = 0; i < 8; ++i) dw |= uint32_t(qv[8 * kg + i] & 15) << (4 * i);
            uint8_t* p = tile + (kg * 8 + n) * 4;
            p[0] = uint8_t(dw);
            p[1] = uint8_t(dw >> 8);
            p[2] = uint8_t(dw >> 16);
            p[3] = uint8_t(dw >> 24);
          }
        }
        if (f == WeightFormat::Q5_K) {
          uint32_t hw = 0;
          for (int kk = 0; kk < 32; ++kk) hw |= uint32_t((qv[kk] >> 4) & 1) << (2 * (kk % 16) + kk / 16);
          uint8_t* p = rec + fmt::kQ5KHigh + j * 32 + n * 4;
          p[0] = uint8_t(hw);
          p[1] = uint8_t(hw >> 8);
          p[2] = uint8_t(hw >> 16);
          p[3] = uint8_t(hw >> 24);
        } else if (f == WeightFormat::Q6_K) {
          // The s2 operand: (q >> 4) ^ 2 is q - 32 = low nibble + 16 s2 with s2 in -2..1.
          for (int kg = 0; kg < 2; ++kg) {
            uint32_t dw = 0;
            for (int i = 0; i < 16; ++i) dw |= uint32_t(((qv[16 * kg + i] >> 4) ^ 2) & 3) << (2 * i);
            uint8_t* p = rec + fmt::kQ6KHigh + j * 64 + (kg * 8 + n) * 4;
            p[0] = uint8_t(dw);
            p[1] = uint8_t(dw >> 8);
            p[2] = uint8_t(dw >> 16);
            p[3] = uint8_t(dw >> 24);
          }
        }
      }
    }
  });
}

Weight repack_format(sycl::queue& q, const ref::Weight& raw, WeightFormat f, bool interleave) {
  Weight w;
  w.k = raw.k;
  w.n = raw.n;
  w.format = f;
  w.tiles = sycl::malloc_device<uint8_t>(size_t(weight_tile_bytes(f, raw.k, raw.n)), q);
  w.scales = reinterpret_cast<sycl::half*>(sycl::malloc_device<uint8_t>(size_t(weight_meta_bytes(f, raw.k, raw.n)), q));
  if (!w.tiles || !w.scales) throw std::runtime_error("xmx: out of device memory repacking weights");
  repack_format_into(q, raw, interleave, w);
  q.wait();
  return w;
}

namespace fmt {

void gemv_any(sycl::queue& q, const Activations& a, const Weight& w, float* out) {
  switch (w.format) {
    case WeightFormat::Q8_0I: return gemv<WeightFormat::Q8_0I>(q, a, w, out);
    case WeightFormat::Q4_K: return gemv<WeightFormat::Q4_K>(q, a, w, out);
    case WeightFormat::Q5_K: return gemv<WeightFormat::Q5_K>(q, a, w, out);
    case WeightFormat::Q6_K: return gemv<WeightFormat::Q6_K>(q, a, w, out);
    case WeightFormat::IQ4_XS: return gemv<WeightFormat::IQ4_XS>(q, a, w, out);
    default: throw std::runtime_error("xmx: no decode GEMV for this weight format");
  }
}

void gemv_grouped_any(sycl::queue& q, const Activations& a, const Activations* ad, const ExpertTable& w, const int32_t* ids, int64_t P,
                      int64_t pe, int64_t row_div, float* out) {
  switch (w.format) {
    case WeightFormat::Q8_0I: return gemv_grouped<WeightFormat::Q8_0I>(q, a, ad, w, ids, P, pe, row_div, out);
    case WeightFormat::Q4_K: return gemv_grouped<WeightFormat::Q4_K>(q, a, ad, w, ids, P, pe, row_div, out);
    case WeightFormat::Q5_K: return gemv_grouped<WeightFormat::Q5_K>(q, a, ad, w, ids, P, pe, row_div, out);
    case WeightFormat::Q6_K: return gemv_grouped<WeightFormat::Q6_K>(q, a, ad, w, ids, P, pe, row_div, out);
    case WeightFormat::IQ4_XS: return gemv_grouped<WeightFormat::IQ4_XS>(q, a, ad, w, ids, P, pe, row_div, out);
    default: throw std::runtime_error("xmx: no grouped decode GEMV for this weight format");
  }
}

}  // namespace fmt

}  // namespace eklow::xmx
