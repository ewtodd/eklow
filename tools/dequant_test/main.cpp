// eklow-dequant-test: dumps rows of a GGUF's tensors dequantized by the reference kernels, for
// tools/ref/dequant_compare.py to set against the gguf Python package's dequantize on the same
// rows (ggml's dequantize_row_*; a mismatch is a bug in ref.cpp, not noise).
//
//   eklow-dequant-test --model GGUF --out DIR [--tensors NAME,...] [--rows N]
//
// Per tensor (default: the first of each quantization type in the file) N rows (default 64) spread
// over all its rows (a 3-d expert tensor counts as ne[1]*ne[2] rows) are written to DIR/<name>.f32 and
// listed in DIR/rows.json with the row numbers.
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "kernels/ref.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string model, out, names;
  int n_rows = 64;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model = next();
    else if (a == "--out") out = next();
    else if (a == "--tensors") names = next();
    else if (a == "--rows") n_rows = std::atoi(next().c_str());
    else {
      std::fprintf(stderr, "usage: eklow-dequant-test --model GGUF --out DIR [--tensors NAME,...] [--rows N]\n");
      return 2;
    }
  }
  try {
    if (model.empty() || out.empty()) throw std::runtime_error("need --model and --out");
    auto f = gguf::File::open(model);
    std::vector<const gguf::Tensor*> chosen;
    if (!names.empty()) {
      std::stringstream ss(names);
      for (std::string n; std::getline(ss, n, ',');) {
        const gguf::Tensor* t = f.tensor(n);
        if (!t) throw std::runtime_error("no tensor " + n);
        chosen.push_back(t);
      }
    } else {
      std::map<gguf::Type, bool> seen;
      using T = gguf::Type;  // the types ref::matmul has kernels for
      for (const auto& t : f.tensors()) {
        const bool known = t.type == T::F32 || t.type == T::F16 || t.type == T::BF16 || t.type == T::Q4_0 ||
                           t.type == T::Q8_0 || t.type == T::Q6_K || t.type == T::Q4_K || t.type == T::Q5_K ||
                           t.type == T::IQ4_XS;
        if (known && t.ne.size() >= 2 && !seen[t.type]) seen[t.type] = true, chosen.push_back(&t);
      }
    }
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    std::ostringstream index;
    for (const gguf::Tensor* t : chosen) {
      const int64_t k = int64_t(t->ne[0]);
      int64_t rows = 1;
      for (size_t d = 1; d < t->ne.size(); ++d) rows *= int64_t(t->ne[d]);
      const int64_t n = std::min<int64_t>(n_rows, rows);
      std::vector<int32_t> ids(static_cast<size_t>(n));
      for (int64_t i = 0; i < n; ++i) ids[size_t(i)] = int32_t(n == 1 ? 0 : i * (rows - 1) / (n - 1));
      auto* dev = sycl::malloc_device<uint8_t>(t->nbytes, q);
      auto* did = sycl::malloc_device<int32_t>(size_t(n), q);
      auto* dout = sycl::malloc_device<float>(size_t(n * k), q);
      q.memcpy(dev, t->data, t->nbytes);
      q.memcpy(did, ids.data(), size_t(n) * sizeof(int32_t)).wait();
      ref::embed(q, ref::Weight{t->type, k, rows, dev, t->nbytes / uint64_t(rows)}, did, n, 1.0f, dout);
      std::vector<float> host(size_t(n * k));
      q.memcpy(host.data(), dout, host.size() * sizeof(float)).wait();
      sycl::free(dev, q), sycl::free(did, q), sycl::free(dout, q);
      std::ofstream(out + "/" + t->name + ".f32", std::ios::binary)
          .write(reinterpret_cast<const char*>(host.data()), std::streamsize(host.size() * sizeof(float)));
      index << (index.tellp() ? ",\n" : "") << "  {\"name\": \"" << t->name << "\", \"type\": \"" << gguf::type_name(t->type)
            << "\", \"k\": " << k << ", \"rows\": [";
      for (size_t i = 0; i < ids.size(); ++i) index << (i ? ", " : "") << ids[i];
      index << "]}";
      std::printf("%-40s %-6s k %lld, %lld rows dumped\n", t->name.c_str(), gguf::type_name(t->type), (long long)k,
                  (long long)n);
    }
    std::ofstream(out + "/rows.json") << "[\n" << index.str() << "\n]\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-dequant-test: %s\n", e.what());
    return 1;
  }
  return 0;
}
