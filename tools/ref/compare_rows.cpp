// compare_rows: two f32 files of [rows][dim] (eklow's and the oracle's), per-row cosine and relative L2.
//
//   compare_rows A.f32 B.f32 DIM       build: c++ -O2 -std=c++17 compare_rows.cpp -o compare_rows
// Prints rows, the minimum and mean cosine, the largest per-row relative L2 (|a-b| / |b|), the
// relative L2 of the whole matrix and the row where each worst case is. Exit 1 if the sizes differ.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

static std::vector<float> load(const char* p) {
  std::ifstream f(p, std::ios::binary);
  std::vector<char> b{std::istreambuf_iterator<char>(f), {}};
  std::vector<float> v(b.size() / sizeof(float));
  std::copy(b.begin(), b.begin() + std::ptrdiff_t(v.size() * sizeof(float)), reinterpret_cast<char*>(v.data()));
  return v;
}

int main(int argc, char** argv) {
  if (argc != 4) return std::fprintf(stderr, "usage: compare_rows A.f32 B.f32 DIM\n"), 2;
  const auto a = load(argv[1]), b = load(argv[2]);
  const size_t dim = size_t(std::atol(argv[3]));
  if (a.size() != b.size() || !dim || a.size() % dim) {
    std::printf("size mismatch: %zu vs %zu floats (dim %zu)\n", a.size(), b.size(), dim);
    return 1;
  }
  const size_t rows = a.size() / dim;
  double min_cos = 2, sum_cos = 0, max_rel = 0, tot_d = 0, tot_b = 0;
  size_t min_row = 0, max_row = 0;
  for (size_t r = 0; r < rows; ++r) {
    double ab = 0, aa = 0, bb = 0, dd = 0;
    for (size_t i = 0; i < dim; ++i) {
      const double x = a[r * dim + i], y = b[r * dim + i];
      ab += x * y, aa += x * x, bb += y * y, dd += (x - y) * (x - y);
    }
    const double c = ab / (std::sqrt(aa * bb) + 1e-30), rel = std::sqrt(dd / (bb + 1e-30));
    if (c < min_cos) min_cos = c, min_row = r;
    if (rel > max_rel) max_rel = rel, max_row = r;
    sum_cos += c, tot_d += dd, tot_b += bb;
  }
  std::printf("rows %zu  cos min %.6f (row %zu) mean %.6f  relL2 max %.3e (row %zu) total %.3e\n", rows, min_cos, min_row,
              sum_cos / double(rows), max_rel, max_row, std::sqrt(tot_d / (tot_b + 1e-30)));
  return 0;
}
