// The file formats vega's checkpoints and laya's policies come in, read without Python: safetensors (the engine and the
// port), NumPy's .npy and .npz (the reader's feature statistics, a policy's heads), and SHA-256 (a policy's manifest
// names every file's, and the engine's, which laya checks before it installs a head).
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "core/json_value.hpp"

namespace eklow::vega {

std::string read_file(const std::string& path);

// Lower-case hex SHA-256.
std::string sha256_hex(std::string_view bytes);

// One array. dtype is NumPy's descr ("<f4", "<f8", "<i4", "<i8", "|b1", "<U12", ...); data holds its bytes, C order.
struct Array {
  std::string dtype;
  std::vector<int64_t> shape;
  std::string data;
  int64_t size() const;
  // The values as float, converting from f4, f8, i4, i8 and b1; throws for others.
  std::vector<float> floats() const;
  // A 0-d or 1-element '<U' array as UTF-8.
  std::string unicode() const;
};

Array parse_npy(std::string_view bytes, const std::string& what);
// The arrays of an .npz, by name without ".npy": a zip of .npy members stored uncompressed, as np.savez writes them
// (zip64 records included); a compressed member is refused.
std::map<std::string, Array> parse_npz(std::string_view bytes, const std::string& what);
// Writes one array as .npy (float32, C order).
void write_npy(const std::string& path, const std::vector<int64_t>& shape, const float* data);

// A safetensors file's tensors as float32 (F32 and BF16 stored), by name.
struct Tensor {
  std::vector<int64_t> shape;
  std::vector<float> data;
};
std::map<std::string, Tensor> read_safetensors(const std::string& path);

// json.dumps as Python writes it: floats as repr, sort_keys, compact (",", ":") or the default (", ", ": ") separators,
// ensure_ascii. vegaml hashes a calibration in this form and renders a structured state or a non-string instruction
// with it.
std::string python_json(const json::Value& v, bool sort_keys, bool compact, bool ensure_ascii);

}  // namespace eklow::vega
