// GGUF reader: the file is mmapped read-only, the header (metadata and tensor table) is parsed and
// checked, and tensor data is handed out as pointers into the mapping. Nothing is copied; the engine
// rearranges weights for XMX when it uploads them.
//
// Format: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md (versions 2 and 3; v1 used
// 32-bit counts and is refused).
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace eklow::gguf {

// ggml tensor types, by their numbers in the file.
enum class Type : uint32_t {
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
  Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
  IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22,
  IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
  TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39,
};

struct TypeTraits {
  const char* name;
  uint32_t block;  // elements per block
  uint32_t bytes;  // bytes per block
};

// Traits of a known type, or nullopt (a tensor of an unknown type makes open() fail).
std::optional<TypeTraits> traits(Type t);
const char* type_name(Type t);

// A metadata value. Arrays hold one element type; nested arrays are allowed by the format and kept.
struct Value;
using Array = std::vector<Value>;
struct Value {
  enum class Kind : uint32_t {
    U8 = 0, I8 = 1, U16 = 2, I16 = 3, U32 = 4, I32 = 5, F32 = 6, Bool = 7, String = 8, Array = 9,
    U64 = 10, I64 = 11, F64 = 12,
  };
  Kind kind;
  Kind elem = Kind::U8;  // for arrays
  std::variant<uint64_t, int64_t, double, bool, std::string, Array> v;

  // Integer value of any integer kind; throws when the value is not an integer or out of range.
  int64_t as_int() const;
  uint64_t as_uint() const;
  double as_float() const;  // F32/F64, or an integer
  const std::string& as_string() const;
  const Array& as_array() const;
};

const char* kind_name(Value::Kind k);

struct Tensor {
  std::string name;
  Type type;
  std::vector<uint64_t> ne;  // ne[0] is the contiguous (row) dimension, as in ggml
  uint64_t offset;           // from the start of the data section
  uint64_t nbytes;
  const uint8_t* data;       // into the mapping

  uint64_t elements() const;
};

// Keys compare with std::less<> so lookups take string_views.
using Metadata = std::map<std::string, Value, std::less<>>;

class File {
 public:
  // Maps and parses `path`; throws std::runtime_error naming the file and what is wrong.
  static File open(const std::string& path);
  File(File&&) noexcept;
  File& operator=(File&&) noexcept;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File();

  uint32_t version() const { return version_; }
  uint64_t alignment() const { return alignment_; }
  uint64_t data_offset() const { return data_offset_; }
  uint64_t size() const { return size_; }
  const std::string& path() const { return path_; }

  const Metadata& metadata() const { return kv_; }
  // nullptr when the key is absent.
  const Value* find(std::string_view key) const;
  // Throws when the key is absent.
  const Value& at(std::string_view key) const;

  const std::vector<Tensor>& tensors() const { return tensors_; }
  const Tensor* tensor(std::string_view name) const;

 private:
  File() = default;
  std::string path_;
  const uint8_t* base_ = nullptr;
  uint64_t size_ = 0;
  uint32_t version_ = 0;
  uint64_t alignment_ = 32;
  uint64_t data_offset_ = 0;
  Metadata kv_;
  std::vector<Tensor> tensors_;
  std::map<std::string, size_t, std::less<>> by_name_;
};

}  // namespace eklow::gguf
