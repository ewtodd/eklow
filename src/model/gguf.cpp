#include "model/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace eklow::gguf {

namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF" little-endian
constexpr uint32_t kMaxDims = 4;         // GGML_MAX_DIMS

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw std::runtime_error(path + ": " + what);
}

// Bounds-checked little-endian reads over the mapping. Every count read from the file is checked
// against the bytes left before anything is allocated for it.
struct Cursor {
  const uint8_t* p;
  uint64_t size;
  uint64_t pos = 0;
  const std::string& path;

  void need(uint64_t n, const char* what) const {
    if (n > size - pos) fail(path, std::string("truncated reading ") + what + " at byte " + std::to_string(pos));
  }
  template <class T>
  T get(const char* what) {
    need(sizeof(T), what);
    T v;
    std::memcpy(&v, p + pos, sizeof(T));
    pos += sizeof(T);
    return v;
  }
  std::string str(const char* what) {
    uint64_t n = get<uint64_t>(what);
    need(n, what);
    std::string s(reinterpret_cast<const char*>(p + pos), n);
    pos += n;
    return s;
  }
};

uint64_t min_bytes(Value::Kind k) {
  switch (k) {
    case Value::Kind::U8: case Value::Kind::I8: case Value::Kind::Bool: return 1;
    case Value::Kind::U16: case Value::Kind::I16: return 2;
    case Value::Kind::U32: case Value::Kind::I32: case Value::Kind::F32: return 4;
    case Value::Kind::U64: case Value::Kind::I64: case Value::Kind::F64: return 8;
    case Value::Kind::String: return 8;  // the length
    case Value::Kind::Array: return 12;  // element type + count
  }
  return 1;
}

bool known_kind(uint32_t k) { return k <= 12; }

Value read_value(Cursor& c, Value::Kind kind, int depth) {
  using K = Value::Kind;
  Value out{kind, K::U8, uint64_t{0}};
  switch (kind) {
    case K::U8: out.v = uint64_t{c.get<uint8_t>("u8")}; break;
    case K::I8: out.v = int64_t{c.get<int8_t>("i8")}; break;
    case K::U16: out.v = uint64_t{c.get<uint16_t>("u16")}; break;
    case K::I16: out.v = int64_t{c.get<int16_t>("i16")}; break;
    case K::U32: out.v = uint64_t{c.get<uint32_t>("u32")}; break;
    case K::I32: out.v = int64_t{c.get<int32_t>("i32")}; break;
    case K::U64: out.v = c.get<uint64_t>("u64"); break;
    case K::I64: out.v = c.get<int64_t>("i64"); break;
    case K::F32: out.v = double{c.get<float>("f32")}; break;
    case K::F64: out.v = c.get<double>("f64"); break;
    case K::Bool: {
      uint8_t b = c.get<uint8_t>("bool");
      if (b > 1) fail(c.path, "bool value " + std::to_string(b) + " at byte " + std::to_string(c.pos - 1));
      out.v = b == 1;
      break;
    }
    case K::String: out.v = c.str("string"); break;
    case K::Array: {
      if (depth > 8) fail(c.path, "arrays nested deeper than 8");
      uint32_t et = c.get<uint32_t>("array type");
      if (!known_kind(et)) fail(c.path, "unknown array element type " + std::to_string(et));
      uint64_t n = c.get<uint64_t>("array length");
      // Each element takes at least min_bytes, so a count beyond what is left is a corrupt file.
      if (n > (c.size - c.pos) / min_bytes(K(et))) fail(c.path, "array of " + std::to_string(n) + " elements runs past the file");
      Array a;
      a.reserve(n);
      for (uint64_t i = 0; i < n; ++i) a.push_back(read_value(c, K(et), depth + 1));
      out.elem = K(et);
      out.v = std::move(a);
      break;
    }
  }
  return out;
}

}  // namespace

std::optional<TypeTraits> traits(Type t) {
  switch (t) {
    case Type::F32: return TypeTraits{"F32", 1, 4};
    case Type::F16: return TypeTraits{"F16", 1, 2};
    case Type::BF16: return TypeTraits{"BF16", 1, 2};
    case Type::F64: return TypeTraits{"F64", 1, 8};
    case Type::I8: return TypeTraits{"I8", 1, 1};
    case Type::I16: return TypeTraits{"I16", 1, 2};
    case Type::I32: return TypeTraits{"I32", 1, 4};
    case Type::I64: return TypeTraits{"I64", 1, 8};
    case Type::Q4_0: return TypeTraits{"Q4_0", 32, 18};
    case Type::Q4_1: return TypeTraits{"Q4_1", 32, 20};
    case Type::Q5_0: return TypeTraits{"Q5_0", 32, 22};
    case Type::Q5_1: return TypeTraits{"Q5_1", 32, 24};
    case Type::Q8_0: return TypeTraits{"Q8_0", 32, 34};
    case Type::Q8_1: return TypeTraits{"Q8_1", 32, 36};
    case Type::Q2_K: return TypeTraits{"Q2_K", 256, 84};
    case Type::Q3_K: return TypeTraits{"Q3_K", 256, 110};
    case Type::Q4_K: return TypeTraits{"Q4_K", 256, 144};
    case Type::Q5_K: return TypeTraits{"Q5_K", 256, 176};
    case Type::Q6_K: return TypeTraits{"Q6_K", 256, 210};
    case Type::Q8_K: return TypeTraits{"Q8_K", 256, 292};
    case Type::IQ4_NL: return TypeTraits{"IQ4_NL", 32, 18};
    case Type::IQ4_XS: return TypeTraits{"IQ4_XS", 256, 136};
    case Type::MXFP4: return TypeTraits{"MXFP4", 32, 17};
    default: return std::nullopt;
  }
}

const char* type_name(Type t) {
  auto tr = traits(t);
  return tr ? tr->name : "unknown";
}

const char* kind_name(Value::Kind k) {
  static const char* names[] = {"u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "array", "u64", "i64", "f64"};
  auto i = static_cast<uint32_t>(k);
  return i < std::size(names) ? names[i] : "unknown";
}

int64_t Value::as_int() const {
  if (auto* u = std::get_if<uint64_t>(&v)) {
    if (*u > uint64_t(std::numeric_limits<int64_t>::max())) throw std::runtime_error("metadata integer out of range");
    return int64_t(*u);
  }
  if (auto* i = std::get_if<int64_t>(&v)) return *i;
  throw std::runtime_error(std::string("metadata value is ") + kind_name(kind) + ", not an integer");
}

uint64_t Value::as_uint() const {
  if (auto* u = std::get_if<uint64_t>(&v)) return *u;
  if (auto* i = std::get_if<int64_t>(&v)) {
    if (*i < 0) throw std::runtime_error("metadata integer is negative");
    return uint64_t(*i);
  }
  throw std::runtime_error(std::string("metadata value is ") + kind_name(kind) + ", not an integer");
}

double Value::as_float() const {
  if (auto* d = std::get_if<double>(&v)) return *d;
  if (auto* u = std::get_if<uint64_t>(&v)) return double(*u);
  if (auto* i = std::get_if<int64_t>(&v)) return double(*i);
  throw std::runtime_error(std::string("metadata value is ") + kind_name(kind) + ", not a number");
}

const std::string& Value::as_string() const {
  if (auto* s = std::get_if<std::string>(&v)) return *s;
  throw std::runtime_error(std::string("metadata value is ") + kind_name(kind) + ", not a string");
}

const Array& Value::as_array() const {
  if (auto* a = std::get_if<Array>(&v)) return *a;
  throw std::runtime_error(std::string("metadata value is ") + kind_name(kind) + ", not an array");
}

uint64_t Tensor::elements() const {
  uint64_t n = 1;
  for (auto d : ne) n *= d;
  return n;
}

File File::open(const std::string& path) {
  File f;
  f.path_ = path;
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) fail(path, std::strerror(errno));
  struct stat st {};
  if (fstat(fd, &st) != 0) {
    int e = errno;
    ::close(fd);
    fail(path, std::strerror(e));
  }
  f.size_ = uint64_t(st.st_size);
  if (f.size_ < 24) {
    ::close(fd);
    fail(path, "too small for a GGUF header");
  }
  void* m = mmap(nullptr, f.size_, PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (m == MAP_FAILED) fail(path, std::string("mmap: ") + std::strerror(errno));
  f.base_ = static_cast<const uint8_t*>(m);

  Cursor c{f.base_, f.size_, 0, f.path_};
  if (c.get<uint32_t>("magic") != kMagic) fail(path, "not a GGUF file (bad magic)");
  f.version_ = c.get<uint32_t>("version");
  if (f.version_ < 2 || f.version_ > 3) fail(path, "GGUF version " + std::to_string(f.version_) + " (only 2 and 3 are read)");
  uint64_t n_tensors = c.get<uint64_t>("tensor count");
  uint64_t n_kv = c.get<uint64_t>("metadata count");

  for (uint64_t i = 0; i < n_kv; ++i) {
    std::string key = c.str("metadata key");
    uint32_t kind = c.get<uint32_t>("metadata type");
    if (!known_kind(kind)) fail(path, "metadata '" + key + "' has unknown type " + std::to_string(kind));
    Value val = read_value(c, Value::Kind(kind), 0);
    if (!f.kv_.emplace(key, std::move(val)).second) fail(path, "duplicate metadata key '" + key + "'");
  }

  if (auto* a = f.find("general.alignment")) {
    if (a->kind != Value::Kind::U32) fail(path, "general.alignment is not a u32");
    f.alignment_ = a->as_uint();
    if (f.alignment_ == 0 || (f.alignment_ & (f.alignment_ - 1)) != 0)
      fail(path, "general.alignment " + std::to_string(f.alignment_) + " is not a power of two");
  }

  // Each tensor info takes at least a name length, a dim count, one dim, a type and an offset.
  if (n_tensors > (c.size - c.pos) / 32) fail(path, "tensor count " + std::to_string(n_tensors) + " runs past the file");
  f.tensors_.reserve(n_tensors);
  for (uint64_t i = 0; i < n_tensors; ++i) {
    Tensor t;
    t.name = c.str("tensor name");
    uint32_t nd = c.get<uint32_t>("tensor dims");
    if (nd == 0 || nd > kMaxDims) fail(path, "tensor '" + t.name + "' has " + std::to_string(nd) + " dims");
    uint64_t elems = 1;
    for (uint32_t d = 0; d < nd; ++d) {
      uint64_t n = c.get<uint64_t>("tensor shape");
      if (n == 0 || n > uint64_t(std::numeric_limits<int64_t>::max()) / elems)
        fail(path, "tensor '" + t.name + "' has an empty or overflowing shape");
      elems *= n;
      t.ne.push_back(n);
    }
    t.type = Type(c.get<uint32_t>("tensor type"));
    auto tr = traits(t.type);
    if (!tr) fail(path, "tensor '" + t.name + "' has unknown type " + std::to_string(uint32_t(t.type)));
    if (t.ne[0] % tr->block != 0)
      fail(path, "tensor '" + t.name + "' row of " + std::to_string(t.ne[0]) + " is not a multiple of " + tr->name + "'s block " + std::to_string(tr->block));
    if (elems / tr->block > std::numeric_limits<uint64_t>::max() / tr->bytes)
      fail(path, "tensor '" + t.name + "' size overflows");
    t.nbytes = elems / tr->block * tr->bytes;
    t.offset = c.get<uint64_t>("tensor offset");
    if (t.offset % f.alignment_ != 0) fail(path, "tensor '" + t.name + "' offset is not aligned");
    if (!f.by_name_.emplace(t.name, f.tensors_.size()).second) fail(path, "duplicate tensor '" + t.name + "'");
    f.tensors_.push_back(std::move(t));
  }

  f.data_offset_ = (c.pos + f.alignment_ - 1) / f.alignment_ * f.alignment_;
  if (f.data_offset_ > f.size_) fail(path, "data section starts past the end of the file");
  const uint64_t data_size = f.size_ - f.data_offset_;

  // Tensors must lie inside the data section and must not overlap.
  std::vector<const Tensor*> order;
  for (auto& t : f.tensors_) {
    if (t.offset > data_size || t.nbytes > data_size - t.offset)
      fail(path, "tensor '" + t.name + "' runs past the end of the file");
    order.push_back(&t);
  }
  std::sort(order.begin(), order.end(), [](auto* a, auto* b) { return a->offset < b->offset; });
  for (size_t i = 1; i < order.size(); ++i)
    if (order[i - 1]->offset + order[i - 1]->nbytes > order[i]->offset)
      fail(path, "tensors '" + order[i - 1]->name + "' and '" + order[i]->name + "' overlap");
  for (auto& t : f.tensors_) t.data = f.base_ + f.data_offset_ + t.offset;
  return f;
}

File::File(File&& o) noexcept { *this = std::move(o); }

File& File::operator=(File&& o) noexcept {
  if (this != &o) {
    if (base_) munmap(const_cast<uint8_t*>(base_), size_);
    path_ = std::move(o.path_);
    base_ = o.base_;
    size_ = o.size_;
    version_ = o.version_;
    alignment_ = o.alignment_;
    data_offset_ = o.data_offset_;
    kv_ = std::move(o.kv_);
    tensors_ = std::move(o.tensors_);
    by_name_ = std::move(o.by_name_);
    o.base_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

File::~File() {
  if (base_) munmap(const_cast<uint8_t*>(base_), size_);
}

const Value* File::find(std::string_view key) const {
  auto it = kv_.find(key);
  return it == kv_.end() ? nullptr : &it->second;
}

const Value& File::at(std::string_view key) const {
  if (auto* v = find(key)) return *v;
  throw std::runtime_error(path_ + ": no metadata key '" + std::string(key) + "'");
}

const Tensor* File::tensor(std::string_view name) const {
  auto it = by_name_.find(name);
  return it == by_name_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace eklow::gguf
