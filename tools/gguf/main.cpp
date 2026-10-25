// eklow-gguf: read a GGUF file with eklow's loader and print what it found.
//
//   eklow-gguf FILE              summary: metadata, tensor types and sizes
//   eklow-gguf FILE --tensors    plus one line per tensor
//   eklow-gguf FILE --json       everything as JSON (tools/ref/gguf_compare.py checks it against
//                                the gguf Python package); arrays give their length and first and
//                                last 8 elements
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

#include "core/json.hpp"
#include "model/gguf.hpp"

using namespace eklow;

namespace {

std::string scalar_text(const gguf::Value& v) {
  using K = gguf::Value::Kind;
  switch (v.kind) {
    case K::String: {
      const auto& s = v.as_string();
      return s.size() > 80 ? json::escape(s.substr(0, 77)) + "..." : json::escape(s);
    }
    case K::Bool: return std::get<bool>(v.v) ? "true" : "false";
    case K::F32: case K::F64: {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%g", v.as_float());
      return buf;
    }
    case K::Array: return "[array]";
    default:
      if (auto* u = std::get_if<uint64_t>(&v.v)) return std::to_string(*u);
      return std::to_string(std::get<int64_t>(v.v));
  }
}

void write_value(json::Writer& w, const gguf::Value& v) {
  using K = gguf::Value::Kind;
  switch (v.kind) {
    case K::String: w.value(v.as_string()); break;
    case K::Bool: w.value(std::get<bool>(v.v)); break;
    case K::F32: case K::F64: w.value(v.as_float()); break;
    case K::Array: {
      const auto& a = v.as_array();
      w.begin_object().kv("type", gguf::kind_name(v.elem)).kv("length", uint64_t(a.size()));
      w.key("head").begin_array();
      for (size_t i = 0; i < a.size() && i < 8; ++i) write_value(w, a[i]);
      w.end_array();
      w.key("tail").begin_array();
      for (size_t i = a.size() > 8 ? a.size() - 8 : 0; i < a.size(); ++i) write_value(w, a[i]);
      w.end_array().end_object();
      break;
    }
    default:
      if (auto* u = std::get_if<uint64_t>(&v.v)) w.value(*u);
      else w.value(std::get<int64_t>(v.v));
  }
}

int usage() {
  std::fprintf(stderr, "usage: eklow-gguf FILE [--tensors] [--json]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  bool as_json = false, list = false;
  for (int i = 2; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--json")) as_json = true;
    else if (!std::strcmp(argv[i], "--tensors")) list = true;
    else return usage();
  }
  try {
    auto f = gguf::File::open(argv[1]);
    if (as_json) {
      json::Writer w;
      w.begin_object().kv("path", f.path()).kv("version", uint64_t(f.version())).kv("alignment", f.alignment())
          .kv("data_offset", f.data_offset()).kv("size", f.size());
      w.key("metadata").begin_object();
      for (const auto& [k, v] : f.metadata()) {
        w.key(k).begin_object().kv("type", gguf::kind_name(v.kind)).key("value");
        write_value(w, v);
        w.end_object();
      }
      w.end_object();
      w.key("tensors").begin_array();
      for (const auto& t : f.tensors()) {
        w.begin_object().kv("name", t.name).kv("type", gguf::type_name(t.type)).key("ne").begin_array();
        for (auto d : t.ne) w.value(d);
        w.end_array().kv("offset", t.offset).kv("nbytes", t.nbytes).end_object();
      }
      w.end_array().end_object();
      std::printf("%s\n", w.str().c_str());
      return 0;
    }

    std::printf("%s: GGUF v%u, %zu metadata keys, %zu tensors, data at %llu, alignment %llu, %.2f GiB\n",
                f.path().c_str(), f.version(), f.metadata().size(), f.tensors().size(),
                (unsigned long long)f.data_offset(), (unsigned long long)f.alignment(), f.size() / 1073741824.0);
    for (const auto& [k, v] : f.metadata()) {
      if (v.kind == gguf::Value::Kind::Array) {
        const auto& a = v.as_array();
        std::string head;
        for (size_t i = 0; i < a.size() && i < 4; ++i) head += (i ? ", " : "") + scalar_text(a[i]);
        std::printf("  %-48s [%zu x %s] %s%s\n", k.c_str(), a.size(), gguf::kind_name(v.elem), head.c_str(),
                    a.size() > 4 ? ", ..." : "");
      } else {
        std::printf("  %-48s %-6s %s\n", k.c_str(), gguf::kind_name(v.kind), scalar_text(v).c_str());
      }
    }
    std::map<std::string, std::pair<size_t, uint64_t>> by_type;
    uint64_t total = 0;
    for (const auto& t : f.tensors()) {
      auto& e = by_type[gguf::type_name(t.type)];
      e.first++;
      e.second += t.nbytes;
      total += t.nbytes;
    }
    std::printf("tensor data: %.3f GiB\n", total / 1073741824.0);
    for (const auto& [name, e] : by_type)
      std::printf("  %-8s %5zu tensors %10.3f GiB\n", name.c_str(), e.first, e.second / 1073741824.0);
    if (list) {
      for (const auto& t : f.tensors()) {
        std::string shape;
        for (size_t d = 0; d < t.ne.size(); ++d) shape += (d ? " x " : "") + std::to_string(t.ne[d]);
        std::printf("  %-48s %-6s %-22s @%-12llu %llu\n", t.name.c_str(), gguf::type_name(t.type), shape.c_str(),
                    (unsigned long long)t.offset, (unsigned long long)t.nbytes);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-gguf: %s\n", e.what());
    return 1;
  }
  return 0;
}
