// eklow-embed: EmbeddingGemma 2 on the card, one input per line of stdin, one JSON line out each:
// {"tokens": N, "embedding": [...]} (normalized). Each input is BOS, its tokens, EOS, as
// llama.cpp's server tokenizes an embedding input.
//
//   eklow-embed --model GGUF < lines
#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

#include "core/json.hpp"
#include "model/embed_gemma2.hpp"
#include "model/gguf.hpp"
#include "rt/device.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  std::string path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) path = argv[++i];
    else {
      std::fprintf(stderr, "usage: eklow-embed --model GGUF < lines\n");
      return 2;
    }
  }
  try {
    auto f = gguf::File::open(path);
    const text::Tokenizer tok = text::Tokenizer::from_gguf(f);
    sycl::queue q(rt::pick_device(), sycl::property::queue::in_order());
    model::EmbedGemma2 m(q, f);
    std::string line;
    while (std::getline(std::cin, line)) {
      std::vector<int32_t> ids{tok.bos()};
      for (int32_t id : tok.encode(line)) ids.push_back(id);
      ids.push_back(tok.eos());
      auto t0 = std::chrono::steady_clock::now();
      const std::vector<float> e = m.embed(ids);
      const double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      json::Writer w(false);
      w.begin_object().kv("tokens", int64_t(ids.size())).kv("ms", ms).key("embedding").begin_array();
      for (float v : e) w.value(double(v));
      w.end_array().end_object();
      std::cout << w.str() << std::endl;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-embed: %s\n", e.what());
    return 1;
  }
  return 0;
}
