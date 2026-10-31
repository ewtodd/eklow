// eklow-tok: eklow's tokenizers from the command line, for comparison against Hugging Face's. The GGUF decides which:
// a Qwen3.6 tokenizer (gpt2 model, qwen35 pre-tokenizer) or Gemma 4's.
//
//   eklow-tok GGUF encode [--no-special] < STRINGS > IDS   STRINGS: NUL-separated; IDS: one line per
//                                                          string, ids separated by spaces
//   eklow-tok GGUF decode [--skip-special] < IDS > STRINGS
//   eklow-tok GGUF bench FILE                              encode FILE, report tokens per second
//   eklow-tok GGUF pretok < STRINGS > SPLITS               (Qwen) NFC and the pre-tokenizer's split: one line per
//                                                          string, the lengths in bytes of its pieces
//   eklow-tok GGUF nfc < STRINGS > STRINGS                 (Qwen) NFC-normalized, NUL-separated
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "model/gguf.hpp"
#include "text/tokenizer.hpp"
#include "text/tokenizer_qwen.hpp"
#include "text/unicode.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: eklow-tok GGUF (encode [--no-special] | decode [--skip-special] | bench FILE | pretok | nfc)\n");
    return 2;
  }
  try {
    auto f = gguf::File::open(argv[1]);
    const auto* model = f.find("tokenizer.ggml.model");
    const bool qwen = model && model->as_string() == "gpt2";
    auto t0 = std::chrono::steady_clock::now();
    std::optional<text::Tokenizer> gemma;
    std::optional<text::QwenTokenizer> q;
    if (qwen) q.emplace(text::QwenTokenizer::from_gguf(f));
    else gemma.emplace(text::Tokenizer::from_gguf(f));
    double load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    auto encode = [&](std::string_view s, bool special) { return qwen ? q->encode(s, special) : gemma->encode(s, special); };
    auto decode = [&](const std::vector<int32_t>& ids, bool skip) { return qwen ? q->decode(ids, skip) : gemma->decode(ids, skip); };
    std::string mode = argv[2];
    bool flag = argc > 3 && (!std::strcmp(argv[3], "--no-special") || !std::strcmp(argv[3], "--skip-special"));
    std::ios::sync_with_stdio(false);
    auto strings = [&](const std::function<void(std::string_view)>& each) {
      std::string in((std::istreambuf_iterator<char>(std::cin)), {});
      size_t start = 0;
      while (start <= in.size()) {
        size_t end = in.find('\0', start);
        if (end == std::string::npos) end = in.size();
        if (end == in.size() && start == end) break;
        each(std::string_view(in).substr(start, end - start));
        start = end + 1;
      }
    };
    if (mode == "encode") {
      strings([&](std::string_view s) {
        auto ids = encode(s, !flag);
        std::string line;
        for (size_t i = 0; i < ids.size(); ++i) line += (i ? " " : "") + std::to_string(ids[i]);
        std::cout << line << '\n';
      });
    } else if (mode == "decode") {
      std::string line;
      while (std::getline(std::cin, line)) {
        std::vector<int32_t> ids;
        std::istringstream ss(line);
        for (int32_t id; ss >> id;) ids.push_back(id);
        std::cout << decode(ids, flag) << '\0';
      }
    } else if (mode == "bench" && argc > 3) {
      std::ifstream in(argv[3], std::ios::binary);
      std::string s((std::istreambuf_iterator<char>(in)), {});
      auto t1 = std::chrono::steady_clock::now();
      auto ids = encode(s, true);
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
      std::printf("load %.2f s; %zu bytes -> %zu tokens in %.3f s (%.0f tokens/s)\n", load, s.size(), ids.size(), secs,
                  ids.size() / secs);
    } else if (qwen && mode == "pretok") {
      strings([&](std::string_view s) {
        std::string line;
        for (auto [b, e] : text::QwenTokenizer::pretokenize(text::unicode::nfc(s)))
          line += (line.empty() ? "" : " ") + std::to_string(e - b);
        std::cout << line << '\n';
      });
    } else if (qwen && mode == "nfc") {
      strings([&](std::string_view s) { std::cout << text::unicode::nfc(s) << '\0'; });
    } else {
      std::fprintf(stderr, "eklow-tok: unknown mode '%s'\n", mode.c_str());
      return 2;
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eklow-tok: %s\n", e.what());
    return 1;
  }
  return 0;
}
