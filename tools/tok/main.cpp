// eklow-tok: eklow's tokenizer from the command line, for comparison against Hugging Face's.
//
//   eklow-tok GGUF encode [--no-special] < STRINGS > IDS   STRINGS: NUL-separated; IDS: one line per
//                                                          string, ids separated by spaces
//   eklow-tok GGUF decode [--skip-special] < IDS > STRINGS
//   eklow-tok GGUF bench FILE                              encode FILE, report tokens per second
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "model/gguf.hpp"
#include "text/tokenizer.hpp"

using namespace eklow;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: eklow-tok GGUF (encode [--no-special] | decode [--skip-special] | bench FILE)\n");
    return 2;
  }
  try {
    auto f = gguf::File::open(argv[1]);
    auto t0 = std::chrono::steady_clock::now();
    auto tok = text::Tokenizer::from_gguf(f);
    double load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string mode = argv[2];
    bool flag = argc > 3 && (!std::strcmp(argv[3], "--no-special") || !std::strcmp(argv[3], "--skip-special"));
    std::ios::sync_with_stdio(false);
    if (mode == "encode") {
      std::string in((std::istreambuf_iterator<char>(std::cin)), {});
      size_t start = 0;
      while (start <= in.size()) {
        size_t end = in.find('\0', start);
        if (end == std::string::npos) end = in.size();
        if (end == in.size() && start == end) break;
        auto ids = tok.encode(std::string_view(in).substr(start, end - start), !flag);
        std::string line;
        for (size_t i = 0; i < ids.size(); ++i) line += (i ? " " : "") + std::to_string(ids[i]);
        std::cout << line << '\n';
        start = end + 1;
      }
    } else if (mode == "decode") {
      std::string line;
      while (std::getline(std::cin, line)) {
        std::vector<int32_t> ids;
        std::istringstream ss(line);
        for (int32_t id; ss >> id;) ids.push_back(id);
        std::cout << tok.decode(ids, flag) << '\0';
      }
    } else if (mode == "bench" && argc > 3) {
      std::ifstream in(argv[3], std::ios::binary);
      std::string s((std::istreambuf_iterator<char>(in)), {});
      auto t1 = std::chrono::steady_clock::now();
      auto ids = tok.encode(s);
      double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
      std::printf("load %.2f s; %zu bytes -> %zu tokens in %.3f s (%.0f tokens/s)\n", load, s.size(), ids.size(), secs,
                  ids.size() / secs);
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
