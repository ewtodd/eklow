// eklow-grammar-oracle: llama.cpp's grammar sampler (llama_sampler_init_grammar, as llama-server builds it for a request
// whose chat format returns a grammar with grammar_lazy false) as the reference for eklow's tool-call grammars.
//
//   eklow-grammar-oracle --model GGUF --cases CASES.txt --out RESULT.txt
//   eklow-grammar-oracle --model GGUF --dump-vocab VOCAB.txt
//
// CASES.txt, one command a line:
//   G PATH                  start a new grammar sampler from the GBNF text in the file PATH (root rule "root")
//   S N ID ID ...           a token sequence of N ids, run against the current grammar
// For each S the oracle writes, for the state before each token and then for the state after the last one, which token
// ids the sampler leaves allowed (every id below the vocabulary's size is a candidate, as the server's candidates are):
//   A K ID ...              the K allowed ids (written when K is at most half the vocabulary), ascending
//   N K ID ...              all ids but these K
// The run stops at the first token the sampler does not allow (it is not accepted) or at an end-of-generation token that
// it does. The last line of each S is
//   Z ok | rej J | eog J | err J TEXT      (J: the index of the token that stopped it)
// --dump-vocab writes "ID EOG HEX" per id: whether the vocabulary says end-of-generation and the token's piece as llama.cpp's
// grammar sees it (llama_token_to_piece, special tokens as text), in hex.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "llama.h"

namespace {

std::string piece_of(const llama_vocab* vocab, llama_token t) {
  std::vector<char> buf(256);
  int n = llama_token_to_piece(vocab, t, buf.data(), int32_t(buf.size()), 0, true);
  if (n < 0) {
    buf.resize(size_t(-n));
    n = llama_token_to_piece(vocab, t, buf.data(), int32_t(buf.size()), 0, true);
  }
  return std::string(buf.data(), size_t(n > 0 ? n : 0));
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    std::exit(1);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, cases_path, out_path, vocab_path;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model_path = next();
    else if (a == "--cases") cases_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--dump-vocab") vocab_path = next();
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (model_path.empty() || (vocab_path.empty() && (cases_path.empty() || out_path.empty()))) {
    std::fprintf(stderr, "usage: eklow-grammar-oracle --model GGUF (--cases CASES --out OUT | --dump-vocab OUT)\n");
    return 2;
  }
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!model) {
    std::fprintf(stderr, "cannot load %s\n", model_path.c_str());
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int32_t n = llama_vocab_n_tokens(vocab);

  if (!vocab_path.empty()) {
    std::ofstream o(vocab_path);
    for (int32_t id = 0; id < n; ++id) {
      o << id << ' ' << int(llama_vocab_is_eog(vocab, id)) << ' ';
      for (unsigned char c : piece_of(vocab, id)) {
        static const char* hx = "0123456789abcdef";
        o << hx[c >> 4] << hx[c & 15];
      }
      o << '\n';
    }
    return 0;
  }

  std::ifstream cases(cases_path);
  std::ofstream out(out_path);
  std::string line;
  llama_sampler* grammar = nullptr;
  std::string grammar_text;
  std::vector<llama_token_data> data(static_cast<size_t>(n));

  // The allowed ids of the sampler's current state, ascending.
  auto allowed = [&]() {
    for (int32_t i = 0; i < n; ++i) data[size_t(i)] = {i, 0.0f, 0.0f};
    llama_token_data_array arr = {data.data(), data.size(), -1, false};
    llama_sampler_apply(grammar, &arr);
    std::vector<int32_t> ok, no;
    for (int32_t i = 0; i < n; ++i) (std::isinf(data[size_t(i)].logit) ? no : ok).push_back(i);
    return std::pair(std::move(ok), std::move(no));
  };
  auto write_set = [&](const std::vector<int32_t>& ok, const std::vector<int32_t>& no) {
    const bool small = ok.size() <= no.size();
    const auto& v = small ? ok : no;
    out << (small ? 'A' : 'N') << ' ' << v.size();
    for (int32_t id : v) out << ' ' << id;
    out << '\n';
  };

  while (std::getline(cases, line)) {
    if (line.empty()) continue;
    if (line[0] == 'G') {
      if (grammar) llama_sampler_free(grammar);
      grammar_text = read_file(line.substr(2));
      grammar = llama_sampler_init_grammar(vocab, grammar_text.c_str(), "root");
      if (!grammar) {
        std::fprintf(stderr, "grammar does not parse: %s\n", line.c_str());
        return 1;
      }
    } else if (line[0] == 'S') {
      std::istringstream is(line.substr(2));
      size_t cnt = 0;
      is >> cnt;
      std::vector<int32_t> seq(cnt);
      for (auto& t : seq) is >> t;
      // A fresh sampler state for each sequence.
      llama_sampler_reset(grammar);
      std::string end = "ok";
      for (size_t j = 0; j <= seq.size(); ++j) {
        auto [ok, no] = allowed();
        write_set(ok, no);
        if (j == seq.size()) break;
        const int32_t t = seq[j];
        const bool is_ok = t >= 0 && t < n && std::isinf(data[size_t(t)].logit) == false;
        if (!is_ok) {
          end = "rej " + std::to_string(j);
          break;
        }
        try {
          llama_sampler_accept(grammar, t);
        } catch (const std::exception& e) {
          end = "err " + std::to_string(j) + " " + e.what();
          break;
        }
        if (llama_vocab_is_eog(vocab, t)) {
          end = "eog " + std::to_string(j);
          break;
        }
      }
      out << "Z " << end << '\n';
    }
  }
  if (grammar) llama_sampler_free(grammar);
  llama_model_free(model);
  return 0;
}
