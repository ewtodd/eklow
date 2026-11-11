// What eklow-sampler-test and llama-sampler-ref share, so both see the same logits, history and settings: the settings
// as one "key=value,..." string, a deterministic row of logits, and the output format (the candidates after the chain as
// "id probability" lines, then a histogram of draws).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace samplertest {

struct Settings {
  float temp = 0.8f, top_p = 0.95f, min_p = 0.05f, typ = 1.0f, sigma = -1.0f, xtc_p = 0.0f, xtc_t = 0.1f, dyn = 0.0f, dyn_exp = 1.0f;
  float repeat = 1.0f, freq = 0.0f, present = 0.0f, tau = 5.0f, eta = 0.1f;
  int top_k = 40, last_n = 64, mirostat = 0, min_keep = 0;
  std::string order = "edskypmxt";  // llama.cpp's sampler chars: the default chain
  std::vector<std::pair<int, float>> bias;
};

inline Settings parse(const std::string& s) {
  Settings o;
  std::stringstream ss(s);
  for (std::string kv; std::getline(ss, kv, ',');) {
    const auto eq = kv.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
    const float f = std::strtof(v.c_str(), nullptr);
    if (k == "temp") o.temp = f;
    else if (k == "top_k") o.top_k = int(f);
    else if (k == "top_p") o.top_p = f;
    else if (k == "min_p") o.min_p = f;
    else if (k == "typ") o.typ = f;
    else if (k == "sigma") o.sigma = f;
    else if (k == "xtc_p") o.xtc_p = f;
    else if (k == "xtc_t") o.xtc_t = f;
    else if (k == "dyn") o.dyn = f;
    else if (k == "dyn_exp") o.dyn_exp = f;
    else if (k == "repeat") o.repeat = f;
    else if (k == "freq") o.freq = f;
    else if (k == "present") o.present = f;
    else if (k == "last_n") o.last_n = int(f);
    else if (k == "mirostat") o.mirostat = int(f);
    else if (k == "tau") o.tau = f;
    else if (k == "eta") o.eta = f;
    else if (k == "min_keep") o.min_keep = int(f);
    else if (k == "order") o.order = v;
    else if (k == "bias") {  // id:bias;id:bias
      std::stringstream bs(v);
      for (std::string b; std::getline(bs, b, ';');)
        if (const auto c = b.find(':'); c != std::string::npos) o.bias.push_back({std::atoi(b.substr(0, c).c_str()), std::strtof(b.substr(c + 1).c_str(), nullptr)});
    }
  }
  return o;
}

inline uint64_t splitmix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// A row of logits shaped like a model's: most near zero, a long tail upward, a few tokens far ahead.
inline std::vector<float> logits(int V, uint64_t seed) {
  std::vector<float> l(static_cast<size_t>(V));
  for (int i = 0; i < V; ++i) {
    const double u = double(splitmix(seed * 1000003ull + uint64_t(i)) >> 11) * 0x1.0p-53;
    l[size_t(i)] = float(14.0 * u * u * u * u * u * u - 2.0 + 0.5 * u);
  }
  return l;
}

// The tokens already seen, for the penalties: drawn among the likeliest so that they matter.
inline std::vector<int> history(const std::vector<float>& l, uint64_t seed, int n) {
  std::vector<int> top;
  for (int i = 0; i < int(l.size()); ++i)
    if (l[size_t(i)] > 6.0f) top.push_back(i);
  std::vector<int> h;
  for (int i = 0; i < n && !top.empty(); ++i) h.push_back(top[splitmix(seed + 77 + uint64_t(i)) % top.size()]);
  return h;
}

}  // namespace samplertest
