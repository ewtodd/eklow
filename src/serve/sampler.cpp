#include "serve/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace eklow::serve {

namespace {

const std::map<std::string, Stage>& stage_names() {
  // common_sampler_types_from_names: the canonical names, their kebab-case and no-underscore forms, and three aliases.
  static const std::map<std::string, Stage> m = [] {
    const std::map<std::string, Stage> canon = {{"dry", Stage::Dry},       {"top_k", Stage::TopK},     {"top_p", Stage::TopP},
                                                {"top_n_sigma", Stage::TopNSigma}, {"typ_p", Stage::TypicalP}, {"min_p", Stage::MinP},
                                                {"temperature", Stage::Temperature}, {"xtc", Stage::Xtc},     {"infill", Stage::Infill},
                                                {"penalties", Stage::Penalties}, {"adaptive_p", Stage::AdaptiveP}};
    std::map<std::string, Stage> all = canon;
    for (const auto& [name, st] : canon) {
      if (name.find('_') == std::string::npos) continue;
      std::string kebab = name, bare;
      std::replace(kebab.begin(), kebab.end(), '_', '-');
      for (char c : name)
        if (c != '_') bare += c;
      all.emplace(kebab, st);
      all.emplace(bare, st);
    }
    all.emplace("nucleus", Stage::TopP);
    all.emplace("temp", Stage::Temperature);
    all.emplace("typ", Stage::TypicalP);
    return all;
  }();
  return m;
}

// Unknown names are skipped, as llama.cpp warns and skips them.
std::vector<Stage> stages_from_names(const std::vector<std::string>& names) {
  std::vector<Stage> out;
  for (std::string n : names) {
    std::transform(n.begin(), n.end(), n.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (auto it = stage_names().find(n); it != stage_names().end()) out.push_back(it->second);
  }
  return out;
}

std::vector<Stage> stages_from_chars(const std::string& s) {
  static const std::map<char, Stage> m = {{'d', Stage::Dry},  {'k', Stage::TopK},        {'y', Stage::TypicalP}, {'p', Stage::TopP},
                                          {'s', Stage::TopNSigma}, {'m', Stage::MinP}, {'t', Stage::Temperature}, {'x', Stage::Xtc},
                                          {'i', Stage::Infill}, {'e', Stage::Penalties}, {'a', Stage::AdaptiveP}};
  std::vector<Stage> out;
  for (char c : s)
    if (auto it = m.find(c); it != m.end()) out.push_back(it->second);
  return out;
}

bool penalties_off(const SamplingParams& p) {
  return p.penalty_last_n == 0 || (p.repeat_penalty == 1.0f && p.frequency_penalty == 0.0f && p.presence_penalty == 0.0f);
}

bool dry_off(const SamplingParams& p) { return p.dry_multiplier == 0.0f || p.dry_base < 1.0f || p.dry_penalty_last_n == 0; }

// What eklow refuses rather than samples differently from llama.cpp.
void check_supported(const SamplingParams& p) {
  if (!std::isfinite(p.repeat_penalty) || p.repeat_penalty <= 0.0f || !std::isfinite(1.0f / p.repeat_penalty))
    throw std::invalid_argument("penalty_repeat must be finite and greater than 0");
  if (!std::isfinite(p.frequency_penalty)) throw std::invalid_argument("penalty_freq must be finite");
  if (!std::isfinite(p.presence_penalty)) throw std::invalid_argument("penalty_present must be finite");
  if (p.mirostat < 0 || p.mirostat > 2) throw std::invalid_argument("mirostat must be 0, 1 or 2");
  if (p.mirostat != 0) return;  // the chain is not used
  for (Stage s : p.samplers) {
    if (s == Stage::Dry && !dry_off(p)) throw std::invalid_argument("DRY sampling (dry_multiplier > 0) is not implemented in eklow");
    if (s == Stage::AdaptiveP) throw std::invalid_argument("the adaptive_p sampler is not implemented in eklow");
    if (s == Stage::Infill) throw std::invalid_argument("the infill sampler is not implemented in eklow");
  }
}

}  // namespace

void apply_gguf_sampling(const gguf::File& f, SamplingParams& p) {
  auto i32 = [&](const char* key, int32_t& dst) {
    if (const gguf::Value* v = f.find(key)) dst = int32_t(v->as_int());
  };
  auto f32 = [&](const char* key, float& dst) {
    if (const gguf::Value* v = f.find(key)) dst = float(v->as_float());
  };
  if (const gguf::Value* v = f.find("general.sampling.sequence")) {
    std::vector<std::string> names;
    std::string cur;
    for (char c : v->as_string() + ";") {
      if (c == ';') names.push_back(cur), cur.clear();
      else cur += c;
    }
    if (std::vector<Stage> s = stages_from_names(names); !s.empty()) p.samplers = s;
  }
  i32("general.sampling.top_k", p.top_k);
  f32("general.sampling.top_p", p.top_p);
  f32("general.sampling.min_p", p.min_p);
  f32("general.sampling.xtc_probability", p.xtc_probability);
  f32("general.sampling.xtc_threshold", p.xtc_threshold);
  f32("general.sampling.temp", p.temperature);
  i32("general.sampling.penalty_last_n", p.penalty_last_n);
  f32("general.sampling.penalty_repeat", p.repeat_penalty);
  i32("general.sampling.mirostat", p.mirostat);
  f32("general.sampling.mirostat_tau", p.mirostat_tau);
  f32("general.sampling.mirostat_eta", p.mirostat_eta);
}

void apply_request_sampling(const json::Value& body, SamplingParams& p,
                            const std::function<std::vector<int32_t>(const std::string&)>& tokenize) {
  auto num = [&](const char* key, double lo, double hi, auto& dst) {
    const json::Value* v = body.find(key);
    if (!v || v->is_null()) return;
    double d;
    if (v->is_int()) d = std::stod(v->int_digits());
    else if (v->is_float()) d = v->as_double();
    else throw std::invalid_argument(std::string(key) + " must be a number");
    if (!(d >= lo && d <= hi)) throw std::invalid_argument(std::string(key) + " is out of range");
    dst = static_cast<std::remove_reference_t<decltype(dst)>>(d);
  };
  constexpr double big = std::numeric_limits<float>::max(), ibig = std::numeric_limits<int32_t>::max();
  num("top_k", -ibig, ibig, p.top_k);
  num("top_p", 0, 1, p.top_p);
  num("min_p", 0, 1, p.min_p);
  num("top_n_sigma", -big, big, p.top_n_sigma);
  num("xtc_probability", 0, 1, p.xtc_probability);
  num("xtc_threshold", 0, 1, p.xtc_threshold);
  num("typical_p", -big, big, p.typical_p);
  num("temperature", 0, std::numeric_limits<double>::infinity(), p.temperature);
  num("dynatemp_range", -big, big, p.dynatemp_range);
  num("dynatemp_exponent", -big, big, p.dynatemp_exponent);
  num("repeat_last_n", 0, ibig, p.penalty_last_n);
  num("repeat_penalty", -big, big, p.repeat_penalty);
  num("frequency_penalty", -big, big, p.frequency_penalty);
  num("presence_penalty", -big, big, p.presence_penalty);
  num("dry_multiplier", -big, big, p.dry_multiplier);
  {
    float base = p.dry_base;
    num("dry_base", -big, big, base);
    if (base >= 1.0f) p.dry_base = base;  // llama-server: below 1 keeps the default
  }
  num("dry_allowed_length", 0, ibig, p.dry_allowed_length);
  num("dry_penalty_last_n", 0, ibig, p.dry_penalty_last_n);
  num("mirostat", 0, 2, p.mirostat);
  num("mirostat_tau", -big, big, p.mirostat_tau);
  num("mirostat_eta", -big, big, p.mirostat_eta);
  num("adaptive_target", -big, 1, p.adaptive_target);
  num("min_keep", 0, ibig, p.min_keep);
  if (const json::Value* v = body.find("samplers"); v && !v->is_null()) {
    if (v->is_array()) {
      std::vector<std::string> names;
      for (const auto& s : v->items())
        if (s.is_string()) names.push_back(s.as_string());
      p.samplers = stages_from_names(names);
    } else if (v->is_string()) {
      p.samplers = stages_from_chars(v->as_string());
    }
  }
  if (const json::Value* v = body.find("logit_bias"); v && !v->is_null()) {
    auto bias_of = [](const json::Value& b, float& out) {
      if (b.is_int()) return out = float(std::stod(b.int_digits())), true;
      if (b.is_float()) return out = float(b.as_double()), true;
      if (b.is_bool() && !b.as_bool()) return out = -std::numeric_limits<float>::infinity(), true;
      return false;
    };
    auto add = [&](const json::Value& key, const json::Value& b) {
      float bias;
      if (!bias_of(b, bias)) return;
      if (key.is_int()) p.logit_bias.push_back({int32_t(std::stoll(key.int_digits())), bias});
      else if (key.is_string())
        for (int32_t t : tokenize(key.as_string())) p.logit_bias.push_back({t, bias});
    };
    if (v->is_array()) {
      for (const auto& el : v->items())
        if (el.is_array() && el.items().size() == 2) add(el.items()[0], el.items()[1]);
    } else if (v->is_object()) {
      for (const auto& [k, b] : v->members()) {
        // An object's keys are token ids when they parse as one, else text (llama-server does the same).
        char* end = nullptr;
        const long long id = std::strtoll(k.c_str(), &end, 10);
        if (!k.empty() && end && *end == '\0') add(json::Value::integer(int64_t(id)), b);
        else add(json::Value::string(k), b);
      }
    }
  }
  if (const json::Value* v = body.find("seed"); v && !v->is_null()) {
    if (!v->is_int()) throw std::invalid_argument("seed must be an integer");
    const long long s = std::stoll(v->int_digits());
    if (s >= 0) p.seed = uint64_t(s);  // -1 (llama's LLAMA_DEFAULT_SEED): the caller's random seed stays
  }
  check_supported(p);
}

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.seed) {
  check_supported(p_);
  greedy_ = p_.mirostat == 0 && !p_.samplers.empty() &&
            ((p_.samplers.back() == Stage::Temperature && p_.temperature == 0.0f && p_.dynatemp_range == 0.0f) ||
             (p_.samplers.back() == Stage::TopK && p_.top_k == 1));
  mu_ = 2.0f * p_.mirostat_tau;
}

void Sampler::accept(int32_t id) {
  if (id < 0) return;
  history_.push_back(id);
  const size_t keep = p_.penalty_last_n > 0 ? size_t(p_.penalty_last_n) : 0;
  if (p_.penalty_last_n >= 0 && history_.size() > keep) history_.erase(history_.begin(), history_.end() - std::ptrdiff_t(keep));
}

namespace {

using Cand = Sampler::Cand;

bool by_logit(const Cand& a, const Cand& b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); }

// llama_sampler_softmax_impl: probabilities in float from the largest logit, sorting first if asked.
void softmax(std::vector<Cand>& c, bool& sorted, bool do_sort) {
  if (do_sort && !sorted) std::sort(c.begin(), c.end(), by_logit), sorted = true;
  float mx = c[0].logit;
  if (!sorted)
    for (const Cand& x : c) mx = std::max(mx, x.logit);
  float sum = 0;
  for (Cand& x : c) sum += x.p = std::exp(x.logit - mx);
  for (Cand& x : c) x.p /= sum;
}

// llama_sampler_temp_impl
void temp_impl(std::vector<Cand>& c, float t) {
  if (c.empty()) return;
  if (t <= 0.0f) {
    size_t mi = 0;
    float ml = c[0].logit;
    for (size_t i = 1; i < c.size(); ++i) {
      if (c[i].logit > ml) {
        c[mi].logit = -INFINITY;
        mi = i, ml = c[i].logit;
      } else {
        c[i].logit = -INFINITY;
      }
    }
    return;
  }
  for (Cand& x : c) x.logit /= t;
}

}  // namespace

void Sampler::run(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned,
                  std::span<const int32_t> extra, int xtc) {
  const int32_t V = int32_t(logits.size());
  // The penalties' window: the last penalty_last_n of the accepted tokens and `extra`.
  std::unordered_map<int32_t, int> counts;
  if (!penalties_off(p_)) {
    const size_t n = history_.size() + extra.size(), keep = p_.penalty_last_n > 0 ? size_t(p_.penalty_last_n) : n;
    for (size_t i = n > keep ? n - keep : 0; i < n; ++i) ++counts[i < history_.size() ? history_[i] : extra[i - history_.size()]];
  }
  std::unordered_map<int32_t, float> bias;
  for (const auto& [id, b] : p_.logit_bias)
    if (id >= 0 && id < V) bias[id] += b;
  auto biased = [&](int32_t id) {
    float l = logits[size_t(id)];
    if (!bias.empty())
      if (auto it = bias.find(id); it != bias.end()) l += it->second;
    return l;
  };
  auto penalized = [&](float l, int32_t id) {
    const auto it = counts.find(id);
    if (it == counts.end()) return l;
    l = l <= 0 ? l * p_.repeat_penalty : l / p_.repeat_penalty;
    return l - float(it->second) * p_.frequency_penalty - float(it->second > 0) * p_.presence_penalty;
  };
  const bool bans = banned && !banned->empty();
  auto is_banned = [&](int32_t id) { return bans && std::binary_search(banned->begin(), banned->end(), id); };

  // Which stages run here: under mirostat only the temperature and its own pick (common_sampler_init).
  const std::vector<Stage> chain = p_.mirostat != 0 ? std::vector<Stage>{Stage::Temperature} : p_.samplers;
  size_t from = 0;
  cur_.clear();
  sorted_ = false;
  // top_k over the whole vocabulary keeps the k largest, so when only stages that are no-ops (or the penalties, applied here)
  // come before it, one pass with a heap picks them: the same set as sorting 262k candidates, at a fraction of the cost.
  size_t k_at = chain.size();
  for (size_t i = 0; i < chain.size(); ++i) {
    const Stage s = chain[i];
    if (s == Stage::TopK) {
      k_at = i;
      break;
    }
    if (!(s == Stage::Penalties || (s == Stage::Dry && dry_off(p_)) || (s == Stage::TopNSigma && p_.top_n_sigma <= 0))) break;
  }
  if (allowed) {
    if (allowed->empty()) throw std::runtime_error("sampler: no token allowed");
    for (int32_t id : *allowed)
      if (!is_banned(id)) cur_.push_back({id, biased(id), 0});
    if (cur_.empty())
      for (int32_t id : *allowed) cur_.push_back({id, biased(id), 0});
  } else if (k_at < chain.size() && p_.top_k > 0) {
    const size_t keep = size_t(p_.top_k);
    float worst = -INFINITY;
    const bool plain = counts.empty() && bias.empty();
    for (int32_t id = 0; id < V; ++id) {
      if (plain && cur_.size() >= keep && !(logits[size_t(id)] > worst)) continue;
      if (is_banned(id)) continue;
      const Cand c{id, penalized(biased(id), id), 0};
      if (cur_.size() < keep) {
        cur_.push_back(c);
        std::push_heap(cur_.begin(), cur_.end(), by_logit);
      } else if (by_logit(c, cur_.front())) {
        std::pop_heap(cur_.begin(), cur_.end(), by_logit);
        cur_.back() = c;
        std::push_heap(cur_.begin(), cur_.end(), by_logit);
      }
      if (cur_.size() >= keep) worst = cur_.front().logit;
    }
    std::sort(cur_.begin(), cur_.end(), by_logit);
    sorted_ = true;
    from = k_at + 1;
  } else {
    cur_.reserve(size_t(V));
    for (int32_t id = 0; id < V; ++id)
      if (!is_banned(id)) cur_.push_back({id, biased(id), 0});
  }
  if (cur_.empty()) throw std::runtime_error("sampler: every token is banned");

  const size_t min_keep = size_t(std::max(p_.min_keep, 0));
  for (size_t si = from; si < chain.size(); ++si) {
    switch (chain[si]) {
      case Stage::Penalties:
        if (penalties_off(p_)) break;
        for (Cand& c : cur_) c.logit = penalized(c.logit, c.id);
        sorted_ = false;
        break;
      case Stage::Dry:  // off (check_supported refuses it on)
        break;
      case Stage::TopNSigma: {
        if (p_.top_n_sigma <= 0.0f || cur_.size() <= 1) break;
        float mx = cur_[0].logit, sum = 0;
        size_t valid = 0;
        for (const Cand& c : cur_)
          if (c.logit != -INFINITY) mx = std::max(mx, c.logit), sum += c.logit, ++valid;
        const float mean = valid ? sum / float(valid) : 0;
        float acc = 0;
        for (const Cand& c : cur_)
          if (c.logit != -INFINITY) acc += float(std::pow(c.logit - mean, 2));
        const float sd = valid ? std::sqrt(acc / float(valid)) : 0;
        for (Cand& c : cur_)
          if (c.logit < mx - p_.top_n_sigma * sd) c.logit = -INFINITY;
        break;
      }
      case Stage::TopK: {
        if (p_.top_k <= 0) break;
        const size_t k = std::min(size_t(p_.top_k), cur_.size());
        if (!sorted_) std::partial_sort(cur_.begin(), cur_.begin() + std::ptrdiff_t(k), cur_.end(), by_logit), sorted_ = true;
        cur_.resize(k);
        break;
      }
      case Stage::TypicalP: {
        if (p_.typical_p >= 1.0f) break;
        softmax(cur_, sorted_, true);
        float entropy = 0;
        for (const Cand& c : cur_) entropy += -c.p * std::log(c.p);
        std::vector<float> shifted(cur_.size());
        for (size_t i = 0; i < cur_.size(); ++i) shifted[i] = std::fabs(-std::log(cur_[i].p) - entropy);
        std::vector<size_t> idx(cur_.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return shifted[a] < shifted[b]; });
        float cum = 0;
        size_t last = idx.size();
        for (size_t i = 0; i < idx.size(); ++i) {
          cum += cur_[idx[i]].p;
          if (cum > p_.typical_p && (min_keep == 0 || i >= min_keep - 1)) {
            last = i + 1;
            break;
          }
        }
        std::vector<Cand> next;
        for (size_t i = 0; i < last; ++i) next.push_back(cur_[idx[i]]);
        cur_.swap(next);
        sorted_ = false;
        break;
      }
      case Stage::TopP: {
        if (p_.top_p >= 1.0f) break;
        softmax(cur_, sorted_, false);
        if (!sorted_) std::sort(cur_.begin(), cur_.end(), by_logit), sorted_ = true;
        float cum = 0;
        size_t last = cur_.size();
        for (size_t i = 0; i < cur_.size(); ++i) {
          cum += cur_[i].p;
          if (cum >= p_.top_p && i + 1 >= min_keep) {
            last = i + 1;
            break;
          }
        }
        cur_.resize(last);
        break;
      }
      case Stage::MinP: {
        if (p_.min_p <= 0.0f || cur_.empty()) break;
        bool applied = false;
        if (!sorted_) {
          float mx = -std::numeric_limits<float>::max();
          for (const Cand& c : cur_) mx = std::max(mx, c.logit);
          const float lo = mx + std::log(p_.min_p);
          std::vector<Cand> kept;
          for (const Cand& c : cur_)
            if (c.logit >= lo) kept.push_back(c);
          if (!kept.empty() && kept.size() >= min_keep) cur_.swap(kept), applied = true;
        }
        if (!applied) {
          if (!sorted_) std::sort(cur_.begin(), cur_.end(), by_logit), sorted_ = true;
          const float lo = cur_[0].logit + std::log(p_.min_p);
          size_t i = 1;
          for (; i < cur_.size(); ++i)
            if (cur_[i].logit < lo && i >= min_keep) break;
          cur_.resize(i);
        }
        break;
      }
      case Stage::Xtc: {
        if (p_.xtc_probability <= 0.0f || p_.xtc_threshold > 0.5f || cur_.size() < 2) break;
        if (xtc == 0) break;
        if (xtc == 2 && std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_) > p_.xtc_probability) break;
        softmax(cur_, sorted_, true);
        size_t last = 0;
        for (size_t i = 0; i < cur_.size(); ++i) {
          if (cur_[i].p >= p_.xtc_threshold) last = i;
          else break;
        }
        if (cur_.size() - last >= min_keep && last > 0) cur_.erase(cur_.begin(), cur_.begin() + std::ptrdiff_t(last));
        break;
      }
      case Stage::Temperature: {
        const float t = p_.temperature;
        if (p_.mirostat == 0 && p_.dynatemp_range > 0) {
          if (cur_.size() <= 1) break;
          const float lo = std::max(0.0f, t - p_.dynatemp_range), hi = t + p_.dynatemp_range;
          const float max_entropy = -std::log(1.0f / float(cur_.size()));
          softmax(cur_, sorted_, true);
          float entropy = 0;
          for (const Cand& c : cur_)
            if (c.p > 0.0f) entropy -= c.p * std::log(c.p);
          const float dyn = lo + (hi - lo) * std::pow(entropy / max_entropy, p_.dynatemp_exponent);
          temp_impl(cur_, dyn);
        } else {
          temp_impl(cur_, t);
        }
        break;
      }
      case Stage::Infill:
      case Stage::AdaptiveP:
        break;  // refused by check_supported
    }
  }
}

void Sampler::final_probs(std::vector<std::pair<int32_t, double>>& out) const {
  // llama_sampler_dist_apply: exp in float from the largest logit, the sum in double.
  out.clear();
  if (greedy_) {
    size_t best = 0;
    for (size_t i = 1; i < cur_.size(); ++i)
      if (cur_[i].logit > cur_[best].logit) best = i;
    out.push_back({cur_[best].id, 1.0});
    return;
  }
  float mx = cur_[0].logit;
  for (const Cand& c : cur_) mx = std::max(mx, c.logit);
  double sum = 0;
  std::vector<float> e(cur_.size());
  for (size_t i = 0; i < cur_.size(); ++i) sum += e[i] = std::exp(cur_[i].logit - mx);
  for (size_t i = 0; i < cur_.size(); ++i)
    if (e[i] > 0) out.push_back({cur_[i].id, double(e[i]) / sum});
}

int32_t Sampler::sample(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned) {
  run(logits, allowed, banned, {}, 2);
  if (p_.mirostat != 0) {
    // llama_sampler_mirostat(_v2)_apply over the tempered candidates; mu moves with the surprise of the pick.
    bool sorted = sorted_;
    softmax(cur_, sorted, true);
    if (p_.mirostat == 1) {
      const int m = 100;
      float sum_ti_bi = 0, sum_ti_sq = 0;
      for (size_t i = 0; i < size_t(m - 1) && i + 1 < cur_.size(); ++i) {
        const float t_i = std::log(float(i + 2) / float(i + 1)), b_i = std::log(cur_[i].p / cur_[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
      }
      const float s_hat = sum_ti_bi / sum_ti_sq, eps = s_hat - 1;
      const float k = std::pow((eps * std::pow(2.0f, mu_)) / (1 - std::pow(float(logits.size()), -eps)), 1 / s_hat);
      cur_.resize(std::min(cur_.size(), size_t(std::max(int(k), 1))));
    } else {
      size_t n = 0;
      while (n < cur_.size() && !(-std::log2(cur_[n].p) > mu_)) ++n;
      cur_.resize(std::max<size_t>(n, 1));
    }
    softmax(cur_, sorted, true);
    std::vector<double> w(cur_.size());
    for (size_t i = 0; i < cur_.size(); ++i) w[i] = cur_[i].p;
    const size_t idx = std::discrete_distribution<size_t>(w.begin(), w.end())(rng_);
    mu_ -= p_.mirostat_eta * (-std::log2(cur_[idx].p) - p_.mirostat_tau);
    return cur_[idx].id;
  }
  if (greedy_) {
    size_t best = 0;
    for (size_t i = 1; i < cur_.size(); ++i)
      if (cur_[i].logit > cur_[best].logit) best = i;
    return cur_[best].id;
  }
  float mx = cur_[0].logit;
  for (const Cand& c : cur_) mx = std::max(mx, c.logit);
  double sum = 0;
  for (Cand& c : cur_) sum += c.p = std::exp(c.logit - mx);
  const double target = sum * std::uniform_real_distribution<double>(0, 1)(rng_);
  double run_sum = 0;
  for (const Cand& c : cur_) {
    run_sum += c.p;
    if (run_sum >= target) return c.id;
  }
  return cur_.back().id;
}

void Sampler::distribution(std::span<const float> logits, const std::vector<int32_t>* allowed, const std::vector<int32_t>* banned,
                           std::span<const int32_t> extra, std::vector<std::pair<int32_t, double>>& out) {
  if (p_.mirostat != 0) throw std::logic_error("sampler: mirostat has no fixed distribution (speculative() is false)");
  const bool xtc = !greedy_ && p_.xtc_probability > 0.0f && p_.xtc_threshold <= 0.5f &&
                   std::find(p_.samplers.begin(), p_.samplers.end(), Stage::Xtc) != p_.samplers.end();
  run(logits, allowed, banned, extra, 0);
  final_probs(out);
  if (!xtc) return;
  // XTC removes the likeliest tokens with probability xtc_probability: the distribution is the mixture of both chains.
  std::vector<std::pair<int32_t, double>> with;
  run(logits, allowed, banned, extra, 1);
  final_probs(with);
  std::unordered_map<int32_t, double> mix;
  for (const auto& [id, pr] : out) mix[id] += (1.0 - double(p_.xtc_probability)) * pr;
  for (const auto& [id, pr] : with) mix[id] += double(p_.xtc_probability) * pr;
  out.assign(mix.begin(), mix.end());
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.second > b.second || (a.second == b.second && a.first < b.first); });
}

}  // namespace eklow::serve
