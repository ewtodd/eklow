#include "vega/policy.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "core/json_value.hpp"
#include "vega/files.hpp"

namespace eklow::vega {

namespace {

double num(const json::Value& v) { return v.is_int() ? std::stod(v.int_digits()) : v.as_double(); }

}  // namespace

std::vector<double> Head::predict(const std::vector<float>& x) const {
  if (int64_t(x.size()) != F) throw std::runtime_error("a policy head reads " + std::to_string(F) + " features, got " + std::to_string(x.size()));
  std::vector<double> zc(static_cast<size_t>(F));
  for (int64_t i = 0; i < F; ++i) zc[size_t(i)] = (double(x[size_t(i)]) - mu[size_t(i)]) / sd[size_t(i)] - c[size_t(i)];
  std::vector<double> zp(static_cast<size_t>(k), 0.0);
  for (int64_t i = 0; i < F; ++i) {
    const float* pr = P.data() + i * k;
    const double v = zc[size_t(i)];
    for (int64_t j = 0; j < k; ++j) zp[size_t(j)] += v * pr[j];
  }
  for (int64_t j = 0; j < k; ++j) zp[size_t(j)] /= s[size_t(j)];
  const size_t L = labels.size();
  std::vector<double> z(L);
  double m = -INFINITY;
  for (size_t l = 0; l < L; ++l) {
    double acc = b[l];
    for (int64_t j = 0; j < k; ++j) acc += zp[size_t(j)] * W[size_t(j) * L + l];
    m = std::max(m, z[l] = acc);
  }
  double sum = 0;
  for (double& v : z) sum += v = std::exp(v - m);
  for (double& v : z) v /= sum;
  return z;
}

Policy Policy::load(const std::string& dir, const std::string& engine_sha256) {
  namespace fs = std::filesystem;
  if (!fs::is_directory(dir)) throw std::runtime_error(dir + " is not a policy folder");
  std::map<std::string, std::string> files;
  for (const char* n : {"heads.npz", "calibration.json", "questions.json", "manifest.json", "manifest.sig"})
    if (fs::is_regular_file(fs::path(dir) / n)) files[n] = read_file((fs::path(dir) / n).string());
  if (!files.count("heads.npz")) throw std::runtime_error(dir + " has no heads.npz; it is not a trained policy");
  Policy p;
  std::string expected;
  if (files.count("manifest.json")) {
    const json::Value m = json::parse(files["manifest.json"]);
    if (!m.is_object()) throw std::runtime_error(dir + ": manifest.json is not a JSON object");
    if (const json::Value* fl = m.find("files"); fl && !fl->is_null()) {
      if (!fl->is_object()) throw std::runtime_error(dir + ": manifest.json's files is not a map of file names to SHA-256 hashes");
      for (const auto& [name, want] : fl->members()) {
        auto it = files.find(name);
        if (it == files.end()) throw std::runtime_error(dir + ": the manifest lists " + name + " but the policy does not contain it");
        const std::string got = sha256_hex(it->second);
        if (!want.is_string() || got != want.as_string())
          throw std::runtime_error(dir + ": " + name + " has been changed: its SHA-256 is " + got.substr(0, 16) + ", the manifest says " +
                                   (want.is_string() ? want.as_string().substr(0, 16) : std::string("?")));
      }
      for (const char* n : {"heads.npz", "calibration.json"})
        if (files.count(n) && !fl->find(n)) throw std::runtime_error(dir + ": " + n + " is in the policy but not in its manifest");
    } else {
      for (const char* n : {"heads.npz", "calibration.json"})
        if (files.count(n)) throw std::runtime_error(dir + ": " + n + " is in the policy but not in its manifest");
    }
    if (const json::Value* kid = m.find("key_id"); kid && kid->truthy())
      throw std::runtime_error(dir + ": a signed policy (key " + (kid->is_string() ? kid->as_string() : std::string("?")) +
                               "); eklow does not verify signatures, so it refuses one rather than pretend");
    if (const json::Value* e = m.find("engine_sha256"); e && e->is_string()) expected = e->as_string();
    p.verified_ = "integrity";
  } else {
    p.verified_ = "unverified (no manifest)";
  }
  json::Value cal;
  if (files.count("calibration.json")) {
    cal = json::parse(files["calibration.json"]);
    if (expected.empty())
      if (const json::Value* pv = cal.find("provenance"); pv && pv->find("engine_sha256") && pv->find("engine_sha256")->is_string())
        expected = pv->find("engine_sha256")->as_string();
  }
  if (!expected.empty() && !engine_sha256.empty() && expected != engine_sha256)
    throw std::runtime_error(dir + ": this policy was trained on engine " + expected.substr(0, 12) + "... but the loaded engine is " +
                             engine_sha256.substr(0, 12) + "...");
  if (!expected.empty()) p.verified_ += ", engine matches";

  const auto z = parse_npz(files["heads.npz"], dir + "/heads.npz");
  auto mt = z.find("meta");
  if (mt == z.end()) throw std::runtime_error(dir + "/heads.npz: no meta");
  const json::Value meta = json::parse(mt->second.unicode());
  for (const auto& [qid, m] : meta.find("heads")->members()) {
    Head h;
    for (const auto& l : m.find("labels")->items()) h.labels.push_back(l.as_string());
    h.k = int64_t(num(*m.find("k")));
    if (const json::Value* cv = m.find("cv"); cv && !cv->is_null()) h.cv = num(*cv);
    if (const json::Value* ea = m.find("engine_accuracy"); ea && !ea->is_null()) h.engine_accuracy = num(*ea);
    auto arr = [&](const char* name) {
      auto it = z.find(qid + "::" + name);
      if (it == z.end()) throw std::runtime_error(dir + "/heads.npz: no " + qid + "::" + name);
      return it->second;
    };
    const Array P = arr("P");
    if (P.shape.size() != 2) throw std::runtime_error(dir + "/heads.npz: " + qid + "::P is not a matrix");
    h.F = P.shape[0];
    const int64_t kk = P.shape[1];
    h.P = P.floats(), h.mu = arr("mu").floats(), h.sd = arr("sd").floats(), h.c = arr("c").floats(), h.s = arr("s").floats();
    h.W = arr("W").floats(), h.b = arr("b").floats();
    h.k = kk;
    if (int64_t(h.mu.size()) != h.F || int64_t(h.s.size()) != kk || h.W.size() != size_t(kk) * h.labels.size() || h.b.size() != h.labels.size())
      throw std::runtime_error(dir + "/heads.npz: " + qid + "'s arrays do not fit together");
    p.heads_.emplace(qid, std::move(h));
  }
  if (cal.is_object()) {
    // vegaml's Calibration.from_payload refuses a calibration whose contents no longer hash to its stamped checksum.
    json::Value payload = json::Value::object();
    payload.set("alpha", *cal.find("alpha"));
    payload.set("questions", *cal.find("questions"));
    const json::Value* band = cal.find("band");
    if (band && !band->is_null()) {
      json::Value b = json::Value::object();
      const json::Value* bq = band->find("questions");
      b.set("questions", bq && !bq->is_null() ? *bq : json::Value::object());
      const json::Value* bd = band->find("default");
      b.set("default", bd && !bd->is_null() ? json::Value::number(num(*bd)) : json::Value::null());
      payload.set("band", b);
    } else {
      payload.set("band", json::Value::null());
    }
    std::string blob;
    blob = python_json(payload, true, true, true);
    if (const json::Value* stamped = cal.find("checksum"); stamped && stamped->is_string() && !stamped->as_string().empty()) {
      const std::string got = sha256_hex(blob);
      if (got != stamped->as_string())
        throw std::runtime_error(dir + "/calibration.json was changed after it was written: it is stamped " + stamped->as_string().substr(0, 12) +
                                 " but its contents hash to " + got.substr(0, 12));
      p.verified_ += ", calibration checksum matches";
    }
    for (const auto& [qid, q] : cal.find("questions")->members()) {
      CalQuestion c;
      c.type = q.find("type")->as_string();
      for (const auto& l : q.find("labels")->items()) c.labels.push_back(l.as_string());
      c.temperature = num(*q.find("temperature"));
      c.qhat = num(*q.find("qhat"));
      p.cal_.emplace(qid, std::move(c));
    }
  }
  return p;
}

const Head* Policy::head_for(const std::string& qid, const std::vector<std::string>& labels) const {
  auto it = heads_.find(qid);
  if (it == heads_.end()) return nullptr;
  const Head& h = it->second;
  if (!h.cv || labels != h.labels) return nullptr;
  double bar = 1.0 / double(std::max<size_t>(labels.size(), 2));
  if (h.engine_accuracy && *h.engine_accuracy > bar) bar = *h.engine_accuracy;
  return *h.cv > bar ? &h : nullptr;
}

const CalQuestion* Policy::calibration(const std::string& qid) const {
  auto it = cal_.find(qid);
  return it == cal_.end() ? nullptr : &it->second;
}

std::vector<double> rescale(const std::vector<double>& probs, double temperature) {
  std::vector<double> z;
  double m = -INFINITY;
  for (double p : probs) m = std::max(m, z.emplace_back(std::log(std::min(std::max(p, 1e-6), 1.0 - 1e-6)) / temperature));
  double s = 0;
  for (double& v : z) s += v = std::exp(v - m);
  for (double& v : z) v /= s;
  return z;
}

double round6(double x) {
  char b[64];
  std::snprintf(b, sizeof b, "%.6f", x);
  return std::strtod(b, nullptr);
}

}  // namespace eklow::vega
