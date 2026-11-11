// laya's trained policies on vega (format laya-policy/1, laya 3a38cbe: vega.py save_calibration / load_calibration,
// policy.py): heads.npz (a TTT head per question: vegaml's PCA + logistic over the reader's pooled vectors, ttt.py), and
// calibration.json (vegaml.calibrate's Calibration: a temperature per question, applied over whichever readout answered),
// under manifest.json (each file's SHA-256, the engine's).
//
// Loading verifies what laya verifies before it installs anything: every file the manifest names has that SHA-256, a
// heads.npz or calibration.json present is named by it, the policy was fitted on this engine (the manifest's
// engine_sha256, else the calibration's provenance), and the calibration's own checksum (vegaml's, over its alpha,
// questions and band) matches its contents. A policy without a manifest loads unverified, as in laya. A signed policy
// (Embedding Space's manifest.sig) is refused: eklow has no Ed25519 and does not pretend to check one.
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace eklow::vega {

struct Head {
  std::vector<std::string> labels;
  int64_t F = 0, k = 0;
  std::vector<float> mu, sd, c, P, s, W, b;  // [F], [F], [F], [F][k], [k], [k][labels], [labels]
  std::optional<double> cv, engine_accuracy;
  // Probabilities in label order (ttt.py TTTHead.predict over all its labels).
  std::vector<double> predict(const std::vector<float>& x) const;
};

struct CalQuestion {
  std::string type;
  std::vector<std::string> labels;
  double temperature = 1.0, qhat = 1.0;
};

class Policy {
 public:
  // engine_sha256: the loaded engine's (Engine::sha256), checked against the policy's.
  static Policy load(const std::string& dir, const std::string& engine_sha256);
  // vegaml's mode "auto": the head answers when its options are the question's and its cross-validated accuracy beats
  // both chance and the engine's own accuracy on the same rows (Vega._head_worth_using).
  const Head* head_for(const std::string& qid, const std::vector<std::string>& labels) const;
  // The question's temperature, when the calibration fitted one.
  const CalQuestion* calibration(const std::string& qid) const;
  const std::string& verified() const { return verified_; }
  const std::map<std::string, Head>& heads() const { return heads_; }

 private:
  std::map<std::string, Head> heads_;
  std::map<std::string, CalQuestion> cal_;
  std::string verified_;
};

// vegaml.calibrate's rescale: each probability clamped to [1e-6, 1 - 1e-6], log / temperature, softmax.
std::vector<double> rescale(const std::vector<double>& probs, double temperature);

// Python's round(x, 6).
double round6(double x);

}  // namespace eklow::vega
