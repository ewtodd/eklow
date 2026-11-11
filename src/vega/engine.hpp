// vega's engine on the CPU (vegaml 0.16.1: _universal.py UniversalEngine, _common.py VegaEngine.forward with halt=True,
// and the thermal calibration of _api.py), from a vega-U model directory: vega_config.json, engine.safetensors,
// port.safetensors and stats.npz.
//
// One question at a time: the pooled vectors are standardized with the reader's extraction statistics and ported into
// the shared space (one linear map per feature layer, the last one's for options and memory chunks); the slots are
// mixed over layers; the world latent (with the memory reader's attention over the chunks when there are any), the
// imagined next state, the probe and the impulse place a particle and its momentum in a 64-d decision space; every
// option is a Gaussian well; the particle moves for n_steps under the wells, friction and a learned memory force (a
// settled particle stays where it is); the wells' energies at its final position are the logits, and the Boltzmann
// distribution at a temperature that depends on how settled it is gives the probabilities.
//
// Arithmetic is double throughout (vegaml runs the engine in fp32); the engine is ~14M parameters, about a millisecond
// a question.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vega/prompt.hpp"
#include "vega/reader.hpp"

namespace eklow::vega {

struct EngineConfig {
  int64_t latent_dim = 64, world_dim = 256, proj_hidden = 2048, n_steps = 12, ssm_state = 32, memory_chunks = 0, d_model = 0, d_shared = 0;
  double dt_max = 0.5, settle_eps = 0.01, force_max = 20.0;
  int64_t max_len = 73728, opt_max_tokens = 32, query_first_max_state = 0, prefix_min_tokens = 4096, memory_min_tokens = 2048,
          memory_chunk_tokens = 128;
  std::vector<int> feature_layers, option_layers;
  double theta[3][5] = {};  // thermal calibration: per question type, log tau = theta[0] + theta[1..4] . phys
  std::string backbone_name;
};

struct EngineAnswer {
  std::vector<double> probs;   // calibrated, in option order
  std::vector<double> logits;  // -(well energy) at the final position
  double phys[4] = {};         // log1p(kinetic), log1p(nearest), steps_used / n_steps, log(k)
  int steps = 0;
};

class Engine {
 public:
  explicit Engine(const std::string& dir);
  const EngineConfig& cfg() const { return cfg_; }
  // SHA-256 of engine.safetensors: what a policy names as the engine it was fitted on.
  const std::string& sha256() const { return sha_; }
  EngineAnswer answer(const Pooled& p, QType t) const;

 private:
  struct Lin {
    std::vector<float> w, b;  // [out][in], [out]
    int64_t in = 0, out = 0;
  };
  struct Norm {
    std::vector<float> w, b;
  };
  struct Mlp {
    Norm ln;
    Lin a, b;
  };
  Lin lin(const std::string& name, const std::map<std::string, struct Tensor>& t) const;
  static std::vector<double> apply(const Lin& l, const std::vector<double>& x);
  static std::vector<double> layernorm(const Norm& n, const std::vector<double>& x);
  static std::vector<double> mlp(const Mlp& m, const std::vector<double>& x);

  EngineConfig cfg_;
  std::string sha_;
  // the port and the reader's statistics
  std::vector<Lin> port_;
  std::vector<float> mu_, sd_, omu_, osd_;  // [Lf][3][d], [Lo][d]
  // the engine
  std::vector<float> mix_, opt_mix_, qtype_emb_, A_log_, pos_;
  double log_kappa_ = 0, force_scale_ = 0;
  Mlp world_, probe_, impulse_, to_z0_, to_p0_, well_, rail_dir_, mem_q_;
  Norm fore_norm_, mem_kv_norm_;
  Lin fore_inp_, fore_out_, mem_kv_, mem_out_, ssm_in_, ssm_out_;
};

}  // namespace eklow::vega
