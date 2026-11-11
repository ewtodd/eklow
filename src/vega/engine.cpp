#include "vega/engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/json_value.hpp"
#include "vega/files.hpp"

namespace eklow::vega {

namespace {

double num(const json::Value& v) { return v.is_int() ? std::stod(v.int_digits()) : v.as_double(); }

const json::Value& at(const json::Value& o, const char* k, const std::string& where) {
  const json::Value* v = o.find(k);
  if (!v) throw std::runtime_error(where + ": no " + k);
  return *v;
}

double gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

std::vector<double> softmax(const std::vector<double>& z) {
  double m = -INFINITY;
  for (double v : z) m = std::max(m, v);
  std::vector<double> e(z.size());
  double s = 0;
  for (size_t i = 0; i < z.size(); ++i) s += e[i] = std::exp(z[i] - m);
  for (double& v : e) v /= s;
  return e;
}

}  // namespace

Engine::Lin Engine::lin(const std::string& name, const std::map<std::string, Tensor>& t) const {
  auto w = t.find(name + ".weight"), b = t.find(name + ".bias");
  if (w == t.end() || b == t.end() || w->second.shape.size() != 2) throw std::runtime_error("vega engine: no linear layer " + name);
  Lin l;
  l.out = w->second.shape[0], l.in = w->second.shape[1];
  l.w = w->second.data, l.b = b->second.data;
  return l;
}

std::vector<double> Engine::apply(const Lin& l, const std::vector<double>& x) {
  if (int64_t(x.size()) != l.in) throw std::runtime_error("vega engine: a linear layer's input width differs");
  std::vector<double> y(size_t(l.out));
  for (int64_t o = 0; o < l.out; ++o) {
    const float* w = l.w.data() + o * l.in;
    double acc = l.b[size_t(o)];
    for (int64_t i = 0; i < l.in; ++i) acc += double(w[i]) * x[size_t(i)];
    y[size_t(o)] = acc;
  }
  return y;
}

// nn.LayerNorm, eps 1e-5.
std::vector<double> Engine::layernorm(const Norm& n, const std::vector<double>& x) {
  double mean = 0, var = 0;
  for (double v : x) mean += v;
  mean /= double(x.size());
  for (double v : x) var += (v - mean) * (v - mean);
  var /= double(x.size());
  const double inv = 1.0 / std::sqrt(var + 1e-5);
  std::vector<double> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = (x[i] - mean) * inv * n.w[i] + n.b[i];
  return y;
}

// vegaml's MLP: LayerNorm, Linear, GELU (erf), Dropout (off at inference), Linear.
std::vector<double> Engine::mlp(const Mlp& m, const std::vector<double>& x) {
  std::vector<double> h = apply(m.a, layernorm(m.ln, x));
  for (double& v : h) v = gelu(v);
  return apply(m.b, h);
}

Engine::Engine(const std::string& dir) {
  const std::string where = dir + "/vega_config.json";
  const json::Value c = json::parse(read_file(where));
  auto gi = [&](const char* k, int64_t def) { const json::Value* v = c.find(k); return v && !v->is_null() ? int64_t(num(*v)) : def; };
  auto gd = [&](const char* k, double def) { const json::Value* v = c.find(k); return v && !v->is_null() ? num(*v) : def; };
  cfg_.latent_dim = gi("latent_dim", 64), cfg_.world_dim = gi("world_dim", 256), cfg_.proj_hidden = gi("proj_hidden", 2048);
  cfg_.n_steps = gi("n_steps", 12), cfg_.ssm_state = gi("ssm_state", 32), cfg_.memory_chunks = gi("memory_chunks", 0);
  cfg_.d_model = gi("d_model", 0), cfg_.d_shared = gi("d_shared", 0);
  cfg_.dt_max = gd("dt_max", 0.5), cfg_.settle_eps = gd("settle_eps", 0.01), cfg_.force_max = gd("force_max", 20.0);
  cfg_.max_len = gi("max_len", 73728), cfg_.opt_max_tokens = gi("opt_max_tokens", 32);
  cfg_.query_first_max_state = gi("query_first_max_state", 0), cfg_.prefix_min_tokens = gi("prefix_min_tokens", 4096);
  cfg_.memory_min_tokens = gi("memory_min_tokens", 2048), cfg_.memory_chunk_tokens = gi("memory_chunk_tokens", 128);
  for (const auto& v : at(c, "feature_layers", where).items()) cfg_.feature_layers.push_back(int(num(v)));
  for (const auto& v : at(c, "option_layers", where).items()) cfg_.option_layers.push_back(int(num(v)));
  if (const json::Value* b = c.find("backbone_name")) cfg_.backbone_name = b->as_string();
  if (!c.find("port") || c.find("port")->is_null()) throw std::runtime_error(where + ": not a universal (vega-U) checkpoint: no port");
  if (c.find("template") && num(*c.find("template")) != 0) throw std::runtime_error(where + ": only template 0 is ported");
  // The thermal calibration. Without one vegaml answers raw softmax probabilities; every shipped checkpoint has it.
  const json::Value& cal = at(c, "calibration", where);
  if (at(cal, "mode", where).as_string() != "thermal") throw std::runtime_error(where + ": calibration mode is not thermal");
  const auto& th = at(cal, "theta", where).items();
  if (th.size() != 3) throw std::runtime_error(where + ": theta is not 3 x 5");
  for (int r = 0; r < 3; ++r) {
    if (th[size_t(r)].items().size() != 5) throw std::runtime_error(where + ": theta is not 3 x 5");
    for (int k = 0; k < 5; ++k) cfg_.theta[r][k] = num(th[size_t(r)].items()[size_t(k)]);
  }

  const std::string engine_bytes = read_file(dir + "/engine.safetensors");
  sha_ = sha256_hex(engine_bytes);
  const auto e = read_safetensors(dir + "/engine.safetensors");
  const auto p = read_safetensors(dir + "/" + at(c, "port", where).as_string());
  const auto st = parse_npz(read_file(dir + "/" + (c.find("raw_stats") ? c.find("raw_stats")->as_string() : std::string("stats.npz"))), "stats.npz");
  auto vec = [&](const std::map<std::string, Tensor>& t, const std::string& name) {
    auto it = t.find(name);
    if (it == t.end()) throw std::runtime_error("vega engine: no tensor " + name);
    return it->second.data;
  };
  auto norm = [&](const std::string& name) { return Norm{vec(e, name + ".weight"), vec(e, name + ".bias")}; };
  auto mlp_of = [&](const std::string& name) { return Mlp{norm(name + ".net.0"), lin(name + ".net.1", e), lin(name + ".net.4", e)}; };
  const int64_t nF = int64_t(cfg_.feature_layers.size());
  for (int64_t i = 0; i < nF; ++i) port_.push_back(lin("lin." + std::to_string(i), p));
  auto stat = [&](const char* k) {
    auto it = st.find(k);
    if (it == st.end()) throw std::runtime_error("stats.npz: no " + std::string(k));
    return it->second.floats();
  };
  mu_ = stat("mu"), sd_ = stat("sd"), omu_ = stat("omu"), osd_ = stat("osd");
  const int64_t d = cfg_.d_model;
  if (int64_t(mu_.size()) != nF * 3 * d || int64_t(omu_.size()) != int64_t(cfg_.option_layers.size()) * d)
    throw std::runtime_error("stats.npz: shapes differ from the configuration");
  mix_ = vec(e, "mix"), opt_mix_ = vec(e, "opt_mix"), qtype_emb_ = vec(e, "qtype_emb.weight"), A_log_ = vec(e, "A_log");
  log_kappa_ = vec(e, "log_kappa").at(0), force_scale_ = vec(e, "force_scale").at(0);
  world_ = mlp_of("world"), probe_ = mlp_of("probe"), impulse_ = mlp_of("impulse"), to_z0_ = mlp_of("to_z0"), to_p0_ = mlp_of("to_p0");
  well_ = mlp_of("well"), rail_dir_ = mlp_of("rail_dir");
  fore_norm_ = norm("foresight.norm"), fore_inp_ = lin("foresight.inp", e), fore_out_ = lin("foresight.out", e);
  ssm_in_ = lin("ssm_in", e), ssm_out_ = lin("ssm_out", e);
  if (cfg_.memory_chunks) {
    mem_q_ = mlp_of("memory.q");
    mem_kv_norm_ = norm("memory.kv.0"), mem_kv_ = lin("memory.kv.1", e), mem_out_ = lin("memory.out", e);
    pos_ = vec(e, "memory.pos");
  }
  for (const auto& [name, t] : e)
    if (name.rfind("gate_nets", 0) == 0) throw std::runtime_error("vega engine: adapters are not ported");
}

EngineAnswer Engine::answer(const Pooled& pl, QType qt) const {
  const int64_t d = cfg_.d_model, nF = int64_t(cfg_.feature_layers.size()), nO = int64_t(cfg_.option_layers.size());
  const int64_t D = cfg_.latent_dim, S = cfg_.world_dim, K = int64_t(pl.opts.size()) / (nO * d), t = int64_t(qt);
  if (int64_t(pl.item.size()) != nF * 3 * d || K < 2) throw std::runtime_error("vega engine: pooled vectors of the wrong shape");
  // The port: standardized with the extraction statistics, one map per feature layer; options and chunks through the last.
  std::vector<std::vector<double>> f(size_t(nF * 3));
  for (int64_t l = 0; l < nF; ++l)
    for (int64_t s = 0; s < 3; ++s) {
      std::vector<double> x(static_cast<size_t>(d));
      const int64_t o = (l * 3 + s) * d;
      for (int64_t i = 0; i < d; ++i) x[size_t(i)] = (double(pl.item[size_t(o + i)]) - mu_[size_t(o + i)]) / sd_[size_t(o + i)];
      f[size_t(l * 3 + s)] = apply(port_[size_t(l)], x);
    }
  // The slots, mixed over feature layers; the options over option layers.
  std::vector<std::vector<double>> slots(3, std::vector<double>(size_t(cfg_.d_shared), 0.0));
  for (int64_t s = 0; s < 3; ++s) {
    std::vector<double> m(static_cast<size_t>(nF));
    for (int64_t l = 0; l < nF; ++l) m[size_t(l)] = mix_[size_t(s * nF + l)];
    const auto w = softmax(m);
    for (int64_t l = 0; l < nF; ++l)
      for (size_t i = 0; i < slots[size_t(s)].size(); ++i) slots[size_t(s)][i] += w[size_t(l)] * f[size_t(l * 3 + s)][i];
  }
  std::vector<double> om(static_cast<size_t>(nO));
  for (int64_t l = 0; l < nO; ++l) om[size_t(l)] = opt_mix_[size_t(l)];
  const auto ow = softmax(om);
  std::vector<std::vector<double>> opts(static_cast<size_t>(K), std::vector<double>(size_t(cfg_.d_shared), 0.0));
  for (int64_t k = 0; k < K; ++k)
    for (int64_t l = 0; l < nO; ++l) {
      std::vector<double> x(static_cast<size_t>(d));
      const int64_t o = (k * nO + l) * d;
      for (int64_t i = 0; i < d; ++i) x[size_t(i)] = (double(pl.opts[size_t(o + i)]) - omu_[size_t(l * d + i)]) / osd_[size_t(l * d + i)];
      const auto y = apply(port_.back(), x);
      for (size_t i = 0; i < y.size(); ++i) opts[size_t(k)][i] += ow[size_t(l)] * y[i];
    }

  std::vector<double> s = mlp(world_, slots[0]);
  // The memory reader: the question reading and the final token attend over the chunks (four heads).
  if (cfg_.memory_chunks && pl.n_chunks > 0) {
    const int64_t C = cfg_.memory_chunks, P = int64_t(pos_.size()) / (2 * S), nh = 4, dh = S / nh, n = pl.n_chunks;
    std::vector<double> qx(slots[1]);
    qx.insert(qx.end(), slots[2].begin(), slots[2].end());
    const auto q = mlp(mem_q_, qx);
    std::vector<std::vector<double>> kv;
    for (int64_t c = 0; c < std::min<int64_t>(n, C); ++c) {
      std::vector<double> x(static_cast<size_t>(d));
      const int64_t o = (nF - 1) * 3 * d;  // the deepest feature layer's state statistics
      for (int64_t i = 0; i < d; ++i) x[size_t(i)] = (double(pl.chunks[size_t(c * d + i)]) - mu_[size_t(o + i)]) / sd_[size_t(o + i)];
      auto y = apply(mem_kv_, layernorm(mem_kv_norm_, apply(port_.back(), x)));
      // Where in the record the chunk sits, computed in fp32 as vegaml does.
      const int64_t at = std::clamp<int64_t>(int64_t((float(c) + 0.5f) / float(n) * float(P)), 0, P - 1);
      for (int64_t i = 0; i < 2 * S; ++i) y[size_t(i)] += pos_[size_t(at * 2 * S + i)];
      kv.push_back(std::move(y));
    }
    std::vector<double> r(static_cast<size_t>(S));
    for (int64_t h = 0; h < nh; ++h) {
      std::vector<double> sc;
      for (const auto& y : kv) {
        double a = 0;
        for (int64_t i = 0; i < dh; ++i) a += q[size_t(h * dh + i)] * y[size_t(h * dh + i)];
        sc.push_back(a / std::sqrt(double(dh)));
      }
      const auto att = softmax(sc);
      for (size_t c = 0; c < kv.size(); ++c)
        for (int64_t i = 0; i < dh; ++i) r[size_t(h * dh + i)] += att[c] * kv[c][size_t(S + h * dh + i)];
    }
    const auto m = apply(mem_out_, r);
    for (int64_t i = 0; i < S; ++i) s[size_t(i)] += m[size_t(i)];
  }
  // One step of world dynamics: the imagined next state.
  std::vector<double> s_next(s);
  {
    const auto u = apply(fore_inp_, layernorm(fore_norm_, s));
    const int64_t half = int64_t(u.size()) / 2;
    std::vector<double> g(static_cast<size_t>(half));
    for (int64_t i = 0; i < half; ++i) g[size_t(i)] = gelu(u[size_t(i)]) * sigmoid(u[size_t(half + i)]);
    const auto o = apply(fore_out_, g);
    for (int64_t i = 0; i < S; ++i) s_next[size_t(i)] += o[size_t(i)];
  }
  std::vector<double> probe = mlp(probe_, slots[1]);
  for (int64_t i = 0; i < D; ++i) probe[size_t(i)] += qtype_emb_[size_t(t * D + i)];
  const std::vector<double> imp = mlp(impulse_, slots[2]);
  std::vector<double> ctx(s);
  ctx.insert(ctx.end(), s_next.begin(), s_next.end());
  ctx.insert(ctx.end(), probe.begin(), probe.end());
  ctx.insert(ctx.end(), imp.begin(), imp.end());
  std::vector<double> z = mlp(to_z0_, ctx), p = mlp(to_p0_, ctx);

  // The landscape: a well per option (centre, log depth, width); score levels on a rail.
  std::vector<std::vector<double>> cen(static_cast<size_t>(K));
  std::vector<double> log_a(static_cast<size_t>(K)), sigma(static_cast<size_t>(K)), gap(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) {
    std::vector<double> x(opts[size_t(k)]);
    x.insert(x.end(), probe.begin(), probe.end());
    const auto e = mlp(well_, x);
    cen[size_t(k)].assign(e.begin(), e.begin() + D);
    for (int64_t i = 0; i < D; ++i) cen[size_t(k)][size_t(i)] += qtype_emb_[size_t(t * D + i)];
    log_a[size_t(k)] = std::clamp(e[size_t(D)], -4.0, 4.0);
    sigma[size_t(k)] = 0.3 + softplus(e[size_t(D + 1)]);
    gap[size_t(k)] = softplus(e[size_t(D + 2)]);
  }
  if (qt == QType::Score) {
    std::vector<double> r0(size_t(D), 0.0);
    for (int64_t k = 0; k < K; ++k)
      for (int64_t i = 0; i < D; ++i) r0[size_t(i)] += cen[size_t(k)][size_t(i)] / double(K);
    std::vector<double> u = mlp(rail_dir_, probe);
    double un = 0;
    for (double v : u) un += v * v;
    un = std::max(std::sqrt(un), 1e-12);
    for (double& v : u) v /= un;
    std::vector<double> tt(static_cast<size_t>(K));
    double run = 0, mean = 0;
    for (int64_t k = 0; k < K; ++k) mean += tt[size_t(k)] = run += gap[size_t(k)];
    mean /= double(K);
    for (int64_t k = 0; k < K; ++k)
      for (int64_t i = 0; i < D; ++i) cen[size_t(k)][size_t(i)] = r0[size_t(i)] + (tt[size_t(k)] - mean) * u[size_t(i)];
  }

  // The integrator: damped symplectic Euler with a selective-SSM thermostat; a settled particle stays where it is.
  const double kappa = std::exp(log_kappa_);
  std::vector<double> grad(static_cast<size_t>(D)), q(static_cast<size_t>(K));
  auto potential = [&] {
    std::fill(grad.begin(), grad.end(), 0.0);
    for (int64_t i = 0; i < D; ++i) grad[size_t(i)] = kappa * z[size_t(i)];
    for (int64_t k = 0; k < K; ++k) {
      double d2 = 0;
      for (int64_t i = 0; i < D; ++i) d2 += (z[size_t(i)] - cen[size_t(k)][size_t(i)]) * (z[size_t(i)] - cen[size_t(k)][size_t(i)]);
      const double s2 = sigma[size_t(k)] * sigma[size_t(k)];
      q[size_t(k)] = d2 / (2 * s2);
      const double g = std::exp(log_a[size_t(k)] - q[size_t(k)]);
      for (int64_t i = 0; i < D; ++i) grad[size_t(i)] += g * (z[size_t(i)] - cen[size_t(k)][size_t(i)]) / s2;
    }
    double gn = 0;
    for (double v : grad) gn += v * v;
    return std::sqrt(gn);
  };
  double gn = potential();
  const int64_t T = cfg_.n_steps, NM = int64_t(A_log_.size());
  std::vector<double> h(static_cast<size_t>(NM), 0.0);
  bool settled = false;
  int steps = int(T);
  for (int64_t step = 0; step < T; ++step) {
    std::vector<double> x(z);
    x.insert(x.end(), p.begin(), p.end());
    x.insert(x.end(), grad.begin(), grad.end());
    x.push_back(gn);
    const auto u = apply(ssm_in_, x);
    const double dt = cfg_.dt_max * sigmoid(u[size_t(NM)]), gamma = softplus(u[size_t(NM + 1)]);
    for (int64_t i = 0; i < NM; ++i) h[size_t(i)] = std::exp(-std::exp(double(A_log_[size_t(i)])) * dt) * h[size_t(i)] + dt * u[size_t(i)];
    const auto fo = apply(ssm_out_, h);
    if (!settled) {
      for (int64_t i = 0; i < D; ++i) {
        const double g_eff = grad[size_t(i)] / (1 + gn / cfg_.force_max), force = force_scale_ * std::tanh(fo[size_t(i)]);
        p[size_t(i)] = p[size_t(i)] * std::exp(-gamma * dt) - dt * g_eff + dt * force;
      }
      for (int64_t i = 0; i < D; ++i) z[size_t(i)] += dt * p[size_t(i)];
    }
    gn = potential();
    double kin = 0;
    for (double v : p) kin += 0.5 * v * v;
    if (!settled && kin < cfg_.settle_eps && gn < 10 * cfg_.settle_eps) settled = true, steps = int(step + 1);
  }
  EngineAnswer a;
  double kin = 0, nearest = INFINITY;
  for (double v : p) kin += 0.5 * v * v;
  for (int64_t k = 0; k < K; ++k) {
    a.logits.push_back(-(q[size_t(k)] - log_a[size_t(k)]));
    nearest = std::min(nearest, q[size_t(k)]);
  }
  a.steps = steps;
  a.phys[0] = std::log1p(kin), a.phys[1] = std::log1p(nearest), a.phys[2] = double(steps) / double(std::max<int64_t>(1, T));
  a.phys[3] = std::log(double(std::max<int64_t>(K, 2)));
  double lt = cfg_.theta[t][0];
  for (int i = 0; i < 4; ++i) lt += cfg_.theta[t][i + 1] * a.phys[i];
  const double tau = std::exp(std::clamp(lt, std::log(0.2), std::log(20.0)));
  std::vector<double> zt;
  for (double l : a.logits) zt.push_back(l / tau);
  a.probs = softmax(zt);
  return a;
}

}  // namespace eklow::vega
