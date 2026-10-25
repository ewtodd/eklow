// eklow-oracle: llama.cpp's CPU backend on the same GGUF, as the reference eklow is checked against.
//
//   eklow-oracle --model GGUF --out DIR (--prompt TEXT | --prompt-file FILE | --tokens 2,818,...)
//                [--tensors PREFIX,...] [--list] [--threads N] [--ngl N] [--logits-last N]
//                [--mmproj GGUF --image FILE ... [--clip-tensors PREFIX,...] [--clip-list]]
//
// Runs one forward pass over the whole prompt (one ubatch, fp32 KV cache, no flash attention) and
// writes to DIR:
//   tokens.json     the token ids and their pieces (llama.cpp's tokenization, with special tokens
//                   parsed and BOS added as the model asks)
//   index.json      one entry per dumped tensor: name, ggml shape (ne[0] fastest), file
//   NAME.f32        the tensor as float32, in ggml order
//   logits.f32      n_vocab logits for every position (or the last --logits-last), row by row
// --tensors takes name prefixes matched against llama.cpp's graph tensor names (default: l_out-,
// result_norm, result_output); --list prints every graph tensor name and shape once. --ngl offloads
// layers to a GPU backend when the linked llama.cpp has one (default 0: CPU only).
//
// With --mmproj, the prompt is a rendered chat whose images are llama.cpp's media marker
// (<__media__>), one per --image in order, and it runs as llama-server does: mtmd tokenizes it
// (the text as written, no BOS added: the prompt carries its own <bos>), each image is encoded on
// the CPU and decoded as embeddings with non-causal attention where mtmd asks for it, every chunk
// in a single ubatch. Image positions are id -1 in tokens.json, which also lists each image's
// first position, token count and patch grid; image_K.f32 holds image K's embeddings
// [tokens][n_embd]. Graph tensors of the text model are not dumped in this mode; the vision
// encoder's are, by --clip-tensors prefixes (default none), into clip.NAME.f32 and clip_index.json.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {

struct Dump {
  std::string dir;
  std::string tag;  // prefix of the dump files' names
  std::vector<std::string> prefixes;
  bool list = false;
  std::set<std::string> listed;
  std::ostringstream index;
  bool first = true;
  std::vector<uint8_t> buf;
};

bool wanted(const Dump& d, const char* name) {
  for (const auto& p : d.prefixes)
    if (std::strncmp(name, p.c_str(), p.size()) == 0) return true;
  return false;
}

std::string json_str(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') o += '\\', o += char(c);
    else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else o += char(c);
  }
  return o + "\"";
}

// The scheduler asks about every graph node (ask = true) and, for the ones we accept, calls again
// once the node is computed.
bool eval_cb(ggml_tensor* t, bool ask, void* user) {
  auto& d = *static_cast<Dump*>(user);
  const char* name = ggml_get_name(t);
  if (ask) {
    if (d.list && d.listed.insert(name).second)
      std::printf("  %-40s %-6s [%lld, %lld, %lld, %lld]\n", name, ggml_type_name(t->type), (long long)t->ne[0],
                  (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
    return wanted(d, name);
  }
  if (!ggml_is_contiguous(t) || (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)) {
    std::fprintf(stderr, "oracle: skipping %s (%s, %s)\n", name, ggml_type_name(t->type),
                 ggml_is_contiguous(t) ? "contiguous" : "not contiguous");
    return true;
  }
  const size_t n = ggml_nelements(t);
  d.buf.resize(ggml_nbytes(t));
  ggml_backend_tensor_get(t, d.buf.data(), 0, d.buf.size());
  std::vector<float> f(n);
  if (t->type == GGML_TYPE_F32) std::memcpy(f.data(), d.buf.data(), n * sizeof(float));
  else ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t*>(d.buf.data()), f.data(), int64_t(n));
  const std::string file = d.tag + name + ".f32";
  std::ofstream(d.dir + "/" + file, std::ios::binary).write(reinterpret_cast<const char*>(f.data()), n * sizeof(float));
  d.index << (d.first ? "\n" : ",\n") << "  {\"name\": " << json_str(name) << ", \"type\": \"" << ggml_type_name(t->type)
          << "\", \"ne\": [" << t->ne[0] << ", " << t->ne[1] << ", " << t->ne[2] << ", " << t->ne[3]
          << "], \"file\": " << json_str(file) << "}";
  d.first = false;
  return true;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string p; std::getline(ss, p, sep);)
    if (!p.empty()) out.push_back(p);
  return out;
}

std::string piece(const llama_vocab* vocab, llama_token t) {
  char buf[256];
  int len = llama_token_to_piece(vocab, t, buf, sizeof buf, 0, true);
  return std::string(buf, len > 0 ? len : 0);
}

int usage() {
  std::fprintf(stderr,
               "usage: eklow-oracle --model GGUF --out DIR (--prompt TEXT | --prompt-file FILE | --tokens IDS)\n"
               "                    [--tensors PREFIX,...] [--list] [--threads N] [--ngl N] [--logits-last N]\n"
               "                    [--mmproj GGUF --image FILE ... [--clip-tensors PREFIX,...] [--clip-list]]\n");
  return 2;
}

llama_context* make_context(llama_model* model, int n_tok, int threads, Dump* d) {
  auto cp = llama_context_default_params();
  cp.n_ctx = uint32_t((n_tok + 255) / 256 * 256);
  cp.n_batch = cp.n_ubatch = uint32_t(n_tok);
  cp.type_k = cp.type_v = GGML_TYPE_F32;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  if (threads > 0) cp.n_threads = cp.n_threads_batch = threads;
  if (d) {
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = d;
  }
  return llama_init_from_model(model, cp);
}

// The prompt through mtmd, chunk by chunk, as llama-server runs it. Returns the exit status.
int run_images(llama_model* model, const std::string& mmproj, const std::vector<std::string>& images,
               const std::string& prompt, int threads, int logits_last, Dump& clip, const std::string& dir) {
  const llama_vocab* vocab = llama_model_get_vocab(model);
  auto mp = mtmd_context_params_default();
  mp.use_gpu = false;
  mp.print_timings = false;
  mp.warmup = false;
  if (threads > 0) mp.n_threads = threads;
  mp.cb_eval = eval_cb;
  mp.cb_eval_user_data = &clip;
  mtmd_context* mctx = mtmd_init_from_file(mmproj.c_str(), model, mp);
  if (!mctx) {
    std::fprintf(stderr, "oracle: cannot load %s\n", mmproj.c_str());
    return 1;
  }
  std::vector<mtmd_bitmap*> bitmaps;
  for (const auto& path : images) {
    auto w = mtmd_helper_bitmap_init_from_file(mctx, path.c_str(), false, mtmd_helper_init_opt_default());
    if (!w.bitmap) {
      std::fprintf(stderr, "oracle: cannot read the image %s\n", path.c_str());
      return 1;
    }
    bitmaps.push_back(w.bitmap);
  }
  mtmd_input_text text{prompt.c_str(), prompt.size(), false, true};
  mtmd_input_chunks* chunks = mtmd_input_chunks_init();
  if (int32_t r = mtmd_tokenize(mctx, chunks, &text, const_cast<const mtmd_bitmap**>(bitmaps.data()), bitmaps.size()); r != 0) {
    std::fprintf(stderr, "oracle: mtmd_tokenize failed (%d)\n", r);
    return 1;
  }
  const int n_tok = int(mtmd_helper_get_n_tokens(chunks));
  const int first_logit = logits_last > 0 && logits_last < n_tok ? n_tok - logits_last : 0;
  llama_context* ctx = make_context(model, n_tok, threads, nullptr);
  if (!ctx) {
    std::fprintf(stderr, "oracle: cannot create a context\n");
    return 1;
  }
  const int n_embd = llama_model_n_embd_inp(model), n_vocab = llama_vocab_n_tokens(vocab);
  std::ofstream lf(dir + "/logits.f32", std::ios::binary);
  std::ostringstream ids, pieces, imgs;
  int pos = 0, n_images = 0;
  auto add_logits = [&](int n) {
    for (int j = 0; j < n; ++j)
      if (pos + j >= first_logit)
        lf.write(reinterpret_cast<const char*>(llama_get_logits_ith(ctx, j)), size_t(n_vocab) * sizeof(float));
  };
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
    const mtmd_input_chunk* c = mtmd_input_chunks_get(chunks, i);
    const auto type = mtmd_input_chunk_get_type(c);
    if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n = 0;
      const llama_token* t = mtmd_input_chunk_get_tokens_text(c, &n);
      llama_batch b = llama_batch_init(int32_t(n), 0, 1);
      for (size_t j = 0; j < n; ++j) {
        b.token[j] = t[j];
        b.pos[j] = pos + int(j);
        b.n_seq_id[j] = 1;
        b.seq_id[j][0] = 0;
        b.logits[j] = true;
        ids << (pos + j ? ", " : "") << t[j];
        pieces << (pos + j ? ", " : "") << json_str(piece(vocab, t[j]));
      }
      b.n_tokens = int32_t(n);
      if (llama_decode(ctx, b) != 0) {
        std::fprintf(stderr, "oracle: llama_decode failed on a text chunk\n");
        return 1;
      }
      add_logits(int(n));
      llama_batch_free(b);
      pos += int(n);
    } else if (type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      if (mtmd_encode_chunk(mctx, c) != 0) {
        std::fprintf(stderr, "oracle: mtmd_encode_chunk failed\n");
        return 1;
      }
      const int n = int(mtmd_input_chunk_get_n_tokens(c));
      const float* embd = mtmd_get_output_embd(mctx);
      std::ofstream(dir + "/image_" + std::to_string(n_images) + ".f32", std::ios::binary)
          .write(reinterpret_cast<const char*>(embd), size_t(n) * n_embd * sizeof(float));
      const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(c);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      imgs << (n_images ? ", " : "") << "{\"pos\": " << pos << ", \"n\": " << n << ", \"nx\": " << mtmd_image_tokens_get_nx(it)
           << ", \"ny\": " << mtmd_image_tokens_get_ny(it) << "}";
#pragma GCC diagnostic pop
      llama_batch b = llama_batch_init(n, n_embd, 1);
      std::memcpy(b.embd, embd, size_t(n) * n_embd * sizeof(float));
      for (int j = 0; j < n; ++j) {
        b.pos[j] = pos + j;
        b.n_seq_id[j] = 1;
        b.seq_id[j][0] = 0;
        b.logits[j] = true;
        ids << (pos + j ? ", " : "") << -1;
        pieces << (pos + j ? ", " : "") << "\"<image>\"";
      }
      b.n_tokens = n;
      llama_set_causal_attn(ctx, !mtmd_decode_use_non_causal(mctx, c));
      const int r = llama_decode(ctx, b);
      llama_set_causal_attn(ctx, true);
      if (r != 0) {
        std::fprintf(stderr, "oracle: llama_decode failed on an image\n");
        return 1;
      }
      add_logits(n);
      llama_batch_free(b);
      pos += n;
      ++n_images;
    } else {
      std::fprintf(stderr, "oracle: only text and images are supported\n");
      return 1;
    }
  }
  std::ofstream(dir + "/tokens.json") << "{\"ids\": [" << ids.str() << "], \"pieces\": [" << pieces.str() << "], \"images\": ["
                                      << imgs.str() << "]}\n";
  std::ofstream(dir + "/index.json") << "{\"n_tokens\": " << n_tok << ", \"n_vocab\": " << n_vocab
                                     << ", \"logits_rows\": " << n_tok - first_logit << ", \"tensors\": []}\n";
  std::ofstream(dir + "/clip_index.json") << "{\"tensors\": [" << clip.index.str() << "\n]}\n";
  std::printf("oracle: %d positions (%d images), %d vocab, wrote %s\n", n_tok, n_images, n_vocab, dir.c_str());
  mtmd_input_chunks_free(chunks);
  for (auto* b : bitmaps) mtmd_bitmap_free(b);
  llama_free(ctx);
  mtmd_free(mctx);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, prompt, tokens_arg, mmproj;
  std::vector<std::string> images;
  bool have_prompt = false;
  int threads = 0, ngl = 0, logits_last = 0;
  Dump d, clip;
  d.prefixes = {"l_out-", "result_norm", "result_output"};
  clip.tag = "clip.";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model_path = next();
    else if (a == "--out") d.dir = next();
    else if (a == "--prompt") prompt = next(), have_prompt = true;
    else if (a == "--prompt-file") {
      std::ifstream f(next(), std::ios::binary);
      prompt.assign(std::istreambuf_iterator<char>(f), {});
      have_prompt = true;
    } else if (a == "--tokens") tokens_arg = next();
    else if (a == "--tensors") d.prefixes = split(next(), ',');
    else if (a == "--list") d.list = true;
    else if (a == "--threads") threads = std::atoi(next().c_str());
    else if (a == "--ngl") ngl = std::atoi(next().c_str());
    else if (a == "--logits-last") logits_last = std::atoi(next().c_str());
    else if (a == "--mmproj") mmproj = next();
    else if (a == "--image") images.push_back(next());
    else if (a == "--clip-tensors") clip.prefixes = split(next(), ',');
    else if (a == "--clip-list") clip.list = true;
    else return usage();
  }
  if (model_path.empty() || d.dir.empty() || (!have_prompt && tokens_arg.empty())) return usage();
  if (!mmproj.empty() && (!have_prompt || images.empty())) return usage();
  clip.dir = d.dir;

  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = ngl;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!model) {
    std::fprintf(stderr, "oracle: cannot load %s\n", model_path.c_str());
    return 1;
  }
  if (!mmproj.empty()) {
    const int r = run_images(model, mmproj, images, prompt, threads, logits_last, clip, d.dir);
    llama_model_free(model);
    llama_backend_free();
    return r;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);

  std::vector<llama_token> toks;
  if (!tokens_arg.empty()) {
    for (auto& s : split(tokens_arg, ',')) toks.push_back(llama_token(std::atoi(s.c_str())));
  } else {
    int n = -llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), nullptr, 0, true, true);
    toks.resize(n);
    if (llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()), toks.data(), n, true, true) != n) {
      std::fprintf(stderr, "oracle: tokenization failed\n");
      return 1;
    }
  }
  const int n_tok = int(toks.size());

  {
    std::ofstream tj(d.dir + "/tokens.json");
    tj << "{\"ids\": [";
    for (int i = 0; i < n_tok; ++i) tj << (i ? ", " : "") << toks[i];
    tj << "], \"pieces\": [";
    for (int i = 0; i < n_tok; ++i) tj << (i ? ", " : "") << json_str(piece(vocab, toks[i]));
    tj << "]}\n";
    if (!tj) {
      std::fprintf(stderr, "oracle: cannot write to %s\n", d.dir.c_str());
      return 1;
    }
  }

  llama_context* ctx = make_context(model, n_tok, threads, &d);
  if (!ctx) {
    std::fprintf(stderr, "oracle: cannot create a context\n");
    return 1;
  }

  llama_batch batch = llama_batch_init(n_tok, 0, 1);
  for (int i = 0; i < n_tok; ++i) {
    batch.token[i] = toks[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = logits_last <= 0 || i >= n_tok - logits_last;
  }
  batch.n_tokens = n_tok;
  if (d.list) std::printf("graph tensors:\n");
  if (llama_decode(ctx, batch) != 0) {
    std::fprintf(stderr, "oracle: llama_decode failed\n");
    return 1;
  }

  const int n_vocab = llama_vocab_n_tokens(vocab);
  const int first = logits_last > 0 && logits_last < n_tok ? n_tok - logits_last : 0;
  std::ofstream lf(d.dir + "/logits.f32", std::ios::binary);
  for (int i = first; i < n_tok; ++i)
    lf.write(reinterpret_cast<const char*>(llama_get_logits_ith(ctx, i)), size_t(n_vocab) * sizeof(float));
  std::ofstream(d.dir + "/index.json") << "{\"n_tokens\": " << n_tok << ", \"n_vocab\": " << n_vocab
                                       << ", \"logits_rows\": " << n_tok - first << ", \"tensors\": [" << d.index.str()
                                       << "\n]}\n";
  std::printf("oracle: %d tokens, %d vocab, wrote %s\n", n_tok, n_vocab, d.dir.c_str());

  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
