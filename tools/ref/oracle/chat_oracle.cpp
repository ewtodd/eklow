// eklow-chat-oracle: llama-server's own chat path (common/chat and the server's request parsing, linked from the same
// llama.cpp build), as the reference for eklow's ported chat template, tokenizer and output parser.
//
//   eklow-chat-oracle --model GGUF --cases CASES.json --out RESULTS.json
//                     [--reasoning-format NAME] [--no-prefill-assistant] [--template-kwargs JSON]
//   eklow-chat-oracle --model GGUF --tokenize STRINGS.json --out TOKENS.json
//
// --tokenize takes an array of strings and writes, for each, llama.cpp's tokens with special tokens as plain text
// ("plain") and parsed ("special"), and the pieces of the parsed ones.
// CASES.json is an array of objects:
//   {"id": "...", "request": {an OpenAI chat request: messages, tools, tool_choice, ...},
//    "outputs": ["raw model output, special tokens as text", ...]}
// The request goes through oaicompat_chat_params_parse as llama-server receives it (--jinja, the GGUF's template, the
// server's defaults unless the options above change them). For each case RESULTS.json holds:
//   prompt             the rendered prompt (what the server tokenizes), and prompt_tokens (tokenized as the server does:
//                      special tokens parsed, BOS as the vocabulary asks)
//   generation_prompt, grammar, grammar_lazy, format, preserved_tokens, additional_stops, parse_error (if any)
//   parsed             for each output: the output's tokens, the message after finish() (to_json_oaicompat), and the
//                      streamed deltas, one entry per token as the server computes them (common_chat_msg_diff)
// Tokens are fed one at a time with their pieces, as the server feeds generated tokens.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "server-common.h"

namespace {

std::string piece(const llama_vocab* vocab, llama_token t) {
  std::vector<char> buf(256);
  int n = llama_token_to_piece(vocab, t, buf.data(), int32_t(buf.size()), 0, true);
  if (n < 0) {
    buf.resize(size_t(-n));
    n = llama_token_to_piece(vocab, t, buf.data(), int32_t(buf.size()), 0, true);
  }
  return std::string(buf.data(), size_t(n > 0 ? n : 0));
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_special) {
  int n = -llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, add_special, true);
  std::vector<llama_token> out(size_t(n > 0 ? n : 0));
  if (n > 0) llama_tokenize(vocab, text.data(), int32_t(text.size()), out.data(), n, add_special, true);
  return out;
}

json tokens_json(const std::vector<llama_token>& toks) {
  json a = json::array();
  for (auto t : toks) a.push_back(json(int64_t(t)));
  return a;
}

json diff_json(const common_chat_msg_diff& d) {
  json o = json::object();
  if (!d.reasoning_content_delta.empty()) o["reasoning_content"] = d.reasoning_content_delta;
  if (!d.content_delta.empty()) o["content"] = d.content_delta;
  if (d.tool_call_index != std::string::npos) {
    o["tool_call_index"] = int64_t(d.tool_call_index);
    if (!d.tool_call_delta.name.empty()) o["name"] = d.tool_call_delta.name;
    if (!d.tool_call_delta.arguments.empty()) o["arguments"] = d.tool_call_delta.arguments;
  }
  return o;
}

int usage() {
  std::fprintf(stderr,
               "usage: eklow-chat-oracle --model GGUF --cases CASES.json --out RESULTS.json\n"
               "                         [--reasoning-format NAME] [--no-prefill-assistant] [--template-kwargs JSON]\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, cases_path, out_path, reasoning = "deepseek", kwargs, tok_path;
  bool prefill_assistant = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model") model_path = next();
    else if (a == "--cases") cases_path = next();
    else if (a == "--out") out_path = next();
    else if (a == "--reasoning-format") reasoning = next();
    else if (a == "--no-prefill-assistant") prefill_assistant = false;
    else if (a == "--template-kwargs") kwargs = next();
    else if (a == "--tokenize") tok_path = next();
    else return usage();
  }
  if (model_path.empty() || (cases_path.empty() == tok_path.empty()) || out_path.empty()) return usage();

  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* model = llama_model_load_from_file(model_path.c_str(), mp);
  if (!model) {
    std::fprintf(stderr, "chat-oracle: cannot load %s\n", model_path.c_str());
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);

  if (!tok_path.empty()) {
    std::ifstream tf(tok_path, std::ios::binary);
    json strings = json::parse(std::string((std::istreambuf_iterator<char>(tf)), {}));
    json out = json::array();
    for (size_t i = 0; i < strings.size(); ++i) {
      const std::string text = strings.at(i).get<std::string>();
      json o = json::object();
      o["text"] = text;
      o["plain"] = tokens_json([&] {
        int n = -llama_tokenize(vocab, text.data(), int32_t(text.size()), nullptr, 0, false, false);
        std::vector<llama_token> t(size_t(n > 0 ? n : 0));
        if (n > 0) llama_tokenize(vocab, text.data(), int32_t(text.size()), t.data(), n, false, false);
        return t;
      }());
      const auto sp = tokenize(vocab, text, false);
      o["special"] = tokens_json(sp);
      json pieces = json::array();
      for (auto t : sp) pieces.push_back(json(piece(vocab, t)));
      o["pieces"] = pieces;
      out.push_back(o);
    }
    std::ofstream(out_path, std::ios::binary) << out.dump(1) << "\n";
    std::printf("chat-oracle: tokenized %zu strings, wrote %s\n", size_t(strings.size()), out_path.c_str());
    llama_model_free(model);
    llama_backend_free();
    return 0;
  }

  // llama-server's chat_params, as server-context.cpp fills them for `--jinja` and no other chat option.
  server_chat_params opt;
  opt.use_jinja = true;
  opt.prefill_assistant = prefill_assistant;
  opt.reasoning_format = common_reasoning_format_from_name(reasoning);
  if (!kwargs.empty()) {
    json kw = json::parse(kwargs);
    for (const auto& item : kw.items()) opt.chat_template_kwargs[item.key()] = item.value().dump();
  }
  opt.tmpls = common_chat_templates_init(model, "");
  opt.allow_image = opt.allow_audio = opt.allow_video = false;
  opt.enable_thinking = common_chat_templates_support_enable_thinking(opt.tmpls.get());

  std::ifstream cf(cases_path, std::ios::binary);
  const std::string cases_text((std::istreambuf_iterator<char>(cf)), {});
  json cases = json::parse(cases_text);

  json results = json::array();
  for (size_t ci = 0; ci < cases.size(); ++ci) {
    const json& c = cases.at(ci);
    json r = json::object();
    r["id"] = c.value("id", std::string());
    json body = c.at("request");
    std::vector<raw_buffer> files;
    common_chat_session session;
    try {
      json params = oaicompat_chat_params_parse(vocab, body, opt, files, session);
      const std::string prompt = session.prompt().str();
      r["prompt"] = prompt;
      r["prompt_tokens"] = tokens_json(tokenize(vocab, prompt, true));
      r["generation_prompt"] = session.generation_prompt();
      r["grammar"] = session.grammar();
      r["format"] = std::string(common_chat_format_name(session.format()));
      json stops = json::array();
      for (const auto& s : session.additional_stops()) stops.push_back(json(s));
      r["additional_stops"] = stops;
      r["thinking_start_tag"] = session.thinking_start_tag();
      json ends = json::array();
      for (const auto& s : session.thinking_end_tags()) ends.push_back(json(s));
      r["thinking_end_tags"] = ends;
    } catch (const std::exception& e) {
      r["error"] = std::string(e.what());
      results.push_back(r);
      continue;
    }

    json parsed = json::array();
    if (c.contains("outputs")) {
      const json& outs = c.at("outputs");
      for (size_t oi = 0; oi < outs.size(); ++oi) {
        const std::string text = outs.at(oi).get<std::string>();
        const auto toks = tokenize(vocab, text, false);
        json p = json::object();
        p["tokens"] = tokens_json(toks);
        // a fresh session for each output: the request parsed again, as a new request would be
        json body2 = c.at("request");
        std::vector<raw_buffer> files2;
        common_chat_session s2;
        oaicompat_chat_params_parse(vocab, body2, opt, files2, s2);
        common_chat_msg prev = s2.msg();
        json deltas = json::array();
        try {
          for (size_t ti = 0; ti < toks.size(); ++ti) {
            common_chat_input chunk;
            chunk.append(piece(vocab, toks[ti]), toks[ti]);
            const bool last = ti + 1 == toks.size();
            common_chat_msg now = last ? s2.finish(chunk) : s2.feed(chunk);
            json step = json::array();
            for (const auto& d : common_chat_msg_diff::compute_diffs(prev, now)) step.push_back(diff_json(d));
            deltas.push_back(step);
            prev = now;
          }
          if (toks.empty()) prev = s2.finish();
          p["message"] = prev.to_json_oaicompat();
        } catch (const std::exception& e) {
          p["parse_error"] = std::string(e.what());
          p["message"] = prev.to_json_oaicompat();
        }
        p["deltas"] = deltas;
        parsed.push_back(p);
      }
    }
    r["parsed"] = parsed;
    results.push_back(r);
  }

  std::ofstream(out_path, std::ios::binary) << results.dump(1) << "\n";
  std::printf("chat-oracle: %zu cases, wrote %s\n", size_t(cases.size()), out_path.c_str());
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
