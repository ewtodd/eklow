# The model files of fiona's deployment, as derivations: Bolt with its projector and DFlash draft, vega's reader for the
# end-of-turn checker, and Qwen3-Reranker (EmbeddingGemma 2 and vega-U are the module's defaults). Published files are
# fetched at a pinned revision by the sha256 Hugging Face lists for them. The two that nobody publishes in the form eklow
# serves are made here, with the flake's pinned llama.cpp:
# - the draft's Q8_0, quantized from NVIDIA's BF16 GGUF;
# - vega's reader, Qwen3.5-0.8B converted to a BF16 GGUF.
# Licenses: Bolt and its draft, NVIDIA's OpenMDW-1.1 (https://openmdw.ai/license/1-1/); the Gemma 4 E4B projector,
# Qwen3.5-0.8B, Qwen3-Reranker-0.6B, EmbeddingGemma 2 and vega-U, Apache-2.0.
{
  lib,
  fetchurl,
  runCommand,
  python3,
  # llama.cpp's source at the flake's pin (its converter and gguf-py) and a build of it (llama-quantize).
  llama-cpp-src,
  llama-cpp,
}:
let
  hf =
    repo: rev: name: sha256:
    fetchurl {
      inherit name sha256;
      url = "https://huggingface.co/${repo}/resolve/${rev}/${name}";
    };

  draftBf16 =
    hf "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB-DFlash" "0949a2eb7152ec6fda1a91ff9302aecbcd464fbf"
      "Nemotron-3.5-Bolt-E4B-DFlash-BF16.gguf"
      "8103aa42fd31fb661c3fbe8701d04d698ea2c5d09351dc3e8c9da639958de989";

  qwen35Rev = "2fc06364715b967f1860aea9cf38778875588b17";
  # What tools/vega/make-reader.sh downloads: the repo's *.json, *.safetensors and *.jinja.
  qwen35Files =
    lib.mapAttrsToList
      (
        name: hash:
        fetchurl {
          inherit name hash;
          url = "https://huggingface.co/Qwen/Qwen3.5-0.8B/resolve/${qwen35Rev}/${name}";
        }
      )
      {
        "chat_template.jinja" = "sha256-Jz2ODmg7iFBx+xfgjXHl8qXd+1MJdWGBaB3k9aGCLYA=";
        "config.json" = "sha256-uQuG81yOaSXvdO4E0OdY8KhFyDpCCJrYK7qpSN6bQgQ=";
        "model.safetensors-00001-of-00001.safetensors" =
          "sha256-BLHDASMd1CK4hg2zExGrJyFRE0ajLLHgecTE5fH+RpY=";
        "model.safetensors.index.json" = "sha256-2KCIOKYTsCXreVLtnbEWliE+V+dqN1Zh71wS+d1dz04=";
        "preprocessor_config.json" = "sha256-JyJUUKycZSmHLuGST8sJYv9WNINPgXBA9EQRgRb05RY=";
        "tokenizer.json" = "sha256-X55NSQGpK5l+RjwfRgVQiLbMpcphplItG59kxLuBy0I=";
        "tokenizer_config.json" = "sha256-SeK245X5WfB38emSsziRnA1KlzL8bmE5leBlV/hDUAw=";
        "video_preprocessor_config.json" = "sha256-d2ivJ8H6+pzJARwdwgBn4D+JFeA7Y1BFUOEdUGaYbRM=";
        "vocab.json" = "sha256-zpm0yymD0RiAbOCot3ejWwk+IAClA+veJYUyhMnfoAM=";
      };

  converterPython = python3.withPackages (ps: [
    ps.numpy
    ps.torch
    ps.safetensors
    ps.transformers
    ps.sentencepiece
    ps.protobuf
    ps.tqdm
    ps.pyyaml
    ps.jinja2
    ps.requests
  ]);
in
{
  bolt-e4b =
    hf "nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB" "1db54d5b12533ff1bcaa8e42cf4c667988a1f1be"
      "Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf"
      "c8af2b8aea10a690a1d9491db744a9446660173cb38fc489612b556615a69612";

  # Google's E4B vision tower, which Bolt's checkpoint carries unchanged (eklow DECISIONS 2026-11-10, "Bolt in eklow-server").
  bolt-e4b-mmproj =
    hf "ggml-org/gemma-4-E4B-it-GGUF" "27cd6836f8e2a6e5e6b57ea0e77fa72490706985"
      "mmproj-gemma-4-E4B-it-BF16.gguf"
      "fc1c6a1b566041bf41268b388bf59f0431ab86b3f4a7e34e996c0b4772509e37";

  # Byte for byte the Q8_0 eklow's draft measurements ran (sha256 46968a8f...), quantized then by a CUDA build of llama.cpp
  # 0.7.0-dev: Q8_0 is ggml's reference quantization in every build.
  bolt-e4b-dflash = runCommand "Nemotron-3.5-Bolt-E4B-DFlash-Q8_0.gguf" { } ''
    ${lib.getExe' llama-cpp "llama-quantize"} ${draftBf16} $out Q8_0
  '';

  # The converter names the model after its directory, so the checkpoint sits in one named by the revision, as in
  # Hugging Face's cache, where make-reader.sh converted it. The result (sha256 2781f548...) is not make-reader.sh's of
  # 2026-11-10 (4f4a6b43..., converted under a pip torch 2.15): 22 values of the ssm_a tensors (-exp(A_log), which the
  # converter computes with torch) are one float32 ulp apart, because nixpkgs' torch 2.13 rounds exp differently.
  vega-reader = runCommand "Qwen3.5-0.8B-BF16.gguf" { nativeBuildInputs = [ converterPython ]; } ''
    export HOME=$TMPDIR HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    src=$TMPDIR/${qwen35Rev}
    mkdir -p $src
    ${lib.concatMapStringsSep "\n" (f: "ln -s ${f} $src/${f.name}") qwen35Files}
    PYTHONPATH=${llama-cpp-src}/gguf-py python3 ${llama-cpp-src}/convert_hf_to_gguf.py $src --outtype bf16 --no-mtp --outfile $out
  '';

  qwen3-reranker =
    hf "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF" "a02f48bb4f057028298c21fa033da2b30d7742d5"
      "qwen3-reranker-0.6b-q8_0.gguf"
      "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48";
}
