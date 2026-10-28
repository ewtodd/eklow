{
  config,
  lib,
  pkgs,
  ...
}:

let
  cfg = config.services.eklow;
in
{
  options.services.eklow = {
    enable = lib.mkEnableOption "eklow, Gemma 4 and EmbeddingGemma 2 on an Intel Arc A770M behind an OpenAI-compatible API";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix { syclTargets = [ "intel_gpu_dg2_g10" ]; };
      defaultText = lib.literalExpression ''pkgs.callPackage ./package.nix { syclTargets = [ "intel_gpu_dg2_g10" ]; }'';
      description = "The eklow package, built with the host's nixpkgs (intel-llvm, compute-runtime) by default.";
    };

    model = lib.mkOption {
      type = lib.types.path;
      # Hugging Face's linked ETag for the file is its sha256.
      default = pkgs.fetchurl {
        name = "gemma-4-12b-it-qat-q4_0.gguf";
        url = "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf/resolve/29d097773436b69ff9feafd636ab4cf873786537/gemma-4-12b-it-qat-q4_0.gguf";
        sha256 = "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b";
      };
      defaultText = "Gemma 4 12B it, QAT Q4_0 (google/gemma-4-12B-it-qat-q4_0-gguf at 29d09777)";
      description = "The chat model's GGUF.";
    };

    embedModel = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = pkgs.fetchurl {
        name = "embeddinggemma-2-Q8_0.gguf";
        url = "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/58faf90940886df2d62be555c8765d00dac362c5/embeddinggemma-2-Q8_0.gguf";
        sha256 = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135";
      };
      defaultText = "EmbeddingGemma 2, Q8_0 (ggml-org/embeddinggemma-2-GGUF at 58faf909)";
      description = "The embedding model's GGUF, served on `embedPort`; null serves chat only.";
    };

    mmproj = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = lib.literalExpression ''
        pkgs.fetchurl {
          name = "gemma-4-26b-a4b-it-mmproj.gguf";
          url = "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/d1c082be9cf3c8a514acf63b8761f4b41935842e/gemma-4-26B-it-mmproj.gguf";
          sha256 = "a359953a076b877db30c31dbbb4c6d93b4a6e017ee5db5784247e4d4c0dd4f3b";
        }
      '';
      description = ''
        The chat model's vision projector (a gemma4v mmproj GGUF: the 26B A4B's), for image_url parts in chat; null
        refuses them. Its card memory (~282 MiB at 280-row images) is reserved before the experts are placed.
      '';
    };

    embedMmproj = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = lib.literalExpression ''
        pkgs.fetchurl {
          name = "mmproj-embeddinggemma-2-Q8_0.gguf";
          url = "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/58faf90940886df2d62be555c8765d00dac362c5/mmproj-embeddinggemma-2-Q8_0.gguf";
          sha256 = "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f";
        }
      '';
      description = "EmbeddingGemma 2's vision projector, for image inputs to /v1/embeddings (~264 MiB of the card); null refuses them.";
    };

    imageTokens = lib.mkOption {
      type = lib.types.ints.between 70 1120;
      default = 280;
      description = "Rows an image becomes at most (Gemma 4's budget; Google's default is 280, the projector's maximum 1120).";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      example = "0.0.0.0";
      description = "Address both ports listen on.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8081;
      description = "Port of the chat API (/v1/chat/completions, /v1/models, /health).";
    };

    embedPort = lib.mkOption {
      type = lib.types.port;
      default = 8082;
      description = "Port of the embedding API (/v1/embeddings, /v1/models, /health).";
    };

    alias = lib.mkOption {
      type = lib.types.str;
      default = "gemma-4-12b";
      description = "Model id the chat API answers to and lists.";
    };

    embedAlias = lib.mkOption {
      type = lib.types.str;
      default = "embeddinggemma-2";
      description = "Model id the embedding API lists.";
    };

    slots = lib.mkOption {
      type = lib.types.ints.positive;
      default = 2;
      description = "Sequences decoded at once, each with its own cached context.";
    };

    contextSize = lib.mkOption {
      type = lib.types.ints.positive;
      default = 262144;
      description = "Context of each slot, in tokens; Gemma 4's full window by default.";
    };

    kvInt8 = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Keep the global layers' K/V in int8. Needed for 2 x 262144 (fp16 would not fit beside the
        weights); fp16 decodes a few percent faster at depth.
      '';
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      example = [
        "--chunk"
        "512"
      ];
      description = "Further eklow-server arguments.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open both ports in the firewall.";
    };
  };

  config = lib.mkIf cfg.enable {
    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall (
      [ cfg.port ] ++ lib.optional (cfg.embedModel != null) cfg.embedPort
    );

    systemd.services.eklow = {
      description = "eklow: Gemma 4 and EmbeddingGemma 2 on the Arc A770M";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        ExecStart = lib.escapeShellArgs (
          [
            (lib.getExe' cfg.package "eklow-server")
            "--model"
            cfg.model
            "--alias"
            cfg.alias
            "--host"
            cfg.host
            "--port"
            (toString cfg.port)
            "--slots"
            (toString cfg.slots)
            "--ctx"
            (toString cfg.contextSize)
            (if cfg.kvInt8 then "--kv-int8" else "--kv-fp16")
          ]
          ++ lib.optionals (cfg.embedModel != null) [
            "--embed-model"
            cfg.embedModel
            "--embed-port"
            (toString cfg.embedPort)
            "--embed-alias"
            cfg.embedAlias
          ]
          ++ lib.optionals (cfg.mmproj != null) [
            "--mmproj"
            cfg.mmproj
          ]
          ++ lib.optionals (cfg.embedMmproj != null) [
            "--embed-mmproj"
            cfg.embedMmproj
          ]
          ++ lib.optionals (cfg.mmproj != null || cfg.embedMmproj != null) [
            "--image-tokens"
            (toString cfg.imageTokens)
          ]
          ++ cfg.extraArgs
        );
        Restart = "on-failure";
        RestartSec = 5;
        DynamicUser = true;
        # The card: /dev/dri/renderD*, which the render group may open.
        SupplementaryGroups = [
          "video"
          "render"
        ];
        # Pinned host memory for transfers.
        LimitMEMLOCK = "infinity";
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ProtectKernelTunables = true;
        ProtectKernelModules = true;
        ProtectControlGroups = true;
        RestrictAddressFamilies = [
          "AF_INET"
          "AF_INET6"
          "AF_UNIX"
        ];
        RestrictNamespaces = true;
        LockPersonality = true;
        SystemCallArchitectures = "native";
      };
    };
  };
}
