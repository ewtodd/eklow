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
      default = 65536;
      description = "Context of each slot, in tokens.";
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
          ]
          ++ lib.optionals (cfg.embedModel != null) [
            "--embed-model"
            cfg.embedModel
            "--embed-port"
            (toString cfg.embedPort)
            "--embed-alias"
            cfg.embedAlias
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
