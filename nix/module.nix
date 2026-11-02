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
    enable = lib.mkEnableOption "eklow, Qwen3.6-35B-A3B with DFlash 2 speculative decoding and EmbeddingGemma 2 on an Intel Arc A770M behind an OpenAI-compatible API";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix { syclTargets = [ "intel_gpu_dg2_g10" ]; };
      defaultText = lib.literalExpression ''pkgs.callPackage ./package.nix { syclTargets = [ "intel_gpu_dg2_g10" ]; }'';
      description = "The eklow package, built with the host's nixpkgs (intel-llvm, compute-runtime) by default.";
    };

    model = lib.mkOption {
      type = lib.types.path;
      example = lib.literalExpression "/models/stock-EKK8.gguf";
      description = ''
        The chat model's GGUF: Qwen3.6-35B-A3B or a finetune of it (llama.cpp's qwen35moe), in the formats the engine runs
        (Q4_K or Q4_0 experts, Q8_0 dense matrices, a Q6_K head). Stock Qwen3.6 and Ornith-1.5 are the two served and measured.
      '';
    };

    draft = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      example = lib.literalExpression "/models/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf";
      description = ''
        A DFlash 2 draft GGUF (arch "dflash"; Inco AI's for stock Qwen3.6, jzinno's for Ornith) for speculative decoding,
        accepted losslessly; null decodes plain. About 1.7 GiB of the card is kept for it when set.
      '';
    };

    draftBlock = lib.mkOption {
      type = lib.types.nullOr (lib.types.ints.between 2 16);
      default = null;
      description = "Tokens of a verify block (the last sampled one and the drafted ones); null: 4, or the draft's trained block if smaller (a block costs ~8.5 ms a row beyond the first on this card; 4 measured best).";
    };

    chatTemplate = lib.mkOption {
      type = lib.types.nullOr (
        lib.types.enum [
          "stock"
          "ornith"
        ]
      );
      default = null;
      description = ''
        Which of Qwen3.6's two chat templates the model's repo uses, when the GGUF's own is not one eklow knows (the official
        Ornith GGUFs carry Unsloth's edit): stock drops earlier turns' reasoning from the prompt, ornith keeps it. null:
        recognized from the GGUF, which then must carry one of the two published templates exactly.
      '';
    };

    expertProfile = lib.mkOption {
      type = lib.types.path;
      default = "${cfg.package}/share/eklow/qwen36-experts.json";
      defaultText = lib.literalExpression ''"''${cfg.package}/share/eklow/qwen36-experts.json"'';
      example = lib.literalExpression ''"''${cfg.package}/share/eklow/ornith-1.5-experts.json"'';
      description = ''
        The routing profile of the model (which experts of each layer are busiest): the package's share/eklow has stock
        Qwen3.6's (qwen36-experts.json) and Ornith-1.5's (ornith-1.5-experts.json). The experts that fit the card stay on it,
        the rest are computed on the CPU from pinned memory.
      '';
    };

    snapshotMiB = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 4096;
      description = ''
        Pinned host memory for snapshots of the DeltaNet state (64 MiB each), which let a conversation's or a document's
        prefix be reused; 0 keeps none, and only a slot's live sequence is reused.
      '';
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
      description = "Rows an image becomes at most in EmbeddingGemma 2's input (Google's default is 280, the projector's maximum 1120).";
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
      default = "qwen3.6-35b-a3b";
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
      description = "Context of each slot, in tokens; Qwen3.6's full window by default.";
    };

    kvInt8 = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Keep the attention layers' K/V in int8. Needed for 2 x 262144 (fp16 would not fit beside the
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
      description = "eklow: Qwen3.6-35B-A3B and EmbeddingGemma 2 on the Arc A770M";
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
            "--expert-profile"
            cfg.expertProfile
            "--snapshot-mib"
            (toString cfg.snapshotMiB)
          ]
          ++ lib.optionals (cfg.draft != null) [
            "--draft"
            cfg.draft
          ]
          ++ lib.optionals (cfg.draftBlock != null) [
            "--draft-block"
            (toString cfg.draftBlock)
          ]
          ++ lib.optionals (cfg.chatTemplate != null) [
            "--chat-template"
            cfg.chatTemplate
          ]
          ++ lib.optionals (cfg.embedModel != null) [
            "--embed-model"
            cfg.embedModel
            "--embed-port"
            (toString cfg.embedPort)
            "--embed-alias"
            cfg.embedAlias
          ]
          ++ lib.optionals (cfg.embedMmproj != null) [
            "--embed-mmproj"
            cfg.embedMmproj
          ]
          ++ lib.optionals (cfg.embedMmproj != null) [
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
