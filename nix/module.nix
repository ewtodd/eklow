{
  config,
  lib,
  pkgs,
  ...
}:

# Each instance is a chat model served by the one eklow-server, as its own unit eklow-<name>.service: the server reads
# the model's architecture (Qwen3.6-35B-A3B and its finetunes, llama.cpp's qwen35moe; Gemma 4 E4B, gemma4, as NVIDIA's
# Nemotron-3.5-Bolt-E4B; anything else is refused) and runs its engine. The instances share the card and the embedding port, so at most one runs: each conflicts with
# the others, and starting one stops the one running (systemctl start, or a mode switch such as the house's llm-mode).
let
  cfg = config.services.eklow;

  unitName = name: "eklow-${name}";
  units = map (name: "${unitName name}.service") (lib.attrNames cfg.instances);

  instanceOptions =
    { name, ... }:
    {
      options = {
        model = lib.mkOption {
          type = lib.types.path;
          example = lib.literalExpression "/models/stock-EKK8.gguf";
          description = ''
            The chat model's GGUF, in the formats the engines run: Qwen3.6-35B-A3B or a finetune of it (Q4_K or Q4_0
            experts, Q8_0 dense matrices, a Q6_K head; stock Qwen3.6, Ornith-1.5 and Sage-A1.5 are served and
            measured), or Gemma 4 E4B as NVIDIA publishes Nemotron-3.5-Bolt-E4B (Q4_K_M, the whole model on the card).
          '';
        };

        alias = lib.mkOption {
          type = lib.types.str;
          default = name;
          defaultText = lib.literalExpression "<name>";
          description = "Model id the chat API answers to and lists.";
        };

        port = lib.mkOption {
          type = lib.types.port;
          default = 8081;
          description = ''
            Port of the chat API (/v1/chat/completions, /v1/models, /health). Give each instance its own when a gateway
            tells them apart by whether their port answers.
          '';
        };

        expertProfile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = lib.literalExpression ''"''${config.services.eklow.package}/share/eklow/ornith-1.5-experts.json"'';
          description = ''
            The routing profile of the model (which experts of each layer are busiest): the package's share/eklow has
            stock Qwen3.6's (qwen36-experts.json) and Ornith-1.5's (ornith-1.5-experts.json). The experts that fit the card stay
            on it, the rest are computed on the CPU from
            pinned memory. null: every expert on the card, which no MoE served here fits at long contexts.
          '';
        };

        slots = lib.mkOption {
          type = lib.types.ints.positive;
          default = 2;
          description = "Sequences decoded at once, each with its own cached context.";
        };

        contextSize = lib.mkOption {
          type = lib.types.nullOr lib.types.ints.positive;
          default = null;
          description = "Context of each slot, in tokens; null: the model's full window (Qwen3.6 262144, Bolt 65536).";
        };

        kvInt8 = lib.mkOption {
          type = lib.types.nullOr lib.types.bool;
          default = null;
          description = ''
            Keep the attention layers' K/V in int8, which 2 x 262144 of Qwen3.6 needs (fp16 would not fit beside the weights);
            fp16 decodes a few percent faster at depth. null: the server's default for the model (int8 for Qwen3.6, fp16 for
            Bolt, whose 2 x 65536 take 2 GiB in fp16).
          '';
        };

        mmproj = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = lib.literalExpression ''
            pkgs.fetchurl {
              name = "mmproj-Ornith-1.5-35B-A3B-bf16.gguf";
              url = "https://huggingface.co/bartowski/Ornith-1.5-35B-A3B-GGUF/resolve/64b0493d34/mmproj-Ornith-1.5-35B-A3B-bf16.gguf";
              sha256 = "...";
            }
          '';
          description = ''
            The chat model's vision projector (llama.cpp's mmproj GGUF: Qwen3.6's tower, or a Gemma 4 one for Bolt),
            for image_url parts in /v1/chat/completions; null refuses them. The encoder's weights and scratch come out of
            the experts' share of the card.
          '';
        };

        draft = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = lib.literalExpression "/models/Qwen3.6-35B-A3B-DFlash2-inco-Q8_0.gguf";
          description = ''
            A DFlash draft GGUF (arch "dflash") for speculative decoding, accepted losslessly; null decodes plain. Qwen3.6:
            a DFlash 2 draft (Inco AI's serves stock Qwen3.6 and Ornith-1.5 alike, DECISIONS 2026-11-01), ~0.8 GiB of the
            card. Bolt: NVIDIA's Nemotron-3.5-Bolt-E4B-DFlash.
          '';
        };

        draftBlock = lib.mkOption {
          type = lib.types.nullOr (lib.types.ints.between 2 16);
          default = null;
          description = "Tokens of a verify block (the last sampled one and the drafted ones); null: Qwen3.6 4, or the draft's trained block if smaller (a block costs ~8.5 ms a row beyond the first on this card; 4 measured best); Bolt the draft's trained block, 8 (dense, its rows are cheap).";
        };

        prefillGroup = lib.mkOption {
          type = lib.types.ints.positive;
          default = 1;
          description = ''
            Qwen3.6 only: prefill chunks a lone long prompt may run layer by layer as one group (`--group`), so that each
            cold expert crosses PCIe once per layer per group instead of once per chunk: 8 prefills 1.2-1.4x faster for
            ~185 MB of card memory (3 experts a layer fewer; DECISIONS 2026-11-03). 1: plain chunks.
          '';
        };

        prefillGroupBusy = lib.mkOption {
          type = lib.types.ints.unsigned;
          default = 0;
          description = ''
            Qwen3.6 only: the most tokens a group may take while another slot is generating (`--group-busy`): a group
            stalls that stream for its whole length (4096: its longest pause 4-5 s against ~2 s with plain chunks). 0:
            plain chunks then.
          '';
        };

        chatTemplate = lib.mkOption {
          type = lib.types.nullOr (
            lib.types.enum [
              "stock"
              "ornith"
              "sage"
            ]
          );
          default = null;
          description = ''
            Qwen3.6 only: which of the Qwen3.6 chat templates eklow renders the model's repo uses, when the GGUF's own is not
            one eklow knows (the official Ornith GGUFs carry Unsloth's edit): stock drops earlier turns' reasoning from the
            prompt, ornith keeps it, sage is stock plus Sage-A1.5's verification switch. null: recognized from the GGUF,
            which then must carry one of the published templates exactly.
          '';
        };

        snapshotMiB = lib.mkOption {
          type = lib.types.nullOr lib.types.ints.unsigned;
          default = null;
          description = ''
            Qwen3.6 only: pinned host memory for snapshots of the DeltaNet state (64 MiB each), which let a conversation's
            or a document's prefix be reused; 0 keeps none, and only a slot's live sequence is reused. null: the server's
            default, 4096.
          '';
        };

        kvPool = lib.mkOption {
          type = lib.types.nullOr lib.types.ints.unsigned;
          default = null;
          example = 262144;
          description = ''
            Qwen3.6 only: the K/V pool of all the slots together, in positions (`--kv-pool`); each slot may still use up to
            contextSize. With 2 slots of 262144 and a pool of 262144 the K/V costs half and the card keeps that memory's
            worth of experts more (on fiona 91 a layer instead of 48). When the pool runs out an idle slot gives its pages
            back and re-prefills later; a request that does not fit waits. null: slots x contextSize, a pool per slot.
          '';
        };

        verification = lib.mkOption {
          type = lib.types.nullOr lib.types.bool;
          default = null;
          description = ''
            Sage-A1.5's template only: turn its verification protocol on by default (`--verification on`) for conversations
            with a system message; a request may still send its own `verification`. null: the server's default, off.
          '';
        };

        preserveThinking = lib.mkOption {
          type = lib.types.nullOr lib.types.bool;
          default = null;
          description = ''
            Stock's and Sage-A1.5's templates, and Bolt's: keep earlier assistant turns' reasoning in the prompt by default
            (`--preserve-thinking on`, the template variable preserve_thinking; Bolt's keeps it for turns that called tools);
            a request may still send its own. Off, every
            new user message re-renders the turn before it without its reasoning, and the server prefills that whole turn
            again (an agentic turn of 190k tokens would take three minutes on fiona). Ornith's template always keeps it. null: the
            server's default, off.
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
      };
    };
in
{
  imports =
    map
      (
        o:
        lib.mkRemovedOptionModule [
          "services"
          "eklow"
          o
        ] "eklow serves named instances now: set services.eklow.instances.<name>.${o}."
      )
      [
        "model"
        "draft"
        "draftBlock"
        "prefillGroup"
        "prefillGroupBusy"
        "chatTemplate"
        "expertProfile"
        "snapshotMiB"
        "mmproj"
        "port"
        "alias"
        "slots"
        "contextSize"
        "kvInt8"
        "extraArgs"
      ];

  options.services.eklow = {
    enable = lib.mkEnableOption "eklow, Qwen3.6-35B-A3B finetunes (with DFlash 2 speculative decoding) or Nemotron-3.5-Bolt-E4B, and EmbeddingGemma 2, on an Intel Arc A770M behind an OpenAI-compatible API";

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.callPackage ./package.nix { syclTargets = [ "intel_gpu_dg2_g10" ]; };
      defaultText = lib.literalExpression ''pkgs.callPackage ./package.nix { syclTargets = [ "intel_gpu_dg2_g10" ]; }'';
      description = "The eklow package, built with the host's nixpkgs (intel-llvm, compute-runtime) by default.";
    };

    instances = lib.mkOption {
      type = lib.types.attrsOf (lib.types.submodule instanceOptions);
      default = { };
      example = lib.literalExpression ''
        {
          ornith = {
            model = /models/Ornith-1.5-35B-A3B-Q4_K-Q8_0.gguf;
            alias = "ornith-1.5-35b-a3b";
            expertProfile = "''${config.services.eklow.package}/share/eklow/ornith-1.5-experts.json";
          };
          bolt = {
            model = /models/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf;
            alias = "nemotron-3.5-bolt-e4b";
            port = 8083;
          };
        }
      '';
      description = ''
        The chat models, each served as eklow-<name>.service. They exclude each other (one card): starting one stops
        another that runs. `autoStart` says which starts at boot.
      '';
    };

    autoStart = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default =
        let
          names = lib.attrNames cfg.instances;
        in
        if lib.length names == 1 then lib.head names else null;
      defaultText = "the only instance, if there is exactly one; otherwise null";
      description = "The instance started at boot; null starts none (a mode switch starts one).";
    };

    embedModel = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = pkgs.fetchurl {
        name = "embeddinggemma-2-Q8_0.gguf";
        url = "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/58faf90940886df2d62be555c8765d00dac362c5/embeddinggemma-2-Q8_0.gguf";
        sha256 = "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135";
      };
      defaultText = "EmbeddingGemma 2, Q8_0 (ggml-org/embeddinggemma-2-GGUF at 58faf909)";
      description = "The embedding model's GGUF, served on `embedPort` by whichever instance runs; null serves chat only.";
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
      description = ''
        Rows an image becomes at most in EmbeddingGemma 2's input (Google's default is 280, the projector's maximum
        1120). Qwen3.6's encoder has its own budget (`--image-max-tokens`).
      '';
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      example = "0.0.0.0";
      description = "Address the ports listen on.";
    };

    embedPort = lib.mkOption {
      type = lib.types.port;
      default = 8082;
      description = "Port of the embedding API (/v1/embeddings, /v1/models, /health).";
    };

    embedAlias = lib.mkOption {
      type = lib.types.str;
      default = "embeddinggemma-2";
      description = "Model id the embedding API lists.";
    };

    checker = {
      reader = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        example = "/var/lib/eklow/Qwen3.5-0.8B-BF16.gguf";
        description = ''
          vega's reader for the end-of-turn checker, laya's System One (POST /v1/systemone on `checker.port`, served by
          whichever instance runs, beside the chat model and the embeddings): Qwen/Qwen3.5-0.8B at 2fc06364 converted by
          the pinned llama.cpp's converter, BF16 (tools/vega/make-reader.sh; sha256 4f4a6b43...). It takes 1,197 MiB of the
          card with checker.maxTokens 16384 (measured on fiona, the driver's buffers included), which comes out of what the
          chat model's experts or K/V would get. null serves no checker.
        '';
      };

      vega = lib.mkOption {
        type = lib.types.path;
        default = pkgs.linkFarm "vega-U-4d205028" (
          map
            (f: {
              inherit (f) name;
              path = pkgs.fetchurl {
                url = "https://huggingface.co/nandakishorm/vega-U/resolve/4d2050287982ca7a7bb0c22fd19a389f5f94474e/${f.name}";
                inherit (f) sha256;
              };
            })
            [
              {
                name = "vega_config.json";
                sha256 = "000461315c798b6eca62323476360c14cf31b4e1adcba21ab8266fca41d3fcf8";
              }
              {
                name = "engine.safetensors";
                sha256 = "78f96be7c6a06abb873e573fcf9e43ef467010a30c72cd6cac32538cd7078db8";
              }
              {
                name = "port.safetensors";
                sha256 = "398ae79d5e3956a3ed809d500fbdf57982d303c719e464bb81ba12a5ee7d23bd";
              }
              {
                name = "stats.npz";
                sha256 = "f891292bb7cb2a7616bb925a3c2fe182c72b2899f36c16494445229bed1d3a12";
              }
            ]
        );
        defaultText = "nandakishorm/vega-U at 4d205028 (its Qwen3.5-0.8B port)";
        description = "The vega-U model directory for the reader: vega_config.json, engine.safetensors, port.safetensors, stats.npz.";
      };

      policy = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = ''
          A laya policy folder (laya-policy/1, as laya's save_calibration writes it: heads.npz, calibration.json,
          manifest.json), fitted on this engine: a question whose id has a head is answered by it, and the calibration's
          temperatures apply. Its manifest's hashes and engine are checked at startup; a signed policy is refused. null:
          every question is the engine's, zero-shot.
        '';
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8084;
        description = "Port of the checker's API (POST /v1/systemone, GET /health).";
      };

      alias = lib.mkOption {
        type = lib.types.str;
        default = "vega-u";
        description = "Model name the checker's answers carry.";
      };

      maxTokens = lib.mkOption {
        type = lib.types.ints.between 1024 73728;
        default = 16384;
        description = ''
          The longest prompt the reader holds (its attention cache, 10 KiB a token); a longer state is answered 413, and
          son-of-anton then keeps the draft. vegaml reads up to 73,728, but a state that long takes the reader seconds,
          past son-of-anton's 3 s timeout.
        '';
      };
    };

    reranker = {
      model = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        example = lib.literalExpression ''
          pkgs.fetchurl {
            name = "qwen3-reranker-0.6b-q8_0.gguf";
            url = "https://huggingface.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/resolve/a02f48bb4f057028298c21fa033da2b30d7742d5/qwen3-reranker-0.6b-q8_0.gguf";
            sha256 = "22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48";
          }
        '';
        description = ''
          A Qwen3-Reranker GGUF with its yes/no head and rerank template (ggml-org's conversion; a plain causal-LM conversion
          has neither), served by whichever instance runs as llama.cpp's POST /v1/rerank on `reranker.port`: 796 MiB of the
          card at maxTokens 4096 (measured on fiona), out of the chat model's share. null serves none.
        '';
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8085;
        description = ''
          Port of the reranker's API (POST /v1/rerank, /rerank, GET /health). Not llama.cpp's 8083 by default: a port another
          service holds stops the whole server (chat included) at startup, so take 8083 only once llama.cpp's reranker is gone.
        '';
      };

      alias = lib.mkOption {
        type = lib.types.str;
        default = "qwen3-reranker-0.6b";
        description = "Model name the reranker's answers carry when a request names none.";
      };

      maxTokens = lib.mkOption {
        type = lib.types.ints.between 256 32768;
        default = 4096;
        description = ''
          The longest query-document pair, in tokens, and the rows of one pass of the reranker (its activations grow with
          it); a longer pair is refused with 400, as llama.cpp refuses a pair past its ubatch (4,096 on fiona).
        '';
      };
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open the instances' ports, the embedding port, the checker's and the reranker's ports in the firewall.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.instances != { };
        message = "services.eklow: define at least one instance (services.eklow.instances.<name>.model).";
      }
      {
        assertion = cfg.autoStart == null || cfg.instances ? ${cfg.autoStart};
        message = "services.eklow.autoStart '${toString cfg.autoStart}' is not an instance (${lib.concatStringsSep ", " (lib.attrNames cfg.instances)}).";
      }
      {
        assertion = lib.all (n: builtins.match "[a-z0-9][a-z0-9-]*" n != null) (
          lib.attrNames cfg.instances
        );
        message = "services.eklow.instances: names are unit name suffixes, [a-z0-9][a-z0-9-]*.";
      }
    ];

    networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall (
      lib.unique (
        lib.mapAttrsToList (_: inst: inst.port) cfg.instances
        ++ lib.optional (cfg.embedModel != null) cfg.embedPort
        ++ lib.optional (cfg.checker.reader != null) cfg.checker.port
        ++ lib.optional (cfg.reranker.model != null) cfg.reranker.port
      )
    );

    systemd.services = lib.mapAttrs' (
      name: inst:
      lib.nameValuePair (unitName name) {
        description = "eklow ${name}: ${inst.alias} and EmbeddingGemma 2 on the Arc A770M";
        after = [ "network-online.target" ];
        wants = [ "network-online.target" ];
        wantedBy = lib.optional (cfg.autoStart == name) "multi-user.target";
        conflicts = lib.remove "${unitName name}.service" units;
        serviceConfig = {
          ExecStart = lib.escapeShellArgs (
            [
              (lib.getExe' cfg.package "eklow-server")
              "--model"
              inst.model
              "--alias"
              inst.alias
              "--host"
              cfg.host
              "--port"
              (toString inst.port)
              "--slots"
              (toString inst.slots)
            ]
            ++ lib.optionals (inst.contextSize != null) [
              "--ctx"
              (toString inst.contextSize)
            ]
            ++ lib.optional (inst.kvInt8 != null) (if inst.kvInt8 then "--kv-int8" else "--kv-fp16")
            ++ lib.optionals (inst.expertProfile != null) [
              "--expert-profile"
              inst.expertProfile
            ]
            ++ lib.optionals (inst.snapshotMiB != null) [
              "--snapshot-mib"
              (toString inst.snapshotMiB)
            ]
            ++ lib.optionals (inst.draft != null) [
              "--draft"
              inst.draft
            ]
            ++ lib.optionals (inst.draftBlock != null) [
              "--draft-block"
              (toString inst.draftBlock)
            ]
            ++ lib.optionals (inst.prefillGroup > 1) [
              "--group"
              (toString inst.prefillGroup)
              "--group-busy"
              (toString inst.prefillGroupBusy)
            ]
            ++ lib.optionals (inst.chatTemplate != null) [
              "--chat-template"
              inst.chatTemplate
            ]
            ++ lib.optionals (inst.kvPool != null) [
              "--kv-pool"
              (toString inst.kvPool)
            ]
            ++ lib.optionals (inst.verification != null) [
              "--verification"
              (if inst.verification then "on" else "off")
            ]
            ++ lib.optionals (inst.preserveThinking != null) [
              "--preserve-thinking"
              (if inst.preserveThinking then "on" else "off")
            ]
            ++ lib.optionals (inst.mmproj != null) [
              "--mmproj"
              inst.mmproj
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
            ++ lib.optionals (cfg.embedMmproj != null || inst.mmproj != null) [
              "--image-tokens"
              (toString cfg.imageTokens)
            ]
            ++ lib.optionals (cfg.checker.reader != null) (
              [
                "--checker-reader"
                cfg.checker.reader
                "--checker-vega"
                cfg.checker.vega
                "--checker-port"
                (toString cfg.checker.port)
                "--checker-alias"
                cfg.checker.alias
                "--checker-max-tokens"
                (toString cfg.checker.maxTokens)
              ]
              ++ lib.optionals (cfg.checker.policy != null) [
                "--checker-policy"
                cfg.checker.policy
              ]
            )
            ++ lib.optionals (cfg.reranker.model != null) [
              "--rerank-model"
              cfg.reranker.model
              "--rerank-port"
              (toString cfg.reranker.port)
              "--rerank-alias"
              cfg.reranker.alias
              "--rerank-max-tokens"
              (toString cfg.reranker.maxTokens)
            ]
            ++ inst.extraArgs
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
      }
    ) cfg.instances;
  };
}
