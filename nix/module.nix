{
  config,
  lib,
  pkgs,
  ...
}:

# Each instance is a chat model served by eklow-server, as its own unit eklow-<name>.service, with the card's other models
# (EmbeddingGemma 2, the end-of-turn checker, the reranker) beside it. The server runs Inkling 2 Mini (inkling2_loop, its
# experts split between the card and the CPU) and Gemma 4 E4B (gemma4: NVIDIA's Nemotron-3.5-Bolt-E4B), and refuses anything
# else; the Qwen3.6-35B-A3B server and its options left on 2026-11-11 (eklow DECISIONS). The instances share the card and the side models' ports, so at most one runs: each conflicts with the others,
# and starting one stops the one running (systemctl start, or a mode switch such as the house's llm-mode).
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
          example = lib.literalExpression "inputs.eklow.packages.x86_64-linux.inkling-2-mini";
          description = ''
            The chat model's GGUF: Inkling 2 Mini as Unsloth publishes it (UD-Q4_K_M, the flake's `inkling-2-mini`; set
            `expertProfile`), or Gemma 4 E4B as NVIDIA publishes Nemotron-3.5-Bolt-E4B (Q4_K_M, the flake's `bolt-e4b`).
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

        slots = lib.mkOption {
          type = lib.types.ints.positive;
          default = 2;
          description = "Sequences decoded at once, each with its own cached context.";
        };

        contextSize = lib.mkOption {
          type = lib.types.nullOr lib.types.ints.positive;
          default = null;
          description = "Context of each slot, in tokens; null: the model's full window (Bolt 65536).";
        };

        kvInt8 = lib.mkOption {
          type = lib.types.nullOr lib.types.bool;
          default = null;
          description = ''
            Keep the global layers' K/V in int8 (`--kv-int8`), half of fp16's; true or false is passed as --kv-int8 or
            --kv-fp16. null: the server's default, fp16 (2 x 65536 take 2.2 GiB).
          '';
        };

        mmproj = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = lib.literalExpression "inputs.eklow.packages.x86_64-linux.inkling-2-mini-mmproj";
          description = ''
            The chat model's vision projector (llama.cpp's mmproj GGUF: Inkling 2 Mini's, the flake's
            `inkling-2-mini-mmproj`, whose weights stay in host memory; or Google's Gemma 4 E4B tower, which Bolt
            carries unchanged, `bolt-e4b-mmproj`), for image_url parts in /v1/chat/completions; null refuses them.
          '';
        };

        imageMaxTokens = lib.mkOption {
          type = lib.types.nullOr (lib.types.ints.between 1 4096);
          default = null;
          description = ''
            Inkling 2's image budget with `mmproj` (`--image-max-tokens`, llama.cpp's image_max_tokens): a larger picture
            is resized down to it. null: the server's default, 1,024 tokens (~1,280 x 1,280 pixels), about the most whose
            encoder activations fit the engine's step buffers at 8,192-token steps.
          '';
        };

        draft = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = lib.literalExpression "inputs.eklow.packages.x86_64-linux.bolt-e4b-dflash";
          description = ''
            A DFlash draft GGUF for speculative decoding, accepted losslessly; null decodes plain. Inkling 2 Mini's is Inco's
            DFlash 2 draft in Q8_0 (the flake's `inkling-2-mini-dflash2`: 714 MiB of the card, which the server reserves before
            it places the experts; ~1.4-1.8x decode, eklow DECISIONS 2026-11-26). Bolt's is NVIDIA's
            Nemotron-3.5-Bolt-E4B-DFlash in Q8_0 (the flake's `bolt-e4b-dflash`, 547 MiB of the card).
          '';
        };

        draftBlock = lib.mkOption {
          type = lib.types.nullOr (lib.types.ints.between 2 16);
          default = null;
          description = "Tokens of a verify block (the last sampled one and the drafted ones); null: the server's default, for Inkling's draft each request's own (2 to 8, from its acceptance so far), for Bolt's the trained 8.";
        };

        preserveThinking = lib.mkOption {
          type = lib.types.nullOr lib.types.bool;
          default = null;
          description = ''
            Keep earlier assistant turns' reasoning in the prompt by default (`--preserve-thinking on`, Bolt's template variable
            preserve_thinking: turns that called tools keep it); a request may still send its own. Off, the next user message
            re-renders the last turn's tool loop without its reasoning and the server prefills that loop again; on, the window
            fills 1.2-1.7x faster (eklow DECISIONS 2026-11-11). null: the server's default, off.
          '';
        };

        expertProfile = lib.mkOption {
          type = lib.types.nullOr lib.types.path;
          default = null;
          example = lib.literalExpression ''"''${inputs.eklow.packages.x86_64-linux.default}/share/eklow/inkling-2-mini-experts.json"'';
          description = ''
            Inkling 2 Mini's routing profile (`--expert-profile`, the package's share/eklow/inkling-2-mini-experts.json): its
            experts do not all fit the card, and the busiest go on it while the card has room (what the side models leave,
            less a 1 GiB margin); the rest run on the CPU. Required for Inkling; Bolt takes none.
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
        "mmproj"
        "port"
        "alias"
        "slots"
        "contextSize"
        "kvInt8"
        "extraArgs"
      ]
    ++
      map
        (
          o:
          lib.mkRemovedOptionModule [
            "services"
            "eklow"
            o
          ] "it was the Qwen3.6-35B-A3B server's, which eklow no longer has (2026-11-11)."
        )
        [
          "prefillGroup"
          "prefillGroupBusy"
          "chatTemplate"
          "expertProfile"
          "snapshotMiB"
        ];

  options.services.eklow = {
    enable = lib.mkEnableOption "eklow: Inkling 2 Mini or Nemotron-3.5-Bolt-E4B, EmbeddingGemma 2, vega's end-of-turn checker and Qwen3-Reranker on an Intel Arc A770M behind OpenAI-compatible APIs";

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
          bolt = {
            model = inputs.eklow.packages.x86_64-linux.bolt-e4b;
            alias = "nemotron-3.5-bolt-e4b";
            draft = inputs.eklow.packages.x86_64-linux.bolt-e4b-dflash;
            mmproj = inputs.eklow.packages.x86_64-linux.bolt-e4b-mmproj;
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

    prefixCache = {
      enable = lib.mkEnableOption ''
        the disk prefix cache (`--prefix-cache`): the K/V of prompts and replies kept on disk page by page, so that a conversation
        whose slot went to another one is read back instead of prefilled, and prefill-only requests (max_tokens 0) that put a
        prompt there ahead of its turn. Each instance keeps its own under /var/lib/eklow-<name>/prefix-cache'';

      keyFile = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "/run/agenix/eklow-prefix-cache-key";
        description = ''
          A root-only file with the cache's AES-256-GCM key: 32 bytes, or 64 hex digits (`openssl rand -hex 32`). The pages are
          conversations, so the cache is always encrypted. The unit reads it through systemd's LoadCredential, so it may stay
          readable by root alone (agenix's default). A path string, not a Nix path, so the key never enters the store.
        '';
      };

      maxGiB = lib.mkOption {
        type = lib.types.ints.positive;
        default = 200;
        description = ''
          The cache's size on disk. Past it the least recently used pages go, a conversation's last pages before its first. A
          page of Bolt's (256 positions) is 4 MiB of global K/V; each restore point adds two 10 MiB pages of sliding K/V.
        '';
      };
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
        1120); a Gemma 4 chat model's projector takes the same budget (Inkling's has its own, imageMaxTokens).
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
        example = lib.literalExpression "inputs.eklow.packages.x86_64-linux.vega-reader";
        description = ''
          vega's reader for the end-of-turn checker, laya's System One (POST /v1/systemone on `checker.port`, served by
          whichever instance runs, beside the chat model and the embeddings): Qwen/Qwen3.5-0.8B at 2fc06364 converted by
          the pinned llama.cpp's converter, BF16 (the flake's `vega-reader`, nix/models.nix). It takes 1,197 MiB of the
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
          every question is the engine's, zero-shot. A folder in the store is world-readable: a private policy goes in
          `policyFiles` instead.
        '';
      };

      policyFiles = lib.mkOption {
        type = lib.types.nullOr (
          lib.types.submodule {
            options = lib.genAttrs [ "heads" "calibration" "manifest" ] (
              f:
              lib.mkOption {
                type = lib.types.strMatching "/.*";
                description = "Absolute path of the policy's ${f} file, read by systemd as root (a string, so that it never enters the store).";
              }
            );
          }
        );
        default = null;
        example = {
          heads = "/run/agenix/eklow-checker-heads";
          calibration = "/run/agenix/eklow-checker-calibration";
          manifest = "/run/agenix/eklow-checker-manifest";
        };
        description = ''
          A private laya policy, given as three root-only files (agenix's, say). Before the server starts, a privileged step
          copies them, owned by the unit's dynamic user and mode 0400, into a folder of its runtime directory
          (/run/eklow-<name>/checker-policy: heads.npz, calibration.json, manifest.json), which systemd removes when the unit
          stops; the server reads it as `policy`, with the same checks. Not systemd's credentials (LoadCredential), which hold
          at most 1 MiB less a byte each (systemd 262 refuses the unit's start, status 243/CREDENTIALS): a policy's heads.npz
          is 6.4 MB.
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
        assertion = cfg.checker.policy == null || cfg.checker.policyFiles == null;
        message = "services.eklow.checker: set policy or policyFiles, not both.";
      }
      {
        assertion = !cfg.prefixCache.enable || cfg.prefixCache.keyFile != null;
        message = "services.eklow.prefixCache: set keyFile (the cache holds conversations and is always encrypted).";
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
            ++ lib.optionals (inst.draft != null) [
              "--draft"
              inst.draft
            ]
            ++ lib.optionals (inst.draftBlock != null) [
              "--draft-block"
              (toString inst.draftBlock)
            ]
            ++ lib.optionals (inst.preserveThinking != null) [
              "--preserve-thinking"
              (if inst.preserveThinking then "on" else "off")
            ]
            ++ lib.optionals (inst.mmproj != null) [
              "--mmproj"
              inst.mmproj
            ]
            ++ lib.optionals (inst.imageMaxTokens != null) [
              "--image-max-tokens"
              (toString inst.imageMaxTokens)
            ]
            ++ lib.optionals (inst.expertProfile != null) [
              "--expert-profile"
              inst.expertProfile
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
              ++ lib.optionals (cfg.checker.policyFiles != null) [
                "--checker-policy"
                "/run/${unitName name}/checker-policy"
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
            # %d is systemd's credentials directory, where LoadCredential puts the key for the dynamic user.
            ++ lib.optionals cfg.prefixCache.enable [
              "--prefix-cache"
              "/var/lib/${unitName name}/prefix-cache"
              "--prefix-cache-key"
              "%d/prefix-cache-key"
              "--prefix-cache-gib"
              (toString cfg.prefixCache.maxGiB)
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
        }
        // lib.optionalAttrs (cfg.checker.reader != null && cfg.checker.policyFiles != null) {
          RuntimeDirectory = unitName name;
          # "+": as root and outside the sandbox, so that it can read the root-only files; the runtime directory is the
          # dynamic user's, whose uid it takes.
          ExecStartPre =
            "+"
            + pkgs.writeShellScript "eklow-${name}-checker-policy" (
              with cfg.checker.policyFiles;
              ''
                set -eu
                d=/run/${unitName name}/checker-policy
                uid=$(${pkgs.coreutils}/bin/stat -c %u /run/${unitName name})
                ${pkgs.coreutils}/bin/rm -rf "$d"
                ${pkgs.coreutils}/bin/install -d -m 0700 -o "$uid" "$d"
                ${pkgs.coreutils}/bin/install -m 0400 -o "$uid" ${lib.escapeShellArg heads} "$d/heads.npz"
                ${pkgs.coreutils}/bin/install -m 0400 -o "$uid" ${lib.escapeShellArg calibration} "$d/calibration.json"
                ${pkgs.coreutils}/bin/install -m 0400 -o "$uid" ${lib.escapeShellArg manifest} "$d/manifest.json"
                ${pkgs.coreutils}/bin/chmod 0500 "$d"
              ''
            );
        }
        // lib.optionalAttrs cfg.prefixCache.enable {
          # /var/lib/private/eklow-<name> for the dynamic user, which systemd hands to the unit's uid at each start.
          StateDirectory = unitName name;
          StateDirectoryMode = "0700";
          LoadCredential = [ "prefix-cache-key:${cfg.prefixCache.keyFile}" ];
        };
      }
    ) cfg.instances;
  };
}
