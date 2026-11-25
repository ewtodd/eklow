# eklow-oracle, eklow-chat-oracle and eklow-grammar-oracle against llama.cpp master c74c71a6 built with CUDA (sm_89, e-workstation's 4090):
# Inkling 2 Mini's reference (DECISIONS 2026-11-24). The same llama.cpp the private replay ran: Inkling's eager tool
# grammar (#32833) and its tokenizer fix (#32799) are in it. Two build fixes: no web UI (its PWA step fails in the
# sandbox), and NVML's headers for ggml-cuda.
#   nix-build --impure nix/oracle-inkling.nix -o oracle-inkling
#   LD_LIBRARY_PATH=/run/opengl-driver/lib oracle-inkling/bin/eklow-oracle --ngl 99 ...
let
  rev = "c74c71a6f7a0b06df30f979520e5d9a86a4d60ad";
  fl = builtins.getFlake "github:ggml-org/llama.cpp/${rev}";
  pkgs = fl.inputs.nixpkgs.legacyPackages.x86_64-linux;
  pkgsCuda = import fl.inputs.nixpkgs {
    system = "x86_64-linux";
    config.cudaSupport = true;
    config.allowUnfree = true;
  };
  llamaCuda =
    (fl.legacyPackages.x86_64-linux.llamaPackagesCuda.llama-cpp.override { useWebUi = false; }).overrideAttrs
      (old: {
        webui = "";
        buildInputs = (old.buildInputs or [ ]) ++ [ pkgsCuda.cudaPackages.cuda_nvml_dev ];
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DCMAKE_CUDA_ARCHITECTURES=89" ];
      });
  # common/, tools/server/ and vendor/ headers of the same revision, for the chat oracle (the package installs only
  # llama.h and ggml's headers, but ships libllama-common and libllama-server-impl).
  src = fl.outPath;
in
pkgs.stdenv.mkDerivation {
  pname = "eklow-oracle-inkling";
  version = "c74c71a6";
  srcs = [ ../tools/ref/oracle ];
  sourceRoot = "oracle";
  buildInputs = [ llamaCuda ];
  buildPhase = ''
    runHook preBuild
    # libggml-cuda leaves the driver API (libcuda.so.1) to the system: run with LD_LIBRARY_PATH=/run/opengl-driver/lib.
    $CXX -O2 -std=c++17 -Wall -Wextra oracle.cpp -o eklow-oracle -lmtmd -lllama -lggml -lggml-base \
      -Wl,--allow-shlib-undefined
    $CXX -O2 -std=c++17 -Wall -Wextra grammar_oracle.cpp -o eklow-grammar-oracle -lllama -lggml -lggml-base \
      -Wl,--allow-shlib-undefined
    $CXX -O2 -std=c++17 -Wall -Wextra chat_oracle.cpp -o eklow-chat-oracle \
      -I${src}/common -I${src}/tools/server -I${src}/vendor -I${src}/include -I${src}/ggml/include \
      -L${llamaCuda}/lib -lllama-server-impl -lllama-common -lmtmd -lllama -lggml -lggml-base \
      -Wl,-rpath,${llamaCuda}/lib -Wl,--allow-shlib-undefined
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    install -Dm755 eklow-oracle $out/bin/eklow-oracle
    install -Dm755 eklow-chat-oracle $out/bin/eklow-chat-oracle
    install -Dm755 eklow-grammar-oracle $out/bin/eklow-grammar-oracle
    runHook postInstall
  '';
}
