# eklow-oracle against a CUDA build of the same llama.cpp revision (the CPU oracle's pin), for measuring the
# noise band of llama.cpp's own backends (CPU against CUDA) on a GPU box such as e-desktop's 4090:
#   nix-build --impure nix/oracle-cuda.nix -o oracle-cuda   (then: oracle-cuda/bin/eklow-oracle ... --ngl 99)
# CUDA architecture 89 (Ada); change it for another card.
let
  fl = builtins.getFlake "github:ggml-org/llama.cpp/ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064";
  llamaCuda = fl.legacyPackages.x86_64-linux.llamaPackagesCuda.llama-cpp.overrideAttrs (old: {
    cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DCMAKE_CUDA_ARCHITECTURES=89" ];
  });
in
# libggml-cuda leaves the driver API (libcuda.so.1) to the system; the link may leave it undefined and run with
# LD_LIBRARY_PATH=/run/opengl-driver/lib.
(fl.inputs.nixpkgs.legacyPackages.x86_64-linux.callPackage ./oracle.nix { llama-cpp = llamaCuda; }).overrideAttrs
  (o: {
    buildPhase = builtins.replaceStrings [ "-lggml-base" ] [ "-lggml-base -Wl,--allow-shlib-undefined" ] o.buildPhase;
  })
