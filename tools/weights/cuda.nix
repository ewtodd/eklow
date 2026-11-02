let
  fl = builtins.getFlake "github:ggml-org/llama.cpp/ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064";
in
fl.legacyPackages.x86_64-linux.llamaPackagesCuda.llama-cpp.overrideAttrs (old: {
  cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DCMAKE_CUDA_ARCHITECTURES=89" ];
})
