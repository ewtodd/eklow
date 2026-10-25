# eklow, built with intel-llvm's clang (SYCL), kernels compiled ahead of time for the A770M so
# nothing is JIT-compiled at startup.
{
  lib,
  intel-llvm,
  intel-compute-runtime,
  intel-graphics-compiler,
  level-zero,
  cmake,
  ninja,
  pkg-config,
  syclTargets,
}:

intel-llvm.stdenv.mkDerivation {
  pname = "eklow";
  version = "0.0.1";

  src = lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      let
        base = baseNameOf path;
      in
      !(lib.hasPrefix "build" base) && base != "result" && !(lib.hasSuffix ".nix" base);
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    # ocloc, which -fsycl-targets=intel_gpu_* calls for the ahead-of-time device code.
    intel-compute-runtime
  ];

  # Level Zero itself, for allocations SYCL cannot ask for (uncached host memory in eklow-probe).
  buildInputs = [ level-zero ];

  # ocloc dlopens the graphics compiler (libigc) by soname, which is not on its runpath.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ intel-graphics-compiler ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "EKLOW_SYCL_TARGETS" (lib.concatStringsSep "," syclTargets))
  ];

  meta = {
    description = "Serving engine for Gemma 4 on an Intel Arc A770M";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
