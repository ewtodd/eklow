# eklow, built with intel-llvm's clang (SYCL), kernels compiled ahead of time for the A770M so
# nothing is JIT-compiled at startup.
{
  lib,
  intel-llvm,
  intel-compute-runtime,
  intel-graphics-compiler,
  level-zero,
  addDriverRunpath,
  cmake,
  httplib,
  makeWrapper,
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
    makeWrapper
    ninja
    pkg-config
    # ocloc, which -fsycl-targets=intel_gpu_* calls for the ahead-of-time device code.
    intel-compute-runtime
  ];

  # Level Zero itself, for allocations SYCL cannot ask for (uncached host memory in eklow-probe);
  # cpp-httplib (header-only) for the server.
  buildInputs = [
    level-zero
    httplib
  ];

  # ocloc dlopens the graphics compiler (libigc) by soname, which is not on its runpath.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [ intel-graphics-compiler ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "EKLOW_SYCL_TARGETS" (lib.concatStringsSep "," syclTargets))
  ];

  # nixpkgs' Level Zero loader has no runpath, so it never finds the system driver in
  # /run/opengl-driver; a runpath on these binaries would not help, since the loader dlopens the
  # driver itself. Sysman's environment path fails under xe, so free-memory queries go through
  # zesInit.
  postFixup = ''
    for b in $out/bin/*; do
      wrapProgram "$b" \
        --prefix LD_LIBRARY_PATH : ${addDriverRunpath.driverLink}/lib \
        --set-default UR_L0_ENABLE_ZESINIT_DEFAULT 1 \
        --set-default UR_L0_ENABLE_SYSMAN_ENV_DEFAULT 0
    done
  '';

  meta = {
    description = "Serving engine for Gemma 4 on an Intel Arc A770M";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
