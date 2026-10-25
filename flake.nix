{
  description = "eklow: a serving engine for Gemma 4 on one Intel Arc A770M (DG2), in SYCL on Level Zero";

  inputs = {
    # The fleet's nixpkgs (/etc/nixos flake.lock), so the host and the engine agree on intel-llvm
    # 7.1.1 and compute-runtime 26.31, the versions checked on fiona (DECISIONS 2026-10-24).
    nixpkgs.url = "github:NixOS/nixpkgs/c3f15ff4df3491351fa2e0c9aa0c2f1bcd290218";
    # The oracle: the llama.cpp fiona serves today (the fleet's pin), on its own nixpkgs as the
    # fleet builds it.
    llama-cpp.url = "github:ggml-org/llama.cpp/ea0c4c2c85fb183ec0fbd30a1af4e4275a18a064";
  };

  outputs =
    {
      self,
      nixpkgs,
      llama-cpp,
    }:
    let
      system = "x86_64-linux";
      # Callers (the NixOS host) pass their own pkgs through `lib.mkEklow` so there is one nixpkgs.
      mkEklow =
        {
          pkgs,
          syclTargets ? [ "intel_gpu_dg2_g10" ],
        }:
        pkgs.callPackage ./nix/package.nix { inherit syclTargets; };

      pkgs = import nixpkgs { inherit system; };
      eklow = mkEklow { inherit pkgs; };
      # llama.cpp's default package is built for generic x86-64 with no SIMD kernels; AVX2, which
      # e-desktop and fiona both have, makes the oracle about 3.5x faster.
      llamaCpuAvx2 = llama-cpp.packages.${system}.default.overrideAttrs (old: {
        cmakeFlags = (old.cmakeFlags or [ ]) ++ [
          "-DGGML_NATIVE=OFF"
          "-DGGML_AVX=ON"
          "-DGGML_AVX2=ON"
          "-DGGML_FMA=ON"
          "-DGGML_F16C=ON"
          "-DGGML_BMI2=ON"
        ];
      });
      oracle = llama-cpp.inputs.nixpkgs.legacyPackages.${system}.callPackage ./nix/oracle.nix {
        llama-cpp = llamaCpuAvx2;
      };
      # Level Zero and OpenCL drivers from this nixpkgs, so the shell works whatever the host's
      # /run/opengl-driver carries (installed binaries use the system's, see nix/package.nix).
      driverLibs = with pkgs; [
        level-zero
        intel-compute-runtime
        intel-graphics-compiler
        ocl-icd
      ];
    in
    {
      lib.${system} = { inherit mkEklow; };

      packages.${system} = {
        default = eklow;
        inherit eklow oracle llamaCpuAvx2;
      };

      devShells.${system}.default = (pkgs.mkShell.override { stdenv = pkgs.intel-llvm.stdenv; }) {
        inputsFrom = [ eklow ];
        packages = with pkgs; [
          clinfo
          gdb
          jq
          (python3.withPackages (ps: [
            ps.numpy
            ps.gguf
            ps.tokenizers
          ]))
        ];
        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath driverLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          export OCL_ICD_VENDORS=${pkgs.intel-compute-runtime}/etc/OpenCL/vendors
          # Free-memory queries go through zesInit; Sysman's environment path fails under xe.
          export UR_L0_ENABLE_ZESINIT_DEFAULT=1 UR_L0_ENABLE_SYSMAN_ENV_DEFAULT=0
        '';
      };

      formatter.${system} = pkgs.nixfmt;
    };
}
