{
  description = "eklow: a serving engine for Gemma 4 on one Intel Arc A770M (DG2), in SYCL on Level Zero";

  inputs = {
    # The fleet's nixpkgs (/etc/nixos flake.lock), so the host and the engine agree on intel-llvm
    # 7.1.1 and compute-runtime 26.31, the versions checked on fiona (DECISIONS 2026-10-24).
    nixpkgs.url = "github:NixOS/nixpkgs/c3f15ff4df3491351fa2e0c9aa0c2f1bcd290218";
  };

  outputs =
    { self, nixpkgs }:
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
        inherit eklow;
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
