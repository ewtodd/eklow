# Environment for tools/weights: the Python the pinned convert_hf_to_gguf.py needs (torch is only used
# for lazy tensor shapes and BF16 casts), curl and jq for the downloads. nixpkgs is the one flake.lock
# pins for eklow itself. Enter with: nix-shell tools/weights/shell.nix
let
  lock = builtins.fromJSON (builtins.readFile ../../flake.lock);
  n = lock.nodes.nixpkgs.locked;
  pkgs = import (builtins.fetchTree {
    type = "github";
    owner = n.owner;
    repo = n.repo;
    rev = n.rev;
    narHash = n.narHash;
  }) { config.allowUnfree = true; };
in
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (ps: [
      ps.numpy
      ps.torch
      ps.transformers
      ps.sentencepiece
      ps.safetensors
      ps.protobuf
      ps.gguf
      ps.pyyaml
      ps.requests
      ps.tqdm
      ps.datasets
    ]))
    pkgs.curl
    pkgs.jq
    pkgs.coreutils
  ];
}
