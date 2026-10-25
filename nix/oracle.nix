# eklow-oracle, built with llama.cpp's own nixpkgs so the process has one C++ runtime.
{
  stdenv,
  llama-cpp,
}:

stdenv.mkDerivation {
  pname = "eklow-oracle";
  version = "0.0.1";
  src = ../tools/ref/oracle;

  buildInputs = [ llama-cpp ];

  buildPhase = ''
    runHook preBuild
    $CXX -O2 -std=c++17 -Wall -Wextra oracle.cpp -o eklow-oracle -lllama -lggml -lggml-base
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    install -Dm755 eklow-oracle $out/bin/eklow-oracle
    runHook postInstall
  '';
}
