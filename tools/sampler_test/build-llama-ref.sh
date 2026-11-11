#!/usr/bin/env bash
# build-llama-ref.sh LLAMA_CPP_PREFIX [OUT]: llama-sampler-ref against a llama.cpp package (its include/llama.h and lib/libllama.so),
# with the host's C++ compiler; run it with LD_LIBRARY_PATH=LLAMA_CPP_PREFIX/lib.
set -euo pipefail
prefix=$1
out=${2:-build/llama-sampler-ref}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$(dirname "$out")"
c++ -std=c++20 -O2 -I"$prefix/include" "$here/llama_ref.cpp" -L"$prefix/lib" -lllama -o "$out"
echo "built $out"
