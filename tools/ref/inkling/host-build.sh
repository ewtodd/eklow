#!/usr/bin/env bash
# host-build.sh [OUTDIR]: builds eklow-ref-run for SYCL's native_cpu target (Inkling's and Gemma's reference forwards on
# the host CPU, for machines without the card, such as e-workstation where the M0 ground truth is). Run from the repo
# root inside `nix develop path:.`. As tools/dflash_test/host-build.sh: this toolchain's driver does not link native_cpu's
# work-item builtins, so the libsycl-nativecpu_utils.bc that ships with it is compiled to an object and linked by hand.
set -euo pipefail
out=${1:-build/hostref}
mkdir -p "$out"
root=$(git rev-parse --show-toplevel)
bc=$(dirname "$(dirname "$(command -v clang++)")")/lib/dpcpp-7/sycl/libsycl-nativecpu_utils.bc
clang++ -c "$bc" -o "$out/nativecpu_utils.o" 2>/dev/null
clang++ -std=gnu++20 -O2 -fsycl -fsycl-targets=native_cpu -DEKLOW_HOST_NO_FP16 -I"$root/src" -Wno-deprecated-declarations \
  "$root"/src/kernels/ref.cpp "$root"/src/kernels/ref_inkling.cpp "$root"/src/model/inkling2.cpp \
  "$root"/src/model/gemma4.cpp "$root"/src/model/gguf.cpp "$root"/src/core/json_value.cpp "$root"/src/core/json.cpp \
  "$root"/tools/ref_run/main.cpp "$out/nativecpu_utils.o" -o "$out/eklow-ref-run"
echo "built $out/eklow-ref-run"
