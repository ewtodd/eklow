#!/usr/bin/env bash
# host-build.sh [OUTDIR]: builds eklow-dflash-host (host_test.cpp) with SYCL's native_cpu target over the plain-SYCL
# sources of the reference forward (the ESIMD ones do not build for it). Run from the repo root inside `nix develop path:.`.
# This toolchain's driver does not link native_cpu's work-item builtins (undefined __dpcpp_nativecpu_get_global_id), so the
# libsycl-nativecpu_utils.bc that ships with it is compiled to an object and linked in by hand.
set -euo pipefail
out=${1:-build/hosttest}
mkdir -p "$out"
root=$(git rev-parse --show-toplevel)
bc=$(dirname "$(dirname "$(command -v clang++)")")/lib/dpcpp-7/sycl/libsycl-nativecpu_utils.bc
clang++ -c "$bc" -o "$out/nativecpu_utils.o" 2>/dev/null
clang++ -std=gnu++20 -O2 -fsycl -fsycl-targets=native_cpu -DEKLOW_HOST_NO_FP16 -I"$root/src" -Wno-deprecated-declarations \
  "$root"/src/kernels/ref.cpp "$root"/src/kernels/ref_dflash.cpp "$root"/src/model/dflash2.cpp "$root"/src/model/gguf.cpp "$root"/src/core/json_value.cpp "$root"/src/core/json.cpp \
  "$root"/src/serve/spec.cpp "$root"/src/serve/sampler.cpp "$root"/tools/dflash_test/host_test.cpp "$out/nativecpu_utils.o" \
  -o "$out/eklow-dflash-host"
echo "built $out/eklow-dflash-host"
