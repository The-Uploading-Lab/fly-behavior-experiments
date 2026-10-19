#!/bin/bash
# Build the CUDA engine. The library carries the SHA-256 of cns_engine.cu
# followed by cns_engine.h; cns_metal.py hashes the same two files at load and
# refuses a mismatch, because a stale library once ran the network silent.
#
# -fmad=false matters: nvcc contracts a*b+c by default and that breaks bit
# parity with the numpy engine. The kernels also spell every parity-path
# operation as an explicit __fadd_rn/__fmul_rn, so the flag is belt and braces.
set -euo pipefail
cd "$(dirname "$0")"
# H100 has a native CUDA toolkit; Freighter's V100 compiler lives in its
# container. Keep the existing V100 target unless the caller names another.
ARCH=${CNS_CUDA_ARCH:-sm_70}
case "$ARCH" in
    sm_70|sm_90) ;;
    *) echo "Unsupported CNS_CUDA_ARCH: $ARCH (use sm_70 or sm_90)" >&2; exit 2 ;;
esac
if command -v nvcc >/dev/null 2>&1; then
    COMPILER=(nvcc)
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    COMPILER=(/usr/local/cuda/bin/nvcc)
elif command -v pytorch-exec >/dev/null 2>&1; then
    COMPILER=(pytorch-exec nvcc)
else
    echo "CUDA compiler missing: load pytorch-volta on Freighter; H100 uses /usr/local/cuda/bin/nvcc" >&2
    exit 2
fi
# Only kernel defines are accepted. Compiler flags must not silently undo
# the floating-point contract or redirect the library output.
for ARG in "$@"; do
    case "$ARG" in
        -D[A-Za-z_]*=*) ;;
        *) echo "Only -DNAME=value kernel defines are accepted: $ARG" >&2; exit 2 ;;
    esac
done
SHA=$(cat cns_engine.cu cns_engine.h | sha256sum | cut -c1-64)
"${COMPILER[@]}" -O3 -arch="$ARCH" -fmad=false -Xcompiler -fPIC -shared \
    -DCNS_ENGINE_SRC_SHA256="\"$SHA\"" \
    "$@" -o libcnsengine.so cns_engine.cu
echo "libcnsengine.so built for $ARCH, source stamp ${SHA:0:16}"
