#!/usr/bin/env bash
# Build .venv-native on HPC with spack toolchain (gcc@12.5.0 + binutils@2.45)
# and llama-cpp-python compiled with -DGGML_NATIVE=ON so AVX512-VNNI / AMX
# kernels actually emit vpdpbusd / tdpbssd instructions.
#
# Run on a SPR compute node (NOT login) so cmake's native detection sees
# the same ISA the sweep will run on. Example:
#   srun -p compute -N1 -n1 -c8 --pty bash
#   bash tools/sweep/build_venv_native.sh
#
# Reuses the existing .venv-avx512 only for Python/uv discovery — the new
# venv is fully isolated under .venv-native/.

set -euo pipefail

PROJ="${PROJ:-$HOME/Projects/cpu-rag}"
VENV="$PROJ/.venv-native"
cd "$PROJ"

echo "==> Loading spack toolchain"
module load spack/1.1.0

# Find the SPR-targeted installs (gcc 12.5.0 + binutils 2.45 sapphirerapids variant).
GCC_PREFIX=$(spack location -i gcc@12.5.0)
# binutils 2.45 has two variants; pick the sapphirerapids one (hash gf5dhlr in current install).
BU_PREFIX=$(spack location -i /gf5dhlr)
echo "    gcc      = $GCC_PREFIX"
echo "    binutils = $BU_PREFIX"

export PATH="$GCC_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$GCC_PREFIX/lib64:${LD_LIBRARY_PATH:-}"
export CC="$GCC_PREFIX/bin/gcc"
export CXX="$GCC_PREFIX/bin/g++"

echo "==> Toolchain check"
gcc --version | head -1
as  --version | head -1
echo "    gcc=12.5.0 (spack), as=2.35.2 (system) -- fine for SPR:"
echo "    AVX512-VNNI (EVEX) and AMX already assemble with 2.35.2."
echo "    Only AVX-VNNI VEX (client variant) is rejected -- not needed on SPR."

# Sanity: confirm AVX512-VNNI (the encoding we actually want) assembles.
if ! echo 'vpdpbusd %zmm0,%zmm1,%zmm2' | as -o /tmp/_vnnitest.o - 2>/dev/null; then
  echo "!! AVX512-VNNI not assembling -- unexpected"; exit 1
fi
if ! echo 'tdpbssd %tmm0,%tmm1,%tmm2' | as -o /tmp/_amxtest.o - 2>/dev/null; then
  echo "!! AMX-INT8 not assembling -- unexpected"; exit 1
fi
rm -f /tmp/_vnnitest.o /tmp/_amxtest.o
echo "    AVX512-VNNI + AMX assemble OK"

echo "==> Creating $VENV"
rm -rf "$VENV"
uv venv --python 3.13 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install -U pip wheel scikit-build-core cmake ninja

echo "==> Installing project deps (without llama-cpp-python)"
# Pull everything else from pyproject so the venv is usable for the server.
# We install llama-cpp-python separately to control CMAKE_ARGS.
uv pip install -e . --no-deps
uv pip install \
  "fastapi>=0.111.0" "uvicorn[standard]" "pydantic>=2" "pydantic-settings>=2" \
  "httpx" "sse-starlette"

echo "==> Building llama-cpp-python with explicit ISA flags"
# IMPORTANT: do NOT use GGML_NATIVE / -march=native here. On Emerald/Sapphire
# Rapids that pulls in AVX512-FP16 (vaddph/vmovsh/vmovw) and AVX-VNNI (VEX
# vpdpbusd), neither of which the system 'as' 2.35.2 can assemble. The RHEL
# 2.35.2 DOES have AMX and AVX512-VNNI (EVEX) backported, so we enable exactly
# those features explicitly -- the kernels we actually want to benchmark.
export CMAKE_ARGS="\
-DGGML_NATIVE=OFF \
-DGGML_AVX=ON \
-DGGML_AVX2=ON \
-DGGML_FMA=ON \
-DGGML_F16C=ON \
-DGGML_AVX512=ON \
-DGGML_AVX512_VBMI=ON \
-DGGML_AVX512_VNNI=ON \
-DGGML_AMX_TILE=ON \
-DGGML_AMX_INT8=ON \
-DGGML_OPENMP=ON \
-DGGML_LLAMAFILE=ON \
-DCMAKE_C_COMPILER=$CC \
-DCMAKE_CXX_COMPILER=$CXX"
export FORCE_CMAKE=1

uv pip install --no-binary=llama-cpp-python --no-cache "llama-cpp-python==0.3.19" 2>&1 | tail -20

echo "==> Verifying build"
python - <<'PY'
import os, glob, subprocess, sys
import llama_cpp
libdir = os.path.dirname(llama_cpp.__file__) + "/lib"
so = glob.glob(libdir + "/libggml-cpu.so")[0]
out = subprocess.check_output(["objdump", "-d", so], text=True, stderr=subprocess.DEVNULL)
counts = {
    "vpdpbusd (AVX512-VNNI)": out.count("vpdpbusd"),
    "tdpbssd (AMX i8)": out.count("tdpbssd"),
    "tileloadd (AMX tile)": out.count("tileloadd"),
}
for k, v in counts.items():
    flag = "OK" if v > 0 else "MISSING"
    print(f"  {k:30s} {v:6d}  [{flag}]")
if counts["vpdpbusd (AVX512-VNNI)"] == 0:
    print("!! VNNI kernel not emitted; build did not pick up flags"); sys.exit(2)

from llama_cpp import llama_print_system_info
info = llama_print_system_info()
info = info.decode() if hasattr(info, "decode") else info
print("system_info:", info)
PY

echo "==> .venv-native ready at $VENV"
