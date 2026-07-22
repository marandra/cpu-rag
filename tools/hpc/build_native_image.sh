#!/usr/bin/env bash
###############################################################################
# build_native_image.sh — build the SPR-native RAG image and a portable tar.
#
# Run on a BUILD HOST WITH DOCKER (laptop, CI, EC2). NOT on the HPC login node.
# Produces a docker-archive tar you scp to the cluster and convert to a SIF
# there with Apptainer (which is the only place the binary can actually RUN —
# it emits AVX512-VNNI + AMX-INT8 and will SIGILL on a non-SPR CPU).
#
# Flags mirror tools/sweep/build_venv_native.sh (the validated .venv-native):
# GGML_NATIVE=OFF + explicit ISA, NOT -march=native. On Sapphire/Emerald Rapids
# -march=native pulls in AVX512-FP16 / AVX-VNNI-VEX that older binutils can't
# assemble and that we don't need; the AVX512-VNNI (EVEX) + AMX kernels are the
# ones the sweep showed give +12-32%.
#
# Usage:
#   tools/hpc/build_native_image.sh            # build + save tar
#   TAG=cpu-rag-api:1.2.0-spr-native tools/hpc/build_native_image.sh
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

TAG="${TAG:-cpu-rag-api:1.2.0-spr-native}"
OUT_TAR="${OUT_TAR:-dist/rag-spr-native.tar}"

# ISA flags — compiler/path flags from the venv build are dropped (Docker uses
# the image's gcc). Everything else matches build_venv_native.sh verbatim.
CMAKE_FLAGS="-DGGML_NATIVE=OFF \
-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
-DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON \
-DGGML_AMX_TILE=ON -DGGML_AMX_INT8=ON \
-DGGML_OPENMP=ON -DGGML_LLAMAFILE=ON"

command -v docker >/dev/null || { echo "ERROR: docker not found on this build host." >&2; exit 1; }

echo "==> Building $TAG"
echo "    CMAKE_FLAGS: $CMAKE_FLAGS"
docker build --build-arg CMAKE_FLAGS="$CMAKE_FLAGS" -t "$TAG" .

mkdir -p "$(dirname "$OUT_TAR")"
echo "==> Saving image to $OUT_TAR"
docker save "$TAG" -o "$OUT_TAR"
ls -lh "$OUT_TAR"

cat <<EOF

==> Done. Next steps (on the HPC):
    scp $OUT_TAR hpc:~/Projects/cpu-rag/
    ssh hpc
      cd ~/Projects/cpu-rag
      module load singularity/1.4.1     # apptainer on PATH
      apptainer build rag-spr-native.sif docker-archive://$(basename "$OUT_TAR")
    # then generate snapshots once per profile, inside the SIF (PROFILE picks
    # both the procedures built and the subdir they land in — bind the root,
    # config.py appends \$P):
      P=glucowise
      mkdir -p ./snapshots
      apptainer exec --bind ./models:/app/models,./snapshots:/app/snapshots \\
        --env RAG_API_KEY=\$RAG_API_KEY --env PROFILE=\$P \\
        rag-spr-native.sif python -m app.generate
    # finally enqueue the pool:
      sbatch --export=ALL,PROFILE=\$P tools/hpc/pool.sbatch   # RAG_API_KEY comes from ./.env
EOF
