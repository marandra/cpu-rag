#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-seed
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=cpu-rag-seed-%j.log
###############################################################################
# seed_sweep.sbatch — measure the real noise floor: vary the RNG seed.
#
# Not a pool: this loads the model in-process on the native venv and generates
# directly, because the seed the service uses is frozen inside the snapshot
# pickle and cannot be overridden through the API. See tools/audit_seed_sweep.py
# for why that is and what it invalidates.
#
#   sbatch tools/hpc/seed_sweep.sbatch
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# -u or the script's progress sits in stdout's buffer until each run exits,
# while llama.cpp's own stderr chatter comes through immediately — which makes
# a healthy job look hung.
PY="./.venv-native/bin/python -u"
export N_THREADS=32 OMP_NUM_THREADS=32 GGML_N_THREADS=32

run() {  # profile procedure
    echo "===================== $2 (profile $1)"
    PROFILE="$1" $PY tools/audit_seed_sweep.py \
        --procedure "$2" --subset argued --seeds 8 --temperatures 0.1,0.8
}

run glucowise diabetes
run aiciblock cirugia-abdominal
run aiciblock hemorroides

echo "==> seed sweep done"
