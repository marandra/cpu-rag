#!/bin/bash
#SBATCH --job-name=rag-reeval
#SBATCH --output=sweep_logs/reeval_%A_%a.out
#SBATCH --error=sweep_logs/reeval_%A_%a.err
#SBATCH --array=0-63
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=cpu

# Re-evaluate all sweep configs with chunk-level metrics.
# One job per chunk config (64 total), each evaluates all retrieval variants.
# No Qdrant lock conflicts since each job opens a unique DB.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

module load llm-fast/py313/torch280

mkdir -p sweep_logs sweep_results

echo "Chunk index: ${SLURM_ARRAY_TASK_ID}"
echo "Start: $(date)"

uv run python sweep_reeval.py --chunk-index "${SLURM_ARRAY_TASK_ID}"

echo "End: $(date)"

# After all jobs: uv run python sweep_reeval.py --aggregate
