#!/bin/bash
#SBATCH --job-name=rag-chunks
#SBATCH --output=sweep_logs/%A_%a.out
#SBATCH --error=sweep_logs/%A_%a.err
#SBATCH --array=0,15,30,45,60,75,90,105,120,135,150,165,180,195,210,225,240,255,270,285,300,315,330,345,360,375,390,405,420,435,450,465,480,495,510,525,540,555,570,585,600,615,630,645,660,675,690,705,720,735,750,765,780,795,810,825,840,855,870,885,900,915,930,945
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu

# Phase 1: Build all 64 unique chunk configs (chunking + embedding).
# No race conditions — each job builds a unique chunk_id.
# Run Phase 2 (sweep.slurm with all 960 configs) after this completes.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

module load llm-fast/py313/torch280

mkdir -p sweep_logs sweep_results

echo "Job array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

uv run python sweep.py --config-index "${SLURM_ARRAY_TASK_ID}"

echo "End: $(date)"
