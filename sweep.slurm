#!/bin/bash
#SBATCH --job-name=rag-sweep
#SBATCH --output=sweep_logs/%A_%a.out
#SBATCH --error=sweep_logs/%A_%a.err
#SBATCH --array=0-959
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=cpu

# --- Environment ---
# Adjust these paths to your HPC setup
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_DIR"

# Load modules
module load llm-fast/py313/torch280

# OpenAI API key loaded from ~/.bashrc

# --- Run single config ---
mkdir -p sweep_logs sweep_results

echo "Job array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

uv run python sweep.py --config-index "${SLURM_ARRAY_TASK_ID}"

echo "End: $(date)"

# --- Notes ---
#
# Total: 960 configurations (array 0-959)
# Unique chunk configs: 64 (chunks + embeddings are reused across retrieval variants)
# Configs needing OpenAI API: 32 (those with llm_enrichment=True)
#
# To run a subset:
#   #SBATCH --array=0-95    # first 10% only
#   #SBATCH --array=0-959%20  # max 20 jobs at a time
#
# After all jobs complete, aggregate results:
#   uv run python sweep.py --aggregate
#
# Estimated time per config:
#   - With re-chunking + embedding: ~2-5 min (depending on LLM enrichment)
#   - With reused chunks/embeddings: ~10-30 sec (evaluation only)
#
# IMPORTANT: Chunk configs are shared across retrieval variants.
# If multiple jobs try to build the same chunks simultaneously, there's a
# race condition. Two options:
#
# Option A (recommended): Run in two phases:
#   Phase 1: Build all 64 chunk configs (use a separate script or --array=0-63)
#   Phase 2: Run all 960 evaluations (chunks already built, fast)
#
# Option B: Use --array=%1 to serialize (slow but safe):
#   #SBATCH --array=0-959%1
#
# For Phase 1, you can use this helper:
#   uv run python -c "
#   from sweep import generate_configs, FIXED
#   seen = set()
#   for i, c in enumerate(generate_configs()):
#       if c.chunk_id not in seen:
#           seen.add(c.chunk_id)
#           print(i)
#   " > chunk_config_indices.txt
#   # Then: --array=$(paste -sd, chunk_config_indices.txt)
