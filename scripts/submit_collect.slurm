#!/bin/bash
#SBATCH --partition=berzelius-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=ebd_collect
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err

# -----------------------------------------------------------------------
# Collect TrainingPair data from SWE-bench instances.
#
# Requires a running vLLM server (start_vllm.slurm).
# The vLLM node is read from logs/vllm_node.txt written by that job.
#
# Usage:
#   sbatch --dependency=after:<vllm_jobid> scripts/submit_collect.slurm
#
# Or set VLLM_NODE manually:
#   VLLM_NODE=node123 sbatch scripts/submit_collect.slurm
#
# MODE: training or zeroshot (default: training)
# -----------------------------------------------------------------------

set -euo pipefail

PROJECT=/proj/test_extender/ICLR/EditBoundaryDetection
cd "$PROJECT"
source gpu_env/bin/activate

mkdir -p logs

MODE=${MODE:-training}
VLLM_NODE=${VLLM_NODE:-$(cat logs/vllm_node.txt 2>/dev/null || echo "localhost")}
VLLM_URL="http://${VLLM_NODE}:8000/v1"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Mode: $MODE"
echo "vLLM server: $VLLM_URL"

python scripts/collect_pairs.py \
    --mode "$MODE" \
    --input  data/raw/swebench_instances.jsonl \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --timeout 120 \
    --filter-prefix "" \
    --repos-dir "${TMPDIR}/repos" \
    --llm-provider openai \
    --llm-api-base "$VLLM_URL" \
    --llm-model Qwen2.5-Coder-32B-Instruct \
    --llm-max-tokens 4096

echo "Job done: $(date)"
OUTFILE="data/cache/${MODE}_pairs.jsonl"
echo "Pairs written: $(wc -l < "$OUTFILE" 2>/dev/null || echo 0)"
