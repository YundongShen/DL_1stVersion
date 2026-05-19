#!/bin/bash
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -A NAISS2026-3-104
#SBATCH --cpus-per-task=8
#SBATCH --time=96:00:00
#SBATCH --job-name=ebd_ablate
#SBATCH --output=logs/ablate_%j.out
#SBATCH --error=logs/ablate_%j.err

set -euo pipefail

PROJECT=/mimer/NOBACKUP/groups/llm-readability/EditBoundaryDetection
cd "$PROJECT"
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

INSTANCES=${INSTANCES:-"data/processed/instances_full.jsonl"}
ENCODER=${ENCODER:-"microsoft/unixcoder-base"}
EPOCHS=${EPOCHS:-10}

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Instances: $INSTANCES"
echo "Encoder: $ENCODER"

# ---- Variant 1: Full (all four pair types) --------------------------------
echo ""
echo "=== Ablation: full (all pair types) ==="
python train.py \
    --instances "$INSTANCES" \
    --encoder   "$ENCODER" \
    --epochs    "$EPOCHS" \
    --checkpoint-dir checkpoints/ablate_full \
    --pair-types req_test req_hunk orig_hunk req_orig

# ---- Variant 2: No ORIG ---------------------------------------------------
echo ""
echo "=== Ablation: no_orig (req_test + req_hunk only) ==="
python train.py \
    --instances "$INSTANCES" \
    --encoder   "$ENCODER" \
    --epochs    "$EPOCHS" \
    --checkpoint-dir checkpoints/ablate_no_orig \
    --pair-types req_test req_hunk

# ---- Variant 3: No TEST ---------------------------------------------------
echo ""
echo "=== Ablation: no_test (req_hunk + orig_hunk + req_orig) ==="
python train.py \
    --instances "$INSTANCES" \
    --encoder   "$ENCODER" \
    --epochs    "$EPOCHS" \
    --checkpoint-dir checkpoints/ablate_no_test \
    --pair-types req_hunk orig_hunk req_orig

# ---- Variant 4: No REQ ----------------------------------------------------
echo ""
echo "=== Ablation: no_req (orig_hunk only) ==="
python train.py \
    --instances "$INSTANCES" \
    --encoder   "$ENCODER" \
    --epochs    "$EPOCHS" \
    --checkpoint-dir checkpoints/ablate_no_req \
    --pair-types orig_hunk

echo ""
echo "All ablations done: $(date)"
