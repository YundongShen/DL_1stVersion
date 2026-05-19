#!/bin/bash
#SBATCH --partition=alvis
#SBATCH --gpus-per-node=A40:1
#SBATCH -A NAISS2026-3-104
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --job-name=ebd_crossrepo_unt
#SBATCH --output=logs/eval_crossrepo_untrained_%j.out
#SBATCH --error=logs/eval_crossrepo_untrained_%j.err

set -euo pipefail
cd /mimer/NOBACKUP/groups/llm-readability/EditBoundaryDetection
module load Python/3.11.3-GCCcore-12.3.0
source gpu_env/bin/activate

python3 - << 'EOF'
import json, math, sys
import numpy as np
import torch
sys.path.insert(0, ".")

from collections import defaultdict
from config import default_config
from data.entailment_dataset import _render_hunk
from data.tier_labeler import label_instance_inplace
from models.entailment_encoder import EntailmentEncoder

cfg = default_config
device = torch.device("cuda")

encoder = EntailmentEncoder(
    model_name="microsoft/unixcoder-base",
    projection_dim=256, dropout=0.0, max_length=512,
).to(device)
# No checkpoint loaded — raw pretrained weights + random projection head
encoder.eval()
print("Untrained encoder ready (no checkpoint).")

tier3_lookup = {}
with open("data/cache/tier3_hunks.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        tier3_lookup[rec["instance_id"]] = rec.get("tier3_hunks", [])

instances = []
with open("data/processed/instances_full.jsonl") as f:
    for line in f:
        instances.append(json.loads(line))

def dcg(rels, k):
    return sum(r / math.log2(rank+2) for rank, r in enumerate(rels[:k]))
def ndcg(rels, k):
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(rels, k) / idcg if idcg > 0 else 0.0

tier_rel = {1: 3.0, 2: 2.0, 3: 0.0}
k_values = [5, 10, 20]

by_repo = defaultdict(list)
for inst in instances:
    repo = inst["instance_id"].split("__")[0]
    by_repo[repo].append(inst)

print("\n=== Per-repo results — UNTRAINED baseline (sorted by repo size) ===")
print(f"{'repo':<15} {'train_size':>10} {'n_eval':>6} {'nDCG@5':>8} {'T2-Recall':>10}")
print("-" * 55)
sys.stdout.flush()

for repo, repo_insts in sorted(by_repo.items(), key=lambda x: len(x[1])):
    ndcg_accum = {k: [] for k in k_values}
    t2_recall = []

    for inst in repo_insts:
        label_instance_inplace(inst)
        req = inst.get("requirement", "")[:2000]
        test_texts = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        orig_texts = [u["code"] for u in inst.get("source_units", []) if u.get("code")]

        candidates = []
        for hunk in inst.get("gold_hunks", []):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, hunk.get("tier_label") or 1))
        for hunk in tier3_lookup.get(inst.get("instance_id",""), []):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, 3))

        if len(candidates) < 2:
            continue

        hunk_texts = [c[0] for c in candidates]
        tiers = [c[1] for c in candidates]

        scores = encoder.entailment_score(
            hunk_texts=hunk_texts,
            req_texts=[req]*len(hunk_texts),
            test_texts=[test_texts]*len(hunk_texts),
            orig_texts=[orig_texts]*len(hunk_texts),
            device=device,
            alpha=cfg.eval.score_alpha,
            beta=cfg.eval.score_beta,
            gamma=cfg.eval.score_gamma,
        ).tolist()

        ranked = sorted(zip(scores, tiers), key=lambda x: x[0], reverse=True)
        ranked_rel = [tier_rel.get(t, 0.0) for _, t in ranked]
        for k in k_values:
            ndcg_accum[k].append(ndcg(ranked_rel, k))

        t2_total = sum(1 for t in tiers if t == 2)
        if t2_total > 0:
            t2_in_top = sum(1 for _, t in ranked[:t2_total] if t == 2)
            t2_recall.append(t2_in_top / t2_total)

    n_eval = len(ndcg_accum[5])
    ndcg5 = float(np.mean(ndcg_accum[5])) if ndcg_accum[5] else 0
    t2 = f"{float(np.mean(t2_recall)):.4f}" if t2_recall else "  n/a"
    print(f"{repo:<15} {len(repo_insts):>10} {n_eval:>6} {ndcg5:>8.4f} {t2:>10}")
    sys.stdout.flush()
EOF
