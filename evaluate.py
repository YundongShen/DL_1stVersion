"""Evaluation entry point for Edit Entailment Learning.

Two experiments:

  Experiment 1 — Geometric Structure Validation
    Verifies that the three-tier necessity gradient exists in the learned
    embedding space.  For each test instance, we compute the Edit Entailment
    Score for every hunk (Tier 1/2 gold + Tier 3 unconstrained extras).
    Mann-Whitney U tests confirm that Tier 1 > Tier 2 > Tier 3 (p < 0.01).
    A histogram is saved to logs/geometry_hist.png.

  Experiment 2 — Retrieval (nDCG@k)
    Given a mixed candidate set (gold + unconstrained hunks), rank by
    entailment score and compute nDCG@k with tier-weighted relevance:
      Tier 1 → relevance 3,  Tier 2 → relevance 2,  Tier 3 → relevance 0.
    Also reports Tier-2-specific recall (the hardest sub-task).

Usage:
    python evaluate.py --exp geometry
    python evaluate.py --exp retrieval
    python evaluate.py --exp all
    python evaluate.py --checkpoint checkpoints/best.pt --instances data/processed/instances_lite.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

from config import Config, default_config
from data.entailment_dataset import _render_hunk, _units_for_hunk
from data.tier_labeler import label_instance_inplace
from models.entailment_encoder import EntailmentEncoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_encoder(cfg: Config, device: torch.device) -> EntailmentEncoder:
    encoder = EntailmentEncoder(
        model_name=cfg.model.encoder_name,
        projection_dim=cfg.model.projection_dim,
        dropout=0.0,
        max_length=cfg.model.max_length,
    ).to(device)

    ckpt_path = Path(cfg.eval.checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state"])
        log.info("Loaded encoder from %s", ckpt_path)
    else:
        log.warning(
            "Checkpoint %s not found — evaluating with random weights.", ckpt_path
        )

    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Tier-3 hunk loading
# ---------------------------------------------------------------------------

def load_tier3_lookup(path: str) -> dict[str, list[dict]]:
    """Load tier3_hunks.jsonl → {instance_id: [hunk_dict, ...]}."""
    result: dict[str, list[dict]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                result[rec["instance_id"]] = rec.get("tier3_hunks", [])
            except Exception:
                pass
    return result


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

def load_test_instances(cfg: Config, all_instances: bool = False) -> list[dict]:
    """Load the held-out test instances (same split logic as train.py)."""
    instances_path = Path(cfg.data.instances_lite_path)
    instances = []
    with open(instances_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                instances.append(json.loads(line))

    if all_instances:
        return instances

    import json as _json
    splits = _json.load(open("data/processed/splits.json"))
    test_ids = set(splits["test_ids"])
    return [inst for inst in instances if inst["instance_id"] in test_ids]


# ---------------------------------------------------------------------------
# Experiment 1 — Geometric Structure Validation
# ---------------------------------------------------------------------------

_TIER_COLORS = {
    "1":  "#2196F3",   # blue
    "2a": "#4CAF50",   # green  — untested, modifies existing code
    "2b": "#FF9800",   # orange — untested, new auxiliary code
    "3":  "#F44336",   # red
}
_TIER_LABELS = {
    "1":  "Tier 1 (tested gold)",
    "2a": "Tier 2a (untested, modifies existing)",
    "2b": "Tier 2b (untested, new code)",
    "3":  "Tier 3 (drift)",
}


def run_geometry(
    cfg: Config,
    encoder: EntailmentEncoder,
    device: torch.device,
    all_instances: bool = False,
    use_projection: bool = True,
    collect_umap: bool = False,
    model_label: str = "m4",
) -> None:
    log.info("=== Experiment 1: Geometric Structure Validation ===")

    instances = load_test_instances(cfg, all_instances=all_instances)
    if cfg.eval.geometry_n_instances > 0:
        instances = instances[: cfg.eval.geometry_n_instances]
    log.info("Test instances: %d", len(instances))

    tier3_lookup = load_tier3_lookup(cfg.eval.tier3_path) if cfg.eval.tier3_path else {}
    if tier3_lookup:
        log.info("Tier-3 lookup loaded: %d instances with scope-creep hunks", len(tier3_lookup))

    # Full entailment scores by tier
    tier_scores: dict[str, list[float]] = {"1": [], "2a": [], "2b": [], "3": []}
    # ORIG-component scores for the Tier-2a vs Tier-2b internal-gradient test
    orig_sim_scores: dict[str, list[float]] = {"2a": [], "2b": []}
    # Hunk embeddings for UMAP (only populated when collect_umap=True)
    embs_for_umap: dict[str, list[np.ndarray]] = {"1": [], "2a": [], "2b": [], "3": []}
    # Entity embeddings for UMAP (REQ/TEST/ORIG, one point per instance)
    entity_embs_for_umap: dict[str, list[np.ndarray]] = {"REQ": [], "TEST": [], "ORIG": []}

    for inst in instances:
        label_instance_inplace(inst)
        req          = inst.get("requirement", "")[:cfg.data.max_req_chars]
        test_texts   = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        source_units = inst.get("source_units", [])

        # Carry base_tier explicitly: tier3_lookup hunks have no tier_label field
        all_hunks: list[tuple[dict, int]] = [
            (h, h.get("tier_label") or 1) for h in inst.get("gold_hunks", [])
        ]
        for h in tier3_lookup.get(inst.get("instance_id", ""), []):
            all_hunks.append((h, 3))

        for hunk, base_tier in all_hunks:
            hunk_text = _render_hunk(hunk)
            if not hunk_text.strip():
                continue

            relevant_units  = _units_for_hunk(hunk, source_units)
            hunk_orig_texts = [u["code"] for u in relevant_units if u.get("code")]

            # Tier-2 → 2a (has relevant ORIG units) or 2b (no ORIG match → new code)
            if base_tier == 2:
                eff_tier = "2a" if hunk_orig_texts else "2b"
            else:
                eff_tier = str(base_tier)

            score = encoder.entailment_score(
                hunk_texts=[hunk_text],
                req_texts=[req],
                test_texts=[test_texts],
                orig_texts=[hunk_orig_texts],
                device=device,
                alpha=cfg.eval.score_alpha,
                beta=cfg.eval.score_beta,
                gamma=cfg.eval.score_gamma,
                use_projection=use_projection,
            ).item()
            tier_scores[eff_tier].append(score)

            # Hunk embedding (shared across UMAP + ORIG-component computation)
            h_emb = encoder.encode([hunk_text], "HUNK", device, use_projection)  # (1, D)

            if collect_umap:
                embs_for_umap[eff_tier].append(h_emb.cpu().numpy()[0])

            # ORIG-component similarity: γ·sim(hunk, mean(relevant_ORIG))
            # Used to confirm the Tier-2a >> Tier-2b internal gradient.
            if base_tier == 2:
                if hunk_orig_texts:  # 2a: has relevant ORIG units
                    o_embs = encoder.encode(hunk_orig_texts, "ORIG", device, use_projection)
                    o_norm = F.normalize(o_embs.mean(dim=0, keepdim=True), dim=-1)
                    orig_sim = (h_emb * o_norm).sum(-1).item()
                else:               # 2b: no ORIG match → similarity is 0 by construction
                    orig_sim = 0.0
                orig_sim_scores[eff_tier].append(orig_sim)

        # Collect instance-level entity embeddings for UMAP (REQ/TEST/ORIG anchors)
        if collect_umap:
            r_emb = encoder.encode([req], "REQ", device, use_projection)
            entity_embs_for_umap["REQ"].append(r_emb.cpu().numpy()[0])
            if test_texts:
                t_embs = encoder.encode(test_texts, "TEST", device, use_projection)
                entity_embs_for_umap["TEST"].append(t_embs.mean(dim=0).cpu().numpy())
            orig_codes = [u["code"] for u in source_units[:10] if u.get("code")]
            if orig_codes:
                o_embs = encoder.encode(orig_codes, "ORIG", device, use_projection)
                entity_embs_for_umap["ORIG"].append(o_embs.mean(dim=0).cpu().numpy())

    # --- Summary statistics ---
    for key in ("1", "2a", "2b", "3"):
        scores = tier_scores[key]
        if scores:
            log.info(
                "%s  n=%d  mean=%.4f  std=%.4f  median=%.4f",
                key, len(scores), np.mean(scores), np.std(scores), np.median(scores),
            )
        else:
            log.info("%s  n=0  (no samples)", key)

    # --- Mann-Whitney U: full entailment score ---
    for ta, tb in [("1", "2a"), ("1", "2b"), ("2a", "2b"), ("2a", "3"), ("2b", "3"), ("1", "3")]:
        sa, sb = tier_scores[ta], tier_scores[tb]
        if len(sa) >= 2 and len(sb) >= 2:
            stat, p = mannwhitneyu(sa, sb, alternative="greater")
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            log.info(
                "Mann-Whitney U [score]  %s > %s:  U=%.0f  p=%.4e  %s",
                ta, tb, stat, p, sig,
            )

    # --- Mann-Whitney U: ORIG-component only (Tier-2 internal gradient) ---
    sa2a, sa2b = orig_sim_scores["2a"], orig_sim_scores["2b"]
    if len(sa2a) >= 2 and len(sa2b) >= 2:
        stat, p = mannwhitneyu(sa2a, sa2b, alternative="greater")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        log.info(
            "Mann-Whitney U [ORIG-sim]  2a > 2b:  U=%.0f  p=%.4e  %s  "
            "(n_2a=%d mean=%.4f | n_2b=%d mean=%.4f)",
            stat, p, sig,
            len(sa2a), np.mean(sa2a),
            len(sa2b), np.mean(sa2b),
        )
    else:
        log.info("Mann-Whitney U [ORIG-sim]  2a > 2b: skipped (n_2a=%d, n_2b=%d)",
                 len(sa2a), len(sa2b))

    _maybe_histogram(tier_scores)
    if collect_umap:
        _maybe_umap(embs_for_umap, entity_embs_for_umap, model_label=model_label)


def _maybe_histogram(tier_scores: dict[str, list[float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        log.info("matplotlib not available — skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    for key in ("1", "2a", "2b", "3"):
        scores = tier_scores.get(key, [])
        if scores:
            ax.hist(scores, bins=30, alpha=0.6, color=_TIER_COLORS[key], label=_TIER_LABELS[key])
    ax.set_xlabel("Edit Entailment Score")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by tier")
    ax.legend()
    out = Path("logs/geometry_hist.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Histogram saved → %s", out)
    plt.close(fig)


def _maybe_umap(
    embs_for_umap: dict[str, list[np.ndarray]],
    entity_embs: dict[str, list[np.ndarray]],
    model_label: str = "m4",
) -> None:
    try:
        import umap as umap_lib          # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
        from matplotlib.cm import ScalarMappable  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401
    except ImportError:
        log.info("umap-learn or matplotlib not available — skipping UMAP.")
        return

    # --- Collect hunk embeddings (256D) for silhouette + UMAP ---
    hunk_embs_256: list[np.ndarray] = []
    hunk_keys:     list[str] = []
    for key in ("1", "2a", "2b", "3"):
        for emb in embs_for_umap.get(key, []):
            hunk_embs_256.append(emb)
            hunk_keys.append(key)

    if len(hunk_embs_256) < 10:
        log.info("Too few hunk embeddings for UMAP (%d) — skipping.", len(hunk_embs_256))
        return

    # --- Silhouette score in original 256D space (cosine) ---
    sil: float | None = None
    try:
        from sklearn.metrics import silhouette_score  # type: ignore
        sil = silhouette_score(np.array(hunk_embs_256), hunk_keys, metric="cosine")
        log.info("Silhouette score (256D cosine, hunk tiers): %.4f", sil)
    except Exception as e:
        log.info("Silhouette score skipped: %s", e)

    # --- Build combined array: hunk + entity embeddings ---
    # Entity anchors: REQ (★), TEST (▲), ORIG (■) — one point per instance
    entity_cfg = {
        "REQ":  ("*", "#9C27B0", "REQ",  100),
        "TEST": ("^", "#00BCD4", "TEST", 60),
        "ORIG": ("s", "#795548", "ORIG", 60),
    }
    all_embs: list[np.ndarray] = list(hunk_embs_256)
    all_tags: list[str] = list(hunk_keys)
    for etype in ("REQ", "TEST", "ORIG"):
        for emb in entity_embs.get(etype, []):
            all_embs.append(emb)
            all_tags.append(etype)

    n_hunks = len(hunk_embs_256)
    log.info("Running 3D UMAP on %d points (%d hunks + %d entity anchors)...",
             len(all_embs), n_hunks, len(all_embs) - n_hunks)

    reducer = umap_lib.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(np.array(all_embs))  # (N, 3)

    # --- Gradient colormap: deep blue (T1, most necessary) → red (T3, scope creep) ---
    tier_cmap = mcolors.LinearSegmentedColormap.from_list(
        "tier_necessity", ["#2196F3", "#4CAF50", "#FF9800", "#F44336"], N=256,
    )
    tier_cval = {"1": 0.0, "2a": 1 / 3, "2b": 2 / 3, "3": 1.0}

    # --- Save raw coords for optional M0 vs M4 paired figure ---
    out_dir = Path("logs")
    out_dir.mkdir(exist_ok=True)
    np.savez(
        out_dir / f"umap_embs_{model_label}.npz",
        coords=coords, tags=np.array(all_tags), n_hunks=n_hunks,
        sil=np.array([sil if sil is not None else float("nan")]),
    )
    log.info("UMAP embeddings saved → logs/umap_embs_%s.npz", model_label)

    # --- Draw single-model 3D figure ---
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")
    _draw_umap_ax(ax, coords, all_tags, n_hunks, entity_cfg, tier_cmap, tier_cval, sil,
                  title=f"4-Entity Embedding Space — {model_label.upper()} (UMAP-3D)")
    sm = ScalarMappable(cmap=tier_cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.12, aspect=20)
    cbar.set_ticks([0.0, 1 / 3, 2 / 3, 1.0])
    cbar.set_ticklabels(
        ["T1 (tested gold)", "T2a (untested, modifies)", "T2b (untested, new)", "T3 (scope creep)"],
        fontsize=7,
    )
    cbar.set_label("HUNK necessity →", fontsize=8)

    out = out_dir / f"umap_3d_{model_label}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("3D UMAP saved → %s", out)
    plt.close(fig)

    # --- Attempt paired M0/M4 comparison if both npz files exist ---
    _maybe_paired_umap(out_dir, entity_cfg, tier_cmap, tier_cval)


def _draw_umap_ax(
    ax,
    coords: np.ndarray,
    all_tags: list[str],
    n_hunks: int,
    entity_cfg: dict,
    tier_cmap,
    tier_cval: dict[str, float],
    sil: float | None,
    title: str = "",
) -> None:
    z_min = float(coords[:, 2].min()) - 0.5

    # HUNK points: gradient color by tier necessity
    for key in ("1", "2a", "2b", "3"):
        idx = [i for i, k in enumerate(all_tags[:n_hunks]) if k == key]
        if not idx:
            continue
        cx, cy, cz = coords[idx, 0], coords[idx, 1], coords[idx, 2]
        color = tier_cmap(tier_cval[key])
        ax.scatter(cx, cy, cz,
                   c=[color] * len(idx), marker="o", label=_TIER_LABELS[key],
                   alpha=0.7, s=12, depthshade=True, linewidths=0)
        ax.scatter(cx, cy, zs=z_min, zdir="z",
                   c=[color] * len(idx), marker="o", alpha=0.08, s=6, linewidths=0)

    # Entity anchors: REQ (★), TEST (▲), ORIG (■) — distinct marker per type
    for etype, (marker, color, label, sz) in entity_cfg.items():
        idx = [i for i, k in enumerate(all_tags) if k == etype]
        if not idx:
            continue
        cx, cy, cz = coords[idx, 0], coords[idx, 1], coords[idx, 2]
        ax.scatter(cx, cy, cz,
                   c=color, marker=marker, label=label,
                   alpha=0.85, s=sz, edgecolors="black", linewidths=0.4, depthshade=False)
        ax.scatter(cx, cy, zs=z_min, zdir="z",
                   c=color, marker=marker, alpha=0.15, s=sz // 2, linewidths=0)

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.set_title(title)
    ax.legend(markerscale=2, loc="upper left", fontsize=8)

    # Silhouette score annotated beside the plot (in axes-fraction coordinates)
    if sil is not None:
        ax.text2D(0.01, 0.97, f"Sil = {sil:.3f}", transform=ax.transAxes,
                  fontsize=9, va="top",
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))


def _maybe_paired_umap(
    out_dir: Path,
    entity_cfg: dict,
    tier_cmap,
    tier_cval: dict[str, float],
) -> None:
    m0_path = out_dir / "umap_embs_m0.npz"
    m4_path = out_dir / "umap_embs_m4.npz"
    if not (m0_path.exists() and m4_path.exists()):
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.colors as mcolors  # type: ignore
        from matplotlib.cm import ScalarMappable  # type: ignore
        from mpl_toolkits.mplot3d import Axes3D  # type: ignore  # noqa: F401
    except ImportError:
        return

    panels = []
    for panel_label, path in [("M0 (untrained UniXCoder)", m0_path), ("M4 (trained)", m4_path)]:
        d = np.load(path, allow_pickle=True)
        sil_val = float(d["sil"][0])
        panels.append((
            panel_label,
            d["coords"],
            list(d["tags"]),
            int(d["n_hunks"]),
            None if np.isnan(sil_val) else sil_val,
        ))

    fig = plt.figure(figsize=(22, 9))
    axes = []
    for col, (panel_label, coords, tags, n_hunks, sil) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        _draw_umap_ax(ax, coords, tags, n_hunks, entity_cfg, tier_cmap, tier_cval, sil,
                      title=panel_label)
        axes.append(ax)

    sm = ScalarMappable(cmap=tier_cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.45, pad=0.04, aspect=25)
    cbar.set_ticks([0.0, 1 / 3, 2 / 3, 1.0])
    cbar.set_ticklabels(
        ["T1 (tested gold)", "T2a (untested, modifies)", "T2b (untested, new)", "T3 (scope creep)"],
        fontsize=8,
    )
    cbar.set_label("HUNK necessity →", fontsize=9)
    fig.suptitle("4-Entity Embedding Space: M0 vs M4 (UMAP-3D)", fontsize=13)

    out = out_dir / "umap_3d_paired.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Paired M0/M4 UMAP saved → %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# nDCG helpers
# ---------------------------------------------------------------------------

def _dcg(relevances: list[float], k: int) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg  = _dcg(ideal, k)
    return _dcg(relevances, k) / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Experiment 2 — Retrieval (nDCG@k)
# ---------------------------------------------------------------------------

def run_retrieval(cfg: Config, encoder: EntailmentEncoder, device: torch.device, all_instances: bool = False, use_projection: bool = True) -> None:
    log.info("=== Experiment 2: Retrieval (nDCG@k) ===")

    instances = load_test_instances(cfg, all_instances=all_instances)
    log.info("Test instances: %d", len(instances))

    tier3_lookup = load_tier3_lookup(cfg.eval.tier3_path) if cfg.eval.tier3_path else {}
    if tier3_lookup:
        log.info("Tier-3 lookup loaded: %d instances with scope-creep hunks", len(tier3_lookup))

    tier_rel = cfg.eval.tier_relevance
    k_values = list(cfg.eval.retrieval_k_values)

    ndcg_accum: dict[int, list[float]] = {k: [] for k in k_values}
    tier2_recall: list[float] = []

    for inst in instances:
        label_instance_inplace(inst)
        req          = inst.get("requirement", "")[:cfg.data.max_req_chars]
        test_texts   = [tf["code"] for tf in inst.get("test_functions", []) if tf.get("code")]
        source_units = inst.get("source_units", [])

        # Track hunk dicts so we can apply _units_for_hunk per candidate (Issue 2 fix)
        candidates: list[tuple[str, int, dict]] = []  # (text, tier, hunk_dict)
        for hunk in inst.get("gold_hunks", []):
            text = _render_hunk(hunk)
            if text.strip():
                tier = hunk.get("tier_label") or 1
                candidates.append((text, tier, hunk))
        for hunk in tier3_lookup.get(inst.get("instance_id", ""), []):
            text = _render_hunk(hunk)
            if text.strip():
                candidates.append((text, 3, hunk))

        if len(candidates) < 2:
            continue

        hunk_texts = [c[0] for c in candidates]
        tiers      = [c[1] for c in candidates]
        # Per-hunk ORIG: only units that the specific hunk modifies, not all source_units
        hunk_orig_texts = [
            [u["code"] for u in _units_for_hunk(c[2], source_units) if u.get("code")]
            for c in candidates
        ]

        scores = encoder.entailment_score(
            hunk_texts=hunk_texts,
            req_texts=[req] * len(hunk_texts),
            test_texts=[test_texts] * len(hunk_texts),
            orig_texts=hunk_orig_texts,
            device=device,
            alpha=cfg.eval.score_alpha,
            beta=cfg.eval.score_beta,
            gamma=cfg.eval.score_gamma,
            use_projection=use_projection,
        ).tolist()

        ranked     = sorted(zip(scores, tiers), key=lambda x: x[0], reverse=True)
        ranked_rel = [tier_rel.get(t, 0.0) for _, t in ranked]

        # nDCG only on instances that have T3 candidates (scope-creep detection task)
        has_tier3 = any(t == 3 for t in tiers)
        if has_tier3:
            for k in k_values:
                ndcg_accum[k].append(_ndcg(ranked_rel, k))

        tier2_total = sum(1 for t in tiers if t == 2)
        if tier2_total > 0:
            tier2_in_top = sum(1 for _, t in ranked[:tier2_total] if t == 2)
            tier2_recall.append(tier2_in_top / tier2_total)

    for k in k_values:
        vals = ndcg_accum[k]
        if vals:
            log.info("nDCG@%2d  mean=%.4f  std=%.4f  n=%d", k, np.mean(vals), np.std(vals), len(vals))

    if tier2_recall:
        log.info(
            "Tier-2 Recall  mean=%.4f  std=%.4f  n=%d",
            np.mean(tier2_recall), np.std(tier2_recall), len(tier2_recall),
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    cfg: Config,
    exp: str,
    all_instances: bool = False,
    use_projection: bool = True,
    collect_umap: bool = False,
) -> None:
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    log.info("Device: %s", device)
    log.info("use_projection: %s", use_projection)

    encoder = load_encoder(cfg, device)

    # Derive a short label for UMAP file naming (m0 for untrained baseline).
    if not use_projection:
        model_label = "m0"
    else:
        ckpt = cfg.eval.checkpoint_path
        p = Path(ckpt)
        parent = p.parent.name
        # "checkpoints/m4/best.pt" → "m4"; legacy "checkpoints/best.pt" → "trained"
        model_label = parent if parent != "checkpoints" else "trained"

    if exp in ("geometry", "all"):
        run_geometry(
            cfg, encoder, device,
            all_instances=all_instances,
            use_projection=use_projection,
            collect_umap=collect_umap,
            model_label=model_label,
        )

    if exp in ("retrieval", "all"):
        run_retrieval(cfg, encoder, device, all_instances=all_instances, use_projection=use_projection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Edit Entailment encoder")
    parser.add_argument(
        "--exp", choices=["geometry", "retrieval", "all"],
        default="all", help="Which experiment to run",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument("--instances", default=None, help="Instances JSONL path")
    parser.add_argument("--encoder", default=None, help="HuggingFace model name")
    parser.add_argument("--tier3", default=None, help="Path to tier3_hunks.jsonl (scope-creep candidates)")
    parser.add_argument("--all-instances", action="store_true", help="Evaluate on all instances, not just test split")
    parser.add_argument("--no-projection", action="store_true", help="Skip MLP projection head (M0 untrained baseline)")
    parser.add_argument("--umap", action="store_true", help="Collect hunk embeddings and save UMAP plot (requires umap-learn)")
    parser.add_argument("--score-alpha", type=float, default=None, help="Override score_alpha")
    parser.add_argument("--score-beta",  type=float, default=None, help="Override score_beta")
    parser.add_argument("--score-gamma", type=float, default=None, help="Override score_gamma")
    args = parser.parse_args()

    cfg = default_config
    if args.checkpoint:
        cfg.eval.checkpoint_path = args.checkpoint
    if args.instances:
        cfg.data.instances_lite_path = args.instances
    if args.encoder:
        cfg.model.encoder_name = args.encoder
    if args.tier3:
        cfg.eval.tier3_path = args.tier3
    if args.score_alpha is not None:
        cfg.eval.score_alpha = args.score_alpha
    if args.score_beta is not None:
        cfg.eval.score_beta = args.score_beta
    if args.score_gamma is not None:
        cfg.eval.score_gamma = args.score_gamma

    use_projection = not args.no_projection
    main(
        cfg, args.exp,
        all_instances=args.all_instances,
        use_projection=use_projection,
        collect_umap=args.umap,
    )
