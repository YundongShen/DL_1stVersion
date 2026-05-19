"""Data pipeline for Edit Entailment Learning."""

from .data_loader import DataLoader, DataSample, GitHubDataLoader
from .entailment_dataset import EntailmentDataset, EntailmentPair
from .tier_labeler import label_tiers, label_instance_inplace
from .utils import hash_signature, normalize_diff, tokenize_diff_hunks

__all__ = [
    # SWE-bench loading (kept for LLM pair generation pipeline)
    "DataLoader",
    "DataSample",
    "GitHubDataLoader",
    # New: Edit Entailment
    "EntailmentDataset",
    "EntailmentPair",
    "label_tiers",
    "label_instance_inplace",
    # Utilities
    "hash_signature",
    "normalize_diff",
    "tokenize_diff_hunks",
]
