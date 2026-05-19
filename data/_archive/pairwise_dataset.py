"""PyTorch Dataset wrapper for TrainingPair objects.

Loads from JSONL files produced by ``scripts/collect_pairs.py``.
"""

from __future__ import annotations

import dataclasses
import json
import random

from torch.utils.data import Dataset

from .pair_types import TrainingPair

# Re-export for convenience and backward compatibility.
from .llm_client import LLMClient  # noqa: F401
from .patch_generator import PatchGenerator  # noqa: F401


class PairwiseDataset(Dataset):
    """PyTorch Dataset wrapping a list of :class:`~data.pair_types.TrainingPair` objects."""

    def __init__(self, pairs: list[TrainingPair]) -> None:
        self._data = pairs

    @classmethod
    def from_jsonl(cls, path: str) -> "PairwiseDataset":
        """Load a dataset from a JSONL file."""
        pairs: list[TrainingPair] = []
        with open(path) as fh:
            for line in fh:
                row = json.loads(line)
                row.pop("sample_id", None)  # attached for resume tracking, not a field
                pairs.append(TrainingPair(**row))
        return cls(pairs)

    def save_jsonl(self, path: str) -> None:
        """Persist pairs to a JSONL file."""
        with open(path, "w") as fh:
            for p in self._data:
                fh.write(json.dumps(dataclasses.asdict(p)) + "\n")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> TrainingPair:
        return self._data[idx]

    def split(
        self, train: float = 0.8, val: float = 0.1, seed: int = 42
    ) -> tuple["PairwiseDataset", "PairwiseDataset", "PairwiseDataset"]:
        """Return (train, val, test) splits."""
        rng = random.Random(seed)
        data = list(self._data)
        rng.shuffle(data)
        n = len(data)
        n_train = int(n * train)
        n_val = int(n * val)
        return (
            PairwiseDataset(data[:n_train]),
            PairwiseDataset(data[n_train : n_train + n_val]),
            PairwiseDataset(data[n_train + n_val :]),
        )

    def filter_by_source(self, positive_source: str) -> "PairwiseDataset":
        """Return subset with the given positive_source (e.g. 'gold' or 'controlled_llm')."""
        return PairwiseDataset([p for p in self._data if p.positive_source == positive_source])
