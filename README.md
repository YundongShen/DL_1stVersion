# EditBoundaryDetection

A PyTorch-based framework for detecting **Scope Creep** in LLM-generated code edits — identifying when a model touches code outside the boundary defined by an issue and its associated test suite.

---

## Overview

The core hypothesis: an edit that stays within the intended boundary should have high cosine similarity with a boundary representation built from the issue text and test signatures. An out-of-boundary edit (scope creep) will diverge.

The model is trained with **InfoNCE contrastive loss** in a dual-tower architecture:

```
Issue + Test Signature  →  BoundaryEncoder  →  anchor embedding
Old Code + Diff         →  EditEncoder      →  edit embedding

InfoNCE: anchor should be close to in-boundary edits, far from out-of-boundary edits
```

Pairs are constructed by prompting an LLM with two different system prompts (controlled/minimal-change vs. unconstrained) on the same issue, then filtering using execution traces to confirm boundary status.

---

## Installation

```bash
pip install -r requirements.txt
```

**Optional GNN support** (for AST-level diff encoding):
```bash
# Install one of:
pip install torch-geometric
pip install dgl
```

Set your OpenAI API key for real pair generation:
```bash
export OPENAI_API_KEY=sk-...
```

Set a GitHub token for `GitHubDataLoader`:
```bash
export GITHUB_TOKEN=ghp_...
```

---

## Data Preparation

### Option 1: JSON Lines file

Create a `.jsonl` file where each line is:
```json
{
  "id": "sample_001",
  "issue": "Fix off-by-one error in pagination",
  "old_files": {"src/pagination.py": "def paginate(items, page):\n    ..."},
  "diff": "--- a/src/pagination.py\n+++ b/src/pagination.py\n...",
  "tests": {"tests/test_pagination.py": "def test_paginate(): ..."},
  "meta": {}
}
```

Load it:
```python
from data.data_loader import JSONDataLoader
loader = JSONDataLoader("path/to/data.jsonl")
```

### Option 2: GitHub Issue-PR pairs

```python
from data.data_loader import GitHubDataLoader
loader = GitHubDataLoader("owner/repo", max_samples=200)
```

### Option 3: Mock data (for testing)

```python
from data.data_loader import MockDataLoader
loader = MockDataLoader(n=50)
```

---

## Generating Paired Data

```python
from data.data_loader import MockDataLoader
from data.pairwise_dataset import LLMClient, PairwiseDataset

loader = MockDataLoader(n=100)
llm = LLMClient(provider="openai", model="gpt-4o")  # or provider="mock"

dataset = PairwiseDataset.from_loader(loader, llm_client=llm, seed=42)
dataset.save_jsonl("data/cache/quintuples.jsonl")
print(f"Generated {len(dataset)} quintuples")
```

Each `Quintuple` contains:
- `issue_text`: the bug/feature description
- `test_suite_signature`: sorted test function names (boundary definition)
- `old_code`: the codebase before the patch
- `in_boundary_diff`: minimal-change patch (positive example)
- `out_boundary_diff`: unconstrained patch with scope creep (negative example)

---

## Training

```bash
python train.py
```

The script will:
1. Look for `data/cache/quintuples.jsonl`. If absent, generates mock pairs automatically.
2. Train with AdamW + linear warmup + cosine decay.
3. Save checkpoints to `checkpoints/epoch_NNN.pt` and `checkpoints/best.pt`.

Key config fields (edit `config.py`):
```python
TrainConfig(
    batch_size=16,
    lr=2e-5,
    epochs=20,
    warmup_steps=200,
    temperature=0.07,   # InfoNCE τ
    checkpoint_dir="checkpoints",
)
```

---

## Evaluation

```bash
python evaluate.py
```

Three protocols are run on the test split:

| Protocol | What it measures |
|----------|-----------------|
| **Linear Probe** | Are edit embeddings linearly separable? (5-fold CV, LogReg) |
| **Boundary Permutation** | Does swapping test signatures degrade similarity? |
| **Randomized Boundary** | Is true similarity higher than randomly shuffled pairs? |

To evaluate a specific checkpoint:
```python
from config import Config, EvalConfig
cfg = Config(eval=EvalConfig(checkpoint_path="checkpoints/best.pt"))

from evaluate import main
main(cfg)
```

---

## Config Reference

All settings live in `config.py` as dataclasses. Key fields:

| Class | Field | Default | Description |
|-------|-------|---------|-------------|
| `ModelConfig` | `issue_encoder_name` | `"roberta-base"` | HuggingFace model for boundary encoder |
| `ModelConfig` | `code_encoder_name` | `"microsoft/codebert-base"` | HuggingFace model for edit encoder |
| `ModelConfig` | `projection_dim` | `256` | Output embedding dimension |
| `ModelConfig` | `use_gnn` | `False` | Enable AST-diff GNN branch |
| `TrainConfig` | `temperature` | `0.07` | InfoNCE softmax temperature |
| `TrainConfig` | `batch_size` | `16` | Training batch size |
| `LLMConfig` | `provider` | `"openai"` | `"openai"` or `"mock"` |
| `EvalConfig` | `n_permutations` | `1000` | Shuffles for randomized boundary test |

---

## End-to-End Example

```python
from config import default_config
from data.data_loader import MockDataLoader
from data.pairwise_dataset import LLMClient, PairwiseDataset
from models.contrastive_model import ContrastiveModel
import torch

# 1. Generate data
loader = MockDataLoader(n=20)
llm = LLMClient(provider="mock")
dataset = PairwiseDataset.from_loader(loader, llm_client=llm)

# 2. Build model
cfg = default_config
model = ContrastiveModel.from_config(cfg)
model.eval()

# 3. Score a single sample
q = dataset[0]
sim = model.similarity(
    issue_texts=[q.issue_text],
    test_signatures=[q.test_suite_signature],
    old_codes=[q.old_code],
    diffs=[q.in_boundary_diff],
)
print(f"In-boundary similarity : {sim.item():.4f}")

sim_out = model.similarity(
    issue_texts=[q.issue_text],
    test_signatures=[q.test_suite_signature],
    old_codes=[q.old_code],
    diffs=[q.out_boundary_diff],
)
print(f"Out-boundary similarity: {sim_out.item():.4f}")
# Expected: in-boundary > out-boundary after training
```

---

## Project Structure

```
EditBoundaryDetection/
├── data/
│   ├── data_loader.py       # Abstract + concrete data loaders
│   ├── execution_tracer.py  # sys.settrace-based execution tracing
│   ├── pairwise_dataset.py  # LLM pair generation, Quintuple dataset
│   └── utils.py             # diff parsing, signature building, etc.
├── models/
│   ├── boundary_encoder.py  # RoBERTa-based anchor encoder
│   ├── edit_encoder.py      # CodeBERT-based edit encoder (+ optional GNN)
│   └── contrastive_model.py # Dual-tower + InfoNCE loss
├── config.py                # All hyperparameters as dataclasses
├── train.py                 # Training loop
├── evaluate.py              # Linear probe, permutation, randomization tests
└── requirements.txt
```
