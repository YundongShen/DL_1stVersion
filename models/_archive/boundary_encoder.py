"""Boundary anchor encoder.

Encodes the *intent boundary* of an edit as a fixed-size vector by jointly
encoding the issue description and the test-suite signature:

    [CLS] <issue_text> [SEP] <test_signature> [SEP]

A linear projection maps the pooled representation to the shared embedding
space used by :class:`~models.contrastive_model.ContrastiveModel`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BoundaryEncoder(nn.Module):
    """Dual-text encoder for issue + test-signature pairs.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (default: ``"roberta-base"``).
    projection_dim:
        Output dimensionality after the linear projection head.
    max_length:
        Maximum token length passed to the tokenizer.
    dropout:
        Dropout applied before the projection layer.
    freeze_base:
        If True, freeze all base-model parameters and only train the head.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        projection_dim: int = 256,
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, projection_dim),
        )

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    def tokenize(
        self,
        issue_texts: list[str],
        test_signatures: list[str],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Tokenize a batch of (issue, signature) pairs.

        The two texts are concatenated with [SEP] so the model can attend
        across both fields.
        """
        encoding = self.tokenizer(
            issue_texts,
            test_signatures,
            max_length=self.max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in encoding.items()}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return L2-normalised boundary embeddings of shape (B, projection_dim).

        Parameters
        ----------
        input_ids, attention_mask, token_type_ids:
            Standard HuggingFace tokeniser outputs.
        """
        kwargs: dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.backbone(**kwargs)
        # Use [CLS] token representation
        cls_repr: torch.Tensor = out.last_hidden_state[:, 0, :]
        cls_repr = self.dropout(cls_repr)
        projected: torch.Tensor = self.proj(cls_repr)
        return nn.functional.normalize(projected, dim=-1)

    # ------------------------------------------------------------------
    # Convenience end-to-end method
    # ------------------------------------------------------------------

    def encode(
        self,
        issue_texts: list[str],
        test_signatures: list[str],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Tokenize and encode in one call. Returns (B, projection_dim)."""
        batch = self.tokenize(issue_texts, test_signatures, device)
        return self.forward(**batch)
