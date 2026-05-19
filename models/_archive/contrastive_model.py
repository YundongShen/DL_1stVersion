"""Contrastive model and InfoNCE loss for edit-boundary detection.

The model jointly trains a :class:`~models.boundary_encoder.BoundaryEncoder`
(anchor) and an :class:`~models.edit_encoder.EditEncoder` (positive/negative)
using InfoNCE loss with temperature τ.

For each sample in a batch of size B:
  - anchor  : BoundaryEncoder(issue_text, test_signature)  → (B, D)
  - positive: EditEncoder(old_code, in_boundary_diff)       → (B, D)
  - negative: EditEncoder(old_code, out_boundary_diff)      → (B, D)

Loss (per anchor i):
    -log( exp(sim(a_i, p_i)/τ) / Σ_j exp(sim(a_i, n_j)/τ) )

where the denominator runs over all negatives in the batch, making each
in-batch negative contribute as a hard negative against every anchor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_encoder import BoundaryEncoder
from .edit_encoder import EditEncoder


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss for triplet (anchor, positive, negative) batches.

    Parameters
    ----------
    temperature:
        Softmax temperature τ.  Smaller values produce sharper distributions.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Parameters
        ----------
        anchors:
            L2-normalised boundary embeddings, shape (B, D).
        positives:
            L2-normalised in-boundary edit embeddings, shape (B, D).
        negatives:
            L2-normalised out-boundary edit embeddings, shape (B, D).

        Returns
        -------
        torch.Tensor
            Scalar loss averaged over the batch.
        """
        # All inputs are assumed unit-norm (encoders already apply F.normalize).
        # sim(a_i, p_i): (B,)
        pos_sim = (anchors * positives).sum(dim=-1) / self.temperature  # (B,)

        # sim(a_i, n_j) for all i,j: (B, B)
        neg_sim = anchors @ negatives.T / self.temperature  # (B, B)

        # For each anchor i, logits = [sim(a_i, n_0), ..., sim(a_i, n_{B-1})]
        # We insert the positive as an additional "class" (index 0 by convention).
        # logits shape: (B, B+1) — column 0 is the positive.
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, B+1)

        # Ground-truth label is always index 0 (the positive column).
        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)

        return F.cross_entropy(logits, labels)


class ContrastiveModel(nn.Module):
    """End-to-end contrastive model for edit-boundary detection.

    Wraps a :class:`BoundaryEncoder` and an :class:`EditEncoder` and exposes
    a single ``forward`` that accepts pre-tokenised inputs for all three roles
    (anchor, positive edit, negative edit) and returns the InfoNCE loss together
    with the three embedding tensors for downstream use.

    Parameters
    ----------
    boundary_encoder:
        Encodes (issue, test-signature) pairs.
    edit_encoder:
        Encodes (old_code, diff) pairs.
    temperature:
        InfoNCE temperature τ (default from :class:`InfoNCELoss`).
    """

    def __init__(
        self,
        boundary_encoder: BoundaryEncoder,
        edit_encoder: EditEncoder,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.boundary_encoder = boundary_encoder
        self.edit_encoder = edit_encoder
        self.loss_fn = InfoNCELoss(temperature=temperature)

    # ------------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg) -> "ContrastiveModel":
        """Build from a :class:`~config.Config` (or compatible namespace)."""
        mc = cfg.model
        tc = cfg.train
        boundary_enc = BoundaryEncoder(
            model_name=mc.issue_encoder_name,
            projection_dim=mc.projection_dim,
            dropout=mc.dropout,
        )
        edit_enc = EditEncoder(
            model_name=mc.code_encoder_name,
            projection_dim=mc.projection_dim,
            dropout=mc.dropout,
            use_gnn=mc.use_gnn,
            gnn_layers=mc.gnn_layers,
            gnn_hidden=mc.gnn_hidden,
        )
        return cls(boundary_enc, edit_enc, temperature=tc.temperature)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        boundary_input_ids: torch.Tensor,
        boundary_attention_mask: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attention_mask: torch.Tensor,
        neg_input_ids: torch.Tensor,
        neg_attention_mask: torch.Tensor,
        boundary_token_type_ids: torch.Tensor | None = None,
        pos_token_type_ids: torch.Tensor | None = None,
        neg_token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode all three roles and compute InfoNCE loss.

        Parameters
        ----------
        boundary_input_ids, boundary_attention_mask:
            BoundaryEncoder tokeniser outputs for the (issue, test-sig) pairs.
        pos_input_ids, pos_attention_mask:
            EditEncoder tokeniser outputs for in-boundary (positive) diffs.
        neg_input_ids, neg_attention_mask:
            EditEncoder tokeniser outputs for out-boundary (negative) diffs.
        *_token_type_ids:
            Optional token-type tensors for BERT-style models.

        Returns
        -------
        loss : torch.Tensor
            Scalar InfoNCE loss.
        anchor_emb : torch.Tensor  (B, projection_dim)
        pos_emb    : torch.Tensor  (B, projection_dim)
        neg_emb    : torch.Tensor  (B, projection_dim)
        """
        anchor_emb: torch.Tensor = self.boundary_encoder(
            boundary_input_ids,
            boundary_attention_mask,
            token_type_ids=boundary_token_type_ids,
        )
        pos_emb: torch.Tensor = self.edit_encoder(
            pos_input_ids,
            pos_attention_mask,
            token_type_ids=pos_token_type_ids,
        )
        neg_emb: torch.Tensor = self.edit_encoder(
            neg_input_ids,
            neg_attention_mask,
            token_type_ids=neg_token_type_ids,
        )

        loss = self.loss_fn(anchor_emb, pos_emb, neg_emb)
        return loss, anchor_emb, pos_emb, neg_emb

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_boundary(
        self,
        issue_texts: list[str],
        test_signatures: list[str],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return boundary embeddings for a list of (issue, test-sig) pairs."""
        return self.boundary_encoder.encode(issue_texts, test_signatures, device)

    @torch.no_grad()
    def encode_edit(
        self,
        old_codes: list[str],
        diffs: list[str],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return edit embeddings for a list of (old_code, diff) pairs."""
        return self.edit_encoder.encode(old_codes, diffs, device)

    @torch.no_grad()
    def similarity(
        self,
        issue_texts: list[str],
        test_signatures: list[str],
        old_codes: list[str],
        diffs: list[str],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return cosine similarities between boundary and edit embeddings.

        A high score means the diff stays within the edit boundary; a low score
        suggests scope creep.

        Returns
        -------
        torch.Tensor of shape (B,) with values in [-1, 1].
        """
        a = self.encode_boundary(issue_texts, test_signatures, device)
        e = self.encode_edit(old_codes, diffs, device)
        return (a * e).sum(dim=-1)
