"""Edit encoder.

Encodes a (old_code, diff) pair into a fixed-size vector representing *what
was changed and how broadly*.  Two encoding paths are available:

1. **Text-only** (default): CodeBERT encodes the concatenation
   ``[CLS] old_code [SEP] diff [SEP]``.
2. **GNN-augmented** (opt-in): an AST change-graph is extracted from the diff
   and processed by a Graph Attention Network whose node embeddings are fused
   with the text representation before the projection head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Optional GNN module (requires torch_geometric or dgl)
# ---------------------------------------------------------------------------

class _DiffGNN(nn.Module):
    """GAT over the diff's AST change-graph.

    Nodes represent AST node types touched by the diff; edges are parent-child
    or data-flow edges. In practice you would extract this graph with a library
    such as tree-sitter; here we provide a self-contained stub.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATConv  # type: ignore[import]
            self._backend = "pyg"
        except ImportError:
            try:
                import dgl.nn as dglnn  # type: ignore[import]
                self._backend = "dgl"
            except ImportError:
                self._backend = "none"

        self.layers: nn.ModuleList = nn.ModuleList()
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        for i in range(layers):
            if self._backend == "pyg":
                from torch_geometric.nn import GATConv  # type: ignore[import]
                self.layers.append(GATConv(dims[i], dims[i + 1], heads=1))
            elif self._backend == "dgl":
                import dgl.nn as dglnn  # type: ignore[import]
                self.layers.append(dglnn.GATConv(dims[i], dims[i + 1], num_heads=1))
            else:
                # Fallback: simple linear (ignores graph structure)
                self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled graph representation.

        Parameters
        ----------
        x:
            Node features (N, in_dim).
        edge_index:
            COO edge index (2, E) for PyG; ignored for the Linear fallback.
        """
        for layer in self.layers:
            if self._backend in ("pyg",):
                x = self.act(layer(x, edge_index))
            elif self._backend == "dgl":
                import dgl  # type: ignore[import]
                g = dgl.graph((edge_index[0], edge_index[1]))
                x = self.act(layer(g, x).squeeze(1))
            else:
                x = self.act(layer(x))
        return x.mean(dim=0, keepdim=True)   # (1, out_dim) — graph-level


# ---------------------------------------------------------------------------
# Main edit encoder
# ---------------------------------------------------------------------------

class EditEncoder(nn.Module):
    """Encode (old_code, diff) into the shared embedding space.

    Parameters
    ----------
    model_name:
        HuggingFace CodeBERT-style model.
    projection_dim:
        Output dimensionality (must match :class:`~models.boundary_encoder.BoundaryEncoder`).
    max_length:
        Token limit for the CodeBERT backbone.
    dropout:
        Dropout rate.
    use_gnn:
        Enable the GNN branch for AST-level structural encoding.
    gnn_layers:
        Depth of the GAT when ``use_gnn=True``.
    gnn_hidden:
        Hidden size of the GAT.
    freeze_base:
        Freeze CodeBERT parameters and train only the head.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        projection_dim: int = 256,
        max_length: int = 512,
        dropout: float = 0.1,
        use_gnn: bool = False,
        gnn_layers: int = 2,
        gnn_hidden: int = 256,
        freeze_base: bool = False,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.use_gnn = use_gnn

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        if freeze_base:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout)

        fusion_dim = hidden
        if use_gnn:
            self.gnn = _DiffGNN(hidden, gnn_hidden, gnn_hidden, layers=gnn_layers)
            fusion_dim = hidden + gnn_hidden

        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, projection_dim),
        )

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    def tokenize(
        self,
        old_codes: list[str],
        diffs: list[str],
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        encoding = self.tokenizer(
            old_codes,
            diffs,
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
        gnn_node_features: torch.Tensor | None = None,
        gnn_edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return L2-normalised edit embeddings of shape (B, projection_dim).

        Parameters
        ----------
        input_ids, attention_mask, token_type_ids:
            CodeBERT tokeniser outputs.
        gnn_node_features:
            Node feature matrix (N, H) for the optional GNN branch.
            If None and ``use_gnn=True``, the GNN branch is skipped.
        gnn_edge_index:
            COO edge index (2, E).
        """
        kwargs: dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.backbone(**kwargs)
        text_repr: torch.Tensor = out.last_hidden_state[:, 0, :]  # [CLS]
        text_repr = self.dropout(text_repr)

        if self.use_gnn and gnn_node_features is not None and gnn_edge_index is not None:
            graph_repr = self.gnn(gnn_node_features, gnn_edge_index)
            # Broadcast graph-level embedding to batch size
            graph_repr = graph_repr.expand(text_repr.size(0), -1)
            combined = torch.cat([text_repr, graph_repr], dim=-1)
        else:
            combined = text_repr

        projected: torch.Tensor = self.proj(combined)
        return nn.functional.normalize(projected, dim=-1)

    # ------------------------------------------------------------------
    # Convenience end-to-end method
    # ------------------------------------------------------------------

    def encode(
        self,
        old_codes: list[str],
        diffs: list[str],
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Tokenize and encode in one call. Returns (B, projection_dim)."""
        batch = self.tokenize(old_codes, diffs, device)
        return self.forward(**batch)
