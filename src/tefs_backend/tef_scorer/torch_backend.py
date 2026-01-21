from typing import Tuple

import torch
from torch import nn


def build_keep_mask_torch(
    gates: torch.Tensor,
    attention_mask: torch.Tensor,
    cumulative_threshold: float,
) -> torch.Tensor:
    """
    Torch implementation of the cumulative variance keep mask builder.
    """
    if attention_mask is None:
        active_tokens = torch.ones_like(gates, dtype=torch.bool)
    else:
        active_tokens = attention_mask.bool()

    gated = gates.masked_fill(~active_tokens, 0.0)
    total = gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    shares = gated / total
    shares = shares.masked_fill(~active_tokens, 0.0)

    sorted_share, sorted_idx = shares.sort(dim=-1, descending=True)
    cumulative = sorted_share.cumsum(dim=-1)
    keep_sorted = cumulative <= cumulative_threshold

    keep_sorted[..., 0] = True

    keep_mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
    keep_mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)
    keep_mask &= active_tokens
    return keep_mask


class TorchTEFScorer(nn.Module):
    """
    Token Estimation Function scorer using pure PyTorch operations.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        cumulative_threshold: float = 0.95,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cumulative_threshold = float(cumulative_threshold)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        inference: bool = False,
        cumulative_threshold: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self.dropout(hidden_states)
        logits = self.projection(hidden_states).squeeze(-1)
        gates = torch.sigmoid(logits)

        if not inference:
            return logits, gates, None

        threshold = float(self.cumulative_threshold if cumulative_threshold is None else cumulative_threshold)
        threshold = min(max(threshold, 0.0), 1.0)
        keep_mask = build_keep_mask_torch(
            gates=gates,
            attention_mask=attention_mask,
            cumulative_threshold=threshold,
        )
        return logits, gates, keep_mask
