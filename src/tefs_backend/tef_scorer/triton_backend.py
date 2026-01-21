from typing import Tuple

import torch
from torch import nn

from src.tefs_backend.tef_scorer.torch_backend import build_keep_mask_torch
from . import triton_kernels as tk


def is_triton_available() -> bool:
    return tk.triton_available() and torch.cuda.is_available()


class TritonKeepMaskBuilder(nn.Module):
    def __init__(self, cumulative_threshold: float):
        super().__init__()
        self.cumulative_threshold = float(cumulative_threshold)

    def forward(
        self,
        gates: torch.Tensor,
        attention_mask: torch.Tensor = None,
        cumulative_threshold: float = None,
    ) -> torch.Tensor:
        threshold = float(self.cumulative_threshold if cumulative_threshold is None else cumulative_threshold)
        threshold = min(max(threshold, 0.0), 1.0)
        return build_keep_mask_triton(
            gates=gates,
            attention_mask=attention_mask,
            cumulative_threshold=threshold,
        )


def build_keep_mask_triton(
    gates: torch.Tensor,
    attention_mask: torch.Tensor,
    cumulative_threshold: float,
) -> torch.Tensor:
    if not (is_triton_available() and gates.is_cuda):
        return build_keep_mask_torch(gates, attention_mask, cumulative_threshold)

    if attention_mask is None:
        active_tokens = None
    else:
        active_tokens = attention_mask.bool()

    if active_tokens is None:
        gated = gates
        total = gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        shares = gated / total
    else:
        gated = gates.masked_fill(~active_tokens, 0.0)
        total = gated.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        shares = gated / total
        shares = shares.masked_fill(~active_tokens, 0.0)
    sorted_share, sorted_idx = shares.sort(dim=-1, descending=True)
    cumulative = sorted_share.cumsum(dim=-1)
    keep_sorted = cumulative <= cumulative_threshold
    keep_sorted[..., 0] = True

    keep_mask = tk.scatter_keep_mask(sorted_idx, keep_sorted)
    if active_tokens is not None:
        keep_mask &= active_tokens
    return keep_mask


class TritonTEFScorer(nn.Module):
    """
    Token Estimation Function scorer backed by Triton kernels where available.
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
        self.keep_mask_builder = TritonKeepMaskBuilder(self.cumulative_threshold)

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

        keep_mask = self.keep_mask_builder(
            gates=gates,
            attention_mask=attention_mask,
            cumulative_threshold=cumulative_threshold,
        )
        return logits, gates, keep_mask
