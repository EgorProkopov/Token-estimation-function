from typing import Tuple

import torch
from torch import nn


class TEFScorer(nn.Module):
    """
    Token Estimation Function scorer.

    This block is intended to sit at the very start of a transformer layer and
    predicts a scalar logit for every token. During training the logits are
    passed through a sigmoid to form continuous gates. During inference the
    gated values are interpreted as a share of explained variance; tokens with
    the smallest share are dropped using a cumulative variance threshold.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        cumulative_threshold: float = 0.95,
    ):
        """
        Args:
            hidden_size: Size of the transformer hidden states.
            dropout: Optional dropout before scoring.
            cumulative_threshold: Default cumulative explained variance threshold
                used to discard low-importance tokens at inference.
        """
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
        """
        Args:
            hidden_states: Input embeddings, shape [batch, seq_len, hidden_size].
            attention_mask: Optional mask with 1 for real tokens and 0 for padding.
            inference: If True, also returns a boolean keep mask after pruning.
            cumulative_threshold: Optional override for the pruning threshold.

        Returns:
            logits: Raw token scores before sigmoid, shape [batch, seq_len].
            gates: Sigmoid(logits) in [0, 1], same shape as logits.
            keep_mask: Only when ``inference`` is True. Boolean mask of tokens to
                keep after applying the cumulative explained variance rule; same
                shape as logits.
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.projection(hidden_states).squeeze(-1)
        gates = torch.sigmoid(logits)

        if not inference:
            return logits, gates, None

        threshold = float(self.cumulative_threshold if cumulative_threshold is None else cumulative_threshold)
        threshold = min(max(threshold, 0.0), 1.0)
        keep_mask = self._build_keep_mask(
            gates=gates,
            attention_mask=attention_mask,
            cumulative_threshold=threshold,
        )
        return logits, gates, keep_mask

    def _build_keep_mask(
        self,
        gates: torch.Tensor,
        attention_mask: torch.Tensor,
        cumulative_threshold: float,
    ) -> torch.Tensor:
        """
        Create a boolean mask that preserves the smallest set of highest-variance
        tokens whose cumulative share exceeds the threshold. Padding positions are
        always dropped.
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
