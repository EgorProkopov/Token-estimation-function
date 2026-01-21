from typing import Tuple

import torch
from torch import nn

from src.tefs_backend.tef_scorer.torch_backend import TorchTEFScorer
from src.tefs_backend.tef_scorer.triton_backend import TritonTEFScorer, is_triton_available


class TEFScorer(nn.Module):
    """
    Token Estimation Function scorer with selectable backend.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        cumulative_threshold: float = 0.95,
        backend: str = "torch",
    ):
        super().__init__()
        backend_key = backend.lower()
        if backend_key == "auto":
            backend_key = "triton" if is_triton_available() else "torch"

        if backend_key == "torch":
            self.impl = TorchTEFScorer(
                hidden_size=hidden_size,
                dropout=dropout,
                cumulative_threshold=cumulative_threshold,
            )
        elif backend_key == "triton":
            self.impl = TritonTEFScorer(
                hidden_size=hidden_size,
                dropout=dropout,
                cumulative_threshold=cumulative_threshold,
            )
        else:
            raise ValueError(f"Unsupported TEF backend: {backend}")

        self.backend = backend_key
        self.hidden_size = self.impl.hidden_size
        self.cumulative_threshold = self.impl.cumulative_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        inference: bool = False,
        cumulative_threshold: float = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.impl(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference=inference,
            cumulative_threshold=cumulative_threshold,
        )
