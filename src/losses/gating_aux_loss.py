from typing import Iterable, Mapping, Sequence, Union

import torch
import torch.nn as nn


class GatedL1Loss(nn.Module):
    """
    Auxiliary loss that minimizes the average gate value.

    Accepts gates from one or multiple TEFScorer layers and averages the per-layer
    penalties to keep the overall scale stable as depth changes.
    """

    def forward(self, gates: Union[torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]]) -> torch.Tensor:
        gate_list = self._flatten_gates(gates)
        if not gate_list:
            raise ValueError("No gates provided to GatedL1Loss")

        losses = []
        for gate in gate_list:
            if gate.ndim == 1:
                batch_size = gate.shape[0]
            else:
                batch_size = gate.shape[0]
            losses.append(gate.mean() / batch_size)

        return torch.stack(losses).mean()

    @staticmethod
    def _flatten_gates(
        gates: Union[torch.Tensor, Sequence[torch.Tensor], Mapping[str, torch.Tensor]],
    ) -> Iterable[torch.Tensor]:
        if isinstance(gates, torch.Tensor):
            return [gates]
        if isinstance(gates, Mapping):
            return list(gates.values())
        if isinstance(gates, (list, tuple)):
            return list(gates)
        raise TypeError(f"Unsupported gates type for GatedL1Loss: {type(gates)}")
