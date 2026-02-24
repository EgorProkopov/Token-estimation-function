import pytest
import torch

from src.losses.gating_aux_loss import GatedL1Loss


def test_gated_l1_loss_accepts_tensor() -> None:
    loss_fn = GatedL1Loss()
    gates = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32)

    loss = loss_fn(gates)

    expected = gates.mean() / gates.shape[0]
    assert torch.isclose(loss, expected)


def test_gated_l1_loss_averages_multiple_gate_collections() -> None:
    loss_fn = GatedL1Loss()
    gates_a = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    gates_b = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.float32)

    loss_list = loss_fn([gates_a, gates_b])
    loss_dict = loss_fn({"layer_1": gates_a, "layer_2": gates_b})

    per_layer = gates_a.mean() / gates_a.shape[0]
    expected = torch.stack([per_layer, per_layer]).mean()
    assert torch.isclose(loss_list, expected)
    assert torch.isclose(loss_dict, expected)


def test_gated_l1_loss_raises_for_empty_collection() -> None:
    loss_fn = GatedL1Loss()

    with pytest.raises(ValueError, match="No gates provided"):
        loss_fn([])


def test_gated_l1_loss_raises_for_unsupported_type() -> None:
    loss_fn = GatedL1Loss()

    with pytest.raises(TypeError, match="Unsupported gates type"):
        loss_fn(42)
