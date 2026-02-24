import pytest
import torch
from torch import nn

pytest.importorskip("transformers")

from src.models.gpt2_tef import TEFTransformerBlock


class RecordingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return kwargs["hidden_states"]


class FakeScorer(nn.Module):
    def __init__(self, logits: torch.Tensor, gates: torch.Tensor, keep_mask: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.gates = gates
        self.keep_mask = keep_mask
        self.last_call = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        inference: bool = False,
        cumulative_threshold: float = None,
    ):
        self.last_call = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "inference": inference,
            "cumulative_threshold": cumulative_threshold,
        }
        return self.logits, self.gates, self.keep_mask


def test_token_mask_from_attention_2d_tensor() -> None:
    attention_mask = torch.tensor([[1, 0, 2]], dtype=torch.long)

    token_mask = TEFTransformerBlock._token_mask_from_attention(attention_mask)

    assert torch.equal(token_mask, torch.tensor([[True, False, True]]))


def test_token_mask_from_attention_4d_float_tensor() -> None:
    attention_mask = torch.tensor(
        [[[[0.0, -float("inf"), 0.0], [0.0, -float("inf"), 0.0], [0.0, -float("inf"), 0.0]]]],
        dtype=torch.float32,
    )

    token_mask = TEFTransformerBlock._token_mask_from_attention(attention_mask)

    assert torch.equal(token_mask, torch.tensor([[True, False, True]]))


def test_apply_keep_mask_to_attention_with_none_attention_mask() -> None:
    keep_mask = torch.tensor([[True, False, True]])

    updated = TEFTransformerBlock._apply_keep_mask_to_attention(
        attention_mask=None,
        keep_mask=keep_mask,
        dtype=torch.float32,
    )

    expected = torch.tensor(
        [[[[0.0, torch.finfo(torch.float32).min, 0.0]]]],
        dtype=torch.float32,
    )
    assert torch.equal(updated, expected)


def test_forward_does_not_apply_keep_mask_when_mode_is_off() -> None:
    block = RecordingBlock()
    tef_block = TEFTransformerBlock(block=block, hidden_size=2, keep_mask_mode="off")
    hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1]], dtype=torch.long)
    gates = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    logits = torch.zeros_like(gates)
    keep_mask = torch.tensor([[False, True]])
    tef_block.scorer = FakeScorer(logits=logits, gates=gates, keep_mask=keep_mask)

    output = tef_block(hidden_states=hidden_states, attention_mask=attention_mask)

    expected_hidden = hidden_states * gates.unsqueeze(-1)
    assert tef_block.scorer.last_call["inference"] is False
    assert torch.equal(tef_block.scorer.last_call["attention_mask"], torch.tensor([[True, True]]))
    assert torch.equal(block.last_kwargs["hidden_states"], expected_hidden)
    assert torch.equal(block.last_kwargs["attention_mask"], attention_mask)
    assert torch.equal(output, expected_hidden)


def test_forward_applies_keep_mask_in_train_mode() -> None:
    block = RecordingBlock()
    tef_block = TEFTransformerBlock(block=block, hidden_size=2, keep_mask_mode="train")
    tef_block.train()

    hidden_states = torch.ones((1, 3, 2), dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    gates = torch.tensor([[0.5, 0.2, 0.8]], dtype=torch.float32)
    logits = torch.zeros_like(gates)
    keep_mask = torch.tensor([[True, False, True]])
    tef_block.scorer = FakeScorer(logits=logits, gates=gates, keep_mask=keep_mask)

    output = tef_block(hidden_states=hidden_states, attention_mask=attention_mask)

    expected_hidden = hidden_states * gates.unsqueeze(-1) * keep_mask.unsqueeze(-1)
    expected_attention = torch.tensor([[1, 0, 1]], dtype=torch.long)
    assert tef_block.scorer.last_call["inference"] is True
    assert torch.equal(block.last_kwargs["hidden_states"], expected_hidden)
    assert torch.equal(block.last_kwargs["attention_mask"], expected_attention)
    assert torch.equal(output, expected_hidden)


def test_use_keep_mask_for_eval_mode_depends_on_training_state() -> None:
    tef_block = TEFTransformerBlock(block=RecordingBlock(), hidden_size=2, keep_mask_mode="eval")
    tef_block.train()
    assert tef_block._use_keep_mask() is False

    tef_block.eval()
    assert tef_block._use_keep_mask() is True
