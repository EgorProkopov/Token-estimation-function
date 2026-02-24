import torch

from src.tefs_backend.tef_scorer.torch_backend import TorchTEFScorer, build_keep_mask_torch


def test_build_keep_mask_torch_respects_attention_mask() -> None:
    gates = torch.tensor([[0.1, 0.2, 0.7, 0.9]], dtype=torch.float32)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    keep_mask = build_keep_mask_torch(
        gates=gates,
        attention_mask=attention_mask,
        cumulative_threshold=0.5,
    )

    expected = torch.tensor([[False, False, True, False]])
    assert torch.equal(keep_mask, expected)


def test_build_keep_mask_torch_keeps_top_token_with_zero_threshold() -> None:
    gates = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)

    keep_mask = build_keep_mask_torch(
        gates=gates,
        attention_mask=None,
        cumulative_threshold=0.0,
    )

    expected = torch.tensor([[False, False, True]])
    assert torch.equal(keep_mask, expected)


def test_torch_tef_scorer_returns_none_keep_mask_when_not_inference() -> None:
    scorer = TorchTEFScorer(hidden_size=1, dropout=0.0, cumulative_threshold=0.95)
    with torch.no_grad():
        scorer.projection.weight.fill_(1.0)
        scorer.projection.bias.zero_()

    hidden_states = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)
    logits, gates, keep_mask = scorer(hidden_states=hidden_states, inference=False)

    assert logits.shape == (1, 3)
    assert gates.shape == (1, 3)
    assert keep_mask is None


def test_torch_tef_scorer_clamps_threshold_in_inference_mode() -> None:
    scorer = TorchTEFScorer(hidden_size=1, dropout=0.0, cumulative_threshold=0.95)
    with torch.no_grad():
        scorer.projection.weight.fill_(1.0)
        scorer.projection.bias.zero_()

    hidden_states = torch.tensor([[[1.0], [2.0], [3.0]]], dtype=torch.float32)

    _, _, keep_all = scorer(
        hidden_states=hidden_states,
        inference=True,
        cumulative_threshold=10.0,
    )
    _, _, keep_top = scorer(
        hidden_states=hidden_states,
        inference=True,
        cumulative_threshold=-10.0,
    )

    assert torch.equal(keep_all, torch.ones_like(keep_all, dtype=torch.bool))
    assert torch.equal(keep_top, torch.tensor([[False, False, True]]))
