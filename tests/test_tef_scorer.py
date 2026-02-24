import pytest

from src.tefs import tef_scorer as tef_scorer_module
from src.tefs.tef_scorer import TEFScorer
from src.tefs_backend.tef_scorer.torch_backend import TorchTEFScorer
from src.tefs_backend.tef_scorer.triton_backend import TritonTEFScorer


def test_tef_scorer_uses_torch_backend() -> None:
    scorer = TEFScorer(hidden_size=8, backend="torch")

    assert scorer.backend == "torch"
    assert isinstance(scorer.impl, TorchTEFScorer)


def test_tef_scorer_auto_falls_back_to_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tef_scorer_module, "is_triton_available", lambda: False)

    scorer = TEFScorer(hidden_size=8, backend="auto")

    assert scorer.backend == "torch"
    assert isinstance(scorer.impl, TorchTEFScorer)


def test_tef_scorer_auto_uses_triton_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tef_scorer_module, "is_triton_available", lambda: True)

    scorer = TEFScorer(hidden_size=8, backend="auto")

    assert scorer.backend == "triton"
    assert isinstance(scorer.impl, TritonTEFScorer)


def test_tef_scorer_raises_for_unsupported_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported TEF backend"):
        TEFScorer(hidden_size=8, backend="something-else")
