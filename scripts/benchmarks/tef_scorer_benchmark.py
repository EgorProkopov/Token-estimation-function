import time
from typing import Optional

import torch

from src.tefs.tef_scorer import TEFScorer
from src.tefs_backend.tef_scorer.triton_backend import is_triton_available


DEVICE = "cuda"
BATCH = 8
SEQ_LEN = 1024
HIDDEN_SIZE = 768
DTYPE = "fp32"  # fp16, bf16, fp32
WARMUP_ITERS = 10
BENCH_ITERS = 50
INFERENCE = False
WITH_MASK = False
DROPOUT = 0.0
CUMULATIVE_THRESHOLD = 0.95


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_inputs(
    batch: int,
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    with_mask: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    hidden_states = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
    attention_mask = None
    if with_mask:
        attention_mask = torch.ones(batch, seq_len, device=device, dtype=torch.bool)
    return hidden_states, attention_mask


def _bench_once(
    scorer: TEFScorer,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    inference: bool,
    iters: int,
    device: torch.device,
) -> float:
    _synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            scorer(
                hidden_states,
                attention_mask=attention_mask,
                inference=inference,
            )
    _synchronize(device)
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000.0


def benchmark_backend(
    backend: str,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    inference: bool,
    warmup: int,
    iters: int,
    device: torch.device,
    dropout: float,
    cumulative_threshold: float,
) -> float:
    scorer = TEFScorer(
        hidden_size=hidden_states.size(-1),
        dropout=dropout,
        cumulative_threshold=cumulative_threshold,
        backend=backend,
    ).to(device=device, dtype=hidden_states.dtype)
    scorer.eval()

    with torch.no_grad():
        for _ in range(warmup):
            scorer(hidden_states, attention_mask=attention_mask, inference=inference)
    _synchronize(device)

    return _bench_once(
        scorer=scorer,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        inference=inference,
        iters=iters,
        device=device,
    )


def _compare_outputs(
    torch_out: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    other_out: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    atol: float,
    rtol: float,
) -> None:
    torch_logits, torch_gates, torch_keep = torch_out
    other_logits, other_gates, other_keep = other_out

    logits_ok = torch.allclose(torch_logits, other_logits, atol=atol, rtol=rtol)
    gates_ok = torch.allclose(torch_gates, other_gates, atol=atol, rtol=rtol)
    keep_ok = True
    if torch_keep is not None or other_keep is not None:
        keep_ok = torch.equal(torch_keep, other_keep)

    print(f"correctness logits: {logits_ok} gates: {gates_ok} keep_mask: {keep_ok}")


def main() -> None:
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[DTYPE]

    if device.type == "cuda":
        torch.set_default_device(device)

    hidden_states, attention_mask = _make_inputs(
        BATCH,
        SEQ_LEN,
        HIDDEN_SIZE,
        device,
        dtype,
        WITH_MASK,
    )

    print(f"Device: {device}")
    print(f"Shape: batch={BATCH} seq_len={SEQ_LEN} hidden_size={HIDDEN_SIZE}")
    print(f"Inference: {INFERENCE}  With mask: {WITH_MASK}")

    torch_scorer = TEFScorer(
        hidden_size=hidden_states.size(-1),
        dropout=DROPOUT,
        cumulative_threshold=CUMULATIVE_THRESHOLD,
        backend="torch",
    ).to(device=device, dtype=hidden_states.dtype)
    torch_scorer.eval()
    torch_state = torch_scorer.state_dict()

    with torch.no_grad():
        torch_out = torch_scorer(
            hidden_states,
            attention_mask=attention_mask,
            inference=INFERENCE,
        )

    torch_ms = benchmark_backend(
        backend="torch",
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        inference=INFERENCE,
        warmup=WARMUP_ITERS,
        iters=BENCH_ITERS,
        device=device,
        dropout=DROPOUT,
        cumulative_threshold=CUMULATIVE_THRESHOLD,
    )
    print(f"torch backend: {torch_ms:.3f} ms")

    if device.type == "cuda" and is_triton_available():
        triton_scorer = TEFScorer(
            hidden_size=hidden_states.size(-1),
            dropout=DROPOUT,
            cumulative_threshold=CUMULATIVE_THRESHOLD,
            backend="triton",
        ).to(device=device, dtype=hidden_states.dtype)
        triton_scorer.load_state_dict(torch_state)
        triton_scorer.eval()

        with torch.no_grad():
            triton_out = triton_scorer(
                hidden_states,
                attention_mask=attention_mask,
                inference=INFERENCE,
            )
        _compare_outputs(torch_out, triton_out, atol=1e-3, rtol=1e-3)

        triton_ms = benchmark_backend(
            backend="triton",
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference=INFERENCE,
            warmup=WARMUP_ITERS,
            iters=BENCH_ITERS,
            device=device,
            dropout=DROPOUT,
            cumulative_threshold=CUMULATIVE_THRESHOLD,
        )
        print(f"triton backend: {triton_ms:.3f} ms")
        if triton_ms > 0:
            print(f"speedup: {torch_ms / triton_ms:.2f}x")
    else:
        print("triton backend: unavailable (requires CUDA + triton)")


if __name__ == "__main__":
    main()
