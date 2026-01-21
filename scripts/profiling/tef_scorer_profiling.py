import os
from contextlib import nullcontext
from typing import Optional

import torch
from torch.profiler import ProfilerActivity

from src.models.gpt2_tef import GPT2TEF
from src.tefs_backend.tef_scorer.triton_backend import is_triton_available


MODEL_NAME = "gpt2"
CACHE_DIR: Optional[str] = None
DEVICE = "cuda:0"  # "auto", "cpu", "cuda", "cuda:0", etc.
DTYPE = "float32"  # "auto", "float16", "bfloat16", "float32"

TEF_BACKEND = "triton"  # "torch" or "triton"
KEEP_MASK_MODE = "eval"  # "off", "eval", "train"

BATCH_SIZE = 2
SEQ_LEN = 256
WARMUP_STEPS = 5
PROFILE_STEPS = 10

SEED = 239

TRACE_DIR = "logs/profiling"
TRACE_NAME = "tef_scorer_trace.json"

RECORD_SHAPES = True
PROFILE_MEMORY = True
WITH_STACK = False


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    return getattr(torch, dtype)


def _maybe_fallback_backend(backend: str, device: torch.device) -> str:
    if backend.lower() != "triton":
        return backend
    if device.type != "cuda" or not is_triton_available():
        print("Triton backend unavailable on this device; falling back to torch backend.")
        return "torch"
    return backend


def _build_inputs(vocab_size: int, device: torch.device, dtype: torch.dtype):
    input_ids = torch.randint(0, vocab_size, (BATCH_SIZE, SEQ_LEN), device=device)
    attention_mask = torch.ones((BATCH_SIZE, SEQ_LEN), device=device, dtype=torch.long)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = _resolve_device(DEVICE)
    dtype = _resolve_dtype(device, DTYPE)
    backend = _maybe_fallback_backend(TEF_BACKEND, device)

    model = GPT2TEF(
        model_name=MODEL_NAME,
        cache_dir=CACHE_DIR,
        tef_backend=backend,
    )
    model.set_keep_mask_mode(KEEP_MASK_MODE)
    model.eval()
    model.to(device=device, dtype=dtype)

    vocab_size = model.model.config.vocab_size
    input_ids, attention_mask, labels = _build_inputs(vocab_size, device, dtype)

    total_steps = WARMUP_STEPS + PROFILE_STEPS
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(TRACE_DIR, exist_ok=True)
    trace_path = os.path.join(TRACE_DIR, TRACE_NAME)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=dtype)
        if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )

    with torch.inference_mode():
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=WARMUP_STEPS, warmup=0, active=PROFILE_STEPS, repeat=1),
            record_shapes=RECORD_SHAPES,
            profile_memory=PROFILE_MEMORY,
            with_stack=WITH_STACK,
        ) as prof:
            for _ in range(total_steps):
                with autocast_ctx:
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                prof.step()

    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved to: {trace_path}")


if __name__ == "__main__":
    main()
