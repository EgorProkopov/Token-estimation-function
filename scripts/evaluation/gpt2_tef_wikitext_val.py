import math
import os
import random
from typing import Any, Iterable, List, Optional, Tuple

import dotenv
import numpy as np
from omegaconf import DictConfig, OmegaConf
from omegaconf.base import ContainerMetadata

import torch
from torch.utils.data import DataLoader

from src.data.wikitext import Wikitext103Dataset
from src.lightning_modules.gpt2 import GPT2LightningModule
from src.lightning_modules.gpt2_tef import GPT2TEFLightningModule
from src.models.gpt2_tef import TEFTransformerBlock

CHECKPOINT_PATH = "weights/gpt2_tef_wikitext/epoch=2-step=43072.ckpt"
MODEL_CONFIG_PATH = "configs/models/gpt2.yaml"
TRAINING_CONFIG_PATH = "configs/training/gpt2_tef.yaml"
TEF_SCORER_CONFIG_PATH = "configs/tefs/tef_scorer.yaml"

THRESHOLDS = [0.5, 0.7, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0]
DEVICE = "cuda:1"
SEED = 42

DATASET_NAME = None
DATASET_CONFIG = None
BLOCK_SIZE = None
VAL_BATCH_SIZE = None
NUM_WORKERS = None
CACHE_DIR = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(path: Optional[str], base_dir: str) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _build_val_dataloader(
    tokenizer,
    dataset_name: str,
    dataset_config: str,
    block_size: int,
    val_batch_size: int,
    num_workers: int,
    cache_dir: Optional[str],
) -> DataLoader:
    val_dataset = Wikitext103Dataset(
        tokenizer=tokenizer,
        split="validation",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        block_size=block_size,
        cache_dir=cache_dir,
    )
    return DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def _evaluate_ppl(model, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.to(device)
    model.eval()
    losses: List[torch.Tensor] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        losses.append(outputs.loss.detach().float())

    mean_loss = torch.stack(losses).mean().item()
    return mean_loss, math.exp(mean_loss)


def _get_tef_blocks(model: GPT2TEFLightningModule) -> List[TEFTransformerBlock]:
    tef_model = getattr(model, "model", None)
    if tef_model is None:
        return []
    core = getattr(tef_model, "model", None)
    if core is None:
        return []
    blocks = getattr(core.transformer, "h", None)
    if blocks is None:
        return []
    return [block for block in blocks if isinstance(block, TEFTransformerBlock)]


def _set_tef_cumulative_threshold(model: GPT2TEFLightningModule, threshold: float) -> None:
    for block in _get_tef_blocks(model):
        scorer = block.scorer
        scorer.cumulative_threshold = float(threshold)
        impl = scorer.impl
        if hasattr(impl, "cumulative_threshold"):
            impl.cumulative_threshold = float(threshold)
        if hasattr(impl, "keep_mask_builder") and hasattr(impl.keep_mask_builder, "cumulative_threshold"):
            impl.keep_mask_builder.cumulative_threshold = float(threshold)


def _format_table(headers: Iterable[str], rows: Iterable[Iterable[str]]) -> str:
    headers = list(headers)
    rows = [list(row) for row in rows]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def _format_row(row_vals: Iterable[str]) -> str:
        return " | ".join(str(val).ljust(widths[idx]) for idx, val in enumerate(row_vals))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_format_row(headers), separator]
    lines.extend(_format_row(row) for row in rows)
    return "\n".join(lines)


def _evaluate_ppl_with_lengths(
    model: GPT2TEFLightningModule,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, List[float]]:
    blocks = _get_tef_blocks(model)
    stats = [{"tokens": 0.0, "sequences": 0} for _ in blocks]
    hooks = []

    for idx, block in enumerate(blocks):
        def _hook(_module, _inputs, output, layer_idx=idx):
            if not isinstance(output, tuple) or len(output) < 3:
                return
            keep_mask = output[2]
            if keep_mask is None:
                return
            stats[layer_idx]["tokens"] += keep_mask.sum().item()
            stats[layer_idx]["sequences"] += keep_mask.size(0)

        hooks.append(block.scorer.register_forward_hook(_hook))

    _, ppl = _evaluate_ppl(model, dataloader, device)

    for hook in hooks:
        hook.remove()

    lengths = []
    for stat in stats:
        sequences = stat["sequences"]
        lengths.append(stat["tokens"] / sequences if sequences else 0.0)

    return ppl, lengths


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    """
    Make checkpoint keys compatible with the current module structure.
    Handles compiled model prefixes and the move from scorer.projection to scorer.impl.projection.
    """
    normalized = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.model._orig_mod.", "model.model.")
        new_key = new_key.replace(".scorer.projection.", ".scorer.impl.projection.")
        normalized[new_key] = value
    return normalized


def _init_tef_model(
    model_config,
    training_config,
    tef_scorer_config,
) -> GPT2TEFLightningModule:
    training_hparams = training_config["training"]
    tef_cumulative_threshold = tef_scorer_config.get("cumulative_threshold", 0.95)
    tef_dropout = tef_scorer_config.get("dropout", 0.0)
    tef_backend = tef_scorer_config.get("backend", "torch")

    return GPT2TEFLightningModule(
        model_name=model_config["model_name"],
        cache_dir=model_config.get("cache_dir", None),
        lr=training_hparams.get("lr", 5e-5),
        log_step=training_hparams.get("log_step", training_config["logging"]["log_every_n_steps"]),
        max_epochs=training_hparams.get("max_epochs", training_config["training"]["max_epochs"]),
        warmup_steps=training_hparams.get("warmup_steps", 0),
        lr_gamma=training_hparams.get("lr_gamma", 0.80),
        weight_decay=training_hparams.get("weight_decay", 0.0),
        generation_kwargs=model_config.get("generation", None),
        compile_model=False,
        aux_alpha=0.0,
        tef_dropout=tef_dropout,
        tef_cumulative_threshold=tef_cumulative_threshold,
        tef_backend=tef_backend,
    )


def main() -> None:
    dotenv.load_dotenv()

    base_dir = os.getcwd()
    model_config = OmegaConf.load(_resolve_path(MODEL_CONFIG_PATH, base_dir))
    training_config = OmegaConf.load(_resolve_path(TRAINING_CONFIG_PATH, base_dir))
    tef_scorer_config = OmegaConf.load(_resolve_path(TEF_SCORER_CONFIG_PATH, base_dir))

    dataset_config = training_config["dataset"]
    dataset_name = DATASET_NAME or dataset_config["name"]
    dataset_cfg = DATASET_CONFIG or dataset_config["config"]
    block_size = BLOCK_SIZE or dataset_config["block_size"]
    val_batch_size = VAL_BATCH_SIZE or dataset_config["val_batch_size"]
    num_workers = NUM_WORKERS or dataset_config["num_workers"]
    cache_dir = CACHE_DIR if CACHE_DIR is not None else dataset_config.get("cache_dir", None)

    thresholds = THRESHOLDS

    _set_seed(SEED)
    device = _resolve_device(DEVICE)

    gpt2_baseline = GPT2LightningModule(
        model_name=model_config["model_name"],
        cache_dir=model_config.get("cache_dir", None),
        lr=training_config["training"].get("lr", 5e-5),
        log_step=training_config["training"].get("log_step", 1000),
        max_epochs=training_config["training"].get("max_epochs", 1),
        warmup_steps=training_config["training"].get("warmup_steps", 0),
        lr_gamma=training_config["training"].get("lr_gamma", 0.8),
        weight_decay=training_config["training"].get("weight_decay", 0.0),
        generation_kwargs=model_config.get("generation", None),
        compile_model=False,
    )

    val_dataloader = _build_val_dataloader(
        tokenizer=gpt2_baseline.tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_cfg,
        block_size=block_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )

    _, baseline_ppl = _evaluate_ppl(gpt2_baseline, val_dataloader, device)
    del gpt2_baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checkpoint_path = _resolve_path(CHECKPOINT_PATH, base_dir)

    torch.serialization.add_safe_globals([DictConfig, ContainerMetadata, Any])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict_raw = checkpoint.get("state_dict", checkpoint)
    state_dict = _normalize_state_dict_keys(state_dict_raw)

    gpt2_tef = _init_tef_model(
        model_config=model_config,
        training_config=training_config,
        tef_scorer_config=tef_scorer_config,
    )
    missing, unexpected = gpt2_tef.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys when loading TEF checkpoint ({len(missing)}): {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading TEF checkpoint ({len(unexpected)}): {unexpected}")

    num_layers = len(_get_tef_blocks(gpt2_tef))
    baseline_lengths = [float(block_size)] * num_layers
    rows = [
        ["gpt2 (raw)", "-", f"{baseline_ppl:.4f}"]
        + [f"{length:.1f}" for length in baseline_lengths],
    ]

    for threshold in thresholds:
        _set_tef_cumulative_threshold(gpt2_tef, threshold)
        ppl, lengths = _evaluate_ppl_with_lengths(gpt2_tef, val_dataloader, device)
        rows.append(
            [f"gpt2_tef", f"{threshold:.4f}", f"{ppl:.4f}"]
            + [f"{length:.1f}" for length in lengths],
        )

    headers = ["model", "cumulative_threshold", "perplexity"] + [
        f"L{idx}_len" for idx in range(num_layers)
    ]
    print(_format_table(headers, rows))


if __name__ == "__main__":
    main()
