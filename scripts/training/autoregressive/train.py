import os
import random
from typing import Optional, Tuple

import numpy as np
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from clearml import Task
from clearml import Logger as ClearMLLogger

from src.data.wikitext import Wikitext103Dataset
from src.lightning_modules.autoregressive import AutoregressiveLightningModule


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_clearml_task(
    project_name: str,
    task_name: str,
    configs_to_log: Optional[dict] = None,
) -> Optional[Task]:
    task = Task.init(
        project_name=project_name,
        task_name=task_name,
    )
    if task and configs_to_log:
        for config_name, config in configs_to_log.items():
            if config is None:
                continue
            config_to_log = (
                OmegaConf.to_container(config, resolve=True)
                if isinstance(config, DictConfig)
                else config
            )
            task.connect_configuration(
                name=config_name,
                configuration=config_to_log,
            )

    return task


def prepare_dataloaders(
    tokenizer,
    block_size: int = 1024,
    train_batch_size: int = 4,
    val_batch_size: int = 4,
    num_workers: int = 2,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_dataset = Wikitext103Dataset(
        tokenizer=tokenizer,
        split="train",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        block_size=block_size,
        cache_dir=cache_dir,
    )
    val_dataset = Wikitext103Dataset(
        tokenizer=tokenizer,
        split="validation",
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        block_size=block_size,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_autoregressive(
    model: AutoregressiveLightningModule,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    training_config: DictConfig,
    clearml_configs: Optional[dict] = None,
):
    seed = training_config["seed"]

    accelerator = training_config["training"]["accelerator"]
    devices = training_config["training"]["devices"]
    strategy = training_config["training"]["strategy"]
    max_epochs = training_config["training"]["max_epochs"]
    gradient_clip_val = training_config["training"].get("gradient_clip_val", 0.0)

    log_every_n_steps = training_config["logging"]["log_every_n_steps"]
    val_check_interval = training_config["logging"]["val_check_interval"]
    checkpoints_dir = training_config["logging"]["checkpoints_dir"]
    log_dir = training_config["logging"]["log_dir"]
    run_name = training_config["logging"].get("run_name", "autoregressive-run")

    clearml_section = training_config.get("clearml", None)
    clearml_task = None
    if clearml_section:
        clearml_project_name = clearml_section["project_name"]
        clearml_task_name = clearml_section["task_name"]
        clearml_task = _init_clearml_task(
            project_name=clearml_project_name,
            task_name=clearml_task_name,
            configs_to_log=clearml_configs,
        )
    else:
        clearml_project_name = "autoregressive"
        clearml_task_name = run_name

    _set_seed(seed=seed)

    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name=clearml_task_name,
        version=None,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(checkpoints_dir, clearml_project_name, clearml_task_name),
        filename="{epoch}-{step}",
    )

    callbacks = [
        checkpoint_cb,
        LearningRateMonitor(logging_interval="step"),
    ]

    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        default_root_dir=log_dir,
        enable_checkpointing=True,
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    return trainer
