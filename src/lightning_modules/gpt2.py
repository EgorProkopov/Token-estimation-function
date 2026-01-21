from typing import Any, Dict, Optional

import torch

from src.lightning_modules.autoregressive import AutoregressiveLightningModule
from src.models.gpt2 import GPT2


class GPT2LightningModule(AutoregressiveLightningModule):
    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        lr: float = 5e-5,
        log_step: int = 1000,
        max_epochs: int = 1,
        warmup_steps: int = 0,
        lr_gamma: float = 0.80,
        weight_decay: float = 0.0,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        compile_model: bool = False,
    ):
        """
        LightningModule tailored for fine-tuning GPT-2 on autoregressive text generation tasks.
        """
        gpt2 = GPT2(model_name=model_name, cache_dir=cache_dir)
        if compile_model:
            gpt2.model = torch.compile(model=gpt2.model)

        super().__init__(
            model=gpt2,
            tokenizer=gpt2.tokenizer,
            lr=lr,
            log_step=log_step,
            max_epochs=max_epochs,
            warmup_steps=warmup_steps,
            lr_gamma=lr_gamma,
            weight_decay=weight_decay,
            generation_kwargs=generation_kwargs,
        )

        self.save_hyperparameters(ignore=["tokenizer", "model"])
