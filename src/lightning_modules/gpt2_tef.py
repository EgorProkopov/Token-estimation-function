from typing import Any, Dict, Optional

import torch

from src.lightning_modules.autoregressive import AutoregressiveLightningModule
from src.losses.gating_aux_loss import GatedL1Loss
from src.models.gpt2_tef import GPT2TEF


class GPT2TEFLightningModule(AutoregressiveLightningModule):
    """
    LightningModule for GPT-2 with TEF scoring. Adds an auxiliary gate sparsity loss.
    """

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
        aux_alpha: float = 0.0,
        tef_dropout: float = 0.0,
        tef_cumulative_threshold: float = 0.95,
        tef_backend: str = "torch",
        tef_mask_warmup_steps: int = 0,
        tef_disable_mask_on_val: bool = True,
    ):
        gpt2_tef = GPT2TEF(
            model_name=model_name,
            cache_dir=cache_dir,
            tef_dropout=tef_dropout,
            tef_cumulative_threshold=tef_cumulative_threshold,
            tef_backend=tef_backend,
            keep_mask_mode="off" if tef_mask_warmup_steps > 0 else "train",
        )
        if compile_model:
            gpt2_tef.model = torch.compile(model=gpt2_tef.model)

        super().__init__(
            model=gpt2_tef,
            tokenizer=gpt2_tef.tokenizer,
            lr=lr,
            log_step=log_step,
            max_epochs=max_epochs,
            warmup_steps=warmup_steps,
            lr_gamma=lr_gamma,
            weight_decay=weight_decay,
            generation_kwargs=generation_kwargs,
        )

        self.aux_alpha = aux_alpha
        self.aux_loss_fn = GatedL1Loss()
        self.tef_mask_warmup_steps = tef_mask_warmup_steps
        self.tef_disable_mask_on_val = tef_disable_mask_on_val
        self._current_tef_mask_mode: str = "off" if tef_mask_warmup_steps > 0 else "train"
        self.save_hyperparameters(ignore=["tokenizer", "model", "aux_loss_fn"])

    def training_step(self, batch, batch_idx):
        self._update_tef_keep_mask_mode(training=True)
        input_ids, attention_mask, labels = self._prepare_batch(batch)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        main_loss = outputs.loss
        aux_loss = torch.zeros((), device=main_loss.device)
        if self.aux_alpha > 0.0:
            gates = getattr(self.model, "collect_gates", lambda: [])()
            if gates:
                aux_loss = self.aux_loss_fn(gates)

        loss = main_loss + self.aux_alpha * aux_loss

        self.log(
            "train_loss",
            main_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        if self.aux_alpha > 0.0:
            self.log(
                "train_aux_loss",
                aux_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=False,
            )
        self.log(
            "train_total_loss",
            loss,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

        if self.global_step != 0 and self.global_step % self.log_step == 0:
            perplexity = torch.exp(main_loss.detach())
            self.log(
                "train_ppl",
                perplexity,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        prev_mode = self._set_tef_keep_mask_mode("off" if self.tef_disable_mask_on_val else "eval")
        input_ids, attention_mask, labels = self._prepare_batch(batch)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        self.val_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self._log_gate_mean()

        if prev_mode is not None:
            self._set_tef_keep_mask_mode(prev_mode)
        return loss

    def _set_tef_keep_mask_mode(self, mode: str) -> Optional[str]:
        prev = None
        if hasattr(self.model, "keep_mask_mode"):
            prev = getattr(self.model, "keep_mask_mode", None)
        if hasattr(self.model, "set_keep_mask_mode"):
            self.model.set_keep_mask_mode(mode)
            self._current_tef_mask_mode = mode
        return prev

    def _update_tef_keep_mask_mode(self, training: bool) -> None:
        if not training:
            return
        if self.global_step < self.tef_mask_warmup_steps:
            self._set_tef_keep_mask_mode("off")
        else:
            self._set_tef_keep_mask_mode("train")

    def _log_gate_mean(self) -> None:
        gates = getattr(self.model, "collect_gates", lambda: [])()
        if not gates:
            return
        means = [g.mean() for g in gates if g is not None]
        if not means:
            return
        gate_mean = torch.stack(means).mean().detach()
        self.log(
            "val_gate_mean",
            gate_mean,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
