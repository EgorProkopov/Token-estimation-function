from typing import Dict, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR


class AutoregressiveLightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        lr: float,
        log_step: int = 1000,
        max_epochs: int = 1,
        warmup_steps: int = 0,
        lr_gamma: float = 0.80,
        weight_decay: float = 0.0,
        generation_kwargs: Optional[Dict[str, Union[int, float, bool]]] = None,
    ):
        """
        Base LightningModule for autoregressive language models.

        Args:
            model: Autoregressive language model that returns a loss when labels are provided.
            tokenizer: Tokenizer paired with the model. Used for generation utilities.
            lr: Learning rate for AdamW.
            log_step: Log metrics every ``log_step`` training steps.
            max_epochs: Total epochs for exponential decay schedule. Set to 0 to disable schedulers.
            warmup_steps: Number of linear warmup steps.
            lr_gamma: Exponential decay factor applied each epoch.
            weight_decay: Weight decay for AdamW.
            generation_kwargs: Default kwargs passed to ``generate_text``.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr
        self.log_step = log_step
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.generation_kwargs = generation_kwargs or {}

        self.val_losses: List[torch.Tensor] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _prepare_batch(
        self,
        batch: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            labels = batch.get("labels")
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
            labels = batch[2] if len(batch) > 2 else None
        else:
            input_ids = batch
            attention_mask = None
            labels = None

        if labels is None:
            labels = input_ids

        return input_ids, attention_mask, labels

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self._prepare_batch(batch)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        if self.global_step != 0 and self.global_step % self.log_step == 0:
            perplexity = torch.exp(loss.detach())
            self.log(
                "train_ppl",
                perplexity,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self._prepare_batch(batch)
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        self.val_losses.append(loss.detach())
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if not self.val_losses:
            return

        epoch_loss = torch.stack(self.val_losses).mean()
        self.log("val_loss_epoch", epoch_loss, prog_bar=True)
        self.log("val_ppl_epoch", torch.exp(epoch_loss), prog_bar=True)

        self.val_losses.clear()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if self.max_epochs <= 0:
            return optimizer

        schedulers = []
        if self.warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_steps,
            )
            schedulers.append({
                "scheduler": warmup,
                "interval": "step",
                "frequency": 1,
                "name": "linear_warmup_lr",
            })

        exp_decay = ExponentialLR(
            optimizer,
            gamma=self.lr_gamma,
        )
        schedulers.append({
            "scheduler": exp_decay,
            "interval": "epoch",
            "frequency": 1,
            "name": "exponential_lr",
        })

        return [optimizer], schedulers

    def on_before_optimizer_step(self, optimizer):
        if self.global_step == 0 or self.global_step % self.log_step != 0:
            return

        grads = [
            p.grad.detach()
            for p in self.model.parameters()
            if p.grad is not None
        ]
        if not grads:
            return

        stacked_norms = torch.stack([g.norm(2) for g in grads])
        total_norm = torch.norm(stacked_norms, 2)
        self.log(
            "grad_norm",
            total_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
        )

    @torch.no_grad()
    def generate_text(
        self,
        prompts: Union[str, List[str]],
        max_length: Optional[int] = 50,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
    ) -> List[str]:
        """
        Generate text continuations for a single prompt or a list of prompts.
        """
        is_single = isinstance(prompts, str)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        eos_token_id = self.tokenizer.eos_token_id

        resolved_max_length = max_length if max_length is not None else 50
        if self.generation_kwargs and max_length is None and "max_length" in self.generation_kwargs:
            resolved_max_length = int(self.generation_kwargs["max_length"])

        resolved_num_beams = num_beams
        if self.generation_kwargs and num_beams is None and "num_beams" in self.generation_kwargs:
            resolved_num_beams = int(self.generation_kwargs["num_beams"])

        resolved_do_sample = do_sample
        if self.generation_kwargs and do_sample is None and "do_sample" in self.generation_kwargs:
            resolved_do_sample = bool(self.generation_kwargs["do_sample"])

        resolved_temperature = temperature
        if self.generation_kwargs and temperature is None and "temperature" in self.generation_kwargs:
            resolved_temperature = float(self.generation_kwargs["temperature"])

        resolved_top_k = top_k
        if self.generation_kwargs and top_k is None and "top_k" in self.generation_kwargs:
            resolved_top_k = int(self.generation_kwargs["top_k"])

        resolved_top_p = top_p
        if self.generation_kwargs and top_p is None and "top_p" in self.generation_kwargs:
            resolved_top_p = float(self.generation_kwargs["top_p"])

        resolved_repetition_penalty = repetition_penalty
        if (
            self.generation_kwargs
            and repetition_penalty is None
            and "repetition_penalty" in self.generation_kwargs
        ):
            resolved_repetition_penalty = float(self.generation_kwargs["repetition_penalty"])

        resolved_no_repeat_ngram_size = no_repeat_ngram_size
        if (
            self.generation_kwargs
            and no_repeat_ngram_size is None
            and "no_repeat_ngram_size" in self.generation_kwargs
        ):
            resolved_no_repeat_ngram_size = int(self.generation_kwargs["no_repeat_ngram_size"])

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"] if "attention_mask" in inputs else None

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=resolved_max_length,
            num_beams=resolved_num_beams,
            do_sample=resolved_do_sample,
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            top_p=resolved_top_p,
            repetition_penalty=resolved_repetition_penalty,
            no_repeat_ngram_size=resolved_no_repeat_ngram_size,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        return decoded if not is_single else decoded[0]
