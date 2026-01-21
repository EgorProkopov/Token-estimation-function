from typing import Optional, Tuple

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.tefs.tef_scorer import TEFScorer


class TEFTransformerBlock(nn.Module):
    """
    Lightweight wrapper over a GPT-2 block that prepends a TEF scorer.

    The scorer produces gates via sigmoid over token logits. During inference
    tokens with low explained variance are removed using a cumulative threshold.
    """

    def __init__(
        self,
        block: nn.Module,
        hidden_size: int,
        dropout: float = 0.0,
        cumulative_threshold: float = 0.95,
        backend: str = "torch",
        keep_mask_mode: str = "eval",
    ):
        super().__init__()
        self.block = block
        self.last_gates: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None
        self.keep_mask_mode = keep_mask_mode
        self.scorer = TEFScorer(
            hidden_size=hidden_size,
            dropout=dropout,
            cumulative_threshold=cumulative_threshold,
            backend=backend,
        )

    def set_keep_mask_mode(self, mode: str) -> None:
        """
        Controls when keep_mask is applied:
        - "off": never apply
        - "eval": apply only when not training (default behavior)
        - "train": always apply, including during training
        """
        self.keep_mask_mode = mode

    def _use_keep_mask(self) -> bool:
        if self.keep_mask_mode == "off":
            return False
        if self.keep_mask_mode == "train":
            return True
        # default/"eval"
        return not self.training

    @staticmethod
    def _token_mask_from_attention(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Convert transformer attention mask (possibly broadcasted) to a 2D token mask.
        """
        if attention_mask is None:
            return None

        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                return attention_mask
            return attention_mask > 0

        mask = attention_mask
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)

        if mask.dim() == 3:
            if torch.is_floating_point(mask):
                reduced = mask.max(dim=-2).values
                threshold = torch.finfo(mask.dtype).min / 2
                return reduced > threshold
            if mask.dtype == torch.bool:
                return mask.any(dim=-2)
            return mask.max(dim=-2).values > 0

        if torch.is_floating_point(mask):
            reduced = mask
            while reduced.dim() > 2:
                reduced = reduced.max(dim=-2).values
            threshold = torch.finfo(mask.dtype).min / 2
            return reduced > threshold
        if mask.dtype == torch.bool:
            reduced = mask
            while reduced.dim() > 2:
                reduced = reduced.any(dim=-2)
            return reduced
        reduced = mask
        while reduced.dim() > 2:
            reduced = reduced.max(dim=-2).values
        return reduced > 0

    @staticmethod
    def _apply_keep_mask_to_attention(
        attention_mask: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            keep = keep_mask[:, None, None, :].to(dtype)
            mask_value = torch.finfo(dtype).min
            return (1.0 - keep) * mask_value

        if attention_mask.dim() == 2:
            return attention_mask * keep_mask.to(attention_mask.dtype)

        broadcast_keep = keep_mask[:, None, None, :]
        if torch.is_floating_point(attention_mask):
            mask_value = torch.finfo(attention_mask.dtype).min
            return attention_mask.masked_fill(~broadcast_keep, mask_value)
        return attention_mask * broadcast_keep

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        token_mask = self._token_mask_from_attention(attention_mask)
        apply_keep_mask = self._use_keep_mask()
        logits, gates, keep_mask = self.scorer(
            hidden_states,
            attention_mask=token_mask,
            inference=apply_keep_mask,
        )
        self.last_logits = logits
        self.last_gates = gates

        gated_states = hidden_states * gates.unsqueeze(-1)
        updated_attention = attention_mask
        if apply_keep_mask and keep_mask is not None:
            gated_states = gated_states * keep_mask.unsqueeze(-1)
            updated_attention = self._apply_keep_mask_to_attention(
                attention_mask=attention_mask,
                keep_mask=keep_mask,
                dtype=hidden_states.dtype,
            )

        return self.block(
            hidden_states=gated_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=updated_attention,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class GPT2TEF(nn.Module):
    def __init__(
        self,
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
        tef_dropout: float = 0.0,
        tef_cumulative_threshold: float = 0.95,
        tef_backend: str = "torch",
        keep_mask_mode: str = "eval",
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.tef_backend = tef_backend
        self.keep_mask_mode = keep_mask_mode
        self.model: GPT2LMHeadModel
        self.tokenizer: GPT2TokenizerFast
        self.model, self.tokenizer = self.load_from_hub(model_name=model_name, cache_dir=cache_dir)
        self._attach_tef_scorers(
            dropout=tef_dropout,
            cumulative_threshold=tef_cumulative_threshold,
            keep_mask_mode=keep_mask_mode,
        )

    @staticmethod
    def load_from_hub(
        model_name: str = "gpt2",
        cache_dir: Optional[str] = None,
    ) -> Tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
        model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

        return model, tokenizer

    def _attach_tef_scorers(self, dropout: float, cumulative_threshold: float, keep_mask_mode: str) -> None:
        hidden_size = getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "n_embd")
        tef_blocks = []
        for block in self.model.transformer.h:
            tef_blocks.append(
                TEFTransformerBlock(
                    block=block,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    cumulative_threshold=cumulative_threshold,
                    backend=self.tef_backend,
                    keep_mask_mode=keep_mask_mode,
                )
            )
        self.model.transformer.h = nn.ModuleList(tef_blocks)

    def set_keep_mask_mode(self, mode: str) -> None:
        self.keep_mask_mode = mode
        for block in self.model.transformer.h:
            if isinstance(block, TEFTransformerBlock):
                block.set_keep_mask_mode(mode)

    def collect_gates(self):
        gates = []
        for block in self.model.transformer.h:
            if isinstance(block, TEFTransformerBlock) and block.last_gates is not None:
                gates.append(block.last_gates)
        return gates

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        resolved_pad_token_id = pad_token_id
        if resolved_pad_token_id is None:
            resolved_pad_token_id = self.tokenizer.pad_token_id
            if resolved_pad_token_id is None:
                resolved_pad_token_id = self.tokenizer.eos_token_id

        resolved_eos_token_id = eos_token_id
        if resolved_eos_token_id is None:
            resolved_eos_token_id = self.tokenizer.eos_token_id

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=resolved_pad_token_id,
            eos_token_id=resolved_eos_token_id,
        )


GPT2 = GPT2TEF
