from typing import Optional, Tuple

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


class GPT2(nn.Module):
    def __init__(self, model_name: str = "gpt2", cache_dir: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model: GPT2LMHeadModel
        self.tokenizer: GPT2TokenizerFast
        self.model, self.tokenizer = self.load_from_hub(model_name=model_name, cache_dir=cache_dir)

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
