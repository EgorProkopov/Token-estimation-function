from typing import Optional, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class Wikitext103Dataset(Dataset):
    """
    Dataset that tokenizes and chunks WikiText-103 into fixed-length sequences for
    autoregressive language modeling.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        block_size: int = 1024,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.cache_dir = cache_dir

        original_max_length = getattr(self.tokenizer, "model_max_length", None)
        if original_max_length is not None and original_max_length < 1_000_000:
            self.tokenizer.model_max_length = 1_000_000

        hf_dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            cache_dir=cache_dir,
        )
        remove_columns = list(hf_dataset.column_names)

        def _tokenize(batch):
            texts = [text for text in batch.get("text", []) if text]
            return tokenizer(
                texts,
                return_attention_mask=False,
                add_special_tokens=False,
            )

        tokenized = hf_dataset.map(
            _tokenize,
            batched=True,
            remove_columns=remove_columns,
        )

        def _group_texts(batch):
            if "input_ids" not in batch:
                return {"input_ids": []}
            concatenated = []
            for ids in batch["input_ids"]:
                concatenated.extend(ids)
            total_length = (len(concatenated) // block_size) * block_size
            if total_length == 0:
                return {"input_ids": []}
            return {
                "input_ids": [
                    concatenated[i : i + block_size]
                    for i in range(0, total_length, block_size)
                ]
            }

        self.dataset = tokenized.map(
            _group_texts,
            batched=True,
            batch_size=1000,
        )

        if original_max_length is not None and original_max_length < 1_000_000:
            self.tokenizer.model_max_length = original_max_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
