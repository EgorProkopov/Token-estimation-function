import os
import random
from typing import List, Optional

import dotenv
from omegaconf import OmegaConf

import torch

from src.lightning_modules.gpt2 import GPT2LightningModule


def _set_seed(seed: int) -> None:
    random.seed(seed)
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


def _read_prompts(prompts: Optional[List[str]], prompt_file: Optional[str], per_line: bool) -> List[str]:
    prompt_list: List[str] = []
    if prompts:
        prompt_list.extend(prompts)

    file_prompts: List[str] = []
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as handle:
            content = handle.read()
        if per_line:
            file_prompts.extend([line.strip() for line in content.splitlines() if line.strip()])
        else:
            stripped = content.strip()
            if stripped:
                file_prompts.append(stripped)

    prompts = prompt_list + file_prompts
    if not prompts:
        return ["Once upon a time,"]
    return prompts


def _build_model_from_config(model_config, training_config) -> GPT2LightningModule:
    training_hparams = training_config["training"]
    return GPT2LightningModule(
        model_name=model_config["model_name"],
        cache_dir=model_config.get("cache_dir", None),
        lr=training_hparams.get("lr", 5e-5),
        log_step=training_hparams.get("log_step", training_config["logging"]["log_every_n_steps"]),
        max_epochs=training_hparams.get("max_epochs", training_config["training"]["max_epochs"]),
        warmup_steps=training_hparams.get("warmup_steps", 0),
        lr_gamma=training_hparams.get("lr_gamma", 0.80),
        weight_decay=training_hparams.get("weight_decay", 0.0),
        generation_kwargs=model_config.get("generation", None),
        compile_model=model_config.get("compile_model", False),
    )


def main() -> None:
    dotenv.load_dotenv()

    base_dir = os.getcwd()
    configs_dir = os.getenv("CONFIGS_DIR") or "configs"
    eval_config_path = os.getenv("EVAL_CONFIG") or os.path.join(
        configs_dir,
        "evaluation",
        "gpt2_base_text_generation.yaml",
    )
    eval_config_path = _resolve_path(eval_config_path, base_dir)
    eval_config = OmegaConf.load(eval_config_path)

    model_config_path = _resolve_path(
        eval_config.get("model", {}).get("config_path", os.path.join(configs_dir, "models", "gpt2.yaml")),
        base_dir,
    )
    training_config_path = _resolve_path(
        eval_config.get("training", {}).get("config_path", os.path.join(configs_dir, "training", "gpt2.yaml")),
        base_dir,
    )

    model_config = OmegaConf.load(model_config_path)
    training_config = OmegaConf.load(training_config_path)

    evaluation = eval_config.get("evaluation", {})
    generation = eval_config.get("generation", {})

    checkpoint_path = _resolve_path(evaluation.get("checkpoint_path", None), base_dir)

    if checkpoint_path:
        model = GPT2LightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
        )
    else:
        model = _build_model_from_config(
            model_config=model_config,
            training_config=training_config,
        )

    device = _resolve_device(evaluation.get("device", "auto"))
    model.to(device)
    model.eval()

    _set_seed(int(evaluation.get("seed", 42)))

    prompt_file = _resolve_path(evaluation.get("prompt_file", None), base_dir)
    prompts = _read_prompts(
        prompts=evaluation.get("prompts", None),
        prompt_file=prompt_file,
        per_line=bool(evaluation.get("prompt_per_line", False)),
    )

    num_samples = int(evaluation.get("num_samples", 1))
    if num_samples > 1:
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * num_samples)
        prompts = expanded_prompts

    with torch.no_grad():
        outputs = model.generate_text(
            prompts=prompts,
            max_length=generation.get("max_length", None),
            num_beams=generation.get("num_beams", None),
            do_sample=generation.get("do_sample", None),
            temperature=generation.get("temperature", None),
            top_k=generation.get("top_k", None),
            top_p=generation.get("top_p", None),
            repetition_penalty=generation.get("repetition_penalty", None),
            no_repeat_ngram_size=generation.get("no_repeat_ngram_size", None),
        )

    if isinstance(outputs, str):
        outputs = [outputs]

    for idx, text in enumerate(outputs, start=1):
        if len(outputs) > 1:
            print(f"=== Sample {idx} ===")
        print(text)
        if idx < len(outputs):
            print()


if __name__ == "__main__":
    main()
