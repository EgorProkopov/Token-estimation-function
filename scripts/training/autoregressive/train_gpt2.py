import os
import dotenv

from omegaconf import DictConfig, OmegaConf

from scripts.training.autoregressive.train import prepare_dataloaders, train_autoregressive
from src.lightning_modules.gpt2 import GPT2LightningModule


def train_gpt2(
    model_config: DictConfig,
    training_config: DictConfig,
):
    generation_kwargs = model_config.get("generation", None)

    training_hparams = training_config["training"]

    model = GPT2LightningModule(
        model_name=model_config["model_name"],
        cache_dir=model_config.get("cache_dir", None),
        lr=training_hparams["lr"],
        log_step=training_hparams.get("log_step", training_config["logging"]["log_every_n_steps"]),
        max_epochs=training_config["training"]["max_epochs"],
        warmup_steps=training_hparams.get("warmup_steps", 0),
        lr_gamma=training_hparams.get("lr_gamma", 0.80),
        weight_decay=training_hparams.get("weight_decay", 0.0),
        generation_kwargs=generation_kwargs,
        compile_model=model_config.get("compile_model", False),
    )

    dataset_config = training_config["dataset"]
    train_dataloader, val_dataloader = prepare_dataloaders(
        tokenizer=model.tokenizer,
        block_size=dataset_config["block_size"],
        train_batch_size=dataset_config["train_batch_size"],
        val_batch_size=dataset_config["val_batch_size"],
        num_workers=dataset_config["num_workers"],
        dataset_name=dataset_config["name"],
        dataset_config=dataset_config["config"],
        cache_dir=dataset_config.get("cache_dir", None),
    )

    train_autoregressive(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_config=training_config,
        clearml_configs={
            "model_config": model_config,
            "training_config": training_config,
        },
    )


if __name__ == "__main__":
    dotenv.load_dotenv()

    configs_dir = os.getenv("CONFIGS_DIR")
    model_config = OmegaConf.load(os.path.join(configs_dir, "models", "gpt2.yaml"))
    training_config = OmegaConf.load(os.path.join(configs_dir, "training", "gpt2.yaml"))

    train_gpt2(
        model_config=model_config,
        training_config=training_config,
    )
