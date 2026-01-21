import os
import dotenv

from omegaconf import DictConfig, OmegaConf

from scripts.training.autoregressive.train import prepare_dataloaders, train_autoregressive
from src.lightning_modules.gpt2_tef import GPT2TEFLightningModule


def train_gpt2_tef(
    model_config: DictConfig,
    training_config: DictConfig,
    tef_scorer_config: DictConfig,
    tef_aux_loss_config: DictConfig,
):
    generation_kwargs = model_config.get("generation", None)
    training_hparams = training_config["training"]

    tef_dropout = tef_scorer_config.get("dropout", 0.0)
    tef_cumulative_threshold = tef_scorer_config.get("cumulative_threshold", 0.95)
    tef_backend = tef_scorer_config.get("backend", "torch")
    tef_mask_warmup_steps = tef_scorer_config.get("mask_warmup_steps", 0)
    tef_disable_mask_on_val = tef_scorer_config.get("disable_mask_on_val", True)
    aux_alpha = tef_aux_loss_config.get("alpha", 0.0)

    model = GPT2TEFLightningModule(
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
        aux_alpha=aux_alpha,
        tef_dropout=tef_dropout,
        tef_cumulative_threshold=tef_cumulative_threshold,
        tef_backend=tef_backend,
        tef_mask_warmup_steps=tef_mask_warmup_steps,
        tef_disable_mask_on_val=tef_disable_mask_on_val,
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
            "tef_scorer_config": tef_scorer_config,
            "tef_aux_loss_config": tef_aux_loss_config,
        },
    )


if __name__ == "__main__":
    dotenv.load_dotenv()

    configs_dir = os.getenv("CONFIGS_DIR")
    model_config = OmegaConf.load(os.path.join(configs_dir, "models", "gpt2.yaml"))
    training_config = OmegaConf.load(os.path.join(configs_dir, "training", "gpt2_tef.yaml"))
    tef_scorer_config = OmegaConf.load(os.path.join(configs_dir, "tefs", "tef_scorer.yaml"))
    tef_aux_loss_config = OmegaConf.load(os.path.join(configs_dir, "tefs", "tef_aux_loss.yaml"))

    train_gpt2_tef(
        model_config=model_config,
        training_config=training_config,
        tef_scorer_config=tef_scorer_config,
        tef_aux_loss_config=tef_aux_loss_config,
    )
