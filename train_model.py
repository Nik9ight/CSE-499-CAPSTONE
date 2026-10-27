import argparse
import json
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

import Config as config
from lightning_module import LViTLightningModule
from data_module import LViTDataModule
from flop_counter import report_model_flops
from experiments import EXPERIMENTS, configure_run


def main(experiment=None, seed=None, batch_size=None, checkpointing=False,
         router_scale=None, router_temperature=None, dry_run=False):
    snapshot = configure_run(
        config, experiment, seed, batch_size, checkpointing, router_scale, router_temperature,
    )
    if dry_run:
        print(json.dumps(snapshot, indent=2))
        return
    os.makedirs(config.model_path, exist_ok=True)
    with open(os.path.join(config.save_path, "run_config.json"), "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    torch.multiprocessing.set_start_method('spawn', force=True)
    pl.seed_everything(config.seed, workers=True)

    dm = LViTDataModule(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model_config = config.get_CTranS_config()
    model = LViTLightningModule(
        model_config=model_config,
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=config.model_path,
        filename="best_model-LViT",
        monitor="val/dice",
        mode="max",
        save_top_k=1,
        verbose=False,
    )

    early_stop_cb = EarlyStopping(
        monitor="val/dice",
        patience=config.early_stopping_patience,
        mode="max",
        verbose=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    from callbacks import ThresholdSweepCallback, BoxMetricsCallback
    box_cb = BoxMetricsCallback(log_file=getattr(config, "logger_path", None))
    threshold_cb = ThresholdSweepCallback(
        save_dir=config.save_path,
        candidates=config.threshold_sweep,
        use_tta=config.use_tta,
        tta_sizes=config.ms_tta_sizes if config.use_ms_tta else (config.img_size,),
    )

    loggers = [
        TensorBoardLogger(save_dir=config.save_path, name="tensorboard_logs"),
        CSVLogger(save_dir=config.save_path, name="csv_logs"),
    ]
    if config.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        loggers.append(WandbLogger(
            project=config.wandb_project,
            name=config.session_name,
            save_dir=config.save_path,
            offline=config.wandb_offline,
        ))

    # Capture the full run config (architecture + data) in every logger's hparams.
    # The module already logs the resolved LR/schedule hparams; these are the rest.
    run_hparams = {
        "backbone": config.backbone,
        "text_encoder": config.text_encoder,
        "text_embed_dim": config.text_embed_dim,
        "img_size": config.img_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "n_channels": config.n_channels,
        "n_labels": config.n_labels,
        "vit_depth": config.vit_depth,
        "vit_heads": config.vit_heads,
        "vit_drop_rate": config.vit_drop_rate,
        "vit_drop_path_rate": config.vit_drop_path_rate,
        "gradient_checkpointing": config.gradient_checkpointing,
        "use_freq": getattr(config, "use_freq", False),
        "freq_sigma_init": getattr(config, "freq_sigma_init", 5.0),
        "use_prev_cross_attention": getattr(config, "use_prev_cross_attention", True),
        "use_adaptive_gated_fusion": getattr(config, "use_adaptive_gated_fusion", True),
        "precision": config.precision,
        "seed": config.seed,
        "task_name": config.task_name,
        "experiment": config.experiment_name,
        "router_config": config.router_config,
    }
    for lg in loggers:
        lg.log_hyperparams(run_hparams)

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices="auto",
        logger=loggers,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor, box_cb, threshold_cb],
        deterministic=config.deterministic,
        precision=config.precision,
        log_every_n_steps=config.print_frequency,
    )

    report_model_flops(model, config)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRACE or a joint-router ablation.")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--router-scale", type=float)
    parser.add_argument("--router-temperature", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without training or downloads.")
    main(**vars(parser.parse_args()))
