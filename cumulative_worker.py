"""One isolated train or test process. Invoked by run_cumulative.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from cumulative_experiments import configure
from experiment_io import atomic_json


def shared_initialization(model, path):
    """Reuse per-seed, untrained common tensors, keyed by name AND shape.

    The first appearance of a tensor records its ordinary constructor initialization.
    Later variants reuse it. Pretrained encoders are excluded. The batch runner holds
    a study lock, so these files have one writer. This never loads trained weights.
    """
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = torch.load(path, map_location="cpu", weights_only=True) if path.exists() else {}
    changed = False
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(("backbone.", "text_encoder.")):
                continue
            key = (name, tuple(parameter.shape))
            if key in archive:
                parameter.copy_(archive[key])
            else:
                archive[key] = parameter.detach().cpu().clone()
                changed = True
            digest.update(repr(key).encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    if changed:
        fd, temporary = tempfile.mkstemp(prefix="init-", dir=path.parent)
        os.close(fd)
        try:
            torch.save(archive, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return digest.hexdigest()


def train(spec, *, accelerator="gpu"):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger
    import Config as config
    from callbacks import BoxMetricsCallback
    from data_module import LViTDataModule
    from lightning_module import LViTLightningModule

    started = time.monotonic()
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU and a CUDA-enabled PyTorch installation are required.")
    snapshot = configure(config, spec)
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "run_config.json", snapshot)
    pl.seed_everything(config.seed, workers=True)
    dm = LViTDataModule(batch_size=config.batch_size, num_workers=config.num_workers)
    dm.setup("fit")
    if len(dm.train_dataset) < 2 or len(dm.val_dataset) < 1:
        raise ValueError("Training requires at least two cases and a nonempty validation split.")
    if config.use_aspp and len(dm.train_dataset) % config.batch_size == 1:
        raise ValueError("The last training batch is a singleton; fix the split/sampler consistently "
                         "for the entire study before training ASPP variants.")
    model = LViTLightningModule(model_config=config.get_CTranS_config())
    init_hash = shared_initialization(model.model, spec["initialization_path"])
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "models", filename="best", monitor="val/dice", mode="max",
        save_top_k=1, save_last=True, auto_insert_metric_name=False,
    )
    early_stop = EarlyStopping(monitor="val/dice", mode="max",
                               patience=config.early_stopping_patience,
                               min_delta=config.early_stopping_min_delta)
    logger = CSVLogger(save_dir=str(run_dir), name="training", flush_logs_every_n_steps=20)
    logger.log_hyperparams(snapshot)
    trainer = pl.Trainer(
        accelerator=accelerator, devices=1, max_epochs=config.epochs,
        precision=config.precision, deterministic=config.deterministic,
        logger=logger, callbacks=[checkpoint, early_stop, LearningRateMonitor("epoch"),
                                  BoxMetricsCallback(config.logger_path)],
        log_every_n_steps=min(config.print_frequency, max(1, len(dm.train_dataloader()))),
        enable_progress_bar=False,
    )
    last = run_dir / "models" / "last.ckpt"
    trainer.fit(model, datamodule=dm, ckpt_path=str(last) if last.exists() else None)
    best = Path(checkpoint.best_model_path)
    if not checkpoint.best_model_path or not best.is_file():
        raise RuntimeError("Training ended without a best validation checkpoint.")
    saved = torch.load(best, map_location="cpu", weights_only=False)
    best_dice = float(checkpoint.best_model_score)
    result = {
        "checkpoint": str(best.resolve()), "best_epoch": int(saved["epoch"]) + 1,
        "epochs_completed": trainer.current_epoch, "optimizer_steps": trainer.global_step,
        "best_val_global_dice": best_dice,
        "best_val_global_iou": best_dice / (2.0 - best_dice),
        "train_seconds_last_attempt": time.monotonic() - started,
        "initialization_hash": init_hash, "train_cases": len(dm.train_dataset),
        "val_cases": len(dm.val_dataset), "torch_version": str(torch.__version__),
        "lightning_version": pl.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    # Completion is committed only after fit and validation-checkpoint verification.
    atomic_json(run_dir / "train_result.json", result)
    return result


def test(spec, *, device_name="cuda"):
    import Config as config
    from test_model import evaluate

    configure(config, spec)
    run_dir = Path(spec["run_dir"])
    training = json.loads((run_dir / "train_result.json").read_text())
    return evaluate(
        training["checkpoint"], protocol="native", batch_size=32,
        results_json=run_dir / "test_result.json", output_dir=run_dir / "predictions",
        save_predictions=spec["evaluation"]["save_predictions"], device_name=device_name,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "test"), required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    (train if args.stage == "train" else test)(spec)


if __name__ == "__main__":
    main()
