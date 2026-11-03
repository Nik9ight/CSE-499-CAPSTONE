"""Short fast_dev_run of the real train_model.py setup; writes no checkpoints or logs.

Usage: python tests/smoke_train.py [--experiment router_region] [--checkpointing] [--batch-size N]
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import pytorch_lightning as pl

import Config as config
from data_module import LViTDataModule
from experiments import EXPERIMENTS, configure_run
from lightning_module import LViTLightningModule


def main(experiment=None, batch_size=None, checkpointing=False, batches=2):
    configure_run(config, experiment, batch_size=batch_size, checkpointing=checkpointing)
    torch.multiprocessing.set_start_method('spawn', force=True)
    pl.seed_everything(config.seed, workers=True)

    dm = LViTDataModule(batch_size=config.batch_size, num_workers=config.num_workers)
    model = LViTLightningModule(model_config=config.get_CTranS_config())
    trainer = pl.Trainer(
        fast_dev_run=batches,
        accelerator="auto",
        devices="auto",
        precision=config.precision,
        deterministic=config.deterministic,
    )
    trainer.fit(model, dm)

    peak = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if torch.cuda.is_available() else "n/a"
    print(f"SMOKE OK experiment={config.experiment_name} task={config.task_name} "
          f"batch_size={config.batch_size} checkpointing={config.gradient_checkpointing} peak_gpu={peak}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a few train/val batches to check the pipeline end to end.")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--batches", type=int, default=2)
    main(**vars(parser.parse_args()))
