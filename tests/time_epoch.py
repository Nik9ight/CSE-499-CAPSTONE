"""Times one full training epoch plus validation with the real train_model.py settings.

Usage: python tests/time_epoch.py --task qata --experiment router_region --batch-size 32 --checkpointing
       add --frozen-backbone to time the first freeze_backbone_epochs epochs instead.
"""
import argparse
import json
import os
import statistics
import sys
import time
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
from check_priority0 import use_task


class EpochTimer(pl.Callback):
    def __init__(self):
        self.batch_ends = []

    @staticmethod
    def _now():
        torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_start = self._now()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_ends.append(self._now())

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_start = self._now()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_end = self._now()

    def on_train_epoch_end(self, trainer, pl_module):
        self.epoch_end = self._now()


def main(task, experiment, batch_size, checkpointing, frozen_backbone):
    torch.multiprocessing.set_start_method('spawn', force=True)
    use_task(task)
    configure_run(config, experiment, batch_size=batch_size, checkpointing=checkpointing)
    pl.seed_everything(config.seed, workers=True)

    dm = LViTDataModule(batch_size=config.batch_size, num_workers=config.num_workers)
    model = LViTLightningModule(
        model_config=config.get_CTranS_config(),
        freeze_backbone_epochs=1 if frozen_backbone else 0,
    )
    timer = EpochTimer()
    trainer = pl.Trainer(
        max_epochs=1, num_sanity_val_steps=0, precision=config.precision,
        accelerator="gpu", devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False,
        deterministic=config.deterministic, callbacks=[timer],
    )
    trainer.fit(model, dm)

    steps = len(timer.batch_ends)
    intervals = [b - a for a, b in zip(timer.batch_ends, timer.batch_ends[1:])][5:]
    result = {
        "task": task, "experiment": experiment, "batch_size": config.batch_size,
        "checkpointing": config.gradient_checkpointing, "backbone": "frozen" if frozen_backbone else "unfrozen",
        "train_steps": steps, "train_seconds": round(timer.batch_ends[-1] - timer.train_start, 1),
        "val_seconds": round(timer.val_end - timer.val_start, 1),
        "epoch_seconds": round(timer.epoch_end - timer.train_start, 1),
        "median_step_seconds": round(statistics.median(intervals), 3),
        "images_per_second": round(config.batch_size / statistics.median(intervals), 1),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }
    print(f"EPOCH {experiment} bs={result['batch_size']} ckpt={'on' if checkpointing else 'off'} "
          f"backbone={result['backbone']}: epoch {result['epoch_seconds'] / 60:.2f} min "
          f"(train {result['train_seconds']}s over {steps} steps, val {result['val_seconds']}s), "
          f"median step {result['median_step_seconds']}s = {result['images_per_second']} img/s")
    print("EPOCHJSON " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time one full training epoch and its validation pass.")
    parser.add_argument("--task", default=config.task_name)
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), default="router_region")
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--frozen-backbone", action="store_true",
                        help="Keep the backbone frozen, as in the first freeze_backbone_epochs epochs.")
    main(**vars(parser.parse_args()))
