"""Measures training data-loader throughput (augmentations on, no model) for several worker counts.

Usage: python tests/time_dataloader.py --task qata --batch-size 32 --workers 2 4 8
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch

import Config as config
from check_priority0 import use_task
from data_module import LViTDataModule


def main(task, batch_size, workers, batches):
    torch.multiprocessing.set_start_method('spawn', force=True)
    use_task(task)
    for count in workers:
        dm = LViTDataModule(batch_size=batch_size, num_workers=count)
        dm.setup("fit")
        batches_iter = iter(dm.train_dataloader())
        for _ in range(5):
            next(batches_iter)   # skip worker start-up
        start = time.perf_counter()
        for _ in range(batches):
            next(batches_iter)
        rate = batches * batch_size / (time.perf_counter() - start)
        print(f"train loader {task}, bs {batch_size}, num_workers={count}: {rate:.0f} img/s", flush=True)
        del batches_iter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time the training data loader without the model.")
    parser.add_argument("--task", default=config.task_name)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--workers", type=int, nargs="+", default=[config.num_workers])
    parser.add_argument("--batches", type=int, default=40)
    main(**vars(parser.parse_args()))
