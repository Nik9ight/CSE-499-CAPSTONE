import random
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import Config as config
from Load_Dataset import ImageToImage2D, get_train_transforms, get_val_transforms


class LViTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_transforms=None,
        val_transforms=None,
        batch_size=None,
        num_workers=None,
    ):
        super().__init__()
        self.train_transforms = train_transforms or get_train_transforms(config.img_size)
        self.val_transforms = val_transforms or get_val_transforms(config.img_size)
        self.batch_size = batch_size or config.batch_size
        self.num_workers = num_workers or config.num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = ImageToImage2D(
                config.train_dataset,
                joint_transform=self.train_transforms,
                image_size=config.img_size,
                text_embed_dim=config.text_embed_dim,
                text_csv_path=config.train_text_csv,
            )
            self.val_dataset = ImageToImage2D(
                config.val_dataset,
                joint_transform=self.val_transforms,
                image_size=config.img_size,
                text_embed_dim=config.text_embed_dim,
                text_csv_path=config.val_text_csv,
            )

        if stage == "test" or stage is None:
            self.test_dataset = ImageToImage2D(
                config.test_dataset,
                joint_transform=self.val_transforms,
                image_size=config.img_size,
                text_embed_dim=config.text_embed_dim,
                text_csv_path=config.test_text_csv,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            worker_init_fn=self._worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    @staticmethod
    def _worker_init_fn(worker_id):
        seed = config.seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
