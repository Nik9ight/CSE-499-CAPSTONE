# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import os
import cv2
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict, List

from monai.transforms import (
    Compose,
    ScaleIntensityd,
    Resized,
    RandRotate90d,
    RandFlipd,
    RandRotated,
    RandAdjustContrastd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGridDistortiond,
    NormalizeIntensityd,
    EnsureTyped,
)

import Config as config

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int) -> Compose:
    """Expects image as [C,H,W] float and label as [1,H,W] float (prepared in __getitem__)."""
    return Compose([
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="bilinear"),
        Resized(keys=["label"], spatial_size=(img_size, img_size), mode="nearest"),
        # Geometric augmentations (image + mask)
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandRotated(
            keys=["image", "label"],
            range_x=0.5236,  # ±30 degrees
            prob=0.25,
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
        ),
        # Intensity augmentations (image only)
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.02),
        RandGridDistortiond(keys=["image", "label"], prob=0.15, distort_limit=(-0.1, 0.1)),
        # Normalize with ImageNet stats
        NormalizeIntensityd(keys=["image"], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


def get_val_transforms(img_size: int) -> Compose:
    """Expects image as [C,H,W] float and label as [1,H,W] float (prepared in __getitem__)."""
    return Compose([
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        Resized(keys=["image"], spatial_size=(img_size, img_size), mode="bilinear"),
        Resized(keys=["label"], spatial_size=(img_size, img_size), mode="nearest"),
        NormalizeIntensityd(keys=["image"], subtrahend=IMAGENET_MEAN, divisor=IMAGENET_STD, channel_wise=True),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    ])


# ---------- Utilities ----------
def correct_dims(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    elif mask.ndim == 3:
        mask = mask[:, :, 0]

    return image, mask


def sorted_image_mask_pairs(img_dir: str, mask_dir: str) -> Tuple[List[str], List[str]]:
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    imgs = sorted([f for f in os.listdir(img_dir) if not f.startswith('.')])
    masks = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

    if len(imgs) == 0:
        raise ValueError(f"No images found in {img_dir}")
    if len(masks) == 0:
        raise ValueError(f"No masks found in {mask_dir}")

    mask_exact_map = {name: name for name in masks}
    mask_basename_map = {os.path.splitext(name)[0]: name for name in masks}

    mask_prefix_map = {}
    for mask_name in masks:
        mask_base = os.path.splitext(mask_name)[0]
        if mask_base.startswith("mask_"):
            original_name = mask_base[5:]
            mask_prefix_map[original_name] = mask_name

    paired_imgs = []
    paired_masks = []

    for im_name in imgs:
        if im_name in mask_exact_map:
            paired_imgs.append(im_name)
            paired_masks.append(mask_exact_map[im_name])
            continue

        base = os.path.splitext(im_name)[0]
        if base in mask_basename_map:
            paired_imgs.append(im_name)
            paired_masks.append(mask_basename_map[base])
            continue

        mask_with_prefix = f"mask_{im_name}"
        if mask_with_prefix in mask_exact_map:
            paired_imgs.append(im_name)
            paired_masks.append(mask_exact_map[mask_with_prefix])
            continue

        if base in mask_prefix_map:
            paired_imgs.append(im_name)
            paired_masks.append(mask_prefix_map[base])
            continue

        mask_base_with_prefix = f"mask_{base}"
        if mask_base_with_prefix in mask_basename_map:
            paired_imgs.append(im_name)
            paired_masks.append(mask_basename_map[mask_base_with_prefix])
            continue

        available_masks = list(mask_basename_map.keys())[:5]
        raise FileNotFoundError(
            f"No mask file found matching image '{im_name}'\n"
            f"Available mask basenames (first 5): {available_masks}"
        )

    return paired_imgs, paired_masks


# ---------- Dataset ----------
class ImageToImage2D(Dataset):
    def __init__(
        self,
        dataset_path: str,
        joint_transform: Optional[Compose] = None,
        one_hot_mask: int = 0,
        image_size: int = 224,
        max_text_tokens: int = 80,
        text_embed_dim: int = 768,
        text_csv_path: str = None,
        text_encoder_name: str = None,
    ):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.max_text_tokens = max_text_tokens
        self.text_embed_dim = text_embed_dim

        self.input_path = os.path.join(dataset_path, "Images")
        self.output_path = os.path.join(dataset_path, "GTs")

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Image directory not found: {self.input_path}")
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"Mask directory not found: {self.output_path}")

        self.images_list, self.mask_list = sorted_image_mask_pairs(
            self.input_path, self.output_path
        )

        self.one_hot_mask = one_hot_mask if one_hot_mask else 0
        self.joint_transform = joint_transform or get_val_transforms(image_size)

        # Text: store raw descriptions; tokenization runs per item (cheap, CPU).
        # The frozen BERT lives inside the model and runs once per batch on GPU,
        # producing per-layer hidden states for HierarchicalTextFusion.
        self.img_to_text: Dict[str, str] = {}
        if text_csv_path and os.path.exists(text_csv_path):
            df = pd.read_csv(text_csv_path)
            self.img_to_text = dict(
                zip(df["Image"].astype(str), df["Description"].astype(str))
            )

        from transformers import AutoTokenizer
        from bertem import _ensure_hf_model_local
        lib_dir = getattr(config, "lib_dir", "./lib/")
        encoder_name = text_encoder_name or config.text_encoder
        local_path = _ensure_hf_model_local(encoder_name, lib_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

    def __len__(self) -> int:
        return len(self.images_list)

    def _tokenize_text(self, image_filename: str):
        if image_filename not in self.img_to_text:
            raise KeyError(
                f"No text description for '{image_filename}'. "
                f"Check that text_csv_path contains an entry for this image."
            )
        enc = self.tokenizer(
            self.img_to_text[image_filename],
            padding="max_length",
            truncation=True,
            max_length=self.max_text_tokens,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], str]:
        image_filename = self.images_list[idx]
        mask_filename = self.mask_list[idx]

        image = cv2.imread(os.path.join(self.input_path, image_filename))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {os.path.join(self.input_path, image_filename)}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(self.output_path, mask_filename), 0)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {os.path.join(self.output_path, mask_filename)}")
        mask = (mask > 0).astype(np.uint8)

        image, mask = correct_dims(image, mask)

        # Convert to CHW float for MONAI transforms (they expect channel-first)
        image = image.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        mask = mask[np.newaxis].astype(np.float32)            # HW -> 1HW

        sample = {"image": image, "label": mask}
        sample = self.joint_transform(sample)

        # One-hot encoding if requested
        if self.one_hot_mask and self.one_hot_mask > 0:
            lbl = sample["label"]
            if lbl.ndim == 3 and lbl.shape[0] == 1:
                lbl = lbl.squeeze(0)
            lbl = lbl.long()
            oh = torch.nn.functional.one_hot(lbl, num_classes=self.one_hot_mask)
            sample["label"] = oh.permute(2, 0, 1).float()

        input_ids, attention_mask = self._tokenize_text(image_filename)
        sample["input_ids"] = input_ids
        sample["attention_mask"] = attention_mask

        return sample, image_filename
