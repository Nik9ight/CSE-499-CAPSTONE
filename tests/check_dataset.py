"""Checks every dataset under ./data/ against what Load_Dataset and Config expect.

Usage: python tests/check_dataset.py [task ...]   (defaults to every task folder in ./data/)
"""
import argparse
import collections
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import cv2
import numpy as np
import pandas as pd

from Load_Dataset import sorted_image_mask_pairs

SPLITS = (("Train", "train"), ("Val", "val"), ("Test", "test"))


def check_split(task, split, csv_name):
    split_dir = Path("data") / task / split
    img_dir, mask_dir = split_dir / "Images", split_dir / "GTs"
    csv_path = Path("data") / task / "prompt" / f"{csv_name}.csv"
    problems = []

    for path in (img_dir, mask_dir, csv_path):
        if not path.exists():
            return [f"missing {path}"], set()

    try:
        imgs, masks = sorted_image_mask_pairs(str(img_dir), str(mask_dir))
    except (FileNotFoundError, ValueError) as err:
        return [str(err)], set()
    orphans = set(os.listdir(mask_dir)) - set(masks)
    if orphans:
        problems.append(f"{len(orphans)} masks without an image, e.g. {sorted(orphans)[:3]}")

    df = pd.read_csv(csv_path)
    text = dict(zip(df["Image"].astype(str), df["Description"].astype(str)))
    no_text = [im for im in imgs if im not in text]
    if no_text:
        problems.append(f"{len(no_text)} images without a prompt, e.g. {no_text[:3]}")
    if len(df) != len(text):
        problems.append(f"{len(df) - len(text)} duplicate prompt rows")

    unreadable, mismatched, empty, soft = [], [], 0, 0
    shapes = collections.Counter()
    for im, m in zip(imgs, masks):
        image = cv2.imread(str(img_dir / im))
        mask = cv2.imread(str(mask_dir / m), 0)
        if image is None or mask is None:
            unreadable.append(im)
            continue
        shapes[image.shape] += 1
        if image.shape[:2] != mask.shape[:2]:
            mismatched.append(im)
        if not (mask > 0).any():
            empty += 1
        if ((mask > 0) & (mask < 255)).any():
            soft += 1
    if unreadable:
        problems.append(f"{len(unreadable)} unreadable files, e.g. {unreadable[:3]}")
    if mismatched:
        problems.append(f"{len(mismatched)} image/mask size mismatches, e.g. {mismatched[:3]}")

    print(f"  {split:5s} pairs={len(imgs)} prompts={len(text)} shapes={dict(shapes.most_common(2))} "
          f"empty_masks={empty} masks_with_gray_edges={soft}")
    return problems, set(imgs)


def main(tasks):
    failed = False
    for task in tasks:
        print(f"=== {task}")
        names = {}
        for split, csv_name in SPLITS:
            problems, names[split] = check_split(task, split, csv_name)
            for problem in problems:
                print(f"    FAIL {split}: {problem}")
            failed |= bool(problems)
        for a, b in (("Train", "Val"), ("Train", "Test"), ("Val", "Test")):
            shared = names[a] & names[b]
            if shared:
                print(f"    FAIL {a}/{b} share {len(shared)} images, e.g. {sorted(shared)[:3]}")
                failed = True
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    available = sorted(p.name for p in (ROOT / "data").iterdir() if (p / "Train").is_dir())
    parser = argparse.ArgumentParser(description="Validate dataset layout, pairing, prompts and files.")
    parser.add_argument("tasks", nargs="*", help=f"any of {available}")
    tasks = parser.parse_args().tasks or available
    unknown = sorted(set(tasks) - set(available))
    if unknown:
        parser.error(f"unknown task(s) {unknown}; choose from {available}")
    sys.exit(main(tasks))
