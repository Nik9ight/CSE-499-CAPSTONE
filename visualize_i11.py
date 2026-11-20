"""Select the best validation-scoring I11 seed and export GRACE XAI figures."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

from cumulative_experiments import DATASETS, SEEDS
from experiment_io import atomic_json


SCORE_KEY = "best_val_global_dice"
SOURCE_LABELS = ("Text 1 (layers 1-3)", "Text 2 (layers 4-6)",
                 "Text 3 (layers 7-9)", "Text 4 (layers 10-12)",
                 "Previous stage", "No update")


def read_json(path: Path) -> dict:
    """Read required run metadata; never infer scores from checkpoint filenames."""
    if not path.is_file():
        raise FileNotFoundError(f"Required cumulative-run metadata is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_best_i11(study_root: Path, dataset: str) -> dict:
    """Compare all three completed training runs using validation Dice only."""
    root = Path(study_root).resolve()
    candidates = []
    for seed in SEEDS:
        run_dir = root / dataset / "I11" / f"seed_{seed}"
        spec = read_json(run_dir / "spec.json")
        if (spec.get("dataset"), spec.get("experiment"), spec.get("seed")) != (dataset, "I11", seed):
            raise ValueError(f"Run identity does not match its directory: {run_dir}")
        result = read_json(run_dir / "train_result.json")
        score = result.get(SCORE_KEY)
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid {SCORE_KEY} in {run_dir}: {score!r}")
        recorded = result.get("checkpoint")
        if not isinstance(recorded, str) or not recorded:
            raise ValueError(f"Missing checkpoint path in {run_dir / 'train_result.json'}")
        candidates.append({"seed": seed, SCORE_KEY: score, "best_epoch": result.get("best_epoch"),
                           "run_dir": str(run_dir), "recorded_checkpoint": recorded})
    # Lower seed wins exact ties, independent of directory traversal order.
    winner = max(candidates, key=lambda row: (row[SCORE_KEY], -row["seed"]))
    recorded = winner["recorded_checkpoint"]
    # Prefer the copied study's checkpoint, supporting Linux-to-Windows transfers.
    basename = recorded.replace("\\", "/").rsplit("/", 1)[-1]
    if not basename.startswith("best") or not basename.endswith(".ckpt"):
        raise ValueError(f"Expected the cumulative runner's best*.ckpt, got {recorded!r}")
    checkpoint = Path(winner["run_dir"]) / "models" / basename
    if not checkpoint.is_file():
        checkpoint = Path(recorded)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"The winning seed {winner['seed']} checkpoint is missing. Copy {basename} into "
            f"{Path(winner['run_dir']) / 'models'}. Selection will not fall back to another seed."
        )
    return {"dataset": dataset, "experiment": "I11", "selection_metric": SCORE_KEY,
            "tie_break": "lowest seed", "candidates": candidates, **winner,
            "checkpoint": str(checkpoint.resolve())}


def select_case_indices(names: list[str], cases: list[str] | None, limit: int) -> list[int]:
    """Use requested filenames or the first sorted cases, never rank test outcomes."""
    if limit < 0:
        raise ValueError("--num-cases must be zero (all) or positive.")
    if not names:
        raise ValueError("The visualization split is empty.")
    if cases:
        if len(set(cases)) != len(cases):
            raise ValueError("--cases contains duplicate filenames.")
        missing = sorted(set(cases) - set(names))
        if missing:
            raise ValueError(f"Requested cases are absent from the dataset: {missing}")
        positions = {name: i for i, name in enumerate(names)}
        return [positions[name] for name in cases]
    indices = sorted(range(len(names)), key=lambda i: names[i])
    return indices[:limit] if limit else indices


def explain_case(model, image, text) -> dict:
    """Compute decoder1 Seg-Grad-CAM and routing diagnostics for one image.

    The score is the mean foreground logit inside the detached prediction (>0.5).
    For an empty prediction, target the maximum-logit pixel and label the fallback.
    Making the decoder activation a leaf avoids retaining the encoder's backward
    graph when model parameters are frozen. Its values and prediction are unchanged.
    """
    import torch
    import torch.nn.functional as functional

    if model.training:
        raise ValueError("XAI requires model.eval().")
    if image.shape[0] != 1:
        raise ValueError("XAI processes one image at a time.")
    if model.router_config["mode"] != "region":
        raise ValueError("I11 XAI requires a regional-router checkpoint.")
    captured = {}

    def capture(_module, _inputs, output):
        activation = output.detach().requires_grad_(True)
        captured["activation"] = activation
        return activation

    handle = model.decoder1.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            output = model(image, text, return_routing=True)
            logits = output["main"]
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits.")
            probability = logits.detach().sigmoid()
            target = probability > 0.5
            target_mode = "predicted_foreground"
            if not target.any():
                target = torch.zeros_like(target)
                target.view(-1)[logits.detach().reshape(-1).argmax()] = True
                target_mode = "max_logit_pixel_empty_prediction"
            score = logits[target].mean()
            activation = captured["activation"]
            gradient, = torch.autograd.grad(score, activation)
            if not torch.isfinite(gradient).all():
                raise ValueError("Grad-CAM produced non-finite gradients.")
            weights = gradient.mean(dim=(-2, -1), keepdim=True)
            raw = (weights * activation.detach()).sum(dim=1, keepdim=True).relu()
            raw = functional.interpolate(raw, size=image.shape[-2:], mode="bilinear", align_corners=False)
            maximum = float(raw.max())
            cam = raw / maximum if maximum > 0 else torch.zeros_like(raw)
            routing = {stage: {key: tensor.detach().cpu() for key, tensor in maps.items()}
                       for stage, maps in output["routing"].items()}
            if any(not torch.isfinite(tensor).all() for maps in routing.values() for tensor in maps.values()):
                raise ValueError("Router diagnostics contain non-finite values.")
            return {"probability": probability[0, 0].cpu().numpy(),
                    "cam": cam[0, 0].detach().cpu().numpy(),
                    "cam_raw": raw[0, 0].detach().cpu().numpy(),
                    "target_mask": target[0, 0].cpu().numpy(), "routing": routing,
                    "target_mode": target_mode, "target_pixels": int(target.sum()),
                    "target_score": float(score.detach()), "cam_raw_max": maximum,
                    "cam_status": "ok" if maximum > 0 else "no_positive_cam_response"}
    finally:
        handle.remove()


def binary_metrics(prediction, truth) -> dict:
    """Match native evaluation's per-case Dice/IoU, including empty pairs."""
    import numpy as np

    prediction, truth = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    intersection = int((prediction & truth).sum())
    cardinality = int(prediction.sum() + truth.sum())
    union = int((prediction | truth).sum())
    return {"dice": 2 * intersection / cardinality if cardinality else 1.0,
            "iou": intersection / union if union else 1.0,
            "pred_area": int(prediction.sum()), "gt_area": int(truth.sum())}


def save_figures(directory: Path, rgb, truth, result: dict, title: str, prompt: str,
                 metrics: dict, dpi: int = 150) -> None:
    """Export a summary and four stage figures with explicit, shared color scales."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    probability, cam = result["probability"], result["cam"]
    prediction = probability > 0.5
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout="constrained")
    titles = ("Preprocessed image", "Ground truth overlay", "Prediction overlay (>0.5)",
              "Foreground probability", "Seg-Grad-CAM: decoder1", "Segmentation errors")
    for ax, caption in zip(axes.flat, titles):
        ax.imshow(rgb)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")
    for ax, mask in ((axes[0, 1], truth), (axes[0, 2], prediction)):
        color = np.zeros((*mask.shape, 4), dtype=np.float32)
        color[mask] = (0.1, 1.0, 0.4, 0.45)
        ax.imshow(color)
    im = axes[1, 0].imshow(probability, cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, label="Probability")
    im = axes[1, 1].imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.55)
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, label="Normalized positive CAM")
    errors = np.zeros((*truth.shape, 4), dtype=np.float32)
    errors[prediction & truth] = (0.1, 0.9, 0.3, 0.65)
    errors[prediction & ~truth] = (1.0, 0.15, 0.15, 0.75)
    errors[~prediction & truth] = (0.1, 0.45, 1.0, 0.75)
    axes[1, 2].imshow(errors)
    axes[1, 2].legend(handles=[Patch(color=c, label=l) for c, l in
                             (("limegreen", "TP"), ("red", "FP"), ("dodgerblue", "FN"))],
                      loc="lower center", ncol=3, fontsize=8)
    target_label = ("predicted foreground" if result["target_mode"] == "predicted_foreground"
                    else "maximum-logit pixel (empty prediction)")
    caption = f"{title}\nDice {metrics['dice']:.4f} | IoU {metrics['iou']:.4f} | CAM target: {target_label}"
    if result["cam_status"] != "ok":
        caption += "\nNo positive CAM response; heatmap is zero."
    if prompt:
        caption += "\nPrompt: " + textwrap.fill(prompt, width=115, max_lines=3, placeholder=" ...")
    fig.suptitle(caption, fontsize=11)
    fig.savefig(directory / "summary.png", dpi=dpi)
    fig.savefig(directory / "summary.pdf")
    plt.close(fig)

    for stage, maps in result["routing"].items():
        weights = maps["weights"][0].numpy()
        norms = maps["contribution_norms"][0].numpy()
        blocks = weights.shape[0]
        norm_max = max(float(norms.max()), 1e-12)
        fig, axes = plt.subplots(2 * blocks, 6, figsize=(17, 4.5 * blocks),
                                 squeeze=False, layout="constrained")
        for block in range(blocks):
            for source, label in enumerate(SOURCE_LABELS):
                for kind, values, cmap, vmax in ((0, weights, "magma", 1), (1, norms, "cividis", norm_max)):
                    ax = axes[2 * block + kind, source]
                    im = ax.imshow(values[block, :, :, source], cmap=cmap, vmin=0, vmax=vmax,
                                   interpolation="nearest")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if block == 0 and kind == 0:
                        ax.set_title(label, fontsize=10)
                    if source == 0:
                        ax.set_ylabel(f"Block {block + 1}\n" + ("Weight" if kind == 0 else "Update L2 norm"))
                    if source == 5:
                        fig.colorbar(im, ax=list(axes[2 * block + kind]), shrink=0.8, pad=0.01)
        fig.suptitle(f"{title} | {stage} ({weights.shape[1]} x {weights.shape[2]})\n"
                     "Weights: fixed 0-1 scale. Update norms: shared scale within this stage.\n"
                     "Text groups are BERT layer depths. No update always has zero contribution.", fontsize=11)
        fig.savefig(directory / f"routing_{stage}.png", dpi=dpi)
        plt.close(fig)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_visualizations(args, selection: dict, dataset_dir: Path, output_dir: Path) -> dict:
    """Load the selected snapshot and export deterministic, sequential case diagnostics."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"Output directory is not empty; choose a new --output-dir: {output_dir}")
    import cv2
    import matplotlib
    import numpy as np
    import torch
    from Load_Dataset import ImageToImage2D, IMAGENET_MEAN, IMAGENET_STD
    from nets.routing import ROUTING_SOURCES
    from test_model import load_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(selection["seed"])
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model, _ = load_model(device, selection["checkpoint"])
    if model.router_config["mode"] != "region" or not model.use_text:
        raise ValueError("Selected checkpoint is not a text-guided regional-router model.")
    model.float().eval().requires_grad_(False)
    prompt_path = dataset_dir / "prompt" / f"{args.split}.csv"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt CSV not found: {prompt_path}")
    dataset = ImageToImage2D(
        str(dataset_dir / args.split.title()), image_size=model.img_size,
        text_embed_dim=model.text_embed_dim, text_csv_path=str(prompt_path),
        text_encoder_name=model.text_encoder_name, use_text=True,
    )
    indices = select_case_indices(dataset.images_list, args.cases, args.num_cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {**selection, "status": "running", "split": args.split,
                "dataset_dir": str(dataset_dir), "output_dir": str(output_dir),
                "checkpoint_sha256": file_sha256(Path(selection["checkpoint"])),
                "prompt_csv_sha256": file_sha256(prompt_path),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "protocol": "native", "image_size": model.img_size, "threshold": 0.5,
                "preprocessing": "Load_Dataset.get_val_transforms", "max_text_tokens": dataset.max_text_tokens,
                "use_tta": False, "precision": "float32", "batch_size": 1, "device": device,
                "cam_layer": "decoder1", "cam_score": "mean foreground logit within detached prediction",
                "empty_prediction_target": "maximum-logit pixel", "routing_sources": list(ROUTING_SOURCES),
                "case_selection": "explicit filenames" if args.cases else "first sorted filenames",
                "cases": [dataset.images_list[i] for i in indices],
                "versions": {"torch": str(torch.__version__), "numpy": np.__version__,
                             "matplotlib": matplotlib.__version__, "opencv": cv2.__version__},
                "interpretation": "CAM and routing maps are diagnostics, not causal proof. "
                                  "CAM is normalized per case. Router norms are not output attribution."}
    atomic_json(output_dir / "selection.json", metadata)
    rows = []
    try:
        for number, index in enumerate(indices, 1):
            sample, name = dataset[index]
            image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
            # MONAI MetaTensor metadata is irrelevant to autograd attribution/export.
            if hasattr(image, "as_tensor"):
                image = image.as_tensor()
            text = {key: sample[key].unsqueeze(0).to(device) for key in ("input_ids", "attention_mask")}
            result = explain_case(model, image, text)
            truth = sample["label"][0].cpu().numpy() > 0.5
            rgb = image[0].detach().cpu().permute(1, 2, 0).numpy()
            rgb = np.clip(rgb * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN), 0, 1)
            prediction = result["probability"] > 0.5
            metrics = binary_metrics(prediction, truth)
            stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).stem)[:80]
            case_dir = output_dir / f"{number:04d}_{stem}"
            case_dir.mkdir()
            prompt = dataset.img_to_text[name]
            save_figures(case_dir, rgb, truth, result,
                         f"GRACE I11 | {selection['dataset']} | seed {selection['seed']} | {name}",
                         prompt, metrics, args.dpi)
            if not cv2.imwrite(str(case_dir / "prediction.png"), prediction.astype(np.uint8) * 255):
                raise OSError(f"Could not write prediction mask to {case_dir}")
            arrays = {"probability": result["probability"], "grad_cam": result["cam"],
                      "grad_cam_raw": result["cam_raw"], "cam_target_mask": result["target_mask"],
                      "ground_truth": truth, "prediction": prediction,
                      "input_ids": text["input_ids"][0].cpu().numpy(),
                      "attention_mask": text["attention_mask"][0].cpu().numpy(),
                      "routing_sources": np.asarray(ROUTING_SOURCES)}
            for stage, maps in result["routing"].items():
                arrays.update({f"{stage}_{key}": tensor[0].numpy() for key, tensor in maps.items()})
            np.savez_compressed(case_dir / "maps.npz", **arrays)
            row = {"case_id": name, "directory": case_dir.name, **metrics,
                   **{key: result[key] for key in ("target_mode", "target_pixels", "target_score",
                                                  "cam_raw_max", "cam_status")}}
            atomic_json(case_dir / "case.json", {**row, "prompt": prompt, "cam_layer": "decoder1"})
            rows.append(row)
            print(f"[{number}/{len(indices)}] {name}: Dice={metrics['dice']:.4f}, "
                  f"IoU={metrics['iou']:.4f}, CAM={result['cam_status']}", flush=True)
        with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata.update(status="completed", exported_cases=len(rows))
    except Exception as error:
        metadata.update(status="failed", exported_cases=len(rows), error=str(error))
        raise
    finally:
        atomic_json(output_dir / "selection.json", metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True, help="Cumulative output root containing qata/ and MosMed/.")
    parser.add_argument("--dataset", choices=DATASETS, required=True, help="Select the best I11 seed within this dataset.")
    parser.add_argument("--data-root", type=Path, help="Root containing dataset folders; defaults to the winning spec's data_root.")
    parser.add_argument("--output-dir", type=Path, help="Empty destination; default: study-root/xai/dataset/seed_TIMESTAMP.")
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--num-cases", type=int, default=8, help="First sorted cases to visualize (default 8; 0 means all).")
    parser.add_argument("--cases", nargs="+", help="Exact image filenames; overrides --num-cases.")
    parser.add_argument("--device", help="Torch device, e.g. cuda, cuda:0, cpu; auto-detected by default.")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="Show selection without importing ML libraries or writing files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_cases < 0 or args.dpi < 1:
        parser.error("--num-cases must be >= 0 and --dpi must be >= 1.")
    selection = select_best_i11(args.study_root, args.dataset)
    spec = read_json(Path(selection["run_dir"]) / "spec.json")
    data_root = args.data_root or Path(spec["data_root"])
    dataset_dir = data_root.resolve() / args.dataset
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = (args.output_dir or args.study_root / "xai" / args.dataset /
                  f"seed_{selection['seed']}_{stamp}").resolve()
    for candidate in selection["candidates"]:
        print(f"I11 seed {candidate['seed']}: validation Dice = {candidate[SCORE_KEY]:.6f}")
    print(f"Selected seed {selection['seed']}: {selection['checkpoint']}")
    print(f"Data: {dataset_dir / args.split.title()}\nOutput: {output_dir}", flush=True)
    if args.dry_run:
        return
    export_visualizations(args, selection, dataset_dir, output_dir)
    print(f"Saved XAI figures and numeric maps to {output_dir}")


if __name__ == "__main__":
    main()
