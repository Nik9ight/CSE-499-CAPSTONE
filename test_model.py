import argparse
import inspect
import json
import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

import Config as config
from Load_Dataset import ImageToImage2D, get_val_transforms
from nets.LViT_new import LViTN
from nets.routing import ROUTING_INTERVENTIONS
from model_settings import normalize_router_config


def tta_inference(model, image, text, sizes=None, routing_override=None):
    """Multi-scale + flip TTA at native resolutions.

    For each size in `sizes`, resize the input to (size, size), run 3 flip variants,
    resize predicted probabilities back to the original (H, W), then average.
    Total: len(sizes) * 3 forward passes. Defaults to Config.ms_tta_sizes.
    """
    if sizes is None:
        sizes = config.ms_tta_sizes if config.use_ms_tta else (image.shape[-1],)
    H, W = image.shape[-2:]
    probs = []
    forward_options = {"routing_override": routing_override} if routing_override is not None else {}
    with torch.no_grad():
        for s in sizes:
            if (s, s) == (H, W):
                x_scaled = image
            else:
                x_scaled = torch.nn.functional.interpolate(
                    image, size=(s, s), mode="bilinear", align_corners=False,
                )
            for flip_dims in ((), (3,), (2,)):
                x_in = torch.flip(x_scaled, dims=flip_dims) if flip_dims else x_scaled
                logits = model(x_in, text, **forward_options)["main"]
                p = torch.sigmoid(logits)
                if flip_dims:
                    p = torch.flip(p, dims=flip_dims)
                if p.shape[-2:] != (H, W):
                    p = torch.nn.functional.interpolate(
                        p, size=(H, W), mode="bilinear", align_corners=False,
                    )
                probs.append(p)
    return torch.stack(probs, dim=0).mean(dim=0)


def load_model(device, checkpoint_path=None, model_kwargs=None):
    """Load trusted Lightning or legacy weights with an explicit architecture."""
    checkpoint_path = checkpoint_path or getattr(config, "test_checkpoint", None)
    if not checkpoint_path:
        raise ValueError("Pass --checkpoint PATH or set Config.test_checkpoint.")
    checkpoint_path = os.fspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_kwargs = checkpoint.get("hyper_parameters", {}).get("model_kwargs")
    if saved_kwargs is None:
        saved_kwargs = checkpoint.get("model_kwargs")
    if saved_kwargs is None and model_kwargs is None:
        raise ValueError(
            "This checkpoint has no architecture snapshot. Pass --model-config "
            "with the LViTN constructor settings used during training."
        )
    resolved = dict(saved_kwargs if saved_kwargs is not None else model_kwargs)
    # Pre-router snapshots omit this field and must retain the legacy guidance path.
    resolved["router_config"] = normalize_router_config(resolved.get("router_config"))
    # JSON represents tuples as lists; normalize before comparing snapshots.
    if "aspp_dilations" in resolved:
        resolved["aspp_dilations"] = tuple(resolved["aspp_dilations"])
    if saved_kwargs is not None and model_kwargs is not None:
        override = dict(model_kwargs)
        override["router_config"] = normalize_router_config(override.get("router_config"))
        if "aspp_dilations" in override:
            override["aspp_dilations"] = tuple(override["aspp_dilations"])
        if override != resolved:
            raise ValueError("Explicit architecture conflicts with the checkpoint snapshot.")
    required = set(inspect.signature(LViTN.__init__).parameters) - {
        "self", "config", "backbone_pretrained",
    }
    missing = required - resolved.keys()
    if missing:
        raise ValueError(f"Incomplete architecture snapshot; missing: {sorted(missing)}")

    model = LViTN(None, backbone_pretrained=False, **resolved)
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("model.") for key in state):
        state = {key[len("model."):]: value for key, value in state.items()
                 if key.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint_path


def evaluate(checkpoint_path=None, model_kwargs=None, routing_override=None, routing_seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = getattr(config, "use_tta", True)
    if routing_override == "shuffle":
        torch.manual_seed(routing_seed)

    model, ckpt_path = load_model(device, checkpoint_path, model_kwargs)
    model.eval()
    if routing_override is not None and model.router_config["mode"] == "off":
        raise ValueError("Routing interventions require a router checkpoint.")
    forward_options = {"routing_override": routing_override} if routing_override is not None else {}

    # Load swept threshold from the same session dir as the loaded checkpoint
    # (sibling to `models/`), not the current Config's session dir.
    threshold_path = os.path.abspath(
        os.path.join(os.path.dirname(ckpt_path), "..", "best_threshold.json")
    )
    pred_threshold = 0.5
    tta_sizes = config.ms_tta_sizes if config.use_ms_tta else (model.img_size,)
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            data = json.load(f)
        pred_threshold = float(data.get("threshold", 0.5))
        use_tta = bool(data.get("use_tta", use_tta))
        tta_sizes = tuple(data.get("tta_sizes", tta_sizes))
        print(f"[test] loaded threshold {pred_threshold:.2f} from {threshold_path}")
    else:
        print(f"[test] no threshold file at {threshold_path}, using {pred_threshold}")

    print("Model loaded.")

    val_tf = get_val_transforms(model.img_size)
    test_dataset = ImageToImage2D(
        config.test_dataset,
        joint_transform=val_tf,
        image_size=model.img_size,
        text_embed_dim=model.text_embed_dim,
        text_csv_path=config.test_text_csv,
        text_encoder_name=model.text_encoder_name,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    dice_metric_pixel = BinaryF1Score().to(device)
    iou_metric_pixel = BinaryJaccardIndex().to(device)

    image_dice_list = []
    image_iou_list = []

    vis_path = f"./test_predictions/{config.task_name}/{config.session_name}/"
    if routing_override is not None:
        intervention_name = f"shuffle_seed{routing_seed}" if routing_override == "shuffle" else routing_override
        vis_path = os.path.join(vis_path, intervention_name)
        print(f"[test] routing intervention: {routing_override}; retaining the calibrated threshold")
    os.makedirs(vis_path, exist_ok=True)

    with tqdm(total=len(test_loader), desc="Testing", ncols=70) as pbar:
        for sampled_batch, names in test_loader:
            images = sampled_batch["image"].to(device)
            masks = sampled_batch["label"].to(device)

            text = None
            if "input_ids" in sampled_batch and "attention_mask" in sampled_batch:
                text = {
                    "input_ids": sampled_batch["input_ids"].to(device),
                    "attention_mask": sampled_batch["attention_mask"].to(device),
                }
            elif "text" in sampled_batch:
                text = sampled_batch["text"].to(device)

            if use_tta:
                prob_avg = tta_inference(model, images, text, sizes=tta_sizes, **forward_options)
            else:
                with torch.no_grad():
                    prob_avg = torch.sigmoid(model(images, text, **forward_options)["main"])

            preds_bin = (prob_avg > pred_threshold).long()

            # Per-pixel (global) update
            dice_metric_pixel.update(preds_bin.view(-1), masks.long().view(-1))
            iou_metric_pixel.update(preds_bin.view(-1), masks.long().view(-1))

            # Per-image (macro) calculation
            preds_flat = preds_bin.view(preds_bin.shape[0], -1)
            masks_flat = masks.long().view(masks.shape[0], -1)
            inter = (preds_flat & masks_flat).sum(dim=1).float()
            union = (preds_flat | masks_flat).sum(dim=1).float()
            card = (preds_flat + masks_flat).sum(dim=1).float()

            dice_scores = torch.where(card == 0, torch.tensor(1.0, device=device), 2.0 * inter / card)
            iou_scores = torch.where(union == 0, torch.tensor(1.0, device=device), inter / union)

            image_dice_list.append(dice_scores.cpu())
            image_iou_list.append(iou_scores.cpu())

            for i in range(preds_bin.shape[0]):
                pred_np = preds_bin[i, 0].cpu().numpy().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(vis_path, names[i]), pred_np)

            pbar.update(1)

    macro_dice = torch.cat(image_dice_list).mean().item() if image_dice_list else 0.0
    macro_iou = torch.cat(image_iou_list).mean().item() if image_iou_list else 0.0

    print(f"\n{'='*45}")
    print(f" Test Dice (Per-Pixel / Global): {dice_metric_pixel.compute().item()}")
    print(f" Test Dice (Per-Image / Macro):  {macro_dice}")
    print(f" Test IoU  (Per-Pixel / Global): {iou_metric_pixel.compute().item()}")
    print(f" Test IoU  (Per-Image / Macro):  {macro_iou}")
    print(f"{'='*45}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a text-guided segmentation checkpoint.")
    parser.add_argument("--checkpoint", help="Checkpoint path (or set Config.test_checkpoint).")
    parser.add_argument("--model-config", help="Architecture JSON for checkpoints without a snapshot.")
    parser.add_argument("--routing-override", choices=ROUTING_INTERVENTIONS,
                        help="Evaluation-only intervention; keep the checkpoint's calibrated threshold.")
    parser.add_argument("--routing-seed", type=int, default=0, help="Seed for shuffled routing interventions.")
    args = parser.parse_args()
    architecture = None
    if args.model_config:
        with open(args.model_config, encoding="utf-8") as f:
            architecture = json.load(f)
    evaluate(args.checkpoint, architecture, args.routing_override, args.routing_seed)
