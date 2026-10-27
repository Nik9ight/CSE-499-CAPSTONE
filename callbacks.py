"""Project-local Lightning callbacks."""
import json
import os

import torch
import pytorch_lightning as pl


class BoxMetricsCallback(pl.Callback):
    """Prints a clean ASCII box summary at the end of each training epoch.

    Displays:
    - Epoch number
    - Train loss
    - Train dice
    - Validation dice
    - Validation loss
    - Number of early stopping epochs
    """

    def __init__(self, log_file: str | None = None):
        super().__init__()
        self.log_file = log_file

    def _get_metric(self, trainer: pl.Trainer, keys: list[str]) -> float | None:
        for k in keys:
            if k in trainer.callback_metrics:
                val = trainer.callback_metrics[k]
                if isinstance(val, torch.Tensor):
                    return val.detach().cpu().item()
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "sanity_checking", False):
            return

        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs

        train_loss = self._get_metric(trainer, ["train/loss_epoch", "train/loss"])
        train_dice = self._get_metric(trainer, ["train/dice", "train/dice_epoch"])
        val_loss = self._get_metric(trainer, ["val/loss", "val/loss_epoch"])
        val_dice = self._get_metric(trainer, ["val/dice", "val/dice_epoch"])

        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        train_dice_str = f"{train_dice:.4f}" if train_dice is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        val_dice_str = f"{val_dice:.4f}" if val_dice is not None else "N/A"

        # Early stopping info
        early_stop_str = "N/A"
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.EarlyStopping):
                wait = getattr(cb, "wait_count", 0)
                pat = getattr(cb, "patience", 0)
                best = getattr(cb, "best_score", None)
                if isinstance(best, torch.Tensor):
                    best = best.detach().cpu().item()
                
                if wait == 0 and best is not None:
                    try:
                        early_stop_str = f"0 / {pat} (Best: {float(best):.4f})"
                    except (ValueError, TypeError):
                        early_stop_str = f"0 / {pat} (Best)"
                elif wait == 0:
                    early_stop_str = f"0 / {pat} (Best)"
                else:
                    early_stop_str = f"{wait} / {pat}"
                break

        # Format ASCII Box
        width = 58
        line_sep = "+" + "-" * width + "+"
        header = f"Epoch: {current_epoch} / {max_epochs}"
        col1_loss = f"Train Loss : {train_loss_str}"
        col2_loss = f"Val Loss   : {val_loss_str}"
        col1_dice = f"Train Dice : {train_dice_str}"
        col2_dice = f"Val Dice   : {val_dice_str}"
        row_es = f"Early Stopping : {early_stop_str}"

        row_header_str = f"| {header:<{width - 2}} |"
        row_loss_str = f"| {col1_loss:<26} | {col2_loss:<27} |"
        row_dice_str = f"| {col1_dice:<26} | {col2_dice:<27} |"
        row_es_str = f"| {row_es:<{width - 2}} |"

        box_lines = [
            "",
            line_sep,
            row_header_str,
            line_sep,
            row_loss_str,
            row_dice_str,
            line_sep,
            row_es_str,
            line_sep,
            "",
        ]
        box_text = "\n".join(box_lines)
        print(box_text, flush=True)

        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(box_text + "\n")
            except Exception:
                pass


class ThresholdSweepCallback(pl.Callback):
    """After training, reload the best checkpoint and sweep prediction thresholds
    on the validation set. Persist the optimum to `<save_dir>/best_threshold.json`.

    Test-time inference should read this file and fall back to 0.5 if absent.
    """

    def __init__(self, save_dir: str, candidates=(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
                 use_tta: bool = True, tta_sizes=None):
        super().__init__()
        self.save_dir = save_dir
        self.candidates = tuple(candidates)
        self.use_tta = use_tta
        self.tta_sizes = tuple(tta_sizes) if tta_sizes is not None else None

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not trainer.is_global_zero:
            return
        # Find best-checkpoint path from the ModelCheckpoint callback
        best_ckpt = None
        for cb in trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint) and cb.best_model_path:
                best_ckpt = cb.best_model_path
                break

        device = pl_module.device
        if best_ckpt and os.path.exists(best_ckpt):
            print(f"[ThresholdSweep] loading best ckpt: {best_ckpt}")
            state = torch.load(best_ckpt, map_location=device, weights_only=False)
            pl_module.load_state_dict(state["state_dict"], strict=True)

        was_training = pl_module.training
        try:
            self._calibrate(trainer, pl_module)
        finally:
            pl_module.train(was_training)

    def _calibrate(self, trainer, pl_module):
        device = pl_module.device
        pl_module.eval()
        val_loader = trainer.datamodule.val_dataloader()

        all_probs, all_masks = [], []
        with torch.no_grad():
            for batch in val_loader:
                sampled_batch, _names = batch
                images = sampled_batch["image"].to(device)
                masks = sampled_batch["label"]
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
                text = None
                if "input_ids" in sampled_batch and "attention_mask" in sampled_batch:
                    text = {
                        "input_ids": sampled_batch["input_ids"].to(device),
                        "attention_mask": sampled_batch["attention_mask"].to(device),
                    }
                if self.use_tta:
                    from test_model import tta_inference
                    probs = tta_inference(pl_module.model, images, text, sizes=self.tta_sizes).cpu()
                else:
                    out = pl_module.model(images, text)
                    probs = torch.sigmoid(out["main"]).cpu()
                all_probs.append(probs)
                all_masks.append(masks.cpu())

        if not all_probs:
            raise ValueError("Threshold selection requires a nonempty validation dataset.")
        probs = torch.cat(all_probs, dim=0)
        masks = torch.cat(all_masks, dim=0).long()
        best_t, best_dice = self._sweep(probs, masks)
        self._write_json(self.save_dir, best_t, best_dice)
        print(f"[ThresholdSweep] best threshold = {best_t:.2f}  val Dice = {best_dice:.4f}")

    def _sweep(self, probs: torch.Tensor, masks: torch.Tensor) -> tuple[float, float]:
        best_t, best_dice = 0.5, -1.0
        if not self.candidates:
            raise ValueError("At least one threshold candidate is required.")
        # probs and masks are already [N, 1, H, W] from on_fit_end
        for t in self.candidates:
            preds = (probs > t).long()
            pred_flat = preds.flatten(start_dim=1)
            mask_flat = masks.flatten(start_dim=1)
            intersection = (pred_flat * mask_flat).sum(dim=1).float()
            cardinality = (pred_flat + mask_flat).sum(dim=1).float()
            scores = torch.where(
                cardinality == 0,
                torch.ones_like(cardinality),
                2.0 * intersection / cardinality.clamp_min(1.0),
            )
            dice = scores.mean().item()
            if dice > best_dice:
                best_t, best_dice = float(t), float(dice)
        return best_t, best_dice

    def _write_json(self, save_dir: str, threshold: float, val_dice: float):
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "best_threshold.json")
        settings = {"threshold": threshold, "val_dice": val_dice,
                    "metric": "macro_dice", "use_tta": self.use_tta}
        if self.tta_sizes is not None:
            settings["tta_sizes"] = list(self.tta_sizes)
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)
