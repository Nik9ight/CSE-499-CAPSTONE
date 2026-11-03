"""Priority 0 feasibility and correctness checks from GRACE_ROUTER_EXPERIMENTS.md.

Every check uses the real datasets and pretrained encoders on the GPU. Each one runs
in its own process so memory peaks and out-of-memory errors stay isolated. Logs go to
experiment_logs/<dataset>/<date>_<time>_priority0/ (one log per check, summary.log,
report.json); round-trip checkpoints go to a temporary directory and are deleted.

Usage:
  python tests/check_priority0.py                                  # all checks, all datasets
  python tests/check_priority0.py --tasks qata --checks masks roundtrip
  python tests/check_priority0.py --unit smoke --task qata --experiment router_region --checkpointing
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

import Config as config
from data_module import LViTDataModule
from experiments import configure_run
from Load_Dataset import ImageToImage2D, get_train_transforms, get_val_transforms, sorted_image_mask_pairs
from lightning_module import LViTLightningModule
from test_model import load_model, tta_inference

COMPARED = ("baseline", "router_region")
ROUTED = (
    "parallel_fixed", "parallel_uniform", "router_static", "router_image", "router_region",
    "router_no_memory", "router_always_memory", "router_assigned_text", "router_no_none",
)
CHECKS = ("forward_backward", "overfit", "smoke", "masks", "roundtrip", "last_batch")
INTERVENTIONS = ("uniform", "fixed", "image_mean", "shuffle", "none")
RESULT_TAG = "P0RESULT "
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Shared helpers ----------
def use_task(task):
    root = f"./data/{task}/"
    config.task_name = task
    config.data_root = root
    config.train_dataset, config.val_dataset, config.test_dataset = (
        f"{root}{split}/" for split in ("Train", "Val", "Test")
    )
    config.task_dataset = config.train_dataset
    for split in ("train", "val", "test"):
        setattr(config, f"{split}_text_csv", f"{root}prompt/{split}.csv")


def build_module(experiment, checkpointing=False, **kwargs):
    configure_run(config, experiment, checkpointing=checkpointing)
    return LViTLightningModule(model_config=config.get_CTranS_config(), **kwargs).to(DEVICE)


def load_batch(split, size, augment=False):
    """First `size` images of a split as one (sample, names) batch on the device."""
    path, csv = {
        "train": (config.train_dataset, config.train_text_csv),
        "val": (config.val_dataset, config.val_text_csv),
    }[split]
    transforms = (get_train_transforms if augment else get_val_transforms)(config.img_size)
    dataset = ImageToImage2D(path, joint_transform=transforms, image_size=config.img_size,
                             text_embed_dim=config.text_embed_dim, text_csv_path=csv)
    sample, names = next(iter(DataLoader(Subset(dataset, range(size)), batch_size=size)))
    return {key: value.to(DEVICE) for key, value in sample.items()}, names


def text_of(sample):
    return {"input_ids": sample["input_ids"], "attention_mask": sample["attention_mask"]}


def autocast(enabled):
    return torch.autocast("cuda", dtype=torch.float16, enabled=enabled)


def trainer_for(callbacks=(), **kwargs):
    return pl.Trainer(
        max_epochs=1, num_sanity_val_steps=0, precision=config.precision,
        accelerator="gpu", devices=1, logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False,
        deterministic=config.deterministic, callbacks=list(callbacks), **kwargs,
    )


def group_of(name):
    name = name.removeprefix("model.")
    if ".router." in name:
        return "router"
    head = name.split(".")[0]
    if head.startswith("stage"):
        return "vit_stages"
    return head if head in ("backbone", "text_fusion") else "decoder_heads"


@torch.no_grad()
def subset_dice(module, batch):
    sample, _ = batch
    module.eval()
    with autocast(True):
        logits = module.model(sample["image"], text_of(sample))["main"]
    pred, target = torch.sigmoid(logits.float()) > 0.5, sample["label"] > 0.5
    return (2 * (pred & target).sum() / (pred.sum() + target.sum()).clamp_min(1)).item()


# ---------- Checks ----------
def check_forward_backward(args):
    """One fp32 training step on real images: finite loss and gradients everywhere."""
    module = build_module(args.experiment, args.checkpointing).train()
    loss, _, _, _, parts = module._shared_step(load_batch("train", 4, augment=True))
    loss.backward()

    trainable = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    no_grad = [n for n, p in trainable if p.grad is None]
    nonfinite = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    squares = {}
    for name, param in trainable:
        if param.grad is not None:
            squares[group_of(name)] = squares.get(group_of(name), 0.0) + param.grad.float().pow(2).sum().item()
    grad_norms = {group: value ** 0.5 for group, value in squares.items()}
    bert_frozen = all(not p.requires_grad and p.grad is None for p in module.model.text_encoder.parameters())
    losses = {"loss": loss.item(), **{k: v.item() for k, v in parts.items()}}

    ok = (all(math.isfinite(v) for v in losses.values()) and not nonfinite and bert_frozen
          and all(v > 0 for v in grad_norms.values()))
    return {
        "ok": ok, "losses": losses, "grad_norms": grad_norms, "bert_frozen": bert_frozen,
        "params_without_grad": no_grad, "params_with_nonfinite_grad": nonfinite,
        "summary": f"loss={loss.item():.3f} groups_with_grad={len(grad_norms)} "
                   f"nonfinite={len(nonfinite)} no_grad={len(no_grad)} bert_frozen={bert_frozen}",
    }


def check_overfit(args, n_images=16, steps=200, eval_every=50):
    """Full-batch AMP training on a fixed 16-image subset at constant LR."""
    module = build_module(args.experiment, args.checkpointing)
    batch = load_batch("train", n_images)
    optimizer = module.configure_optimizers()["optimizer"]
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"]   # skip the 1% warmup start; constant LR
    scaler = torch.amp.GradScaler("cuda")

    dice, losses = {0: subset_dice(module, batch)}, []
    for step in range(1, steps + 1):
        module.train()
        with autocast(True):
            loss = module._shared_step(batch)[0]
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
        if step % eval_every == 0:
            dice[step] = subset_dice(module, batch)

    ok = all(math.isfinite(v) for v in losses) and dice[steps] >= 0.8
    return {
        "ok": ok, "dice_by_step": dice, "loss_first": losses[0], "loss_last": losses[-1],
        "lr": [g["lr"] for g in optimizer.param_groups],
        "summary": f"dice {dice[0]:.3f} -> {dice[steps]:.3f} after {steps} steps; "
                   f"loss {losses[0]:.3f} -> {losses[-1]:.3f}",
    }


class StabilityMonitor(pl.Callback):
    def __init__(self):
        self.losses, self.grad_norms, self.scales = [], [], []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Lightning unscales AMP gradients before this hook.
        norms = [p.grad.detach().float().norm() for g in optimizer.param_groups
                 for p in g["params"] if p.grad is not None]
        self.grad_norms.append(torch.stack(norms).norm().item() if norms else 0.0)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(float(outputs["loss"]))
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        self.scales.append(scaler.get_scale() if scaler is not None else None)


def check_smoke(args, steps=30):
    """Lightning 16-mixed training at the selected batch size with the backbone unfrozen."""
    module = build_module(args.experiment, args.checkpointing, freeze_backbone_epochs=0)
    dm = LViTDataModule(batch_size=args.batch_size, num_workers=config.num_workers)
    monitor = StabilityMonitor()
    trainer = trainer_for([monitor], limit_train_batches=steps, limit_val_batches=4)
    trainer.fit(module, dm)

    finite = [math.isfinite(g) for g in monitor.grad_norms]
    first_finite = finite.index(True) if True in finite else len(finite)
    late_nonfinite = [i for i, f in enumerate(finite) if not f and i > first_finite]
    params_finite = all(torch.isfinite(p).all() for p in module.parameters())
    val_loss = float(trainer.callback_metrics.get("val/loss", float("nan")))
    total = torch.cuda.get_device_properties(0).total_memory
    reserved = torch.cuda.max_memory_reserved()

    ok = (all(math.isfinite(v) for v in monitor.losses) and not late_nonfinite and params_finite
          and math.isfinite(val_loss) and first_finite < len(finite) and reserved / total < 0.95)
    grad_tail = [g for g in monitor.grad_norms if math.isfinite(g)]
    return {
        "ok": ok, "steps": len(monitor.losses), "losses": monitor.losses,
        "grad_norms": monitor.grad_norms, "loss_scales": monitor.scales,
        "skipped_steps_at_start": first_finite, "nonfinite_after_start": late_nonfinite,
        "params_finite": params_finite, "val_loss": val_loss,
        "summary": f"{len(monitor.losses)} steps, loss {monitor.losses[0]:.3f} -> {monitor.losses[-1]:.3f}, "
                   f"grad_norm {min(grad_tail, default=float('nan')):.2f}-{max(grad_tail, default=float('nan')):.2f}, "
                   f"AMP skipped {first_finite} initial step(s), val_loss={val_loss:.3f}, "
                   f"peak reserved {reserved / 2**30:.1f}/{total / 2**30:.1f} GiB",
    }


def expected_availability(settings, stage):
    if settings["language"] == "all":
        available = [True] * 4
    else:
        available = [group == stage - 1 for group in range(4)]
    memory_on = stage > 1 and settings["memory"] != "off"
    available += [memory_on and settings["memory"] == "selective", settings["allow_none"]]
    return torch.tensor(available, device=DEVICE), memory_on


def mask_failures(label, settings, routing, override=None):
    failures, worst = [], {"masked_weight": 0.0, "masked_norm": 0.0}
    mode = override or settings["mode"]
    for stage in range(1, 5):
        weights = routing[f"stage{stage}"]["weights"]
        norms = routing[f"stage{stage}"]["contribution_norms"]
        where = f"{label} stage{stage}"
        side = config.img_size // 2 ** (stage + 1)
        if tuple(weights.shape[2:]) != (side, side, 6):
            failures.append(f"{where}: grid {tuple(weights.shape[2:4])}, expected {side}x{side}")

        available, memory_on = expected_availability(settings, stage)
        if override == "none":
            available = torch.zeros_like(available)
            available[5] = True
        masked_weight = weights[..., ~available].abs().max().item() if (~available).any() else 0.0
        worst["masked_weight"] = max(worst["masked_weight"], masked_weight)
        if masked_weight != 0.0:
            failures.append(f"{where}: unavailable sources carry weight {masked_weight:g}")
        if (weights.sum(-1) - 1).abs().max().item() > 1e-5:
            failures.append(f"{where}: weights do not sum to 1")

        responding = expected_availability(settings, stage)[0].clone()
        responding[5] = False
        if settings["memory"] == "always" and memory_on:
            responding[4] = True
            if not (norms[..., 4] > 0).any():
                failures.append(f"{where}: always-on memory contributes nothing")
        masked_norm = norms[..., ~responding].abs().max().item()
        worst["masked_norm"] = max(worst["masked_norm"], masked_norm)
        if masked_norm != 0.0:
            failures.append(f"{where}: unavailable/none sources contribute norm {masked_norm:g}")

        softmax_sources = expected_availability(settings, stage)[0]
        if mode == "fixed":
            target = torch.zeros(6, device=DEVICE)
            target[stage - 1] = 1.0
            target[4] = float(softmax_sources[4])
            target = target / target.sum()
        elif mode == "uniform":
            target = softmax_sources.float() / softmax_sources.sum()
        else:
            target = None
        if target is not None and (weights - target).abs().max().item() > 1e-6:
            failures.append(f"{where}: {mode} weights differ from {target.tolist()}")
        if override == "image_mean" and weights.flatten(2, 3).std(dim=2).max().item() > 1e-6:
            failures.append(f"{where}: image_mean weights vary across tokens")
    return failures, worst


def check_masks(args):
    """Stage-1 memory and disabled sources get exactly zero weight in every routed preset."""
    sample, _ = load_batch("val", 2)
    failures, worst, checked = [], {"masked_weight": 0.0, "masked_norm": 0.0}, 0
    for preset in ROUTED:
        module = build_module(preset)
        model = module.model.eval()
        runs = [(f"{preset} fp32", None, False), (f"{preset} fp16", None, True)]
        if preset == "router_region":
            runs += [(f"{preset} override={o}", o, True) for o in INTERVENTIONS]
        for label, override, amp in runs:
            with torch.no_grad(), autocast(amp):
                out = model(sample["image"], text_of(sample), return_routing=True, routing_override=override)
            found, stats = mask_failures(label, model.router_config, out["routing"], override)
            failures += found
            worst = {k: max(worst[k], stats[k]) for k in worst}
            checked += 1
        del module, model, out
        torch.cuda.empty_cache()
    return {
        "ok": not failures, "failures": failures, "forward_passes_checked": checked, **worst,
        "summary": f"{len(ROUTED)} presets, {checked} routed forwards x 4 stages; "
                   f"max weight on unavailable sources={worst['masked_weight']:g}, "
                   f"max masked contribution={worst['masked_norm']:g}, failures={len(failures)}",
    }


@torch.no_grad()
def predict(model, sample, amp, tta):
    with autocast(amp):
        if tta:
            return tta_inference(model, sample["image"], text_of(sample), sizes=config.ms_tta_sizes).float()
        return torch.sigmoid(model(sample["image"], text_of(sample))["main"].float())


def check_roundtrip(args):
    """Train a few steps, save a Lightning checkpoint, restore with test_model.load_model, compare."""
    module = build_module(args.experiment, args.checkpointing)
    trainer = trainer_for(limit_train_batches=3, limit_val_batches=1, default_root_dir=args.work_dir)
    trainer.fit(module, LViTDataModule(batch_size=8, num_workers=config.num_workers))
    path = Path(args.work_dir) / f"roundtrip_{args.task}_{args.experiment}.ckpt"
    trainer.save_checkpoint(path)
    size_gib = path.stat().st_size / 2**30
    try:
        restored, _ = load_model(DEVICE, path)
    finally:
        path.unlink()
    original = module.model.to(DEVICE).eval()

    state_a, state_b = original.state_dict(), restored.state_dict()
    mismatched = sorted(set(state_a) ^ set(state_b)) + [
        k for k in state_a if k in state_b and not torch.equal(state_a[k], state_b[k])
    ]
    sample, _ = load_batch("val", 8)
    comparisons = {}
    for amp in (False, True):
        for tta in (False, True):
            a, b = predict(original, sample, amp, tta), predict(restored, sample, amp, tta)
            comparisons[f"{'fp16' if amp else 'fp32'}{'+tta' if tta else ''}"] = {
                "max_abs_diff": (a - b).abs().max().item(),
                "mask_agreement": ((a > 0.5) == (b > 0.5)).float().mean().item(),
            }
    same_router = original.router_config == restored.router_config
    ok = (not mismatched and same_router and all(
        c["mask_agreement"] == 1.0 and c["max_abs_diff"] <= (1e-3 if "fp16" in name else 1e-5)
        for name, c in comparisons.items()))
    worst = max(c["max_abs_diff"] for c in comparisons.values())
    return {
        "ok": ok, "mismatched_tensors": mismatched, "router_config_restored": same_router,
        "comparisons": comparisons, "checkpoint_gib": size_gib,
        "summary": f"{len(state_a)} tensors, mismatched={len(mismatched)}, router_config match={same_router}, "
                   f"max |dp| over fp32/fp16 x single/TTA={worst:g}, ckpt {size_gib:.1f} GiB",
    }


def check_last_batch(args, batch_sizes=(8, 16, 32)):
    """No batch size in use may leave a 1-image final training batch (pooled ASPP BatchNorm)."""
    counts = {}
    for split, path in (("train", config.train_dataset), ("val", config.val_dataset)):
        counts[split] = len(sorted_image_mask_pairs(path + "Images", path + "GTs")[0])
    sizes = sorted(set(batch_sizes) | {args.batch_size})
    last = {bs: counts["train"] % bs or bs for bs in sizes}
    singleton = [bs for bs, n in last.items() if n == 1]

    module = build_module("baseline")
    sample, _ = load_batch("train", 1)
    model = module.model.train()
    try:
        model(sample["image"], text_of(sample))
        train_error = None
    except ValueError as err:
        train_error = str(err).splitlines()[0]
    with torch.no_grad():
        eval_ok = torch.isfinite(model.eval()(sample["image"], text_of(sample))["main"]).all().item()

    ok = args.batch_size not in singleton and eval_ok
    return {
        "ok": ok, "counts": counts, "devices": 1, "last_train_batch": last,
        "singleton_batch_sizes": singleton, "singleton_train_error": train_error,
        "singleton_eval_ok": eval_ok,
        "summary": f"train={counts['train']} last batch at bs {sizes}: {[last[b] for b in sizes]}; "
                   f"singleton at selected bs {args.batch_size}: {args.batch_size in singleton}; "
                   f"1-image train forward -> {train_error or 'no error'}",
    }


UNITS = {
    "forward_backward": check_forward_backward, "overfit": check_overfit, "smoke": check_smoke,
    "masks": check_masks, "roundtrip": check_roundtrip, "last_batch": check_last_batch,
}


# ---------- Unit and driver ----------
def run_unit(args):
    torch.multiprocessing.set_start_method("spawn", force=True)
    use_task(args.task)
    pl.seed_everything(config.seed, workers=True)
    start = time.time()
    try:
        result = UNITS[args.unit](args)
    except torch.OutOfMemoryError as err:
        result = {"ok": False, "oom": True, "summary": "CUDA out of memory: " + str(err).splitlines()[0][:120]}
    result.update(
        seconds=round(time.time() - start, 1),
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
    )
    print(RESULT_TAG + json.dumps(result, default=str), flush=True)


def plan(tasks, checks):
    units = []
    for task in tasks:
        for check in checks:
            if check in ("masks", "last_batch"):
                units.append((check, task, None, True, False))
            elif check == "smoke":
                for experiment in COMPARED:
                    units.append((check, task, experiment, True, False))
                    units.append((check, task, experiment, False, True))   # memory probe
            else:
                units.extend((check, task, experiment, True, False) for experiment in COMPARED)
    return units


def run_driver(args):
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_dirs = {task: Path(args.log_dir) / task / f"{stamp}_{args.name}" for task in args.tasks}
    for run_dir in run_dirs.values():
        run_dir.mkdir(parents=True, exist_ok=True)
    units = plan(args.tasks, args.checks)
    print(f"Running {len(units)} checks at batch size {args.batch_size}; logs in "
          f"{', '.join(str(d) for d in run_dirs.values())}", flush=True)

    results = []
    with tempfile.TemporaryDirectory(prefix="grace_priority0_") as scratch:
        work_dir = args.work_dir or scratch
        for check, task, experiment, checkpointing, probe in units:
            run_dir = run_dirs[task]
            name = "_".join(filter(None, (check, experiment, "ckpt" if checkpointing else "nockpt")))
            cmd = [sys.executable, str(SCRIPT), "--unit", check, "--task", task,
                   "--batch-size", str(args.batch_size), "--work-dir", str(work_dir)]
            cmd += ["--experiment", experiment] if experiment else []
            cmd += ["--checkpointing"] if checkpointing else []
            proc = subprocess.run(cmd, capture_output=True, text=True)
            (run_dir / f"{name}.log").write_text(proc.stdout + proc.stderr)
            lines = [l for l in proc.stdout.splitlines() if l.startswith(RESULT_TAG)]
            if lines:
                result = json.loads(lines[-1][len(RESULT_TAG):])
            else:
                tail = [l for l in proc.stderr.strip().splitlines() if l.strip()][-1:] or ["no output"]
                result = {"ok": False, "summary": f"crashed (exit {proc.returncode}): {tail[0][:160]}"}
            result.update(check=check, task=task, experiment=experiment, checkpointing=checkpointing, probe=probe)
            results.append(result)
            status = "OOM" if result.get("oom") else "PASS" if result["ok"] else "FAIL"
            if probe:
                status = f"probe:{status}"
            line = (f"[{status:10s}] {check:16s} {task:7s} {experiment or '-':14s} "
                    f"ckpt={'on ' if checkpointing else 'off'} {result.get('seconds', '-')}s  {result['summary']}")
            print(line, flush=True)
            with open(run_dir / "summary.log", "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    for task, run_dir in run_dirs.items():
        task_results = [r for r in results if r["task"] == task]
        (run_dir / "report.json").write_text(json.dumps(task_results, indent=2, default=str))
    failed = [r for r in results if not r["ok"] and not r["probe"]]
    print(f"RESULT: {'FAIL' if failed else 'PASS'} ({len(results) - len(failed)}/{len(results)} ok "
          f"incl. probes); reports in {', '.join(str(d) for d in run_dirs.values())}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GRACE Priority 0 feasibility/correctness checks.")
    parser.add_argument("--tasks", nargs="+", default=["qata", "MosMed"])
    parser.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--log-dir", default=str(ROOT / "experiment_logs"),
                        help="Root for per-dataset run folders (default: experiment_logs/).")
    parser.add_argument("--name", default="priority0", help="Suffix of the run folder name.")
    parser.add_argument("--work-dir", help="Where round-trip checkpoints are written (default: a temp dir).")
    parser.add_argument("--unit", choices=CHECKS, help=argparse.SUPPRESS)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    parser.add_argument("--experiment", help=argparse.SUPPRESS)
    parser.add_argument("--checkpointing", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.unit:
        run_unit(args)
    else:
        sys.exit(run_driver(args))
