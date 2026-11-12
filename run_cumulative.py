"""Resumable single-GPU orchestration for I0–I11; no ML imports in dry-run mode."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from cumulative_experiments import DATASETS, SEEDS, VARIANTS, make_spec
from experiment_io import atomic_json

ROOT = Path(__file__).resolve().parent
FIELDS = (
    "dataset", "experiment", "description", "seed", "status", "batch_size",
    "train_precision", "eval_precision", "checkpointing", "geometric_augmentation",
    "use_visual_refinement", "use_prev_cross_attention", "use_freq", "use_aspp",
    "use_high_res_tail", "deep_supervision", "use_text", "text_fusion_mode",
    "peal_loss_weight", "router_mode", "best_epoch", "epochs_completed", "optimizer_steps",
    "best_val_global_dice", "best_val_global_iou", "test_macro_dice", "test_macro_iou",
    "test_global_dice", "test_global_iou", "test_cases", "threshold", "use_tta",
    "delta_macro_dice_previous_pp", "delta_macro_dice_vs_I0_pp", "delta_macro_dice_vs_I6_pp",
    "completed_seeds", "seed_mean_macro_dice", "seed_std_macro_dice",
    "train_seconds_total", "test_seconds_total", "train_attempts", "test_attempts",
    "gpu", "torch_version", "lightning_version", "initialization_hash",
    "checkpoint", "run_dir", "error",
)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def source_hash():
    digest = hashlib.sha256()
    for path in sorted([*ROOT.glob("*.py"), *ROOT.glob("nets/*.py"), ROOT / "run_cumulative.sh"]):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_data(specs):
    """Check all requested splits/prompts before any GPU training; hash their contents."""
    fingerprints = {}
    for dataset in sorted({spec["dataset"] for spec in specs}):
        selected = [s for s in specs if s["dataset"] == dataset]
        root = Path(selected[0]["data_root"]) / dataset
        needs_text = any(s["settings"]["use_text"] for s in selected)
        needs_aspp = any(s["settings"]["use_aspp"] for s in selected)
        digest = hashlib.sha256()
        counts = {}
        for split in ("Train", "Val", "Test"):
            images_dir, masks_dir = root / split / "Images", root / split / "GTs"
            images = sorted(p for p in images_dir.glob("*") if p.is_file() and not p.name.startswith("."))
            masks = sorted(p for p in masks_dir.glob("*") if p.is_file() and not p.name.startswith("."))
            if not images or not masks:
                raise ValueError(f"Missing or empty split: {root / split} (Images/ and GTs/ required)")
            counts[split] = len(images)
            if split == "Train" and (len(images) < 2 or (needs_aspp and len(images) % 32 == 1)):
                raise ValueError(f"{dataset}: training count {len(images)} creates an invalid singleton "
                                 "ASPP batch at batch size 32; fix the split/sampler consistently.")
            mask_names, mask_stems = {p.name for p in masks}, {p.stem for p in masks}
            for image in images:
                if not (image.name in mask_names or image.stem in mask_stems or
                        f"mask_{image.name}" in mask_names or f"mask_{image.stem}" in mask_stems):
                    raise ValueError(f"No matching mask for {image}")
            files = images + masks
            if needs_text:
                prompt = root / "prompt" / f"{split.lower()}.csv"
                with prompt.open(newline="", encoding="utf-8-sig") as handle:
                    reader = csv.DictReader(handle)
                    if not {"Image", "Description"} <= set(reader.fieldnames or ()):
                        raise ValueError(f"{prompt}: required columns are Image,Description")
                    rows = list(reader)
                descriptions = {}
                for row in rows:
                    if row["Image"] in descriptions:
                        raise ValueError(f"{prompt}: duplicate Image entry {row['Image']}")
                    descriptions[row["Image"]] = row["Description"]
                for image in images:
                    description = descriptions.get(image.name)
                    if not description or not description.strip() or description.lower().strip() == "nan":
                        raise ValueError(f"{prompt}: missing/empty description for {image.name}")
                files.append(prompt)
            for path in files:
                digest.update(str(path.relative_to(root)).encode())
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
        fingerprints[dataset] = {"sha256": digest.hexdigest(), "counts": counts}
    return fingerprints


def run_stage(spec_path, stage, log_path):
    command = [sys.executable, "-u", str(ROOT / "cumulative_worker.py"),
               "--stage", stage, "--spec", str(spec_path)]
    spec = read_json(spec_path)
    environment = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(spec["seed"]))
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(f"\n--- {stage} attempt at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ---\n")
        log.flush()
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise


def completed(spec):
    run_dir = Path(spec["run_dir"])
    train_result = read_json(run_dir / "train_result.json")
    result = read_json(run_dir / "test_result.json")
    if not result:
        return False
    if not train_result or result.get("checkpoint") != train_result.get("checkpoint"):
        raise ValueError(f"Mismatched training/test checkpoint metadata in {run_dir}")
    if (result.get("dataset") != spec["dataset"] or result.get("protocol") != "native" or
            result.get("threshold") != 0.5 or result.get("use_tta") is not False or
            result.get("batch_size") != 32 or result.get("test_cases", 0) < 1):
        raise ValueError(f"Unexpected evaluation protocol in {run_dir}")
    for key in ("test_macro_dice", "test_macro_iou", "test_global_dice", "test_global_iou"):
        value = result.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {key} in {run_dir}")
    return True


def write_results(specs, output):
    rows = []
    for spec in specs:
        run_dir = Path(spec["run_dir"])
        settings = spec["settings"]
        state = read_json(run_dir / "state.json")
        row = {**settings, **spec, **read_json(run_dir / "train_result.json"),
               **state, **read_json(run_dir / "test_result.json")}
        row.update(status="completed" if completed(spec) else state.get("status", "pending"),
                   train_precision=settings["precision"], eval_precision="32",
                   checkpointing=settings["gradient_checkpointing"],
                   router_mode=settings["router_config"]["mode"])
        rows.append(row)
    by_key = {(r["dataset"], r["experiment"], r["seed"]): r for r in rows}
    for row in rows:
        if row["status"] != "completed":
            continue
        index = VARIANTS.index(row["experiment"])
        references = {"delta_macro_dice_vs_I0_pp": "I0", "delta_macro_dice_vs_I6_pp": "I6"}
        if index:
            references["delta_macro_dice_previous_pp"] = VARIANTS[index - 1]
        for field, variant in references.items():
            reference = by_key.get((row["dataset"], variant, row["seed"]))
            if reference and reference["status"] == "completed":
                row[field] = 100 * (row["test_macro_dice"] - reference["test_macro_dice"])
        values = [r["test_macro_dice"] for r in rows if r["dataset"] == row["dataset"] and
                  r["experiment"] == row["experiment"] and r["status"] == "completed"]
        row["completed_seeds"] = len(values)
        row["seed_mean_macro_dice"] = statistics.mean(values)
        row["seed_std_macro_dice"] = statistics.stdev(values) if len(values) > 1 else ""
    output = Path(output)
    fd, temporary = tempfile.mkstemp(prefix="results-", dir=output.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def execute(specs, output_root, *, fail_fast=False):
    """Resume completed phases; a failed test never causes a completed fit to repeat."""
    results_path = Path(output_root) / "results.csv"
    write_results(specs, results_path)
    failed = 0
    for index, spec in enumerate(specs, 1):
        run_dir = Path(spec["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        spec_path = run_dir / "spec.json"
        if spec_path.exists() and read_json(spec_path) != spec:
            raise ValueError(f"Run configuration changed: {spec_path}; use a new output directory.")
        atomic_json(spec_path, spec)
        if completed(spec):
            print(f"[{index}/{len(specs)}] completed: {spec['dataset']} {spec['experiment']} seed {spec['seed']}", flush=True)
            continue
        state_path = run_dir / "state.json"
        state = read_json(state_path)
        for stage in ("train", "test"):
            if (run_dir / f"{stage}_result.json").exists():
                continue
            state.update(status="training" if stage == "train" else "testing", error="")
            state[f"{stage}_attempts"] = state.get(f"{stage}_attempts", 0) + 1
            atomic_json(state_path, state)
            write_results(specs, results_path)
            log_path = run_dir / f"{stage}.log"
            print(f"[{index}/{len(specs)}] {stage}: {spec['dataset']} {spec['experiment']} "
                  f"seed {spec['seed']} batch 32 — {log_path}", flush=True)
            started = time.monotonic()
            try:
                code = run_stage(spec_path, stage, log_path)
                if code != 0:
                    raise RuntimeError(f"{stage} exited {code}; see {log_path}")
                if not (run_dir / f"{stage}_result.json").exists():
                    raise RuntimeError(f"{stage} produced no completion metadata; see {log_path}")
                if stage == "test" and not completed(spec):
                    raise RuntimeError("Testing did not produce valid results.")
                state["status"] = "trained" if stage == "train" else "completed"
            except KeyboardInterrupt:
                state.update(status="interrupted", error=f"Interrupted during {stage}")
                raise
            except Exception as error:
                state.update(status=f"{stage}_failed", error=str(error))
                print(state["error"], file=sys.stderr, flush=True)
                failed += 1
                break
            finally:
                key = f"{stage}_seconds_total"
                state[key] = state.get(key, 0.0) + time.monotonic() - started
                atomic_json(state_path, state)
                write_results(specs, results_path)
        if failed and fail_fast:
            break
    print(f"Results: {results_path} ({sum(completed(s) for s in specs)}/{len(specs)} completed)", flush=True)
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "cumulative_results")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--experiments", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--precision", choices=("16-mixed", "bf16-mixed", "32"), default="16-mixed")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--legacy-geometric-augmentation", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List resolved runs without imports, downloads, or training.")
    parser.add_argument("--preflight-only", action="store_true", help="Check data and CUDA environment without training.")
    args = parser.parse_args(argv)
    if args.workers < 0 or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("workers must be nonnegative and seeds must be in [0, 2**32).")
    for values in (args.datasets, args.seeds, args.experiments):
        if len(values) != len(set(values)):
            parser.error("Duplicate datasets, seeds, or experiments would duplicate runs.")
    specs = [make_spec(dataset, variant, seed, args.data_root, args.output_root,
                       checkpointing=not args.no_checkpointing, workers=args.workers,
                       precision=args.precision, geometric_augmentation=args.legacy_geometric_augmentation,
                       save_predictions=args.save_predictions)
             for dataset in args.datasets for seed in args.seeds for variant in args.experiments]
    print(f"{len(specs)} runs; batch size 32; train then native test for each run.", flush=True)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return 0
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".study.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another runner is using this output directory.") from None
        print("Checking split/prompt files and hashing inputs...", flush=True)
        fingerprints = validate_data(specs)
        check = "import torch, torchvision, pytorch_lightning, monai, timm, transformers, cv2, ml_collections; " \
                "assert torch.cuda.is_available(), 'CUDA GPU/PyTorch unavailable'; " \
                "print('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)"
        subprocess.run([sys.executable, "-c", check], check=True, cwd=ROOT)
        manifest = {"schema_version": 1, "source_sha256": source_hash(),
                    "data": fingerprints, "runs": specs}
        manifest_path = args.output_root / "study_manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError("Code, inputs, or study settings changed. Use a new --output-root "
                             "to avoid mixing incompatible experiments.")
        print("Checking cumulative model forwards, gradients, checkpoint reloads, and train/test integration...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", str(ROOT / "tests/test_cumulative_models.py"), "-q"],
            check=True, cwd=ROOT,
            env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
        )
        if args.preflight_only:
            print("Preflight passed. No training started.")
            return 0
        atomic_json(manifest_path, manifest)
        return execute(specs, args.output_root, fail_fast=args.fail_fast)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command to resume; see results.csv.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
