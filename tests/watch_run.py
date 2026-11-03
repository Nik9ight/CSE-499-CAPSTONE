"""Watches a running train_model.py session and prints one line per event worth acting on.

Events: a progress summary every --every epochs, NaN/inf losses, a val Dice drop of more
than --drop below the best, early-stopping patience reaching 10 and 15, a stall (no new
epoch for --stall-min minutes) and the process exiting. On start it reports nothing about
epochs already finished, so it can be re-armed mid-run. Only needs the standard library.

Usage: python -u tests/watch_run.py qata/LViT/<session> --pid <train_model.py pid>
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

PATIENCE_MILESTONES = (10, 15)


def read_epochs(session):
    """Returns {epoch: metrics} for epochs whose train row has been written (val row comes first)."""
    path = session / "csv_logs" / "version_0" / "metrics.csv"
    val, done = {}, {}
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    for row in rows:
        if not row.get("epoch"):
            continue
        epoch = int(row["epoch"])
        if row.get("val/dice"):
            val[epoch] = {k: float(row[k]) for k in ("val/dice", "val/iou", "val/loss")}
        if row.get("train/loss_epoch") and epoch in val:
            done[epoch] = {**val[epoch], "train/loss": float(row["train/loss_epoch"]),
                           "train/dice": float(row["train/dice"])}
    return done


def best_and_wait(epochs, upto, min_delta=0.0):
    """Best val Dice over epochs <= upto (the saved checkpoint), its epoch, and EarlyStopping's
    wait count, which only resets on a gain of more than min_delta."""
    best_epoch = max((e for e in epochs if e <= upto), key=lambda e: epochs[e]["val/dice"])
    stop_best, wait = None, 0
    for e in sorted(e for e in epochs if e <= upto):
        dice = epochs[e]["val/dice"]
        if stop_best is None or dice - min_delta > stop_best:
            stop_best, wait = dice, 0
        else:
            wait += 1
    return epochs[best_epoch]["val/dice"], best_epoch, wait


def stopping_settings(session):
    """(patience, min_delta) from the run's config snapshot; runs before min_delta existed used 0."""
    try:
        training = json.loads((session / "run_config.json").read_text())["training"]
    except (OSError, ValueError, KeyError):
        return 20, 0.0
    return training.get("early_stopping_patience", 20), training.get("early_stopping_min_delta", 0.0)


def gpu_memory():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, total = (int(x) for x in out.split(",")[:2])
        return f"{used / 1024:.1f}/{total / 1024:.1f} GiB"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "n/a"


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    # A zombie still answers kill(0); treat it as exited.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def last_error(log):
    """Last line of the run's stdout log that looks like an error, if a log was given."""
    if log is None:
        return None
    try:
        lines = log.read_text(errors="replace").replace("\r", "\n").splitlines()
    except OSError:
        return None
    errors = [l.strip() for l in lines if any(s in l for s in ("Error", "Exception", "Killed", "out of memory"))]
    return errors[-1] if errors else None


def fmt_minutes(seconds):
    return f"{seconds / 60:.1f} min"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, help="the run's stdout log, quoted if the run exits without finishing")
    parser.add_argument("--max-epochs", type=int, default=201)
    parser.add_argument("--every", type=int, default=10, help="progress line every N epochs")
    parser.add_argument("--drop", type=float, default=0.05, help="val Dice drop below best that triggers a warning")
    parser.add_argument("--stall-min", type=float, default=10.0)
    parser.add_argument("--poll", type=float, default=30.0)
    args = parser.parse_args()
    patience, min_delta = stopping_settings(args.session)

    epochs = read_epochs(args.session) or {}
    seen = max(epochs, default=-1)
    last_new_epoch_at = time.time()
    epoch_times = []  # (epoch, detection time), to estimate epoch length
    stalled = False

    while True:
        if not alive(args.pid):
            epochs = read_epochs(args.session) or epochs
            last = max(epochs, default=-1)
            summary = f"last finished epoch {last + 1}/{args.max_epochs}"
            if epochs:
                best, best_epoch, wait = best_and_wait(epochs, last, min_delta)
                summary += f", best val Dice {best:.4f} at epoch {best_epoch + 1}, patience {wait}/{patience}"
            if (args.session / "best_threshold.json").exists():
                print(f"FINISHED: process {args.pid} exited after on_fit_end (best_threshold.json written); {summary}")
            else:
                error = last_error(args.log)
                print(f"EXITED WITHOUT FINISHING: process {args.pid} is gone and best_threshold.json is missing "
                      f"(crash, OOM or Ctrl+C); {summary}; last error: {error or 'none found, see the launching terminal'}")
            return

        current = read_epochs(args.session)
        if current is not None:
            epochs = current
        now = time.time()
        for epoch in sorted(e for e in epochs if e > seen):
            m = epochs[epoch]
            seen = epoch
            last_new_epoch_at = now
            epoch_times.append((epoch, now))
            stalled = False
            label = f"epoch {epoch + 1}/{args.max_epochs}"

            bad = [k for k, v in m.items() if not math.isfinite(v)]
            if bad:
                print(f"NAN: {label} has non-finite {', '.join(bad)}")
                continue

            best, best_epoch, wait = best_and_wait(epochs, epoch, min_delta)
            if best - m["val/dice"] > args.drop:
                print(f"DROP: {label} val Dice {m['val/dice']:.4f} is {best - m['val/dice']:.4f} "
                      f"below the best {best:.4f} (epoch {best_epoch + 1})")
            if wait in PATIENCE_MILESTONES:
                print(f"PLATEAU: {label} is {wait}/{patience} epochs without a gain over {min_delta:g} "
                      f"(best {best:.4f} at epoch {best_epoch + 1})")
            if (epoch + 1) % args.every == 0:
                pace = ""
                if len(epoch_times) >= 2:
                    (e0, t0), (e1, t1) = epoch_times[0], epoch_times[-1]
                    if e1 > e0:
                        per_epoch = (t1 - t0) / (e1 - e0)
                        pace = (f", {fmt_minutes(per_epoch)}/epoch, at most "
                                f"{(args.max_epochs - epoch - 1) * per_epoch / 3600:.1f} h left")
                print(f"PROGRESS: {label} val Dice {m['val/dice']:.4f} / IoU {m['val/iou']:.4f}, "
                      f"train loss {m['train/loss']:.4f}, val loss {m['val/loss']:.4f}; "
                      f"best {best:.4f} at epoch {best_epoch + 1}, patience {wait}/{patience}; "
                      f"GPU {gpu_memory()}{pace}")

        if not stalled and now - last_new_epoch_at > args.stall_min * 60:
            stalled = True
            print(f"STALL: no new epoch for {fmt_minutes(now - last_new_epoch_at)} "
                  f"(last finished epoch {seen + 1}); process {args.pid} is still alive")
        sys.stdout.flush()
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
