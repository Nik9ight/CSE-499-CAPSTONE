"""Dependency-free orchestration checks; synthetic worker outputs are not model results."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_cumulative as runner
from cumulative_experiments import DATASETS, SEEDS, VARIANTS, make_spec, variant_settings
from experiment_io import atomic_json


def specs_at(tmp_path, variants=VARIANTS, datasets=DATASETS, seeds=SEEDS):
    return [make_spec(d, v, s, tmp_path / "data", tmp_path)
            for d in datasets for s in seeds for v in variants]


def fake_worker(calls, fail=None):
    def worker(path, stage, log):
        spec = runner.read_json(path)
        calls.append((spec["dataset"], spec["experiment"], spec["seed"], stage))
        run_dir = Path(spec["run_dir"])
        if fail and fail(spec, stage):
            return 1
        checkpoint = str(run_dir / "models" / "best.ckpt")
        if stage == "train":
            atomic_json(run_dir / "train_result.json", {
                "checkpoint": checkpoint, "best_val_global_dice": 0.8, "best_epoch": 3,
            })
        else:
            # Testing must follow the matching training process, never a globbed checkpoint.
            assert runner.read_json(run_dir / "train_result.json")["checkpoint"] == checkpoint
            score = 0.7 + VARIANTS.index(spec["experiment"]) * 0.01
            atomic_json(run_dir / "test_result.json", {
                "checkpoint": checkpoint, "dataset": spec["dataset"], "protocol": "native",
                "threshold": 0.5, "use_tta": False, "batch_size": 32, "test_cases": 2,
                "test_macro_dice": score, "test_macro_iou": 0.6,
                "test_global_dice": 0.8, "test_global_iou": 2 / 3,
            })
        return 0
    return worker


def read_rows(tmp_path):
    with (tmp_path / "results.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_shell_dry_run_has_exactly_72_unique_runs_without_creating_outputs(tmp_path):
    result = subprocess.run(["bash", str(ROOT / "run_cumulative.sh"), "--dry-run",
                             "--output-root", str(tmp_path / "unused")],
                            capture_output=True, text=True, check=True)
    specs = json.loads(result.stdout.split("\n", 1)[1])
    assert len(specs) == 72
    assert len({(s["dataset"], s["experiment"], s["seed"]) for s in specs}) == 72
    assert not (tmp_path / "unused").exists()
    assert all(s["settings"]["batch_size"] == s["evaluation"]["batch_size"] == 32 for s in specs)
    assert all(s["settings"]["gradient_checkpointing"] for s in specs)


def test_cumulative_treatments_do_not_leak_text_or_losses():
    for variant in VARIANTS[:7]:
        settings = variant_settings(variant)
        assert settings["use_text"] is False and settings["peal_loss_weight"] == 0
        assert settings["router_config"]["mode"] == "off"
    for variant in VARIANTS[:6]:
        assert variant_settings(variant)["deep_supervision_loss_weight"] == 0
    assert variant_settings("I7")["text_fusion_mode"] == "final"
    assert variant_settings("I8")["text_fusion_mode"] == "uniform"
    assert variant_settings("I9")["text_fusion_mode"] == "learned"
    assert variant_settings("I9")["peal_loss_weight"] == 0
    assert variant_settings("I10")["peal_loss_weight"] == 0.2
    assert variant_settings("I11")["router_config"]["mode"] == "region"


def test_all_72_train_then_test_and_resume_without_duplicate_rows(tmp_path, monkeypatch):
    specs = specs_at(tmp_path)
    calls = []
    monkeypatch.setattr(runner, "run_stage", fake_worker(calls))
    assert runner.execute(specs, tmp_path) == 0
    assert len(calls) == 144
    assert [c[-1] for c in calls] == ["train", "test"] * 72
    rows = read_rows(tmp_path)
    assert len(rows) == 72 and all(r["status"] == "completed" for r in rows)
    full = next(r for r in rows if r["experiment"] == "I11")
    assert float(full["delta_macro_dice_previous_pp"]) == pytest.approx(1.0)
    assert float(full["delta_macro_dice_vs_I0_pp"]) == pytest.approx(11.0)
    assert float(full["delta_macro_dice_vs_I6_pp"]) == pytest.approx(5.0)
    assert int(full["completed_seeds"]) == 3
    assert float(full["seed_std_macro_dice"]) == 0
    assert runner.execute(specs, tmp_path) == 0
    assert len(calls) == 144 and len(read_rows(tmp_path)) == 72


@pytest.mark.parametrize("failed_stage", ["train", "test"])
def test_failed_phase_retries_without_repeating_completed_work(tmp_path, monkeypatch, failed_stage):
    specs = specs_at(tmp_path, variants=("I0", "I1"), datasets=("qata",), seeds=(42,))
    calls = []
    monkeypatch.setattr(runner, "run_stage", fake_worker(
        calls, lambda spec, stage: spec["experiment"] == "I0" and stage == failed_stage))
    assert runner.execute(specs, tmp_path) == 1
    rows = read_rows(tmp_path)
    assert rows[0]["status"] == f"{failed_stage}_failed" and rows[1]["status"] == "completed"
    if failed_stage == "train":
        assert ("qata", "I0", 42, "test") not in calls
    prior_calls = len(calls)
    monkeypatch.setattr(runner, "run_stage", fake_worker(calls))
    assert runner.execute(specs, tmp_path) == 0
    assert len(calls) - prior_calls == (2 if failed_stage == "train" else 1)
    assert all(r["status"] == "completed" for r in read_rows(tmp_path))


def test_interrupt_records_state_and_never_tests_partial_training(tmp_path, monkeypatch):
    specs = specs_at(tmp_path, variants=("I0",), datasets=("qata",), seeds=(42,))
    def interrupt(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(runner, "run_stage", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(specs, tmp_path)
    assert read_rows(tmp_path)[0]["status"] == "interrupted"
    assert not (Path(specs[0]["run_dir"]) / "test_result.json").exists()


def test_mismatched_test_metadata_cannot_be_marked_complete(tmp_path, monkeypatch):
    specs = specs_at(tmp_path, variants=("I0",), datasets=("qata",), seeds=(42,))
    monkeypatch.setattr(runner, "run_stage", fake_worker([]))
    runner.execute(specs, tmp_path)
    path = Path(specs[0]["run_dir"]) / "test_result.json"
    result = runner.read_json(path)
    result["threshold"] = 0.4
    atomic_json(path, result)
    with pytest.raises(ValueError, match="protocol"):
        runner.completed(specs[0])


def test_preflight_checks_prompts_and_input_fingerprint(tmp_path):
    specs = specs_at(tmp_path, variants=("I7",), datasets=("qata",), seeds=(42,))
    dataset = tmp_path / "data" / "qata"
    (dataset / "prompt").mkdir(parents=True)
    for split in ("Train", "Val", "Test"):
        for folder in ("Images", "GTs"):
            path = dataset / split / folder
            path.mkdir(parents=True)
            for i in range(2):
                (path / f"{i}.png").write_bytes(b"preflight hashes bytes; image decoding occurs in the worker")
        (dataset / "prompt" / f"{split.lower()}.csv").write_text("Image,Description\n0.png,finding\n1.png,finding\n")
    before = runner.validate_data(specs)
    (dataset / "Train" / "Images" / "0.png").write_bytes(b"changed image")
    assert before != runner.validate_data(specs)
    (dataset / "prompt" / "test.csv").write_text("Image,Description\n0.png,finding\n")
    with pytest.raises(ValueError, match="description"):
        runner.validate_data(specs)
    image_only = specs_at(tmp_path, variants=("I0",), datasets=("qata",), seeds=(42,))
    assert runner.validate_data(image_only)
