"""Selection/CLI checks run without PyTorch or pretrained assets."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import visualize_i11 as xai
from cumulative_experiments import SEEDS, make_spec
from experiment_io import atomic_json


def make_study(root, scores=(0.81, 0.87, 0.85)):
    for seed, score in zip(SEEDS, scores):
        spec = make_spec("qata", "I11", seed, root / "data", root)
        run_dir = Path(spec["run_dir"])
        atomic_json(run_dir / "spec.json", spec)
        atomic_json(run_dir / "train_result.json", {
            "best_val_global_dice": score, "best_epoch": 12,
            "checkpoint": f"C:\\old_cloud_copy\\qata\\I11\\seed_{seed}\\models\\best-v1.ckpt",
        })
        # Deliberately disagreeing test scores must never influence selection.
        atomic_json(run_dir / "test_result.json", {"test_macro_dice": 1 - score})
        (run_dir / "models").mkdir()
        (run_dir / "models" / "best-v1.ckpt").write_bytes(b"selection-only fixture")


def test_select_validation_winner_and_relocate_checkpoint(tmp_path):
    make_study(tmp_path)
    selected = xai.select_best_i11(tmp_path, "qata")
    assert selected["seed"] == 1337
    assert selected["selection_metric"] == "best_val_global_dice"
    assert Path(selected["checkpoint"]) == tmp_path / "qata/I11/seed_1337/models/best-v1.ckpt"
    assert len(selected["candidates"]) == 3


def test_tie_selects_lowest_seed(tmp_path):
    make_study(tmp_path, (0.9, 0.9, 0.9))
    assert xai.select_best_i11(tmp_path, "qata")["seed"] == 42


def test_requires_all_three_training_results(tmp_path):
    make_study(tmp_path)
    (tmp_path / "qata/I11/seed_2026/train_result.json").unlink()
    with pytest.raises(FileNotFoundError, match="metadata is missing"):
        xai.select_best_i11(tmp_path, "qata")


def test_missing_winner_never_falls_back_to_lower_score(tmp_path):
    make_study(tmp_path)
    (tmp_path / "qata/I11/seed_1337/models/best-v1.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="will not fall back"):
        xai.select_best_i11(tmp_path, "qata")


@pytest.mark.parametrize("score", [float("nan"), 1.1, -0.2, True, "0.9"])
def test_reject_invalid_validation_score(tmp_path, score):
    make_study(tmp_path)
    path = tmp_path / "qata/I11/seed_42/train_result.json"
    result = json.loads(path.read_text())
    result["best_val_global_dice"] = score
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Invalid best_val_global_dice"):
        xai.select_best_i11(tmp_path, "qata")


def test_reject_wrong_variant_identity(tmp_path):
    make_study(tmp_path)
    path = tmp_path / "qata/I11/seed_42/spec.json"
    spec = json.loads(path.read_text())
    spec["experiment"] = "I10"
    atomic_json(path, spec)
    with pytest.raises(ValueError, match="identity"):
        xai.select_best_i11(tmp_path, "qata")


def test_dry_run_without_site_packages_or_writes(tmp_path):
    make_study(tmp_path)
    output = tmp_path / "figures"
    run = subprocess.run([
        sys.executable, "-S", str(ROOT / "visualize_i11.py"), "--study-root", str(tmp_path),
        "--dataset", "qata", "--output-dir", str(output), "--dry-run",
    ], capture_output=True, text=True, check=True)
    assert "Selected seed 1337" in run.stdout
    assert not output.exists()


def test_case_selection_is_explicit_or_sorted():
    names = ["z.png", "a.png", "b.png"]
    assert xai.select_case_indices(names, None, 2) == [1, 2]
    assert xai.select_case_indices(names, None, 0) == [1, 2, 0]
    assert xai.select_case_indices(names, ["b.png", "z.png"], 1) == [2, 0]
    with pytest.raises(ValueError, match="absent"):
        xai.select_case_indices(names, ["missing.png"], 1)
    with pytest.raises(ValueError, match="duplicate"):
        xai.select_case_indices(names, ["a.png", "a.png"], 1)
