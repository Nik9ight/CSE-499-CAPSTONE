"""Dependency-free definition of the I0–I11 study and its resolved run settings."""
from pathlib import Path

from model_settings import model_kwargs_from_config


VARIANT_NAMES = (
    "Image-only scaffold", "Visual transformer", "Previous-stage memory",
    "Frequency enhancement", "ASPP", "High-resolution decoder",
    "Deep supervision (strong image-only)", "Final-layer text guidance",
    "Fixed hierarchical text groups", "Learned hierarchical text mixing",
    "PEAL alignment", "Joint regional router (full model)",
)
VARIANTS = tuple(f"I{i}" for i in range(12))
DATASETS = ("qata", "MosMed")
SEEDS = (42, 1337, 2026)


def variant_settings(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown cumulative variant: {variant}")
    i = VARIANTS.index(variant)
    return {
        "use_visual_refinement": i >= 1,
        "use_prev_cross_attention": i >= 2,
        "use_freq": i >= 3,
        "use_aspp": i >= 4,
        "use_high_res_tail": i >= 5,
        "deep_supervision": i >= 6,
        "deep_supervision_loss_weight": 0.3 if i >= 6 else 0.0,
        "use_text": i >= 7,
        "text_fusion_mode": "final" if i <= 7 else "uniform" if i == 8 else "learned",
        "peal_loss_weight": 0.2 if i >= 10 else 0.0,
        "use_adaptive_gated_fusion": False,
        "router_config": {"mode": "region" if i == 11 else "off"},
    }


def make_spec(dataset, variant, seed, data_root, output_root, *, checkpointing=True,
              workers=2, precision="16-mixed", geometric_augmentation=False,
              save_predictions=False):
    """Freeze the common recipe rather than inherit mutable Config.py edits."""
    settings = {
        "backbone": "convnext_tiny.in12k_ft_in1k",
        "text_encoder": "microsoft/BiomedVLP-CXR-BERT-specialized",
        "text_embed_dim": 768, "img_size": 224, "n_channels": 3, "n_labels": 1,
        "vit_depth": 2, "vit_heads": 8, "vit_drop_rate": 0.0,
        "vit_drop_path_rate": 0.2, "freq_sigma_init": 5.0,
        "aspp_in_out_ch": 384, "aspp_dilations": [1, 3, 6, 9], "aspp_dropout": 0.1,
        "aux_loss_weights": [0.3, 0.5], "epochs": 201, "batch_size": 32,
        "learning_rate": 3e-4, "warmup_epochs": 5, "freeze_backbone_epochs": 5,
        "early_stopping_patience": 20, "early_stopping_min_delta": 5e-4,
        "precision": precision, "deterministic": "warn", "num_workers": workers,
        "gradient_checkpointing": checkpointing, "use_mixup_cutmix": False,
        "geometric_augmentation": geometric_augmentation,
        "seed": seed, "use_tta": False, "use_ms_tta": False,
        "ms_tta_sizes": [224], "threshold_sweep": [0.5],
        **variant_settings(variant),
    }
    return {
        "schema_version": 1, "dataset": dataset, "experiment": variant,
        "description": VARIANT_NAMES[VARIANTS.index(variant)], "seed": seed,
        "data_root": str(Path(data_root).resolve()),
        "run_dir": str(Path(output_root).resolve() / dataset / variant / f"seed_{seed}"),
        "initialization_path": str(Path(output_root).resolve() / "initializations" / f"seed_{seed}.pt"),
        "settings": settings,
        "evaluation": {"protocol": "native", "threshold": 0.5, "use_tta": False,
                       "batch_size": 32, "precision": "32", "save_predictions": save_predictions},
    }


def configure(config, spec):
    for key, value in spec["settings"].items():
        setattr(config, key, value)
    config.task_name = spec["dataset"]
    dataset_dir = Path(spec["data_root"]) / spec["dataset"]
    config.data_root = str(dataset_dir)
    for split in ("train", "val", "test"):
        setattr(config, f"{split}_dataset", str(dataset_dir / split.title()))
        setattr(config, f"{split}_text_csv", str(dataset_dir / "prompt" / f"{split}.csv"))
    config.task_dataset = config.train_dataset
    config.experiment_name = spec["experiment"]
    config.session_name = f"{spec['experiment']}_seed{spec['seed']}"
    config.save_path = spec["run_dir"]
    config.model_path = str(Path(spec["run_dir"]) / "models")
    config.logger_path = str(Path(spec["run_dir"]) / "metrics.log")
    config.flops_batch_size = 32
    return {
        **spec, "model_kwargs": model_kwargs_from_config(config),
        "data": {name: getattr(config, name) for name in (
            "train_dataset", "val_dataset", "test_dataset",
            "train_text_csv", "val_text_csv", "test_text_csv")},
    }
