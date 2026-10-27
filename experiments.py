"""Named GRACE router comparisons; datasets and common training settings stay in Config."""
from datetime import datetime
from pathlib import Path

from model_settings import model_kwargs_from_config, normalize_router_config


EXPERIMENTS = {
    "baseline": {"mode": "off"},
    "baseline_no_memory": {"mode": "off", "previous_attention": False},
    "adaptive_gate": {"mode": "off", "adaptive_gate": True},
    "parallel_fixed": {"mode": "fixed"},
    "parallel_uniform": {"mode": "uniform"},
    "router_static": {"mode": "static"},
    "router_image": {"mode": "image"},
    "router_region": {"mode": "region"},
    "router_no_memory": {"mode": "region", "memory": "off"},
    "router_always_memory": {"mode": "region", "memory": "always"},
    "router_assigned_text": {"mode": "region", "language": "assigned"},
    "router_no_none": {"mode": "region", "allow_none": False},
    "baseline_no_peal": {"mode": "off", "peal_weight": 0.0},
    "router_no_peal": {"mode": "region", "peal_weight": 0.0},
}


def configure_run(config, experiment=None, seed=None, batch_size=None,
                  checkpointing=False, router_scale=None, router_temperature=None):
    """Apply an explicit preset and give the run its own output paths."""
    if experiment is not None:
        spec = dict(EXPERIMENTS[experiment])
        config.use_prev_cross_attention = spec.pop("previous_attention", True)
        config.use_adaptive_gated_fusion = spec.pop("adaptive_gate", False)
        if "peal_weight" in spec:
            config.peal_loss_weight = spec.pop("peal_weight")
        config.router_config = normalize_router_config(spec)
    config.router_config = normalize_router_config(getattr(config, "router_config", None))
    if router_scale is not None:
        config.router_config["residual_scale"] = router_scale
    if router_temperature is not None:
        config.router_config["temperature"] = router_temperature
    config.router_config = normalize_router_config(config.router_config)
    if seed is not None:
        config.seed = seed
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        config.batch_size = batch_size
        config.flops_batch_size = batch_size
    if checkpointing:
        config.gradient_checkpointing = True
    config.experiment_name = experiment or "manual"
    stamp = datetime.now().strftime("%m.%d_%Hh%Mm%Ss_%f")
    config.session_name = f"{config.experiment_name}_seed{config.seed}_{stamp}"
    run_dir = Path(config.task_name) / config.model_name / config.session_name
    config.save_path = str(run_dir)
    config.model_path = str(run_dir / "models")
    config.logger_path = str(run_dir / "metrics.log")
    config.tensorboard_folder = str(run_dir / "tensorboard_logs")
    config.visualize_path = str(run_dir / "visualize_val")
    return {
        "architecture": "GRACE",
        "experiment": config.experiment_name,
        "model_kwargs": model_kwargs_from_config(config),
        "training": {key: getattr(config, key) for key in (
            "seed", "batch_size", "epochs", "learning_rate", "warmup_epochs",
            "freeze_backbone_epochs", "precision", "num_workers", "deterministic",
            "peal_loss_weight", "deep_supervision_loss_weight", "aux_loss_weights",
            "early_stopping_patience", "use_mixup_cutmix",
        )},
        "evaluation": {key: getattr(config, key) for key in (
            "use_tta", "use_ms_tta", "ms_tta_sizes", "threshold_sweep",
        )},
        "data": {key: getattr(config, key) for key in (
            "task_name", "train_dataset", "val_dataset", "test_dataset",
            "train_text_csv", "val_text_csv", "test_text_csv",
        )},
        "save_path": config.save_path,
    }
