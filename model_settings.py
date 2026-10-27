"""Resolve a serializable snapshot of the LViTN constructor settings."""
import math


DEFAULT_ROUTER_CONFIG = {
    "mode": "off",
    "language": "all",
    "memory": "selective",
    "allow_none": True,
    "hidden_dim": 64,
    "temperature": 1.0,
    "residual_scale": 1.0,
}


def normalize_router_config(settings=None):
    """Resolve router options; omitted settings preserve pre-router checkpoints."""
    settings = {} if settings is None else dict(settings)
    unknown = settings.keys() - DEFAULT_ROUTER_CONFIG.keys()
    if unknown:
        raise ValueError(f"Unknown router settings: {sorted(unknown)}")
    result = {**DEFAULT_ROUTER_CONFIG, **settings}
    if result["mode"] not in {"off", "fixed", "uniform", "static", "image", "region"}:
        raise ValueError(f"Unknown router mode: {result['mode']}")
    if result["language"] not in {"all", "assigned"}:
        raise ValueError("Router language must be 'all' or 'assigned'.")
    if result["memory"] not in {"selective", "always", "off"}:
        raise ValueError("Router memory must be 'selective', 'always', or 'off'.")
    if not isinstance(result["allow_none"], bool):
        raise ValueError("Router allow_none must be a boolean.")
    if not isinstance(result["hidden_dim"], int) or result["hidden_dim"] < 1:
        raise ValueError("Router hidden_dim must be a positive integer.")
    for key in ("temperature", "residual_scale"):
        result[key] = float(result[key])
        if not math.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"Router {key} must be finite and positive.")
    return result


def model_kwargs_from_config(config):
    """Keep architecture settings in checkpoints, including behavior-only flags."""
    return {
        "n_channels": config.n_channels,
        "n_classes": config.n_labels,
        "img_size": config.img_size,
        "backbone_name": config.backbone,
        "text_encoder_name": config.text_encoder,
        "text_embed_dim": config.text_embed_dim,
        "bert_num_layers": 12,
        "num_text_groups": 4,
        "vit_depth": config.vit_depth,
        "vit_heads": config.vit_heads,
        "drop_rate": config.vit_drop_rate,
        "drop_path_rate": config.vit_drop_path_rate,
        "layerscale_init": 0.1,
        "use_freq": getattr(config, "use_freq", False),
        "freq_sigma_init": getattr(config, "freq_sigma_init", 5.0),
        "use_checkpoint": config.gradient_checkpointing,
        "use_prev_cross_attention": getattr(config, "use_prev_cross_attention", True),
        "use_adaptive_gated_fusion": getattr(config, "use_adaptive_gated_fusion", True),
        "deep_supervision": config.deep_supervision,
        "aspp_in_out_ch": config.aspp_in_out_ch,
        "aspp_dilations": tuple(config.aspp_dilations),
        "aspp_dropout": config.aspp_dropout,
        "router_config": normalize_router_config(getattr(config, "router_config", None)),
    }
