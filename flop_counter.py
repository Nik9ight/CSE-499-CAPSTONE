from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception:
    def rank_zero_only(fn):
        return fn


@dataclass(frozen=True)
class FlopsReport:
    total_flops: float
    per_sample_flops: float
    batch_size: int
    includes_text_encoder: bool


def format_flops(flops: float) -> str:
    """Format a FLOP count with readable SI units."""
    flops = float(flops)
    abs_flops = abs(flops)
    for unit, scale in (
        ("TFLOPs", 1_000_000_000_000),
        ("GFLOPs", 1_000_000_000),
        ("MFLOPs", 1_000_000),
        ("KFLOPs", 1_000),
    ):
        if abs_flops >= scale:
            return f"{flops / scale:.2f} {unit}"
    return f"{flops:.0f} FLOPs"


def build_dummy_flops_inputs(
    cfg: Any,
    *,
    include_text_encoder: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, dict[str, torch.Tensor] | torch.Tensor]:
    """Build image and text inputs matching the training forward signature."""
    batch_size = int(getattr(cfg, "flops_batch_size", getattr(cfg, "batch_size", 1)))
    n_channels = int(getattr(cfg, "n_channels", 3))
    img_size = getattr(cfg, "img_size", 224)
    if isinstance(img_size, (tuple, list)):
        height, width = int(img_size[0]), int(img_size[1])
    else:
        height = width = int(img_size)

    text_tokens = int(getattr(cfg, "flops_text_tokens", 80))
    image = torch.zeros(
        (batch_size, n_channels, height, width),
        device=device,
        dtype=dtype,
    )

    if include_text_encoder:
        text = {
            "input_ids": torch.zeros(
                (batch_size, text_tokens), device=device, dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                (batch_size, text_tokens), device=device, dtype=torch.long,
            ),
        }
        return image, text

    text_layers = int(getattr(cfg, "flops_text_layers", 12))
    text_embed_dim = int(getattr(cfg, "text_embed_dim", 768))
    text_features = torch.zeros(
        (batch_size, text_layers, text_tokens, text_embed_dim),
        device=device,
        dtype=dtype,
    )
    return image, text_features


class _TokenizerTextForward(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.model(
            image,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return _main_output(out)


class _PrecomputedTextForward(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        out = self.model(image, text_features)
        return _main_output(out)


def _main_output(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        if "main" in out:
            return out["main"]
        for value in out.values():
            if torch.is_tensor(value):
                return value
        raise ValueError("Model output dict did not contain a tensor.")
    if isinstance(out, (tuple, list)):
        if not out:
            raise ValueError("Model output sequence was empty.")
        first = out[0]
        if torch.is_tensor(first):
            return first
    if torch.is_tensor(out):
        return out
    raise ValueError(f"Unsupported model output type for FLOPs tracing: {type(out)}")


def _rank_zero_print(message: str) -> None:
    rank_zero_only(print)(message)


def _load_flop_count_analysis():
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception as exc:
        raise RuntimeError(
            "fvcore is required for FLOPs reporting. Install it with `pip install fvcore`."
        ) from exc
    return FlopCountAnalysis


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cpu")
    dtype = torch.float32
    for tensor in module.parameters():
        device = tensor.device
        if tensor.is_floating_point():
            dtype = tensor.dtype
            return device, dtype
    for tensor in module.buffers():
        device = tensor.device
        if tensor.is_floating_point():
            dtype = tensor.dtype
            return device, dtype
    return device, dtype


@contextmanager
def _preserve_training_state(module: nn.Module):
    states = {child: child.training for child in module.modules()}
    try:
        module.eval()
        yield
    finally:
        for child, was_training in states.items():
            child.training = was_training


def _estimate_total_flops(
    model: nn.Module,
    inputs: tuple[Any, ...],
    flop_analysis_cls: Callable[[nn.Module, tuple[Any, ...]], Any],
) -> float:
    analysis = flop_analysis_cls(model, inputs)
    return float(analysis.total())


def _print_estimate(
    printer: Callable[[str], None],
    *,
    label: str,
    total_flops: float,
    batch_size: int,
) -> None:
    printer(f"[FLOPs] {label} (batch_size={batch_size}): {format_flops(total_flops)}")
    printer(f"[FLOPs] Per sample: {format_flops(total_flops / batch_size)}")


def report_model_flops(
    lightning_or_model: nn.Module,
    cfg: Any,
    *,
    flop_analysis_cls: Callable[[nn.Module, tuple[Any, ...]], Any] | None = None,
    printer: Callable[[str], None] | None = None,
) -> FlopsReport | None:
    """Estimate and print forward FLOPs for the configured profiling batch.

    This function is intentionally non-fatal: any profiling failure is printed as a
    warning and returns None so training can continue.
    """
    if not bool(getattr(cfg, "show_flops", True)):
        return None

    emit = printer or _rank_zero_print
    target_model = getattr(lightning_or_model, "model", lightning_or_model)

    try:
        analysis_cls = flop_analysis_cls or _load_flop_count_analysis()
        batch_size = int(getattr(cfg, "flops_batch_size", getattr(cfg, "batch_size", 1)))
        device, dtype = _module_device_dtype(target_model)
        include_text_encoder = bool(getattr(cfg, "flops_include_text_encoder", True))

        with _preserve_training_state(target_model), torch.no_grad():
            if include_text_encoder:
                try:
                    image, text = build_dummy_flops_inputs(
                        cfg,
                        include_text_encoder=True,
                        device=device,
                        dtype=dtype,
                    )
                    total_flops = _estimate_total_flops(
                        _TokenizerTextForward(target_model),
                        (image, text["input_ids"], text["attention_mask"]),
                        analysis_cls,
                    )
                    _print_estimate(
                        emit,
                        label="Estimated forward including text encoder",
                        total_flops=total_flops,
                        batch_size=batch_size,
                    )
                    return FlopsReport(
                        total_flops=total_flops,
                        per_sample_flops=total_flops / batch_size,
                        batch_size=batch_size,
                        includes_text_encoder=True,
                    )
                except Exception as exc:
                    emit(
                        "[FLOPs] Could not estimate full forward including the "
                        f"text encoder: {exc}"
                    )
                    emit(
                        "[FLOPs] Retrying with precomputed text features; text "
                        "encoder FLOPs will be excluded."
                    )
            else:
                emit(
                    "[FLOPs] Using precomputed text features; text encoder FLOPs "
                    "are excluded by config."
                )

            image, text_features = build_dummy_flops_inputs(
                cfg,
                include_text_encoder=False,
                device=device,
                dtype=dtype,
            )
            total_flops = _estimate_total_flops(
                _PrecomputedTextForward(target_model),
                (image, text_features),
                analysis_cls,
            )
            _print_estimate(
                emit,
                label="Estimated forward with precomputed text features",
                total_flops=total_flops,
                batch_size=batch_size,
            )
            return FlopsReport(
                total_flops=total_flops,
                per_sample_flops=total_flops / batch_size,
                batch_size=batch_size,
                includes_text_encoder=False,
            )
    except Exception as exc:
        emit(f"[FLOPs] Warning: FLOPs estimation failed; continuing training. {exc}")
        return None
