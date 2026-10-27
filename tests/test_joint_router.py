"""Behavioral checks for GRACE's joint routing and experimental controls."""
import copy
from types import SimpleNamespace

import pytest
import torch

import Config as config
from experiments import EXPERIMENTS, configure_run
from model_settings import normalize_router_config
from nets.routing import JointGuidanceRouter
from nets.ViT2 import CascadeViTBlock, CascadeViTStage


torch.set_num_threads(1)


def router_inputs():
    torch.manual_seed(19)
    return torch.randn(2, 7, 8), torch.randn(2, 7, 6, 8)


@pytest.mark.parametrize("mode", ["fixed", "uniform", "static", "image", "region"])
def test_availability_masks_are_exact(mode):
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": mode, "allow_none": False}, 2)
    if hasattr(router, "source_bias"):
        with torch.no_grad():
            router.source_bias[4:] = 1000
    _, weights, norms = router(visual, candidates, False, return_diagnostics=True)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 7))
    assert torch.count_nonzero(weights[..., 4:]) == 0
    assert torch.count_nonzero(norms[..., 4:]) == 0
    assert torch.isfinite(weights).all()


def test_fixed_assignment_and_guidance_scale():
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": "fixed", "residual_scale": 2.0}, 2)
    update, weights, _ = router(visual, candidates, True, return_diagnostics=True)
    torch.testing.assert_close(update, candidates[..., 2, :] + candidates[..., 4, :])
    assert (weights[..., 2] == 0.5).all() and (weights[..., 4] == 0.5).all()


@pytest.mark.parametrize("memory", ["selective", "always", "off"])
def test_memory_controls_have_distinct_semantics(memory):
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": "uniform", "memory": memory,
                                     "allow_none": False}, 0)
    update, weights, norms = router(visual, candidates, True, return_diagnostics=True)
    if memory == "selective":
        expected = candidates[..., :5, :].mean(dim=2)
        assert (weights[..., 4] > 0).all()
    elif memory == "always":
        expected = candidates[..., :4, :].mean(dim=2) + candidates[..., 4, :]
        assert torch.count_nonzero(weights[..., 4]) == 0
        assert (norms[..., 4] > 0).all()
    else:
        expected = candidates[..., :4, :].mean(dim=2)
        assert torch.count_nonzero(weights[..., 4]) == 0
        assert torch.count_nonzero(norms[..., 4]) == 0
    torch.testing.assert_close(update, expected)


def test_assigned_language_masks_other_groups():
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": "region", "language": "assigned",
                                     "memory": "off", "allow_none": False}, 1)
    update, weights, _ = router(visual, candidates, False, return_diagnostics=True)
    torch.testing.assert_close(update, candidates[..., 1, :])
    assert (weights[..., 1] == 1).all()


def test_none_is_zero_even_if_supplied_candidate_is_nonzero():
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": "region"}, 0).eval()
    update, weights, norms = router(visual, candidates, True, override="none",
                                    return_diagnostics=True)
    assert torch.count_nonzero(update) == 0
    assert torch.count_nonzero(norms) == 0
    assert (weights[..., 5] == 1).all()


@pytest.mark.parametrize("mode", ["static", "image", "region"])
def test_conditioning_and_spatial_scope(mode):
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": mode}, 0)
    _, weights, _ = router(visual, candidates, True, return_diagnostics=True)
    _, changed, _ = router(-visual, candidates.roll(1, dims=2), True, return_diagnostics=True)
    if mode == "static":
        torch.testing.assert_close(weights, changed)
    else:
        assert not torch.allclose(weights, changed)
    if mode in {"static", "image"}:
        torch.testing.assert_close(weights, weights[:, :1].expand_as(weights))
    else:
        assert not torch.allclose(weights, weights[:, :1].expand_as(weights))


def test_inference_interventions_preserve_valid_distributions():
    visual, candidates = router_inputs()
    router = JointGuidanceRouter(8, {"mode": "region"}, 0).eval()
    _, original, _ = router(visual, candidates, False, return_diagnostics=True)
    for override in ("uniform", "fixed", "image_mean", "shuffle", "none"):
        _, weights, _ = router(visual, candidates, False, override=override,
                                return_diagnostics=True)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 7))
        assert torch.count_nonzero(weights[..., 4]) == 0
        if override == "image_mean":
            torch.testing.assert_close(weights, original.mean(1, keepdim=True).expand_as(weights))
        if override == "shuffle":
            torch.testing.assert_close(weights.mean(1), original.mean(1))
    router.train()
    with pytest.raises(ValueError, match="evaluation-only"):
        router(visual, candidates, False, override="uniform")


@pytest.mark.parametrize("checkpointed", [False, True])
def test_masked_group_gradients_and_memory(checkpointed):
    stage = CascadeViTStage(
        4, 8, 8, 2, 2, prev_dim=4, text_dim=8, use_freq=True,
        use_checkpoint=checkpointed, use_adaptive_gated_fusion=False,
        router_config={"mode": "region"},
    )
    image = torch.randn(2, 8, 4, 4, requires_grad=True)
    text = torch.randn(2, 4, 5, 8, requires_grad=True)
    previous = torch.randn(2, 20, 4, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    out, weights, norms = stage(image, text, previous, mask, return_routing=True)
    (out * torch.randn_like(out)).sum().backward()
    assert weights.shape == (2, 2, 16, 6)
    assert not weights.requires_grad and not norms.requires_grad
    padding = (mask[:, None, :, None] == 0).expand_as(text)
    assert torch.count_nonzero(text.grad[padding]) == 0
    for value in (image.grad, text.grad, previous.grad):
        assert torch.isfinite(value).all() and torch.count_nonzero(value) > 0
    for block in stage.blocks:
        for param in block.router.parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all()
        assert torch.count_nonzero(block.router.score.weight.grad) > 0


def test_checkpointed_forward_and_gradients_match():
    plain = CascadeViTStage(4, 8, 8, 1, 2, text_dim=8, use_adaptive_gated_fusion=False,
                             router_config={"mode": "region"})
    checkpointed = copy.deepcopy(plain)
    checkpointed.use_checkpoint = True
    image = torch.randn(2, 8, 4, 4)
    text = torch.randn(2, 4, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    target = torch.randn(2, 16, 8)
    results = []
    for stage in (plain, checkpointed):
        x, t = image.clone().requires_grad_(), text.clone().requires_grad_()
        output = stage(x, t, text_mask=mask)
        (output * target).sum().backward()
        results.append((output.detach(), x.grad, t.grad))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected)


def test_cpu_autocast_has_finite_forward_and_gradients():
    stage = CascadeViTStage(4, 8, 8, 1, 2, text_dim=8, use_adaptive_gated_fusion=False,
                            router_config={"mode": "region"})
    image = torch.randn(2, 8, 4, 4, requires_grad=True)
    text = torch.randn(2, 4, 5, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = stage(image, text, text_mask=mask)
        loss = (output * torch.randn_like(output)).sum()
    loss.backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(image.grad).all() and torch.isfinite(text.grad).all()


def test_text_attention_parameters_are_shared_and_padding_invariant():
    block = CascadeViTBlock(8, 2, text_dim=8, use_adaptive_gated_fusion=False,
                            router_config={"mode": "region"}).eval()
    x = torch.randn(2, 9, 8)
    text = torch.randn(2, 4, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])
    calls = []
    hook = block.text_cross_attn.register_forward_hook(lambda module, args, out: calls.append(id(module)))
    expected = block(x, text, text_mask=mask)
    hook.remove()
    assert calls == [id(block.text_cross_attn)] * 4
    changed = text.clone()
    changed[(mask[:, None, :, None] == 0).expand_as(changed)] = 100
    torch.testing.assert_close(block(x, changed, text_mask=mask), expected)


def test_unsafe_inputs_are_rejected():
    with pytest.raises(ValueError, match="alternative"):
        CascadeViTBlock(8, 2, router_config={"mode": "region"})
    block = CascadeViTBlock(8, 2, text_dim=8, use_adaptive_gated_fusion=False,
                            router_config={"mode": "region"})
    with pytest.raises(ValueError, match="unmasked"):
        block(torch.randn(2, 7, 8), torch.randn(2, 4, 5, 8), text_mask=torch.zeros(2, 5))
    with pytest.raises(ValueError, match="positive"):
        normalize_router_config({"mode": "region", "temperature": 0})


def test_presets_record_actual_settings_and_distinct_paths():
    paths = set()
    for name in EXPERIMENTS:
        cfg = SimpleNamespace(**{key: getattr(config, key) for key in dir(config)
                                 if not key.startswith("__")})
        snapshot = configure_run(cfg, name, seed=17, batch_size=2, checkpointing=True)
        assert snapshot["training"]["seed"] == 17
        assert snapshot["model_kwargs"]["router_config"] == cfg.router_config
        assert cfg.gradient_checkpointing and cfg.batch_size == 2
        assert cfg.save_path not in paths
        paths.add(cfg.save_path)
