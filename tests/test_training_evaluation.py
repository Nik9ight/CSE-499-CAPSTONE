"""Regression checks using real attention/decoder/losses and tiny local encoders."""
import inspect
import json
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

import Config as config
import bertem
import nets.LViT_new as architecture
import test_model as evaluation
from callbacks import ThresholdSweepCallback
from lightning_module import LViTLightningModule
from model_settings import model_kwargs_from_config
from nets.ViT2 import CascadeViTBlock, CascadeViTStage
from Load_Dataset import ImageToImage2D
from transformers import AutoTokenizer


torch.set_num_threads(1)


class LocalTextEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, input_ids, attention_mask):
        basis = torch.arange(1, 9, device=input_ids.device).float()
        tokens = torch.sin(input_ids.float().unsqueeze(-1) * basis)
        return tokens.unsqueeze(1).expand(-1, 12, -1, -1)


class LocalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Conv2d(3, 8, 4, stride=4),
            nn.Conv2d(8, 16, 2, stride=2),
            nn.Conv2d(16, 32, 2, stride=2),
            nn.Conv2d(32, 64, 2, stride=2),
        ])

    def forward(self, image):
        features = []
        for stage in self.stages:
            image = stage(image)
            features.append(image)
        return features


@pytest.fixture
def tiny_settings(monkeypatch):
    monkeypatch.setattr(architecture, "TextEncoder", LocalTextEncoder)
    monkeypatch.setattr(
        architecture, "get_backbone_features",
        lambda *args, **kwargs: (LocalBackbone(), [8, 16, 32, 64]),
    )
    monkeypatch.setattr(config, "use_mixup_cutmix", False)
    settings = model_kwargs_from_config(config)
    settings.update(img_size=64, text_embed_dim=8, vit_depth=1, vit_heads=2,
                    aspp_in_out_ch=32, drop_path_rate=0.0, aspp_dropout=0.0)
    return settings


def sample_batch():
    torch.manual_seed(31)
    return {
        "image": torch.randn(2, 3, 64, 64),
        "label": torch.randint(0, 2, (2, 1, 64, 64)).float(),
        "input_ids": torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
    }


def sample_loader():
    batch = sample_batch()
    items = [({key: value[i] for key, value in batch.items()}, f"{i}.png")
             for i in range(2)]
    return DataLoader(items, batch_size=2)


def test_snapshot_covers_all_constructor_options():
    parameters = set(inspect.signature(architecture.LViTN.__init__).parameters)
    assert set(model_kwargs_from_config(config)) == parameters - {
        "self", "config", "backbone_pretrained",
    }


def test_dataset_uses_checkpoint_tokenizer(tmp_path, monkeypatch):
    import numpy as np

    (tmp_path / "Images").mkdir()
    (tmp_path / "GTs").mkdir()
    evaluation.cv2.imwrite(str(tmp_path / "Images/sample.png"),
                           np.full((20, 20, 3), 128, dtype=np.uint8))
    evaluation.cv2.imwrite(str(tmp_path / "GTs/sample.png"),
                           np.full((20, 20), 255, dtype=np.uint8))
    csv_path = tmp_path / "prompts.csv"
    csv_path.write_text("Image,Description\nsample.png,a finding\n", encoding="utf-8")
    selected = []

    def local_snapshot(name, lib_dir):
        selected.append(name)
        return str(tmp_path)

    class Tokenizer:
        def __call__(self, text, **kwargs):
            ids = torch.zeros(1, kwargs["max_length"], dtype=torch.long)
            ids[:, :2] = torch.tensor([1, 2])
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    monkeypatch.setattr(bertem, "_ensure_hf_model_local", local_snapshot)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(config, "text_encoder", "wrong-current-encoder")
    dataset = ImageToImage2D(str(tmp_path), image_size=64, text_csv_path=str(csv_path),
                             text_encoder_name="saved-encoder")
    sample, name = dataset[0]
    assert selected == ["saved-encoder"]
    assert name == "sample.png"
    assert sample["image"].shape == (3, 64, 64)
    assert sample["label"].shape == (1, 64, 64)
    assert sample["input_ids"].shape == (80,)
    assert sample["attention_mask"].sum() == 2


def test_attention_ignores_padding_but_uses_valid_tokens():
    torch.manual_seed(3)
    block = CascadeViTBlock(8, 2, text_dim=8).eval()
    image = torch.randn(2, 4, 8)
    text = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    changed = text.clone()
    changed[mask == 0] = torch.randn_like(changed[mask == 0]) * 100
    expected = block(image, text, text_mask=mask)
    torch.testing.assert_close(block(image, changed, text_mask=mask), expected)
    changed[:, 0] += torch.randn_like(changed[:, 0]) * 10
    assert not torch.allclose(block(image, changed, text_mask=mask), expected)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_stage_masks_padding_during_backward(checkpointed):
    stage = CascadeViTStage(4, 8, 8, 1, 2, text_dim=8, use_freq=True,
                            use_checkpoint=checkpointed)
    image = torch.randn(2, 8, 4, 4, requires_grad=True)
    text = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]])
    out = stage(image, text, text_mask=mask)
    (out * torch.randn_like(out)).sum().backward()
    assert torch.isfinite(image.grad).all()
    assert torch.isfinite(text.grad).all()
    assert torch.count_nonzero(text.grad[mask == 0]) == 0
    assert torch.count_nonzero(text.grad[mask == 1]) > 0


def test_full_model_propagates_mask_to_all_stages(tiny_settings):
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings).eval()
    batch = sample_batch()
    text = {key: batch[key] for key in ("input_ids", "attention_mask")}
    changed = {key: value.clone() for key, value in text.items()}
    changed["input_ids"][text["attention_mask"] == 0] = 91
    with torch.no_grad():
        expected = model(batch["image"], text)
        actual = model(batch["image"], changed)
    assert expected["main"].shape == (2, 1, 64, 64)
    torch.testing.assert_close(actual["main"], expected["main"])
    for actual_text, expected_text in zip(actual["peal_text"], expected["peal_text"]):
        torch.testing.assert_close(actual_text, expected_text)


@pytest.mark.parametrize("router_mode", ["off", "region"])
def test_multiscale_tta_preserves_padding_mask(tiny_settings, router_mode):
    tiny_settings["router_config"]["mode"] = router_mode
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings).eval()
    batch = sample_batch()
    text = {key: batch[key].clone() for key in ("input_ids", "attention_mask")}
    expected = evaluation.tta_inference(model, batch["image"], text, sizes=(32, 64))
    text["input_ids"][text["attention_mask"] == 0] = 91
    actual = evaluation.tta_inference(model, batch["image"], text, sizes=(32, 64))
    assert actual.shape == (2, 1, 64, 64)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_training_preserves_pairs_and_backpropagates(tiny_settings):
    module = LViTLightningModule(model_kwargs=tiny_settings, backbone_pretrained=False)
    batch = sample_batch()
    original = {key: value.clone() for key, value in batch.items()}
    loss, preds, targets, bs, terms = module._shared_step((batch, ["a", "b"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert terms["deep_supervision_loss"] > 0
    assert torch.isfinite(module.model.seg_head.weight.grad).all()
    assert torch.isfinite(module.model.text_fusion.alpha.grad).all()
    assert bs == 2
    assert set(targets.tolist()) == {0, 1}
    for key in batch:
        torch.testing.assert_close(batch[key], original[key])


def test_unsafe_batch_mixing_is_rejected(tiny_settings, monkeypatch):
    module = LViTLightningModule(model_kwargs=tiny_settings, backbone_pretrained=False)
    monkeypatch.setattr(config, "use_mixup_cutmix", True)
    with pytest.raises(ValueError, match="paired image/text"):
        module._shared_step((sample_batch(), ["a", "b"]))


def test_threshold_selection_averages_images_and_handles_empty_masks(tmp_path):
    callback = ThresholdSweepCallback(str(tmp_path), candidates=(0.3, 0.5, 0.7))
    probs = torch.tensor([[[[0.4, 0.2]]], [[[0.1, 0.1]]]])
    masks = torch.tensor([[[[1, 0]]], [[[0, 0]]]])
    threshold, dice = callback._sweep(probs, masks)
    assert threshold == 0.3
    assert dice == 1.0
    # One missed foreground image and one correct empty image: macro Dice = 0.5.
    callback.candidates = (0.5,)
    assert callback._sweep(probs, masks) == (0.5, 0.5)


def test_calibration_writes_settings_and_restores_mode(tiny_settings, tmp_path):
    module = LViTLightningModule(model_kwargs=tiny_settings, backbone_pretrained=False)
    trainer = SimpleNamespace(is_global_zero=True, callbacks=[],
                              datamodule=SimpleNamespace(val_dataloader=sample_loader))
    callback = ThresholdSweepCallback(str(tmp_path), use_tta=False, tta_sizes=(64,))
    callback.on_fit_end(trainer, module)
    saved = json.loads((tmp_path / "best_threshold.json").read_text())
    assert saved["metric"] == "macro_dice"
    assert saved["use_tta"] is False
    assert saved["tta_sizes"] == [64]
    assert module.training


def test_calibration_empty_loader_restores_mode(tiny_settings, tmp_path):
    module = LViTLightningModule(model_kwargs=tiny_settings, backbone_pretrained=False)
    trainer = SimpleNamespace(is_global_zero=True, callbacks=[],
                              datamodule=SimpleNamespace(val_dataloader=lambda: []))
    callback = ThresholdSweepCallback(str(tmp_path), use_tta=False)
    with pytest.raises(ValueError, match="nonempty"):
        callback.on_fit_end(trainer, module)
    assert module.training


@pytest.mark.parametrize("router_mode", ["off", "region"])
def test_checkpoint_roundtrip_survives_config_changes(tiny_settings, tmp_path, monkeypatch, router_mode):
    tiny_settings["router_config"]["mode"] = router_mode
    module = LViTLightningModule(model_kwargs=tiny_settings, backbone_pretrained=False,
                                 freeze_backbone_epochs=0, warmup_epochs=1, max_epochs=2)
    trainer = pl.Trainer(
        default_root_dir=tmp_path, accelerator="cpu", devices=1, max_epochs=1,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=sample_loader(), val_dataloaders=sample_loader())
    checkpoint_path = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    module.eval()
    batch = sample_batch()
    text = {key: batch[key] for key in ("input_ids", "attention_mask")}
    with torch.no_grad():
        expected = module.model(batch["image"], text)["main"]
    monkeypatch.setattr(config, "backbone", "does-not-exist")
    monkeypatch.setattr(config, "text_encoder", "does-not-exist")
    monkeypatch.setattr(config, "img_size", 256)
    monkeypatch.setattr(config, "use_freq", False)
    monkeypatch.setattr(config, "use_adaptive_gated_fusion", True)
    restored, _ = evaluation.load_model("cpu", checkpoint_path)
    lightning_restored = LViTLightningModule.load_from_checkpoint(
        checkpoint_path, backbone_pretrained=False,
    ).eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["image"], text)["main"], expected)
        torch.testing.assert_close(lightning_restored.model(batch["image"], text)["main"], expected)
    assert restored.img_size == 64
    assert restored.text_encoder_name == tiny_settings["text_encoder_name"]
    assert restored.router_config["mode"] == router_mode


def test_pre_router_snapshot_preserves_legacy_behavior(tiny_settings, tmp_path, monkeypatch):
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings).eval()
    old_settings = dict(tiny_settings)
    old_settings.pop("router_config")
    path = tmp_path / "old_snapshot.ckpt"
    torch.save({"hyper_parameters": {"model_kwargs": old_settings},
                "state_dict": model.state_dict()}, path)
    monkeypatch.setattr(config, "router_config", {"mode": "region"})
    restored, _ = evaluation.load_model("cpu", path)
    assert restored.router_config["mode"] == "off"
    batch = sample_batch()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["image"], batch)["main"],
                                   model(batch["image"], batch)["main"])


def test_full_router_maps_and_interventions(tiny_settings):
    tiny_settings["router_config"]["mode"] = "region"
    tiny_settings["vit_depth"] = 2
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings).eval()
    batch = sample_batch()
    with torch.no_grad():
        out = model(batch["image"], batch, return_routing=True)
        null = model(batch["image"], batch, return_routing=True, routing_override="none")
    assert out["main"].shape == (2, 1, 64, 64)
    for stage, grid in enumerate((16, 8, 4, 2), start=1):
        weights = out["routing"][f"stage{stage}"]["weights"]
        assert weights.shape == (2, 2, grid, grid, 6)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, grid, grid))
        assert torch.count_nonzero(null["routing"][f"stage{stage}"]["contribution_norms"]) == 0
    assert torch.count_nonzero(out["routing"]["stage1"]["weights"][..., 4]) == 0
    assert not torch.allclose(out["main"], null["main"])


def test_legacy_weights_require_explicit_architecture(tiny_settings, tmp_path):
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings).eval()
    path = tmp_path / "legacy.pth.tar"
    torch.save({"state_dict": model.state_dict()}, path)
    with pytest.raises(ValueError, match="no architecture snapshot"):
        evaluation.load_model("cpu", path)
    restored, _ = evaluation.load_model(
        "cpu", path, model_kwargs=json.loads(json.dumps(tiny_settings)),
    )
    assert restored.img_size == 64
    assert not restored.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_conflicting_snapshot_is_rejected(tiny_settings, tmp_path):
    path = tmp_path / "snapshot.ckpt"
    torch.save({"hyper_parameters": {"model_kwargs": tiny_settings}, "state_dict": {}}, path)
    override = dict(tiny_settings, use_adaptive_gated_fusion=True)
    with pytest.raises(ValueError, match="conflicts"):
        evaluation.load_model("cpu", path, model_kwargs=override)


def test_checkpoint_path_is_required(monkeypatch):
    monkeypatch.setattr(config, "test_checkpoint", None)
    with pytest.raises(ValueError, match="--checkpoint"):
        evaluation.load_model("cpu")


@pytest.mark.parametrize("calibrated_tta", [False, True])
@pytest.mark.parametrize("routing_override", [None, "none"])
def test_evaluation_uses_saved_preprocessing_and_calibration(
    tiny_settings, tmp_path, monkeypatch, calibrated_tta, routing_override,
):
    if routing_override is not None:
        tiny_settings["router_config"]["mode"] = "region"
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    checkpoint_path = model_dir / "model.ckpt"
    torch.save({"hyper_parameters": {"model_kwargs": tiny_settings},
                "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()}},
               checkpoint_path)
    (tmp_path / "best_threshold.json").write_text(json.dumps({
        "threshold": 0.0, "use_tta": calibrated_tta, "tta_sizes": [32, 64],
    }))
    captured = {}

    def dataset(*args, **kwargs):
        captured.update(kwargs)
        return sample_loader().dataset

    tta_calls = []
    original_tta = evaluation.tta_inference

    def recorded_tta(model, images, text, sizes=None, **kwargs):
        tta_calls.append(tuple(sizes))
        return original_tta(model, images, text, sizes=sizes, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(evaluation, "ImageToImage2D", dataset)
    monkeypatch.setattr(evaluation, "tta_inference", recorded_tta)
    monkeypatch.setattr(config, "use_tta", not calibrated_tta)
    monkeypatch.setattr(config, "img_size", 256)
    monkeypatch.setattr(config, "text_encoder", "wrong-tokenizer")
    monkeypatch.setattr(config, "task_name", "synthetic")
    monkeypatch.setattr(config, "session_name", "evaluation")
    evaluation.evaluate(checkpoint_path, routing_override=routing_override)
    assert tta_calls == ([(32, 64)] if calibrated_tta else [])
    assert captured["image_size"] == 64
    assert captured["text_encoder_name"] == tiny_settings["text_encoder_name"]
    prediction_dir = tmp_path / "test_predictions/synthetic/evaluation"
    if routing_override is not None:
        prediction_dir = prediction_dir / routing_override
    predictions = list(prediction_dir.glob("*.png"))
    assert len(predictions) == 2
    for path in predictions:
        mask = evaluation.cv2.imread(str(path), 0)
        assert mask.shape == (64, 64)
        assert (mask == 255).all()
