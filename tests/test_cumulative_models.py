"""Run in the ML environment; uses local tiny encoders, no pretrained downloads."""
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Config as config
import nets.LViT_new as architecture
import test_model as evaluation
from cumulative_experiments import VARIANTS, variant_settings, make_spec
from lightning_module import LViTLightningModule
from Load_Dataset import ImageToImage2D
from model_settings import CUMULATIVE_MODEL_DEFAULTS
from cumulative_worker import shared_initialization
from test_training_evaluation import tiny_settings, sample_batch, sample_loader


def model_settings(tiny_settings, variant, checkpointed=False):
    changes = variant_settings(variant)
    return {**tiny_settings, **{k: v for k, v in changes.items() if k in tiny_settings},
            "use_checkpoint": checkpointed}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("checkpointed", [False, True])
def test_each_variant_backward_and_checkpoint_roundtrip(tiny_settings, variant, checkpointed, tmp_path, monkeypatch):
    settings = model_settings(tiny_settings, variant, checkpointed)
    if not settings["use_text"]:
        def forbidden_encoder(**kwargs):
            raise AssertionError("An image-only variant must not construct a text encoder")
        monkeypatch.setattr(architecture, "TextEncoder", forbidden_encoder)
    treatment = variant_settings(variant)
    module = LViTLightningModule(model_kwargs=settings, backbone_pretrained=False,
                                 peal_loss_weight=treatment["peal_loss_weight"],
                                 deep_supervision_loss_weight=treatment["deep_supervision_loss_weight"])
    batch = sample_batch()
    if not settings["use_text"]:
        batch = {k: v for k, v in batch.items() if k in ("image", "label")}
    loss, _, _, _, terms = module._shared_step((batch, ["a", "b"]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(module.model.seg_head.weight.grad).all()
    if not settings["use_text"]:
        assert module.model.text_encoder is None
        assert terms["peal_loss"] == 0
    if not settings["deep_supervision"]:
        assert terms["deep_supervision_loss"] == 0
    if variant == "I0":
        assert all(b.self_attn is None for b in module.model.stage1.blocks)
    if variant == "I8":
        assert module.model.text_fusion.alpha.grad is None
        torch.testing.assert_close(module.model.text_fusion.alpha, torch.zeros_like(module.model.text_fusion.alpha))
    module.eval()
    text = batch if settings["use_text"] else None
    with torch.no_grad():
        expected = module.model(batch["image"], text)["main"]
        if not settings["use_text"]:
            torch.testing.assert_close(module.model(batch["image"], {"ignored": "text"})["main"], expected)
    assert expected.shape == (2, 1, 64, 64)
    path = tmp_path / "model.ckpt"
    torch.save({"hyper_parameters": {"model_kwargs": settings}, "state_dict": module.state_dict()}, path)
    restored, _ = evaluation.load_model("cpu", path)
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["image"], text)["main"], expected)


def test_text_sources_match_layer12_and_group_means(tiny_settings):
    hidden = torch.arange(12.0).view(1, 12, 1, 1).expand(2, 12, 5, 8)
    final = architecture.LViTN(None, backbone_pretrained=False,
                               **model_settings(tiny_settings, "I7"))
    for group in final._fuse_text(hidden):
        torch.testing.assert_close(group, hidden[:, -1])
    uniform = architecture.LViTN(None, backbone_pretrained=False,
                                 **model_settings(tiny_settings, "I8"))
    for i, group in enumerate(uniform._fuse_text(hidden)):
        torch.testing.assert_close(group, hidden[:, i*3:i*3+3].mean(1))


def test_image_dataset_needs_no_csv_or_tokenizer(tmp_path, monkeypatch):
    import numpy as np
    from transformers import AutoTokenizer
    for folder in ("Images", "GTs"):
        (tmp_path / folder).mkdir()
    evaluation.cv2.imwrite(str(tmp_path / "Images/x.png"), np.zeros((16, 16, 3), dtype=np.uint8))
    evaluation.cv2.imwrite(str(tmp_path / "GTs/x.png"), np.zeros((16, 16), dtype=np.uint8))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("Tokenizer used"))
    dataset = ImageToImage2D(str(tmp_path), image_size=64, use_text=False)
    sample, name = dataset[0]
    assert name == "x.png" and set(sample) == {"image", "label"}


def test_legacy_snapshot_defaults_keep_full_architecture(tiny_settings, tmp_path):
    model = architecture.LViTN(None, backbone_pretrained=False, **tiny_settings).eval()
    old_settings = {k: v for k, v in tiny_settings.items() if k not in CUMULATIVE_MODEL_DEFAULTS}
    path = tmp_path / "legacy.ckpt"
    torch.save({"hyper_parameters": {"model_kwargs": old_settings}, "state_dict": model.state_dict()}, path)
    restored, _ = evaluation.load_model("cpu", path)
    batch = sample_batch()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch["image"], batch)["main"], model(batch["image"], batch)["main"])


def test_common_initialization_reuses_weights_despite_constructor_rng_changes(tiny_settings, tmp_path):
    torch.manual_seed(42)
    first = architecture.LViTN(None, backbone_pretrained=False, **model_settings(tiny_settings, "I0"))
    path = tmp_path / "init.pt"
    shared_initialization(first, path)
    torch.manual_seed(1337)
    full = architecture.LViTN(None, backbone_pretrained=False, **model_settings(tiny_settings, "I11"))
    assert not torch.equal(first.stage1.patch_embed.proj.weight, full.stage1.patch_embed.proj.weight)
    shared_initialization(full, path)
    torch.testing.assert_close(first.stage1.patch_embed.proj.weight, full.stage1.patch_embed.proj.weight)
    torch.testing.assert_close(first.decoder1.transp_conv.conv.weight, full.decoder1.transp_conv.conv.weight)


def test_native_evaluation_ignores_saved_tta_and_writes_metrics(tiny_settings, tmp_path, monkeypatch):
    settings = model_settings(tiny_settings, "I6")
    model = architecture.LViTN(None, backbone_pretrained=False, **settings)
    (tmp_path / "models").mkdir()
    path = tmp_path / "models" / "best.ckpt"
    torch.save({"hyper_parameters": {"model_kwargs": settings}, "state_dict": model.state_dict()}, path)
    (tmp_path / "best_threshold.json").write_text(json.dumps({"threshold": 0.0, "use_tta": True}))
    monkeypatch.setattr(evaluation, "ImageToImage2D", lambda *a, **k: sample_loader().dataset)
    monkeypatch.setattr(evaluation, "tta_inference", lambda *a, **k: pytest.fail("TTA called"))
    monkeypatch.setattr(config, "task_name", "synthetic")
    result = evaluation.evaluate(path, protocol="native", batch_size=32,
                                 results_json=tmp_path / "test_result.json",
                                 save_predictions=False, device_name="cpu")
    assert result["threshold"] == 0.5 and result["use_tta"] is False
    assert result["test_cases"] == 2 and result["batch_size"] == 32
    assert (tmp_path / "test_cases.csv").exists()
    assert json.loads((tmp_path / "test_result.json").read_text()) == result


@pytest.mark.parametrize("variant", ["I0", "I11"])
def test_worker_real_fit_then_test_on_local_encoders(tiny_settings, tmp_path, monkeypatch, variant):
    import pytorch_lightning as pl
    import data_module
    import cumulative_worker

    class TinyData(pl.LightningDataModule):
        def __init__(self, **kwargs):
            super().__init__()
        def setup(self, stage=None):
            self.train_dataset = self.val_dataset = sample_loader().dataset
        def train_dataloader(self):
            return sample_loader()
        def val_dataloader(self):
            return sample_loader()

    monkeypatch.setattr(data_module, "LViTDataModule", TinyData)
    monkeypatch.setattr(evaluation, "ImageToImage2D", lambda *a, **k: sample_loader().dataset)
    spec = make_spec("qata", variant, 42, tmp_path / "data", tmp_path)
    spec["settings"].update(img_size=64, text_embed_dim=8, aspp_in_out_ch=32,
                            vit_depth=1, vit_heads=2, precision="32", epochs=2,
                            warmup_epochs=1, freeze_backbone_epochs=0,
                            vit_drop_path_rate=0.0, aspp_dropout=0.0, num_workers=0)
    # Fixture substitutes local encoders; the worker still exercises real Lightning,
    # losses, callbacks, best/last checkpoints, strict reload, and test CSV/JSON export.
    trained = cumulative_worker.train(spec, accelerator="cpu")
    assert Path(trained["checkpoint"]).is_file()
    assert (Path(spec["run_dir"]) / "models" / "last.ckpt").is_file()
    assert trained["epochs_completed"] == 2
    result = cumulative_worker.test(spec, device_name="cpu")
    assert result["checkpoint"] == trained["checkpoint"]
    assert result["protocol"] == "native" and result["test_cases"] == 2
