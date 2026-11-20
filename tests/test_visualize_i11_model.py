"""Analytic Grad-CAM checks and a full tiny-GRACE checkpoint/figure export."""
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import visualize_i11 as xai


class AnalyticModel(torch.nn.Module):
    def __init__(self, negative=False, bias=0):
        super().__init__()
        self.router_config = {"mode": "region"}
        self.decoder1 = torch.nn.Identity()
        self.negative, self.bias = negative, bias

    def forward(self, image, text, return_routing=False):
        feature = self.decoder1(image)
        logit = (feature[:, :1] + 2 * feature[:, 1:2]) * (-1 if self.negative else 1) + self.bias
        return {"main": logit, "routing": {}}


def test_cam_matches_analytic_derivative_and_preserves_prediction():
    model = AnalyticModel().eval()
    image = torch.arange(1, 33, dtype=torch.float32).reshape(1, 2, 4, 4)
    expected = model(image, None)["main"].detach()
    result = xai.explain_case(model, image, None)
    np.testing.assert_allclose(result["probability"], expected[0, 0].sigmoid())
    # Mean regional logit gives derivative (1, 2) / 16 at every feature location.
    raw = expected[0, 0].numpy() / 16
    np.testing.assert_allclose(result["cam_raw"], raw)
    np.testing.assert_allclose(result["cam"], raw / raw.max())
    assert result["target_mode"] == "predicted_foreground"
    assert result["target_pixels"] == 16
    assert not model.decoder1._forward_hooks
    torch.testing.assert_close(model(image, None)["main"], expected)


def test_empty_prediction_targets_one_pixel_and_zero_cam_is_explicit():
    model = AnalyticModel(negative=True).eval()
    image = torch.arange(1, 33, dtype=torch.float32).reshape(1, 2, 4, 4)
    result = xai.explain_case(model, image, None)
    assert result["target_mode"] == "max_logit_pixel_empty_prediction"
    assert result["target_pixels"] == 1
    assert result["target_mask"][0, 0]
    assert result["cam_status"] == "no_positive_cam_response"
    assert np.isfinite(result["cam"]).all() and not result["cam"].any()


def test_cam_hook_removed_after_failed_forward():
    model = AnalyticModel(bias=float("nan")).eval()
    with pytest.raises(ValueError, match="non-finite logits"):
        xai.explain_case(model, torch.ones(1, 2, 4, 4), None)
    assert not model.decoder1._forward_hooks


def test_binary_metrics_empty_and_false_positive():
    assert xai.binary_metrics(np.zeros((2, 2)), np.zeros((2, 2)))["dice"] == 1
    assert xai.binary_metrics(np.ones((2, 2)), np.zeros((2, 2)))["iou"] == 0


def test_real_i11_checkpoint_to_figures(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import cv2
    import bertem
    import nets.LViT_new as architecture
    from cumulative_experiments import SEEDS, make_spec, variant_settings
    from experiment_io import atomic_json
    from model_settings import model_kwargs_from_config
    import Config as config
    from test_training_evaluation import LocalBackbone, LocalTextEncoder
    from transformers import AutoTokenizer

    monkeypatch.setattr(architecture, "TextEncoder", LocalTextEncoder)
    monkeypatch.setattr(architecture, "get_backbone_features",
                        lambda *a, **kw: (LocalBackbone(), [8, 16, 32, 64]))
    settings = model_kwargs_from_config(config)
    settings.update({key: value for key, value in variant_settings("I11").items() if key in settings})
    settings.update(img_size=64, text_embed_dim=8, vit_depth=2, vit_heads=2,
                    aspp_in_out_ch=32, drop_path_rate=0.0, aspp_dropout=0.0, use_checkpoint=True)
    model = architecture.LViTN(None, backbone_pretrained=False, **settings).eval()
    study = tmp_path / "study"
    for seed, score in zip(SEEDS, (0.8, 0.9, 0.85)):
        spec = make_spec("qata", "I11", seed, tmp_path / "data", study)
        run_dir = Path(spec["run_dir"])
        path = run_dir / "models/best.ckpt"
        path.parent.mkdir(parents=True)
        atomic_json(run_dir / "spec.json", spec)
        atomic_json(run_dir / "train_result.json", {"best_val_global_dice": score, "checkpoint": str(path)})
        if seed == 1337:
            torch.save({"hyper_parameters": {"model_kwargs": settings},
                        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()}}, path)

    dataset_dir = tmp_path / "data/qata"
    for subdir in ("Test/Images", "Test/GTs", "prompt"):
        (dataset_dir / subdir).mkdir(parents=True)
    pixels = np.linspace(0, 255, 64 * 64 * 3).reshape(64, 64, 3).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 15:45] = 255
    cv2.imwrite(str(dataset_dir / "Test/Images/example.png"), pixels)
    cv2.imwrite(str(dataset_dir / "Test/GTs/example.png"), mask)
    (dataset_dir / "prompt/test.csv").write_text("Image,Description\nexample.png,test finding\n")
    monkeypatch.setattr(bertem, "_ensure_hf_model_local", lambda *a: str(tmp_path))

    class Tokenizer:
        def __call__(self, text, **kwargs):
            ids = torch.zeros(1, kwargs["max_length"], dtype=torch.long)
            ids[:, :2] = torch.tensor([1, 2])
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
    # Independently establish prediction parity before the explanation hook.
    from Load_Dataset import ImageToImage2D
    dataset = ImageToImage2D(str(dataset_dir / "Test"), image_size=64,
                             text_csv_path=str(dataset_dir / "prompt/test.csv"))
    sample, _ = dataset[0]
    text = {k: sample[k].unsqueeze(0) for k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        expected = model(sample["image"].unsqueeze(0), text)["main"].sigmoid()[0, 0].numpy()

    output = tmp_path / "figures"
    xai.main(["--study-root", str(study), "--dataset", "qata", "--output-dir", str(output),
              "--device", "cpu", "--num-cases", "1", "--dpi", "45"])
    manifest = json.loads((output / "selection.json").read_text())
    assert manifest["seed"] == 1337 and manifest["status"] == "completed"
    assert manifest["use_tta"] is False and manifest["threshold"] == 0.5
    case_dir = output / "0001_example"
    for name in ("summary.png", "summary.pdf", "prediction.png", "routing_stage1.png",
                 "routing_stage2.png", "routing_stage3.png", "routing_stage4.png", "case.json"):
        assert (case_dir / name).stat().st_size > 100
    with np.load(case_dir / "maps.npz", allow_pickle=False) as maps:
        np.testing.assert_allclose(maps["probability"], expected, rtol=1e-5, atol=1e-6)
        assert maps["stage1_weights"].shape == (2, 16, 16, 6)
        np.testing.assert_allclose(maps["stage1_weights"].sum(-1), 1, atol=1e-6)
        assert not maps["stage1_weights"][..., 4].any()
        assert not maps["stage4_contribution_norms"][..., 5].any()
        assert maps["grad_cam"].shape == (64, 64) and np.isfinite(maps["grad_cam"]).all()
    assert (output / "cases.csv").exists()
    with pytest.raises(ValueError, match="not empty"):
        xai.main(["--study-root", str(study), "--dataset", "qata", "--output-dir", str(output),
                  "--device", "cpu"])
