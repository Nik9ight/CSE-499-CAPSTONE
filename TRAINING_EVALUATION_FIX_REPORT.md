# Training and Evaluation Fix Report

Date: 2026-10-22

## Scope

This change addresses the training-quality and evaluation problems identified in the repository review of **GRACE** (implemented under legacy LViT identifiers). It preserves the model's parameter layout while correcting text masking, paired-data handling, checkpoint reconstruction, and threshold calibration. It does not establish a Dice/IoU improvement without a dataset experiment.

## Changes and Rationale

### 1. Preserve image, mask, and description correspondence

Previously, batch-level MixUp/CutMix changed images and masks while retaining only the original description. Both text-guided segmentation and PEAL alignment therefore received mismatched pairs.

`Config.use_mixup_cutmix` now defaults to `False`, and the image-only mixing implementation has been removed from `lightning_module.py`. Re-enabling the flag raises an actionable error during training instead of silently restoring mismatched supervision. Existing per-sample dataset transforms remain active. This is a conservative fix; it avoids inventing a text-mixing objective without validating it.

Training metric targets are now explicitly thresholded at 0.5 instead of truncated to integers. The segmentation loss continues to receive floating-point targets.

### 2. Exclude padded text from visual cross-attention

`LViTN.forward()` now passes the tokenizer's attention mask through all four cascade stages and every transformer block. `nn.MultiheadAttention` receives `key_padding_mask=(text_mask == 0)`: `True` means the position is ignored.

The mask also travels through the gradient-checkpointing path. Precomputed embeddings can supply the optional `text_mask` argument; tensors without a mask retain the legacy assumption that all positions are valid. PEAL pooling continues to respect padding.

Files: `nets/LViT_new.py`, `nets/ViT2.py`.

### 3. Reconstruct checkpoints from saved architecture settings

`model_settings.py` resolves the complete LViTN constructor configuration. `LViTLightningModule` saves it as `hyper_parameters.model_kwargs`, including behavior-only switches such as gated fusion. Loss weights, auxiliary weights, and schedule length are also saved and accepted as constructor arguments.

Standalone evaluation reads this snapshot and strictly loads model weights. It initializes the visual backbone without downloading pretrained backbone weights that the checkpoint will replace. Image size, embedding dimension, and tokenizer identity come from the restored model; `Load_Dataset.py` accepts an explicit text encoder name.

Both Lightning-prefixed and raw legacy state dictionaries use the same loader. The obsolete `text_group_order` call and hardcoded checkpoint path have been removed. Missing or conflicting architecture metadata produces an explicit error. Older checkpoints require their original constructor settings through `--model-config`; missing behavior settings cannot reliably be recovered from weights alone.

Files: `model_settings.py`, `lightning_module.py`, `test_model.py`, `Load_Dataset.py`, `Config.py`.

### 4. Make threshold selection valid and reproducible

The threshold sweep now computes Dice for every validation image and averages those scores before selecting the best candidate. Correctly empty prediction/target pairs score 1, matching standalone per-image evaluation. This fixes the previous `.item()` call on a multi-element tensor.

Calibration writes `metric`, `use_tta`, and `tta_sizes` alongside the threshold and score. Evaluation reuses those augmentation settings. The training callback now respects `use_tta` separately from the multi-scale flag, so single-scale flip augmentation is calibrated consistently too.

Calibration runs on global rank zero, reports an empty validation loader clearly, and restores the original training/evaluation mode even if calibration fails. Checkpoint selection still monitors global validation Dice at 0.5; threshold calibration optimizes per-image macro Dice afterward.

Files: `callbacks.py`, `train_model.py`, `test_model.py`.

## Usage and Compatibility

Train normally:

```text
python train_model.py
```

Evaluate a new checkpoint:

```text
python test_model.py --checkpoint qata/LViT/<session>/models/best_model-LViT.ckpt
```

Alternatively, set `Config.test_checkpoint`. Dataset and prompt paths remain configured in `Config.py`.

For an older checkpoint, restore the **original training architecture settings** in `Config.py`, then export a JSON snapshot with Python:

```python
import json
import Config as config
from model_settings import model_kwargs_from_config

with open("training_architecture.json", "w", encoding="utf-8") as handle:
    json.dump(model_kwargs_from_config(config), handle, indent=2)
```

Then run:

```text
python test_model.py --checkpoint <old-checkpoint> --model-config training_architecture.json
```

Only matching legacy state dictionaries are supported; structural incompatibilities still fail strict loading. Corrected attention changes predictions for padded descriptions, including when using old weights. Previously selected thresholds should be recalibrated on validation data under the corrected inference behavior. Retraining is required to measure the training-quality changes.

## Validation

Regression tests are in `tests/test_training_evaluation.py` and are explicitly allowed by `.gitignore`.

```text
python -m pytest tests/test_training_evaluation.py -q
```

The tests use real attention, frequency enhancement, MONAI decoder blocks, losses, and Lightning, with small local substitutes for BERT and ConvNeXt. They cover padding invariance and gradients, paired-input preservation, rejection of unsafe mixing, macro threshold selection, calibration metadata and mode restoration, checkpoint round trips after configuration changes, legacy loading, and synthetic evaluation output.

Result: **18 passed**, with three non-failing warnings, in 55.04 seconds. Source compilation also passed. The warnings concern worker counts in the synthetic loaders and PyTorch's scheduler epoch-argument deprecation.

The checkpoint test executes a real Lightning training/validation epoch with one synthetic batch, saves the checkpoint, changes the global backbone, text encoder, image size, and architecture flags, and verifies matching predictions after both standalone and Lightning reloads. Synthetic evaluation checks saved tokenizer/image-size selection, threshold application, TTA settings, and exported mask dimensions and values.

Validation environment: Python 3.12.3; PyTorch 2.8.0+cpu; torchvision 0.23.0+cpu; MONAI 1.5.1; timm 1.0.20; PyTorch Lightning 2.5.5; TorchMetrics 1.8.2; transformers 4.57.1; pytest 9.1.1. Dependencies were installed in the ignored `.venv/` directory. Tests ran with Hugging Face offline mode enabled and did not download pretrained model weights. These versions describe the test environment; project dependency pins were not changed.

## Remaining Limits

No real dataset or production checkpoint is present in this checkout, so this report contains no measured accuracy improvement. Full GPU/mixed-precision and multi-process validation remain necessary in the training environment. The previously identified singleton-batch ASPP BatchNorm issue and repeated BERT encoding during TTA remain separate follow-up work.
