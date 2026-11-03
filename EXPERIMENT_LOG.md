# GRACE Experiment Log

Record of every experiment run for this project: what was run, with which settings, what came out, and where the logs are. Add a new entry after each experiment. The plan these experiments follow is in [GRACE_ROUTER_EXPERIMENTS.md](GRACE_ROUTER_EXPERIMENTS.md).

**Logs:** each experiment has a folder `experiment_logs/<dataset>/<YYYY-MM-DD_HHMM>_<name>/`. `.gitignore` excludes these folders (`qata/`, `MosMed/` and `*.log` match at any depth), so they exist only on the machine that ran them. Training outputs from `train_model.py` (checkpoints, CSV and TensorBoard logs) are written separately to `<dataset>/LViT/<session>/`.

**Environment for all entries so far:** NVIDIA RTX 4070 (12 GB), conda env `capstone` with Python 3.11, torch 2.11.0+cu128, pytorch-lightning 2.6.6, MONAI 1.6.1, transformers 5.21.0, timm 1.0.30. Code at commit `4c1eae8`, plus the uncommitted scripts in `tests/`.

## Index

| ID | Date | Experiment | Datasets | Result | Logs |
|---|---|---|---|---|---|
| [E01](#e01-dataset-validation) | 2026-10-28 | Dataset validation | qata, MosMed | PASS | [qata](experiment_logs/qata/2026-10-28_0013_dataset_check/), [MosMed](experiment_logs/MosMed/2026-10-28_0013_dataset_check/) |
| [E02](#e02-priority-0-feasibility-and-correctness) | 2026-10-27 | Priority 0: feasibility and correctness | qata, MosMed | PASS, 6/6 checks on both | [qata](experiment_logs/qata/2026-10-27_2358_priority0/), [MosMed](experiment_logs/MosMed/2026-10-27_2358_priority0/) |
| [E03](#e03-priority-0-at-batch-size-24) | 2026-10-28 | Priority 0 at batch size 24 (batch-size-dependent checks) | qata | PASS; router still needs checkpointing | [qata](experiment_logs/qata/2026-10-28_0029_priority0_bs24/) |
| [E04](#e04-priority-0-at-batch-size-16) | 2026-10-28 | Priority 0 at batch size 16 (batch-size-dependent checks) | qata | PASS; router fits without checkpointing | [qata](experiment_logs/qata/2026-10-28_0037_priority0_bs16/) |
| [E05](#e05-epoch-timing) | 2026-10-28 | Full-epoch timing and data-loader throughput | qata | Router 3.15 min/epoch, baseline 1.96 min/epoch (batch 32 with checkpointing) | [qata](experiment_logs/qata/2026-10-28_0045_epoch_timing/) |
| [E06](#e06-router_region-qata-seed-42) | 2026-10-28 | Full training: `router_region`, seed 42, batch 16 | qata | Best val Dice 0.8943 at epoch 132; early-stopped after 152 | [qata](experiment_logs/qata/2026-10-28_0123_router_region_seed42_bs16/) |
| [E07](#e07-parallel_uniform-qata-seed-42) | 2026-10-28 | Full training: `parallel_uniform`, seed 42, batch 16 | qata | Stopped by the user after 6 epochs (val Dice 0.8565); incomplete | [qata](experiment_logs/qata/2026-10-28_0801_parallel_uniform_seed42_bs16/) |
| [E08](#e08-test-router_region-qata-seed-42) | 2026-10-28 | Test-set evaluation of the E06 checkpoint | qata | Test Dice 0.9176 per-pixel, 0.8514 per-image | [qata](experiment_logs/qata/2026-10-28_0818_test_router_region_seed42/) |
| [E09](#e09-baseline-qata-seed-42) | 2026-11-01 | Full training: `baseline`, seed 42, batch 16, `min_delta=5e-4` | qata | Best val Dice 0.8932 at epoch 94; early-stopped after 108 | [qata](experiment_logs/qata/2026-11-01_1642_baseline_seed42/) |
| [E10](#e10-test-baseline-qata-seed-42) | 2026-11-01 | Test-set evaluation of the E09 checkpoint | qata | Test Dice 0.9166 per-pixel, 0.8501 per-image | [qata](experiment_logs/qata/2026-11-01_2015_test_baseline_seed42/) |
| [E11](#e11-parallel_uniform-qata-seed-42-rerun) | 2026-11-01 | Full training: `parallel_uniform`, seed 42, batch 16, `min_delta=5e-4` (rerun of E07) | qata | Best val Dice 0.8932 at epoch 86; early-stopped after 106 | [qata](experiment_logs/qata/2026-11-01_2023_parallel_uniform_seed42/) |
| [E12](#e12-test-parallel_uniform-qata-seed-42) | 2026-11-02 | Test-set evaluation of the E11 checkpoint | qata | Test Dice 0.9172 per-pixel, 0.8503 per-image | [qata](experiment_logs/qata/2026-11-02_0033_test_parallel_uniform_seed42/) |
| [E13](#e13-router_region-qata-seed-42-rerun) | 2026-11-02 | Full training: `router_region`, seed 42, batch 16, `min_delta=5e-4` (rerun of E06) | qata | Best val Dice 0.8928 at epoch 58; early-stopped after 78 | [qata](experiment_logs/qata/2026-11-02_0042_router_region_seed42/) |
| [E14](#e14-test-router_region-rerun-qata-seed-42) | 2026-11-03 | Test-set evaluation of the E13 checkpoint | qata | Test Dice 0.9179 per-pixel, 0.8518 per-image | [qata](experiment_logs/qata/2026-11-03_1752_test_router_region_seed42_rerun/) |
| [E15](#e15-router_region-depth-4-drop-path-03-mixup-qata-seed-42) | 2026-11-03 | `router_region` with `vit_depth=4`, `vit_drop_path_rate=0.3`, `use_mixup_cutmix=True` | qata | FAILED at the first training step (MixUp/CutMix guard) | [qata](experiment_logs/qata/2026-11-03_1937_router_region_seed42_depth4_dp03_mixup/) |
| [E16](#e16-router_region-vit_depth-4-qata-seed-42) | 2026-11-03 | `router_region` with `vit_depth=4`, otherwise identical to E13 | qata | FAILED: CUDA out of memory at the first training step | [qata](experiment_logs/qata/2026-11-03_2005_router_region_seed42_depth4/) |

---

## E01: Dataset validation

**Purpose:** confirm that both datasets match the layout `Config.py` and `Load_Dataset.py` expect before running anything on them.

**Command:** `python tests/check_dataset.py <dataset>`

**What it checks:**
- `Train/Val/Test` each contain `Images/` and `GTs/` folders.
- Every image pairs with a mask using the loader's own `sorted_image_mask_pairs`.
- Every image has a row in `prompt/<split>.csv`.
- No image appears in more than one split.
- Every file decodes, and each image is the same size as its mask.

**Result: PASS on both datasets.**

| Dataset | Train | Val | Test | Image size | Empty masks (Train/Val/Test) | Masks with gray edge pixels (Train/Val/Test) |
|---|---|---|---|---|---|---|
| qata | 6074 | 1071 | 2113 | 224×224 | 0 / 0 / 0 | 0 / 0 / 0 |
| MosMed | 1910 | 409 | 410 | 224×224 | 2 / 1 / 1 | 66 / 21 / 12 |

**Notes:**
- **Gray edge pixels:** 99 MosMed masks contain the values 12, 17 and 21 as well as 0/255. All of these pixels lie within 2 px of a lesion edge, so they are conversion anti-aliasing. The loader binarizes with `mask > 0`, which counts them as lesion. On average they are about 14% of the foreground pixels in those masks, and up to 28%. Thresholding at `> 127` would drop them. This is not changed yet.
- **Empty masks:** four MosMed masks are empty. Per-image Dice/IoU gives a correctly empty prediction a score of 1, so the empty-mask policy has to be stated when reporting results.
- **Earlier run:** the same check was first run on 2026-10-27 with an earlier script and gave identical results. This logged run replaces it.

---

## E02: Priority 0: feasibility and correctness

**Purpose:** run the six Priority 0 checks from [GRACE_ROUTER_EXPERIMENTS.md](GRACE_ROUTER_EXPERIMENTS.md#priority-0-feasibility-and-correctness) before any full training run.

**Command:** `python tests/check_priority0.py --tasks qata MosMed` (24 checks, each in its own process, 13 minutes in total)

**Settings:**
- **Models:** `baseline` and `router_region`. The mask check covers all nine routed presets.
- **Pretrained encoders:** ConvNeXt-tiny (`convnext_tiny.in12k_ft_in1k`) and CXR-BERT (`microsoft/BiomedVLP-CXR-BERT-specialized`), with BERT kept frozen.
- **Training settings:** seed 42, batch size 32, precision `16-mixed`.
- **Gradient checkpointing:** on (the `--checkpointing` flag) for every check except the memory probes.

**Result: PASS on both datasets.** All 20 required checks passed. The 4 runs without checkpointing only measure memory.

| Check | Required outcome | qata | MosMed | How it was tested and what was measured |
|---|---|---|---|---|
| Real-data forward/backward | Finite loss and gradients with the pretrained encoders | PASS | PASS | One fp32 step on 4 augmented training images. Loss was 2.45 (baseline) / 2.42 (router) on qata and 1.62 / 1.61 on MosMed. Every gradient was finite. Backbone, text fusion, ViT stages, decoder/heads and (for the router) router parameters all received non-zero gradients. BERT received no gradients. |
| Small-subset overfit | Both models can learn a small subset | PASS | PASS | 200 full-batch fp16 steps on 16 fixed training images at full LR (head 3e-4, backbone 3e-5). Eval-mode Dice went from 0.000 to 0.995 / 0.996 on qata and 0.000 to 1.000 / 0.999 on MosMed. Dice was already 0.96 or higher by step 50. |
| GPU mixed-precision smoke run | Stable gradients, no NaNs, enough memory at the selected batch size | PASS | PASS | 30 Lightning `16-mixed` steps at batch size 32, plus 4 validation batches, with the backbone unfrozen so memory is at its worst case. There were no non-finite gradients, the AMP loss scale stayed at 65536 with no skipped steps, all parameters stayed finite and validation loss was finite. Unscaled gradient norms were 0.8–1.9 on qata and 0.3–0.7 on MosMed. Peak memory is in the table below. |
| Routing masks | Stage-1 memory and disabled sources get exactly zero weight | PASS | PASS | 9 routed presets, run in fp32 and fp16, plus 5 inference-time interventions on `router_region`: 23 forward passes × 4 stages. The largest weight on any unavailable source was exactly 0, and so was the largest contribution from a masked source. Grid sizes were 56/28/14/7. Weights sum to 1. `fixed` and `uniform` produced their exact specified weights. `always_memory` contributes through memory with weight 0. |
| Checkpoint round trip | Restored predictions match at the same precision and inference settings | PASS | PASS | Trained 3 steps, saved a Lightning checkpoint, and restored it with `test_model.load_model`. All state tensors were identical (874 for baseline, 914 for router) and `router_config` was restored. Predictions matched bit for bit (largest difference 0) in fp32 and fp16, both single-pass and with the 9-pass TTA. |
| Last training batch | No single-image batch per device | PASS | PASS | Last training batch size at batch sizes 8 / 16 / 32: 2 / 10 / 26 on qata and 6 / 6 / 22 on MosMed, on 1 GPU. A forward pass on a single training image fails as the plan document predicts (`Expected more than 1 value per channel when training`, raised by the pooled ASPP BatchNorm). A single image in eval mode works. |

**Peak GPU memory at batch size 32, backbone unfrozen** (peak reserved, out of 11.6 GiB; the same on both datasets):

| Model | Checkpointing on | Checkpointing off |
|---|---|---|
| `baseline` | 6.6 GiB | 10.1 GiB |
| `router_region` | 7.6 GiB | out of memory |

**Decisions:**
- Run every comparison at **batch size 32 with `--checkpointing`**. The router needs checkpointing to fit, and the plan requires the same batch size and checkpointing policy for all conditions.

**Notes:**
- **Unused parameters:** 16 `text_gate` parameters receive no gradient in every preset except `adaptive_gate`, because the gate module is built even when `use_adaptive_gated_fusion=False`. This is harmless on one GPU. Multi-GPU (DDP) training would need `find_unused_parameters=True`.
- **MosMed validation at batch size 8:** the last validation batch would hold 1 image (409 = 51×8 + 1). That's harmless, because validation runs in eval mode, and a single image was checked in eval mode. Only training batches of one image fail.
- **Checkpoint size:** each round-trip checkpoint was 1.1 GiB. This was measured with the backbone frozen, so it doesn't include optimizer state for the backbone. Checkpoints saved after the backbone is unfrozen will be larger.

**Limitations:**
- **Smoke run length:** the smoke run covers only 30 steps of epoch 0, where the real schedule's warmup holds the learning rate at 1% of its base value. So it shows the start of training is stable. Stability at full learning rate is covered only by the 200-step overfit check, which ran in fp16 on 16 images. Long-run stability will be shown by the first full training run.
- **No accuracy results:** these are pilot checks and none of their numbers are accuracy results. The validation loss after 30 steps carries no information about model quality.

---

## E03: Priority 0 at batch size 24

**Purpose:** see whether batch size 24 works on qata, and whether it lets `router_region` train without gradient checkpointing.

**Command:** `python tests/check_priority0.py --tasks qata --checks smoke last_batch --batch-size 24 --name priority0_bs24`

**Settings:** the same as [E02](#e02-priority-0-feasibility-and-correctness) except batch size 24. Only the two checks that depend on batch size were rerun. The other four use fixed batch sizes (4, 16, 2 and 8 images), so their E02 results apply unchanged.

**Result: PASS.** Batch size 24 is stable and fits with checkpointing. `router_region` still runs out of memory without checkpointing.

| Check | Result | Details |
|---|---|---|
| GPU mixed-precision smoke run | PASS | 30 `16-mixed` steps at batch size 24 with the backbone unfrozen. No non-finite gradients, AMP loss scale steady at 65536 with no skipped steps, parameters finite. Gradient norms 0.8–1.9. Validation loss 1.870 (baseline) and 1.831 (router). |
| Last training batch | PASS | 6074 training images leave a final batch of 2 at batch size 24 (6074 = 253×24 + 2). The final validation batch holds 15 images. |

**Peak GPU memory, qata, backbone unfrozen** (peak reserved, out of 11.6 GiB):

| Model | Batch 24, ckpt on | Batch 24, ckpt off | Batch 32, ckpt on (E02) | Batch 32, ckpt off (E02) |
|---|---|---|---|---|
| `baseline` | 5.2 GiB | 7.8 GiB | 6.6 GiB | 10.1 GiB |
| `router_region` | 6.1 GiB | out of memory | 7.6 GiB | out of memory |

**Notes:**
- **Checkpointing:** batch size 24 does not remove the need for `--checkpointing`, so the E02 decision stands. If a run uses batch size 24 instead of 32, every comparison condition must use 24 too.
- **Convergence:** these results show only feasibility. Whether 24 or 32 trains better can only be judged from full training runs on validation data, never test data.

---

## E04: Priority 0 at batch size 16

**Purpose:** see whether batch size 16 works on qata, and whether it lets `router_region` train without gradient checkpointing.

**Command:** `python tests/check_priority0.py --tasks qata --checks smoke last_batch --batch-size 16 --name priority0_bs16`

**Settings:** the same as [E02](#e02-priority-0-feasibility-and-correctness) except batch size 16. As in [E03](#e03-priority-0-at-batch-size-24), only the two batch-size-dependent checks were rerun.

**Result: PASS.** Batch size 16 is stable. It is the first batch size tested at which **both models fit without checkpointing**.

| Check | Result | Details |
|---|---|---|
| GPU mixed-precision smoke run | PASS | 30 `16-mixed` steps at batch size 16 with the backbone unfrozen. No non-finite gradients, AMP loss scale steady at 65536 with no skipped steps, parameters finite. Gradient norms 0.7–2.0. Validation loss 1.826 (baseline) and 1.791 (router). |
| Last training batch | PASS | 6074 training images leave a final batch of 10 at batch size 16. The final validation batch holds 15 images. |

**Peak GPU memory on qata across batch sizes, backbone unfrozen** (peak reserved, out of 11.6 GiB; from E02, E03 and E04):

| Model | Checkpointing | Batch 16 | Batch 24 | Batch 32 |
|---|---|---|---|---|
| `baseline` | on | 4.1 GiB | 5.2 GiB | 6.6 GiB |
| `baseline` | off | 5.8 GiB | 7.8 GiB | 10.1 GiB |
| `router_region` | on | 4.7 GiB | 6.1 GiB | 7.6 GiB |
| `router_region` | off | 8.8 GiB | out of memory | out of memory |

**Notes:**
- **Two uniform policies now fit both models:** batch 32 with `--checkpointing`, or batch 16 without it. Either can be applied to every condition. Batch 16 without checkpointing leaves about 2.8 GiB free for the router, about 24% of the card, and anything else using the GPU (such as the desktop display) reduces that.
- **Speed:** the run times in the logs include model loading and data-loader start-up, so they can't be used to compare training speed between policies. That needs a separate throughput measurement.
- **Convergence:** as in E03, which batch size trains better can only be judged from full training runs on validation data.

---

## E05: Epoch timing

**Purpose:** measure how long one training epoch takes on qata for each candidate policy, to plan the full training runs.

**Commands:**
- `python tests/time_epoch.py --task qata --experiment <preset> --batch-size <N> [--checkpointing] [--frozen-backbone]`
- `python tests/time_dataloader.py --task qata --batch-size 32 --workers 2 4 8`

**Settings:**
- **What was timed:** one full epoch of Lightning training (6074 images) plus a full validation pass (1071 images), with the real data module, augmentations, `16-mixed` precision, `deterministic="warn"` and `num_workers=2`.
- **Not included:** loggers, checkpoint saving and the sanity-check validation.
- **Timing method:** the clock is synchronized with the GPU at each step boundary.

**Result** (`Epoch` includes validation):

| Model | Policy | Backbone | Epoch | Train / val | Median step | Throughput | Peak reserved |
|---|---|---|---|---|---|---|---|
| `router_region` | batch 32, ckpt on | unfrozen | **3.15 min** | 172 s / 10.5 s | 0.85 s | 37.6 img/s | 7.6 GiB |
| `baseline` | batch 32, ckpt on | unfrozen | **1.96 min** | 105 s / 6.9 s | 0.50 s | 64.3 img/s | 6.6 GiB |
| `router_region` | batch 16, ckpt off | unfrozen | **2.68 min** | 144 s / 10.8 s | 0.35 s | 45.5 img/s | 8.8 GiB |
| `baseline` | batch 16, ckpt off | unfrozen | **1.81 min** | 96 s / 7.1 s | 0.23 s | 71.2 img/s | 5.9 GiB |
| `router_region` | batch 32, ckpt on | frozen (epochs 0–4) | 2.92 min | 159 s / 10.5 s | 0.81 s | 39.8 img/s | 5.6 GiB |
| `baseline` | batch 32, ckpt on | frozen (epochs 0–4) | 1.74 min | 92 s / 7.0 s | 0.45 s | 70.4 img/s | 4.4 GiB |

**Data loader alone** (batch 32, augmentations and tokenization on, no model): 493 img/s with 2 workers, 889 with 4 and 1438 with 8. Even with 2 workers this is about 7× what training consumes, so the GPU is the bottleneck and more workers would not speed up training.

**Estimated length of a full training run** (201 epochs at most, of which the first 5 have the backbone frozen; early stopping with patience 20 may end runs earlier):

| Model | Batch 32, ckpt on | Batch 16, ckpt off |
|---|---|---|
| `router_region` | about 10.5 h | about 9.0 h* |
| `baseline` | about 6.5 h | about 6.1 h* |

\* The frozen-backbone phase was not timed for batch 16, so these figures use the unfrozen epoch time for all 201 epochs and are slight overestimates.

**Notes:**
- **Policy speed:** batch 16 without checkpointing is 15% faster per epoch for the router and 8% faster for the baseline, because checkpointing recomputes activations during the backward pass. It leaves less memory headroom (see [E04](#e04-priority-0-at-batch-size-16)).
- **Later epochs:** each measured epoch includes about 10 s of data-loader start-up. In real training the workers start once and are kept (`persistent_workers=True`), so epochs after the first should be slightly shorter.
- **MosMed (estimate, not measured):** with 31% as many training images and 38% as many validation images, an epoch should take about 1.0 min for the router and 0.6 min for the baseline, which is about 3.3 h and 2.0 h for 201 epochs.
- **Plan budget:** at the maximum epoch count, the 48 Priority-1 runs (8 presets × 2 datasets × 3 seeds) come to roughly 280 GPU-hours, about 12 days of continuous training on this card. This assumes the three baseline-type presets cost about as much as `baseline` and the five parallel/router presets about as much as `router_region`. Early stopping will shorten it by an amount that isn't known yet.

---

## E06: router_region, qata, seed 42

**Purpose:** the first full training run. It trains the `router_region` preset on qata with seed 42, using the batch-16 policy without checkpointing from [E04](#e04-priority-0-at-batch-size-16) and [E05](#e05-epoch-timing).

**Command:** `python train_model.py --experiment router_region --seed 42 --batch-size 16`. It was started from a terminal, so there is no stdout log; the per-epoch summaries are in `metrics.log`.

**Settings:** qata, batch size 16, gradient checkpointing off, `16-mixed` precision, up to 201 epochs. The backbone is frozen for the first 5 epochs, with 5 warm-up epochs and learning rate 3e-4. Early stopping watches `val/dice` with patience 20 and Lightning's default `min_delta=0`; `num_workers=2`. Everything else is as in `Config.py`; `run_config.json` has the full resolved settings. Code at commit `4c1eae8`.

**Result:** early stopping ended the run after epoch 152. The best val Dice was **0.8943** (IoU 0.8088), at epoch 132.

| Metric | qata |
|---|---|
| Best val Dice / IoU (epoch 132) | 0.8943 / 0.8088 |
| Val loss at the best epoch | 0.2906 (the lowest was 0.2500, at epoch 29) |
| Train loss / Dice at the best epoch | 0.0976 / 0.9621 |
| Epochs run | 152 of 201 |
| Chosen threshold (multi-scale and flip TTA) | 0.40, macro val Dice 0.8322 |
| Training time | 6 h 08 min, 2.42 min/epoch |
| Threshold sweep | 29.5 min, run on the CPU (see notes) |
| GPU memory (whole card, from `nvidia-smi`) | 9.3 / 12.0 GiB |

Best val Dice reached by epoch 10, 20, 40, 60, 80, 100, 120 and 140: 0.8737, 0.8848, 0.8900, 0.8905, 0.8913, 0.8923, 0.8935, 0.8943.

**Notes:**
- **Slow tail:** val Dice reached 0.8900 by epoch 38, and the remaining 114 epochs added only 0.0043. Ten small new bests, each 0.0001–0.0010, kept resetting the patience counter, because `EarlyStopping` counts any gain when `min_delta=0`. Over the same epochs val loss rose from 0.250 (epoch 29) to 0.300 and train loss fell to 0.087. So the model overfits by its loss while Dice still creeps up. Replaying this val Dice curve under Lightning's stopping rule with `min_delta=1e-3` (0.001 Dice): the run would have stopped after epoch 49 and kept the epoch-38 checkpoint (val Dice 0.8900, 0.0043 lower), about 4.2 h sooner. Epoch 38 never counted as an improvement because it beat epoch 29 (0.8895) by less than 0.001. With `min_delta=5e-4` it would have stopped after epoch 58 with the epoch-57 checkpoint (0.8904), 3.8 h sooner. **Decision (2026-11-01):** `Config.early_stopping_min_delta = 5e-4` from now on. It is passed to `EarlyStopping` and recorded in each run's `run_config.json`. E06 and E07 ran with 0, so E06 has to be rerun before it can be compared with runs that use the new setting. This needs deciding before the Priority-1 runs, because changing it partway through the set would make runs stop under different rules.
- **Two kinds of Dice:** `val/dice` (0.8943) is torchmetrics' `BinaryF1Score` pooled over every validation pixel. The threshold sweep's 0.8322 is the mean of per-image Dice with TTA, where an empty prediction on an empty mask scores 1. Compare runs on the same measure.
- **The threshold sweep ran on the CPU:** Lightning 2.6.6 moves the model to the CPU in `Trainer._teardown()` before it calls `on_fit_end`. As a result `ThresholdSweepCallback` ran 1071 images × 9 TTA passes on the CPU, taking 29.5 min instead of about 2 min. After this run, `callbacks.py` was fixed to move the module to `trainer.strategy.root_device` for the sweep, and `tests/test_training_evaluation.py` gained a regression test. [E07](#e07-parallel_uniform-qata-seed-42) and later runs include the fix.
- **Epoch summary display:** the early-stopping line in `metrics.log` is one epoch behind, because it is printed before `EarlyStopping` updates its counter. This affects the display only.
- **Aborted start:** a start 3 minutes earlier left `qata/LViT/router_region_seed42_10.28_01h20m39s_159592/` behind. It ran for 17 s and completed no epochs.

**Logs:** [experiment_logs/qata/2026-10-28_0123_router_region_seed42_bs16/](experiment_logs/qata/2026-10-28_0123_router_region_seed42_bs16/) holds copies of `metrics.log`, `metrics.csv`, `run_config.json` and `best_threshold.json`. Training outputs and the best checkpoint are in `qata/LViT/router_region_seed42_10.28_01h23m48s_757816/`.

---

## E07: parallel_uniform, qata, seed 42

**Status: stopped, incomplete.** It started automatically at 08:01 on 2026-10-28, when [E06](#e06-router_region-qata-seed-42) finished. At the user's request it was stopped at 08:15 with SIGINT, partway through epoch 7. Six epochs had finished, all with the backbone frozen. Val Dice was 0.8565 (IoU 0.7490) at epoch 6, compared with 0.8577 for E06 at the same epoch. No threshold sweep ran. The run needs to be repeated before it can be compared with E06.

**Command:** `python train_model.py --experiment parallel_uniform --seed 42 --batch-size 16`, run detached with stdout in `train.log`.

**Settings:** the same as E06 except for the preset (`router_config.mode = "uniform"`). The code includes the threshold-sweep device fix described in E06.

**Logs:** [experiment_logs/qata/2026-10-28_0801_parallel_uniform_seed42_bs16/](experiment_logs/qata/2026-10-28_0801_parallel_uniform_seed42_bs16/); training outputs in `qata/LViT/parallel_uniform_seed42_10.28_08h01m54s_729110/`.

---

## E08: Test router_region, qata, seed 42

**Purpose:** evaluate the best [E06](#e06-router_region-qata-seed-42) checkpoint (epoch 132) on the qata test set.

**Command:** `python test_model.py --checkpoint qata/LViT/router_region_seed42_10.28_01h23m48s_757816/models/best_model-LViT.ckpt`

**Settings:** 2113 test images in batches of 16, fp32. The script read threshold 0.40 and the TTA settings (sizes 192, 224 and 256, each with no flip, a horizontal flip and a vertical flip) from E06's `best_threshold.json`. The architecture comes from the checkpoint's own snapshot.

**Result:**

| Metric | qata test |
|---|---|
| Dice, per-pixel (global) | **0.9176** |
| Dice, per-image (macro) | 0.8514 |
| IoU, per-pixel (global) | 0.8477 |
| IoU, per-image (macro) | 0.7706 |
| Run time | 4 min 31 s on the GPU (2.04 s per batch of 16) |

**Notes:**
- **Not comparable with the validation numbers:** test per-pixel Dice (0.9176) is higher than E06's best val Dice (0.8943). Training validation used threshold 0.5 without TTA, while this test used threshold 0.40 with TTA, so the gap mixes the effect of TTA and the threshold with any difference between the two image sets. For comparison, the closest validation figure is the threshold sweep's 0.8322 per-image Dice, the same metric as the 0.8514 here.
- **Predictions:** 2113 binary masks in `test_predictions/qata/Test_session_10.28_08h18/`. That folder is named after `Config.session_name`, not the checkpoint, so later test runs will create separate timestamped folders.

**Logs:** [experiment_logs/qata/2026-10-28_0818_test_router_region_seed42/](experiment_logs/qata/2026-10-28_0818_test_router_region_seed42/) (`test.log`, `command.txt`).

---

## E09: baseline, qata, seed 42

**Purpose:** full training of the `baseline` preset (router off) on qata with seed 42. It is the reference that the router presets are compared against.

**Command:** `python train_model.py --experiment baseline --seed 42`, with stdout in `train.log`.

**Settings:** the same as [E06](#e06-router_region-qata-seed-42) except for the preset (`router_config.mode = "off"`) and the stopping rule, which is now `early_stopping_min_delta = 5e-4`. Batch size 16 (the new `Config` default), no checkpointing. Code at commit `4c1eae8` plus the uncommitted changes from E06 and later: the threshold-sweep device fix, `min_delta`, and the batch-16 default.

**Result:** early stopping ended the run after epoch 108. The best val Dice was **0.8932** (IoU 0.8071), at epoch 94.

| Metric | qata |
|---|---|
| Best val Dice / IoU (epoch 94) | 0.8932 / 0.8071 |
| Val loss at the best epoch | 0.2710 (the lowest was 0.2511, at epoch 27) |
| Train loss / Dice at the best epoch | 0.1367 / 0.9479 |
| Epochs run | 108 of 201 |
| Chosen threshold (multi-scale and flip TTA) | 0.45, macro val Dice 0.8328 |
| Training time | 2 h 45 min, 1.52 min/epoch |
| Threshold sweep | about 2 min on the GPU (E06's 29.5 min on the CPU confirms the fix) |

Best val Dice reached by epoch 10, 20, 40, 60, 80 and 100: 0.8782, 0.8858, 0.8890, 0.8900, 0.8910, 0.8932.

**Notes:**
- **Same overfitting pattern as E06:** val loss was lowest at epoch 27 and rose afterwards, while val Dice kept creeping up.
- **Aborted start:** a start 2 minutes earlier left `qata/LViT/baseline_seed42_11.01_16h40m01s_484273/` with one epoch.

**Logs:** [experiment_logs/qata/2026-11-01_1642_baseline_seed42/](experiment_logs/qata/2026-11-01_1642_baseline_seed42/) (`train.log`); training outputs and the best checkpoint in `qata/LViT/baseline_seed42_11.01_16h42m09s_721830/`.

---

## E10: Test baseline, qata, seed 42

**Purpose:** evaluate the best [E09](#e09-baseline-qata-seed-42) checkpoint (epoch 94) on the qata test set.

**Command:** `python test_model.py --checkpoint qata/LViT/baseline_seed42_11.01_16h42m09s_721830/models/best_model-LViT.ckpt`

**Settings:** the same as [E08](#e08-test-router_region-qata-seed-42). The threshold was 0.45, read from E09's `best_threshold.json`.

**Result** (the router_region results from E08 are shown alongside):

| Metric | `baseline` (E10) | `router_region` (E08) | Router minus baseline |
|---|---|---|---|
| Dice, per-pixel (global) | 0.9166 | 0.9176 | +0.0010 |
| Dice, per-image (macro) | 0.8501 | 0.8514 | +0.0013 |
| IoU, per-pixel (global) | 0.8460 | 0.8477 | +0.0017 |
| IoU, per-image (macro) | 0.7680 | 0.7706 | +0.0026 |
| Run time | 3 min 42 s | 4 min 31 s | |

**Notes:**
- **Not a fair comparison yet:** router_region (E06) trained with `min_delta=0` for 152 epochs, and the baseline (E09) with `min_delta=5e-4` for 108. Replaying E06 under the baseline's rule stops it after epoch 58 with val Dice 0.8904, below the baseline's 0.8932. So under the same rule the router is *behind* on validation, while under the unequal rules it is slightly ahead on test. Its lead of about 0.001 on test may come from the extra training rather than from the router. Rerun router_region with `min_delta=5e-4`, and use more seeds, before drawing a conclusion.
- **Predictions:** 2113 masks in `test_predictions/qata/Test_session_11.01_20h15/`.

**Logs:** [experiment_logs/qata/2026-11-01_2015_test_baseline_seed42/](experiment_logs/qata/2026-11-01_2015_test_baseline_seed42/) (`test.log`, `command.txt`).

---

## E11: parallel_uniform, qata, seed 42 (rerun)

**Purpose:** full training of `parallel_uniform` on qata with seed 42. It replaces the incomplete [E07](#e07-parallel_uniform-qata-seed-42).

**Command:** `python train_model.py --experiment parallel_uniform --seed 42`, run detached with stdout in `train.log`.

**Settings:** the same as [E09](#e09-baseline-qata-seed-42) (batch 16, no checkpointing, `min_delta=5e-4`) except for the preset (`router_config.mode = "uniform"`).

**Result:** early stopping ended the run after epoch 106. The best val Dice was **0.8932** (IoU 0.8071), at epoch 86.

| Metric | `parallel_uniform` (E11) | `baseline` (E09) |
|---|---|---|
| Best val Dice / IoU | 0.8932 / 0.8071 (epoch 86) | 0.8932 / 0.8071 (epoch 94) |
| Val loss at the best epoch | 0.2673 (the lowest was 0.2526, at epoch 42) | 0.2710 (the lowest was 0.2511, at epoch 27) |
| Epochs run | 106 | 108 |
| Chosen threshold, macro val Dice | 0.45, 0.8324 | 0.45, 0.8328 |
| Training time | 3 h 55 min, 2.22 min/epoch | 2 h 45 min, 1.52 min/epoch |
| Threshold sweep | 2.4 min on the GPU | about 2 min |
| GPU memory (whole card) | 8.1 / 12.0 GiB | not recorded |

Best val Dice reached by epoch 10, 20, 40, 60, 80 and 100: 0.8770, 0.8874, 0.8883, 0.8907, 0.8913, 0.8932.

**Notes:**
- **Tied with the baseline on validation:** the best val Dice matches the baseline to four decimals (0.893235 against 0.893240), even though the two runs took different paths to get there. For pixel-pooled metrics IoU equals Dice / (2 − Dice), so equal Dice also means equal IoU. Uniform parallel guidance gives no validation gain over the baseline for this seed, and it costs 46% more time per epoch.
- **The stopping rule in action:** at epoch 34 a gain of 0.0001 (0.8874 to 0.8875) updated the saved checkpoint but did not reset patience. Later gains of more than 0.0005, at epochs 36, 48, 56, 65 and 86, did reset it.

**Logs:** [experiment_logs/qata/2026-11-01_2023_parallel_uniform_seed42/](experiment_logs/qata/2026-11-01_2023_parallel_uniform_seed42/); training outputs in `qata/LViT/parallel_uniform_seed42_11.01_20h23m53s_652158/`.

---

## E12: Test parallel_uniform, qata, seed 42

**Purpose:** evaluate the best [E11](#e11-parallel_uniform-qata-seed-42-rerun) checkpoint (epoch 86) on the qata test set.

**Command:** `python test_model.py --checkpoint qata/LViT/parallel_uniform_seed42_11.01_20h23m53s_652158/models/best_model-LViT.ckpt`

**Settings:** the same as [E08](#e08-test-router_region-qata-seed-42). The threshold was 0.45, read from E11's `best_threshold.json`.

**Result** (all qata test results so far, seed 42):

| Metric | `baseline` (E10) | `parallel_uniform` (E12) | `router_region` (E08)* |
|---|---|---|---|
| Dice, per-pixel (global) | 0.9166 | 0.9172 | 0.9176 |
| Dice, per-image (macro) | 0.8501 | 0.8503 | 0.8514 |
| IoU, per-pixel (global) | 0.8460 | 0.8471 | 0.8477 |
| IoU, per-image (macro) | 0.7680 | 0.7688 | 0.7706 |
| Threshold | 0.45 | 0.45 | 0.40 |
| Run time | 3 min 42 s | 4 min 23 s | 4 min 31 s |

\* router_region was trained with `min_delta=0` for 152 epochs. The other two used `min_delta=5e-4`, so it is not yet comparable with them (see [E10](#e10-test-baseline-qata-seed-42)).

**Notes:**
- **parallel_uniform against baseline:** it is 0.0006 ahead on per-pixel Dice and 0.0002 ahead on per-image Dice. With one seed each, this is a tie, which matches their identical best val Dice. More seeds are needed to tell whether there is any real difference.
- **Predictions:** 2113 masks in `test_predictions/qata/Test_session_11.02_00h33/`.

**Logs:** [experiment_logs/qata/2026-11-02_0033_test_parallel_uniform_seed42/](experiment_logs/qata/2026-11-02_0033_test_parallel_uniform_seed42/) (`test.log`, `command.txt`).

---

## E13: router_region, qata, seed 42 (rerun)

**Purpose:** rerun [E06](#e06-router_region-qata-seed-42) under the current stopping rule, so that router_region can be compared fairly with [E09](#e09-baseline-qata-seed-42) and [E11](#e11-parallel_uniform-qata-seed-42-rerun).

**Command:** `python train_model.py --experiment router_region --seed 42`, run detached with stdout in `train.log`.

**Settings:** the same as E09 and E11 (batch 16, no checkpointing, `min_delta=5e-4`) except for the preset (`router_config.mode = "region"`).

**Result:** early stopping ended the run after epoch 78. The best val Dice was **0.8928** (IoU 0.8064), at epoch 58.

| Metric | qata |
|---|---|
| Best val Dice / IoU (epoch 58) | 0.8928 / 0.8064 |
| Val loss at the best epoch | 0.2607 (the lowest was 0.2513, at epoch 25) |
| Train loss / Dice at the best epoch | 0.1896 / 0.9306 |
| Epochs run | 78 of 201 |
| Chosen threshold, macro val Dice | 0.45, 0.8315 |
| Training time | 3 h 06 min, 2.38 min/epoch |
| Threshold sweep | 2.5 min on the GPU |

Best val Dice reached by epoch 10, 20, 40 and 60: 0.8763, 0.8855, 0.8891, 0.8928.

**Notes:**
- **Runs are not reproducible epoch for epoch:** the seed, data order and code are the same as E06 except for `min_delta`, but per-epoch val Dice differs from E06 by 0.0026 on average and by up to 0.0159 (during the frozen-backbone epochs). The cause is the non-deterministic attention backward passes (Flash and memory-efficient attention) that the run warns about under `deterministic="warn"`. Replaying E06's curve had predicted a stop after epoch 58 at 0.8904. The rerun instead reached 0.8928 at epoch 58 and ran to 78.
- **What this says about noise:** two runs of the same setup ended 0.0024 apart in best val Dice under one stopping rule (0.8904 replayed against 0.8928). That is larger than every difference between presets so far (0.0004 or less on val). It is only one pair of runs, but differences between presets need several seeds before they mean anything.

**Logs:** [experiment_logs/qata/2026-11-02_0042_router_region_seed42/](experiment_logs/qata/2026-11-02_0042_router_region_seed42/); training outputs in `qata/LViT/router_region_seed42_11.02_00h42m13s_311085/`.

---

## E14: Test router_region rerun, qata, seed 42

**Purpose:** evaluate the best [E13](#e13-router_region-qata-seed-42-rerun) checkpoint (epoch 58) on the qata test set. Together with E10 and E12, it gives the first comparison of the three presets under the same stopping rule.

**Command:** `python test_model.py --checkpoint qata/LViT/router_region_seed42_11.02_00h42m13s_311085/models/best_model-LViT.ckpt`

**Settings:** the same as [E08](#e08-test-router_region-qata-seed-42). The threshold was 0.45, read from E13's `best_threshold.json`.

**Result** (qata test, seed 42; all three trained with `min_delta=5e-4`):

| Metric | `baseline` (E10) | `parallel_uniform` (E12) | `router_region` (E14) | Router minus baseline |
|---|---|---|---|---|
| Dice, per-pixel (global) | 0.9166 | 0.9172 | **0.9179** | +0.0013 |
| Dice, per-image (macro) | 0.8501 | 0.8503 | **0.8518** | +0.0017 |
| IoU, per-pixel (global) | 0.8460 | 0.8471 | **0.8483** | +0.0023 |
| IoU, per-image (macro) | 0.7680 | 0.7688 | **0.7709** | +0.0029 |
| Best val Dice | 0.8932 | 0.8932 | 0.8928 | −0.0004 |
| Epochs run | 108 | 106 | 78 | |
| Run time | 3 min 42 s | 4 min 23 s | 4 min 32 s | |

**Notes:**
- **Ranking:** router_region > parallel_uniform > baseline on all four test metrics. On validation the order differs: the three are within 0.0004, with router_region slightly lowest. The margins are 0.001–0.003, and there is one seed per preset, so this is a consistent direction but not yet a significant result.
- **The two router_region runs compared on test:** E06 (`min_delta=0`, epoch-132 checkpoint) scored 0.9176 / 0.8514 per-pixel / per-image, and E13 (`min_delta=5e-4`, epoch-58 checkpoint) scored 0.9179 / 0.8518. The 74 extra epochs of the old rule gave no test benefit, which supports the `min_delta` change. The two runs differ by 0.0003–0.0004 on test, against 0.0024 on validation. On test, the router's lead over the baseline (+0.0013 / +0.0017) is about 4× that rerun gap. This is one pair of runs, so treat it as a first estimate of the noise.
- **Predictions:** 2113 masks in `test_predictions/qata/Test_session_11.03_17h52/`.

**Logs:** [experiment_logs/qata/2026-11-03_1752_test_router_region_seed42_rerun/](experiment_logs/qata/2026-11-03_1752_test_router_region_seed42_rerun/) (`test.log`, `command.txt`).

---

## E15: router_region, depth 4, drop path 0.3, mixup, qata, seed 42

**Purpose:** try router_region with a deeper cascade, more stochastic depth and MixUp/CutMix.

**Command:** `python train_model.py --experiment router_region --seed 42`, run detached with stdout in `train.log`.

**Settings:** the same as [E13](#e13-router_region-qata-seed-42-rerun) except for three `Config` changes: `vit_depth` 2 → 4, `vit_drop_path_rate` 0.2 → 0.3 and `use_mixup_cutmix` False → True.

**Result: FAILED** 18 s after launch, on the first training step, before any epoch finished: `ValueError: MixUp/CutMix is unsupported for paired image/text training` (`lightning_module.py:111`, in `_shared_step`). This guard is deliberate. MixUp and CutMix blend or paste images while each text prompt still describes only its original image, which breaks image–text correspondence. The user was told before launch and chose to run as is.

**Notes:** nothing was learned about `vit_depth=4` or `vit_drop_path_rate=0.3`, including whether depth 4 fits in 12 GB at batch 16 without checkpointing. Running with `use_mixup_cutmix = False` would test those two changes on their own.

**Logs:** [experiment_logs/qata/2026-11-03_1937_router_region_seed42_depth4_dp03_mixup/](experiment_logs/qata/2026-11-03_1937_router_region_seed42_depth4_dp03_mixup/) (`train.log` with the full traceback); empty session `qata/LViT/router_region_seed42_11.03_19h37m12s_134810/`.

---

## E16: router_region, vit_depth 4, qata, seed 42

**Result: FAILED** 18 s after launch (20:05), at the first training step of epoch 0, in the forward pass (`nets/LViT_new.py:480`, called from `lightning_module.py:116`): `torch.OutOfMemoryError: CUDA out of memory`. PyTorch had 11.10 GiB allocated, against the card's 11.62 GiB. This was with the backbone still frozen, which uses less memory than later epochs. At depth 2 the unfrozen peak at batch 16 without checkpointing was 8.8 GiB ([E04](#e04-priority-0-at-batch-size-16)), so depth 4 adds well over 2.3 GiB. **Depth 4 does not fit at batch 16 without gradient checkpointing.** No epochs finished.

**Purpose:** measure the effect of a deeper cascade (`vit_depth` 2 → 4 transformer blocks per stage, 8 → 16 in total) on router_region.

**Command:** `python train_model.py --experiment router_region --seed 42`, with `Config.vit_depth = 4`, run detached with stdout in `train.log`.

**Settings:** identical to [E13](#e13-router_region-qata-seed-42-rerun) except for `vit_depth`. This was checked by diffing the full resolved `run_config.json` against E13's. Batch 16, no checkpointing, `vit_drop_rate=0.0`, `vit_drop_path_rate=0.2` (now spread over 16 blocks instead of 8), `min_delta=5e-4`, mixup off. GPU memory at depth 4 without checkpointing was not tested beforehand.

**Logs:** [experiment_logs/qata/2026-11-03_2005_router_region_seed42_depth4/](experiment_logs/qata/2026-11-03_2005_router_region_seed42_depth4/); training outputs in `qata/LViT/router_region_seed42_11.03_20h05m12s_285704/`.

---

<!-- Template for new entries:
## E0N: <name>
**Purpose:** ...
**Command:** `...`
**Settings:** dataset(s), experiment preset, seed, batch size, precision, checkpointing, epochs; anything that differs from Config.py
**Result:** headline outcome
| Metric | qata | MosMed |
**Notes / decisions / limitations:** ...
**Logs:** experiment_logs/<dataset>/<folder>/ ; training outputs <dataset>/LViT/<session>/
-->
