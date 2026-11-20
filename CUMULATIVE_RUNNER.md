# Run the 72 cumulative experiments on a cloud GPU

`run_cumulative.sh` runs **I0–I11 × qata/MosMed × seeds 42, 1337, 2026**. Both training and testing use **batch size 32**. Each training process finishes before a separate process tests its best validation checkpoint. The consolidated result is `<output-root>/results.csv`, with one row per run, including pending/failed/completed status.

## Start

Use a CUDA-enabled PyTorch environment with the dependencies in `requirements.txt` installed. Copy the project code and both datasets to the cloud machine. Run from a terminal session that will remain alive, such as tmux. Use persistent storage for the output directory.

Expected data layout, including case-sensitive dataset and folder names:

```text
/workspace/data/
  qata/
    Train/Images/    Train/GTs/
    Val/Images/      Val/GTs/
    Test/Images/     Test/GTs/
    prompt/train.csv
    prompt/val.csv
    prompt/test.csv
  MosMed/
    ...same layout...
```

Prompt CSVs need `Image,Description` columns, with `Image` matching each image filename, including its extension. Image-only models themselves never load prompts or BERT; the batch preflight checks prompts because later rows require them.

```bash
# List all 72 resolved configurations; works without PyTorch or local datasets.
bash run_cumulative.sh --dry-run

# Verify dataset/prompt files, CUDA, and tiny-model regression tests.
# This does not start any full dataset training run.
bash run_cumulative.sh \
  --data-root /workspace/data \
  --output-root /workspace/results/cumulative72 \
  --preflight-only

# Launch all 72 train → test pairs.
bash run_cumulative.sh \
  --data-root /workspace/data \
  --output-root /workspace/results/cumulative72
```

If the environment's Python is not `python3`, select it explicitly:

```bash
PYTHON=/opt/conda/bin/python bash run_cumulative.sh \
  --data-root /workspace/data \
  --output-root /workspace/results/cumulative72
```

The runner uses one visible GPU, respecting `CUDA_VISIBLE_DEVICES`. It does not provision a cloud instance or install dependencies. Preflight also runs the cumulative model regression tests using tiny local encoders, including two miniature CPU training/evaluation integrations. A failed preflight stops before the full study. Pretrained weights for the real runs are fetched by the existing model/tokenizer loaders into the project's configured `lib/` cache when absent.

## What runs

| ID | Addition / treatment |
|---|---|
| I0 | Image-only ConvNeXt, common feature interface, UNETR decoder, bilinear output |
| I1 | Add visual self-attention/MLP refinement |
| I2 | Add previous-stage memory |
| I3 | Add frequency enhancement |
| I4 | Add ASPP |
| I5 | Add high-resolution stems and learned upsampling |
| I6 | Add deep supervision; strong image-only reference |
| I7 | Add frozen BERT final-layer text guidance |
| I8 | Use fixed uniform mixtures in four text-depth groups |
| I9 | Learn the within-group mixtures |
| I10 | Add PEAL |
| I11 | Add joint regional routing; full architecture |

The recipe is resolved by `cumulative_experiments.py`, independently of edits to common training values in `Config.py`: depth 2, image size 224, 201 maximum epochs, five warmup/frozen-backbone epochs, LR 3e-4 for heads and 3e-5 for the backbone, early-stopping patience 20 and min_delta 5e-4. Every row trains independently. Shared untrained parameter tensors are cached by seed/name/shape across variants; pretrained backbone/BERT weights are excluded from that cache.

**Batch-32 protocol choices:** gradient checkpointing is enabled for every row, and training uses `16-mixed`. Use `--no-checkpointing` or `--precision bf16-mixed` only as a consistent whole-study choice with a separate output directory. The runner never silently reduces the requested batch size on OOM. Small final batches are retained; preflight rejects a training split whose final batch would contain only one case when ASPP is used.

**Augmentation:** by default, the runner disables geometric transforms for every row, preserving correspondence with directional text; photometric augmentation remains. `--legacy-geometric-augmentation` restores the existing flips/rotations/grid distortion consistently across rows. This is recorded in every run and the result CSV. New batch/initialization/augmentation settings mean older seed-42 results should not be inserted into this study.

**Testing:** E0/native protocol, FP32, one 224 × 224 forward, threshold 0.5, no TTA. Saved calibration files are ignored. Test data is never used for checkpoint or threshold selection. This runner does not execute the plan's optional calibrated/TTA, boundary-metric, or router-removal studies.

## Results and resume

```text
<output-root>/
  results.csv                   # one consolidated table, 72 rows by default
  study_manifest.json           # configurations, source hash, input hashes/counts
  initializations/seed_42.pt    # shared untrained tensors; also 1337 and 2026
  qata/I0/seed_42/
    spec.json
    run_config.json
    state.json
    train.log
    test.log
    train_result.json
    test_result.json
    test_cases.csv
    training/version_*/metrics.csv
    models/best*.ckpt
    models/last.ckpt
  ...remaining dataset/variant/seed directories...
```

`results.csv` is rewritten atomically after each phase. It contains the configuration, best validation Dice/IoU, best epoch, epochs/optimizer steps, test macro/global Dice and IoU, case count, phase durations/attempts, checkpoint location, and errors. Best validation IoU is derived from the corresponding pooled Dice using `Dice / (2 - Dice)`. Test metrics are computed from predictions.

It also contains paired macro-Dice differences in percentage points versus the preceding variant, I0, and I6. These cells fill when both matching seed results exist. Mean and sample standard deviation across completed seeds are repeated on each completed row of the same dataset/variant; `completed_seeds` makes partial summaries explicit. Individual scores are on [0,1]. No failed/missing score is replaced by zero.

**Resume with the identical launch command and output directory.** Completed train/test pairs are skipped. If training completed but testing failed, only testing repeats. Interrupted training resumes from `last.ckpt` when available, otherwise starts again. Resume is at a saved epoch boundary; worker augmentation/data-order RNG state is not fully restored, so interrupted training is not promised to be bitwise identical to uninterrupted training. Resume attempts are recorded.

By default, a failed run is recorded and the runner continues with the other runs, then returns a nonzero exit code. Use `--fail-fast` to stop at the first failure. Ctrl-C/SIGTERM stops the active child and records interruption. Avoid launching multiple runners against the same output directory; an advisory lock enforces one writer. Changing source, data, or settings requires a new output directory rather than mixing incompatible results.

Monitor progress with:

```bash
tail -f /workspace/results/cumulative72/qata/I0/seed_42/train.log
```

Add `--save-predictions` at study creation to save prediction masks per run. The default saves per-case metric CSVs without thousands of mask images.

For a limited pilot, use a different output directory and selectors, for example `--datasets qata --seeds 42 --experiments I0 I7 I8 I11`. Selectors change the study manifest; a pilot directory cannot later be reused as the full-study directory without moving to a new study. The default command always schedules all 72 runs.

## Visualize the best I11 seed

`visualize_i11.py` compares **validation global Dice** in the three I11
`train_result.json` files (seeds 42, 1337, and 2026) for one dataset, then loads
the winning run's best checkpoint. It requires all three training results;
test scores do not participate in selection. Exact ties choose the lowest seed.
The script restores architecture settings from the checkpoint and needs no retraining.

```bash
# Check the selected seed and paths without loading PyTorch or model weights.
python visualize_i11.py --study-root /workspace/results/cumulative72 --dataset qata --dry-run

# Export eight test cases from the winning seed.
python visualize_i11.py \
  --study-root /workspace/results/cumulative72 \
  --dataset qata \
  --data-root /workspace/data \
  --num-cases 8

# Choose exact cases, or use --num-cases 0 to export the entire split.
python visualize_i11.py \
  --study-root /workspace/results/cumulative72 \
  --dataset MosMed \
  --data-root /workspace/data \
  --cases example1.png example2.png
```

Replace example filenames with names from the dataset. `--data-root` is the
parent of `qata/` and `MosMed/`; omit it to use the path recorded in `spec.json`.
If copying results from another machine, retain each seed's `spec.json` and
`train_result.json`, plus the winning `models/best*.ckpt`. The script first looks
for that exact checkpoint basename inside the copied run, so old absolute paths
do not prevent loading. A missing winning checkpoint raises an error rather than
silently selecting a different seed.

Matplotlib is included in `requirements.txt`; install it with
`python -m pip install matplotlib` in an existing training environment if needed.
Existing model/tokenizer dependencies and cached BERT assets are also required.
The default device is CUDA when available; `--device cpu` is supported.
`--split val` selects validation images instead of test images.

Outputs go to a new timestamped directory under
`<study-root>/xai/<dataset>/seed_<winner>_<timestamp>/`. Use `--output-dir` to
choose an empty destination. The script does not overwrite previous exports.

Each case contains:

- `summary.png` and `summary.pdf`: preprocessed image, ground truth and prediction
  overlays, foreground probability, Seg-Grad-CAM, and false-positive/false-negative maps.
- `routing_stage1.png` through `routing_stage4.png`: all six source weights and
  contribution L2 norms, separately for every transformer block at native feature resolution.
- `prediction.png`: the binary mask, stored as 0/255.
- `maps.npz`: probabilities, raw and normalized CAM, the CAM target mask, ground truth,
  prediction, token IDs/mask, and unaveraged routing arrays. Routing arrays have shape
  `[blocks, height, width, 6]`; source names are included. Load with `allow_pickle=False`.
- `case.json`: the prompt, CAM target/fallback, and per-case Dice/IoU.

The export root contains `selection.json` with all three validation scores, the
winning checkpoint and hash, protocol, versions, selected filenames, and export
status, plus `cases.csv` with per-case metrics. Cases default to the first sorted
filenames, not those with the highest test scores. Subset metrics are not full-test results.

The prediction protocol remains one native-size FP32 forward (224 square for the
cumulative study), threshold **strictly greater than 0.5**, no TTA, and no calibration
file. XAI processes one case at a time to limit memory. No optimizer is used.

Seg-Grad-CAM uses `decoder1` (56 square for a 224-square input), which combines
the cascade features before the high-resolution tail. Its scalar target is the
mean foreground logit within the detached predicted mask. An empty prediction
instead targets the maximum-logit pixel, explicitly labelled in the figure and
metadata. Ground truth is used for overlays and metrics, not to choose the CAM
target. A zero positive CAM response is exported as zeros and explicitly flagged.
The method follows the regional target principle in
[SEG-GRAD-CAM](https://arxiv.org/abs/2002.11434).

CAM is normalized independently for each case, so color intensity is not directly
comparable across cases. Router weights use a fixed 0–1 scale; contribution norms
share a scale across all blocks/sources within each stage figure. Raw values are
preserved in `maps.npz`. Text groups represent BERT layer depths, not labelled
anatomical concepts. No-update has zero contribution even when its weight is large;
previous-stage memory is unavailable at stage 1. These maps describe internal
computation and sensitivity, not causal proof or calibrated uncertainty.

## Validation

XAI selection and model/figure export checks:

```bash
python -m pytest tests/test_visualize_i11.py tests/test_visualize_i11_model.py -q
```

Local validation on 2026-11-20: these 17 checks and the cumulative model,
training/evaluation, and joint-router suites completed with **94 passed, 1 GPU-only
test skipped**. The export check uses a saved tiny GRACE I11 checkpoint, the real
dataset preprocessing, and local substitute encoders; it verifies prediction
parity and writes all figures/arrays without pretrained downloads. The cumulative
worker test fixture now restores global configuration between tests. Linux-only
orchestration tests could not be rerun on this Windows host because WSL failed to
start. Real pretrained checkpoints, datasets, and GPU execution remain unverified.

Dependency-free batch tests:

```bash
python -m pytest tests/test_cumulative_runner.py -q
```

In the ML environment, run the model and integration checks before the long study. These substitute tiny local encoders and require no model downloads:

```bash
python -m pytest tests/test_cumulative_models.py tests/test_training_evaluation.py tests/test_joint_router.py -q
```

Local validation on 2026-11-12: the eight orchestration tests passed, including all 72 simulated train/test pairs, resume, failure handling, summary arithmetic, and data preflight. Shell syntax and Python compilation passed. The model tests were added but could not run locally because PyTorch is absent and dependency installation was declined. Real-data cloud GPU training and batch-32 GPU memory validation have not been performed here.
