# GRACE Joint Router: Prototype and Evaluation Plan

Date: 2026-10-26

## Implementation Status

The joint router is implemented as an optional GRACE guidance path. It has passed synthetic CPU regression tests; no real-data training, accuracy comparison, or GPU benchmark has been run.

The original corrected GRACE path remains `router_config.mode="off"`. Launch the proposed architecture explicitly with the `router_region` experiment. Frequency enhancement, frozen BERT, hierarchical within-group text mixing, PEAL, ASPP, decoder, and deep supervision retain their existing behavior unless a named ablation changes a setting.

Key files:

- [nets/routing.py](nets/routing.py): source scoring, availability masking, routing controls, and interventions.
- [nets/ViT2.py](nets/ViT2.py): candidate attention and routing inside each transformer block.
- [nets/LViT_new.py](nets/LViT_new.py): all four language groups at every routed stage and optional diagnostic maps.
- [experiments.py](experiments.py): named training comparisons and reproducible run snapshots.
- [model_settings.py](model_settings.py): validated router configuration saved in architecture snapshots.

## Prototype Definition

Each block computes four text-attention responses using **one shared text-attention module**. Each group retains its own token softmax and padding mask. A fifth candidate comes from previous-stage attention, and the sixth is a zero response.

Text and memory candidates use the incoming visual tokens as their queries, so they are parallel. Responses pass through the existing post-attention normalization and LayerScale. For dynamic routing, a shared small scorer combines normalized visual tokens and normalized candidate responses:

```text
logit_g(p) = score(GELU(W_visual LN(V(p)) + W_candidate LN(C_g(p)))) + bias_g
weights(p) = masked_softmax(logits(p) / temperature)
update(p) = residual_scale * sum_g weights_g(p) * candidate_g(p)
V_guided(p) = V(p) + DropPath(update(p))
```

The candidate responses already contain image/text or previous-stage context, so the scorer is conditioned on those inputs without another text encoder. Self-attention and the MLP follow the guidance update.

- Stage 1 masks memory out of the softmax exactly.
- Every stage uses the same six-source order: `text_1`, `text_2`, `text_3`, `text_4`, `memory`, `none`.
- The default hidden dimension is 64, temperature is 1, and residual scale is **fixed at 1**. Existing per-channel LayerScale parameters remain learnable.
- One DropPath decision is applied to the combined guidance residual. All routed controls share this construction.
- Two blocks per stage produce eight routing distributions. Grid sizes at 224 input are 56, 28, 14, and 7 square.
- Three-dimensional legacy embeddings are rejected when routing is enabled: four copies of the same embedding would not support a meaningful depth-selection experiment.
- Routing and the original adaptive gate are alternative guidance paths; enabling both raises an error.

With the current channel widths and two blocks per stage, the dynamic scorer adds approximately **369,712 parameters**. It does not create four independent text-attention parameter sets. Dense text-attention evaluations increase from 8 to 32 per model forward. Measure actual latency and memory; these counts do not imply a fourfold whole-model slowdown.

## Running the Prototype

Configure the dataset and prompt CSV paths in `Config.py`. Run commands from the repository root using the project's Python environment.

Preview the resolved settings without loading datasets or downloading weights:

```text
python train_model.py --experiment router_region --seed 42 --dry-run
```

Start the proposed model and a baseline comparison:

```text
python train_model.py --experiment router_region --seed 42
python train_model.py --experiment baseline --seed 42
```

Memory-conscious pilot configuration:

```text
python train_model.py --experiment router_region --seed 42 --batch-size 8 --checkpointing
```

Apply the same batch size and checkpointing policy to comparison runs. Batch size 8 is an example, not a measured hardware recommendation.

Named presets reset router-specific options to their declared defaults, with common training/data settings read from `Config.py`. `--router-scale` and `--router-temperature` override those two defaults. Each training run writes `run_config.json` and receives a distinct directory containing its experiment name, seed, and timestamp. New checkpoints save router settings with the architecture. Pre-router snapshots missing `router_config` restore the original path with routing off.

## Experiment Matrix

Run the following on **both target datasets** using the same predetermined splits. The currently configured dataset is `qata`; configure the second dataset's paths before its runs.

### Priority 0: Feasibility and Correctness

| Check | Required outcome |
|---|---|
| Small real-data forward/backward pass | Finite loss and gradients with the actual pretrained encoders |
| Small-subset overfit | Both baseline and router can learn a small training subset |
| GPU mixed-precision smoke run | Stable gradients, no NaNs, and sufficient memory at the selected batch size |
| Routing masks | Stage-1 memory and disabled sources have exactly zero weight |
| Checkpoint round trip | Restored predictions match using the same precision and inference settings |
| Last training batch | Avoid a singleton per-device batch: the existing pooled ASPP BatchNorm limitation remains |

Keep pilot runs separate from reported full-training results. Do not use test performance to choose hyperparameters.

### Priority 1: Main Mechanism Comparisons

| ID | Training preset | What it tests |
|---|---|---|
| B0 | `baseline` | Corrected GRACE with sequential text and memory attention |
| B1 | `baseline_no_memory` | Whether the original unconditional memory branch helps at all |
| B2 | `adaptive_gate` | Whether the existing conventional text gate explains the benefit |
| C0 | `parallel_fixed` | Parallel candidates with fixed assigned-text/memory allocation |
| C1 | `parallel_uniform` | Flexible candidate availability with equal weights |
| C2 | `router_static` | Learned input-independent weights per block |
| C3 | `router_image` | One input-dependent distribution per image and block |
| C4 | `router_region` | Input-dependent selection at every visual token |

Important control definitions:

- `parallel_fixed` assigns all weight to the stage's original text group at stage 1; later stages split weight equally between that group and memory. None receives zero weight. All four text responses are still computed.
- `parallel_uniform` distributes weight equally over available sources, including none. Stage 1 has five available sources and later stages have six.
- `router_static` learns six logits per block, masked for unavailable sources.
- `router_image` averages normalized visual/candidate features over spatial tokens before scoring. It has the same scorer parameter count as `router_region`.
- All routed controls share the same parallel candidate graph and combined DropPath policy. They therefore isolate routing more directly than comparison to B0 alone.

### Priority 2: Component Ablations

| ID | Training preset | Question |
|---|---|---|
| A0 | `router_no_memory` | Does language-depth selection help without inherited context? |
| A1 | `router_always_memory` | Does selective memory outperform a full parallel memory residual? |
| A2 | `router_assigned_text` | Does selecting among linguistic depths help beyond selecting memory/none? |
| A3 | `router_no_none` | Is explicit guidance suppression useful? |

For A1, memory is excluded from the routing softmax and added at full residual scale outside it. Its reported routing weight is zero but its contribution norm can be nonzero. A none decision suppresses only the routed text part in this control.

For A2, the three unassigned text groups are masked out. Four candidate attentions are still evaluated, maintaining the prototype's dense construction.

### Priority 3: PEAL and Guidance Strength

| Comparison | Runs |
|---|---|
| PEAL interaction | B0 versus `baseline_no_peal`; C4 versus `router_no_peal` |
| Residual-scale sensitivity | C0, C1, and C4 with scales 0.5, 1, and 2 |
| Temperature sensitivity | C4 with temperatures 0.5, 1, and 2 |

Example:

```text
python train_model.py --experiment router_region --seed 42 --router-scale 2
python train_model.py --experiment router_region --seed 42 --router-temperature 0.5
```

The four PEAL runs separate routing usefulness from compatibility with the fixed stage/group alignment target. No PEAL target is automatically changed by the router.

Scale controls are essential: a convex mixture shares a unit guidance budget, whereas the original block adds two residual branches. Fixed routing at scale 2 restores full assigned-text plus memory amplitudes in stages 2–4, but doubles stage-1 text amplitude. It also retains parallel attention queries, so it is **not** an exact recovery of the sequential baseline.

### Additional Capacity Control Before a Strong Mechanistic Claim

Build and train a conventional fusion MLP over the same candidate responses, adjusting its hidden width to approximately match the router's additional parameters. Keep candidate generation, normalization, loss, and DropPath identical. Record parameter and compute differences explicitly.

This comparator is **not implemented by the current presets**. The existing adaptive-gate baseline is useful but does not substitute for a comparison that matches capacity and candidate access.

## Seed and Run Budget

Use shared seeds across conditions, for example **42, 1337, and 2026**, and report every completed seed rather than selecting the best one.

- The eight Priority-1 presets across two datasets and three seeds require **48 full runs**.
- The four component ablations add **24 runs**.
- The two additional PEAL-disabled presets add **12 runs**; reuse B0/C4 from Priority 1.
- All 14 implemented presets therefore require **84 full runs**, excluding sensitivity studies and the additional capacity control.

A staged approach can first screen all eight core presets with seed 42 on both datasets, then complete the remaining seeds for the planned confirmatory comparisons. Treat single-seed screening as exploratory evidence.

## Evaluation Protocol and Metrics

### Fixed protocol

- Use identical data splits, prompt CSVs, augmentation, image resolution, pretrained weights, optimizer settings, maximum epochs, and stopping criteria across conditions.
- Use patient-level separation where patient identifiers are available; avoid repeated slices or images from the same patient crossing splits.
- Train each ablation from the same initialization policy. Loading an already-trained baseline into only the router would confound the comparison.
- Keep frequency enhancement and the initial PEAL weight fixed except in their explicit ablations.
- Select checkpoints using validation only. Keep the existing global validation Dice criterion consistent across conditions.
- Calibrate thresholds on validation only, separately for each trained model and inference configuration. Never select a threshold using the test set.
- Report single-pass and TTA results separately. For native routing analysis, start with `Config.use_tta=False` before training/calibration. Changing that flag only at test time does not override saved calibration settings.

### Required measurements

| Category | Measurements |
|---|---|
| Segmentation | Per-image macro Dice/IoU plus global Dice/IoU |
| Boundaries | HD95 and/or ASSD, with declared spatial units and evaluation resolution |
| Stability | Mean and standard deviation across seeds; paired differences between conditions |
| Uncertainty | Paired confidence intervals, preferably resampling patients when identifiers permit |
| Efficiency | Trainable and total parameters, latency, peak GPU memory, and forward FLOPs with unsupported operations disclosed |
| Routing | Source utilization, entropy, spatial variation, and weighted contribution norms by stage/block |

Define empty-mask handling before evaluation. Correctly empty pairs currently receive Dice/IoU 1 in per-image evaluation. For boundary metrics, explicitly report empty-pair and one-empty-pair policies and case counts; silently discarding failed cases can bias results.

Benchmark every method using the same device, batch size, precision, warmup, and synchronization. Measure single-pass inference separately from the nine-forward TTA configuration. Record whether frozen BERT computation is included.

The current test script reports Dice/IoU and writes masks. **Boundary metrics, per-case metric exports, confidence intervals, and a hardware benchmark harness still need evaluation tooling.** They are requirements for the experimental study, not results produced by this prototype.

## Routing Interventions and Diagnostics

The evaluation CLI supports interventions on a trained router checkpoint:

```text
python test_model.py --checkpoint <path> --routing-override uniform
python test_model.py --checkpoint <path> --routing-override fixed
python test_model.py --checkpoint <path> --routing-override image_mean
python test_model.py --checkpoint <path> --routing-override shuffle --routing-seed 42
python test_model.py --checkpoint <path> --routing-override none
```

These interventions retain the original calibrated threshold and TTA settings, isolating a change at inference. They complement separately trained ablations; performance drops under an unfamiliar routing distribution alone do not establish the benefit of learning that mechanism. Repeat shuffled interventions with several seeds. Output masks use intervention-specific subdirectories.

`image_mean` broadcasts the learned spatial mean distribution; `shuffle` permutes distributions across tokens within each image. Both preserve source availability. `none` requires that the checkpoint allowed none. Interventions are rejected during training.

For diagnostic maps, use a native-resolution, single forward in evaluation mode:

```python
import torch
from test_model import load_model
from nets.routing import ROUTING_SOURCES

model, _ = load_model(device, checkpoint_path)
with torch.no_grad():
    out = model(images, text, return_routing=True)

# Each tensor has shape [batch, blocks, height, width, 6].
maps = {
    stage: {name: tensor.cpu() for name, tensor in values.items()}
    for stage, values in out["routing"].items()
}
torch.save({"sources": ROUTING_SOURCES, "maps": maps}, "routing_maps.pt")
```

Here `images` and `text` are a prepared batch on the model's device; `text` contains token IDs and padding masks. Diagnostic tensors are detached and are returned only when requested. Contribution norms include candidate LayerScale and residual scale, measured before DropPath; use evaluation mode for interpretation.

Examine maps and contribution magnitudes for foreground, boundary, and background regions using ground truth only for retrospective analysis. Measure candidate-response similarity to determine whether the four depths offer distinguishable information. Do not infer fixed anatomical/disease meanings for particular BERT groups from attractive visualizations alone.

Evaluate prompt sensitivity with predefined empty, shuffled, incomplete, and paraphrased descriptions. Preserve tokenizer special tokens for empty prompts; all-masked descriptions are invalid. Laterality tests require consistent handling of current flips/rotations and the prompt's anatomical frame.

## Evidence Needed to Support the Contribution

- Regional routing improves over learned-static and image-level routing across seeds and both datasets.
- Joint language/memory selection improves over its component controls.
- Gains survive guidance-strength and approximate capacity controls.
- The no-update option has measurable value beyond merely reducing average guidance strength.
- Routing responds to image/text changes, and diagnostic interventions support the interpretation without replacing trained comparisons.
- Efficiency costs and dataset-specific failures are reported alongside accuracy improvements.

If static routing matches regional routing, the supported contribution is flexible depth mixing. If memory-only selection explains the gain, narrow the claim to selective cascade guidance. A null result remains useful evidence for choosing the simpler architecture.

## Validation Completed

The CPU regression suite passed **46 tests in 14.88 seconds**, including source masking, no-update behavior, routing scope, shared attention, padded-token gradients, checkpointed/plain gradient agreement, CPU bfloat16 autocast, multi-scale inference, synthetic Lightning training/checkpoint round trips, evaluation interventions, and pre-router checkpoint compatibility. Six non-failing warnings concern synthetic loader worker counts and the existing scheduler deprecation. The `router_region` CLI dry run also passed. Full pretrained-model training, GPU mixed precision, and real-data performance remain to be evaluated.
