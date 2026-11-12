# GRACE: from image-only segmentation to the full architecture

Implementation update — **2026-11-12:** the I0–I11 cumulative study now has an executable entry point, [run_cumulative.sh](run_cumulative.sh), and instructions in [CUMULATIVE_RUNNER.md](CUMULATIVE_RUNNER.md). It schedules all 72 train/test pairs at the user's requested **batch size 32**, with checkpointing enabled, native FP32 testing at threshold 0.5, and one consolidated `results.csv`. The runner defaults to photometric-only augmentation for every row, records common initialization and input/source hashes, and resumes completed phases. These explicit settings supersede the batch-16 execution recipe below for this new study; historical results are not interchangeable. The I/W implementation-status descriptions below record the original plan; cumulative I0–I11 switches and E0 exports are now implemented, while W12 and additional router-removal studies remain outside this runner. The eight local orchestration tests passed; ML tests could not run on this machine because PyTorch is absent and installation was declined. Cloud preflight runs the added model tests before full training.

Updated: **2026-11-03 (Asia/Dhaka)**. Source review: working tree at commit `e33f89b`.

**Study focus:** measure the improvement from an image-only model to the full GRACE architecture, and quantify what each added component contributes. The main paper table is a **12-row cumulative study, I0–I11**. A second table removes one component at a time from the full model. Keep both a simple image-only reference and a strong image-only model with all visual enhancements, so the language comparison is credible.

**Run budget:** 72 full runs for the cumulative table across two datasets and three seeds; 84 with two additional matched language controls; 138 including all planned component removals. Detailed router comparisons are a supplementary extension, bringing the complete optional package to 198 runs. Begin with the five anchor conditions in section 11. No positive gain is assumed: fill tables with measured results, including ties and regressions.

This is a prospective research plan, not a record of newly completed experiments or a guarantee of publication. Historical numbers below come from [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); their checkpoints and raw datasets were not independently evaluated during this review. This document updates the experimental priorities in [GRACE_ROUTER_EXPERIMENTS.md](GRACE_ROUTER_EXPERIMENTS.md), whose initial “no real-data training” status predates that log. Keep the log as the record of actual runs.

## 1. Research question and claim boundaries

Working question: **How much does the complete architecture improve over image-only segmentation, how much comes from better visual processing versus language, and which components account for that improvement?**

| Proposed claim | Evidence needed | Interpretation if unsupported |
|---|---|---|
| The architecture improves the starting image-only system | I11 versus I0 | This is the total system effect, not evidence for any single module |
| The text-guided system improves a strong image-only model | I11 versus I6, plus routed image-only W12 | Do not attribute all gains over the simpler I0 to language |
| Visual components add value | I1−I0 through I6−I5, supported by W1–W6 | Keep negative or uncertain increments in the table |
| Text guidance contributes information | I7−I6; I9−I6; W9 versus W12 under matched routing and no PEAL | Robustness to wrong text alone does not establish useful text conditioning |
| Text hierarchy and learned layer mixing help | I8−I7, I9−I8; W7/W8 versus I11 | Separate depth grouping from learnable within-group weights |
| PEAL helps | I10−I9; I11−W9 | Alignment may interact with routing; show both settings |
| Joint routing adds value beyond corrected GRACE | I11−I10, supported by section 8 controls | This step changes a guidance package; do not label it a pure spatial-routing effect |
| Regional selection and the zero option are useful | C4 versus C3/C2, and I11 versus W11, with scale controls | Narrow the mechanism claim if simpler fusion explains the gain |

The architecture is implemented under `LViT`/`LViTN` names, but this repository's `baseline` is **text-guided corrected GRACE**, not an image-only model and not a faithful reproduction of the published LViT. In this plan, `baseline` = I10/B0 and `router_region` = I11/C4. The new I/W IDs specify experiments; they are not automatically implemented CLI presets.

## 2. Architecture to freeze

Read together: [Config.py](Config.py), [model_settings.py](model_settings.py), [nets/LViT_new.py](nets/LViT_new.py), [nets/ViT2.py](nets/ViT2.py), [nets/routing.py](nets/routing.py), and [lightning_module.py](lightning_module.py).

| Part | Reference configuration / actual behavior |
|---|---|
| Input/output | RGB image at 224 × 224; one binary foreground mask |
| Visual backbone | `convnext_tiny.in12k_ft_in1k`; channels 96, 192, 384, 768; pretrained weights, followed by fine-tuning |
| Text encoder | Frozen `microsoft/BiomedVLP-CXR-BERT-specialized`; 12 transformer hidden layers, width 768; maximum 80 tokens including special tokens |
| Text hierarchy | Four groups: layers 1–3, 4–6, 7–9, 10–12; learnable softmax mixing inside each group |
| Visual cascade | Four stages, two transformer blocks each, eight heads; grids 56², 28², 14², 7² |
| Frequency module | Enabled at each stage before tokenization; positive learned Gaussian width initialized to 5; per-channel residual coefficient initialized to zero |
| Legacy guidance, B0 | Assigned text group, then previous-stage cross-attention; memory query sees text-updated tokens; adaptive gate disabled |
| Proposed guidance, C4 | Four text responses, memory response when available, and zero response; text attention weights shared across groups within each block |
| Router | Dense token-level mixture; hidden width 64, temperature 1, residual scale 1; LayerScale initialized to 0.1 remains learnable |
| Source availability | No stage-1 memory; six source slots retain the same order at every stage |
| Transformer regularization | Dropout 0; maximum stochastic depth 0.2, linearly spread over eight blocks |
| Decoder | Stage-3 ASPP with dilations (1, 3, 6, 9), dropout 0.1; UNETR upsampling; raw-image detail skips at full and half resolution |
| Main segmentation loss | MONAI DiceFocalLoss: Dice + focal, both coefficients 1; focal gamma 2 |
| PEAL | Mean of four stage-wise cosine distances between projected pooled visual features and assigned pooled text groups; coefficient 0.2 |
| Deep supervision | Two auxiliary heads; coefficient 0.3; normalized internal weights 0.3 at H/8 and 0.5 at H/4 |

The training objective is:

```text
L = L_main + 0.2 L_PEAL + 0.3 [(0.3 L_aux2 + 0.5 L_aux1) / 0.8].
```

PEAL is a cosine alignment objective, not an InfoNCE loss. Its targets remain tied to the original stage/group assignment even when the router can select other groups. The zero candidate disables the guidance residual only; self-attention, the MLP, and information inherited from earlier stages remain active.

The default command without `--experiment` currently leaves routing **off**. Use `--experiment router_region` for C4. `Config.pretrain=False` does not turn off backbone pretraining in the current training path: `LViTLightningModule` defaults `backbone_pretrained=True`. Report resolved behavior, not unused configuration labels.

## 3. Existing evidence and what can be reused

Matched stopping-policy results reported in E09–E14, **QaTa, seed 42, calibrated nine-forward TTA**:

| Condition | Best global validation Dice | Test global Dice | Test macro Dice | Epochs run |
|---|---:|---:|---:|---:|
| B0 `baseline` | 0.8932 | 0.9166 | 0.8501 | 108 |
| C1 `parallel_uniform` | 0.8932 | 0.9172 | 0.8503 | 106 |
| C4 `router_region` | 0.8928 | 0.9179 | 0.8518 | 78 |

C4's reported macro Dice advantage over B0 is **0.0017 = 0.17 percentage points**, not 1.7%. Validation ranks C4 slightly below B0. These observations motivate replication; they do not establish statistical significance or single-pass improvement.

- E06/E08 used `early_stopping_min_delta=0`, unlike the later `5e-4` runs. Keep them as historical experiments, outside the matched primary table.
- E07 was incomplete. E15 failed on the paired-text MixUp/CutMix guard. E16 failed with depth 4, batch 16, checkpointing off. None is a completed accuracy result.
- E13 documents same-seed variation with nondeterministic attention. A replay of another run's learning curve is not an independent matched replicate; do not use that replay as a formal noise estimate.
- E09, E11, and E13 may fill three existing seed-42 cells only if checkpoints, configurations, data, and the final training protocol match. Re-evaluate them under the newly locked evaluation protocol. Changes to augmentation, mask targets, splits, or initialization policy require new training cells.
- The QaTa test set has already been inspected. This plan is a prospective protocol lock, not retrospective preregistration. Stop selecting architectures from that test set; seek a genuinely untouched cohort for stronger confirmation.

**No image-only result has been established by these three logged comparisons.** Their scores cannot fill I0, I6, or W12, and do not yet demonstrate the requested improvement over an image-only model.

## 4. Resolve data and reproducibility issues before the run matrix

### 4.1 Dataset identity, leakage, and text provenance

The local experiment log reports these image counts; verify them against a frozen manifest:

| Local dataset | Train | Validation | Test | Empty GT masks, train/val/test |
|---|---:|---:|---:|---|
| `qata` | 6074 | 1071 | 2113 | 0 / 0 / 0 |
| `MosMed` | 1910 | 409 | 410 | 2 / 1 / 1 |

1. Identify exact dataset releases and preprocessing. The folder name `MosMed` alone does not establish whether it is MosMedData+, another derivative, or the original collection. The [official LViT repository](https://github.com/HUANGLIZI/LViT) provides dataset and text-annotation provenance to check against local files.
2. Export patient/study IDs, source IDs, image/mask/prompt hashes, split membership, dimensions, and label statistics. Filename uniqueness does not establish patient independence; check duplicate and near-duplicate images and related CT slices across splits.
3. Split by patient/study when possible. If IDs cannot be recovered, disclose that limitation and avoid patient-generalization claims. Do not invent IDs from unverified filename conventions.
4. Record how descriptions were created and what is available at inference. If test descriptions contain information derived from the target mask, identify the task as segmentation with that additional annotation; do not present it as ordinary clinical-report inference. Use the same prompt information for every text-based comparator.
5. Audit missing/duplicate CSV entries, literal `nan` descriptions, and truncation at 80 tokens. Publish prompt templates and a manifest without exposing restricted raw records.

Train separately on the two datasets. This establishes performance in two settings; it is not external validation of one frozen model. An additional external test should use an appropriate same-modality/task cohort, a compatible prompt protocol, and no target-test tuning. Do not treat an X-ray-to-CT transfer as an ordinary external test.

### 4.2 Mask and augmentation policy

- The loader currently binarizes original masks with `mask > 0`. E01 reports 99 MosMed masks with gray edge values, including 12, 17, and 21. Their origin was interpreted as antialiasing in the log, but original annotation provenance still needs checking.
- Preserve `>0` for historical comparability unless source documentation supports a correction. If corrected, freeze the conversion and retrain every affected comparator. A `>127` sensitivity report must be labeled as evaluation against a different target definition; never select a label threshold by model test score.
- Current transforms rotate by 90°, flip horizontally and vertically, and sometimes rotate ±30°, while leaving descriptions unchanged. Audit anatomical laterality and positional language against these transforms. Recommended publication policy for directional descriptions: disable transforms that invalidate the description unless a validated image/mask/text transformation is implemented. Apply the resulting policy to every condition and rerun affected historical cells.
- Keep `use_mixup_cutmix=False`. The current implementation deliberately rejects image-only mixing with paired text.
- Check the last per-device training batch: pooled ASPP BatchNorm cannot train on a singleton. Preserve the existing loader policy when safe; if a sampling/drop policy must change, lock it for all conditions and report discarded samples.

### 4.3 Initialization and randomness

Use **42, 1337, 2026** for the first complete matrix. Same seed does not imply identical common weights: extra router modules consume random draws during model construction. For stronger pairing, export a per-seed initialization of common trainable tensors, copy matching tensors into each condition before optimization, and initialize condition-specific tensors from a separate RNG stream. This is an initialization snapshot, not a trained checkpoint. Verify common tensor hashes.

Freeze dataloader ordering, worker settings, and augmentation RNG policy. Audit MONAI transform RNG initialization explicitly; Python/NumPy seeding alone does not demonstrate identical transform draws. Record CUDA/attention backend flags. Either use a deterministic backend for the full study after a feasibility pilot, or retain `deterministic="warn"` and disclose/measure residual nondeterminism. Do not change backends between conditions.

Use independent processes per training run. `configure_run()` mutates global configuration and only resets PEAL weight for presets that explicitly supply it; an in-process loop can accidentally carry `peal_loss_weight=0` into a later run. Every proposed new flag must be saved in `model_kwargs` or the relevant run metadata and restored by evaluation.

## 5. Fixed training and evaluation protocol

### 5.1 Training recipe

| Setting | Locked starting value |
|---|---|
| Maximum training | 201 epochs |
| Optimizer | AdamW; head LR 3e-4, backbone LR 3e-5; weight decay 1e-4 |
| Schedule | Five-epoch linear warmup from factor 0.01, then cosine decay to 1e-6 |
| Backbone freeze | First five epochs; frozen BERT throughout |
| Batch/precision | Batch 16, `16-mixed`, checkpointing off for the depth-2 core; two workers |
| Checkpoint selection | Highest global validation Dice at threshold 0.5, native 224 input, no TTA |
| Early stopping | Same global validation Dice; patience 20; `min_delta=5e-4` |
| Components/loss terms | Section 2 is the full model; enable only the components listed for each I/W row. All other settings follow section 4 |

Early stopping's `min_delta` controls patience resets; it does not prevent `ModelCheckpoint` from saving a smaller improvement. Equal stopping rules do not imply equal epoch counts or GPU-hours. Record both. If fixed-budget training is investigated, run it as a separate matched study with the scheduler duration declared in advance.

The depth-2 router previously reached about 8.8 GiB peak reserved memory on the logged RTX 4070. Recheck after changes. Depth 4 at batch 16 without checkpointing already failed; do not repeat that setup. If a later depth study requires checkpointing or a smaller batch, rerun its depth-2 reference with the same policy. Gradient accumulation does not reproduce the same BatchNorm behavior as a larger physical batch.

### 5.2 Three inference protocols, using the same selected checkpoint

| ID | Protocol | Purpose |
|---|---|---|
| E0 — primary | One 224 × 224 forward, threshold 0.5 | Test architectural benefit without threshold search or TTA |
| E1 — secondary | One 224 × 224 forward; threshold selected on validation macro Dice | Measure the calibrated deployment pipeline |
| E2 — secondary | Sizes 192, 224, 256; original, horizontal flip, vertical flip; validation-calibrated threshold | Compare the existing nine-forward TTA pipeline and its cost |

Use the same threshold candidates for all models: `(0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)`. Preserve the current tie rule: the first candidate with the best score wins, so exact ties choose the lowest threshold. Calibrate E1 and E2 separately for each seed/checkpoint. TTA flips require the same prompt-consistency audit as training augmentation; if invalid, replace E2 with a declared scale-only protocol for every comparator and label historical nine-forward scores separately.

Report E0 even when E2 scores higher. No test-based threshold, checkpoint, TTA, or postprocessing selection. Keep checkpoint selection and final macro evaluation distinct: the current selection criterion is global Dice, not macro Dice.

**Tooling gap:** `test_model.py` has no CLI switch for E0, E1, inference precision, or a chosen calibration file. Saved `best_threshold.json` overrides current TTA settings. Merely setting `Config.use_tta=False` at test time does not reliably select E0. Add explicit evaluation/calibration selection and separate metadata files before running the study; do not overwrite historical calibration artifacts. The current standalone evaluator does not enter an AMP autocast context, so training `16-mixed` does not imply mixed-precision evaluation.

### 5.3 Metrics and statistical analysis

- **Primary endpoint:** mean per-image Dice under E0, then mean and sample standard deviation across training seeds. Export each seed and each paired seed difference. Report percentages and percentage-point differences consistently.
- **Secondary endpoints:** per-image IoU, global Dice/IoU, HD95 and ASSD, and foreground precision/recall. Global Dice and IoU are algebraically related for the same pooled binary confusion counts; they are not independent confirmations. Include patient-balanced Dice as a secondary summary when patients have multiple images.
- **Boundary units:** use the 224 × 224 evaluation grid and report pixels for the current pre-resized files. Do not label distances in millimetres without original spacing and a validated mapping. Use original-resolution evaluation only as a separately specified protocol available to all models.
- **Empty masks:** overlap scores are 1 when both masks are empty, 0 when exactly one is empty. For HD95/ASSD, assign 0 to both-empty cases; count one-empty cases explicitly and assign a declared finite penalty equal to the grid diagonal, `sqrt((H-1)^2 + (W-1)^2)` pixels, in an all-case summary. Also report conditional distances for pairs with both masks nonempty. The diagonal assignment is a reporting convention, not a measured surface distance. Never silently discard failures.
- **Case uncertainty:** use 10,000 paired bootstrap replicates, resampling patients/studies as clusters and retaining all their images and paired model predictions. Recompute the chosen image-macro statistic in each replicate. If verified independent images are the only units available, bootstrap images and disclose that unresolved patient clustering limits inference. Never bootstrap pixels as independent samples.
- **Training uncertainty:** bootstrap intervals over cases with trained models fixed do not include training variability. Report the paired seed deltas separately. A crossed bootstrap resampling both seed pairs and patient clusters can be a sensitivity analysis; with only three seeds, its training-variance estimate remains weak. Do not pool seed × image rows as independent observations.
- **Prespecified main contrasts:** I11−I0 (total system), I11−I6 (beyond strong visual processing), and W9−W12 (language guidance with matched routed topology and PEAL off), separately on each dataset. Report all six, including adverse outcomes. If formal p-values are required, use a justified paired patient-level procedure on seed-averaged outcomes and Holm correction across these six tests; label inference as conditional on the fitted seeds. Cumulative increments, removals, and router comparisons are secondary with their own declared multiplicity treatment if significance claims are made.
- **Practical relevance:** before new results, document the smallest useful accuracy gain and acceptable latency/memory cost for the intended use. No clinical relevance threshold is established here. Statistical significance alone does not establish usefulness, and a confidence interval crossing zero does not prove equivalence.

Complementing overlap with boundary measures and respecting nested data follows the validation principles in [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z). The concrete endpoints and analysis choices above are this project's proposed protocol.

## 6. Main experiment: add components to an image-only model

### 6.1 Define the baselines before measuring gains

**I0 — simple image-only scaffold.** Use the same pretrained ConvNeXt-Tiny and the three existing UNETR decoder blocks. Retain the per-stage token projection, positional embedding, and final normalization as a common feature interface. Bypass all cascade self-attention/MLP blocks, text attention, previous-stage attention, and frequency processing. Replace ASPP with identity. Predict at H/4 from d1 and bilinearly resize logits to 224 × 224. There are no raw-image detail stems, learned high-resolution upsampling, auxiliary losses, BERT, text fusion, routing, or PEAL. Train with the same main DiceFocal loss. This is a project-matched visual scaffold, not an official U-Net implementation.

**I6 — strong image-only reference.** Keep every visual enhancement used by the final model: visual transformer blocks, previous-stage memory, frequency enhancement, ASPP, the high-resolution tail, and deep supervision. Remove the language branch and PEAL entirely. This tests whether language improves an already capable image-only architecture. It corresponds to the earlier plan's proposed S0.

**W12 — routed image-only reference.** Keep the full model's routed guidance topology and scorer width, but make only memory and zero candidates available; disable every text candidate and PEAL. At stage 1 only zero is available. This requires a new language-off routing mode. Pair it with W9, the text-guided router without PEAL, to test language access with matched routing and training objectives. Report the changed active parameter count and candidate cost.

I0, I6, and W12 must train and evaluate without prompt CSVs, tokenization, a loaded text encoder, or text-derived supervision. Passing empty prompts to the current model is not image-only training. Setting PEAL to zero alone also leaves text attention active. If optional text arguments are supplied, verify that these image-only models' outputs do not depend on them.

### 6.2 Cumulative architecture table

Each row retains all earlier additions unless the row explicitly replaces a mechanism. **Train every row independently from initialization; do not fine-tune I1 from I0, I2 from I1, and so on.** Keep shared initial tensors, data, optimizer schedule, stopping rules, resolution, and evaluation identical. The order below is fixed before new results, rather than sorted afterward to make the curve increase.

| ID | Input | Add/change relative to the preceding row | Components introduced so far / interpretation |
|---|---|---|---|
| I0 | Image | Common visual scaffold | ConvNeXt + common feature interface + UNETR decoder + bilinear output; main loss only |
| I1 | Image | Add visual transformer refinement | Two self-attention/MLP blocks at each stage; text and memory still disabled |
| I2 | Image | Add previous-stage memory attention | Stages 2–4 attend to the previous stage; no text or router |
| I3 | Image | Add frequency enhancement | Learned frequency residual before tokenization at all four stages |
| I4 | Image | Add ASPP | Existing stage-3 skip ASPP replaces identity |
| I5 | Image | Add high-resolution detail reconstruction | Raw-image stems and two learned upsampling blocks replace the H/4 head plus bilinear output |
| I6 | Image | Add deep supervision | Existing auxiliary heads and weighted auxiliary loss; **strong image-only baseline** |
| I7 | Image + text | Add simple language guidance | Frozen CXR-BERT layer 12 supplies every stage's text attention; text residual then memory, no adaptive gate; PEAL off |
| I8 | Image + text | Replace single-layer guidance with fixed hierarchical groups | Layers 1–3, 4–6, 7–9, 10–12 averaged uniformly and assigned to stages 1–4; PEAL off |
| I9 | Image + text | Learn the within-group mixture | Trainable softmax layer weights replace fixed means; equals `baseline_no_peal` / L0 |
| I10 | Image + text | Add PEAL alignment | Existing stage/group cosine alignment with coefficient 0.2; equals `baseline` / B0 |
| I11 | Image + text | Add joint regional guidance routing | Replace sequential assigned-text/memory updates with parallel four-group/memory/zero routing; equals `router_region` / C4; **full model** |

Keep depth 2, eight heads, LayerScale initialization 0.1, and the same stochastic-depth schedule wherever the corresponding blocks are active. The common feature interface is trained, not frozen. Initialize both the H/4 and full-resolution segmentation heads with the same 0.01 foreground-probability bias prior. Do not add an adaptive gate as an extra step: it is an alternative comparator, not a component enabled in the final routed architecture. Deep supervision remains active from I6 onward; PEAL is off through I9.

I7 uses the same frozen encoder as all later rows, so I8 tests grouping rather than a different text pretraining source. I8 must freeze `alpha=0`; merely initializing learnable `alpha` to zero does not make a fixed-mean control. I9 unfreezes these values. In I7–I10, previous-stage attention queries the text-updated visual tokens, matching the existing legacy guidance path.

I11 changes candidate access, sequential versus parallel queries, residual normalization, and routing together. Label I11−I10 as the **joint-routing package effect**, not the isolated effect of spatial routing. Section 8 supplies the narrower comparisons.

### 6.3 What to report for every row

For model i, dataset d, and training seed s, let `D[i,d,s]` be test macro Dice under E0:

```text
Increment from the previous row = mean_s(D[i,d,s] - D[i-1,d,s])
Gain over simple image-only     = mean_s(D[i,d,s] - D[I0,d,s])
Gain over strong image-only     = mean_s(D[i,d,s] - D[I6,d,s])
```

Compute differences from paired seeds first. Use `100 × difference` for percentage points when Dice is stored on [0,1]. Report the per-seed deltas and paired case-based intervals alongside the seed mean ± sample SD. I0 has no preceding-row increment. Negative values remain negative.

The paper's main ablation table should have this layout, with a separate dataset panel or table:

| Model | Added component | Macro Dice, mean ± SD | Macro IoU | HD95, px | Delta versus previous, pp | Delta versus I0, pp | Delta versus I6, pp |
|---|---|---|---|---|---|---|---|
| I0 | Image-only scaffold | To measure | To measure | To measure | — | 0 | To measure |
| I1–I5 | One row per visual addition above | To measure | To measure | To measure | To measure | To measure | To measure |
| I6 | Deep supervision; strong image-only | To measure | To measure | To measure | To measure | To measure | 0 |
| I7 | Single-layer text guidance | To measure | To measure | To measure | To measure | To measure | To measure |
| I8 | Fixed hierarchical text groups | To measure | To measure | To measure | To measure | To measure | To measure |
| I9 | Learnable group mixing | To measure | To measure | To measure | To measure | To measure | To measure |
| I10 | PEAL | To measure | To measure | To measure | To measure | To measure | To measure |
| I11 | Joint regional routing | To measure | To measure | To measure | To measure | To measure | To measure |

Expand I1–I5 into five separate rows in the final manuscript. Add parameters and latency in an adjacent table if the main table becomes too wide. No expected or invented scores belong in these cells.

Report these distinct summaries:

- **Visual improvements:** I6−I0.
- **Simple language guidance:** I7−I6.
- **Hierarchical text guidance without PEAL:** I9−I6, with I8−I7 and I9−I8 separating grouping and learned mixing.
- **Complete multimodal extension beyond strong vision:** I11−I6. This includes language, PEAL, and joint routing; it is not language alone.
- **Language with matched routed topology, PEAL off:** W9−W12.
- **Full architecture over the starting system:** I11−I0.

A cumulative increment is conditional on the components already present. Its value can differ under another ordering. The increments sum to I11−I0 for this path, but they are not order-independent causal shares of the final score. Use the removal study below to check each component in the final context.

## 7. Confirmation: remove one component from the full model

Train each removal independently using the full model's settings except the named change. Compare **I11 minus the removal model**, with a positive Dice difference indicating that removing the component hurts. Also report the removal model's own score and uncertainty. For HD95, use the opposite sign or label the direction explicitly.

| ID | Model | Precise change from I11 | Existing equivalent / implementation |
|---|---|---|---|
| W0 | Full architecture | No change | Reuse I11 / C4; no extra run |
| W1 | Without visual transformer refinement | Bypass only the visual self-attention and MLP residuals in each cascade block; retain token interface, text/memory guidance, routers, frequency, and decoder | New independent block switches required |
| W2 | Without previous-stage memory | Remove memory candidates; preserve all text sources and zero | `router_no_memory` / A0 |
| W3 | Without frequency enhancement | Disable all four frequency modules | Existing `use_freq=False`; add a named preset |
| W4 | Without ASPP | Replace stage-3 ASPP with identity | New saved architecture option |
| W5 | Without high-resolution tail | Use a head on d1 plus bilinear output; remove raw-image stems and two learned upsampling blocks | New saved decoder option |
| W6 | Without deep supervision | Disable auxiliary-head execution and set auxiliary-loss coefficient to zero | Existing settings; add a named preset |
| W7 | Without learned within-group mixing | Freeze all group mixing weights to uniform means, in guidance and PEAL | New saved freeze policy; four groups remain |
| W8 | Without text-depth hierarchy | Use layer 12 in all four guidance slots and as every PEAL text target | New text-source option; candidate count and shared attention retained |
| W9 | Without PEAL | Set PEAL coefficient to zero; keep text guidance | `router_no_peal` / L1 |
| W10 | Without the joint-routing package | Restore sequential assigned-text/memory guidance | Reuse I10 / B0; a package removal, not an isolated learned-weight ablation |
| W11 | Without zero-update candidate | Remove zero from source availability | `router_no_none` / A3; also inspect residual magnitude |
| W12 | Without the language package | No BERT, text attention, hierarchy, or PEAL; route only memory/zero, keeping visual components and scorer width | New genuinely image-only routed path |

W1 must not bypass the entire cascade stage: that would also remove the language and memory guidance being held constant. W8 removes the whole hierarchical text package, including its alignment targets. A guidance-only final-layer control retaining hierarchical PEAL is a different optional experiment; do not silently substitute it for W8.

W4/W5 and other removals can change active parameters, compute, or regularization. Report those changes. They estimate the component's contribution as implemented; they are not automatically capacity-matched comparisons. W12 removes multiple language-dependent parts because PEAL cannot remain active in a genuinely image-only training run. Therefore compare W9 with W12 for language guidance with PEAL off, and I11 with W9 for alignment in the routed model.

**Paired reporting template:** `component | full Dice | removal Dice | full minus removal (pp) | paired interval | seed deltas | parameter/latency change`. Do not declare a component unnecessary solely because a small study fails to detect a difference, or necessary solely because one seed improves.

### Interactions that reuse completed cells

The PEAL × routing 2 × 2 design is already available:

| Guidance | PEAL off | PEAL on |
|---|---|---|
| Sequential hierarchical text | I9 / L0 | I10 / B0 |
| Joint regional router | W9 / L1 | I11 / C4 |

```text
PEAL × routing interaction = (Dice_I11 - Dice_W9) - (Dice_I10 - Dice_I9).
```

Compute this per dataset and paired seed. If an addition helps early in the chain but its removal barely affects the final model, investigate redundancy or interaction rather than choosing whichever comparison looks better. Targeted 2 × 2 follow-ups for frequency × routing or deep supervision × routing can be added if those interactions become part of the paper's claim; they are outside the core budget.

## 8. Supplementary study: explain the routing step

The main paper story remains I0 → I11. Use these controls to explain the final I11−I10 transition and avoid attributing a package gain entirely to regional decisions. B0/C4 and the other aliases below retain their meanings from the older project documents.

| Existing preset | Alias | Role / reuse |
|---|---|---|
| `baseline` | B0 | I10, text-guided corrected GRACE |
| `baseline_no_memory` | B1 | Extra control for memory in sequential guidance |
| `adaptive_gate` | B2 | Alternative conventional text gate; not part of the full model |
| `parallel_fixed` | C0 | Parallel assigned-text/memory control |
| `parallel_uniform` | C1 | All available candidates averaged without learned selection |
| `router_static` | C2 | Learned input-independent source weights |
| `router_image` | C3 | One input-dependent distribution per image/block |
| `router_region` | C4 | I11, one distribution per visual token/block |
| `router_no_memory` | A0 | W2 |
| `router_always_memory` | A1 | Extra control with memory added outside the routing simplex |
| `router_assigned_text` | A2 | Extra control exposing only the stage-assigned text group, memory, and zero |
| `router_no_none` | A3 | W11 |
| `baseline_no_peal` | L0 | I9 |
| `router_no_peal` | L1 | W9 |

**Eight extra presets remain after the cumulative/removal studies:** B1, B2, C0, C1, C2, C3, A1, A2. The other six conditions are reused. C4−C3 tests the benefit of spatial decisions with matched scorer capacity; C3/C4 versus C2 tests input conditioning; C4 versus C1 tests learning selection instead of uniform averaging.

Important qualifications:

1. C0 uses assigned text weight 1 at stage 1 and text/memory weights 0.5/0.5 later. B0 adds full sequential residuals with separate DropPath calls. Their difference includes query ordering, strength, and stochastic-depth grouping.
2. C1 gives weight 1/5 to each available slot at stage 1 and 1/6 later, including zero. Removing zero redistributes the guidance budget.
3. C3 pools normalized features before nonlinear scoring. Averaging C4's learned spatial weights at inference is a different operation.
4. C0/A2 still compute all four text attentions. Masked candidate access is not a computational saving in this implementation.
5. A1's memory routing weight is zero but its contribution can be nonzero. Zero selection suppresses only its routed text part.
6. B2 lacks C4's four-group access and does not match its additional capacity.

Two additional, currently unimplemented controls strengthen the mechanism claim:

- **K1 — conventional fusion with matched capacity:** retain C4's candidate construction and one combined DropPath, but use a bottleneck MLP on normalized visual/candidate features to predict the residual without a source softmax. Match added active parameters within a declared tolerance such as 5%, record compute, and match initial update scale using a fixed training-only calibration batch.
- **K2 — attention-order control:** relative to B0, form assigned-text and memory responses from the same incoming visual tokens and add both at full strength using separate DropPath calls. Keep everything else unchanged. This isolates query ordering more cleanly than C0 versus B0.

### Sensitivity, after the main table is complete

| Factor | Values | Models | Interpretation |
|---|---|---|---|
| Residual multiplier | 0.5, 1, 2 | C0, C1, C4 | Check whether weaker/stronger guidance explains gains |
| Router temperature | 0.5, 1, 2 | C4 | Check stability of selection sharpness |
| Depth | 1, 2, 4 | B0 and C4, optional | Change depth alone; match checkpointing/batch policy |
| Router width | 32, 64, 128 | C4, optional | Additional capacity sensitivity |

The first two rows add 16 training runs for a one-seed/two-dataset screen after reusing defaults, or 48 for three-seed coverage. These are outside section 11's totals. Scale 2 does not exactly reproduce B0: it doubles C0's stage-1 text while restoring full text/memory amplitudes later. If W11 changes update norms substantially, add a validation-defined scale-matched W11 comparison. Keep such tuned comparisons separate from the default table.

Do not tune components on test performance or change the cumulative order to hide regressions. Each row should answer a prespecified architectural question.

## 9. Diagnostics and robustness without retraining

Run these on frozen C4 checkpoints from every retained seed. Use E0, fixed thresholds, and fixed example selection. They complement trained controls; distribution shifts induced after training are not evidence that a separately trained simpler model is inferior.

### 9.1 Existing router interventions

| Existing override | What changes | Interpretation |
|---|---|---|
| `uniform` | Equal weights over available sources | Dependence on the learned distribution |
| `fixed` | Original assigned-text/memory allocation | Dependence on flexible candidate use |
| `image_mean` | Broadcast spatially averaged learned weights within each image | Sensitivity to removing local variation; differs from trained C3 |
| `shuffle` | Permute routing distributions over spatial positions within each image | Sensitivity to spatial placement |
| `none` | Select zero at every router | Dependence on routed guidance; valid only when zero is available |

Repeat `shuffle` with routing seeds **0, 1, 2, 3, 4**, distinct from training seeds. Report the average and spread for each trained model. The current CLI applies the override to all routed blocks; stage-specific interventions need new tooling. On C4, `none` suppresses text and selective memory guidance, so it does not isolate language alone.

### 9.2 Router measurements

Use `model(images, text, return_routing=True)` in evaluation mode. The existing API returns `weights` and `contribution_norms` per stage with shape `[B, blocks, H, W, 6]`; source order is `text_1, text_2, text_3, text_4, memory, none`.

Export source means, normalized entropy `-sum(w log w) / log(K_available)`, spatial variance, and contribution norms for each stage/block. Handle masked zero weights without `log(0)`. Summarize per image/patient before aggregation, so stage-1 token counts do not dominate stage-4 statistics. Also record learned text-group mixing weights, frequency sigma/gamma, and LayerScale magnitudes.

Report the norm of the combined update alongside individual weighted-candidate norms: candidates can cancel. A large routing weight is not necessarily a large contribution, and a large contribution is not a causal explanation. Inspect pairwise candidate-response cosine similarity and variation across images; near-identical language responses weaken a depth-specialization interpretation. Candidate responses and combined update norms need additional diagnostic exports.

Compare foreground, a fixed two-pixel GT boundary band on the evaluation grid, and remaining background. Declare the band construction and map downsampling rules before analysis. GT is used for retrospective grouping only, never as router input. Show preselected examples across lesion-size strata, including failures. Do not assign “anatomy” or “disease” meanings to language groups without evidence.

### 9.3 Prompt robustness

Use the same predefined perturbations and case mappings for B0, B2, C3, and C4:

| Prompt condition | Construction | Report |
|---|---|---|
| Correct | Original audited description | Reference quality |
| Empty | Tokenize an empty string while retaining valid special tokens | Sensitivity to missing description |
| Mismatched | Fixed derangement of prompts within the evaluation split, excluding identical descriptions where possible | Absolute quality and drop from correct prompts |
| Paraphrased | Verified meaning-preserving paraphrases, fixed before scoring | Stability to wording |
| Incomplete | Predefined removal of location or extent clauses | Dependence on those cues |
| Contradictory location | Controlled, clearly labeled stress test where original text has such information | Failure behavior; not ordinary deployment performance |

Never use all-masked token sequences. Keep model and calibration fixed for perturbation comparisons. Use no test-derived prompt rewriting or selection. Limited degradation is not automatically robustness: a model that ignores text can also be insensitive. Interpret together with I6/I9 and W12/W9, plus correct-prompt performance.

## 10. Efficiency measurements

Dense routing evaluates every candidate; it is not sparse expert execution. The reviewed depth-2 design uses 32 text-attention evaluations instead of eight per forward, with shared parameters across groups. The dynamic router adds approximately **369,712 parameters** under the current widths; verify counts from the instantiated models.

Measure every I0–I11 row and the W9/W12 controls under one hardware/software policy. Add supplementary router and K1 comparisons when run:

- Total stored parameters, nominal trainable parameters, and parameters active in the selected forward/loss. Frozen BERT is part of total model size. Unused gate, memory, auxiliary, or PEAL modules can remain instantiated; a raw `requires_grad` count alone may overstate active capacity.
- Native E0 latency at batch 1 and throughput at a common feasible batch such as 16; separate E2 latency. Use at least 50 warm-up iterations and 200 timed iterations, CUDA synchronization/events, and three measurement repeats. Report median and p95 latency, GPU model, precision, input size, and text length.
- Peak allocated and reserved GPU memory measured from clean runs, separately for inference and an unfrozen training step. Whole-card `nvidia-smi` usage is not the same measurement.
- Training time per epoch, number of optimization steps/epochs, total GPU-hours, and time to the selected checkpoint. Keep model-forward latency separate from data loading, tokenization, metric computation, and writing masks; report end-to-end latency separately if deployment is discussed.
- FLOPs/MACs with the profiler and convention identified. The current `flop_counter.py` can exclude BERT on fallback and may not account for every FFT/attention operation. Include unsupported-operator coverage and do not call an incomplete profile an exact total.

Include frozen BERT in the default end-to-end model measurement. The current TTA implementation reruns BERT for each forward. If caching frozen text representations is added, report cached and uncached modes separately and apply identical caching to all eligible methods. Compare accuracy and latency at the same declared inference precision.

## 11. Execution order and compute budget

Counts use **two datasets × three seeds = six runs per unique condition**. Reuse a condition whenever it appears in multiple tables; never retrain it merely to give it another ID. Counts exclude evaluation passes and assume no historical reuse.

| Phase | Conditions added | New full runs | Cumulative runs | Purpose |
|---|---|---:|---:|---|
| P0 | Data/protocol audit; implement image-only and evaluation paths; smoke checks | No full runs | 0 | Make the intended comparisons executable and valid |
| P1 | I0, I6, I11, W9, W12 | 30 | 30 | Establish simple/strong image-only anchors, full-model gain, and matched language effect |
| P2 | I1, I2, I3, I4, I5, I7, I8, I9, I10 | 54 | 84 | Complete every step of the cumulative table |
| P3 | W1, W2, W3, W4, W5, W6, W7, W8, W11 | 54 | 138 | Complete all component removals; W0/W9/W10/W12 already available |
| P4 — supplementary | B1, B2, C0, C1, C2, C3, A1, A2 | 48 | 186 | Explain the joint-routing step using all remaining existing presets |
| P5 — supplementary | K1, K2 | 12 | 198 | Match fusion capacity and isolate attention order |

The **12-row cumulative table alone costs 72 runs**. Adding W9/W12 produces the recommended 84-run build-up and language-control package. Completing full-model removals takes 138 runs. P4/P5 are additional mechanism studies; they need not displace the requested image-only/component story.

Run the five P1 anchors with seed 42 on both datasets first, then complete seeds 1337 and 2026 for all five. Use this pilot to catch invalid controls and estimate resources. Continue the declared matrix or explicitly narrow the paper's scope; do not select only favorable seeds or datasets. Report negative architectural results as well as positive ones.

The matched historical B0 and C4 checkpoints could fill I10/I11 seed-42 QaTa cells after the reuse audit in section 3. Historical C1 is relevant only when P4 is included. Revised augmentation, targets, splits, or common-initialization policies invalidate such reuse. Treat these as at most two reusable cells in P1–P3 and one additional cell in P4, not evidence for image-only performance.

If a five-seed anchor study is funded, add seeds **7 and 31415** to I0/I6/I11/W9/W12 on both datasets: **20 additional runs**. Decide the seed count before confirmatory test inspection. Five seeds improve the variance estimate but do not guarantee power or significance.

Historical QaTa full-training times were about 2 h 45 min for B0, 3 h 55 min for C1, and 3 h 06 min for matched C4. Image-only and new removal variants have not been timed. As a rough scheduling proxy, 69 QaTa runs at 3–4.5 hours plus 69 MosMed runs at 1–1.5 hours gives **276–414 GPU-hours** for the 138-run package, about 11.5–17.3 continuous days on a comparable single GPU. This is not a measured prediction: image-only runs may be cheaper, and longer training or deterministic kernels may increase time. Reforecast from the five P1 conditions; the older timing log estimated roughly nine hours for a full 201-epoch QaTa router run.

External comparator training, sensitivity sweeps, repeatability pilots, and new datasets are additional. For example, three reproduced external methods across two datasets and three seeds add 18 runs. Archive failures and include their cost in the actual study compute report.

## 12. External comparators and related-work positioning

Use the same evaluation manifests, available prompt information, and metric definitions for local reproductions. Distinguish published numbers from numbers produced by the common evaluator; differences in splits, target masks, resolution, supervision, or TTA prevent direct ranking. Keep official-recipe reproductions distinct from backbone-matched adaptations, and budget tuning transparently.

| Comparator | Role | Source / qualification |
|---|---|---|
| I0, I6, W12 image-only models | Simple, strong, and routed same-project visual controls | Include all three; they do not replace established external methods |
| Published LViT | Closest historical text-guided reference | [Paper](https://arxiv.org/abs/2206.14718) and [official implementation](https://github.com/HUANGLIZI/LViT); separate from this repository's B0 |
| RecLMIS | Alternative cross-modal alignment/reconstruction mechanism | [Primary paper](https://arxiv.org/abs/2404.02845); reproduce only after checking its data and supervision protocol |
| RADAR | Relevant contemporary depth-aggregation and alignment comparison | [Primary paper](https://arxiv.org/abs/2610.14944); its foundation models and adaptation regime differ, so report pretraining and total/frozen/trainable parameters |

RADAR describes spatial, text-conditioned selection of historical visual features, and its methodology also discusses text-branch depth aggregation. Therefore “first spatial text routing” or “first use of language depth” is not a supported novelty statement. A narrower candidate distinction is **joint allocation among separately normalized linguistic-depth attention responses, previous-stage context, and a zero-update candidate**. Treat this as a positioning hypothesis requiring the controls above and a broader literature review, not an established priority claim. See [RADAR, section 3.2](https://arxiv.org/html/2610.14944v1#S3.SS2).

The current encoder is explicitly specialized for chest X-rays according to [Microsoft's model card](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized). Discuss its use on CT descriptions as a design choice; a second dataset does not by itself establish that this is the best language encoder.

## 13. Paper deliverables and reporting templates

### Tables and figures

| Artifact | Required content |
|---|---|
| Table 1 — main results | External methods, I0, I6, W12, W9, and I11; both datasets; E0 metrics and mean ± sample SD; E1/E2 reported separately |
| Table 2 — cumulative additions | All 12 I0–I11 rows; each added component, score, delta from previous row, and gains over both image-only baselines |
| Table 3 — full-model removals | W0–W12, with repeated conditions reused; full-minus-removal effects, uncertainty, and changed compute |
| Table 4 — mechanisms/interactions | PEAL × routing cells, optional C0–C4/source controls and K1/K2; separate package effects from isolated mechanisms |
| Figure 1 — architecture | Four language groups, shared text attention, memory availability, zero response, and explicit query ordering |
| Figure 2 — architecture progression | Measured I0–I11 scores with uncertainty and image-only references; an adjacent plot of full-minus-removal effects; show any decreases |
| Figure 3 — mechanism | Weight and contribution maps, entropy/utilization, candidate similarity; matched successes and failures |
| Figure 4 — trade-offs | Accuracy versus native latency/memory; sensitivity curves; E0/E1/E2 comparison |
| Supplement | Individual seeds, complete hyperparameters, split/prompt provenance, all sensitivity and failed-run records, subgroup and robustness analyses |

Use lesion-area bins derived from training data and frozen before analysis, for example training-set tertiles of foreground area fraction; report sample counts. Do not define “small lesion” from whichever test split makes the gain largest. Additional subgroup analyses need sufficient cases and available, reliable metadata.

Suggested machine-readable outputs to implement:

```text
results/run_registry.csv
  run_id, protocol_id, dataset, split_manifest_hash, prompt_manifest_hash,
  experiment_id, seed, git_commit, config_hash, init_hash, checkpoint_hash,
  status, best_epoch, epochs_run, optimizer_steps, gpu_hours, artifact_paths

results/per_case_metrics.csv
  run_id, eval_protocol, case_id, patient_id_if_known, dice, iou,
  tp, fp, fn, gt_area, pred_area, hd95_px, assd_px, empty_case_type,
  threshold, inference_precision, prompt_condition, routing_override, routing_seed

results/summary.csv
  experiment_id, dataset, eval_protocol, seed, macro_dice, macro_iou,
  global_dice, global_iou, boundary_summary, failed_boundary_count,
  parameter_counts, latency_ms, peak_allocated_mib, peak_reserved_mib
```

Store full resolved run configuration, frozen transform/label policy, environment lock, pretrained model revisions, commands, warnings, per-epoch curves, best checkpoint, validation calibration metadata, and case-level metrics with each run. Aggregate global scores from confusion counts, not averages of per-image Dice. Store probabilities losslessly when feasible; current image mask exports alone cannot reproduce a threshold sweep.

The repository's requirements are unpinned and its data/run folders are ignored. Archive artifacts and dependency versions separately; a Git commit alone does not reproduce a run. Do not place patient identifiers or restricted raw data in a public release.

For manuscript reporting, use [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300) to check that data provenance, partitions, preprocessing, model selection, external testing limitations, and failure analysis are described. This is reporting guidance, not evidence that the study has satisfied those requirements.

## 14. Implementation backlog before launching publication runs

| Priority | Work | Completion criterion |
|---|---|---|
| Required | Freeze dataset/prompt/label/augmentation manifests | Patient overlap and directional-text checks documented; exact protocol version recorded |
| Required | Evaluation and calibration selection | E0/E1/E2 selectable explicitly; calibration bound to checkpoint, validation manifest, precision, and inference settings; no accidental reuse |
| Required | Case-level metrics and uncertainty export | CSVs reproduce printed overlap results; boundary conventions and cluster-aware analysis checked on known masks |
| Required | Reproducible run setup | Independent processes, all settings serialized, common initialization verified, transform RNG/backend policy recorded |
| Required first | I0, I6, W12 image-only paths | No text encoder/tokenizer/CSV dependency or PEAL; memory/zero routing works at stage 1; output invariant to optional text |
| Required | Independent I/W component switches | Visual refinement, memory, frequency, ASPP, tail, deep supervision, text source, fixed/learned mixing, PEAL, and routing serialize and reload correctly |
| Required | Benchmark harness | Synchronized native/TTA timing, allocated/reserved memory, profiler coverage, BERT inclusion documented |
| Required for mechanism figures | Diagnostic/perturbation exporter | Stable case selection, paired perturbations, response norms/similarity and stage/block maps |

These are planned tasks. This document does not implement them. Existing tests provide useful starting points in [tests/test_training_evaluation.py](tests/test_training_evaluation.py), [tests/test_joint_router.py](tests/test_joint_router.py), and [tests/check_priority0.py](tests/check_priority0.py). Re-run the relevant checks after actual code changes rather than treating historical pass counts as validation of new features.

## 15. Commands supported by the current repository

Run from the project root in the established training environment. Commands here are instructions for future execution, not commands executed when writing this memory file.

Preview a configuration without training or downloading pretrained weights:

```bash
python train_model.py --experiment router_region --seed 42 --dry-run
```

These commands cover the already implemented I9, I10, I11, and W9 respectively, after P0 and the protocol lock. Other available presets are mapped in section 8. These commands do not cover the image-only anchors:

```bash
python train_model.py --experiment baseline_no_peal --seed 42 --batch-size 16
python train_model.py --experiment baseline --seed 42 --batch-size 16
python train_model.py --experiment router_region --seed 42 --batch-size 16
python train_model.py --experiment router_no_peal --seed 42 --batch-size 16
```

Each command launches an independent process. Repeat with seeds 1337 and 2026, and with the second dataset's configuration. **There is no `--dataset` argument in `train_model.py`.** Configure all dataset paths before launch and verify the dry-run snapshot. Changing only `config.task_name` after `Config.py` has been imported does not recompute its already-assigned paths. The checkpointing flag only enables checkpointing; verify `Config.gradient_checkpointing=False` for the reference policy.

Existing sensitivity and historical-style evaluation commands:

```bash
python train_model.py --experiment router_region --seed 42 --router-scale 2
python train_model.py --experiment router_region --seed 42 --router-temperature 0.5
python test_model.py --checkpoint path/to/best_model-LViT.ckpt
python test_model.py --checkpoint path/to/best_model-LViT.ckpt --routing-override image_mean
python test_model.py --checkpoint path/to/best_model-LViT.ckpt --routing-override shuffle --routing-seed 0
```

Replace the checkpoint placeholder. These evaluation commands reuse saved threshold/TTA metadata; they are not guaranteed to implement E0. The test script also reads test paths from current `Config.py`, so verify that the selected checkpoint and dataset match. I0–I8, W12, and the other planned switches need implementation before runnable commands can be supplied. In particular, current router normalization does not accept a language-off mode. Do not invent CLI names or use blank prompts as a substitute.

## 16. Persistent decisions and next action

- Main architecture story: I0 image-only scaffold → I6 strong image-only → I7 simple text → I8 fixed hierarchy → I9 learned hierarchy → I10 PEAL → I11 joint router. Visual components are added individually at I1–I6.
- Primary evaluation: native single-pass macro Dice at threshold 0.5; calibrated and TTA performance are separate outcomes.
- Existing gains are small and single-seed. Replication and matched controls take priority over increasing depth.
- First full experiment block: I0/I6/I11/W9/W12 × two datasets × seeds 42/1337/2026, after P0. Then complete all cumulative steps and full-model removals.
- Before that block: implement genuine image-only paths, independent component switches, explicit evaluation selection and per-case exports; resolve patient/prompt/augmentation/mask provenance.
- Preserve completed, negative, failed, and interrupted experiments in the log. Never replace an inconvenient completed seed with a favorable rerun; mark technical reruns and their exclusion reasons explicitly.
- Update this memory only when a protocol decision changes; append actual outcomes and artifact locations to `EXPERIMENT_LOG.md`. Record deviations with a date and whether they were decided before or after inspecting test results.

The paper should show measured improvement over image-only models and measured component contributions. If a component has zero or negative benefit, retain that result and qualify the claim; the plan does not require a monotonically improving table.
