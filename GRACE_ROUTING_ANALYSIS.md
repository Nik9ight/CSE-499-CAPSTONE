# GRACE: Region-wise Language-Depth Routing Analysis

Date: 2026-10-23  
Status: Original design assessment; see the implementation update below.

Implementation update (2026-10-26): the optional prototype is now implemented and synthetically tested. See [GRACE_ROUTER_EXPERIMENTS.md](GRACE_ROUTER_EXPERIMENTS.md) for the implemented choices and evaluation plan. The analysis below records the original design assessment; no real-data performance improvement has been established.

## Overall Assessment

The region-wise language-depth router is the strongest fit for making GRACE's contribution more identifiable. It targets an existing design constraint: each stage currently receives one assigned text group and adds previous-stage context through a separate branch.

Its potential benefit is better control over guidance. Its main risks are increased computation, correlated text candidates, and interaction with PEAL. Accuracy improvements remain hypotheses until tested.

## 1. Relationship to Current GRACE

Current GRACE already performs spatially varying text attention: different visual tokens can attend to different words. The proposed addition would select **which linguistic depth and guidance source** those tokens use.

| Mechanism | Current implementation | Proposed change |
|---|---|---|
| Within-group BERT-layer mixture | Learned weights shared across samples | Can remain unchanged |
| Text-group assignment | Group 1 to stage 1, etc. | All four groups available at every stage |
| Previous-stage context | Separate residual attention update | Competes with language candidates |
| Guidance suppression | Residual scaling; optional adaptive gate currently disabled | Explicit no-update candidate |
| Spatial decisions | Attention over words and visual tokens | Additional distribution over guidance sources |

The existing `AdaptiveGatedFusion` is an especially important baseline. It already provides input-dependent, per-token/channel control over the text update, although it is currently disabled. The router needs to demonstrate value beyond simply enabling that gate.

Relevant implementation: [nets/ViT2.py](nets/ViT2.py), [nets/LViT_new.py](nets/LViT_new.py), and [Config.py](Config.py).

## 2. Integration Point

The natural insertion point is the guidance section of `CascadeViTBlock`.

```text
Current:
visual -> text attention + residual
       -> previous-stage attention + residual
       -> self-attention -> MLP

Proposed:
visual -> four text responses + previous-stage response
       -> regional source routing + residual
       -> self-attention -> MLP
```

The frequency module, backbone, ASPP, decoder, and auxiliary segmentation heads can remain unchanged.

**Attention order is a confound.** Current memory attention queries visual features after their text update. The proposal instead suggests parallel candidates computed from the same incoming visual representation. Replacing the block therefore changes both routing and attention order.

A learned-static-routing control must use the same parallel candidate construction; otherwise, improvements could come from that structural change.

With two blocks per stage, routing inside every existing guidance block would produce **eight routing distributions**, each shaped `[B, N_s, 6]`. A single router per stage would be a separate architectural variant.

## 3. Mathematical Definition

Let $T_g$ denote the four existing text groups and $V_s$ the incoming visual tokens. The block index is omitted for readability.

Construct text candidates using shared attention parameters within each block:

$$
C_{s,g}
=
\operatorname{CrossAttn}_{s}
\left(\operatorname{LN}(V_s),T_g,T_g;\text{text mask}\right).
$$

For stages with previous-stage tokens:

$$
M_s
=
\operatorname{CrossAttn}^{\mathrm{memory}}_s
\left(\operatorname{LN}(V_s),V_{s-1},V_{s-1}\right).
$$

These responses already have the current stage's channel dimension, so no spatial resizing of previous-stage tokens is required.

Normalize and scale candidates consistently, then route:

$$
r_s(p)
=
\operatorname{softmax}
\left(a_s(p)/\tau+\mathcal{M}_s\right),
$$

$$
V'_s(p)
=
V_s(p)+
\eta_s
\left[
\sum_{g=1}^{4}r_{s,g}(p)\Delta C_{s,g}(p)
+
r_{s,m}(p)\Delta M_s(p)
\right].
$$

Here:

- $a_s(p)$ contains six router logits; $\tau$ is the softmax temperature.
- $\mathcal{M}_s$ masks unavailable candidates. At stage 1, set the memory logit to $-\infty$, excluding it from normalization.
- The none candidate contributes exactly zero.
- $\Delta C$ and $\Delta M$ denote consistently normalized/scaled candidate responses.
- $\eta_s$ controls the overall guidance magnitude.
- The existing text padding mask must reach every candidate attention operation.

The router can use local visual features, masked text summaries, and the local memory response. Including candidate-response information would let it judge what guidance actually says at that location.

Two qualifications matter:

1. Finite softmax weights generally provide approximate guidance suppression rather than an exact identity decision.
2. Selecting none preserves the input to this guidance residual. Self-attention and the MLP still update it, and earlier stages may already contain language information.

This is a dense mixture of guidance responses. Computing all candidates does not itself provide sparse execution or computational savings.

## 4. Potential Segmentation Impact

| Potential benefit | Why it could help | What could prevent it |
|---|---|---|
| Better foreground localization | Different locations can use different text representations | BERT groups may carry nearly identical task-relevant information |
| Fewer background false positives | Background tokens can suppress unnecessary guidance | The router may learn a global none preference instead |
| More useful cascade context | Memory can be suppressed where it propagates misleading evidence | Memory can dominate because its response scale differs |
| Better boundary behavior | Fine-scale tokens can retain visual detail | Routing stops at backbone resolutions; boundary reconstruction also depends on the decoder |
| Better text robustness | Weak guidance can receive less weight | A router can confidently select an incorrect description |

Region granularity is stage-dependent:

| Stage | Routing grid at a 224-pixel input |
|---|---|
| 1 | $56\times56$ |
| 2 | $28\times28$ |
| 3 | $14\times14$ |
| 4 | $7\times7$ |

Stage 4 makes relatively coarse spatial decisions. Its routing maps should not be interpreted as precise lesion-boundary explanations.

Avoid assigning fixed meanings such as "anatomy" or "disease semantics" to particular BERT groups. Measure group similarity and examine routing behavior before making those interpretations.

## 5. Interaction with PEAL

Current PEAL aligns each stage's pooled visual representation with its originally assigned text group. See [lightning_module.py](lightning_module.py).

That creates a possible conflict:

```text
Router: this region benefits from group 4.
PEAL: this stage's pooled representation should align with group 1.
```

Because PEAL operates globally and through learned visual projections, this is not a strict contradiction. It can nevertheless bias the model toward the original stage/group assignment.

Keeping PEAL unchanged initially is appropriate for isolating the module. However, if routing collapses toward the original assignment, that would not establish that dynamic selection is useless.

A follow-up four-run comparison would help:

| Model | Original PEAL weight | PEAL disabled |
|---|---|---|
| Corrected GRACE baseline | Run | Run |
| GRACE with router | Run | Run |

This separates the router's usefulness from its compatibility with the existing alignment objective.

## 6. Computation and Parameter Impact

BERT does not need extra forwards: GRACE already extracts all 12 layers and constructs all four groups.

The extra work occurs in visual-to-text attention. For the current configuration, an analytical count gives:

| Quantity | Current | Four candidates in every block |
|---|---|---|
| Text-attention evaluations | 8 | 32 |
| Text-attention parameters with independent modules | 7.57 million | 30.27 million |

Four independent attention modules would therefore add approximately **22.71 million parameters**, excluding the router itself.

These parameter estimates use two blocks per stage, visual dimensions $(96,192,384,768)$, and text dimension 768. A text-attention module at visual dimension $C$ has $2C^2+2C\cdot768+4C$ parameters, including projection biases. They are analytical counts, not measured whole-model FLOPs or latency.

**Share text-attention projections across the four groups within each block.** That keeps the comparison focused on linguistic depth and avoids attributing improvements to a large capacity increase. It still increases attention computation and activation storage.

The router's six-weight output is inexpensive; candidate generation is the main cost. Overall inference would not necessarily become four times slower because the backbone, BERT, self-attention, memory attention, and decoder remain shared.

Concatenating all four groups and running one ordinary attention operation is not equivalent to separately normalizing attention within each group and then routing their responses.

## 7. Essential Experimental Controls

| Comparison | Main question |
|---|---|
| Corrected GRACE versus existing adaptive gate enabled | Does a conventional gate already explain the benefit? |
| Fixed versus learned-static source weights, using identical candidates | Does flexible source mixing suffice? |
| Static versus input-dependent image-level routing | Does conditioning help? |
| Image-level versus token-level routing | Is spatial variation necessary? |
| Language routing only versus memory selection only versus joint routing | Does joint selection provide additional value? |
| With versus without none | Does guidance suppression help? |
| Comparable-capacity conventional fusion | Are gains explained by extra capacity? |

### Control guidance magnitude

Current GRACE can add a full text residual and a full memory residual. A softmax mixture shares a unit budget between candidates. Uniform initialization therefore reduces individual updates substantially.

An improvement could partly result from weaker guidance. Match residual scaling across controls and report the learned scale.

### Evaluation protocol

- Use the corrected GRACE implementation as the baseline.
- Keep splits, training budgets, augmentations, threshold selection, and TTA consistent.
- Report multiple seeds, Dice, IoU, and a boundary metric with explicit units and empty-mask handling on both datasets.
- Report parameters, latency, and peak GPU memory; separate single-pass and TTA costs.
- Measure candidate similarity, routing entropy, none usage, and memory usage.
- Show both routing weights and weighted response magnitudes. Large weights do not necessarily mean large contributions.
- Test uniform, spatially averaged, or shuffled routing at inference as diagnostic interventions, alongside separately trained controls.
- Do not force balanced candidate usage merely to obtain attractive routing maps. A consistently useful source may legitimately dominate.

## 8. Contribution Positioning Relative to RADAR

The cited paper confirms spatial, text-conditioned aggregation over historical visual representations. It also applies feature-intrinsic depth aggregation inside the text encoder, so "using language depth" alone would be too broad a distinction. See [RADAR, Section 3.2](https://arxiv.org/html/2610.14944v1#S3.SS2).

The more defensible proposed contribution is:

> Joint, spatially conditioned allocation of guidance across linguistic-depth responses, inherited cross-stage context, and a zero-update candidate.

That is a mechanism to investigate, not yet a verified novelty claim. A broader related-work review would still be necessary.

## 9. Alternative Directions

| Direction | Fit with current GRACE | Assessment |
|---|---|---|
| Selective cascade block | Small change to the existing memory branch | Best low-cost diagnostic. If memory selection alone explains the gains, the full language-depth router may be unnecessary. |
| Anatomical phrase-grounding block | Requires phrase handling and explicit spatial correspondence | Potentially strong, but substantially larger scope. Current flips, rotations, and flip TTA would need consistent handling of laterality/location phrases. |

## Recommendation

Prototype the shared-projection joint router, preserve frequency enhancement and PEAL initially, and make the parallel static-mixture and existing adaptive-gate baselines mandatory.

Those comparisons will determine whether GRACE gains from meaningful regional source selection, reduced guidance strength, or conventional fusion flexibility.
