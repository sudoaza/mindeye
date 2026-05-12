# Phase 3.5: Full-Window ZUNA Recovery Report
**Date**: May 12, 2026
**Target**: Validate if using a continuous 5s window (-1.0s to +4.0s) of ZUNA-denoised EEG recovers semantic retrieval (EEG-to-CLIP).

## Pipeline Execution Details
The automated `execute_recovery_v2.sh` pipeline successfully ran end-to-end:
1. Downloaded 8 runs for `sub-01` from OpenNeuro.
2. Synced only the necessary 4000 targeted ImageNet stimuli from S3 (preventing previous storage explosions).
3. Processed continuous runs through the 15-step offline ZUNA batch pipeline.
4. Cropped the full 5s windows around the `stim_on` event (physically anchored at +1.0s).
5. Generated HuggingFace CLIP image (`sub01_image_embeddings.pt`) and text (`imagenet_text_embeddings.pt`) embeddings natively.
6. Conducted a baseline matrix training run using `temporal_attn`.

## Matrix Results (Sprint 2 Gate)
The matrix evaluated `zuna_runheldout` across 50 epochs using the `image_text` semantic target.

| Metric | Result | Target/Expected Random | Status |
|--------|--------|------------------------|--------|
| **Top-1** | 0.000 | ~0.008 | ❌ Below random |
| **Top-5** | 0.033 | ~0.041 | ❌ Below random |
| **Top-10**| 0.074 | ~0.082 | ❌ Below random |
| **MRR** | 0.038 | - | ❌ Poor |
| **Collapse Score** | 0.088 | > 0.1 | ❌ Collapsed |

*(Note: The matrix automatically skipped processing the control conditions because the primary `zuna_runheldout` completely failed the absolute collapse score gate.)*

## Conclusion
The full 5s window **did not** recover the semantic signal. The model experienced severe dimensional collapse (`collapse_score = 0.088`), outputting essentially the same average vector regardless of the input EEG, leading to below-random retrieval performance.

The label noise diagnostic during cropping showed an average of `1.98` other stimuli within each 5s window. It is highly likely that the long 5-second window is polluting the attention mechanism with multiple overlapping semantic events, destroying the contrastive learning signal.

## Recommended Pivot
As established in the protocol, since the full-window fallback failed to beat the baseline, we must abandon the `image_text` CLIP target for this stage and pivot to the **text-only supervision fallback**. We should attempt to map EEG directly to simpler, highly discriminative text embeddings, avoiding the noisy image modality for the foundational alignment.
