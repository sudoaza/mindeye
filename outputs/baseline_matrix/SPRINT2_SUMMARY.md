# Sprint 2 Baseline Matrix Summary

## Commit
Local state (not yet pushed to master)

## Data
- **Subject**: sub-01, Session: ImageNet01, Runs: 01–05
- **ZUNA Crop duration**: 1.25s ([-0.25, 1.0] relative to stim_on)
- **Validation**: run-heldout (Runs 01–04 train, Run 05 val)
- **Items**: 495 Train / 123 Val

## Conditions
- `raw_runheldout`: Raw EEG (250Hz)
- `resample_runheldout`: Resampled EEG (256Hz)
- `zuna_runheldout`: ZUNA Output (256Hz)
- `zuna_shuffled_labels`: ZUNA + Permuted CLIP targets
- `zuna_random_targets`: ZUNA + Gaussian random targets
- `zuna_sameclass`: ZUNA + different image from same synset

## Gate Result: FAIL ❌
The ZUNA-real condition did not outperform randomized controls.

## Metrics Table (Val set, n=123)
| Condition | top1 | top5 | top10 | mrr | median_rank | collapse_score | status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **zuna_runheldout** | **0.000** | **0.041** | **0.081** | **0.042** | **54.0** | **0.428** | ok |
| raw_runheldout | 0.008 | 0.033 | 0.073 | 0.042 | 62.0 | 0.279 | ok |
| resample_runheldout | 0.008 | 0.041 | 0.081 | 0.045 | 58.0 | 0.275 | ok |
| zuna_shuffled_labels | 0.008 | 0.033 | 0.089 | 0.043 | 62.0 | 0.065 | ok |
| zuna_random_targets | 0.000 | 0.016 | 0.114 | 0.039 | 52.0 | 0.915 | ok |
| zuna_sameclass | 0.000 | 0.041 | 0.081 | 0.042 | 54.0 | 0.428 | ok |

## Notes & Observations
- **Retrieval Performance**: The Top-10 accuracy of ~8% is almost exactly the random chance for a 123-item pool (~8.1%). 
- **ZUNA vs Raw**: ZUNA denoising shows no measurable gain over raw or resampled EEG yet.
- **Control Mismatch**: `zuna_random_targets` actually showed a higher Top-10 (0.114) than the real condition, which strongly suggests the model is mapping EEG to a generic region of the CLIP hypersphere rather than learning semantic features.
- **Collapse Score**: `0.428` is healthy (not collapsed), but the `pred_std` is low (~0.012), indicating the model output is quite clustered.

## Immediate Action: Debugging
As per Step 4 "Fail condition", we will now investigate:
1. **Event/Crop Alignment**: Verify `stim_on` annotations are correctly aligned with the actual visual onset.
2. **CLIP Centering**: Check if subtracting the global CLIP mean or applying stronger normalization to targets helps.
3. **Architecture**: Consider if a deeper encoder or longer window is needed.
4. **Data Normalization**: Check the range of ZUNA outputs (std~0.1 is expected; currently we normalize per-crop).
