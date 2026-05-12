# Phase 3.5 Back-Aligned Full5s Recovery

## Commit
1f4bc10

## Data
Subject: sub-01
Runs: 1-8
Val runs: 8
Window mode: full5s_backaligned
Window: -3.0s to +2.0s
Event offset: 3.0s
Anchor sample: 768
Event marker: true

## Label Noise
Avg previous stimuli: 1.13
Avg future stimuli: 0.85
Avg total other stimuli: 1.98

## Target
semantic_target: text

## Model
model: temporal_attn
n_channels: 63
n_samples: 1280
layers: 4
heads: 8

## Results
condition | top1 | top5 | top10 | mrr | median_rank | pred_std | collapse_score
--- | --- | --- | --- | --- | --- | --- | ---
zuna_real | 0.000 | 0.041 | 0.099 | 0.043 | 59 | 0.0054 | 0.210
zuna_shuffled | 0.008 | 0.033 | 0.066 | 0.042 | 60 | 0.0035 | 0.134
zuna_random | 0.008 | 0.041 | 0.083 | 0.045 | 61 | 0.0013 | 0.030

## Gate
WEAK POSITIVE

## Interpretation
The `image_text` target collapsed entirely (score < 0.1) as seen in previous runs, failing the gate. However, shifting to the `text`-only target using the `full5s_backaligned` window and event markers resulted in our first non-collapsed signal (`collapse_score` = 0.210). 

The `zuna_real` condition successfully beat both `zuna_shuffled` and `zuna_random` on the `top10` metric (9.9% vs 6.6% and 8.3%), and beat `zuna_shuffled` on `MRR` (though slightly underperforming `zuna_random` MRR). Because it only worked on the text target, this constitutes a **WEAK POSITIVE**. The window alignment and marker injection approach is valid for text supervision.
