# Comparison

Held-out test photos: 123. Ranges are 95% bootstrap intervals.

| Metric | clip_zeroshot | efficientnet_b0_finetune | efficientnet_b0_linear |
|:---|---:|---:|---:|
| Macro AP (ranking quality, 8 damage types) | 0.50 (0.46-0.56) | 0.53 (0.48-0.59) | 0.54 (0.50-0.60) |
| Macro AP expected from guessing (average share of photos with each damage) | 0.42 | 0.42 | 0.42 |
| Micro F1 at validation thresholds | 0.66 (0.62-0.69) | 0.71 (0.67-0.74) | 0.71 (0.68-0.74) |
| Photos with every label right | 0.02 | 0.04 | 0.04 |
| Calibration error (lower is better) | 0.12 | 0.13 | 0.14 |
| Photos automated at <= 10% error | 0.00 | 0.00 | 0.00 |
| Error among automated photos | 0.00 | 0.00 | 0.00 |
| Macro AP, blur test photos | n/a | 0.47 | 0.51 |
| Macro AP, low light test photos | n/a | 0.50 | 0.51 |
| Macro AP, jpeg test photos | n/a | 0.46 | 0.51 |
| Macro AP, low res test photos | n/a | 0.45 | 0.49 |

## Is the best run really better? (same photos, paired comparison)

| Best run | Compared with | Macro AP difference | 95% range | Clearly better? |
|:---|:---|---:|---:|:---|
| efficientnet_b0_linear | clip_zeroshot | +0.04 | -0.02 to +0.11 | not clearly |
| efficientnet_b0_linear | efficientnet_b0_finetune | +0.01 | -0.04 to +0.07 | not clearly |

## Average precision per damage type

Guess = what a model with no skill would score (the share of test photos with that damage).

| Damage | Test photos | Guess | clip_zeroshot | efficientnet_b0_finetune | efficientnet_b0_linear |
|:---|---:|---:|---:|---:|---:|
| Dent | 94 | 0.76 | 0.81 | 0.79 | 0.77 |
| Scratch | 78 | 0.63 | 0.76 | 0.66 | 0.62 |
| Broken part | 93 | 0.76 | 0.87 | 0.84 | 0.83 |
| Missing part | 66 | 0.54 | 0.56 | 0.69 | 0.75 |
| Paint chip | 42 | 0.34 | 0.37 | 0.47 | 0.44 |
| Flaking | 17 | 0.14 | 0.25 | 0.27 | 0.31 |
| Corrosion | 16 | 0.13 | 0.12 | 0.23 | 0.33 |
| Cracked | 11 | 0.09 | 0.26 | 0.30 | 0.31 |

## Photos that could skip review, one damage type at a time

Share of test photos where the decision for that damage type could be automated at <= 10% error (cutoff chosen on validation), with the error actually seen on the test photos in brackets. Far less strict than needing all 8 types right at once.

- n/a (guess) = answering the same way for every photo is already within the error limit.
- n/a (no skill) = that run's AP for the type is within 0.10 of guessing, so a high share would only reflect how common the type is, not what the model learned.

| Damage | clip_zeroshot | efficientnet_b0_finetune | efficientnet_b0_linear |
|:---|---:|---:|---:|
| Dent | n/a (no skill) | n/a (no skill) | n/a (no skill) |
| Scratch | 0.04 (error 0.00) | n/a (no skill) | n/a (no skill) |
| Broken part | 0.12 (error 0.00) | n/a (no skill) | n/a (no skill) |
| Missing part | n/a (no skill) | 0.02 (error 0.00) | 0.02 (error 0.50) |
| Paint chip | n/a (no skill) | 0.00 (error 0.00) | n/a (no skill) |
| Flaking | 0.00 (error 0.00) | 0.01 (error 0.00) | 0.35 (error 0.14) |
| Corrosion | n/a (no skill) | 0.74 (error 0.15) | 0.50 (error 0.11) |
| Cracked | n/a (guess) | n/a (guess) | n/a (guess) |
