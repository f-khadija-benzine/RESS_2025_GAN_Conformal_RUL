# Calibration-size check: 2 calibration bearings vs 14 (out-of-fold pooled)

PICP±std / clipped MPIW, alpha = 0.10


## near-failure windows

| calibration | grouping | baseline | +GAN | +MT | +GAN+MT |
|---|---|---|---|---|---|
| 2 bearings (current) | single q | 0.871±0.105 / 0.69 | 0.872±0.103 / 0.70 | 0.854±0.104 / 0.67 | 0.839±0.080 / 0.63 |
| 2 bearings (current) | true stage | 0.860±0.027 / 0.47 | 0.857±0.036 / 0.47 | 0.853±0.056 / 0.48 | 0.864±0.025 / 0.49 |
| 2 bearings (current) | predicted stage | 0.887±0.046 / 0.65 | 0.902±0.037 / 0.65 | 0.901±0.057 / 0.69 | 0.884±0.046 / 0.67 |
| 14 bearings (pooled OOF) | single q | 0.966±0.040 / 0.88 | 0.968±0.033 / 0.91 | 0.974±0.022 / 0.90 | 0.960±0.034 / 0.83 |
| 14 bearings (pooled OOF) | true stage | 0.864±0.050 / 0.45 | 0.872±0.044 / 0.46 | 0.820±0.066 / 0.45 | 0.864±0.027 / 0.44 |
| 14 bearings (pooled OOF) | predicted stage | 0.960±0.032 / 0.84 | 0.961±0.032 / 0.86 | 0.959±0.034 / 0.83 | 0.949±0.025 / 0.82 |

## all test windows

| calibration | grouping | baseline | +GAN | +MT | +GAN+MT |
|---|---|---|---|---|---|
| 2 bearings (current) | single q | 0.793±0.069 / 0.68 | 0.804±0.069 / 0.69 | 0.833±0.052 / 0.64 | 0.830±0.050 / 0.59 |
| 2 bearings (current) | true stage | 0.786±0.052 / 0.55 | 0.792±0.051 / 0.57 | 0.819±0.051 / 0.49 | 0.811±0.034 / 0.46 |
| 2 bearings (current) | predicted stage | 0.740±0.022 / 0.63 | 0.757±0.016 / 0.64 | 0.798±0.028 / 0.60 | 0.817±0.047 / 0.54 |
| 14 bearings (pooled OOF) | single q | 0.893±0.092 / 0.85 | 0.909±0.075 / 0.87 | 0.909±0.075 / 0.84 | 0.862±0.072 / 0.76 |
| 14 bearings (pooled OOF) | true stage | 0.882±0.074 / 0.76 | 0.896±0.062 / 0.78 | 0.890±0.062 / 0.75 | 0.872±0.051 / 0.68 |
| 14 bearings (pooled OOF) | predicted stage | 0.849±0.037 / 0.66 | 0.845±0.044 / 0.67 | 0.878±0.037 / 0.63 | 0.884±0.037 / 0.53 |
