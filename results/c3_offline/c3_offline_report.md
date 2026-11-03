# Offline C3 analysis (100 saved prediction files, no retraining)

## Shared stage classifier, test windows pooled over folds and seeds

Confusion matrix (rows = true stage, columns = predicted), row %:

| true \ pred | healthy | early | near-failure | n |
|---|---|---|---|---|
| healthy | 77.5 | 9.6 | 12.9 | 30260 |
| early | 10.5 | 39.5 | 50.1 | 7330 |
| near-failure | 1.6 | 15.8 | 82.5 | 6165 |

| stage assignment | accuracy | near-failure recall | near-failure precision |
|---|---|---|---|
| raw | 0.686 | 0.803 | 0.586 |
| ratchet k=5 | 0.656 | 0.840 | 0.587 |
| ratchet k=10 | 0.668 | 0.821 | 0.614 |
| ratchet k=20 | 0.692 | 0.776 | 0.660 |
| ratchet k=30 | 0.706 | 0.725 | 0.728 |

## Coverage / width on near-failure windows (true stage), alpha = 0.1

PICP | MPIW raw | MPIW clipped to [0,1] — mean ± std over 5 seeds

| variant | baseline | +GAN | +MT | +GAN+MT |
|---|---|---|---|---|
| single q (no groups) | 0.871±0.105 / 1.22 / 0.69 | 0.872±0.103 / 1.20 / 0.70 | 0.854±0.104 / 1.10 / 0.67 | 0.839±0.080 / 1.01 / 0.63 |
| true stage (oracle) | 0.860±0.027 / 0.72 / 0.47 | 0.857±0.036 / 0.71 / 0.47 | 0.853±0.056 / 0.66 / 0.48 | 0.864±0.025 / 0.66 / 0.49 |
| predicted stage | 0.879±0.057 / inf / 0.65 | 0.894±0.047 / inf / 0.65 | 0.893±0.064 / inf / 0.69 | 0.876±0.049 / inf / 0.67 |
| predicted stage + ratchet k=5 | 0.882±0.044 / inf / 0.63 | 0.895±0.041 / inf / 0.63 | 0.898±0.060 / inf / 0.66 | 0.880±0.049 / inf / 0.64 |
| predicted stage + ratchet k=10 | 0.874±0.044 / 1.04 / 0.62 | 0.887±0.041 / 1.02 / 0.63 | 0.896±0.060 / 1.01 / 0.65 | 0.875±0.052 / 0.97 / 0.64 |
| predicted stage + ratchet k=20 | 0.854±0.054 / inf / 0.62 | 0.862±0.053 / inf / 0.63 | 0.866±0.068 / inf / 0.64 | 0.859±0.066 / inf / 0.63 |
| predicted stage + ratchet k=30 | 0.833±0.051 / inf / 0.60 | 0.840±0.055 / inf / 0.60 | 0.844±0.085 / inf / 0.62 | 0.847±0.063 / inf / 0.61 |
| predicted-RUL bins rul3 | 0.890±0.051 / 1.24 / 0.73 | 0.896±0.055 / 1.20 / 0.73 | 0.889±0.060 / inf / 0.73 | 0.869±0.059 / inf / 0.70 |
| predicted-RUL bins rul4 | 0.882±0.062 / inf / 0.72 | 0.893±0.059 / inf / 0.71 | 0.893±0.060 / inf / 0.73 | 0.867±0.053 / inf / 0.68 |
| predicted-RUL bins rul5 | 0.886±0.063 / inf / 0.72 | 0.881±0.064 / inf / 0.72 | 0.890±0.069 / inf / 0.73 | 0.849±0.075 / inf / 0.66 |

## Coverage / width on all test windows, alpha = 0.1

PICP | MPIW raw | MPIW clipped to [0,1] — mean ± std over 5 seeds

| variant | baseline | +GAN | +MT | +GAN+MT |
|---|---|---|---|---|
| single q (no groups) | 0.793±0.069 / 1.22 / 0.68 | 0.804±0.069 / 1.20 / 0.69 | 0.833±0.052 / 1.10 / 0.64 | 0.830±0.050 / 1.01 / 0.59 |
| true stage (oracle) | 0.786±0.052 / 1.01 / 0.55 | 0.792±0.051 / 1.02 / 0.57 | 0.819±0.051 / 0.84 / 0.49 | 0.811±0.034 / 0.79 / 0.46 |
| predicted stage | 0.657±0.057 / inf / 0.62 | 0.673±0.055 / inf / 0.64 | 0.715±0.055 / inf / 0.60 | 0.734±0.053 / inf / 0.53 |
| predicted stage + ratchet k=5 | 0.728±0.025 / inf / 0.61 | 0.751±0.024 / inf / 0.61 | 0.797±0.022 / inf / 0.60 | 0.786±0.033 / inf / 0.54 |
| predicted stage + ratchet k=10 | 0.725±0.026 / inf / 0.60 | 0.747±0.025 / inf / 0.60 | 0.794±0.020 / inf / 0.60 | 0.783±0.031 / inf / 0.53 |
| predicted stage + ratchet k=20 | 0.732±0.044 / inf / 0.64 | 0.749±0.039 / inf / 0.63 | 0.792±0.024 / inf / 0.65 | 0.778±0.032 / inf / 0.56 |
| predicted stage + ratchet k=30 | 0.727±0.042 / inf / 0.63 | 0.744±0.041 / inf / 0.63 | 0.799±0.029 / inf / 0.65 | 0.774±0.031 / inf / 0.56 |
| predicted-RUL bins rul3 | 0.762±0.078 / inf / 0.59 | 0.775±0.071 / inf / 0.62 | 0.744±0.096 / inf / 0.59 | 0.790±0.102 / inf / 0.56 |
| predicted-RUL bins rul4 | 0.759±0.103 / inf / 0.59 | 0.791±0.063 / inf / 0.62 | 0.745±0.094 / inf / 0.59 | 0.778±0.100 / inf / 0.54 |
| predicted-RUL bins rul5 | 0.789±0.101 / inf / 0.61 | 0.788±0.061 / inf / 0.63 | 0.728±0.104 / inf / 0.57 | 0.750±0.104 / inf / 0.52 |

## Coverage / width on post-FPT windows, alpha = 0.1

PICP | MPIW raw | MPIW clipped to [0,1] — mean ± std over 5 seeds

| variant | baseline | +GAN | +MT | +GAN+MT |
|---|---|---|---|---|
| single q (no groups) | 0.757±0.129 / 1.22 / 0.70 | 0.755±0.134 / 1.20 / 0.71 | 0.766±0.116 / 1.10 / 0.67 | 0.724±0.099 / 1.01 / 0.63 |
| true stage (oracle) | 0.812±0.031 / 1.07 / 0.64 | 0.819±0.028 / 1.05 / 0.65 | 0.838±0.016 / 0.99 / 0.64 | 0.839±0.020 / 0.96 / 0.64 |
| predicted stage | 0.733±0.095 / inf / 0.66 | 0.743±0.080 / inf / 0.67 | 0.746±0.090 / inf / 0.67 | 0.723±0.060 / inf / 0.65 |
| predicted stage + ratchet k=5 | 0.713±0.071 / inf / 0.63 | 0.717±0.068 / inf / 0.63 | 0.743±0.094 / inf / 0.64 | 0.703±0.073 / inf / 0.62 |
| predicted stage + ratchet k=10 | 0.710±0.073 / inf / 0.63 | 0.707±0.074 / inf / 0.63 | 0.747±0.090 / inf / 0.64 | 0.706±0.073 / inf / 0.62 |
| predicted stage + ratchet k=20 | 0.725±0.079 / inf / 0.63 | 0.716±0.083 / inf / 0.64 | 0.739±0.104 / inf / 0.64 | 0.708±0.092 / inf / 0.62 |
| predicted stage + ratchet k=30 | 0.714±0.087 / inf / 0.62 | 0.705±0.097 / inf / 0.63 | 0.729±0.122 / inf / 0.63 | 0.700±0.101 / inf / 0.61 |
| predicted-RUL bins rul3 | 0.730±0.125 / inf / 0.66 | 0.759±0.111 / inf / 0.69 | 0.722±0.115 / inf / 0.68 | 0.702±0.113 / inf / 0.65 |
| predicted-RUL bins rul4 | 0.725±0.136 / inf / 0.66 | 0.763±0.107 / inf / 0.69 | 0.721±0.115 / inf / 0.68 | 0.693±0.104 / inf / 0.64 |
| predicted-RUL bins rul5 | 0.746±0.133 / inf / 0.68 | 0.753±0.110 / inf / 0.69 | 0.702±0.127 / inf / 0.67 | 0.681±0.112 / inf / 0.63 |
