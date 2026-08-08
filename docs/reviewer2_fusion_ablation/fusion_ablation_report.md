# Reviewer 2 Unity Fusion Baseline and Ablation Report

n_frames = 677
3D MPJPE is reported in Unity/avatar coordinate units using the same 15-joint matched evaluation convention as the manuscript. Saved proposed-fusion rows are loaded from the manuscript output files; rows marked recomputed are direct internal baseline/ablation replays.

## Method Summary

| Method | Mean MPJPE | SD | Median | vs best single | vs unweighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Left single-view | 0.794983 | 0.444630 | 0.625640 | 0.000% | -6.361% |
| Right single-view | 0.849810 | 0.496988 | 0.682264 | -6.897% | -13.696% |
| Best single-view (oracle frame) | 0.769241 | 0.441040 | 0.606273 | 3.238% | -2.917% |
| Unweighted average fusion | 0.747441 | 0.486957 | 0.571170 | 5.980% | 0.000% |
| Two-view median fusion | 0.747441 | 0.486957 | 0.571170 | 5.980% | 0.000% |
| Confidence best-view selection | 0.834166 | 0.463537 | 0.671002 | -4.929% | -11.603% |
| 2D confidence only | 0.748022 | 0.487120 | 0.571178 | 5.907% | -0.078% |
| 3D consistency only | 0.747441 | 0.486957 | 0.571170 | 5.980% | 0.000% |
| 2D x 3D multiplicative (recomputed) | 0.747422 | 0.486923 | 0.571170 | 5.983% | 0.003% |
| 2D + 3D additive (recomputed) | 0.747723 | 0.487040 | 0.571174 | 5.945% | -0.038% |
| Proposed confidence-guided fusion (saved raw) | 0.513083 | 0.583041 | 0.288579 | 35.460% | 31.355% |
| Proposed fusion + smoothing (saved) | 0.513038 | 0.582972 | 0.288129 | 35.466% | 31.361% |
| 2D x 3D multiplicative (0.5x sigma, recomputed) | 0.747438 | 0.486958 | 0.571170 | 5.981% | 0.000% |
| 2D x 3D multiplicative (2.0x sigma, recomputed) | 0.746961 | 0.486094 | 0.571139 | 6.041% | 0.064% |

## Paired Frame-Level Statistics

Positive deltas mean the comparison method has higher MPJPE than the reference method. Saved proposed-fusion rows use the existing manuscript output files; rows marked recomputed are direct replay baselines from the same left/right predictions.

| Comparison | Mean delta | 95% CI | p | Holm p | Cohen dz |
| --- | ---: | ---: | ---: | ---: | ---: |
| Left single-view - Proposed confidence-guided fusion (saved raw) | 0.281900 | [0.270537, 0.292964] | 4.99975e-05 | 0.000649968 | 1.893 |
| Right single-view - Proposed confidence-guided fusion (saved raw) | 0.336727 | [0.327242, 0.346026] | 4.99975e-05 | 0.000649968 | 2.733 |
| Best single-view (oracle frame) - Proposed confidence-guided fusion (saved raw) | 0.256158 | [0.244618, 0.267706] | 4.99975e-05 | 0.000649968 | 1.686 |
| Unweighted average fusion - Proposed confidence-guided fusion (saved raw) | 0.234358 | [0.225875, 0.242800] | 4.99975e-05 | 0.000649968 | 2.114 |
| Two-view median fusion - Proposed confidence-guided fusion (saved raw) | 0.234358 | [0.225875, 0.242800] | 4.99975e-05 | 0.000649968 | 2.114 |
| Confidence best-view selection - Proposed confidence-guided fusion (saved raw) | 0.321083 | [0.310716, 0.331575] | 4.99975e-05 | 0.000649968 | 2.335 |
| 2D confidence only - Proposed confidence-guided fusion (saved raw) | 0.234939 | [0.226461, 0.243449] | 4.99975e-05 | 0.000649968 | 2.119 |
| 3D consistency only - Proposed confidence-guided fusion (saved raw) | 0.234358 | [0.225875, 0.242800] | 4.99975e-05 | 0.000649968 | 2.114 |
| 2D x 3D multiplicative (recomputed) - Proposed confidence-guided fusion (saved raw) | 0.234338 | [0.225853, 0.242797] | 4.99975e-05 | 0.000649968 | 2.113 |
| 2D + 3D additive (recomputed) - Proposed confidence-guided fusion (saved raw) | 0.234639 | [0.226177, 0.243124] | 4.99975e-05 | 0.000649968 | 2.117 |
| Proposed fusion + smoothing (saved) - Proposed confidence-guided fusion (saved raw) | -0.000045 | [-0.000146, 0.000056] | 0.367182 | 0.367182 | -0.034 |
| 2D x 3D multiplicative (0.5x sigma, recomputed) - Proposed confidence-guided fusion (saved raw) | 0.234355 | [0.225876, 0.242798] | 4.99975e-05 | 0.000649968 | 2.114 |
| 2D x 3D multiplicative (2.0x sigma, recomputed) - Proposed confidence-guided fusion (saved raw) | 0.233878 | [0.225340, 0.242459] | 4.99975e-05 | 0.000649968 | 2.093 |
