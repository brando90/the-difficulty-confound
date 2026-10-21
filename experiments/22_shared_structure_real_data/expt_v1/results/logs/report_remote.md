Runs: 154 (txt_real 60, txt_shuf 24, vis_real 50, vis_shuf 20); structure measures present: True

| phase | mean in-dist test acc | min | share best epoch = last | median train s |
|---|---|---|---|---|
| txt_real | 0.843 | 0.812 | 0.00 | 50 |
| txt_shuf | 0.761 | 0.702 | 0.00 | 48 |
| vis_real | 0.662 | 0.624 | 0.18 | 297 |
| vis_shuf | 0.483 | 0.415 | 0.05 | 250 |

## Pre-registered predictions (p = 0)

| arm | P1 shuffled-map transfer | real-map transfer | P2 Spearman rho (margin) | R² structure | R² complexity | R² diff [CI] | verdicts |
|---|---|---|---|---|---|---|---|
| text | 0.022 [-0.131, 0.189] (n=16) | 0.620 [0.470, 0.761] (n=16) | 0.800 [0.505, 0.946] (n=16) | 0.701 | 0.120 | 0.581 [0.230, 0.799] | P1 SUPPORTED, P2 SUPPORTED, P3 SUPPORTED |
| vision | -0.014 [-0.032, 0.006] (n=100) | 0.487 [0.420, 0.555] (n=100) | 0.498 [0.324, 0.642] (n=100) | 0.222 | 0.011 | 0.211 [0.101, 0.310] | P1 SUPPORTED, P2 NOT_SUPPORTED, P3 SUPPORTED |

## Secondary structure measures (real map, p = 0): Spearman rho with transfer

| arm | measure | rho [95% CI] | n |
|---|---|---|---|
| text | gzip_bits | 0.341 [-0.253, 0.740] | 16 |
| text | jaccard_top200 | 0.762 [0.317, 0.937] | 16 |
| text | margin | 0.800 [0.505, 0.946] | 16 |
| text | nc_share | 0.897 [0.672, 0.976] | 16 |
| text | pad | -0.597 [-0.889, -0.110] | 16 |
| text | margin, real + shuffled units pooled | 0.923 [0.818, 0.964] | 32 |
| vision | gzip_bits | 0.070 [-0.149, 0.287] | 100 |
| vision | margin | 0.498 [0.320, 0.644] | 100 |
| vision | nc_share | 0.523 [0.364, 0.654] | 100 |
| vision | pad | -0.330 [-0.514, -0.121] | 100 |
| vision | pixel_margin | 0.443 [0.268, 0.589] | 100 |
| vision | margin, real + shuffled units pooled | 0.839 [0.795, 0.871] | 200 |

## The p dial (share of held-out-class training items mixed in, fixed budget)

| arm | map / p | normalized transfer [95% CI] | held-out acc | in-dist acc |
|---|---|---|---|---|
| text | real/p=0 | 0.620 [0.469, 0.762] (n=16) | 0.620 | 0.848 |
| text | real/p=0.01 | 0.634 [0.500, 0.761] (n=16) | 0.629 | 0.848 |
| text | real/p=0.05 | 0.713 [0.585, 0.829] (n=16) | 0.675 | 0.847 |
| text | real/p=0.2 | 0.841 [0.750, 0.926] (n=16) | 0.750 | 0.845 |
| text | real/p=1 | 1.033 [0.962, 1.094] (n=16) | 0.844 | 0.826 |
| text | shuffled/p=0 | 0.022 [-0.135, 0.188] (n=16) | 0.263 | 0.786 |
| text | shuffled/p=1 | 1.037 [0.952, 1.124] (n=16) | 0.753 | 0.736 |
| vision | real/p=0 | 0.487 [0.419, 0.557] (n=100) | 0.353 | 0.673 |
| vision | real/p=0.01 | 0.508 [0.442, 0.575] (n=100) | 0.365 | 0.671 |
| vision | real/p=0.05 | 0.568 [0.502, 0.635] (n=100) | 0.403 | 0.671 |
| vision | real/p=0.2 | 0.741 [0.679, 0.802] (n=100) | 0.505 | 0.664 |
| vision | real/p=1 | 1.075 [1.025, 1.124] (n=100) | 0.675 | 0.631 |
| vision | shuffled/p=0 | -0.014 [-0.032, 0.006] (n=100) | 0.043 | 0.524 |
| vision | shuffled/p=1 | 1.165 [1.088, 1.244] (n=100) | 0.506 | 0.442 |
