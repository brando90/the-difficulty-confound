Runs: 80 (vis_real 40, vis_shuf 40); structure measures present: True

| phase | mean in-dist test acc | min | share best epoch = last | median train s |
|---|---|---|---|---|
| vis_real | 0.662 | 0.626 | 0.03 | 1482 |
| vis_shuf | 0.517 | 0.456 | 0.05 | 1453 |

Budget check: share of real-map vision runs whose best epoch is the last evaluation = 0.025 (limit 0.05): long enough

## Pre-registered predictions (p = 0; 95% intervals resample whole splits)

| arm | P1 shuffled-map transfer | real-map transfer | P2 Spearman rho (margin) | R² structure | R² complexity | R² diff [CI] | verdicts |
|---|---|---|---|---|---|---|---|
| vision | -0.002 [-0.015, 0.010] (n=200) | 0.491 [0.472, 0.509] (n=200) | 0.488 split-level [0.377, 0.595], unit-level [0.365, 0.595] (n=200, 10 splits) | 0.223 | 0.010 | 0.214 [0.141, 0.294] | P1 SUPPORTED, P2 NOT_SUPPORTED, P3 SUPPORTED |

## Secondary structure measures (real map, p = 0): Spearman rho with transfer

| arm | measure | rho [95% CI] | n |
|---|---|---|---|
| vision | gzip_bits | 0.060 [-0.005, 0.124] | 200 |
| vision | margin | 0.488 [0.376, 0.594] | 200 |
| vision | nc_share | 0.525 [0.446, 0.596] | 200 |
| vision | pad | -0.288 [-0.393, -0.192] | 200 |
| vision | pixel_margin | 0.481 [0.382, 0.573] | 200 |
| vision | margin, real + shuffled units pooled | 0.804 [0.776, 0.829] | 400 |

## Descriptive (no threshold)

- vision rho_splits_0_4: rho 0.484 (n=100)
- vision rho_splits_5_9: rho 0.487 (n=100)
- vision class-level rho (each fine class's two splits averaged): 0.496 [0.320, 0.643] (n=100)
- vision real map seed-to-seed Spearman of unit transfers: 0.961 (mean of 2 seeds: 0.980, n=200)
- vision real per-split mean transfer: 0: 0.455, 1: 0.495, 2: 0.463, 3: 0.498, 4: 0.523, 5: 0.525, 6: 0.480, 7: 0.509, 8: 0.530, 9: 0.435
- vision shuffled map seed-to-seed Spearman of unit transfers: 0.781 (mean of 2 seeds: 0.877, n=200)
- vision shuffled per-split mean transfer: 0: -0.002, 1: -0.030, 2: 0.018, 3: -0.017, 4: -0.037, 5: -0.012, 6: 0.026, 7: 0.009, 8: 0.022, 9: 0.002

## The p dial (share of held-out-class training items mixed in, fixed budget)

| arm | map / p | normalized transfer [95% CI] | held-out acc | in-dist acc |
|---|---|---|---|---|
| vision | real/p=0 | 0.491 [0.471, 0.510] (n=200) | 0.361 | 0.684 |
| vision | real/p=1 | 1.078 [1.050, 1.104] (n=200) | 0.685 | 0.639 |
| vision | shuffled/p=0 | -0.002 [-0.015, 0.010] (n=200) | 0.049 | 0.557 |
| vision | shuffled/p=1 | 1.114 [1.068, 1.159] (n=200) | 0.527 | 0.478 |
