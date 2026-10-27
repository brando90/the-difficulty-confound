# Experiment 22b: all 80 runs plus feature extractor, run on mercury1 A4000s (finished 10-26-2026)

**Status:** DONE, exploratory hardware look; not the frozen FarmShare L40S launch, and not verified. All 80 vision cells ran to epoch 300 with rc=0 and no `.error` file (40 real-map, 40 shuffled-map; 5 real-map cells of shard 0/8 came from the 10-25 run, 75 from this run; the feature extractor is the one from 10-25). `structure.py`, `analyze.py` and `report.py` were run unchanged on the 80 cells. `verify.py` (replay and manifest) was NOT run, so `RESULTS.md` is not written and `MANIFEST.sha256` does not exist. PROTOCOL.md was not edited; the A4000 deviation is in its Section 9.

**Takeaway:** P1 SUPPORTED (shuffled-map transfer -0.004, split-level 95% [-0.017, 0.009], inside +-0.05). P2 NOT_SUPPORTED: Spearman rho(transfer, centroid margin) = 0.533, split-level 95% [0.409, 0.641], so the 0.6 bar is not reached by the point-estimate rule, although the interval contains 0.6. P3 SUPPORTED: R2 structure 0.269 minus R2 gzip 0.010 = 0.259, split-level 95% [0.181, 0.350], above 0. Budget check passes: the share of real-map runs whose best epoch is the last evaluation is 0.025 (limit 0.05). With 300 epochs and ten splits rho moved from Experiment 22's 0.498 to 0.533; it did not reach 0.6. Seed-to-seed Spearman of real-map unit transfers is 0.962, so seed noise does not explain it.

Tables below are the verbatim output of `report.py` on `expt_v1/results`.

Runs: 80 (vis_real 40, vis_shuf 40); structure measures present: True

| phase | mean in-dist test acc | min | share best epoch = last | median train s |
|---|---|---|---|---|
| vis_real | 0.659 | 0.617 | 0.03 | 1597 |
| vis_shuf | 0.517 | 0.446 | 0.03 | 1591 |

Budget check: share of real-map vision runs whose best epoch is the last evaluation = 0.025 (limit 0.05): long enough

## Pre-registered predictions (p = 0; 95% intervals resample whole splits)

| arm | P1 shuffled-map transfer | real-map transfer | P2 Spearman rho (margin) | R² structure | R² complexity | R² diff [CI] | verdicts |
|---|---|---|---|---|---|---|---|
| vision | -0.004 [-0.017, 0.009] (n=200) | 0.485 [0.459, 0.509] (n=200) | 0.533 split-level [0.409, 0.641], unit-level [0.420, 0.630] (n=200, 10 splits) | 0.269 | 0.010 | 0.259 [0.181, 0.350] | P1 SUPPORTED, P2 NOT_SUPPORTED, P3 SUPPORTED |

## Secondary structure measures (real map, p = 0): Spearman rho with transfer

| arm | measure | rho [95% CI] | n |
|---|---|---|---|
| vision | gzip_bits | 0.058 [-0.005, 0.113] | 200 |
| vision | margin | 0.533 [0.410, 0.639] | 200 |
| vision | nc_share | 0.584 [0.502, 0.655] | 200 |
| vision | pad | -0.310 [-0.416, -0.202] | 200 |
| vision | pixel_margin | 0.470 [0.364, 0.559] | 200 |
| vision | margin, real + shuffled units pooled | 0.817 [0.791, 0.840] | 400 |

## Descriptive (no threshold)

- vision rho_splits_0_4: rho 0.549 (n=100)
- vision rho_splits_5_9: rho 0.516 (n=100)
- vision class-level rho (each fine class's two splits averaged): 0.541 [0.378, 0.674] (n=100)
- vision real map seed-to-seed Spearman of unit transfers: 0.962 (mean of 2 seeds: 0.981, n=200)
- vision real per-split mean transfer: 0: 0.460, 1: 0.490, 2: 0.451, 3: 0.485, 4: 0.500, 5: 0.548, 6: 0.466, 7: 0.509, 8: 0.539, 9: 0.401
- vision shuffled map seed-to-seed Spearman of unit transfers: 0.737 (mean of 2 seeds: 0.849, n=200)
- vision shuffled per-split mean transfer: 0: -0.022, 1: -0.020, 2: 0.019, 3: -0.023, 4: -0.036, 5: -0.015, 6: 0.035, 7: 0.013, 8: 0.005, 9: 0.003

## The p dial (share of held-out-class training items mixed in, fixed budget)

| arm | map / p | normalized transfer [95% CI] | held-out acc | in-dist acc |
|---|---|---|---|---|
| vision | real/p=0 | 0.485 [0.458, 0.509] (n=200) | 0.356 | 0.680 |
| vision | real/p=1 | 1.076 [1.047, 1.104] (n=200) | 0.681 | 0.637 |
| vision | shuffled/p=0 | -0.004 [-0.017, 0.009] (n=200) | 0.048 | 0.557 |
| vision | shuffled/p=1 | 1.111 [1.066, 1.157] (n=200) | 0.525 | 0.478 |

## Settings
- Host mercury1.stanford.edu, GPUs 0-6 (NVIDIA RTX A4000 16 GB; the frozen plan is L40S), torch 2.8.0+cu128, bf16 autocast, one cell at a time per GPU, seven shards in parallel: real-map shards 1-7 of 8 first, then shuffled-map shards 0-6 of 7. Same code, config and seeds (23101, 23202) as the frozen protocol.
- Train time per cell 1576-6008 s (sum 38.1 GPU-hours over the 80 cells; the feature extractor and five shard-0 cells came earlier). Seconds are not comparable between cells because GPUs were shared with other jobs at times.
- Raw per-run JSONs: `expt_v1/results/{vis_real,vis_shuf}/`; `structure.json` (400 units) and `analysis.json` are in `expt_v1/results/`. The 44 MB features file (`results/feat/cifar100_features_from_cifar10_vit.npz`) is not committed (see results_shard0of8_mercury1.md for its hashes).

## Caveats
- No replay (`verify.py`) was run; the replay in the frozen plan reproduces the first cell on the FarmShare GPU. A4000 numbers may differ slightly from L40S numbers.
- One generation per cell (one run per spec); intervals resample the ten splits, as in the protocol.
