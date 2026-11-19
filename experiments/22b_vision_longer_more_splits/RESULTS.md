# Results: Experiment 22b, vision shared-structure rerun (300 epochs, ten class splits)

Frozen design: PROTOCOL.md at commit e73471cc12ad080521bc368f644e82d8e7caa74b (10-21-2026). Primary run on FarmShare L40S, the frozen route (array 1863627, post job 1863628, both exit 0); results in `expt_v1/results_farmshare/`; tarball sha256 16a4cb68c98362124bf8bcac0905c5ad99155d94598ce9c5b6f866e674afcb95. No deviation from the protocol (Section 9: none).

## Takeaway

Vision ρ does **not** reach the 0.6 bar: P2 NOT_SUPPORTED, ρ = 0.488, split-level 95% interval [0.377, 0.595] (the interval excludes 0.6 narrowly, upper end 0.595). The 3× longer budget and ten splits tightened the estimate (Experiment 22: 0.498 [0.307, 0.664] at split level) but did not move it. P1 SUPPORTED (shuffled-map transfer −0.002 [−0.015, 0.010]), P3 SUPPORTED (R² structure − R² gzip = 0.214 [0.141, 0.294]). The budget check passes (2.5% of real-map runs end on their best epoch at epoch 300, limit 5%). An independent replicate of the same grid on mercury1 A4000s gives the same verdicts with rho 0.533 [0.409, 0.641]: across both runs the answer to "does rho reach 0.6" is no, with rho between 0.49 and 0.53.

## Numbers (verbatim from report.py)

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

## Independent replicate and comparison with Experiment 22 (descriptive; compare_runs.py, compare_runs.txt)

The same frozen grid (same code, config and seeds) was also run earlier on mercury1 RTX A4000s (results_mercury1_full.md, `expt_v1/results/`, not verified, PR #27 note): P1 SUPPORTED, P2 NOT_SUPPORTED with rho 0.533 [0.409, 0.641], P3 SUPPORTED (0.259 [0.181, 0.350]). Both runs reach the same verdicts and neither reaches 0.6 by the point-estimate rule.

- Unit transfers of the two 22b runs agree: Spearman 0.981 over the 200 real units, mean shift 0.006 [-0.003, 0.016].
- The gap in rho (0.488 vs 0.533) comes from the frozen feature extractor, which each run trained on its own hardware: margin Spearman across the 200 real units between the two runs is 0.970 (mean 0.659 vs 0.648), while the pixel-margin is identical (1.000). So the 0.045 gap is the run-to-run (hardware/feature-extractor) noise of the structure measure, comparable to the width of the split-level interval; rho in the 0.49-0.53 range, with 0.6 outside or at the edge of both intervals.
- Against Experiment 22 on the same 100 real units (splits 0-4, 100 epochs): FarmShare Spearman 0.978, mean shift 0.0001 [-0.011, 0.009]; mercury Spearman 0.980, mean shift -0.009 [-0.013, -0.006]. The longer budget does not change the ranking or level of transfer.
- Seed-to-seed Spearman of unit transfers is 0.961, so seed noise is not what holds rho below 0.6.

## Verification

`verify.py --stage remote` PASS on FarmShare (post job 1863628, MANIFEST.sha256 written, one replay from scratch within 0.03). `python verify.py --stage local --results results_farmshare` (here, CPU, sklearn 1.9.1, see verify_local_farmshare.log): manifest PASS (89 files), analysis recomputed within 1e-9 PASS, RESULTS.md matches analysis.json PASS, structure recomputed FAIL at the frozen tolerance 1e-4: margin, nc_share, pixel_margin and gzip_bits agree to 3e-16, and only the proxy A-distance (`pad`) of 2 of 400 units differs, by at most 0.004 (Spearman of pad between the two computations 0.99999986). `pad` is a secondary, thresholdless measure (logistic-regression solver, the same CPU-instruction-set dependence disclosed in Experiment 22); P1-P3 use margin and gzip bits only, which recompute exactly. The tolerance in verify.py was not changed.

## Deviations

None from the frozen protocol for the FarmShare run. The earlier mercury1 A4000 run (Section 9 of PROTOCOL.md) is reported as a replicate, not as the primary result. The `compare_runs.py` comparison is descriptive and was written after the results.
