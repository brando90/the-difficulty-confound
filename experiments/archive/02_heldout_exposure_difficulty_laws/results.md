# Results: held-out exposure and difficulty laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/02_heldout_exposure_difficulty_laws/results.md>

**TLDR:** Complete and verified. Combined exposure/difficulty covariates did not meet the predeclared held-out improvement criterion; the predeclared difficulty-only ablation is promising. The actual alignment-law breadth proxy did not improve prediction.

Updated: 10-16-2026. Execution status: COMPLETE. Scientific verdict: NOT_ESTABLISHED. Frozen source: `337149b051256420e82f746712cf00a79316b682`.

| Measure | Final value |
|---|---:|
| Intervention trajectories | 200 / 200 |
| Warm starts / independent seed blocks | 40 / 8 |
| Training time on skampere2 | 25.55 minutes |
| Joint held-out aggregate targets per law | 384 |
| Primary joint error reduction | 6.80% [-3.52, 14.80] |
| Secondary difficulty-only error reduction | 15.80% [10.27, 20.52] |
| Actual-data breadth-proxy error reduction | -1.14% [-9.58, 0.57]; 24 observations |
| Exhaustive addition oracle | 500,500 / 500,500 passed |
| Train/probe/calibration/test overlap | 0 |
| Per-item predictions / aggregate outcomes checked | 3,000,000 / 4,800 |
| Law forecasts checked | 10,752 |
| Warm/final weights replayed | 240 / 240; maximum bits difference 0 |
| Known-law recovery / child-death detection / primary bootstrap | passed |

Controlled intervals are 95% paired seed bootstrap intervals over eight blocks, 20,000 draws; original-data intervals are descriptive over four size blocks with overlapping fit folds. p-val=n/a. The practical primary threshold is 10%, chosen prospectively, with a positive lower bound and improvement at both held widths. Negative-loss forecasts remain: 52/384 for difficulty-only and 64/384 for the combined law.

[Controlled report, absolute errors, bias, all ablations and limits](RESULTS_heldout_laws.md) · [Actual-data companion with authorized observations and predictions](RESULTS_actual_math.md) · [Complete verification](expt_v1/results/verification.json).

Brando authorized publication of the companion summary, used observations, and predictions on 10-16-2026. All fitting and forecasting code and configurations remain unchanged from the pre-outcome freeze; reporting and verification were added afterward.

Published in [pull request #1](https://github.com/brando90/the-difficulty-confound/pull/1), merged commit `330f3555312b0780327c0e77cd60599c6e40b525`. All 1,520 original files were preserved and checked before removing the completed remote checkout; [cleanup receipt](expt_v1/results/cleanup_receipt.json).

Expanded publication authorization on 10-16-2026 covers all project research artifacts without further confirmation. [Complete original inputs](expt_v1/source_data/) and [original companion outputs and transfer receipts](expt_v1/results/private_math/) are now public; the 27 source-grid observations outside the 24-row fixed-budget analysis remain unused, and all reported results are unchanged.
