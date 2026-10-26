# Results: Experiment 23, Experiment 13-small and Experiment 17b to convergence, 8 seeds

Generated 2026-10-21 22:13 UTC by report23.py from /home/users/brando9/e23/out. Protocol: PROTOCOL.md (frozen before any job). Verdicts are the frozen gates; everything else is reported.

## Experiment 13 (e13): complexity versus alignment, 256 runs

Stopped at T_stop = 2^26 tokens (cap). ε* = 0.1 bits/token by the Section 5.1 rule (n = 32, α = 0.4, T = 2^25: median ε_learn 0.072; share reaching each level 0.05: 0.00, 0.1: 1.00, 0.2: 1.00, 0.3: 1.00, 0.5: 1.00, 0.75: 1.00, 1.1: 1.00).

Stopping decisions:

| stage | T | cells converged | decision |
|---|---|---|---|
| 0 | 2^20 | 0/16 | CONTINUE  |
| 1 | 2^22 | 0/16 | CONTINUE  |
| 2 | 2^24 | 1/16 | CONTINUE  |
| 3 | 2^26 | 8/16 | STOP cap |

Least-converged cells at the last decision: n=32|alpha=0.05: 0.400 -> 0.229 bits (threshold 0.020); n=32|alpha=0.15: 0.205 -> 0.131 bits (threshold 0.020); n=16|alpha=0.05: 0.219 -> 0.146 bits (threshold 0.020); n=8|alpha=0.05: 0.162 -> 0.120 bits (threshold 0.020)

| gate | Exp 23 at ε* (gated) | Exp 23 at ε* = 1.10 | Exp 13-small (4 seeds, 2^20, ε* = 1.10) |
|---|---|---|---|
| G1 | SUPPORTED: M_K 35.51 [31.782, 39.541], two-stage [31.029, 40.047] | SUPPORTED: M_K 2.62 [2.566, 2.672], two-stage [2.480, 2.766] | SUPPORTED: M_K 2.61 [2.521, 2.726], two-stage [2.417, 2.853] |
| G2 | SUPPORTED: M_K_coef cuts held-out RMSE 51.7% [50.7, 52.7] | SUPPORTED: M_K_coef cuts held-out RMSE 51.7% [50.7, 52.7] | SUPPORTED: M_K_coef cuts held-out RMSE 52.3% [49.4, 55.0] |
| G3a | NOT_ESTABLISHED: κ 0.789 [0.780, 0.797], γ₁ -0.019 [-0.023, -0.017] | NOT_ESTABLISHED: κ 0.789 [0.780, 0.797], γ₁ -0.019 [-0.023, -0.017] | NOT_ESTABLISHED: κ 0.789 [0.770, 0.808], γ₁ -0.174 [-0.185, -0.164] |
| G3b | NOT_ESTABLISHED: exp vs coef -74.1% | NOT_ESTABLISHED: exp vs coef -74.1% | NOT_ESTABLISHED: exp vs coef -68.1% |
| G4 | NOT_ESTABLISHED: E 0.67 [0.660, 0.675] | NOT_ESTABLISHED: E 0.75 [0.740, 0.767] | NOT_ESTABLISHED: E 0.80 [0.786, 0.830] |
| G5 | SUPPORTED: r_α early 0.80, late 0.42 | SUPPORTED: r_α early 0.80, late 0.42 | SUPPORTED: r_α early 0.62, late 0.37 |

C3 (coefficient or exponent): unresolved. Independent fitter: AGREE.

Mean excess ε over 8 seeds × 2 targets (bits/token, all positions; 95% seed bootstrap), at 2^20 and at T_stop, and after the cooldown branch at T_stop (reported only):

| n | α | ε at 2^20 | ε at T_stop | ε_learn at T_stop | after cooldown |
|---|---|---|---|---|---|
| 4 | 0.05 | 0.247 [0.238, 0.258] | 0.075 [0.071, 0.078] | 0.041 [0.037, 0.044] | 0.044 [0.044, 0.044] |
| 4 | 0.15 | 0.162 [0.151, 0.172] | 0.045 [0.043, 0.048] | 0.024 [0.022, 0.026] | 0.026 [0.026, 0.026] |
| 4 | 0.4 | 0.082 [0.076, 0.090] | 0.021 [0.021, 0.022] | 0.011 [0.010, 0.012] | 0.012 [0.012, 0.013] |
| 4 | 1.0 | 0.044 [0.037, 0.054] | 0.002 [0.002, 0.003] | 0.002 [0.002, 0.003] | 0.000 [0.000, 0.001] |
| 8 | 0.05 | 0.628 [0.602, 0.653] | 0.118 [0.114, 0.122] | 0.085 [0.081, 0.089] | 0.059 [0.059, 0.060] |
| 8 | 0.15 | 0.406 [0.394, 0.419] | 0.070 [0.068, 0.072] | 0.048 [0.046, 0.050] | 0.033 [0.033, 0.034] |
| 8 | 0.4 | 0.291 [0.285, 0.297] | 0.038 [0.037, 0.040] | 0.028 [0.026, 0.030] | 0.016 [0.016, 0.016] |
| 8 | 1.0 | 0.214 [0.209, 0.219] | 0.009 [0.009, 0.010] | 0.009 [0.009, 0.010] | 0.002 [0.002, 0.002] |
| 16 | 0.05 | 1.024 [0.997, 1.050] | 0.146 [0.142, 0.150] | 0.113 [0.109, 0.117] | 0.070 [0.069, 0.071] |
| 16 | 0.15 | 0.599 [0.589, 0.612] | 0.088 [0.086, 0.090] | 0.066 [0.065, 0.068] | 0.038 [0.038, 0.038] |
| 16 | 0.4 | 0.416 [0.407, 0.424] | 0.047 [0.045, 0.048] | 0.036 [0.035, 0.038] | 0.018 [0.018, 0.018] |
| 16 | 1.0 | 0.285 [0.281, 0.290] | 0.014 [0.013, 0.015] | 0.014 [0.013, 0.015] | 0.003 [0.003, 0.003] |
| 32 | 0.05 | 1.759 [1.729, 1.792] | 0.228 [0.225, 0.232] | 0.195 [0.191, 0.198] | 0.104 [0.103, 0.105] |
| 32 | 0.15 | 1.162 [1.146, 1.177] | 0.130 [0.125, 0.133] | 0.108 [0.104, 0.112] | 0.054 [0.054, 0.055] |
| 32 | 0.4 | 0.864 [0.856, 0.871] | 0.074 [0.072, 0.076] | 0.064 [0.062, 0.065] | 0.026 [0.025, 0.026] |
| 32 | 1.0 | 0.651 [0.644, 0.656] | 0.030 [0.029, 0.031] | 0.030 [0.029, 0.031] | 0.007 [0.007, 0.007] |

## Experiment 17b (e17b): coverage-matched easy pools, 160 runs

Stopped at T_stop = 2^26 tokens (cap). 16 (seed, target) pairs per contrast, verdict needs ≥ 14/16.

Stopping decisions:

| stage | T | cells converged | decision |
|---|---|---|---|
| 0 | 2^20 | 0/12 | CONTINUE  |
| 1 | 2^22 | 0/12 | CONTINUE  |
| 2 | 2^24 | 0/12 | CONTINUE  |
| 3 | 2^26 | 5/12 | STOP cap |

Least-converged cells at the last decision: easy|alpha=0.05|n=32: 0.882 -> 0.379 bits (threshold 0.020); cov32|alpha=0.05|n=32: 0.909 -> 0.408 bits (threshold 0.020); cov16|alpha=0.05|n=32: 0.901 -> 0.549 bits (threshold 0.027); hard|alpha=0.05|n=32: 0.672 -> 0.329 bits (threshold 0.020)

| pool | Q (hard-only still better zero-shot?) | H1′ | H2′ | gap closed (α = 0) | Exp 17b (3 seeds, 2^20) |
|---|---|---|---|---|---|
| cov32 (primary) | NO RELIABLE DIFFERENCE | NOT_ESTABLISHED | NOT_ESTABLISHED | 1.03 [0.976, 1.087] | Q NO RELIABLE DIFFERENCE, H1′ NOT_ESTABLISHED, H2′ NOT_ESTABLISHED, gap closed 1.01 |
| cov16 | YES (hard-only still better) | REFUTED | REFUTED | 0.82 [0.727, 0.896] | Q NO RELIABLE DIFFERENCE, H1′ NOT_ESTABLISHED, H2′ NOT_ESTABLISHED, gap closed 0.99 |

Contrasts at T_stop (positive = the coverage-matched pool wins; bits/token):

| pool | targets | α | Δ mean | 95% CI | pairs > 0 | verdict | Exp 17b Δ (verdict) |
|---|---|---|---|---|---|---|---|
| cov16 | easy | 0.00 | -0.744 | [-1.026, -0.454] | 1/16 | MISMATCHED_POOL_BETTER | -0.051 (NOT_ESTABLISHED) |
| cov16 | easy | 0.05 | -0.049 | [-0.071, -0.028] | 0/16 | MISMATCHED_POOL_BETTER | -0.017 (NOT_ESTABLISHED) |
| cov16 | hard | 0.00 | 0.975 | [0.632, 1.352] | 16/16 | MATCHED_POOL_BETTER | 0.685 (MATCHED_POOL_BETTER) |
| cov16 | hard | 0.05 | 0.215 | [0.119, 0.328] | 16/16 | MATCHED_POOL_BETTER | -0.223 (MISMATCHED_POOL_BETTER) |
| cov32 | easy | 0.00 | 0.115 | [-0.109, 0.314] | 12/16 | NOT_ESTABLISHED | 0.027 (NOT_ESTABLISHED) |
| cov32 | easy | 0.05 | -0.006 | [-0.012, 0.000] | 6/16 | NOT_ESTABLISHED | 0.010 (NOT_ESTABLISHED) |
| cov32 | hard | 0.00 | 0.278 | [0.110, 0.443] | 13/16 | NOT_ESTABLISHED | 0.111 (NOT_ESTABLISHED) |
| cov32 | hard | 0.05 | 0.075 | [0.051, 0.099] | 15/16 | MATCHED_POOL_BETTER | -0.165 (MISMATCHED_POOL_BETTER) |

Training-pool excess at T_stop: mean 0.090, max 0.276 bits/token over 1280 (run, pool concept) pairs.

## Compute

- e13: 256 runs, 4 stages, 569 core-hours of run wall time, median 14684 training tokens/s per run, hosts barley-01, barley-02, barley-04, rye-01, wheat-01, wheat-02 ...; non-finite runs: 0.
- e17b: 160 runs, 4 stages, 465 core-hours of run wall time, median 14178 training tokens/s per run, hosts barley-01, barley-02, barley-04, rye-01, wheat-01, wheat-02 ...; non-finite runs: 0.

## Verification

- e13_complete: PASS (256 runs to T_stop 67108864; missing 0, non-finite 0, wrong checkpoints 0)
- e13_rebuild: PASS (aggregated files rebuilt byte for byte)
- e13_reload_eval: PASS (4 runs x 2 checkpoints re-evaluated, max |diff| 4.44e-16 bits)
- e13_hashes: PASS (prep files 26, mismatches []; sources []; source experiments [])
- e13_analysis_rerun: PASS (analysis rerun identical)
- e13_replication_2^20: PASS (2560 overlapping rows; max |diff| 2.21e-02, median 2.64e-08 bits (identical streams; differences come only from floating point on other CPUs))
- e17b_complete: PASS (160 runs to T_stop 67108864; missing 0, non-finite 0, wrong checkpoints 0)
- e17b_rebuild: PASS (aggregated files rebuilt byte for byte)
- e17b_reload_eval: PASS (4 runs x 2 checkpoints re-evaluated, max |diff| 8.88e-16 bits)
- e17b_hashes: PASS (prep files 61, mismatches []; sources []; source experiments [])
- e17b_analysis_rerun: PASS (analysis rerun identical)
- e17b_replication_2^20: PASS (1920 overlapping rows; max |diff| 6.10e-05, median 4.93e-08 bits (identical streams; differences come only from floating point on other CPUs))
- overall: PASS

## Deviations from PROTOCOL.md

- None recorded by the pipeline; add any here.

## Deviations from the source experiments (declared in PROTOCOL.md)

- Experiment 17b: the hard and easy pools are re-run to T_stop instead of reused; the near pool and the α = 1 target-only runs are not run, so t_equiv is not reported.
- Experiment 13: ε* comes from the Section 5.1 rule on the grid's own fit rows instead of the pilot rule.
