# Completed experiments

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/README.md>

**TLDR:** Completed studies retain every measured outcome, code, and per-item table. Completion describes execution, not whether the scientific hypothesis passed.

| Canonical experiment | Outcome | Previous location |
|---|---|---|
| [experiments/archive/01_fixed_width_addition_exposure/](01_fixed_width_addition_exposure/) | Long-carry accuracy 9.60% → 94.89%; 94.34% of baseline gap recovered, but retention criterion failed | experiments/01_fixed_width_addition_exposure/ |
| [experiments/archive/02_heldout_exposure_difficulty_laws/](02_heldout_exposure_difficulty_laws/) | 200/200 trajectories; combined covariates: 6.80% error reduction, primary criterion not established; difficulty-only secondary: 15.80%. [Full uncertainty and companion test](02_heldout_exposure_difficulty_laws/RESULTS_heldout_laws.md) | experiments/02_heldout_exposure_difficulty_laws/ |
| [experiments/archive/03_answer_matched_alignment/](03_answer_matched_alignment/) | 200/200 trajectories; matched-answer ranking reversal supported, starting-difficulty forecast improvement not established. [Full uncertainty and failure diagnosis](03_answer_matched_alignment/RESULTS_matched_alignment.md) | experiments/03_answer_matched_alignment/ |
| [experiments/archive/04_monotone_ridge_heldout_addition/](04_monotone_ridge_heldout_addition/) | 200/200 trajectories; constrained monotone ridge extrapolation repair supported (100.00% [99.90, 100.00] reduction vs legacy difficulty), starting-difficulty feature value under shared warmup 3.48% [1.12, 6.24] (below 10% gate), ranking reversal replicated. [Full report](04_monotone_ridge_heldout_addition/RESULTS_stable.md) | experiments/04_monotone_ridge_heldout_addition/ |
| [experiments/archive/05_pretraining_history_and_retention/](05_pretraining_history_and_retention/) | 200 pretraining checkpoints + 200 post-training trajectories; starting-difficulty feature value under heterogeneous history supported (**47.20% [42.88, 53.08]** RMSE reduction vs `stable_base`, **36.43%** vs oracle history), targeted remediation recovers **87.55% [83.60, 90.72]** of carry gap with **+1.09 [+0.33, +1.89] pp** no-carry retention (**all gates supported**). [Full report](05_pretraining_history_and_retention/RESULTS_history_retention.md) | experiments/05_pretraining_history_and_retention/ |
| [experiments/archive/06_compositional_carry_marginal_matched/](06_compositional_carry_marginal_matched/) | 200 trajectories + 40 warm starts; exact answer and per-column carry-marginal matched (50% ones, 50% tens; input-digit TV `0.0398`); pure compositional ranking reversal between single carries and cascading two-carry chains supported (**22.35 [20.60, 24.00] pp** and **17.74 [14.62, 21.12] pp**); shared-warmup starting-difficulty gain 1.94% [-3.10, 6.50]. [Full report](06_compositional_carry_marginal_matched/RESULTS_compositional.md) | experiments/06_compositional_carry_marginal_matched/ |
| [experiments/archive/07_prequential_mdl_and_length_transfer/](07_prequential_mdl_and_length_transfer/) | 200 pretraining checkpoints + 200 balanced adaptation trajectories under 50%/50% carry-marginal invariance; starting-difficulty conditional loss RMSE reduction **36.22% [32.45, 40.51]**, prequential online MDL RMSE reduction **56.93% [55.01, 58.67]**, online MDL reversal supported (**0.4423** vs **0.1773** bits/item), gap recovery **104.29%** with **+5.13 pp** single-carry retention (pooled 5-width target accuracy 81.09%, below strict 85% sub-gate). [Full report](07_prequential_mdl_and_length_transfer/RESULTS_mdl_history.md) | experiments/07_prequential_mdl_and_length_transfer/ |
| [experiments/archive/08_rlvr_verifiable_reward_utility_laws/](08_rlvr_verifiable_reward_utility_laws/) | 200 pretraining checkpoints + 200 pure 0/1 verifiable-reward RL trajectories (leave-one-out GRPO, $G=16$); post-RLVR loss RMSE reduction **39.87% [36.20, 50.17]**, RL improvement utility reversal supported (**29.07 [26.32, 32.33] pp** and **32.70 [30.18, 35.20] pp**), RLVR remediation recovers **78.23% [72.58, 84.23]** of gap with **+6.45 [+4.69, +7.91] pp** natural no-harm retention; additive linear RL utility law reduces width-96 RMSE by **48.56%** (`14.59 → 7.50` pp) but overshoots ceiling headroom at width 384 (`-15.78%` pooled). [Full report](08_rlvr_verifiable_reward_utility_laws/RESULTS_rlvr_utility.md) | experiments/08_rlvr_verifiable_reward_utility_laws/ |

Moved on 10-16-2026 after all 32 cells completed. The original protocol and launch records preserve their source-era paths; use the source revision in the report for historical links.

The held-out law study moved on 10-16-2026 after all 200 trajectories and 240 weight replays completed. Its frozen protocol, dispatch, and execution receipts retain source-era paths at revision `337149b051256420e82f746712cf00a79316b682`; publication does not rewrite those records.

The answer-matched study moved on 10-16-2026 after completion and all 240 weight replays. Its frozen protocol and dispatch retain source-era paths; use source revision `fd284d4` for historical records.

The constrained monotone-ridge study moved on 10-16-2026 after completion and all 240 weight replays; its frozen protocol and dispatch retain source-era paths at revision `39cd0b78eea437f38b89cbf36928b8e0e9947613`.

The heterogeneous pretraining-history and no-harm retention study moved on 10-16-2026 after completion and all 400 weight replays; its frozen protocol and dispatch retain source-era paths at revision `13a0bf6a4e0f5d2320b2ae29ee60faf27deef3c4`.

The compositional carry-marginal matched study moved on 10-16-2026 after completion and all 240 weight replays; its frozen protocol and dispatch retain source-era paths at revision `e76ac4084019845181ce072c92dc06875463063d`.

The prequential MDL and carry-marginal matched heterogeneous-history study moved on 10-16-2026 after completion and all 400 weight replays; its frozen protocol and dispatch retain source-era paths at revision `b4044e8097ef3919178a34c63f2a66628902eb45`.

The verifiable-reward RL (RLVR) utility-law study moved on 10-16-2026 after completion and all 400 weight replays; its frozen protocol and dispatch retain source-era paths at revision `95b9c80a5e871b4d0d512c92fb3a788d4f7fae0f`.
