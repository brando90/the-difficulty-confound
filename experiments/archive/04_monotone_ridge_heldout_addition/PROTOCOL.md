# Fresh held-out addition with constrained predictive laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/04_monotone_ridge_heldout_addition/PROTOCOL.md>

**TLDR:** Test a fixed regularization and coefficient-constraint repair of the previous extrapolation failure on fresh models, questions and exposure mixtures. Improving extrapolation and proving that starting difficulty adds predictive value are separate decisions; every outcome will be published.

Frozen on 10-16-2026 before any new training. Experiment 03 is prior development evidence, including its failed law. No experiment-04 outcomes select the laws, penalty, bounds, checkpoints or thresholds.

## Fixed skill and fresh data

Reuse the byte-identical addition reference, six one-hot input digits, four output digits, two-hidden-layer network, carry definition, optimizer, learning rate, batch size, 500 balanced warmup updates and 300/1000/3000 intervention checkpoints from experiment 03. The target remains exact addition of integers 000–999. Use all 117 previously eligible sums, no-carry and two-carry strata with totals below 1000; exactly 40 training pairs per sum, 4,680 per intervention.

New seeds: 3101/3202/3303/3404/3505/3606/3707/3808. Fit widths 32/64/128, held-out widths 96/384; width 384 is outside all previously tested widths. Fit mixtures .1/.5/.9, held-out mixtures .2/.8, both new. The experiment has 200 trajectories and 40 shared warm starts. Initialization and warm state are paired across mixtures within seed/width.

New split seed 20261020. Reserve eight test, four calibration and four probe pairs per sum/stratum, excluding EVERY experiment-03 test/calibration/probe item from these new evaluation splits. Train from all remaining pairs, which may include old evaluation items: these are fresh networks and no new evaluation item is trained on. Save old split file hash and exact new item/training IDs. Swapped pairs cannot cross splits. No-carry/two-carry answer histograms match exactly. Input-digit marginals remain unmatched. Primary endpoint warmup-plus-intervention unions match; intermediate unions differ as in experiment 03.

## Frozen predictive laws

The response is log mean answer loss in bits, fitted separately within each seed. All laws include intercept, log parameter count, log intervention updates, a two-carry indicator and log training-stratum fraction. Difficulty laws add log independently probed starting loss after warmup. Five laws:

1. `legacy_base`: ordinary least squares, no starting difficulty, reproducing the prior baseline form.
2. `legacy_difficulty`: ordinary least squares plus starting difficulty, reproducing the failed form.
3. `ridge_difficulty`: starting difficulty plus ridge regularization, without coefficient constraints, isolating shrinkage from the shape constraints.
4. `stable_base`: ridge plus nonpositive coefficients on log parameters, updates and exposure; carry coefficient and intercept unrestricted.
5. `stable_difficulty`: the same stable base plus starting-difficulty elasticity constrained to [0,1].

For regularized fits, center and scale every non-intercept feature using ONLY the 54 allowed calibration fit rows per seed. Minimize mean squared log error plus 0.01 times squared standardized slopes; do not penalize the intercept. The fixed penalty and unit elasticity cap are project-chosen inductive assumptions, not literature constants or empirically selected settings. Translate the difficulty bound to standardized coordinates. Use bounded least squares; independently verify its optimum by enumerating feasible active constraints. Exponentiate forecasts without clipping, fallback routing or removing seeds.

These restrictions directly prevent lower starting loss from raising a forecast through an inverse difficulty power; increasing size, updates or exposure cannot raise the forecast when the other features are fixed. They do not guarantee accurate predictions or imply universal monotonic learning. Bounds, shrinkage and a different fresh population can affect accuracy. Publish both failure controls and the shrinkage-only ablation.

## Held-out evaluation and decisions

Fit only widths 32/64/128 AND mixtures .1/.5/.9, all checkpoints and strata: 54 calibration rows/seed. Freeze coefficients and all 3,840 forecasts before reading final-test outcomes. Independent probes are measured before intervention. Score joint, model-only and mixture-only splits and each held-out width. Joint evaluation contains 192 targets/law across eight seeds, two widths, two mixtures, three checkpoints and two carry strata.

**Primary feature-value criterion:** `stable_difficulty` must lower joint root mean squared error by at least 10% relative to `stable_base`, have a positive 95% interval lower bound, and improve at both held-out widths. **Separate repair criterion:** the identical gate comparing `stable_difficulty` with `legacy_difficulty`. Ten percent is the frozen project practical threshold. Neither decision can substitute for the other. Report all five laws and all six comparisons: stable difficulty versus stable base, legacy difficulty, and ridge difficulty; ridge difficulty versus legacy difficulty; legacy difficulty versus legacy base; stable base versus legacy base.

Use 20,000 paired bootstrap resamples of eight whole seed blocks, analysis seed 20102026; intervals condition on the fixed test set, p-val=n/a. Report root mean squared error with intervals, mean absolute error, signed bias (forecast minus observed), both means, worst forecast, invalid counts, seed variability and every failure. Any invalid forecast disqualifies its law from a success claim; never silently omit targets. Treat the eight seeds, not correlated targets, as replicates. The experiment-03 rank-reversal contrast at 3000 updates is a secondary replication, retaining its original 10-point conjunction.

## Verification, budget and publication

Before training: exhaustive 500,500-pair oracle check; source identity; exact histogram matching; disjoint new splits and zero overlap with prior evaluation; coefficient-bound/monotonic stress checks on synthetic inputs without training outcomes. Afterward: reconcile 2,400 aggregates from 1,684,800 item rows, replay all 240 weights, reconstruct all forecasts and bootstrap comparisons independently, verify fit masks and source hashes. Keep every cell, including nonfinite or failed ones, and report missing results explicitly.

Run on skampere2 with two central processing unit threads, no graphics processor or model-provider calls; estimated 25–40 minutes, training cap 90 minutes, artifacts below 2 GiB. No scientific retries, outcome-driven tuning or extra training. A persistent ordinary-code supervisor detects child exit and timeout. Publish through a verified pull request, synchronize the canonical Mac, verify transferred hashes, remove the owned temporary checkout, and send the required run receipts. This is supervised addition; it is not a reinforcement-learning experiment or a validation of Varun's original mathematics law.
