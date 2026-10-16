# Frozen protocol: held-out exposure and difficulty laws

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/02_heldout_exposure_difficulty_laws/PROTOCOL.md>

**TLDR:** Compare Varun's additive alignment-law functional form with exposure and starting-difficulty covariates on genuinely withheld network widths, training distributions, and questions. A separate retrospective check uses actual mathematics observations and a breadth proxy. Frozen before new outcomes on 10-16-2026; no adaptive scientific retries.

## Claim and scope

The prospective task is exact addition of two three-digit integers. All conditions use the same notation, answer function, four-digit output, test questions, optimizer, and budgets. Network width is deliberately varied to test prediction across model sizes. This is a transfer test of the **functional form**, not a reproduction of Varun's language-model results or the reinforcement-learning utility law. The latter's local data lack independently measured difficulty paired with post-training outcomes.

The existing alignment implementation is `m_additive` in `madanva/alignment-scaling-laws`, `experiments/00_neurips2026_rebuttal/scripts/fitlib.py`: loss = a + b N^(-beta) + c T^(-gamma) - d alpha. In our controlled task alpha is explicitly replaced by known stratum training fraction q, not represented as the original semantic alignment score.

## Prospective design

- Eight new independent random-seed blocks: 1101, 1202, 1303, 1404, 1505, 1606, 1707, 1808.
- Law fitting widths: 32, 64, 128. Entire held-out widths: 96 (interpolation) and 256 (extrapolation). Two-hidden-layer rectified-linear networks; input 60 one-hot values, output 40 logits.
- Training long-carry fractions: .02, .25, .80. Held-out fractions: .10, .50. The remaining probability is equally divided among maximum carry chains 0, 1, 2. Every training set has 18,000 unique unordered input pairs. All five mixtures are trained for every width and seed: **200 intervention trajectories**.
- Every width/seed starts from a separately trained, shared balanced warm start: 500 updates, 18,000 examples, batch 256. The five interventions copy exactly that state and reset optimizer state identically. They share batch-index random streams. Warm starts use no probe/calibration/test inputs. There are 40 warm starts.
- Fixed learning rate .001, AdamW weight decay .0001, 2 CPU threads. Evaluate after 300, 1,000 and 3,000 intervention updates. No early stopping or selection of a favorable checkpoint.
- Exhaustive input population: 500,500 unordered pairs. Fixed split seed 20261017: per carry stratum, 1,000 final-test questions, 250 law-calibration questions, and 250 starting-difficulty probes. All three splits and all training pools are disjoint, including swapped operands. The 4,000 final questions are identical across all trajectories and never fit any law.
- **D**, starting difficulty, is mean correct-answer negative log probability in bits on the 250 independent probe questions for that stratum, evaluated after the balanced warm start and before intervention. Its inference cost is 1,000 questions for each new starting model. This covariate is available for held-out widths, but no held-out final outcomes enter fitting.
- **E**, additional exposure, is log(q). At a given intervention budget this is the relative expected sampling exposure; it is not a count of latent pretraining examples. The recorded q and actual training counts are known by construction. Exposure interventions also change digit/answer marginals; a unique carry-specific causal mediator is not identified.
- Structural carry-chain length remains recorded, with an explicitly secondary categorical adjustment. Exact Kolmogorov complexity is neither measured nor substituted by D; unchanged inputs and task specifications keep the formal conditional complexity fixed.

## Frozen predictive models and fitting

Use per-stratum mean correct-answer bits as the target. Normalize N by 10,000 parameters and T by 1,000 intervention updates, which only changes amplitudes. Fit separately inside each independent seed block using only calibration outcomes for 3 training widths × 3 training mixtures × 3 budgets × 4 strata = **108 observations**. This gives eight independent forecast replications, rather than pretending thousands of questions are independent training runs.

1. `resources`: a + b N^(-beta) + c T^(-gamma).
2. `additive`: resources - d q, the transferred alignment-law baseline.
3. `exposure`: additive + e log(q).
4. `difficulty`: additive + f D.
5. `full`: additive + e log(q) + f D.
6. `structural`: additive + three carry-category offsets.
7. `full_structural`: full + three carry-category offsets.

For every model, exhaust the same 36 predeclared exponent pairs beta,gamma in {.05,.1,.25,.5,1,2}. At each pair solve bounded linear least squares: b,c,d,f nonnegative, e nonpositive; intercept and category offsets unrestricted. Select the lowest calibration squared error, deterministic grid order breaks ties. No outcome-dependent hyperparameter search, regularization, full-data starting point, or prediction clipping. Report any negative loss forecasts. These bounds and discretized exponents are declared adaptations of the original fitter.

Save immutable coefficients and all test forecasts with a calibration-input hash **before reading any final-test outcome file**. Evaluate separately: model-only holdout (held widths, fitting mixtures), mixture-only holdout (fitting widths, held mixtures), and joint holdout (held widths and held mixtures). Primary joint comparison contains 2 widths × 2 mixtures × 3 budgets × 4 strata = **48 aggregate targets per seed**, 384 total. No fitting rows appear in a scored holdout.

## Decision rule and uncertainty

Primary metric: root mean squared error (RMSE), bits per answer, for `full` versus `additive` on the joint holdout. Also report mean absolute error, mean signed prediction bias, prediction and outcome means, every individual seed, and both held widths separately. A convincing predictive improvement requires at least a **10% reduction in pooled joint RMSE**, a **positive lower bound** on its paired 95% percentile interval, and positive point improvements at both held widths. The 10% threshold is our practical design choice, not an external standard.

Bootstrap 20,000 paired seed-block draws using analysis seed 17102026, recomputing pooled RMSE and relative reduction each time. Eight independently fitted seeds are the resampling units; questions and checkpoints are not independent replicas. Intervals are conditional on this fixed item population, model family, and holdout design. Report p-val=n/a (estimation, no null test). Ablations, width-specific results, and other holdout axes are descriptive secondary comparisons, without multiple-testing claims. A full-model win alone does not prove both covariates help: compare full separately to exposure and difficulty, and report failures/negative results without redesigning this run.

## Companion existing-data test

The source repository is private: retain copied inputs and derived outputs in ignored private storage unless explicit publication permission is obtained. This does not block the public prospective study. Use byte-exact copies of `math_grid.csv` and `E5_decontam_verification.json` from Varun's alignment repository. Restrict to the 24 observations at 21 million tokens and 0.6, 1.7, 4, 8 billion parameters. The task/evaluation corpus remains the original mathematics setup. Compare original additive form a + b N^(-beta) - d alpha (the fixed token term is absorbed into a) with an added standardized log(training-example count), coefficient unrestricted. Counts are independent metadata but are only a **breadth/length proxy**, not observed exposure or starting difficulty. Alpha and breadth may be strongly collinear.

Evaluate leave-one-size-out, leave-one-training-condition-out, and joint size/condition holdouts. For each of the 24 joint cells, exclude its entire size and condition before fitting (15 training cells). Use the same six beta values, cold bounded linear fits, and training-fold-only centering/scaling. Report all predictions and paired RMSE differences. A descriptive paired 95% size-block bootstrap (4 sizes, 20,000 draws) shows sensitivity; overlapping training folds and only four sizes prevent treating it as confirmatory population inference. Original observations are retrospective, usually one seed per cell; no new original-language-model runs or independent evaluation questions are claimed.

## Execution and preservation

Execute pure PyTorch/SciPy numerical computation on skampere2 using two processor threads, no model-provider calls or purchased credits. Estimate 15–30 minutes, hard training wall budget 5,400 seconds, expected files under 1 GiB. Keep all 200 cells and failures; no selective retries. A named tmux supervisor records liveness, failures and timeout. First run exhaustive oracle/split checks and a child-death supervisor test. Publish source/config hashes, manifests, per-item forecasts, final weights, outcomes, coefficients, statistics, and deterministic artifact verification. Preserve before removing the owned remote checkout.

SNAP PRE-APPROVAL (Brando, 2026-09-14): you run with full access (permission prompts and sandbox bypassed). Every command, install, Docker/GPU use, file and git operation (branch, commit, push, PR, merge per Rule 46) and every budgeted experiment call needed for this task is already approved. Never stop to ask for permission or confirmation; treat any approval step as granted, proceed, and log what you did. Pause only for a real blocker (missing credential, exhausted budget, a human-only login step).
