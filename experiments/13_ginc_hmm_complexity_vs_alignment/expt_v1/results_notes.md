## Plain-language takeaway

On small GINC-style HMM mixtures (exact minimal state count n = 4 to 32, alignment α = 0.05 to 1, 128 runs, 4 seeds), **complexity clearly costs data, and here it moves the data bill more than alignment does**:

- **Complexity is real (G1 SUPPORTED).** At α = 1, a 32-state concept needs at least 2.6× the training tokens of a 4-state one to reach the gated excess (seed interval [2.52, 2.73]). It is a lower bound: every 4-state run was already below ε* at the first checkpoint, 2^16 tokens. The fitted law puts the ratio at 17×.
- **Knowing K helps forecasting (G2 SUPPORTED).** Adding K to the best alignment-only law cuts held-out error by 52% [49%, 55%], and also on the unseen n = 16 level (0.33 vs 0.79).
- **Alignment does not dominate (G4 NOT_ESTABLISHED, in the reverse direction).** Per log-unit, complexity moves the data bill more than alignment: E = e_α/e_K = 0.80 [0.79, 0.83] in bits, 0.75 in states, 0.73 model-free, all below 1. The draft range ratio says the same (R = 0.84 [0.83, 0.85]). E falls further for stricter targets (0.24 at ε* = 0.01, from the fitted law).
- **Coefficient or exponent: both (G3 unresolved).** The joint law has a clear constant-multiplier term (κ = 0.79 [0.77, 0.81]) *and* a slower learning rate for complex concepts (γ1 = −0.17 [−0.19, −0.16]). The coefficient-only law forecasts held-out cells far better than the exponent-only law (RMSE 0.51 vs 0.85), and the joint law is best (0.43), so neither single picture passes its gate.
- **Alignment acts early, complexity late (G5 SUPPORTED).** The alignment share of explained variance is 0.62 [0.61, 0.64] in positions 1–16 and 0.37 [0.36, 0.37] in positions 65–128.

## Caveats a reader needs

- **The target excess is loose.** The pilot rule raised ε* from 0.05 to 1.10 bits/token, about 38% of the targets' entropy rates (2.73–2.96 bits). "Trained" here means partly trained: at 2^20 tokens and α = 1 the 32-state targets are still 0.65 bits above the oracle. The multiplier curves at 0.01–0.2 bits rest on the fitted law extrapolating past the data (20–32 of the 32 α = 1 target-seeds never reach those levels).
- **Identification is easy in this construction.** After 16 tokens the oracle posterior on the true target is 1.000 for every target and α, and ε_id over all 128 positions equals log2(1/α)/128 exactly (0.034 bits at α = 0.05), independent of K. So alignment acts here almost entirely through effective data (η = 0.79 [0.78, 0.81] in both M_align_eff and the K law), not through a prior-identification cost. The 13-GPU arm or harder distractors would be needed to test the identification route.
- **Only 4 seeds.** The seed-block bootstrap has 35 distinct resamples, so its intervals understate between-seed uncertainty; the two-stage interval for G1 ([2.42, 2.85]) is the more honest one.
- **No support for the quanta mechanism.** Spearman between each target's γ at α = 1 and its stationary tail exponent is 0.0 over 8 targets.
- **Model-free vs law M_K disagree** (2.6 vs 17.1 at ε*): the model-free value is held down by left-censoring at n = 4; the law value depends on the exponent term extrapolating below the first checkpoint for n = 4.
- **Relation to Experiment 11.** No Exp 13 gate uses an Exp 11 number; only Exp 11's gate definitions were adopted (PROTOCOL Section 5 note). Exp 11's verdicts (all NOT_ESTABLISHED, T* capped) are not inputs here.

## Deviations (also in CKPT_ginc_hmm.md)

1. **Wall cap raised before the freeze**, 14,400 → 18,000 s, because the pilot projected 4.58 h even with the fallback T range. In the event the grid took 12,067 s (3.35 h) cumulative, so it would have fit the original cap.
2. **Grid ran in 4 resume-safe chunks** (one foreground command is limited to 60 minutes). Chunk 1 was killed by the command timeout at 3,500 s with 4 runs in flight; their temporary checkpoint folders were discarded and the 4 runs redone from scratch in chunk 2 (runs are deterministic from seeds; verify.py replays them). Chunk 1's run_state entry was reconstructed by hand from the shell timestamps (3,501 s, 36 runs).
3. **Background CPU load** of about 3 from other jobs on the 4-core box during pilot and grid (lower throughput, no effect on results).
4. **ε_res = 0.015 bits** set at the freeze (this protocol had no value; Exp 11's censored residual needs one). No cell came near it (smallest cell mean 0.042).
5. **report.py** E panel changed after the analysis (linear axis plus the gated ε* point); reporting only, no number changed.
6. **13-GPU not run** (no GPU on this box). Its code path is not implemented in this commit: `run.py --arm gpu` exits with "pending, not run", and the `--pilot-job`, `--pilot-select`, `--merge` and verify modes that launch/README.md describes still have to be written (and the GPU `--check-only` run) before any GPU launch. Diversity coefficient: follow-up (Exp 14 code did not exist at run time).
7. **Not pushed**: committed locally on results/cpu-expts-11-15; the parent task pushes.
