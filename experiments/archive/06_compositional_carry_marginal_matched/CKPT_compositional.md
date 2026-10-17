# Experiment 06 checkpoint
Created: 10-16-2026 17:10 PDT
Last updated: 10-16-2026 17:17 PDT

## Goal
Test compositional carry difficulty reversal (isolated single carries vs cascading two-carry chains) while holding both exact answer histograms and per-column carry rates (50% ones, 50% tens) strictly invariant across mixtures.
## What ran
Frozen source `e76ac4084019845181ce072c92dc06875463063d`; all preflight checks passed; 200/200 trajectories and 40 warm starts completed in 2.12 minutes on skampere2; all 1,507 artifacts transferred and verified on canonical Mac.
## Results
- Pure compositional reversal (`(1,0)+(0,1)` vs `(1,1)`): **SUPPORTED** (`+22.35 [20.60, 24.00]` pp single-carry lead at $q=0.1$; `+17.74 [14.62, 21.12]` pp double-carry lead at $q=0.9$; positive at all five widths).
- Regime reversal ($S_{\text{iso}}$ vs $S_{\text{comp}}$): **SUPPORTED** (`22.39 [20.78, 23.93]` pp and `18.07 [14.78, 21.61]` pp).
- Input-digit marginal total variation distance across six columns reduced from mean `0.443` (max `0.656`) in Expts 03–04 to mean `0.0398` (range `0.0049`–`0.0972`).
- Shared-warmup starting-difficulty feature gain: `1.94% [-3.10, 6.50]` (**NOT_ESTABLISHED**), confirming Experiment 04's finding that starting difficulty requires variation in prior training history (as in Experiment 05's **47.20%** gain) to add >10% predictive value beyond scale and structure.
## Stuck / blocked
None.
## Next steps
Archive under `experiments/archive/06_compositional_carry_marginal_matched/` and update project-wide documentation.
