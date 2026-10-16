# Fresh held-out constrained-law results

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/04_monotone_ridge_heldout_addition/results.md>

Status: DONE — 200/200 trajectories and 40 warm starts complete; deterministic verification passed.

**TLDR:** Constrained monotone ridge (`stable_difficulty`) repairs the previous extrapolation failure (**SUPPORTED**, reducing joint held-out RMSE from 1,935,297,769,746.72 to 0.4835 bits/answer with 0 invalid forecasts) and replicates the answer-matched ranking reversal (**SUPPORTED**, 24.07 and 24.48 percentage-point endpoint advantages). However, adding starting difficulty under a shared balanced warmup improves over `stable_base` by only 3.48% [1.12, 6.24], missing the 10% practical threshold (**NOT_ESTABLISHED**). [Full results and uncertainty](RESULTS_stable.md).

| Frozen comparison | Measured result | Verdict |
|---|---|---|
| Extrapolation repair (`legacy_difficulty` → `stable_difficulty`) | 1935297769746.7185 → 0.4835 bits/answer; reduction 100.00% [99.90, 100.00] | Supported |
| Starting-difficulty feature value (`stable_base` → `stable_difficulty`) | 0.5010 → 0.4835 bits/answer; reduction 3.48% [1.12, 6.24] | Not established |
| Secondary ranking-reversal replication (`e_low`, `e_high`) | 24.07 [22.80, 25.46] and 24.48 [22.77, 26.06] percentage points | Supported |

95% paired bootstrap intervals over eight seed blocks (20,000 draws); fixed test set with 936 questions per carry stratum; joint prediction comparison has 192 targets per law. p-val=n/a.

Training took 15.88 minutes on skampere2 with two CPU threads; frozen source `39cd0b78eea437f38b89cbf36928b8e0e9947613`. All 200 final weights and 40 warm weights replayed with 0.0 bits discrepancy; 1,684,800 per-item rows reconciled; all 3,840 forecasts and constrained active-set optima independently reconstructed. Canonical Mac verified all 1,521 artifact SHA-256 hashes. [Full report](RESULTS_stable.md) · [Protocol](PROTOCOL.md) · [Checkpoint](CKPT_stable.md).
