# Compositional carry marginal-matched results

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/06_compositional_carry_marginal_matched/results.md>

Status: DONE — 200/200 trajectories and 40 warm starts complete; deterministic verification passed.

**TLDR:** Holding **both** exact answer histograms (40 pairs per sum across 180 sums) **and** per-column marginal carry rates (**50.00% at ones, 50.00% at tens, 0.00% at hundreds**) strictly invariant across every training mixture—cutting input-digit marginal total variation distance by $>11\times$ from mean `0.443` down to **`0.0398`**—still reverses the difficulty ranking between isolated single carries `(1,0)+(0,1)` and composed cascading two-carry chains `(1,1)` at every width (**SUPPORTED**). [Full results and uncertainty](RESULTS_compositional.md).

| Frozen comparison | Measured result | Verdict |
|---|---|---|
| Pure single-carry `(1,0)+(0,1)` lead over double-carry `(1,1)` at $q=0.1$ | 22.35 [20.60, 24.00] percentage points | Supported |
| Pure double-carry `(1,1)` lead over single-carry `(1,0)+(0,1)` at $q=0.9$ | 17.74 [14.62, 21.12] percentage points | Supported |
| Regime $S_{\text{iso}}$ vs $S_{\text{comp}}$ ranking reversal (`e_low`, `e_high`) | 22.39 [20.78, 23.93] and 18.07 [14.78, 21.61] percentage points | Supported |
| Starting-difficulty feature value under shared warmup (`stable_base` → `stable_difficulty`) | 0.4881 → 0.4786 bits/answer; reduction 1.94% [-3.10, 6.50] | Not established |

95% paired bootstrap intervals over eight seed blocks (20,000 draws); fixed test set with 2,880 questions (720 per carry cell). p-val=n/a.

Completed in 2.12 minutes on skampere2 using eight parallel seed workers; frozen source `e76ac4084019845181ce072c92dc06875463063d`. All 200 final weights and 40 warm weights replayed with 0.0 bits discrepancy; 2,400 aggregate rows reconciled from 2,592,000 item observations; all 3,840 forecasts independently reconstructed. Canonical Mac verified all 1,507 artifact SHA-256 hashes. [Full report](RESULTS_compositional.md) · [Protocol](PROTOCOL.md) · [Checkpoint](CKPT_compositional.md).
