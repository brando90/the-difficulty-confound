# Answer-matched alignment results

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/archive/03_answer_matched_alignment/results.md>

Status: DONE — 200/200 trajectories and 40 warm starts complete; deterministic verification passed.

**TLDR:** Exposure reverses the no-carry versus two-carry ranking at every width, passing the frozen criterion. The independent starting-difficulty prediction-law test fails badly. [Full results and uncertainty](RESULTS_matched_alignment.md).

| Frozen comparison | Measured result | Verdict |
|---|---|---|
| No-carry advantage with 90% no-carry training | 25.30 [23.14, 27.26] percentage points | Pass |
| Two-carry advantage with 90% two-carry training | 24.58 [22.88, 26.19] percentage points | Pass |
| Added starting difficulty, joint held-out forecast error | 0.4727 → 292.7941 bits/answer; relative reduction -61835.64% [-140226.76, 27.71] | Not established |

95% paired bootstrap intervals over eight seed blocks; one fixed test set, 936 questions per carry stratum; prediction comparison has 192 targets per law. p-val=n/a. Both primary advantage means had to be at least 10 points, have positive lower bounds, and remain positive at every width. Prediction gate required 10% error reduction with a positive lower bound and improvement at both held-out widths.

Training took 23.93 minutes on skampere2 with two central processing unit threads. Source `fd284d4`; protocol `34dc473`. All 200 final weights and 40 warm weights replayed exactly; 1,684,800 per-item rows reconciled; all 3,840 forecasts independently reconstructed. The post hoc failure diagnostic retains every seed and leaves both gates unchanged.

START notification: `1a14670035f2ac4f`, verified all-four blind carbon copy including sender and To. Merged [pull request 2](https://github.com/brando90/the-difficulty-confound/pull/2) at `1a3e778`. Canonical Mac verified all 1,524 artifact hashes; the owned temporary cluster clone, launch files and branch are removed. DONE message `1a1468e1104f48d8` and team update `1a1468e6dc69ff7b` are verified SENT, each explicitly blind-copying all four required addresses. [Publication and cleanup receipt](PUBLICATION.md).

**TLDR-end:** [the-difficulty-confound: experiments/archive/03_answer_matched_alignment] Answer matching preserves a clear exposure-induced ranking reversal; positive predictive-law transfer is not supported.
