## Date

10-19-2026 (pilot, freeze and gated run the same day)

## Takeaway

**No. On these toys, Varun's measured α (the combined score c, which is Eq. 2 applied to the average of the compression and probe-embedding scores) does not recover the true alignment within complexity levels.**

- **DFA (Experiment 11 mixes):** α̂_c is nearly flat. Averaged over targets it moves from 0.699 at α = 0.05 to 0.700 at α = 1. The within-level Spearman ρ is 0.39, 0.44, 0.28 and −0.16 for n = 4, 8, 16, 32, against the bar of 0.90 with a lower bound of 0.80.
- **DSL (Experiment 12 mixes):** α̂_c ranks mixes moderately but stays below the bar: ρ 0.87, 0.85, 0.82, 0.82, 0.82, with lower bounds 0.78 to 0.83. The embedding half carries no information here. Target and distractors share their inputs, and the probe-embedding score ignores labels. On DSL every reference input is in the pool, so that score is constant (Deviation 2) and α̂_c is half of the compression score's α̂.
- **Gate 2 passes only because the measure is insensitive.** A score that hardly moves with α cannot move with K either. The slope per doubling of K is −0.0015 [−0.0031, −0.0002] on DFA and −0.0009 [−0.0087, +0.0072] on DSL. Both sit inside the ±0.02 band and match the oracle's slopes (−0.0015 and −0.0021).
- **Gate 3 passes but says little.** With calibrated α̂_c in place of the true α, the best Experiment 11/12 law's held-out RMSE rises by 7.5% [6.5, 8.6] on DFA and falls by 6.9% [6.7, 7.1] on DSL. Uncalibrated α̂ raises it by 5.8% and 2.1%. On DFA, a random α̂ also stays inside the 10% bar (+4.6%), because Experiment 11's large targets never learned and the law depends little on α there. On DSL the random score fails (+16.8%), so the DSL pass is informative.
- **Which scores do track α.**
  - The frozen-gradient score d is the best non-leaking score on DFA: ρ 0.81 to 0.91 (lower bounds 0.63 to 0.84). Permuting its labels cuts its lowest ρ to 0.01, so it reads the input-to-label rule. It still misses the Gate 1 bar.
  - On DSL, b2 (probe loss) clears the bar at every level: ρ 0.92 to 0.98, lower bounds 0.91 to 0.98. c2 is close (lowest ρ 0.86). Both read labels, and both fail on DFA (lowest ρ 0.24).
  - Task2Vec e is noisy: lowest ρ 0.58 on DFA and 0.49 on DSL.
  - When the task token is left in, every score improves sharply (c_leak, lowest ρ 0.78 on both families). This shows how much identity cues can flatter a measure.
- **Graded arm (160 new runs on DFA).** Graded relevance predicts loss better than target share: chance-corrected α_soft cuts held-out RMSE by 11.2% [10.1, 12.3] against α_share, and α_agree by 9.6%. α̂_c does not follow it: ρ with α_soft is −0.03 and with α_share 0.12, a difference of −0.15 [−0.93, +0.25]. The gradient score d does follow it (ρ_soft 0.78 vs ρ_share 0.05, difference 0.73 [0.24, 0.90]), and so do b2 and c2.

**Read with the earlier results.**

- Experiment 11's T* is censored by the 2^16 budget, and its n = 16 and n = 32 targets stayed near chance. That is why the Gate 3 comparison on DFA barely separates good α̂ from random α̂.
- Experiment 13 came out the reverse of C2 (E = 0.80). Experiment 14 found that Task2Vec does not track K. Here the Task2Vec-based α̂ (e) is also the noisiest estimator of α.
- Together these say that neither complexity nor alignment, as measured by the proposed proxies, can yet be read off a fitted law with confidence on these toys.
- The measured α is mostly a share-of-similar-inputs score. Where target and distractors share inputs, as they do here, it does not see the alignment that matters.

## Section 6 row

**The closest row is "G1 NOT_ESTABLISHED for (c), with better numbers for (d)".** Similarity and compression scores miss conditional (input-to-label) alignment, and the optimizer-aware gradient score catches more of it. But (d) does not reach Gate 1 numbers: its lowest ρ is 0.81 on DFA and 0.72 on DSL. So the row holds only in direction.

Three more rows apply:

- **Permuted labels (DFA c, b1).** On DFA, permuted-label and input-only references recover α no worse than true references do: lowest ρ about −0.1 to −0.2 for all of them. The combined measure ignores labels. On DSL, permuting labels does hurt c (lowest ρ 0.82 → 0.28), so there its compression half reads labels.
- **Leak control.** With the task token left in, scores reach ρ 0.73 to 0.97. Stripped of it, c on DFA falls to a mean ρ of 0.24 (random baseline 0.02). Identity cues alone can make α̂ look good.
- **G4a SUPPORTED.** Partially relevant data acts as partial alignment, so "share of target data" understates alignment.

## Gate 2 consequence

Gate 2 is SUPPORTED, so the protocol does not require this paragraph. The pass is vacuous, though. On DFA, a target's α̂_c (averaged over probes) changes by at most 0.005 across the whole α range from 0.05 to 1; on DSL by at most 0.08 (median 0.055), against a true range of 0.95. A score that barely responds to α cannot show a complexity slope either. This run therefore shows no evidence that the measured α is a hidden complexity measure. It also gives no assurance that a measure which does track α would stay neutral to complexity. Among the label-reading scores, b2 on DFA has a slope of −0.024 [−0.034, −0.013] and c2 on DFA one of −0.012 [−0.018, −0.006], while the oracle's slope stays inside the band. That is the pattern Section 6 warns about. These are not gated measures.

## Deviations

1. **No PR and no archive move.** The work was committed on the existing branch results/cpu-expts-11-15 and merged onto main, instead of opening a new branch and a pull request. Experiment 15 stays at experiments/15_alignment_alpha_recovery/. Both were written into PROTOCOL Section 2.8 before the freeze.
2. **Scores constant over a pool (found after the freeze, while analyzing).**
   - What happened: the DSL label-blind embedding score b1, and its permuted-label and input-only controls, are exactly 1.0 for every unit, because every DSL pool input appears among the references. Min-max normalization of a constant is undefined, and analyze.py stopped on it.
   - The fix: following scikit-learn's MinMaxScaler, a score constant over the pool now normalizes to 0. This is `minmax_or_zero` in scores.py, used in analyze.py and in the Eq. 2 check, and every case is listed in gates.json under constant_scores.
   - Consequences: DSL α̂_b1 is 0 and its ρ is undefined, and DSL α̂_c equals half of α̂_a. Any constant would give the same Gates 1, 2, 3 (calibrated) and 4b, because those are invariant to an affine shift and to constant terms. Only raw α̂ levels and Gate 3 raw depend on the choice.
   - The DSL rows of the Eq. 2 check had come out NaN for the same reason, and were recomputed with the same rule.
   - No gate, threshold, seed or estimator changed.
3. **Bootstrap intervals with no finite draw.** percentile_ci returned an index error when every draw was NaN, which only happens for the constant DSL b1 scores. It now returns [nan, nan].
4. **Run accounting for chunk 1.** The first gated chunk ran into the 60-minute command limit. Four scoring jobs had started with 149 s of budget left, and the command was stopped before run_state.json was written. The chunk-1 entry was added by hand from the command's start and stop times (3,600 s). After that, run.py starts a job only while at least 900 s of the chunk budget remain. Every killed job reran from scratch, and the outputs are atomic, so no result depends on the kill. The "started" field of the later entries records when run.py wrote them, which is the end of each chunk.
5. **verify.py tolerance for gradient cosines.** The protocol asks for a maximum relative difference of 1e-6 on the 10% recompute sample of b1, d and e.
   - For d and d_perm (cosines of float32 per-example gradients), the recomputed values differ by at most 3.2e-8 in absolute terms. Batching differs, which changes the float32 summation order.
   - Some cosines lie near 0 (about 1e-4), and there the plain relative difference reaches 7e-5.
   - verify.py therefore rounds the recompute to float32 (the precision the scores are saved in) and measures the difference relative to the target's pool range, the scale that min-max normalization divides by, with the same 1e-6 bar. It also prints the absolute and plain relative values.
   - verify.py's independent-refit check also counted a Spearman ρ that is undefined in both the main and the independent computation (the constant DSL b1) as a mismatch, because NaN never equals NaN. It now counts it as agreement.
   - Both check fixes were made after the gate numbers had been seen. Neither changes a score, an estimator or a gate.

## Scope

These results hold for the Experiment 11 DFA and Experiment 12 DSL toys, the four 0.1M-parameter probes and Eq. 2 as implemented here. They do not show whether the paper's α is right or wrong on its real-text grid. Real examples are hundreds to thousands of tokens long, the dictionary is 128 KB rather than 2–4 KB, and real targets and pools differ mostly in their inputs, which is exactly where similarity and compression scores are strongest. Isotonic calibration uses the true α on its fit cells, which real data never provides, so the raw α̂ numbers are the realistic ones. The graded arm covers DFA only, at two K levels (n = 8 and 16).
