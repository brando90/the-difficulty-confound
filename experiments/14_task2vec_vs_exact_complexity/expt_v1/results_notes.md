## Takeaway

On these small probes, the Task2Vec norm does not track exact complexity K. The DFA result even goes the wrong way. In the DFA family the norm falls as K rises (Spearman −0.36, 95% interval [−0.56, −0.16]). Once label entropy and sensitivity are taken out, nothing is left (partial 0.02 [−0.21, 0.21]). In the DSL family the link is weak and positive (0.19 [0.01, 0.36]), and it disappears once the same covariates are removed (−0.07 [−0.27, 0.12]).

What the norm does follow is how well the probe's head already fits a task. Across tasks, the Spearman correlation of the norm score with the fitted head's loss is −0.96 in both families. It also follows how quickly a task is learned, but with the opposite sign to the prediction: a larger norm goes with a smaller T\*. The correlation with T\* is −0.44 [−0.70, 0.07] for DFA (13 of 16 targets never reached ε\* and are treated as +∞) and −0.56 [−0.86, −0.06] for DSL.

In the DSL family the norm predicts log T\* far better than K does. In a leave-one-target-out fit on the 15 targets that reached ε\*, the error falls from 0.65 with log K alone to 0.27 with the norm added. So the norm carries information about learnability, but as an "easiness" signal, not as complexity. The pattern holds:
- for all four estimators (E0 through E3);
- for the L2 and log-sum summaries and for either transformer block alone;
- across probe seeds (Kendall's W 0.91 for DFA, 0.95 for DSL).

Label noise lowered the norm in both families rather than raising it, the opposite of Gate 5's prediction (25% noise: DFA ratio 0.59 [0.45, 0.65], DSL 0.52 [0.50, 0.59]; the gate asked for at least 1.2).

**Gate 4 (diversity coefficient) is NOT_ESTABLISHED in both families.** It ran only after the 4-hour cap was lifted, once Gates 1–3 had been seen (Deviation 3). On the coverage mixes, the diversity coefficient of interleaved batches barely follows mixture entropy H: Spearman 0.23 [−0.02, 0.45] for DFA and −0.13 [−0.37, 0.19] for DSL (the gate asked for at least 0.7 with a lower bound above 0.5). It also does not follow H more than it follows K (gap −0.08 [−0.40, 0.28] and −0.23 [−0.50, 0.11]; the gate asked for at least 0.3). Interleaved-batch diversity stays in a narrow band (DFA 0.026–0.037, DSL 0.017–0.021). Reported, not gated: computed from single-task batches instead, diversity does follow H (0.73 DFA, 0.70 DSL) and Rao's Q (0.91, 0.89). So the coefficient registers which tasks a mix holds only when its batches are not mixed.

The rescaling check passes. Rescaling the probe's parameterization leaves the ranking of tasks unchanged (Spearman with the original 0.996 for DFA, 0.999 for DSL), so the handoff fails on Gates 1–3 alone. With a randomly initialized probe, the DFA norm carries no K or T\* signal (−0.05 with K, −0.04 partial, −0.02 with T\*; point estimates only). The DSL norm still follows T\* (−0.72), so in DSL the easiness signal does not need pretraining.

So: does Task2Vec track exact K, and T\*? It does not track K. It tracks T\* only with the opposite sign: a higher norm means faster learning, through how well the head fits, not through complexity.

## Timing pilot and freeze

See PROTOCOL Section 2.8. The pilot ran on a contended machine and projected 12.9 h, so the pre-declared fallback (8 diversity batches instead of 12) was applied before the freeze (commit d22ad34e).

## Deviations (details in CKPT_task2vec_complexity.md)

1. **Wall cap and job order (results not seen).** With most of the 4-hour cap used up because of machine contention, the remaining jobs were reordered so that gate inputs ran first. The order became:
   1. all-estimator embeddings;
   2. label noise;
   3. diversity jobs holding the coverage mixes;
   4. everything else.

   The run stopped at 14,121 s with 363 of 503 jobs not run:
   - 4 DSL label-noise jobs;
   - all 224 diversity jobs;
   - 108 rescaled-probe jobs;
   - 27 random-init-probe jobs.

   The finished jobs were merged, and the 10-19 report marked analyses whose inputs were missing as not run (NOT_RUN meant "not tested", not "failed"). The run can be resumed (`run.py --output results --resume`) if the cap is raised. It was resumed on 10-20 (Deviation 3).
2. **Embedding storage.** The two families' embeddings have different dimensions (DFA 1,564, DSL 1,566), so `embeddings.npz` holds one set of arrays per family (`dfa_keys`, `dfa_f`, …). The keys are as in PROTOCOL Section 8.
3. **Wall cap lifted after results were seen (10-20).** Gates 1–3 in both families and DFA Gate 5 had been seen on 10-19. Gate 4, DSL Gate 5 and the rescaled-probe and random-init checks had not, because their jobs never ran. Gate 4 had been cut by the cap, not tested, so `config.json` `max_wall_seconds` was raised from 14,400 to 57,600 s. That is the only change: no analysis code, seed, threshold, gate or job definition changed. The run resumed with `run.py --output results --resume` on a free machine, and the 363 jobs finished in 5,042 s (cumulative wall 19,163 s, 5.3 h, over the pre-registered 4 h). Gates 1–3 and DFA Gate 5 recomputed identically and stand as reported on 10-19. Gate 4, DSL Gate 5 and the two checks are results from past the pre-registered budget.

## Verification

`verify.py` passes (results/verify.json, verify.log), rerun on 10-20 over the complete results. It reloads the 5 probes per family and re-evaluates them (max difference 0 bits). It recomputes exact K for every task and rebuilds the roster identically with disjoint roles. It recomputes 3,634 saved embeddings with max relative difference 0: a 10% random sample of task embeddings plus every Gate-input embedding (1,959), and a 10% random sample of diversity batch embeddings plus every interleaved coverage-mix batch behind Gate 4 (1,675). It recomputes 245,776 statistics, bootstrap draws and table entries with the independently written independent_stats.py (max absolute difference 8.3e-16). One verify.py bookkeeping fix was needed on 10-20 (a recomputation job holding both families' embeddings could not be stacked; rows are now padded and sliced back), with no change to what is checked (CKPT).

## Limitations

- Gate 4, DSL Gate 5 and the rescaling and random-init checks ran past the pre-registered 4 h cap, after Gates 1–3 had been seen (Deviation 3). Only the cap changed, so no choice made after seeing results enters their numbers, but the run as a whole exceeded its frozen budget.
- T\* is heavily right-censored for DFA (104 of 128 α = 1 runs never reached ε\* = 0.5 bits), so the DFA Gate 3 estimate rests on 3 finite targets plus ties.
- The probes are tiny transformers pretrained on the same family. A large pretrained probe on real data could behave differently. These results speak about this setting only.
- The negative link between the norm and head loss suggests the E0 norm mostly measures how well the frozen features fit the task. The research note's Kolmogorov reading of the norm does not hold here.
