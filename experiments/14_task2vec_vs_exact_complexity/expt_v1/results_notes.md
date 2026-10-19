## Takeaway

On these small probes, the Task2Vec norm does not track exact complexity K. The DFA result even goes the wrong way. In the DFA family the norm falls as K rises (Spearman −0.36, 95% interval [−0.56, −0.16]). Once label entropy and sensitivity are taken out, nothing is left (partial 0.02 [−0.21, 0.21]). In the DSL family the link is weak and positive (0.19 [0.01, 0.36]), and it disappears once the same covariates are removed (−0.07 [−0.27, 0.12]).

What the norm does follow is how well the probe's head already fits a task. Across tasks, the Spearman correlation of the norm score with the fitted head's loss is −0.96 in both families. It also follows how quickly a task is learned, but with the opposite sign to the prediction: a larger norm goes with a smaller T\*. The correlation with T\* is −0.44 [−0.70, 0.07] for DFA (13 of 16 targets never reached ε\* and are treated as +∞) and −0.56 [−0.86, −0.06] for DSL.

In the DSL family the norm predicts log T\* far better than K does. In a leave-one-target-out fit on the 15 targets that reached ε\*, the error falls from 0.65 with log K alone to 0.27 with the norm added. So the norm carries information about learnability, but as an "easiness" signal, not as complexity. The pattern holds:
- for all four estimators (E0 through E3);
- for the L2 and log-sum summaries and for either transformer block alone;
- across probe seeds (Kendall's W 0.91 for DFA, 0.95 for DSL).

Label noise lowered the DFA norm rather than raising it (ratio 0.59 [0.45, 0.65]), the opposite of Gate 5's prediction.

**Gate 4 (diversity coefficient) and the rescaling check were not run.** The run hit its pre-registered 4-hour wall-clock cap. Other jobs kept the shared machine's load average at 7–20 for most of the run, so units of work took 2–20 times longer than on a free CPU. Gate 5 is therefore complete for DFA but not for DSL.

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

   The finished jobs were merged, and analyses whose inputs are missing are marked not run. Gate verdict NOT_RUN means "not tested", not "failed". The run can be resumed (`run.py --output results --resume`) if the cap is raised.
2. **Embedding storage.** The two families' embeddings have different dimensions (DFA 1,564, DSL 1,566), so `embeddings.npz` holds one set of arrays per family (`dfa_keys`, `dfa_f`, …). The keys are as in PROTOCOL Section 8.

## Verification

`verify.py` passes (results/verify.json, verify.log). It reloads the 5 probes per family and re-evaluates them (max difference 0 bits); recomputes exact K for every task and rebuilds the roster identically with disjoint roles; recomputes 1,788 saved embeddings (a 10% random sample plus every Gate-input embedding; max relative difference 0); and recomputes 142,352 statistics, bootstrap draws and table entries with the independently written independent_stats.py (max absolute difference 8.3e-16). Diversity batch embeddings were not checked because none were computed.

## Limitations

- Gate 4 and the rescaling (parameterization) check are untested. The handoff condition is therefore unresolved on rescaling, but it already fails on Gates 1–3.
- T\* is heavily right-censored for DFA (104 of 128 α = 1 runs never reached ε\* = 0.5 bits), so the DFA Gate 3 estimate rests on 3 finite targets plus ties.
- The probes are tiny transformers pretrained on the same family. A large pretrained probe on real data could behave differently. These results speak about this setting only.
- The negative link between the norm and head loss suggests the E0 norm mostly measures how well the frozen features fit the task. The research note's Kolmogorov reading of the norm does not hold here.
